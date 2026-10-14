"""SinWave を数行で使うための高水準 API。

    from sinwave import SinWaveClassifier, merge

    a = SinWaveClassifier(classes=["挨拶", "天気"], slots=[0, 1], n_slots=8).fit(texts_a, labels_a)
    b = SinWaveClassifier(classes=["食べ物", "乗り物"], slots=[2, 3], n_slots=8).fit(texts_b, labels_b)
    m = merge([a, b])          # 別々に学習した技能を足すだけで融合
    m.predict(["こんにちは"])
    m.remove("天気")           # 技能（光路）を外す = 完全な取消

各技能（SinWaveClassifier）は 1 本の光路で、出力面の検出器のうち自分の slots だけを担当する。
融合する技能どうしは n_slots と光学設定 (preset) を揃え、slots が重ならないようにする。

注意: 1 文字は S 標本を使うので、使われるのは各文字列の先頭 L/S 文字だけ（既定の "fast" では 8 文字）。
それより長い部分は無視される（max_chars で確かめられる。超えると警告を出す）。
"""
from __future__ import annotations

import copy
import warnings

import torch
import torch.nn.functional as F

from .model import Config, SinWave, char_table, encode, encode_ids, field_from_ids

# 実験で良かった設定。S は「区別したい文字の種類」に合わせる（ひらがな+カタカナなら 256、漢字を含む文なら 1024）
PRESETS = {
    # 計算 1 倍。平均の滲み層で干渉の雑音を減らす
    "fast": dict(L=2048, S=256, D=16, P=64, blur="mean", incoh=0),
    # 計算 8 倍。平均の滲み層 + tilt 照明（入射角を変えた平面波 8 通り）
    "best": dict(L=2048, S=256, D=16, P=64, blur="mean", incoh=8, illum="tilt"),
    # 漢字を含む実テキスト向け（周波数の数を増やす。使われるのは先頭 2 文字）
    "text": dict(L=2048, S=1024, D=16, P=64, blur="", incoh=0),
}

LEARNED = ("A", "w", "phi", "gain", "bias")   # 学習する値（保存・足し引きの対象）


def _check_classes(classes, slots):
    classes = list(classes)
    slots = list(slots) if slots is not None else list(range(len(classes)))
    if not classes or len(slots) != len(classes):
        raise ValueError("classes は空でなく、slots と同じ数にする")
    if len(set(classes)) != len(classes) or len(set(slots)) != len(slots):
        raise ValueError("classes と slots に重複があってはいけない")
    return classes, slots


def _device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class SinWaveClassifier:
    """正弦の干渉・回折による文字列分類器（1 技能 = 1 光路）。

    classes  : この技能が担当するラベルの列
    slots    : 各ラベルに割り当てる検出器の番号（省略時 0, 1, 2, ...）
    n_slots  : 出力面に並べる検出器の総数（融合する技能どうしで揃える）
    preset   : "fast" / "best" / "text"。個別の値 (L, S, D, P, blur, incoh, illum) は overrides で上書きできる
    freeze_w : True なら角速度 ω を学習しない（パラメータが 2/3 になり、性能は同等かわずかに良い）
    """

    def __init__(self, classes, slots=None, n_slots=None, preset="fast", name=None, seed=0,
                 device=None, freeze_w=False, **overrides):
        if preset not in PRESETS:
            raise ValueError(f"preset は {list(PRESETS)} のどれか")
        self.classes, self.slots = _check_classes(classes, slots)
        self.n_slots = n_slots or max(self.slots) + 1
        if max(self.slots) >= self.n_slots:
            raise ValueError(f"slots の番号は n_slots ({self.n_slots}) 未満にする")
        self.name = name or "+".join(map(str, self.classes))
        self.seed = seed
        self.freeze_w = freeze_w
        self.opts = dict(PRESETS[preset], **overrides)
        self.preset = preset
        self.device = _device(device)
        self._build()

    # ------------------------------------------------------------ 構築
    def _config(self):
        o = self.opts
        return Config(L=o["L"], C=self.n_slots, far=True, bias=True, slot=o["S"], blur=o.get("blur", ""),
                      incoh=o.get("incoh", 0), illum=o.get("illum", "random"))

    def _build(self):
        torch.manual_seed(self.seed)
        self.model = SinWave(self._config()).to(self.device)
        self.model.add_layers(self.opts["D"], 1, "phase", tag=self.name, seed=self.seed, P=self.opts["P"])
        self.model.bias.requires_grad_(True)
        if self.freeze_w:
            for layer in self.model.layers:
                layer.w.requires_grad_(False)

    @property
    def max_chars(self):
        """1 つの文字列のうち使われる文字数（レイヤ長 L / 1 文字の標本数 S）。"""
        return self.opts["L"] // self.opts["S"]

    def _warn_long(self, texts):
        n = sum(len(t) > self.max_chars for t in texts)
        if n:
            warnings.warn(f"{n} 個の文字列が {self.max_chars} 文字を超えている。超えた部分は使われない",
                          stacklevel=3)

    def encode(self, texts):
        """文字列を光の場 (N, 1, L) にする。"""
        return encode(list(texts), self.opts["L"], 1, self.device, self.opts["S"], "freq")

    # ------------------------------------------------------------ 学習・推論
    def fit(self, texts, labels, steps=1500, lr=0.01, batch_size=256, verbose=False):
        texts, labels = list(texts), list(labels)
        if not texts or len(texts) != len(labels):
            raise ValueError("texts と labels は同じ長さで、空でないこと")
        idx = dict(zip(self.classes, self.slots))
        unknown = sorted({repr(y) for y in labels if y not in idx})
        if unknown:
            raise ValueError(f"classes にないラベル: {', '.join(unknown[:5])}")
        self._warn_long(texts)
        # 文字番号の表だけを持ち、光の場はミニバッチごとに作る（GPU メモリはバッチ分しか使わない）
        L, S = self.opts["L"], self.opts["S"]
        chars, ids = encode_ids(texts, L, S)
        tab = char_table(chars, 1, S, "freq", self.device)
        Y = torch.tensor([idx[y] for y in labels])
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        gen = torch.Generator().manual_seed(self.seed)   # ミニバッチの選び方も seed で決まる
        for s in range(steps):
            i = torch.randint(len(Y), (min(batch_size, len(Y)),), generator=gen)
            loss = F.cross_entropy(self.model(field_from_ids(ids[i], tab, L)), Y[i].to(self.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if verbose and s % max(steps // 10, 1) == 0:
                print(f"step {s:5d}  loss {loss.item():.4f}")
        return self

    @torch.no_grad()
    def logits(self, texts, batch_size=512):
        """全検出器の logit (N, n_slots)。"""
        texts = list(texts)
        self._warn_long(texts)
        return torch.cat([self.model(self.encode(texts[i:i + batch_size]))
                          for i in range(0, len(texts), batch_size)])

    @torch.no_grad()
    def predict_proba(self, texts):
        """自分のクラスについての確率 (N, len(classes))。"""
        return torch.softmax(self.logits(texts)[:, self.slots], 1).cpu()

    def predict(self, texts):
        return [self.classes[i] for i in self.predict_proba(texts).argmax(1).tolist()]

    def score(self, texts, labels):
        """正解率。"""
        pred = self.predict(texts)
        return sum(a == b for a, b in zip(pred, labels)) / len(labels)

    # ------------------------------------------------------------ 保存
    def save(self, path):
        """学習した値（A, ω, φ, 温度, 背景光）と設定だけを保存する（1 技能 約 23KB）。"""
        state = {k: v for k, v in self.model.state_dict().items() if k.split(".")[-1] in LEARNED}
        torch.save(dict(classes=self.classes, slots=self.slots, n_slots=self.n_slots, name=self.name,
                        seed=self.seed, freeze_w=self.freeze_w, preset=self.preset, opts=self.opts,
                        state=state), path)

    @classmethod
    def load(cls, path, device=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(d["classes"], d["slots"], d["n_slots"], d["preset"], d["name"], d["seed"], device,
                  d["freeze_w"], **d["opts"])
        _, unexpected = obj.model.load_state_dict(d["state"], strict=False)
        if unexpected:
            raise ValueError(f"保存ファイルに知らない値がある: {unexpected[:3]}")
        return obj

    # ------------------------------------------------------------ 光路の足し算・引き算の準備
    def derive(self, classes, slots=None, name=None):
        """この技能を「共通の出発点」として、新しい技能を作る（パラメータを引き継ぐ。あとで fit する）。
        同じ出発点から作った技能どうしは combine で足し引きできる。"""
        new = copy.copy(self)
        new.classes, new.slots = _check_classes(classes, slots)
        if max(new.slots) >= self.n_slots:
            raise ValueError(f"slots の番号は n_slots ({self.n_slots}) 未満にする")
        new.name = name or "+".join(map(str, new.classes))
        new.model = copy.deepcopy(self.model)
        new.base = self
        return new

    def n_params(self):
        """学習するパラメータ数（レイヤの A, ω, φ ＋ 温度 1 ＋ 背景光 n_slots）。"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self):
        return f"SinWaveClassifier({self.name!r}, classes={self.classes}, slots={self.slots}, preset={self.preset!r})"


class MergedClassifier:
    """別々に学習した技能（光路）を並べたもの。強度を足すだけで融合し、外せば完全に元に戻る。

    mode="avg" : 各光路の logit の平均（＝検出強度の平均）
    mode="gate": 各光路の「担当検出器に入った光の割合」を技能の確信度にする
    """

    def __init__(self, skills, mode="avg"):
        if mode not in ("avg", "gate"):
            raise ValueError('mode は "avg" か "gate"')
        self.skills = list(skills)
        self.mode = mode
        self._check()

    def _check(self):
        if not self.skills:
            raise ValueError("技能が空")
        s0 = self.skills[0]
        used = {}
        for s in self.skills:
            if s.n_slots != s0.n_slots:
                raise ValueError("融合する技能どうしで n_slots を揃える")
            if (s.opts["L"], s.opts["S"]) != (s0.opts["L"], s0.opts["S"]):
                raise ValueError("融合する技能どうしで L と S を揃える")
            for c, sl in zip(s.classes, s.slots):
                if sl in used:
                    raise ValueError(f"検出器 {sl} が {used[sl]!r} と {c!r} で重なっている")
                used[sl] = c

    @property
    def classes(self):
        return [c for s in self.skills for c in s.classes]

    @property
    def slots(self):
        return [sl for s in self.skills for sl in s.slots]

    def add(self, skill):
        """技能を足す（融合）。"""
        self.skills.append(skill)
        try:
            self._check()
        except ValueError:
            self.skills.pop()
            raise
        return self

    def remove(self, name_or_class):
        """技能を名前か、その技能が持つクラス名で外す（取消）。最後の 1 つは外せない。"""
        keep = [s for s in self.skills if s.name != name_or_class and name_or_class not in s.classes]
        if len(keep) == len(self.skills):
            raise ValueError(f"{name_or_class!r} を持つ技能がない")
        if not keep:
            raise ValueError("すべての技能を外すことはできない")
        self.skills = keep
        return self

    @torch.no_grad()
    def logits(self, texts):
        """融合した各クラスのスコア (N, len(classes))。列の並びは self.classes。"""
        texts = list(texts)
        if self.mode == "avg":
            return (sum(s.logits(texts) for s in self.skills) / len(self.skills))[:, self.slots].cpu()
        out = []
        for s in self.skills:
            inten = torch.cat([s.model.intensity(s.encode(texts[i:i + 512])) for i in range(0, len(texts), 512)])
            f = inten @ s.model.det.T / s.model.cfg.det_width
            own = torch.tensor(s.slots, device=f.device)
            r = f[:, own].sum(1) / (f.sum(1) + 1e-9)
            lg = s.model.detect(inten)[:, own]
            out.append(torch.log(r + 1e-9)[:, None] + torch.log_softmax(lg, 1))
        return torch.cat(out, 1).cpu()

    def predict_proba(self, texts):
        return torch.softmax(self.logits(texts), 1)

    def predict(self, texts):
        cls = self.classes
        return [cls[i] for i in self.logits(texts).argmax(1).tolist()]

    def score(self, texts, labels):
        pred = self.predict(texts)
        return sum(a == b for a, b in zip(pred, labels)) / len(labels)

    def __repr__(self):
        return f"MergedClassifier({[s.name for s in self.skills]}, mode={self.mode!r})"


def merge(skills, mode="avg"):
    """技能を融合する。各技能は互いのデータを見ずに学習してよい。"""
    return MergedClassifier(skills, mode)


def combine(base, terms, classes=None, slots=None, name=None):
    """タスク算術: base + Σ 係数 × (技能 − base) を 1 本の光路として作る。

    terms: [(係数, 技能), ...]。技能は base.derive(...) から学習したもの。
    係数 +1 で技能を足す（1 本の光路での融合、計算は増えない）。
    負の係数で「その技能に似た入力への反応」を弱める（例: トマトを引いて、リンゴ技能のトマトへの誤反応を減らす）。
    classes / slots を省略すると、係数が正の技能のクラスをまとめる。
    """
    for _, sk in terms:
        if getattr(sk, "base", None) is not base:
            raise ValueError("同じ base から derive した技能だけを足し引きできる")
    out = copy.copy(base)
    out.model = copy.deepcopy(base.model)
    if classes is None:
        pos = [sk for c, sk in terms if c > 0]
        classes = [c for sk in pos for c in sk.classes]
        slots = [sl for sk in pos for sl in sk.slots]
    out.classes, out.slots = _check_classes(classes, slots)
    out.name = name or "combine(" + ", ".join(f"{c:+g}·{sk.name}" for c, sk in terms) + ")"
    b0 = base.model.state_dict()
    states = [(c, sk.model.state_dict()) for c, sk in terms]
    new = {k: v + sum(c * (st[k] - v) for c, st in states) for k, v in b0.items() if k.split(".")[-1] in LEARNED}
    out.model.load_state_dict(new, strict=False)
    return out
