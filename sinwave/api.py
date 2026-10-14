"""SinWave を数行で使うための高水準 API。

    from sinwave import SinWaveClassifier, merge

    a = SinWaveClassifier(classes=["挨拶", "天気"], slots=[0, 1], n_slots=8).fit(texts_a, labels_a)
    b = SinWaveClassifier(classes=["食べ物", "乗り物"], slots=[2, 3], n_slots=8).fit(texts_b, labels_b)
    m = merge([a, b])          # 別々に学習した技能を足すだけで融合
    m.predict(["こんにちは"])
    m.remove("天気")           # 技能（光路）を外す = 完全な取消

各技能（SinWaveClassifier）は 1 本の光路で、出力面の検出器のうち自分の slots だけを担当する。
融合する技能どうしは n_slots と光学設定 (preset) を揃え、slots が重ならないようにする。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import Config, SinWave, encode

# 実験で良かった設定。S は「区別したい文字の種類」に合わせる（ひらがな+カタカナなら 256、漢字を含む文なら 1024）
PRESETS = {
    # 計算 1 倍。平均の滲み層で干渉の雑音を減らす
    "fast": dict(L=2048, S=256, D=16, P=64, blur="mean", incoh=0),
    # 計算 8 倍。平均の滲み層 + tilt 照明（入射角を変えた平面波 8 通り）
    "best": dict(L=2048, S=256, D=16, P=64, blur="mean", incoh=8, illum="tilt"),
    # 漢字を含む実テキスト向け（周波数の数を増やす）
    "text": dict(L=2048, S=1024, D=16, P=64, blur="", incoh=0),
}


def _device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class SinWaveClassifier:
    """正弦の干渉・回折による文字列分類器（1 技能 = 1 光路）。

    classes : この技能が担当するラベルの列
    slots   : 各ラベルに割り当てる検出器の番号（省略時 0, 1, 2, ...）
    n_slots : 出力面に並べる検出器の総数（融合する技能どうしで揃える）
    preset  : "fast" / "best" / "text"。個別の値は overrides で上書きできる
    """

    def __init__(self, classes, slots=None, n_slots=None, preset="fast", name=None, seed=0,
                 device=None, freeze_w=False, **overrides):
        self.classes = list(classes)
        self.slots = list(slots) if slots is not None else list(range(len(self.classes)))
        assert len(self.slots) == len(self.classes), "classes と slots の数を揃える"
        self.n_slots = n_slots or max(self.slots) + 1
        assert max(self.slots) < self.n_slots
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

    def encode(self, texts):
        return encode(list(texts), self.opts["L"], 1, self.device, self.opts["S"], "freq")

    # ------------------------------------------------------------ 学習・推論
    def fit(self, texts, labels, steps=1500, lr=0.01, batch_size=256, verbose=False):
        idx = {c: s for c, s in zip(self.classes, self.slots)}
        X = self.encode(texts)
        Y = torch.tensor([idx[y] for y in labels], device=self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        self.model.train()
        for s in range(steps):
            i = torch.randint(len(Y), (min(batch_size, len(Y)),), device=self.device)
            loss = F.cross_entropy(self.model(X[i]), Y[i])
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
        X = self.encode(texts)
        return torch.cat([self.model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])

    @torch.no_grad()
    def predict_proba(self, texts):
        lg = self.logits(texts)[:, self.slots]
        return torch.softmax(lg, 1).cpu()

    def predict(self, texts):
        p = self.predict_proba(texts)
        return [self.classes[i] for i in p.argmax(1).tolist()]

    def score(self, texts, labels):
        pred = self.predict(texts)
        return sum(a == b for a, b in zip(pred, labels)) / len(labels)

    # ------------------------------------------------------------ 保存
    def save(self, path):
        torch.save(dict(classes=self.classes, slots=self.slots, n_slots=self.n_slots, name=self.name,
                        seed=self.seed, freeze_w=self.freeze_w, preset=self.preset, opts=self.opts,
                        state=self._learned_state()), path)

    def _learned_state(self):
        # 学習した値（A, ω, φ, 温度, 背景光）だけを保存する。伝搬・検出器などの固定部品は設定から作り直せる
        keep = ("A", "w", "phi", "gain", "bias")
        return {k: v for k, v in self.model.state_dict().items() if k.split(".")[-1] in keep}

    @classmethod
    def load(cls, path, device=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(d["classes"], d["slots"], d["n_slots"], d["preset"], d["name"], d["seed"], device,
                  d["freeze_w"], **d["opts"])
        missing, unexpected = obj.model.load_state_dict(d["state"], strict=False)
        assert not unexpected and all(k.split(".")[-1] not in ("A", "w", "phi", "gain", "bias") for k in missing)
        return obj

    def n_params(self):
        """学習するパラメータ数（レイヤの A, ω, φ ＋ 温度 1 ＋ 背景光 n_slots）。"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self):
        return f"SinWaveClassifier({self.name!r}, classes={self.classes}, slots={self.slots}, preset={self.preset!r})"


class MergedClassifier:
    """別々に学習した技能（光路）を並べたもの。強度を足すだけで融合し、外せば完全に元に戻る。

    mode="avg" : 各光路の logit の平均（＝検出強度の平均）
    mode="gate": 各光路の「担当検出器に入った光の割合」を技能の確信度にする（技能が多いとき少し良い）
    """

    def __init__(self, skills, mode="avg"):
        self.skills = list(skills)
        self.mode = mode
        self._check()

    def _check(self):
        assert self.skills, "技能が空"
        s0 = self.skills[0]
        used = {}
        for s in self.skills:
            assert s.n_slots == s0.n_slots, "n_slots を揃える"
            assert {k: s.opts[k] for k in ("L", "S")} == {k: s0.opts[k] for k in ("L", "S")}, "L と S を揃える"
            for c, sl in zip(s.classes, s.slots):
                assert sl not in used, f"検出器 {sl} が {used[sl]} と {c} で重なっている"
                used[sl] = c

    @property
    def classes(self):
        return [c for s in self.skills for c in s.classes]

    @property
    def slots(self):
        return [sl for s in self.skills for sl in s.slots]

    def add(self, skill):
        self.skills.append(skill)
        self._check()
        return self

    def remove(self, name_or_class):
        """技能を名前か、その技能が持つクラス名で外す。"""
        self.skills = [s for s in self.skills if s.name != name_or_class and name_or_class not in s.classes]
        return self

    @torch.no_grad()
    def logits(self, texts):
        """融合した各クラスのスコア (N, len(classes))。"""
        cols = self.slots
        if self.mode == "avg":
            return (sum(s.logits(texts) for s in self.skills) / len(self.skills))[:, cols].cpu()
        if self.mode == "gate":
            out = []
            for s in self.skills:
                X = s.encode(texts)
                inten = s.model.intensity(X)
                f = inten @ s.model.det.T / s.model.cfg.det_width
                own = torch.tensor(s.slots, device=f.device)
                r = f[:, own].sum(1) / (f.sum(1) + 1e-9)
                lg = s.model.detect(inten)[:, own]
                out.append(torch.log(r + 1e-9)[:, None] + torch.log_softmax(lg, 1))
            return torch.cat(out, 1).cpu()
        raise ValueError(self.mode)

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
