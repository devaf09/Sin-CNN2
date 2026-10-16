"""sinwave_minimal.py — 正弦の干渉と回折だけで作る文字列分類器（単独で動く 1 ファイル版）

このファイルだけをコピーすれば使えます（依存は PyTorch のみ）。実験用のオプションは入れず、
これまでの実験（NOTES.md、実験 01〜27）で分かっている「最も効率の良い構成」だけを実装しています。

    import sinwave_minimal as sw

    net = sw.SinWaveNet(n_slots=8, own=[0, 1, 2, 3])       # 検出器 8 個のうち 0〜3 番を担当する技能
    sw.train(net, texts, labels)                           # labels は検出器の番号 (0〜3)
    sw.predict(net, ["こんにちは"])                         # → [検出器の番号]

    merged = sw.merge([net_a, net_b])                      # 別々に学習した技能を足すだけで融合
    merged = sw.merge([net_a])                             # 外せば完全に元どおり（取消）

しくみ（光の実験装置に例えると）
    文字列 → 光にする → 照明 → 滲み層 → [Sin の板 → 回折] × 16 → レンズ → スクリーンの窓 → 答え
    学習するのは「Sin の板」の模様（区画ごとの 振幅 A と 初期位相 φ）と、窓ごとの背景光・全体の温度だけ。
    重み行列（ニューロンのマップ）は持たない。

前提知識なしの解説は docs/introduction.md、実験の記録は NOTES.md。

変更履歴（成果が出たらここを更新する）
    1.0.0 (2026-10-14)  最初の版。文字=角速度の符号化、平均の滲み層、区画ごとの Sin レイヤ（ω 固定）、
                        遠方場での検出、背景光、任意の tilt 照明、並列光路の融合、光路の足し算・引き算。
                        性能（2 技能 × 4 クラスの合成課題、3 seed、experiments/bench_minimal.py）:
                          patterns=8: 技能単独 97.0±0.3%、2 技能の融合 93.1±0.2%（計算 8 倍）
                          patterns=0: 技能単独 94.6±1.3%、2 技能の融合 86.9±1.1%（計算 1 倍）

ライセンス: MIT
"""
from __future__ import annotations

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.0.0"


# =============================================================================================
# 1. 文字列を光にする（符号化）
# =============================================================================================
# 1 文字に S 個の標本（光の画素）を割り当て、文字ごとに「縞の細かさ（角速度）」が違う波を置く。
#   波形: exp(j·(2π·k·s/S + θ))、k = 文字コード mod S、θ = 文字ごとの固定の位相（ハッシュ）
# 縞の細かさが違う光は進む向きが違うので、レンズの後では「文字 = スクリーン上の場所」になる。
# 文字列はレイヤ長 L まで繰り返す。L/S 文字を超える部分は使われない（既定では 8 文字まで）。
# S は区別したい文字の種類より大きくする（ひらがな+カタカナなら 256、漢字を含むなら 1024 以上）。

def _hash01(x: int) -> float:
    """整数 → [0, 1) の擬似乱数（splitmix64）。文字ごとの固定の位相 θ に使う。"""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) / 2**64


def coprime_stride(L: int) -> int:
    """L と約数を共有しない（互いに素な）、L の約 0.618 倍の歩幅。周波数 f を f·g mod L に並べ直すのに使う。
    オシロスコープで「ピックアップ間隔」を正しく選ぶのと同じで、小さな L でも周波数が重ならない（実験35）。"""
    g = int(L * 0.6180339887) | 1
    while math.gcd(g, L) != 1:
        g += 2
    return g


def encode_wave(texts: list[str], L: int, n: int, mix: str = "rainbow", V: int = 256, rb: float = 0.5,
                T: int = 8, device=None) -> torch.Tensor:
    """文字列 → 光の場 (N, T, L)。各文字の光がレイヤ全体を照らす（v1.1、実験30・33・35）。

    文字 c は、レイヤ全体に広がる平面波 exp(j·(2π·f·x/L + θ_c + 2π·0.618·m))。
      f = (文字コード mod V)·n·g mod L（g は L と互いに素な歩幅）、θ_c = 文字ごとの固定の位相、m = 文字の位置
    文字どうしの合わせ方 mix:
      coh    : 全部干渉させる（T=1、計算 1 倍。順序に強い）
      color  : 文字ごとに別の色（干渉しない。T=n。文字の頻度に最も強いが、順序は表せない）
      rainbow: 位置 m の光の周波数を m·rb だけずらし、時間 T 標本で平均（虹）。隣の文字どうしは干渉し、
               離れた文字どうしは干渉しない。rb=0.5 で順序と頻度を両立（実験33）
    先頭の n 文字だけを使う。"""
    if any(len(t) == 0 for t in texts):
        raise ValueError("空の文字列は符号化できない")
    g = coprime_stride(L)
    x = torch.arange(L, dtype=torch.float64)
    N = len(texts)
    W = torch.zeros(N, n, L, dtype=torch.complex64)
    for b, t in enumerate(texts):
        t = t[:n]
        for m, c in enumerate(t):
            f = ((ord(c) % V) * n * g) % L
            ph = 2 * math.pi * f * x / L + 2 * math.pi * _hash01(ord(c) * 1000003) + 2 * math.pi * 0.6180339887 * m
            W[b, m] = torch.polar(torch.ones(L, dtype=torch.float64), ph).to(torch.complex64) / math.sqrt(len(t))
    if mix == "coh":
        U = W.sum(1, keepdim=True)
    elif mix == "color":
        U = W
    elif mix == "rainbow":
        tau = torch.arange(T, dtype=torch.float32)
        m = torch.arange(n, dtype=torch.float32)
        code = torch.exp(1j * (2 * math.pi * rb * m[None, :] * tau[:, None] / T)).to(torch.complex64)  # (T, n)
        U = torch.einsum("tn,bnl->btl", code, W)
    else:
        raise ValueError("mix は coh / color / rainbow")
    return U.to(device)


def encode(texts: list[str], L: int, S: int, device=None) -> torch.Tensor:
    """文字列の並び → 光の場 (N, L)（複素数）。v1.0 からの「区画（スロット）」方式。"""
    if any(len(t) == 0 for t in texts):
        raise ValueError("空の文字列は符号化できない")
    chars = sorted(set("".join(texts)))
    s = torch.arange(S, dtype=torch.float64)
    # 文字ごとの波形 (V, S)
    table = torch.stack([2 * math.pi * (ord(c) % S) * s / S + 2 * math.pi * _hash01(ord(c) * 1000003)
                         for c in chars]).float()
    table = torch.polar(torch.ones_like(table), table).to(device)
    cid = {c: i for i, c in enumerate(chars)}
    n = -(-L // S)                                                    # 置ける文字数（切り上げ）
    ids = torch.tensor([[cid[c] for c in (t * (n // len(t) + 1))[:n]] for t in texts], device=device)
    return table[ids].reshape(len(texts), -1)[:, :L]                  # 文字の波形を並べる


# =============================================================================================
# 2. 学習器本体
# =============================================================================================

class SinWaveNet(nn.Module):
    """1 本の光路（= 1 つの技能）。

    n_slots  : スクリーンに並べる検出器（窓）の総数。融合する技能どうしで揃える
    own      : この技能が担当する検出器の番号（クラス）。融合する技能どうしで重ならないようにする
    L        : レイヤ長（光の画素数）。2 の累乗が速い
    S        : 1 文字の標本数。区別したい文字の種類より大きくする
    layers   : Sin の板の枚数
    patches  : 1 枚の板の区画数。区画ごとに (振幅, 初期位相) を学習する
    patterns : （input="slot" のとき）照明パターンの数。0 なら照明なし（計算 1 倍）。8 にすると tilt 照明（計算 8 倍）。
               融合して使うなら 8 を勧める（2 技能の融合で 86.9% → 93.1%）
    seed     : 板の初期模様の乱数
    input    : 文字の入れ方。"slot"（v1.0。1 文字に S 画素の区画）/ "wave"（v1.1。各文字の光がレイヤ全体を照らす）
    mix      : （input="wave" のとき）文字どうしの合わせ方。"rainbow"（既定）/ "color" / "coh"（encode_wave を参照）
    max_chars: （input="wave" のとき）使う文字数
    V        : （input="wave" のとき）文字コードを V で割った余りで文字を区別する（ひらがな+カタカナなら 256）
    """

    def __init__(self, n_slots: int, own: list[int] | None = None, L: int = 2048, S: int = 256,
                 layers: int = 16, patches: int = 64, patterns: int = 0, seed: int = 0,
                 input: str = "slot", mix: str = "rainbow", max_chars: int = 10, V: int = 256,
                 rb: float = 0.5, T: int = 8):
        super().__init__()
        self.n_slots, self.L, self.S = n_slots, L, S
        self.input, self.mix, self.max_chars, self.V, self.rb, self.T = input, mix, max_chars, V, rb, T
        if input not in ("slot", "wave"):
            raise ValueError('input は "slot" か "wave"')
        if input == "wave" and V * max_chars > L * 64:
            warnings.warn("V·max_chars が L に比べて大きいと、文字の周波数が重なりやすい", stacklevel=2)
        self.own = list(own) if own is not None else list(range(n_slots))
        if max(self.own) >= n_slots or len(set(self.own)) != len(self.own):
            raise ValueError("own は n_slots 未満の、重複しない番号の列")
        self.config = dict(n_slots=n_slots, own=self.own, L=L, S=S, layers=layers, patches=patches,
                           patterns=patterns, seed=seed, input=input, mix=mix, max_chars=max_chars, V=V, rb=rb, T=T)

        # --- Sin の板（学習する）------------------------------------------------------------
        # 板 d の区画 p は、位置 t の光の位相を A·sin(ω·t + φ) だけ遅らせる。
        # ω（縞の細かさ）は 2π/L〜π の対数一様な乱数で決めて固定する（学習しない方が同等かわずかに良い: 実験19b・22）。
        g = torch.Generator().manual_seed(seed)
        lo, hi = math.log(2 * math.pi / L), math.log(math.pi)
        A, w, phi = [], [], []
        for _ in range(layers):
            r = torch.rand(3, patches, 1, generator=g)
            A.append(0.5 + 0.5 * r[0])
            w.append(torch.exp(lo + (hi - lo) * r[1]))
            phi.append(2 * math.pi * r[2])
        self.A = nn.Parameter(torch.stack(A)[..., 0])                 # (layers, patches) 振幅
        self.phi = nn.Parameter(torch.stack(phi)[..., 0])             # (layers, patches) 初期位相
        self.register_buffer("w", torch.stack(w)[..., 0])            # (layers, patches) 角速度（固定）
        self.register_buffer("patch", torch.arange(L) * patches // L)  # 位置 t → 区画番号
        self.register_buffer("t", torch.arange(L, dtype=torch.float32))

        # --- 検出器（学習するのは温度と背景光だけ）------------------------------------------
        self.gain = nn.Parameter(torch.tensor(5.0))                   # 明るさ → スコアの倍率
        self.bias = nn.Parameter(torch.zeros(n_slots))                # 窓ごとの背景光（クラスの起こりやすさ）

        # --- 回折（固定）: 角スペクトル法。波長 1、板の間隔 64 画素 ---------------------------
        kx = 2 * math.pi * torch.fft.fftfreq(L)                       # 横方向の空間周波数
        kz = torch.sqrt((2 * math.pi) ** 2 - kx ** 2)                 # 進行方向の波数
        self.register_buffer("H", torch.polar(torch.ones(L), kz * 64.0))

        # --- 照明（固定、任意）: tilt 照明 = 文字の位置 m に位相 2π·j·m/n を与えた n 通りの照明 ---
        # n 通りの明るさを平均すると、違う位置の文字どうしの干渉（位置の雑音）がちょうど 0 になる。
        # 携帯電話の CDMA と同じ「直交符号で混信を消す」考え方（実験10・11）。
        n_pos = -(-L // S)
        if patterns > 0:
            j = torch.arange(patterns, dtype=torch.float32)[:, None]
            m = torch.arange(n_pos, dtype=torch.float32)[None]
            psi = (2 * math.pi * j * m / n_pos).repeat_interleave(S, -1)[:, :L]
        else:
            psi = torch.zeros(1, L)
        self.register_buffer("illum", torch.polar(torch.ones_like(psi), psi))   # (patterns, L)

        # --- 検出器の窓（固定）: スクリーンに n_slots 個の幅 4 の窓を等間隔に並べる ----------------
        det = torch.zeros(n_slots, L)
        for i in range(n_slots):
            c = int((i + 0.5) * L / n_slots)
            det[i, (torch.arange(4) + c - 2) % L] = 0.25              # 窓の中の明るさの平均
        self.register_buffer("det", det)

    # -------------------------------------------------------------------------------------
    def intensity(self, u: torch.Tensor) -> torch.Tensor:
        """光の場 (N, L)（slot）または (N, T, L)（wave）→ スクリーンの明るさ (N, L)（平均 1）。"""
        if u.dim() == 2:   # slot: 照明と滲み層
            u = u[:, None] * self.illum[None]                         # 照明パターンごとに (N, P, L)
            # 滲み層: 前後の文字（±S 画素）との平均。文字そのものの縞は残し、位置の雑音を弱める（実験18c）
            u = (torch.roll(u, -self.S, -1) + u + torch.roll(u, self.S, -1)) / 3
        # wave: (N, T, L) の T は「色・時間」の軸。最後に明るさを平均する（色どうし・時間で干渉が消える）
        for d in range(self.A.shape[0]):
            # Sin の板: 区画ごとの A·sin(ω t + φ) だけ位相を遅らせる
            p = self.patch
            phase = self.A[d, p] * torch.sin(self.w[d, p] * self.t + self.phi[d, p])
            u = u * torch.polar(torch.ones_like(phase), phase)
            # 回折: 次の板まで 64 画素進む（周期境界）
            u = torch.fft.ifft(torch.fft.fft(u) * self.H)
            # 増幅: 平均の明るさを 1 に戻す（値が潰れたり発散したりしないように）
            u = u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()   # 全部 0 でも微分が壊れないように
        u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)  # レンズ: 同じ向きの光を 1 点に集める
        inten = u.abs().pow(2).mean(1)                                # 明るさ、照明パターンについて平均
        return inten / (inten.mean(-1, keepdim=True) + 1e-12)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """光の場 (N, L) → 検出器ごとのスコア (N, n_slots)。"""
        return self.gain * (self.intensity(u) @ self.det.T) + self.bias

    def encode(self, texts: list[str], device=None) -> torch.Tensor:
        """文字列 → 光の場（既定ではこの光路と同じ装置上）。長すぎる文字列には警告を出す。"""
        dev = self.det.device if device is None else device
        limit = self.max_chars if self.input == "wave" else self.L // self.S
        n = sum(len(t) > limit for t in texts)
        if n:
            warnings.warn(f"{n} 個の文字列が {limit} 文字を超えている。超えた部分は使われない", stacklevel=3)
        if self.input == "wave":
            return encode_wave(texts, self.L, self.max_chars, self.mix, self.V, self.rb, self.T, dev)
        return encode(texts, self.L, self.S, dev)


# =============================================================================================
# 3. 学習と予測
# =============================================================================================

def train(net: SinWaveNet, texts: list[str], labels: list[int], steps: int = 1500, lr: float = 0.01,
          batch_size: int = 256, seed: int = 0, verbose: bool = False) -> SinWaveNet:
    """labels は検出器の番号（net.own のどれか）。正解の窓が明るくなるよう板の模様を少しずつ変える。"""
    if not texts or len(texts) != len(labels):
        raise ValueError("texts と labels は同じ長さで、空でないこと")
    if set(labels) - set(net.own):
        raise ValueError(f"labels は own {net.own} のどれかにする")
    dev = net.det.device
    X = net.encode(list(texts), device="cpu")                         # 光の場は CPU に置き、バッチごとに送る
    Y = torch.tensor(list(labels))
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        i = torch.randint(len(Y), (min(batch_size, len(Y)),), generator=gen)
        loss = F.cross_entropy(net(X[i].to(dev)), Y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % max(steps // 10, 1) == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    return net


@torch.no_grad()
def scores(net, texts: list[str], batch_size: int = 512) -> torch.Tensor:
    """各検出器のスコア (N, n_slots)。担当外の検出器は -inf。net は SinWaveNet でも Merged でもよい。"""
    out = torch.cat([net(net.encode(texts[i:i + batch_size])) for i in range(0, len(texts), batch_size)])
    mask = torch.full((net.n_slots,), float("-inf"), device=out.device)
    mask[net.own] = 0
    return (out + mask).cpu()


def predict(net, texts: list[str]) -> list[int]:
    """一番明るい（スコアの高い）担当検出器の番号。"""
    return scores(net, list(texts)).argmax(1).tolist()


# =============================================================================================
# 4. 融合・取消（並列光路）と、光路の足し算・引き算
# =============================================================================================

class Merged(nn.Module):
    """別々に学習した技能（光路）を並べたもの。互いに干渉しない光（色や偏光の違い）を想定し、
    スクリーンでは明るさを足すだけ。融合に追加の学習は要らず、光路を外せば完全に元に戻る。"""

    def __init__(self, nets: list[SinWaveNet]):
        super().__init__()
        if not nets:
            raise ValueError("技能が空")
        owned = [o for n in nets for o in n.own]
        if len(set(owned)) != len(owned):
            raise ValueError("融合する技能どうしで担当する検出器 (own) が重なっている")
        if len({(n.n_slots, n.L, n.S, n.input, n.mix, n.max_chars, n.V, n.rb, n.T) for n in nets}) != 1:
            raise ValueError("融合する技能どうしで n_slots, L, S と入力の設定 (input, mix, max_chars, V, rb, T) を揃える")
        self.nets = nn.ModuleList(nets)
        self.n_slots, self.L, self.S, self.own = nets[0].n_slots, nets[0].L, nets[0].S, owned

    def forward(self, u):
        return sum(n(u) for n in self.nets) / len(self.nets)        # 各光路のスコアの平均（＝明るさの平均）

    def encode(self, texts, device=None):
        return self.nets[0].encode(texts, device)


def merge(nets: list[SinWaveNet]) -> Merged:
    """融合。取消は、外したい技能を除いたリストで merge し直せばよい（残りは元と完全に同じ）。"""
    return Merged(list(nets))


def combine(base: SinWaveNet, terms: list[tuple[float, SinWaveNet]], own: list[int] | None = None) -> SinWaveNet:
    """光路の足し算・引き算（タスク算術）: base + Σ 係数 × (技能 − base) を 1 本の光路として作る。

    技能は base をコピーして追加学習したもの（derive を使う）。共通の出発点が必要で、
    ゼロから別々に学習したものどうしでは壊れる（実験23）。
      係数 +1 : 技能を足す。1 本の光路のまま融合できる（計算は増えない）
      係数 −1 : その技能に似た入力への反応を弱める（例: リンゴ技能 − トマト技能 で、トマトへの誤反応が減る）
    own を省略すると、係数が正の技能の担当をまとめる。
    """
    for _, t in terms:
        if t.w.shape != base.w.shape or not torch.equal(t.w.cpu(), base.w.cpu()):
            raise ValueError("共通の出発点（derive で作ったもの）どうしでないと足し引きできない")
    out = copy.deepcopy(base)
    with torch.no_grad():
        for name, p in out.named_parameters():
            b = dict(base.named_parameters())[name]
            p.copy_(b + sum(c * (dict(t.named_parameters())[name] - b) for c, t in terms))
    out.own = list(own) if own is not None else [o for c, t in terms if c > 0 for o in t.own]
    return out


def derive(base: SinWaveNet, own: list[int]) -> SinWaveNet:
    """base を共通の出発点として、担当だけを変えたコピーを作る（あとで train する）。"""
    out = copy.deepcopy(base)
    out.own = list(own)
    return out


# =============================================================================================
# 5. 語根（部品）の検出器: 例のない新しい物を、部品の組み合わせとして認識する（v1.1、実験34・36）
# =============================================================================================
# クラスごとではなく、部品（特徴）ごとに窓を持たせ、「どの部品が含まれているか」を学習する。
# 新しいクラスは、例を 1 つも見せずに「部品 a + 部品 b + …」と定義するだけ（造語のように組み立てる）。
# 実験34 では、見せていない新しい 4 種を 89.7% で当てた（例 5 個で覚える方法は 21%）。

def train_roots(net: SinWaveNet, texts: list[str], roots: list[list[int]], steps: int = 1500, lr: float = 0.01,
                batch_size: int = 256, seed: int = 0) -> SinWaveNet:
    """roots[i] = texts[i] に含まれる部品（窓の番号）の列。部品ごとの窓が「有る／無い」を学ぶ（多ラベル学習）。"""
    if len(texts) != len(roots) or not texts:
        raise ValueError("texts と roots は同じ長さで、空でないこと")
    dev = net.det.device
    X = net.encode(list(texts), device="cpu")
    Y = torch.zeros(len(texts), net.n_slots)
    for i, r in enumerate(roots):
        Y[i, list(r)] = 1
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(Y), (min(batch_size, len(Y)),), generator=gen)
        loss = F.binary_cross_entropy_with_logits(net(X[i].to(dev)), Y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.root_prior = Y.mean(0).clamp(0.02, 0.98)        # 学習データでの部品の出やすさ（偏りの補正に使う）
    return net


@torch.no_grad()
def compose(net: SinWaveNet, texts: list[str], classes: dict, seen: list | None = None, prior_adjust: bool = True,
            gamma: float = 0.0) -> list:
    """部品の組み合わせで定義したクラス classes = {名前: [部品の番号, ...]} のどれに当たるかを答える。
    点数 = Σ_{部品∈クラス} log P(有る) + Σ_{部品∉クラス} log P(無い)。
    prior_adjust: 学習データでの部品の出やすさの偏りを差し引く（新しい組み合わせを既知の物と取り違えにくくする）
    gamma       : seen（学習で見たクラスの名前）の点数から gamma を引く（較正つきの積み重ね）"""
    z = torch.cat([net(net.encode(texts[i:i + 512])) for i in range(0, len(texts), 512)]).cpu()
    if prior_adjust and hasattr(net, "root_prior"):
        z = z - torch.logit(net.root_prior.cpu())
    names = list(classes)
    M = torch.zeros(len(names), net.n_slots)
    for k, nm in enumerate(names):
        M[k, list(classes[nm])] = 1
    s = F.logsigmoid(z) @ M.T + F.logsigmoid(-z) @ (1 - M).T
    if seen:
        s[:, [names.index(nm) for nm in seen]] -= gamma
    return [names[i] for i in s.argmax(1).tolist()]


# =============================================================================================
# 6. 保存と読み込み
# =============================================================================================

def save(net: SinWaveNet, path: str) -> None:
    """学習した値（A, φ, 温度, 背景光）と設定だけを保存する（1 技能 約 10KB）。ω などの固定部品は設定から作り直す。"""
    torch.save({"version": __version__, "config": dict(net.config, own=net.own),
                "state": {k: v for k, v in net.state_dict().items() if k in ("A", "phi", "gain", "bias")},
                "root_prior": getattr(net, "root_prior", None)}, path)


def load(path: str, device=None) -> SinWaveNet:
    d = torch.load(path, map_location="cpu", weights_only=False)
    net = SinWaveNet(**d["config"])            # v1.0 の保存ファイルは input="slot" として読める
    net.load_state_dict(d["state"], strict=False)
    if d.get("root_prior") is not None:
        net.root_prior = d["root_prior"]
    return net.to(device) if device is not None else net


# =============================================================================================
# 動作例: python sinwave_minimal.py
# =============================================================================================
if __name__ == "__main__":
    import random

    # 合成データ: 技能 A（検出器 0,1）と技能 B（検出器 2,3）。各クラスは好む文字を持つ
    keys = {0: "あいうえ", 1: "かきくけ", 2: "さしすせ", 3: "たちつて"}
    pool = "なにぬねのはひふへほまみむめもやゆよらりるれろわを"

    def sample(r, y):
        return "".join(r.choice(keys[y]) if r.random() < 0.6 else r.choice(pool) for _ in range(r.randint(4, 8)))

    r = random.Random(0)
    data = {y: [sample(r, y) for _ in range(300)] for y in keys}
    test = {y: [sample(r, y) for _ in range(100)] for y in keys}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    a = train(SinWaveNet(n_slots=4, own=[0, 1]).to(dev), data[0] + data[1], [0] * 300 + [1] * 300)
    b = train(SinWaveNet(n_slots=4, own=[2, 3], seed=1).to(dev), data[2] + data[3], [2] * 300 + [3] * 300)
    acc = lambda net, ys: sum(predict(net, test[y]).count(y) for y in ys) / (100 * len(ys))
    print(f"技能 A 単独: {acc(a, [0, 1]):.3f}   技能 B 単独: {acc(b, [2, 3]):.3f}")
    print(f"融合 (4 クラス): {acc(merge([a, b]), [0, 1, 2, 3]):.3f}")
    print(f"B を外す: {acc(merge([a]), [0, 1]):.3f}（A 単独と同じ）")
    print(f"学習するパラメータ: {sum(p.numel() for p in a.parameters())} 個")
