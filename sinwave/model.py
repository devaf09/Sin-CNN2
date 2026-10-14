"""正弦の干渉・回折による学習器 (SinWave) の本体。

光（複素数の場 u）が「Sin レイヤ → 回折」を繰り返し、出力面の明るさで判定する。重み行列は持たない。

    文字列 ─ encode ─▶ 照明 ─▶ 滲み層 ─▶ [Sin レイヤ ─▶ 回折] × D ─▶ レンズ ─▶ 検出器 ─▶ logit

- encode   : 1 文字 = S 標本。freq 符号化では文字コード mod S を角速度にした搬送波 exp(j(2πk·s/S + θ))。
             文字列はレイヤ長 L まで繰り返す（L/S 文字を超える部分は使われない）。
- 照明     : (incoh>0) 照明パターンを incoh 通り用意し、明るさを平均する。異なる位置の文字どうしの干渉を打ち消す。
- 滲み層   : (blur="mean") 前後の文字との平均。周波数領域では櫛形フィルタで、位置の雑音を弱める。
- Sin レイヤ: u ← u · exp(j · A sin(ωt + φ))。レイヤを P 区画に分け、区画ごとに (A, ω, φ) を持つ。学習するのはここだけ。
- 回折     : 角スペクトル法による自由空間伝搬（距離 z、周期境界）。
- レンズ   : (far=True) フーリエ変換して遠方場で検出する。
- 検出器   : 出力面の C 個の窓の明るさ → logit（温度 gain と、任意で背景光 bias）。

Config には実験で試した多くのオプションがある（効かなかったものも含む）。良かった組み合わせは sinwave/api.py の PRESETS。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------- 符号化

def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def char_phase(c: str, m: int) -> float:
    """文字 c の波長 m における固定位相 (0..2π)。"""
    h = _splitmix(ord(c) * 1000003 + m * 7919)
    return 2 * math.pi * (h / 2**64)


def char_wave(c: str, m: int, S: int, enc: str) -> list[float]:
    """1文字が占める S 標本の位相列。
    phase: 全標本が同じ位相 θ(c)（S=1 なら元の方式）
    code : 標本ごとに独立な擬似乱数位相（固定の「字形」）
    freq : 文字固有の角速度 ω_c の搬送波 ω_c·s + θ(c)。ω_c = 2π·(文字コード mod S)/S"""
    if enc == "phase":
        return [char_phase(c, m)] * S
    if enc == "code":
        return [2 * math.pi * (_splitmix(ord(c) * 1000003 + m * 7919 + s * 104729) / 2**64) for s in range(S)]
    if enc == "freq":
        k = (ord(c) * (2 * m + 1)) % S   # 文字コードをそのまま角速度にする（S≥128 ならひらがなは衝突しない）
        th = char_phase(c, m)
        return [2 * math.pi * k * s / S + th for s in range(S)]
    raise ValueError(enc)


def encode_ids(texts: list[str], L: int, S: int):
    """文字列を「文字番号の表」にする。戻り値 (chars, ids)。ids: (B, ceil(L/S)) の LongTensor。
    文字列はレイヤ長まで繰り返し、L/S 文字を超える部分は使わない。"""
    if any(len(s) == 0 for s in texts):
        raise ValueError("空の文字列は符号化できない")
    chars = sorted(set("".join(texts)))
    cid = {c: i for i, c in enumerate(chars)}
    n_chars = -(-L // S)
    ids = torch.tensor([[cid[c] for c in (s * (n_chars // len(s) + 1))[:n_chars]] for s in texts])
    return chars, ids


def char_table(chars: list[str], M: int, S: int, enc: str, device) -> torch.Tensor:
    """各文字の S 標本の複素波形 (V, M, S)。"""
    table = torch.tensor([[char_wave(c, m, S, enc) for m in range(M)] for c in chars])
    return torch.polar(torch.ones_like(table), table).to(device)


def field_from_ids(ids: torch.Tensor, tab: torch.Tensor, L: int, lens: torch.Tensor | None = None) -> torch.Tensor:
    """文字番号の表と文字の波形から場 (B, M, L) を作る。lens を渡すと各文字列の長さ以降を暗く (0) する。"""
    w = tab[ids.to(tab.device)]                                   # (B, n_chars, M, S)
    out = w.permute(0, 2, 1, 3).reshape(w.shape[0], w.shape[2], -1)[..., :L]
    if lens is not None:
        out = out * (torch.arange(L, device=out.device)[None] < lens.to(out.device)[:, None])[:, None]
    return out


def encode(texts: list[str], L: int, M: int, device, S: int = 1, enc: str = "phase", repeat: bool = True) -> torch.Tensor:
    """文字列を位相場にする（1文字 = S 標本）。repeat=True なら長さ L まで繰り返し、
    False なら先頭に1回だけ置いて残りは暗く（0）する。戻り値 (B, M, L) complex64。"""
    chars, ids = encode_ids(texts, L, S)
    tab = char_table(chars, M, S, enc, device)
    lens = None if repeat else torch.tensor([min(len(s) * S, L) for s in texts])
    out = torch.empty(len(texts), M, L, dtype=torch.complex64, device=device)
    for i in range(0, len(texts), 4096):
        out[i:i + 4096] = field_from_ids(ids[i:i + 4096], tab, L, None if lens is None else lens[i:i + 4096])
    return out


# ---------------------------------------------------------------- レイヤ（ユニット）

class SinLayer(nn.Module):
    """振幅・角速度・初期位相だけを持つレイヤ。K 個の正弦の和。
    P>1 ならレイヤを P 区画に分け、区画ごとに (A, ω, φ) を持つ（P=1: レイヤ全体で1組、P=L: 要素ごと）。"""

    def __init__(self, K: int = 1, mode: str = "phase", L: int = 256, tag: str = "base",
                 gen: torch.Generator | None = None, P: int = 1):
        super().__init__()
        self.mode, self.tag, self.P = mode, tag, P
        # ω は 2π/L .. π の対数一様で初期化（どの空間周波数にも手がかりがあるように）
        lo, hi = math.log(2 * math.pi / L), math.log(math.pi)
        r = torch.rand(3, P, K, generator=gen)
        self.A = nn.Parameter(0.5 + 0.5 * r[0])
        self.w = nn.Parameter(torch.exp(lo + (hi - lo) * r[1]))
        self.phi = nn.Parameter(2 * math.pi * r[2])
        self.register_buffer("patch", (torch.arange(L) * P // L).long())

    def _params_at(self):
        # 各位置 t に対応する区画のパラメータ (L, K)
        return self.A[self.patch], self.w[self.patch], self.phi[self.patch]

    def wave(self, tt: torch.Tensor) -> torch.Tensor:
        """Σ_k A_k sin(ω_k t + φ_k)。
        tt: 位置を時間とみなすなら (1, L)、本物の時間なら (T, 1)。戻り値 (T or 1, L)"""
        A, w, phi = self._params_at()
        return (A * torch.sin(w * tt[..., None] + phi)).sum(-1)

    def forward(self, u: torch.Tensor, tt: torch.Tensor, lam_ratio: torch.Tensor) -> torch.Tensor:
        # u: (B, T, M, L)、lam_ratio: (M,) = λ0/λ_m
        if self.mode == "phase":
            ph = self.wave(tt)[None, :, None, :] * lam_ratio[None, None, :, None]
            return u * torch.polar(torch.ones_like(ph), ph)
        if self.mode == "amp":
            return u * (1 + self.wave(tt))[None, :, None, :]
        if self.mode == "add":
            A, w, phi = self._params_at()
            ph = w * tt[..., None] + phi
            ref = (A * torch.polar(torch.ones_like(ph), ph)).sum(-1)
            return u + ref[None, :, None, :]
        raise ValueError(self.mode)


# ---------------------------------------------------------------- 全体

@dataclass
class Config:
    L: int = 256            # レイヤ長（標本数）
    M: int = 1              # 波長（色）の数
    lam0: float = 1.0       # 基準波長 [標本間隔]
    lam_spread: float = 0.3  # 波長の広がり（M>1 のとき λ_m = lam0·(1 + spread·m/(M-1))）
    z: float = 64.0         # レイヤ間の伝搬距離 [標本間隔]
    C: int = 10             # 検出器（クラス）数
    det_width: int = 4      # 検出器の窓幅
    nonlin: float = 0.0     # >0 なら強度依存位相 (Kerr 風) をレイヤ毎に加える（任意）
    far: bool = False       # True: 最後に固定レンズを置き、遠方場（フーリエ面）で検出する
    bias: bool = False      # True: 検出器ごとの背景光（定数、C 個）を足す
    incoh: int = 0          # >0: 非コヒーレント光源。文字スロットごとにランダム位相を incoh 通り与えて平均する
    slot: int = 1           # 文字スロットの幅（= 符号化の S）
    T: int = 0              # 0: 位置 t を時間とみなす。>0: 本物の時間 τ=0..T-1 で振動させ、検出器で時間平均
    # ---- 光学的な改良の候補（実験10）
    z_sched: str = "const"  # const: 全層 z / geom: z/4 → 4z の等比 / learn: 層ごとに z を学習（4つ目のパラメータ）
    illum: str = "random"   # incoh>0 のときの照明パターン。random / tilt（入射角を変えた平面波）/ hadamard（0/π）
    feedback: str = ""      # "" / field: 出力の場を入力に足して2周 / phase: 出力強度で入力の位相を変えて2周（非線形）
    fb: float = 0.3         # 帰還の強さの初期値（学習する。スカラ1個）
    out_pad: int = 1        # >1: 最後にレイヤの out_pad 倍の広さの出力面へ z_out 伝搬して検出（周期境界なし）
    z_out: float = 0.0
    detlog: bool = False    # True: 検出器を対数応答にする（logit = gain·log 強度）
    diff: bool = False      # True: 差動検出。1クラスに検出器2つ (+/-)、logit = gain·(I+ − I−)/(I+ + I−)
    readout: str = "win"    # 検出の方法。win: 集光点の窓の明るさ /
                            #   fcos: 出力面全体を縞模様 cos(2π k_c x / L) の感度で読む（全部の光を使う、符号付き）/
                            #   fabs: 出力面の明るさの模様をフーリエ変換した振幅 |FT(I)|(k_c) を読む（場の自己相関、位置不変）
    readout_k0: int = 1     # 読む周波数 k_c = k0 + c·dk
    readout_dk: int = 1
    pat_w: str = ""         # 照明パターンの平均の重み（逆拡散の重み付け）。"" 均等 / adj: 隣のスロットどうしの干渉だけ残す /
                            #   learn: 重みを学習する（パターン数個、非負）。tilt 照明と組み合わせる
    pat_a: float = 0.5      # adj のとき、隣どうしの干渉を残す割合（0〜0.5）
    pre_layers: int = 0     # >0: 背景光 → 前 Sin レイヤ（前処理光）→ 入力 → 後 Sin レイヤ。前段の層数
    pre_P: int = 64         # 前段レイヤの区画数
    pre_A0: float = 1.0     # 前段レイヤの振幅の初期倍率（小さくすると「照明そのまま」から学習が始まる）
    blur: str = ""          # 滲み層（入力と最初のレイヤの間）。mean: 前後 r 文字の平均（効く）/
                            #   std: 前後 r 文字とのばらつき（標準偏差）/ ustd: 元の場 × ばらつき /
                            #   cstd: 前の r 文字と自分だけのばらつき（因果的）/ diff: 直前の文字との差 u_n − u_{n−1}
    blur_r: int = 1         # 滲み層の半径（文字数）
    blur_mix: bool = False  # True: 色0 に元の入力、色1 に滲みを載せる（M≥2 が必要。色どうしは強度で足される）
    input_ft: bool = False  # True: 入力（文字列の波形）をフーリエ変換してからレイヤへ渡す
    lens_every: int = 0     # >0: その層数ごとに固定レンズ（フーリエ変換）を挟み、実空間と周波数空間を交互にする


class SinWave(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        self.gain = nn.Parameter(torch.tensor(5.0))  # logit の温度（スカラ1個）
        L, M = cfg.L, cfg.M
        lams = torch.tensor([cfg.lam0 * (1 + cfg.lam_spread * m / max(M - 1, 1)) for m in range(M)])
        self.register_buffer("lam_ratio", cfg.lam0 / lams)
        self.register_buffer("t", torch.arange(L, dtype=torch.float32))
        tt = torch.arange(cfg.T, dtype=torch.float32)[:, None] if cfg.T > 0 else self.t[None]
        self.register_buffer("tt", tt)
        # 角スペクトル伝搬の伝達関数 H_m(f) = exp(j z sqrt(k^2 - kx^2))、エバネッセントは減衰
        kx = 2 * math.pi * torch.fft.fftfreq(L)
        k = (2 * math.pi / lams)[:, None]
        kz2 = k**2 - kx[None] ** 2
        kz = torch.sqrt(kz2.clamp(min=0))
        decay = torch.sqrt((-kz2).clamp(min=0))
        self.register_buffer("kz", kz)
        self.register_buffer("decay", decay)
        H = torch.polar(torch.exp(-decay * cfg.z), kz * cfg.z)
        self.register_buffer("H", H)
        self.L_out = L * cfg.out_pad
        if cfg.out_pad > 1:   # 広い出力面への伝搬（零詰めして周期境界の影響をなくす）
            kxo = 2 * math.pi * torch.fft.fftfreq(self.L_out)
            kz2o = k**2 - kxo[None] ** 2
            Ho = torch.polar(torch.exp(-torch.sqrt((-kz2o).clamp(min=0)) * cfg.z_out),
                             torch.sqrt(kz2o.clamp(min=0)) * cfg.z_out)
            self.register_buffer("H_out", Ho)
        self.set_detectors(2 * cfg.C if cfg.diff else cfg.C)
        if cfg.incoh > 0:
            n_slots = -(-L // cfg.slot)
            j = torch.arange(cfg.incoh, dtype=torch.float32)[:, None]
            sl = torch.arange(n_slots, dtype=torch.float32)[None]
            if cfg.illum == "random":
                g = torch.Generator().manual_seed(12345)
                psi = 2 * math.pi * torch.rand(cfg.incoh, n_slots, generator=g)
            elif cfg.illum == "tilt":      # j 番目の照明 = スロットあたり 2πj/n 傾いた平面波
                psi = 2 * math.pi * j * sl / n_slots
            elif cfg.illum == "hadamard":  # ウォルシュ・アダマール符号（0/π）
                bits = (j.long() & sl.long())
                par = torch.zeros_like(bits)
                for b in range(12):
                    par ^= (bits >> b) & 1
                psi = math.pi * par.float()
            else:
                raise ValueError(cfg.illum)
            psi = psi.repeat_interleave(cfg.slot, -1)[:, :L]
            self.register_buffer("psi", torch.polar(torch.ones_like(psi), psi))   # (incoh, L)
        self.pre = nn.ModuleList()
        if cfg.pre_layers > 0:
            g = torch.Generator().manual_seed(777)
            for _ in range(cfg.pre_layers):
                layer = SinLayer(1, "phase", L, "pre", g, cfg.pre_P)
                layer.A.data *= cfg.pre_A0
                self.pre.append(layer)
        if cfg.incoh > 0 and cfg.pat_w:
            n = cfg.incoh
            j = torch.arange(n, dtype=torch.float32)
            if cfg.pat_w == "adj":
                # tilt 照明で重み w_j = 1 + 2a·cos(2πj/n) とすると、スロット間の相関は
                # K(Δ) = Σ_j w_j e^{i2πjΔ/n} / Σ w_j → Δ=0 で 1、|Δ|=1 で a、それ以外 0（隣どうしだけ干渉が残る）
                self.register_buffer("pw", (1 + 2 * cfg.pat_a * torch.cos(2 * math.pi * j / n)))
            elif cfg.pat_w == "learn":
                self.pw_logit = nn.Parameter(torch.zeros(n))
            else:
                raise ValueError(cfg.pat_w)
        if cfg.feedback:
            self.fb_beta = nn.Parameter(torch.tensor(cfg.fb))
        self.bias = nn.Parameter(torch.zeros(cfg.C), requires_grad=cfg.bias)

    def set_detectors(self, C: int):
        """出力面に C 個の窓（幅 det_width）を等間隔に置く。"""
        L, w = self.L_out, self.cfg.det_width
        centers = [int((i + 0.5) * L / C) for i in range(C)]
        mask = torch.zeros(C, L)
        for i, c in enumerate(centers):
            mask[i, (torch.arange(w) + c - w // 2) % L] = 1.0
        self.register_buffer("det", mask.to(self.t.device))

    def add_layers(self, n: int, K: int = 1, mode: str = "phase", tag: str = "base", seed: int = 0, P: int = 1):
        g = torch.Generator().manual_seed(seed)
        for _ in range(n):
            layer = SinLayer(K, mode, self.cfg.L, tag, g, P).to(self.t.device)
            if self.cfg.z_sched == "learn":
                layer.logz = nn.Parameter(torch.tensor(math.log(self.cfg.z), device=self.t.device))
            self.layers.append(layer)

    def init_w(self, kind: str, tag: str | None = None):
        """角速度 ω の初期配置を決め直す（区間 [2π/L, π] に写す）。
        logrand: 対数一様の乱数（既定）/ rand: 一様の乱数 / grid: 等間隔 / prime: 素数 p_n /
        zeta: リーマンゼータのゼロ点の高さ t_n（間隔が反発しあい、近づきすぎない）。
        各層が全帯域の値を持つよう、小さい順の値を層に交互に配る。"""
        if kind == "logrand":
            return
        layers = [l for l in self.layers if tag is None or l.tag == tag]
        N = sum(l.w.numel() for l in layers)
        if kind == "rand":
            v = torch.rand(N, generator=torch.Generator().manual_seed(4321))
        elif kind == "grid":
            v = torch.arange(N, dtype=torch.float64)
        elif kind == "prime":
            ps, x = [], 2
            while len(ps) < N:
                if all(x % p for p in ps if p * p <= x):
                    ps.append(x)
                x += 1
            v = torch.tensor(ps, dtype=torch.float64)
        elif kind == "zeta":
            import pathlib
            zs = [float(l) for l in (pathlib.Path(__file__).parent / "zeta_zeros.txt").read_text().split()]
            assert len(zs) >= N, f"ゼロ点が足りない ({len(zs)} < {N})。tools/zeta_zeros.py で増やす"
            v = torch.tensor(zs[:N], dtype=torch.float64)
        else:
            raise ValueError(kind)
        v = (v - v.min()) / (v.max() - v.min())
        lo, hi = 2 * math.pi / self.cfg.L, math.pi
        v = (lo + (hi - lo) * v).sort().values.float()
        nl = len(layers)
        for d, l in enumerate(layers):
            vals = v[d::nl][: l.w.numel()]   # 層 d には d, d+nl, d+2nl, ... 番目の値
            perm = torch.randperm(vals.numel(), generator=torch.Generator().manual_seed(99 + d))
            vals = vals[perm]                 # 区画の並びは全方式で共通の固定順列（空間的なチャープを作らない）
            l.w.data = vals.reshape(l.w.shape).to(l.w.device)

    def layer_H(self, d, layer):
        cfg = self.cfg
        if cfg.z_sched == "const":
            return self.H
        if cfg.z_sched == "geom":
            n = len(self.layers)
            z = cfg.z * 4 ** (2 * d / max(n - 1, 1) - 1)
        elif cfg.z_sched == "learn":
            z = layer.logz.exp()
        else:
            raise ValueError(cfg.z_sched)
        return torch.polar(torch.exp(-self.decay * z), self.kz * z)

    def propagate(self, u, H=None):
        return torch.fft.ifft(torch.fft.fft(u) * (self.H if H is None else H)[None], dim=-1)

    def field(self, u, active=None):
        for d, layer in enumerate(self.layers):
            if active is not None and layer.tag not in active:
                continue
            u = self.propagate(layer(u, self.tt, self.lam_ratio), self.layer_H(d, layer))
            if self.cfg.lens_every and (d + 1) % self.cfg.lens_every == 0:
                u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
            # 増幅: レイヤを通るたびに平均強度を 1 に戻す（減衰・発散で値が潰れないように。位相の情報は変えない）
            u = u / (u.abs().pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-12)
            if self.cfg.nonlin > 0:
                inten = u.abs() ** 2
                u = u * torch.polar(torch.ones_like(inten), self.cfg.nonlin * inten / inten.mean(-1, keepdim=True))
        return u

    def intensity(self, u0, active=None):
        """出力面の強度 (B, L)。平均 1 に正規化。"""
        u = u0[:, None]
        if len(self.pre) > 0:    # 背景光を前段の Sin レイヤで整形してから、入力（透過マスク）に当てる
            u = u * self.illumination()[None]
        elif self.cfg.incoh > 0:   # 位相のばらつきを時間方向に並べる（T と同じ扱いで最後に平均される）
            u = u * self.psi[None, :, None, :]
        u = self.preprocess(u)
        u_in = u
        u = self.field(u, active)                          # (B, T, M, L)
        if self.cfg.feedback == "field":     # 出力光の一部を入力側の背景光として戻し、もう一周
            u = self.field(u_in + self.fb_beta * u, active)
        elif self.cfg.feedback == "phase":   # 出力の強度に応じて入力の位相を変え、もう一周（非線形）
            I = u.abs() ** 2
            I = I / I.mean(-1, keepdim=True)
            u = self.field(u_in * torch.polar(torch.ones_like(I), self.fb_beta * I), active)
        if self.cfg.out_pad > 1:            # レイヤより広い出力面へ伝搬
            pad = (self.L_out - self.cfg.L) // 2
            u = torch.nn.functional.pad(u, (pad, self.L_out - self.cfg.L - pad))
            u = torch.fft.ifft(torch.fft.fft(u) * self.H_out[None], dim=-1)
        if self.cfg.far:
            u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)  # 集光レンズ
        inten = (u.abs() ** 2).sum(2)                      # 波長は非干渉に足す (B, T, L)
        if self.cfg.incoh > 0 and self.cfg.pat_w:          # 照明パターンの重み付き平均（逆拡散の重み付け）
            w = self.pw if self.cfg.pat_w == "adj" else torch.nn.functional.softplus(self.pw_logit) + 1e-3
            inten = (inten * (w / w.sum())[None, :, None]).sum(1)
        else:
            inten = inten.mean(1)                          # 時間（照明パターン）は平均 (B, L)
        return inten / (inten.mean(-1, keepdim=True) + 1e-12)

    def preprocess(self, u):
        """入力と最初のレイヤの間の処理（滲み層・入力のフーリエ変換）。u: (B, T, M, L)"""
        cfg = self.cfg
        if cfg.blur:
            S, r = cfg.slot, cfg.blur_r
            src = u[:, :, :1] if cfg.blur_mix else u
            if cfg.blur == "diff":
                v = (src - torch.roll(src, S, dims=-1)) / 2
            else:
                ks = range(0, r + 1) if cfg.blur == "cstd" else range(-r, r + 1)   # roll(+kS) = k 文字前
                nb = torch.stack([torch.roll(src, k * S, dims=-1) for k in ks])     # 前後 r 文字（巡回）
                mean = nb.mean(0)
                if cfg.blur == "mean":
                    v = mean
                else:
                    std = ((nb.abs().pow(2).mean(0) - mean.abs().pow(2)).clamp(min=0) + 1e-4).sqrt()  # 全部同じ文字でも 0 除算しない
                    v = src * std if cfg.blur == "ustd" else std.to(u.dtype)
            if cfg.blur_mix:
                u = torch.cat([u[:, :, :1], v, u[:, :, 2:]], dim=2)
            else:
                u = v
        if cfg.input_ft:
            u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
        return u

    def illumination(self):
        """前処理光 (P_illum, M, L)。源は平面波（incoh>0 なら照明パターンごと）。入力に依存しない。"""
        src = self.psi if self.cfg.incoh > 0 else torch.ones(1, self.cfg.L, dtype=torch.complex64, device=self.t.device)
        v = src[None, :, None, :].expand(1, -1, self.cfg.M, -1)
        for d, layer in enumerate(self.pre):
            v = self.propagate(layer(v, self.tt, self.lam_ratio))
            v = v / (v.abs().pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-12)
        return v[0]

    def detect(self, inten):
        if self.cfg.readout != "win":
            C = self.det.shape[0]
            ks = torch.arange(C, device=inten.device) * self.cfg.readout_dk + self.cfg.readout_k0
            spec = torch.fft.rfft(inten, dim=-1)[:, ks] / inten.shape[-1]      # 平均強度 1 で正規化済み
            f = spec.real * 2 if self.cfg.readout == "fcos" else spec.abs() * 2
            return self.gain * f + self.bias
        f = inten @ self.det.T / self.cfg.det_width
        if self.cfg.diff:
            fp, fm = f[:, 0::2], f[:, 1::2]
            f = (fp - fm) / (fp + fm + 1e-6)
        if self.cfg.detlog:
            f = torch.log(f + 1e-3)
        return self.gain * f + self.bias   # (B, C)

    def forward(self, u0, active=None):
        return self.detect(self.intensity(u0, active))

    def n_params(self):
        return sum(p.numel() for p in self.layers.parameters()) + sum(p.numel() for p in self.pre.parameters())


class Parallel(nn.Module):
    """互いに干渉しない光路（色・偏光などを想定）を並べ、検出器で強度を足す。
    融合 = 光路を足す、取消 = 光路を外す。各光路は独立に学習できる。
    mode="gate" を使うときは、各光路に担当する検出器の番号の列 own を持たせておく（例: path.own = [0, 1, 2, 3]）。"""

    def __init__(self, paths: list[SinWave]):
        super().__init__()
        self.paths = nn.ModuleList(paths)

    def forward(self, u0, active=None, mode: str = "avg"):
        paths = [p for p in self.paths if active is None or p.layers[0].tag in active]
        if mode == "avg":
            outs = [p.detect(p.intensity(u0)) for p in paths]
            return sum(outs) / len(outs)   # 強度の平均を検出するのと同じ（各光路の温度・背景光も平均される）
        if mode == "gate":
            # 各光路について「自分の担当検出器に入った光の割合 r」を技能の確信度にする。
            # スコア = log r_i + (光路 i の担当検出器内での log-softmax)。担当外のクラスは -inf。
            C = paths[0].det.shape[0]
            out = torch.full((u0.shape[0], C), -1e9, device=u0.device)
            for p in paths:
                inten = p.intensity(u0)
                f = inten @ p.det.T / p.cfg.det_width                # (B, C) 各検出器の強度
                if not hasattr(p, "own"):
                    raise ValueError("gate には各光路の担当検出器 own が必要（path.own = [...]）")
                own = torch.tensor(p.own, device=u0.device)
                r = f[:, own].sum(1) / (f.sum(1) + 1e-9)
                lg = p.detect(inten)[:, own]
                out[:, own] = torch.log(r + 1e-9)[:, None] + torch.log_softmax(lg, 1)
            return out
        raise ValueError(mode)
