"""正弦の干渉・回折による学習器 (SinWave)。

構成
----
- 入力: 文字列をレイヤ長 L まで繰り返し、各文字を固定の位相 θ(c) に変換した複素場 u0[t] = exp(jθ(c_t))。
  θ は文字コードのハッシュで決まり、学習しない（「字形」に相当する固定の符号化）。
  M 波長（色）を使う場合は波長ごとに独立なハッシュで位相を作る。
- レイヤ: 各レイヤは (振幅 A, 角速度 ω, 初期位相 φ) だけを持つ。位置 t を時間とみなし
      phase 型: u ← u · exp(j · A·sin(ωt+φ) · λ0/λ)     (位相変調 = 乗算)
      amp   型: u ← u · (1 + A·sin(ωt+φ))              (振幅変調 = 乗算)
      add   型: u ← u + A · exp(j(ωt+φ))                (参照波の注入 = 加算)
  1レイヤに K 個の正弦を持たせることもできる（既定 K=1）。重み行列は持たない。
- 伝搬: レイヤ間は角スペクトル法による自由空間伝搬（周期境界、固定。学習しない）。
- 出力: 出力面の C 個の窓（検出器）に集まった強度 |u|^2 を特徴とし、正規化して logit にする。

学習・追加・取消・融合
--------------------
レイヤは「ユニット」の列。ユニットには task タグを付けられ、凍結/削除/結合が自由にできる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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


def encode(texts: list[str], L: int, M: int, device, S: int = 1, enc: str = "phase") -> torch.Tensor:
    """文字列を長さ L まで繰り返して位相場にする（1文字 = S 標本）。戻り値 (B, M, L) complex64。"""
    chars = sorted(set("".join(texts)))
    cid = {c: i for i, c in enumerate(chars)}
    table = torch.tensor([[char_wave(c, m, S, enc) for m in range(M)] for c in chars])  # (V, M, S)
    n_chars = -(-L // S)
    ids = torch.tensor([[cid[c] for c in (s * (n_chars // len(s) + 1))[:n_chars]] for s in texts])
    out = torch.empty(len(texts), M, L, dtype=torch.complex64, device=device)
    tab = torch.polar(torch.ones_like(table), table).to(device)
    for i in range(0, len(texts), 4096):
        w = tab[ids[i:i + 4096].to(device)]                 # (b, n_chars, M, S)
        out[i:i + 4096] = w.permute(0, 2, 1, 3).reshape(w.shape[0], M, -1)[..., :L]
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
        H = torch.polar(torch.exp(-decay * cfg.z), kz * cfg.z)
        self.register_buffer("H", H)
        self.set_detectors(cfg.C)
        if cfg.incoh > 0:
            g = torch.Generator().manual_seed(12345)
            n_slots = -(-L // cfg.slot)
            psi = 2 * math.pi * torch.rand(cfg.incoh, n_slots, generator=g)
            psi = psi.repeat_interleave(cfg.slot, -1)[:, :L]
            self.register_buffer("psi", torch.polar(torch.ones_like(psi), psi))   # (incoh, L)
        self.bias = nn.Parameter(torch.zeros(cfg.C), requires_grad=cfg.bias)

    # 検出器：出力面に C 個の窓を等間隔に置く
    def set_detectors(self, C: int, offset: int = 0, n_total: int | None = None):
        n_total = n_total or C
        L, w = self.cfg.L, self.cfg.det_width
        centers = [int((i + 0.5) * L / n_total) for i in range(offset, offset + C)]
        mask = torch.zeros(C, L)
        for i, c in enumerate(centers):
            mask[i, (torch.arange(w) + c - w // 2) % L] = 1.0
        self.register_buffer("det", mask.to(self.t.device))

    def add_layers(self, n: int, K: int = 1, mode: str = "phase", tag: str = "base", seed: int = 0, P: int = 1):
        g = torch.Generator().manual_seed(seed)
        for _ in range(n):
            self.layers.append(SinLayer(K, mode, self.cfg.L, tag, g, P).to(self.t.device))

    def propagate(self, u):
        return torch.fft.ifft(torch.fft.fft(u) * self.H[None], dim=-1)

    def field(self, u, active=None):
        for layer in self.layers:
            if active is not None and layer.tag not in active:
                continue
            u = self.propagate(layer(u, self.tt, self.lam_ratio))
            # 増幅: レイヤを通るたびに平均強度を 1 に戻す（減衰・発散で値が潰れないように。位相の情報は変えない）
            u = u / (u.abs().pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-12)
            if self.cfg.nonlin > 0:
                inten = u.abs() ** 2
                u = u * torch.polar(torch.ones_like(inten), self.cfg.nonlin * inten / inten.mean(-1, keepdim=True))
        return u

    def intensity(self, u0, active=None):
        """出力面の強度 (B, L)。平均 1 に正規化。"""
        u = u0[:, None]
        if self.cfg.incoh > 0:   # 位相のばらつきを時間方向に並べる（T と同じ扱いで最後に平均される）
            u = u * self.psi[None, :, None, :]
        u = self.field(u, active)                          # (B, T, M, L)
        if self.cfg.far:
            u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)  # 集光レンズ
        inten = (u.abs() ** 2).sum(2).mean(1)              # 波長は非干渉に足し、時間は平均 (B, L)
        return inten / inten.mean(-1, keepdim=True)

    def detect(self, inten):
        return self.gain * (inten @ self.det.T / self.cfg.det_width) + self.bias   # (B, C)

    def forward(self, u0, active=None):
        return self.detect(self.intensity(u0, active))

    def n_params(self):
        return sum(p.numel() for p in self.layers.parameters())


class Parallel(nn.Module):
    """互いに干渉しない光路（色・偏光などを想定）を並べ、検出器で強度を足す。
    融合 = 光路を足す、取消 = 光路を外す。各光路は独立に学習できる。"""

    def __init__(self, paths: list[SinWave]):
        super().__init__()
        self.paths = nn.ModuleList(paths)

    def forward(self, u0, active=None):
        outs = [p.detect(p.intensity(u0)) for p in self.paths if active is None or p.layers[0].tag in active]
        return sum(outs) / len(outs)   # 強度の平均を検出するのと同じ（各光路の温度・背景光も平均される）
