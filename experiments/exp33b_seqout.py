"""実験33b: 出力を「順番つきの文字列」として取り出す（ユーザー案、2026-10-16）。

課題: 16 種類の文字からなる長さ n=4 の文字列。
  copy : 各位置 m で、入力の m 番目の文字を答える（写し取り）
  shift: 各位置 m で、入力の (m+1) 番目の文字を答える（ずらし。別の位置の情報が要る）
検出器: 出力面に n×16 個の窓。位置 m の答えは m 番目の 16 個の窓で読む。

取り出し方 (--read)
  shared: 全部の文字を 1 つの光で干渉させ、1 つの出力面で全位置を読む（文字と位置の組ごとの向きで入れる）
  color : 位置ごとに別の色。出力側で色を分け、色 m の光で位置 m を読む（波長分割）
  mix   : code の取り出し側の符号を学習する（実験37。入力のどの位置を出力のどの位置へ送るかの配線）
  code  : 位置 m に位相の符号 c_j(m) = exp(i2πjm/n) を付けた照明 n 通り。出力の場に符号の逆を掛けて足し、
          位置 m の光だけを取り出して読む（CDMA の逆拡散。光学系は場に対して線形なので、通った後で分けられる）
予想: color と code は「その位置の入力 → その位置の出力」しか作れない（他の位置の情報が混ざらない）ので、
      copy は解けるが shift は解けない。shared は両方解ける可能性がある。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from sinwave.model import Config, SinWave, char_phase

V, N = 16, 4
CHARS = [chr(0x3041 + i) for i in range(V)]   # 文字コード mod 16 がすべて異なる 16 字


def make_data(n_samples, seed, task):
    r = random.Random(seed)
    X = torch.tensor([[r.randrange(V) for _ in range(N)] for _ in range(n_samples)])
    Y = {"copy": X, "shift": torch.roll(X, -1, dims=1), "reverse": X.flip(1)}[task]
    return X, Y


def waves(X, L, dev, pair):
    """文字ごとの波 (B, N, L)。pair なら文字と位置の組ごとの向き、そうでなければ文字の向きだけ。"""
    x = torch.arange(L, device=dev, dtype=torch.float32)
    pos = torch.arange(N, device=dev)[None]
    codes = torch.tensor([ord(c) % V for c in CHARS], device=dev)[X.to(dev)]
    th = torch.tensor([char_phase(c, 0) for c in CHARS], device=dev)[X.to(dev)]
    f = codes * N + (pos if pair else 0)
    return torch.exp(1j * (2 * math.pi * f[..., None].float() * x / L + th[..., None])) / math.sqrt(N)


def readout(model, X, mode, L, dev, E=None):
    """(B, N, V) の点数"""
    def lens_detect(u):                       # u: (..., L) → 全窓の点数 (..., N*V)
        u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
        I = u.abs().pow(2)
        I = I / (I.mean(-1, keepdim=True) + 1e-12)
        return model.detect(I.reshape(-1, L)).reshape(*I.shape[:-1], N, V)

    if mode == "shared":
        u = model.field(waves(X, L, dev, True).sum(1)[:, None, None])[:, 0, 0]       # (B, L)
        return lens_detect(u)
    w = waves(X, L, dev, False)
    if mode == "color":
        u = model.field(w[:, :, None])[:, :, 0]                                       # 位置ごとに別の色 (B, N, L)
        s = lens_detect(u)                                                            # (B, N色, N, V)
        return torch.stack([s[:, m, m] for m in range(N)], 1)                         # 色 m の光で位置 m を読む
    if mode in ("code", "mix"):
        j = torch.arange(N, device=dev, dtype=torch.float32)
        Cm = torch.exp(1j * (2 * math.pi * j[:, None] * j[None, :] / N))             # c_j(m) (j, m)
        U = torch.einsum("jm,bml->bjl", Cm, w)                                        # 照明 j の場
        u = model.field(U[:, :, None])[:, :, 0]                                       # 光学系を通す (B, j, L)
        D = Cm.conj() if mode == "code" else torch.complex(E[0], E[1])                # mix: 取り出しの符号を学習（実験37）
        v = torch.einsum("jm,bjl->bml", D, u) / N                                     # 逆拡散: code なら位置 m の光だけ
        s = lens_detect(v)
        return torch.stack([s[:, m, m] for m in range(N)], 1)
    raise ValueError(mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="copy")
    ap.add_argument("--read", default="shared")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    Xtr, Ytr = make_data(2000, a.seed + 1, a.task)
    Xte, Yte = make_data(1000, a.seed + 2, a.task)
    model = SinWave(Config(L=a.L, C=N * V, far=True)).to(dev)
    model.add_layers(a.D, 1, "phase", seed=a.seed, P=a.P)
    # 実験37 (mix): 出力の位置 k を読む符号の組 E (N×N の複素数) を学習する。入力と同じ符号で初期化。
    # 入力側の符号とずらせば「入力のどの位置の光を、出力のどの位置へ送るか」の配線になる（パラメータ 32 個）
    j0 = torch.arange(N, dtype=torch.float32)
    C0 = torch.exp(1j * (2 * math.pi * j0[:, None] * j0[None, :] / N)).conj()
    E = torch.nn.Parameter(torch.stack([C0.real, C0.imag]).to(dev)) if a.read == "mix" else None
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad] + ([E] if E is not None else []), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(a.steps):
        i = torch.randint(len(Ytr), (256,), generator=gen)
        out = readout(model, Xtr[i], a.read, a.L, dev, E)
        loss = F.cross_entropy(out.reshape(-1, V), Ytr[i].to(dev).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = torch.cat([readout(model, Xte[j:j + 250], a.read, a.L, dev, E).argmax(-1).cpu() for j in range(0, len(Yte), 250)])
    per_pos = (pred == Yte).float().mean(0)
    print(json.dumps(dict(vars(a), per_position=[round(v, 3) for v in per_pos.tolist()],
                          char_acc=round((pred == Yte).float().mean().item(), 4),
                          string_acc=round((pred == Yte).all(1).float().mean().item(), 4),
                          sec=round(time.time() - t0, 1)), ensure_ascii=False), flush=True)
