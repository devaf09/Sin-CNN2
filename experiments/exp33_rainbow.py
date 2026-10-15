"""実験33a: 文字の位置に合わせて色を少しずつ変える「虹」（ユーザー案、2026-10-16）。

(1) 虹のコヒーレンス (--mix rainbow): 位置 m の文字の光の周波数を m·Δ だけずらし、検出器で時間平均する。
    色の差が小さい隣どうしは干渉が残り、離れた文字どうしは干渉が消える（うなりが平均で打ち消される）。
    時刻 τ = 0..T-1 で位置 m の光に位相 2π·Δ·m·τ/T を与え、明るさを平均する。
    スロット間の相関は K(d) = |mean_τ exp(i2πΔdτ/T)|。Δ が小さいほど遠くまで干渉が残る（Δ=1 で全部消える）。
(2) 虹の分散 (--mix disp): 位置 m の文字を、波長 λ_m = λ0·(1 + spread·m/(n−1)) の光にする（色が違うので干渉しない）。
    Sin の板の位相（λ0/λ 倍）と回折の仕方が、位置ごとに少しずつ変わる。
文字の入れ方・課題・学習は実験30 と同じ（exp30_fullaperture.py の Encoder, load_task, make_fields を使う）。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from exp30_fullaperture import Encoder, load_task, make_fields
from sinwave.model import Config, SinWave


def fields(T_, a, n, dev):
    """(B, T, M, L) の光の場"""
    codes, theta, pos, mask = T_
    if a.mix in ("rainbow", "disp"):
        w = make_fields(codes, theta, pos, mask, a.L, n, a.pos, "color", a.d0, dev)   # 文字ごとの波 (B, n, L)
        if a.mix == "disp":
            return w[:, None]                                                         # 位置 m → 波長チャネル m
        tau = torch.arange(a.T, device=dev, dtype=torch.float32)
        code = torch.exp(1j * (2 * math.pi * a.rb * pos.to(dev).float()[:, None, :] * tau[None, :, None] / a.T))
        return torch.einsum("btn,bnl->btl", code, w)[:, :, None, :]
    return make_fields(codes, theta, pos, mask, a.L, n, a.pos, a.mix, a.d0, dev)[:, :, None, :]


def forward(model, U):
    u = model.field(U)
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)                   # レンズ
    I = u.abs().pow(2).sum(2).mean(1)                                                 # 波長は足し、時間は平均
    return model.detect(I / (I.mean(-1, keepdim=True) + 1e-12))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="order")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--pos", default="phase")
    ap.add_argument("--mix", default="rainbow")
    ap.add_argument("--rb", type=float, default=0.5, help="虹: 位置 1 つあたりの周波数のずれ（時間窓あたりの周期数）")
    ap.add_argument("--T", type=int, default=8, help="虹: 時間の標本数")
    ap.add_argument("--spread", type=float, default=0.05, help="分散: 波長の広がり")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--d0", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--freeze_w", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    xtr, ytr, xte, yte, C, n, cd, is_text = load_task(a.task, a.n)
    if a.max_train:
        xtr, ytr, xte, yte = xtr[:a.max_train], ytr[:a.max_train], xte[:a.max_train // 2], yte[:a.max_train // 2]
    steps = a.steps or (3000 if is_text else 1500)
    enc = Encoder(n, cd["V"], cd["code"], xtr, from_end=is_text)
    Ttr, Tte = enc.table(xtr), enc.table(xte)
    Ytr, Yte = torch.tensor(ytr), torch.tensor(yte)
    M = n if a.mix == "disp" else 1
    cfg = Config(L=a.L, M=M, lam_spread=a.spread, C=C, far=True, bias=is_text)
    model = SinWave(cfg).to(dev)
    model.add_layers(a.D, 1, "phase", seed=a.seed, P=a.P)
    model.bias.requires_grad_(is_text)
    if a.freeze_w:
        for l in model.layers:
            l.w.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(steps):
        i = torch.randint(len(Ytr), (min(256, len(Ytr)),), generator=gen)
        loss = F.cross_entropy(forward(model, fields(tuple(t[i] for t in Ttr), a, n, dev)), Ytr[i].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    @torch.no_grad()
    def evaluate(T_, Y):
        lg = torch.cat([forward(model, fields(tuple(t[j:j + 256] for t in T_), a, n, dev)) for j in range(0, len(Y), 256)]).cpu()
        return round((lg.argmax(1) == Y).float().mean().item(), 4), round(F.cross_entropy(lg, Y).item(), 4)

    # 位置が d 離れた文字どうしに残る干渉の割合 K(d)（d = 0..3）
    kern = [round(abs(sum(complex(math.cos(x), math.sin(x)) for x in (2 * math.pi * a.rb * d * t / a.T for t in range(a.T)))) / a.T, 3)
            for d in range(4)] if a.mix == "rainbow" else None
    print(json.dumps(dict(vars(a), n_pos=n, kernel=kern, train=evaluate(Ttr, Ytr)[0], test=evaluate(Tte, Yte),
                          sec=round(time.time() - t0, 1)), ensure_ascii=False), flush=True)
