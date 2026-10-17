"""実験52: うなり読みで実テキストの次の文字を予測する（実験51 の方式を CC-100 日本語に使う）。

文脈の n 文字を虹で一度に入れる（位置 m の文字の光の周波数を時間とともに rb·m だけずらす）。
文字は平面波（出現順位の上位 V−1 字 + その他、周波数 = 番号·g mod L、実験41 と同じ）。
出力は 64 クラス（頻出 63 字 + その他）、窓は各クラスの文字の周波数の位置（実験41 と同じ）。
読み方:
  beat : 明るさを時間方向にフーリエ変換し、平均（成分 0）と、うなりの成分 d=1..dmax の実部・虚部を窓ごとの重みで足す
  mean : 平均の明るさだけ（今の虹）。rb=0, T=1 なら全部干渉（coh）
比べる相手: 実験41 の光の記憶（同じ大きさ）、n-gram。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import math
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp06_nextchar import load
from sinwave.model import Config, SinWave, char_phase


def coprime_stride(L):
    g = int(L * 0.6180339887) | 1
    while math.gcd(g, L) != 1:
        g += 2
    return g


ap = argparse.ArgumentParser()
ap.add_argument("--read", default="beat")
ap.add_argument("--n", type=int, default=8, help="文脈の文字数")
ap.add_argument("--rb", type=float, default=1.0)
ap.add_argument("--T", type=int, default=20)
ap.add_argument("--dmax", type=int, default=7)
ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--V", type=int, default=512)
ap.add_argument("--D", type=int, default=16)
ap.add_argument("--P", type=int, default=64)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--max_train", type=int, default=60000)
ap.add_argument("--decay", type=float, default=1.0, help="位置ごとの明るさ: 直前の文字を 1、k 文字前を decay^k（光の記憶と同じ考え方）")
ap.add_argument("--learn_amp", action="store_true", help="位置ごとの明るさを学習する（n 個）")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
data = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "cc100_ja.txt")
tr, te, vocab, tr_docs, idx = load(data, a.n, 64, a.max_train, 10000, a.seed)
rank = {c: i for i, (c, _) in enumerate(Counter("".join(tr_docs)).most_common(a.V - 1))}
code = lambda c: rank.get(c, a.V - 1)
Xtr = torch.tensor([[code(c) for c in x] for x, _, _ in tr])
Ytr = torch.tensor([y for _, y, _ in tr])
Xte = torch.tensor([[code(c) for c in x] for x, _, _ in te])
Yte = torch.tensor([y for _, y, _ in te])

L = a.L
g = coprime_stride(L)
freq = torch.tensor([(k * g) % L for k in range(a.V)], device=dev)
chars_by_code = {v: k for k, v in rank.items()}
theta = torch.tensor([char_phase(chars_by_code.get(k, "〓"), 0) for k in range(a.V)], device=dev)
x = torch.arange(L, device=dev, dtype=torch.float32)
waves = torch.exp(1j * (2 * math.pi * freq[:, None] * x / L + theta[:, None]))            # (V, L)
m = torch.arange(a.n, device=dev, dtype=torch.float32)
tau = torch.arange(a.T, device=dev, dtype=torch.float32)
code_t = torch.exp(1j * (2 * math.pi * (a.rb * m[None, :] * tau[:, None] / a.T + 0.6180339887 * m[None, :])))   # (T, n)

age = (a.n - 1 - m)                                         # 0 = 直前の文字
amp = nn.Parameter((a.decay ** age).clone(), requires_grad=a.learn_amp)
model = SinWave(Config(L=L, C=64, far=True)).to(dev)
model.add_layers(a.D, 1, "phase", seed=a.seed, P=a.P)
for l in model.layers:
    l.w.requires_grad_(False)
det = torch.zeros(64, L, device=dev)                       # クラス k の窓 = その文字の周波数の位置
for k in range(64):
    f = freq[code(vocab[k]) if k < 63 else a.V - 1].item()
    c = (f + L // 2) % L
    det[k, [(c - 1) % L, c, (c + 1) % L]] = 1 / 3
nb = a.dmax if a.read == "beat" else 0
bins = [round(a.rb * d) % a.T for d in range(1, nb + 1)]
gain = nn.Parameter(torch.tensor(5.0, device=dev))
bias = nn.Parameter(torch.zeros(64, device=dev))
g0 = nn.Parameter(torch.ones(64, device=dev))
gr = nn.Parameter(torch.zeros(64, nb, device=dev))
gi = nn.Parameter(torch.zeros(64, nb, device=dev))


def scores(X):
    """X (B, n) の文字番号 → 点数 (B, 64)"""
    W = waves[X.to(dev)]                                    # (B, n, L)
    U = torch.einsum("tn,bnl->btl", code_t * amp, W) / math.sqrt(a.n)   # (B, T, L) 虹
    v = model.field(U[:, :, None, :])[:, :, 0]               # (B, T, L)
    I = torch.fft.fftshift(torch.fft.fft(v, norm="ortho"), dim=-1).abs().pow(2)
    I = I / (I.mean((1, 2), keepdim=True) + 1e-12)
    S = torch.fft.fft(I, dim=1) / a.T
    out = g0 * (S[:, 0].real @ det.T)
    for j, k in enumerate(bins):
        out = out + gr[:, j] * (S[:, k].real @ det.T) + gi[:, j] * (S[:, k].imag @ det.T)
    return gain * out + bias


params = [p for p in model.parameters() if p.requires_grad] + [gain, bias, g0, gr, gi] + ([amp] if a.learn_amp else [])
opt = torch.optim.Adam(params, lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
gen = torch.Generator().manual_seed(a.seed)
t0 = time.time()
for s in range(a.steps):
    i = torch.randint(len(Ytr), (256,), generator=gen)
    loss = F.cross_entropy(scores(Xtr[i]), Ytr[i].to(dev))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
with torch.no_grad():
    out = torch.cat([scores(Xte[j:j + 500]) for j in range(0, len(Yte), 500)]).cpu()
R = dict(vars(a), acc=round((out.argmax(1) == Yte).float().mean().item() * 100, 2),
         ce=round(F.cross_entropy(out, Yte).item(), 3), amp=[round(v, 3) for v in amp.tolist()], params=sum(p.numel() for p in params), sec=round(time.time() - t0, 1))
print(json.dumps(R, ensure_ascii=False), flush=True)
