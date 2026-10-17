"""実験51: うなり読み（FMCW レーダー式の取り出し）。

虹（実験33）では、位置 m の文字の光の周波数を m·rb だけずらし、T 個の時間の標本で明るさを平均していた。
2 つの文字（位置 m, n）の光は、明るさに周波数 rb·(m−n) の「うなり」を作る。FMCW レーダーで距離がうなりの周波数になるのと同じ。
平均するとこのうなりは消える（離れた文字どうしの関係が捨てられる）。
今回: 平均せず、スクリーンの各点の明るさを時間方向にフーリエ変換し、うなりの周波数 d ごとの成分を読む。
  成分 d には「d 文字離れた 2 文字の組」の干渉が乗る（どちらが先かは成分の位相に出るので、実部と虚部を別々に読む）。
  点数_c = gain·[ g0_c·窓_c(平均の明るさ) + Σ_d (gr_cd·窓_c(Re 成分 d) + gi_cd·窓_c(Im 成分 d)) ] + 背景光_c
  学習が増えるのは、窓ごとの重み g（クラス数 × (1 + 2·dmax) 個）だけ。光の計算は虹と同じ（T 個の標本）。
読み方 --read:
  mean : 平均の明るさだけ（今の虹）
  beat : うなりの成分も読む
課題 --task:
  skills4 : 16 クラスの文字列分類（文字の頻度で解ける）
  dist    : 「あ」と「い」が 1 回ずつ入り、何文字離れているか（1〜4）× どちらが先か の 8 クラス（頻度では解けない）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sinwave_minimal as sw
from exp01_learnability import task_skills4
from sinwave.tasks import HIRA


def task_dist(n_train=2000, n_test=1000, seed=0):
    others = HIRA[2:20]

    def sample(r):
        y = r.randrange(8)
        d, first = y % 4 + 1, y // 4                     # d 文字離れている、first=0 なら あ が先
        n = r.randint(max(6, d + 1), 10)
        body = [r.choice(others) for _ in range(n)]
        i = r.randrange(n - d)
        body[i], body[i + d] = ("あ", "い") if first == 0 else ("い", "あ")
        return "".join(body), y

    r1, r2 = random.Random(seed * 10 + 1), random.Random(seed * 10 + 2)
    tr = [sample(r1) for _ in range(n_train)]
    te = [sample(r2) for _ in range(n_test)]
    return tuple(map(list, zip(*tr))), tuple(map(list, zip(*te)))


ap = argparse.ArgumentParser()
ap.add_argument("--task", default="dist")
ap.add_argument("--read", default="beat")
ap.add_argument("--rb", type=float, default=1.0, help="位置 1 つあたりの周波数のずれ（整数にすると、うなりが標本の上でちょうど分かれる）")
ap.add_argument("--T", type=int, default=20, help="時間の標本の数（うなりの周波数 rb·9 まで分けるには 2·rb·9 より多く）")
ap.add_argument("--dmax", type=int, default=9, help="読むうなりの周波数の最大（文字の距離）")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

(xtr, ytr), (xte, yte) = task_skills4() if a.task == "skills4" else task_dist(seed=a.seed)
C = max(ytr) + 1
net = sw.SinWaveNet(C, L=a.L, max_chars=10, rb=a.rb, T=a.T, seed=a.seed).to(dev)
Xtr, Xte = net.encode(xtr), net.encode(xte)
Ytr, Yte = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
nb = a.dmax if a.read == "beat" else 0
g0 = nn.Parameter(torch.ones(C, device=dev))
gr = nn.Parameter(torch.zeros(C, nb, device=dev))
gi = nn.Parameter(torch.zeros(C, nb, device=dev))
bins = [round(a.rb * d) % a.T for d in range(1, nb + 1)]   # 距離 d のうなりが落ちる時間の周波数の番号


def scores(U):
    """最小版の intensity と同じ光の計算。ただし時間の標本ごとの明るさを残す → 点数 (N, C)。"""
    u = U
    for d in range(net.A.shape[0]):
        p = net.patch
        phase = net.A[d, p] * torch.sin(net.w[d, p] * net.t + net.phi[d, p])
        u = u * torch.exp(1j * phase)
        u = torch.fft.ifft(torch.fft.fft(u) * net.H)
        u = u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    I = u.abs().pow(2)                                       # (N, T, L) 時間の標本ごとの明るさ
    I = I / (I.mean((1, 2), keepdim=True) + 1e-12)
    S = torch.fft.fft(I, dim=1) / I.shape[1]                  # 時間方向のフーリエ変換: 成分 0 = 平均、成分 d = うなり
    W = net.det                                              # (C, L) 窓
    out = g0 * (S[:, 0].real @ W.T)
    for j, k in enumerate(bins):
        out = out + gr[:, j] * (S[:, k].real @ W.T) + gi[:, j] * (S[:, k].imag @ W.T)
    return net.gain * out + net.bias


params = [p for p in net.parameters() if p.requires_grad] + [g0, gr, gi]
opt = torch.optim.Adam(params, lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
gen = torch.Generator().manual_seed(a.seed)
t0 = time.time()
for s in range(a.steps):
    i = torch.randint(len(Ytr), (256,), generator=gen).to(dev)
    loss = F.cross_entropy(scores(Xtr[i]), Ytr[i])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
with torch.no_grad():
    acc = lambda X, Y: (torch.cat([scores(X[j:j + 250]).argmax(1) for j in range(0, len(Y), 250)]) == Y).float().mean().item()
    R = dict(vars(a), test_acc=round(acc(Xte, Yte) * 100, 1), train_acc=round(acc(Xtr, Ytr) * 100, 1),
             params=sum(p.numel() for p in params), sec=round(time.time() - t0, 1))
print(json.dumps(R, ensure_ascii=False), flush=True)
