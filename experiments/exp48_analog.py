"""実験48: アナログな装置で作ったときの強さ（精度・雑音・作りの誤差）。

学習済みの光路（最小版 v1.2、16 クラス課題、虹、L=1024）を、次の不完全さを入れて評価する（学習は理想の装置で行う）。
  bits  : 板の位相の遅れを 2^bits 段階に丸める（位相を何段階で作れれば足りるか）
  fab   : 板の画素ごとに、固定の位相の誤差 N(0, σ²) [rad] を足す（作りのばらつき）
  shift : 板ごとに、位置が ±s 画素ずれる（組み立ての位置合わせの誤差、板ごとに乱数）
  zerr  : 板の間隔が板ごとに (1 + N(0, e²)) 倍になる（距離の誤差）
  det   : スクリーンの明るさに、平均の明るさに対して N(0, σ²) の雑音（検出器の雑音、評価のたびに変わる）
不完全さはそれぞれ乱数を 5 回引いて平均する。
--train_noise を付けると、学習中に fab と det の雑音を入れて学習する（雑音に慣れさせる）。
実験49: --train_shift / --train_zerr で、学習中に位置のずれ・距離の誤差も入れる。--wmax で板の縞の細かさ ω の上限を下げる
（既定 π = 2 画素周期の縞。粗い縞なら 1 画素ずれても位相がほとんど変わらない）。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import sinwave_minimal as sw
from exp01_learnability import task_skills4

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--train_noise", type=float, default=0.0, help="学習中の作りのばらつき σ [rad]（検出器の雑音も同じ値）。0 なら理想の装置で学習")
ap.add_argument("--train_shift", type=int, default=0, help="学習中の板ごとの位置のずれ（±画素、乱数）")
ap.add_argument("--train_zerr", type=float, default=0.0, help="学習中の板の間隔の誤差（相対、乱数）")
ap.add_argument("--wmax", type=float, default=math.pi, help="板の縞の細かさ ω の上限（既定 π）")
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

(xtr, ytr), (xte, yte) = task_skills4()
net = sw.SinWaveNet(16, L=a.L, max_chars=10, seed=a.seed).to(dev)
Xtr, Xte = net.encode(xtr), net.encode(xte)
Ytr, Yte = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
L, Dn = net.L, net.A.shape[0]
if a.wmax < math.pi:   # ω を対数一様の範囲 [2π/L, π] → [2π/L, wmax] に縮める（並び順はそのまま）
    lo = math.log(2 * math.pi / L)
    with torch.no_grad():
        net.w.copy_(torch.exp(lo + (net.w.log() - lo) * (math.log(a.wmax) - lo) / (math.log(math.pi) - lo)))
kx = 2 * math.pi * torch.fft.fftfreq(L, device=dev)
kz = torch.sqrt((2 * math.pi) ** 2 - kx ** 2)


def run(U, bits=0, fab=None, shift=None, zerr=None, det=0.0, gen=None):
    """最小版の intensity と同じ計算に、不完全さを入れたもの → スコア (N, 16)。"""
    u = U
    for d in range(Dn):
        p = net.patch
        phase = net.A[d, p] * torch.sin(net.w[d, p] * net.t + net.phi[d, p])
        if fab is not None:
            phase = phase + fab[d]
        if bits:
            q = 2 * math.pi / 2 ** bits
            phase = phase + (torch.round(phase / q) * q - phase).detach()   # 丸め（学習時は勾配をそのまま通す）
        if shift is not None:
            phase = torch.roll(phase, int(shift[d]), -1)
        u = u * torch.exp(1j * phase)
        H = net.H if zerr is None else torch.exp(1j * kz * 64.0 * (1 + zerr[d]))
        u = torch.fft.ifft(torch.fft.fft(u) * H)
        u = u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    inten = u.abs().pow(2).mean(1)
    inten = inten / (inten.mean(-1, keepdim=True) + 1e-12)
    if det > 0:
        inten = inten + det * torch.randn(inten.shape, generator=gen, device="cpu").to(dev)
    return net.gain * (inten @ net.det.T) + net.bias


opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
g = torch.Generator().manual_seed(a.seed)
t0 = time.time()
for s in range(a.steps):
    i = torch.randint(len(Ytr), (256,), generator=g).to(dev)
    kw = {}
    if a.train_noise > 0:
        kw.update(fab=a.train_noise * torch.randn(Dn, L, generator=g).to(dev), det=a.train_noise, gen=g)
    if a.train_shift > 0:
        kw.update(shift=torch.randint(-a.train_shift, a.train_shift + 1, (Dn,), generator=g))
    if a.train_zerr > 0:
        kw.update(zerr=(a.train_zerr * torch.randn(Dn, generator=g)).tolist())
    out = run(Xtr[i], **kw)
    loss = F.cross_entropy(out, Ytr[i])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()


@torch.no_grad()
def acc(**kw):
    return (torch.cat([run(Xte[j:j + 250], **kw).argmax(1) for j in range(0, len(Yte), 250)]) == Yte).float().mean().item()


R = dict(vars(a), ideal=round(acc() * 100, 1), train_sec=round(time.time() - t0, 1))
eg = torch.Generator().manual_seed(1234)
mean = lambda f: round(100 * sum(f() for _ in range(a.draws)) / a.draws, 1)
R["bits"] = {b: round(acc(bits=b) * 100, 1) for b in (1, 2, 3, 4, 6, 8)}
R["fab"] = {s: mean(lambda: acc(fab=s * torch.randn(Dn, L, generator=eg).to(dev))) for s in (0.05, 0.1, 0.2, 0.5, 1.0)}
R["shift"] = {s: mean(lambda: acc(shift=torch.randint(-s, s + 1, (Dn,), generator=eg))) for s in (1, 2, 4, 8)}
R["zerr"] = {e: mean(lambda: acc(zerr=(e * torch.randn(Dn, generator=eg)).tolist())) for e in (0.01, 0.05, 0.1, 0.3)}
R["det"] = {s: mean(lambda: acc(det=s, gen=eg)) for s in (0.1, 0.3, 1.0, 3.0)}
print(json.dumps(R, ensure_ascii=False), flush=True)
