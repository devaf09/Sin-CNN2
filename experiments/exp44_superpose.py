"""実験44: 答えが同じ入力を重ねて、1 回の計算で学習する（ユーザー案「周波数的にまとめて一発にする」）。

例:「やぁ」「こんにちは」「あぁ」「本日は」→ 答えはどれも「いい天気ですね。」なら、4 つの光を足し合わせて 1 回だけ通し、
   その答えの窓が明るくなるように学習する。光の計算は 1 件分なので、同じ計算量で k 倍の例を見られる。
   評価は 1 件ずつ（重ねない）入れて行う。

重ね方 mode:
  single : 重ねない（ふつうの学習、k=1）
  coh    : 光の場をそのまま足す（同じ色。入力どうしが干渉する。干渉の模様は組み合わせごとに決まっている）
  rphase : 足す前に、入力ごとに毎回ばらばらの位相をかける（光源どうしの位相がそろっていない状態）。
           干渉の模様が毎回変わるので、学習を重ねると平均で打ち消し合う
  incoh  : 色・時間の軸に並べる（干渉しない）。計算は k 倍になる（比較用: 干渉のない重ね方の上限）
組み合わせ:
  既定は毎ステップ組み合わせを作り直す。--fixed なら最初に作った組を使い回す（「同じ入力セットを使い回さない方が良い」の確認）。

課題: 16 クラスの文字列分類（実験30 と同じ。訓練 2000 件 = 各クラス約 125 件、テスト 1000 件）。最小版 v1.2 の虹の入力、L=1024。
計算量: 1 ステップに通す光の本数（組の数 × 色・時間の数）で数える。
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
ap.add_argument("--mode", default="single", help="single / coh / rphase / incoh")
ap.add_argument("--k", type=int, default=1, help="1 回に重ねる入力の数")
ap.add_argument("--fixed", action="store_true", help="組み合わせを最初に決めて使い回す")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--groups", type=int, default=256, help="1 ステップの組の数（single なら件数）")
ap.add_argument("--mix", default="rainbow")
ap.add_argument("--L", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if a.mode == "single":
    a.k = 1
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

(xtr, ytr), (xte, yte) = task_skills4()
net = sw.SinWaveNet(16, L=a.L, mix=a.mix, max_chars=10, seed=a.seed).to(dev)
Xtr = net.encode(xtr)                                     # (N, T, L)
Ytr = torch.tensor(ytr, device=dev)
by_class = [torch.nonzero(Ytr == c).flatten() for c in range(16)]
gen = torch.Generator(device="cpu").manual_seed(a.seed)


def make_groups(G):
    """G 組: 各組はクラス 1 つと、そのクラスの訓練例 k 個（重複なし）の番号。"""
    ys = torch.randint(16, (G,), generator=gen)
    idx = torch.stack([by_class[y][torch.randperm(len(by_class[y]), generator=gen)[:a.k].to(dev)] for y in ys.tolist()])
    return idx, ys.to(dev)


fixed = make_groups(a.groups * 8) if a.fixed else None   # 使い回す組（8 ステップ分の組を繰り返し使う）
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
light = 0
t0 = time.time()
for s in range(a.steps):
    if fixed is not None:
        j = torch.randint(len(fixed[1]), (a.groups,), generator=gen).to(dev)
        idx, y = fixed[0][j], fixed[1][j]
    else:
        idx, y = make_groups(a.groups)
    U = Xtr[idx]                                           # (G, k, T, L)
    if a.mode in ("single", "coh"):
        U = U.sum(1) / math.sqrt(a.k)
    elif a.mode == "rphase":
        ph = torch.rand(U.shape[:2], generator=gen).to(dev) * 2 * math.pi
        U = (U * torch.exp(1j * ph)[..., None, None]).sum(1) / math.sqrt(a.k)
    elif a.mode == "incoh":
        U = U.flatten(1, 2) / math.sqrt(a.k)               # (G, k·T, L) 色・時間の軸に並べる
    light += U.shape[0] * U.shape[1]
    loss = F.cross_entropy(net(U), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

with torch.no_grad():
    acc = sum(p == t for p, t in zip(sw.predict(net, xte), yte)) / len(yte)
    acc_tr = sum(p == t for p, t in zip(sw.predict(net, xtr), ytr)) / len(ytr)
print(json.dumps(dict(vars(a), test_acc=round(acc, 4), train_acc=round(acc_tr, 4), examples_seen=a.steps * a.groups * a.k,
                      light_per_step=light // a.steps, sec=round(time.time() - t0, 1)), ensure_ascii=False), flush=True)
