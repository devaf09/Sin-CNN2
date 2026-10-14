"""実験29: 学習した板を敷き詰めて、大きな板へ移す（ユーザー案 3b）。

長さ L で学習した光路を、長さ 2L の光路にする。
  板   : 区画を 2 回並べる。2 回目の区画は初期位相を φ − ωL にずらす（sin(ω(t+L)+φ') = sin(ωt+φ) にするため）
  入力 : 長さ L の場をそのまま 2 回並べる（tiled）。ふつうに文字列を繰り返した場（natural）でも試す
  検出 : 周期 L の場はレンズの後で偶数番目の周波数にだけ現れるので、窓を中心からの距離 2 倍・幅 2 倍にする
         （窓内の平均がちょうど元と同じ値になる）

確かめること
  1. tiled 入力で元と同じ出力になるか（数値誤差の範囲）
  2. natural 入力でもそのまま使えるか
  3. 敷き詰めた板から追加学習すると、2L の板をゼロから学習するより良いか（同じステップ数）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う

import argparse
import json
import math
import warnings

import torch

import sinwave_minimal as sw
from sinwave.tasks import task_skills

warnings.filterwarnings("ignore", message=".*文字を超えている")


def tile(net: sw.SinWaveNet) -> sw.SinWaveNet:
    L, P = net.L, net.A.shape[1]
    big = sw.SinWaveNet(net.n_slots, net.own, L=2 * L, S=net.S, layers=net.A.shape[0], patches=2 * P,
                        patterns=0, seed=0).to(net.det.device)
    with torch.no_grad():
        big.A.copy_(torch.cat([net.A, net.A], 1))
        big.w.copy_(torch.cat([net.w, net.w], 1))
        big.phi.copy_(torch.cat([net.phi, (net.phi - net.w * L) % (2 * math.pi)], 1))
        big.gain.copy_(net.gain)
        big.bias.copy_(net.bias)
        # 検出窓: 元の窓（中心からの位置 o..o+3）→ 2o..2o+7
        det = torch.zeros_like(big.det)
        for i in range(net.n_slots):
            idx = net.det[i].nonzero()[:, 0] - L // 2          # 元の窓の、中心からの位置
            lo = 2 * idx.min().item()
            det[i, (torch.arange(2 * len(idx)) + lo + L) % (2 * L)] = 1 / (2 * len(idx))
        big.det.copy_(det)
    return big


def acc(net, X, Y):
    with torch.no_grad():
        s = net(X)
        mask = torch.full((net.n_slots,), float("-inf"), device=s.device)
        mask[net.own] = 0
        return ((s + mask).argmax(1).cpu() == Y).float().mean().item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ft_steps", type=int, default=1500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sk = task_skills(4, seed=a.seed)
    xtr = sum([tr[0] for tr, _ in sk], []); ytr = sum([tr[1] for tr, _ in sk], [])
    xte = sum([te[0] for _, te in sk], []); yte = torch.tensor(sum([te[1] for _, te in sk], []))
    L, S = 2048, 256
    R = dict(seed=a.seed)

    small = sw.train(sw.SinWaveNet(16, list(range(16)), L=L, S=S, seed=a.seed).to(dev), xtr, ytr, steps=a.steps, seed=a.seed)
    Xs = sw.encode(xte, L, S, dev)
    R["small_L2048"] = acc(small, Xs, yte)

    big = tile(small)
    X_tiled = torch.cat([Xs, Xs], 1)                         # 長さ L の場を 2 回並べる
    X_nat = sw.encode(xte, 2 * L, S, dev)                    # 文字列をふつうに繰り返す
    with torch.no_grad():
        R["tiled_max_logit_diff"] = (big(X_tiled) - small(Xs)).abs().max().item()
    R["tiled_input"] = acc(big, X_tiled, yte)
    R["natural_input_no_ft"] = acc(big, X_nat, yte)

    # 追加学習: 敷き詰めた板から / 2L の板をゼロから（同じステップ数、natural 入力）
    sw.train(big, xtr, ytr, steps=a.ft_steps, seed=a.seed)
    R["tiled_then_ft"] = acc(big, X_nat, yte)
    scratch = sw.train(sw.SinWaveNet(16, list(range(16)), L=2 * L, S=S, patches=128, seed=a.seed).to(dev),
                       xtr, ytr, steps=a.ft_steps, seed=a.seed)
    R["scratch_L4096"] = acc(scratch, X_nat, yte)
    print(json.dumps(R), flush=True)
