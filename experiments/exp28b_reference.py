"""実験28b: フーリエの指紋に、距離測定用の参照波を加える（ユーザー案）。

実験28 では、出力の明るさの模様をフーリエ変換した「振幅」だけを指紋にすると、模様を横にずらしても
値が変わらない（位置の情報が消える）ため、判定がうまくいかなかった。
案: 出力の手前で、周期が互いに素な参照波 (1 + cos(2πx/p_j)) を掛けてから振幅を取る。ずれ方で参照波との
重なりが変わるので、振幅だけでも位置が読める。周期が互いに素なら、中国剰余定理で長い距離も一意になる
（多波長干渉による絶対距離の測定と同じ考え方）。
比較: 複素数のまま（位相も残す）の指紋。計算機の中ではこれと等価な情報を、参照波なしで持てる。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う

import argparse
import json
import math
import random
import warnings

import torch

import sinwave_minimal as sw
from exp28_proto import evaluate, make_task, patterns

warnings.filterwarnings("ignore", message=".*文字を超えている")
K = 64


def mag(P):
    return torch.fft.rfft(P, dim=-1).abs()[:, 1:K + 1]


def cplx(P):
    F = torch.fft.rfft(P, dim=-1)[:, 1:K + 1]
    return torch.cat([F.real, F.imag], 1)


def with_refs(P, periods=(37, 41, 47)):
    x = torch.arange(P.shape[-1], dtype=torch.float32)
    feats = [mag(P)]
    for p in periods:
        feats.append(mag(P * (1 + torch.cos(2 * math.pi * x / p))))
    return torch.cat(feats, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patterns", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes, train, test = make_task(a.seed)
    net = sw.SinWaveNet(8, list(range(8)), patterns=a.patterns, seed=a.seed).to(dev)
    sw.train(net, sum([train[c] for c in range(8)], []), sum([[c] * len(train[c]) for c in range(8)], []), seed=a.seed)
    R = dict(seed=a.seed, patterns=a.patterns)
    pat = lambda t: patterns(net, t)
    fns = {"full": pat, "mag": lambda t: mag(pat(t)), "complex": lambda t: cplx(pat(t)),
           "refs": lambda t: with_refs(pat(t))}
    for name, fn in fns.items():
        R[f"{name}_base8"] = evaluate(fn, train, test, classes, list(range(8)), 0, random.Random(0))["base"]
        r = evaluate(fn, train, test, classes, list(range(12)), 20, random.Random(a.seed + 20))
        R[f"{name}_add4_k20_base"], R[f"{name}_add4_k20_novel"] = r["base"], r["novel"]
    print(json.dumps(R), flush=True)
