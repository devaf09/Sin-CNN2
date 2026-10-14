"""sinwave_minimal.py の性能確認（README と変更履歴の数字の出どころ）。
2 技能 × 4 クラスの合成課題（実験17・21 と同じデータ）で、技能単独と融合の正解率を 3 seed で測る。
    python experiments/bench_minimal.py [--patterns 8]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う

import argparse
import statistics as st

import torch

import sinwave_minimal as sw
from sinwave.tasks import task_skills

ap = argparse.ArgumentParser()
ap.add_argument("--patterns", type=int, default=8)
ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
alone, merged = [], []
for seed in map(int, a.seeds.split(",")):
    sk = task_skills(2, seed=seed)
    nets = []
    for i, ((x, y), _) in enumerate(sk):
        own = list(range(4 * i, 4 * i + 4))
        nets.append(sw.train(sw.SinWaveNet(8, own, patterns=a.patterns, seed=seed + 100 * i).to(dev), x, y, seed=seed))
    acc = lambda net, xs, ys: sum(p == t for p, t in zip(sw.predict(net, xs), ys)) / len(ys)
    al = st.mean(acc(n, te[0], te[1]) for n, (_, te) in zip(nets, sk))
    mg = acc(sw.merge(nets), sk[0][1][0] + sk[1][1][0], sk[0][1][1] + sk[1][1][1])
    alone.append(al * 100); merged.append(mg * 100)
    print(f"seed {seed}: 単独 {al:.3f}  融合 {mg:.3f}", flush=True)
se = lambda v: st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0
print(f"patterns={a.patterns}: 単独 {st.mean(alone):.1f}±{se(alone):.1f}  融合 {st.mean(merged):.1f}±{se(merged):.1f}")
