"""実験04: 実験02/03 の主要シナリオを複数 seed で測り直す。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import copy
import json

import torch

from exp02_modular import DEV, evaluate, new_model, split, tensors
from exp03_parallel import fit_path, probes
from sinwave.model import Config, Parallel, encode
from sinwave.tasks import task_topic


def run(seed, D, P, L, S, steps):
    torch.manual_seed(seed)
    cfg = Config(L=L, C=8, far=True)
    T = {k: tensors(v, cfg, S, "freq") for k, v in split(task_topic(C=8, seed=seed)).items()}
    (XA, YA), (XB, YB) = (T["A"][0], T["A"][1]), (T["B"][0], T["B"][1])
    Xp = encode(probes(2000, seed + 7), cfg.L, cfg.M, DEV, S, "freq")
    ownA, ownB = [0, 1, 2, 3], [4, 5, 6, 7]
    R = {}
    mA = new_model(cfg, D, P, "A", seed)
    fit_path(mA, XA, YA, ownA, steps=steps)
    mB = new_model(cfg, D, P, "B", seed + 100)
    fit_path(mB, XB, YB, ownB, steps=steps)
    R["A_alone"] = evaluate(mA, T)
    R["B_alone"] = evaluate(mB, T)
    R["parallel_merge"] = evaluate(Parallel([mA, mB]), T)

    def add_B(keep):
        m = copy.deepcopy(mA)
        for p in m.parameters():
            p.requires_grad_(False)
        m.add_layers(D, 1, "phase", tag="B", seed=seed + 200, P=P)
        fit_path(m, XB, YB, ownB, steps=steps, keep=keep)
        return evaluate(m, T)

    R["add_naive"] = add_B(None)
    with torch.no_grad():
        kA, kP = (XA, mA(XA)), (Xp, mA(Xp))
    R["add_keepA_data"] = add_B(kA)
    R["add_keep_probe"] = add_B(kP)
    # 同時学習（同じ総パラメータ: 2D 層、1 光路）
    mJ = new_model(cfg, 2 * D, P, "J", seed + 300)
    XJ, YJ = torch.cat([XA, XB]), torch.cat([YA, YB])
    fit_path(mJ, XJ, YJ, ownA + ownB, steps=steps)
    R["joint"] = evaluate(mJ, T)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="results/exp04.json")
    a = ap.parse_args()
    allr = {}
    for s in map(int, a.seeds.split(",")):
        allr[s] = run(s, a.D, a.P, a.L, a.S, a.steps)
        print("seed", s, json.dumps(allr[s]), flush=True)
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(allr, open(a.out, "w"), indent=1)
    # 平均 ± 標準誤差
    keys = next(iter(allr.values())).keys()
    for k in keys:
        cols = {}
        for m in ("A_task", "A_class", "B_task", "B_class"):
            v = torch.tensor([allr[s][k][m] for s in allr]) * 100
            se = v.std() / len(v) ** 0.5 if len(v) > 1 else torch.tensor(0.)
            cols[m] = f"{v.mean():.1f}±{se:.1f}"
        print(f"{k:16s}", "  ".join(f"{m}={c}" for m, c in cols.items()))
