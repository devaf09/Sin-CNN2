"""実験05: 並列光路に技能を何本まで足せるか。

n 技能 × 4 クラス。各技能は自分のデータだけで自分の光路を学習し、最後に強度を足して融合する。
比較: 1 光路で全技能を同時学習（パラメータ数は融合後と同じ D·n 層）、ユニグラム回帰（全クラス同時）。
"""
import argparse
import json

import torch

from baseline import run as baseline_run
from exp03_parallel import fit_path
from exp02_modular import DEV, new_model
from sinwave.model import Config, Parallel, encode
from sinwave.tasks import task_skills


@torch.no_grad()
def acc(m, X, Y, lo=None):
    lg = m(X)
    if lo is not None:
        return (lg[:, lo:lo + 4].argmax(1) + lo == Y).float().mean().item()
    return (lg.argmax(1) == Y).float().mean().item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--incoh", type=int, default=0)
    ap.add_argument("--illum", default="random")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    sk = task_skills(a.n, seed=a.seed)
    C = 4 * a.n
    cfg = Config(L=a.L, C=C, far=True, incoh=a.incoh, slot=a.S, illum=a.illum)
    enc = lambda xs: encode(xs, cfg.L, cfg.M, DEV, a.S, "freq")
    data = [(enc(tr[0]), torch.tensor(tr[1], device=DEV), enc(te[0]), torch.tensor(te[1], device=DEV))
            for tr, te in sk]
    paths, alone = [], []
    for i, (X, Y, Xt, Yt) in enumerate(data):
        m = new_model(cfg, a.D, a.P, f"s{i}", a.seed + 100 * i)
        fit_path(m, X, Y, list(range(4 * i, 4 * i + 4)), steps=a.steps)
        alone.append(acc(m, Xt, Yt, 4 * i))
        paths.append(m)
    par = Parallel(paths)
    Xt_all = torch.cat([d[2] for d in data]); Yt_all = torch.cat([d[3] for d in data])
    res = dict(n=a.n, seed=a.seed, incoh=a.incoh, illum=a.illum,
               alone_task=sum(alone) / a.n,
               merged_task=sum(acc(par, d[2], d[3], 4 * i) for i, d in enumerate(data)) / a.n,
               merged_class=acc(par, Xt_all, Yt_all),
               params_merged=sum(p.n_params() for p in paths))
    # 1 本ずつ外したとき、残りの技能の class 精度
    if a.n > 1:
        sub = Parallel(paths[:-1])
        Xs = torch.cat([d[2] for d in data[:-1]]); Ys = torch.cat([d[3] for d in data[:-1]])
        res["minus_last_class"] = acc(sub, Xs, Ys)
    xtr = sum([tr[0] for tr, _ in sk], []); ytr = sum([tr[1] for tr, _ in sk], [])
    xte = sum([te[0] for _, te in sk], []); yte = sum([te[1] for _, te in sk], [])
    res["unigram_class"] = baseline_run(((xtr, ytr), (xte, yte)), False)[1]
    if a.joint:
        mJ = new_model(cfg, a.D * a.n, a.P, "J", a.seed + 999)
        X = torch.cat([d[0] for d in data]); Y = torch.cat([d[1] for d in data])
        fit_path(mJ, X, Y, list(range(C)), steps=a.steps)
        res["joint_class"] = acc(mJ, Xt_all, Yt_all)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
