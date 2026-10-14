"""実験03: 並列学習と融合を成り立たせる。

P1 並列光路（素朴）       : A, B を別々の光路で学習（8 検出器の CE、自分のデータのみ）→ 強度を足して融合
P2 並列光路＋背景抑制     : さらに共通のランダム文字列（どちらのタスクのデータでもない）で
                            「自分の検出器を光らせない」よう学ぶ
C1 直列追加＋データなし保持: A を凍結して B 層を足すとき、ランダム文字列に対する A の出力を保つ
各光路は相手のデータを一切見ない。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import copy
import json
import random

import torch
import torch.nn.functional as F

from exp02_modular import DEV, evaluate, layer_params, new_model, split, tensors
from sinwave.model import Config, Parallel, encode
from sinwave.tasks import HIRA, task_topic


def probes(n, seed):
    r = random.Random(seed)
    return ["".join(r.choice(HIRA) for _ in range(r.randint(4, 10))) for _ in range(n)]


def fit_path(m, X, Y, own, Xp=None, w_bg=1.0, steps=1500, lr=0.01, bs=256, keep=None):
    """own: 自分の検出器の番号。Xp: 背景（ランダム文字列）。keep: (Xk, 目標logit) 直列追加用"""
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(len(Y), (bs,), device=DEV)
        loss = F.cross_entropy(m(X[idx]), Y[idx])
        if Xp is not None:
            j = torch.randint(len(Xp), (bs,), device=DEV)
            inten = m.intensity(Xp[j])
            loss = loss + w_bg * (inten @ m.det[own].T / m.cfg.det_width).mean()
        if keep is not None:
            Xk, Tk = keep
            j = torch.randint(len(Xk), (bs,), device=DEV)
            loss = loss + F.kl_div(F.log_softmax(m(Xk[j]), 1), F.softmax(Tk[j], 1), reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="P1,P2,C1")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = Config(L=a.L, C=8, far=True)
    T = {k: tensors(v, cfg, a.S, "freq") for k, v in split(task_topic(C=8, seed=a.seed)).items()}
    (XA, YA), (XB, YB) = (T["A"][0], T["A"][1]), (T["B"][0], T["B"][1])
    Xp = encode(probes(2000, a.seed + 7), cfg.L, cfg.M, DEV, a.S, "freq")
    ownA, ownB = [0, 1, 2, 3], [4, 5, 6, 7]
    R = {}
    todo = a.only.split(",")

    for name, bg in (("P1", None), ("P2", Xp)):
        if name not in todo:
            continue
        mA = new_model(cfg, a.D, a.P, "A", a.seed)
        fit_path(mA, XA, YA, ownA, bg, steps=a.steps)
        mB = new_model(cfg, a.D, a.P, "B", a.seed + 100)
        fit_path(mB, XB, YB, ownB, bg, steps=a.steps)
        par = Parallel([mA, mB])
        R[name + "_A_alone"] = evaluate(mA, T)
        R[name + "_B_alone"] = evaluate(mB, T)
        R[name + "_merged"] = evaluate(par, T)
        R[name + "_merged_minus_B"] = evaluate(par, T, active={"A"})

    if "C1" in todo:
        mA = new_model(cfg, a.D, a.P, "A", a.seed)
        fit_path(mA, XA, YA, ownA, steps=a.steps)
        with torch.no_grad():
            keep = (Xp, mA(Xp))
        m = copy.deepcopy(mA)
        for p in m.parameters():
            p.requires_grad_(False)
        m.add_layers(a.D, 1, "phase", tag="B", seed=a.seed + 200, P=a.P)
        fit_path(m, XB, YB, ownB, steps=a.steps, keep=keep)
        R["C1_A"] = evaluate(mA, T)
        R["C1_add_B_keep_probe"] = evaluate(m, T)

    for k, v in R.items():
        print(f"{k:22s}", json.dumps(v))
