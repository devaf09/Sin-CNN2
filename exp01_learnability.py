"""実験01: そもそも学習できるか。T1(暗記) / T2(文字頻度) / T3(順序) を設定を変えて試す。"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from sinwave.model import Config, SinWave, encode
from sinwave.tasks import task_order, task_skills, task_topic, task_words


def task_skills4():
    sk = task_skills(4)
    cat = lambda j, k: sum([x[j][k] for x in sk], [])
    return (cat(0, 0), cat(0, 1)), (cat(1, 0), cat(1, 1))


def train(task, cfg: Config, D=16, K=1, mode="phase", steps=1500, lr=0.03, bs=256, seed=0, dev="cuda", log=False, S=1, enc="phase", P=1, repeat=True):
    torch.manual_seed(seed)
    (xtr, ytr), (xte, yte) = task
    m = SinWave(cfg).to(dev)
    m.add_layers(D, K, mode, seed=seed, P=P)
    Xtr, Xte = encode(xtr, cfg.L, cfg.M, dev, S, enc, repeat), encode(xte, cfg.L, cfg.M, dev, S, enc, repeat)
    Ytr, Yte = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        idx = torch.randint(len(Ytr), (min(bs, len(Ytr)),), device=dev)
        loss = F.cross_entropy(m(Xtr[idx]), Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and s % 250 == 0:
            print(s, round(loss.item(), 4))
    with torch.no_grad():
        acc_tr = (m(Xtr).argmax(1) == Ytr).float().mean().item()
        acc_te = (m(Xte).argmax(1) == Yte).float().mean().item()
    return m, acc_tr, acc_te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="words")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--z", type=float, default=64)
    ap.add_argument("--mode", default="phase")
    ap.add_argument("--nonlin", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--enc", default="phase")
    ap.add_argument("--P", type=int, default=1, help="レイヤの区画数")
    ap.add_argument("--T", type=int, default=0, help="時間標本数 (0=位置を時間とみなす)")
    ap.add_argument("--far", action="store_true", help="遠方場で検出")
    ap.add_argument("--incoh", type=int, default=0, help="非コヒーレント光源の位相パターン数")
    ap.add_argument("--z_sched", default="const")
    ap.add_argument("--illum", default="random")
    ap.add_argument("--feedback", default="")
    ap.add_argument("--out_pad", type=int, default=1)
    ap.add_argument("--z_out", type=float, default=0.0)
    ap.add_argument("--detlog", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--lens_every", type=int, default=0)
    ap.add_argument("--norepeat", action="store_true", help="入力を繰り返さず1回だけ置く")
    ap.add_argument("--dw", type=int, default=4, help="検出窓幅。0 で出力面を C 等分")
    a = ap.parse_args()
    task = {"words": task_words, "topic": task_topic, "order": task_order, "skills4": task_skills4}[a.task]()
    C = max(task[0][1]) + 1
    cfg = Config(L=a.L, M=a.M, z=a.z, C=C, nonlin=a.nonlin, T=a.T, far=a.far, incoh=a.incoh, slot=a.S, z_sched=a.z_sched, illum=a.illum, feedback=a.feedback, out_pad=a.out_pad, z_out=a.z_out, detlog=a.detlog, diff=a.diff, lens_every=a.lens_every, det_width=a.dw or a.L // C)
    t0 = time.time()
    m, tr, te = train(task, cfg, a.D, a.K, a.mode, a.steps, a.lr, seed=a.seed, log=True, S=a.S, enc=a.enc, P=a.P, repeat=not a.norepeat)
    print(json.dumps(dict(vars(a), params=m.n_params(), train=round(tr, 4), test=round(te, 4),
                          sec=round(time.time() - t0, 1)), ensure_ascii=False))
