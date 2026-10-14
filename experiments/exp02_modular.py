"""実験02: 学習の追加・取消・融合・並列学習。

T2(文字頻度) の 8 クラスを A(0-3) と B(4-7) の二つの「技能」に分ける。キー文字は重ならない。
検出器は 8 個。A は 0-3、B は 4-7 を使う。
評価は二通り:
  task-IL : 自分の 4 検出器の中で argmax（どちらの技能を使うかは分かっている）
  class-IL: 8 検出器全体で argmax（どちらの技能かも当てる）

シナリオ
  S1 単独    : A だけ学習 / B だけ学習
  S2 追加    : A を学習 → A を凍結して B のレイヤを後ろに足して学習
  S3 取消    : S2 から B のレイヤを外す
  S4 並列融合: A と B を別々に学習 → レイヤ列を連結（直列融合）
  S5 重ね融合: 同じ位置のレイヤ同士の位相を足し合わせる（重ね合わせ融合）
  S6 同時    : A と B を最初から一緒に学習（上限の目安）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from sinwave.model import Config, SinWave, SinLayer, encode
from sinwave.tasks import task_topic

DEV = "cuda"


def split(task):
    (xtr, ytr), (xte, yte) = task
    out = {}
    for name, lo in (("A", 0), ("B", 4)):
        f = lambda xs, ys: ([x for x, y in zip(xs, ys) if lo <= y < lo + 4], [y for y in ys if lo <= y < lo + 4])
        out[name] = (f(xtr, ytr), f(xte, yte))
    return out


def tensors(d, cfg, S, enc):
    (xtr, ytr), (xte, yte) = d
    return (encode(xtr, cfg.L, cfg.M, DEV, S, enc), torch.tensor(ytr, device=DEV),
            encode(xte, cfg.L, cfg.M, DEV, S, enc), torch.tensor(yte, device=DEV))


def fit(m, data, params, active=None, steps=1500, lr=0.01, bs=256, keep=None, keep_w=1.0):
    """data: [(X, Y), ...] を混ぜて学習。keep=(X, 目標logit) があれば、その出力を保つ罰則を足す。"""
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = 0
        for X, Y in data:
            idx = torch.randint(len(Y), (min(bs, len(Y)),), device=DEV)
            loss = loss + F.cross_entropy(m(X[idx], active), Y[idx])
        if keep is not None:
            Xk, Tk = keep
            idx = torch.randint(len(Xk), (min(bs, len(Xk)),), device=DEV)
            loss = loss + keep_w * F.kl_div(F.log_softmax(m(Xk[idx], active), 1), F.softmax(Tk[idx], 1),
                                            reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def evaluate(m, T, active=None):
    res = {}
    for name, (_, _, Xte, Yte) in T.items():
        lg = m(Xte, active)
        lo = 0 if name == "A" else 4
        res[name + "_task"] = round((lg[:, lo:lo + 4].argmax(1) + lo == Yte).float().mean().item(), 4)
        res[name + "_class"] = round((lg.argmax(1) == Yte).float().mean().item(), 4)
    return res


def new_model(cfg, D, P, tag, seed):
    m = SinWave(cfg).to(DEV)
    m.gain.requires_grad_(False)
    m.add_layers(D, 1, "phase", tag=tag, seed=seed, P=P)
    return m


def layer_params(m, tag):
    return [p for l in m.layers if l.tag == tag for p in l.parameters()]


def overlay(mA, mB):
    """重ね合わせ融合: 同じ深さのレイヤの位相変調 A sin(ωt+φ) を足す（K 方向に連結）。"""
    m = copy.deepcopy(mA)
    for la, lb in zip(m.layers, mB.layers):
        for n in ("A", "w", "phi"):
            setattr(la, n, torch.nn.Parameter(torch.cat([getattr(la, n), getattr(lb, n)], -1)))
        la.tag = "AB"
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = Config(L=a.L, C=8, far=True)
    T = {k: tensors(v, cfg, a.S, "freq") for k, v in split(task_topic(C=8, seed=a.seed)).items()}
    dA, dB = (T["A"][0], T["A"][1]), (T["B"][0], T["B"][1])
    R = {}

    # S1 単独
    mA = new_model(cfg, a.D, a.P, "A", a.seed)
    fit(mA, [dA], layer_params(mA, "A"), steps=a.steps)
    R["S1_A_only"] = evaluate(mA, T)
    mB = new_model(cfg, a.D, a.P, "B", a.seed + 100)
    fit(mB, [dB], layer_params(mB, "B"), steps=a.steps)
    R["S1_B_only"] = evaluate(mB, T)

    # S2 追加（A を凍結し、B のレイヤを後ろに足す）
    mAB = copy.deepcopy(mA)
    mAB.add_layers(a.D, 1, "phase", tag="B", seed=a.seed + 200, P=a.P)
    fit(mAB, [dB], layer_params(mAB, "B"), steps=a.steps)
    R["S2_add_B"] = evaluate(mAB, T)
    # S2' 追加 + A の出力を保つ罰則（A の訓練入力に対する A モデルの出力を教師にする）
    with torch.no_grad():
        keepA = (dA[0], mA(dA[0]))
    mAB2 = copy.deepcopy(mA)
    mAB2.add_layers(a.D, 1, "phase", tag="B", seed=a.seed + 200, P=a.P)
    fit(mAB2, [dB], layer_params(mAB2, "B"), steps=a.steps, keep=keepA)
    R["S2b_add_B_keepA"] = evaluate(mAB2, T)

    # S3 取消（B のレイヤを外す）
    R["S3_remove_B"] = evaluate(mAB, T, active={"A"})
    R["S3b_remove_B"] = evaluate(mAB2, T, active={"A"})

    # S4 並列学習 → 直列融合
    mcat = copy.deepcopy(mA)
    for l in mB.layers:
        mcat.layers.append(copy.deepcopy(l))
    R["S4_parallel_concat"] = evaluate(mcat, T)

    # S5 並列学習 → 重ね合わせ融合
    R["S5_parallel_overlay"] = evaluate(overlay(mA, mB), T)

    # S6 同時学習（同じパラメータ数 = 2D 層）
    mJ = new_model(cfg, 2 * a.D, a.P, "J", a.seed + 300)
    fit(mJ, [dA, dB], layer_params(mJ, "J"), steps=a.steps)
    R["S6_joint_2D"] = evaluate(mJ, T)

    for k, v in R.items():
        print(f"{k:22s}", json.dumps(v))
