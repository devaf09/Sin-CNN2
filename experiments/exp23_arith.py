"""実験23: 光路の引き算（タスク算術）。似た物どうしの違いを、引き算で強調できるか。

クラス（好む文字）: リンゴ {あいうえ}、トマト {あい かき}（似ている）、バナナ {さしすせ}、ブドウ {たちつて}
--share 3 にするとトマトは {あいう き}（もっと似ている）。検出器: 0 リンゴ / 1 トマト / 2 バナナ / 3 ブドウ。

θ0 : バナナ・ブドウだけを学習した共通の光路（リンゴもトマトも知らない）
θA : θ0 をリンゴ+バナナ+ブドウで追加学習（トマトは見ない）
θT : θ0 をトマト+バナナ+ブドウで追加学習（リンゴは見ない）
ΔA = θA − θ0、ΔT = θT − θ0。ω は固定し、振幅 A・初期位相 φ・背景光・温度だけを足し引きする。

評価
  auc     : リンゴの確度（4 検出器の softmax のリンゴ成分）で、リンゴの入力とトマトの入力をどれだけ分けられるか
  fp_tom  : トマトの入力を「リンゴ」と答える率（リンゴ・バナナ・ブドウの 3 択で）= やってはいけない誤反応
  rec_app : リンゴの入力を「リンゴ」と答える率（同じ 3 択）
  acc_bg  : バナナ・ブドウの正答率（4 択）
  acc4    : 4 種すべての正答率（4 択）
"""
import argparse
import copy
import json
import random

import torch
import torch.nn.functional as F

from sinwave.model import Config, SinWave, encode
from sinwave.tasks import HIRA, KATA

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_data(share, n_train, n_test, seed):
    shared = list("あいう")[:share]
    keys = {0: shared + list("うえ")[: 4 - share] if share < 3 else shared + ["え"],
            1: shared + list("かき")[: 4 - share],
            2: list("さしすせ"), 3: list("たちつて")}
    pool = HIRA + KATA

    def sample(r, y):
        n = r.randint(4, 10)
        return "".join(r.choice(keys[y]) if r.random() < 0.6 else r.choice(pool) for _ in range(n))

    def split(r, n):
        return {y: [sample(r, y) for _ in range(n)] for y in range(4)}

    return keys, split(random.Random(seed * 10 + 1), n_train), split(random.Random(seed * 10 + 2), n_test)


def new_model(cfg, seed, P, D):
    torch.manual_seed(seed)
    m = SinWave(cfg).to(DEV)
    m.add_layers(D, 1, "phase", seed=seed, P=P)
    for l in m.layers:
        l.w.requires_grad_(False)          # ω は固定（全モデルで共通の格子）
    m.bias.requires_grad_(True)
    return m


def fit(m, data, classes, steps, lr, enc):
    X = torch.cat([enc(data[c]) for c in classes])
    Y = torch.cat([torch.full((len(data[c]),), c, device=DEV) for c in classes])
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(len(Y), (256,), device=DEV)
        loss = F.cross_entropy(m(X[i]), Y[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


NAMES = ("A", "phi", "gain", "bias")


def params_of(m):
    out = {}
    for d, l in enumerate(m.layers):
        out[f"{d}.A"] = l.A.detach().clone()
        out[f"{d}.phi"] = l.phi.detach().clone()
    out["gain"] = m.gain.detach().clone()
    out["bias"] = m.bias.detach().clone()
    return out


def with_params(template, p):
    m = copy.deepcopy(template)
    with torch.no_grad():
        for d, l in enumerate(m.layers):
            l.A.copy_(p[f"{d}.A"])
            l.phi.copy_(p[f"{d}.phi"])
        m.gain.copy_(p["gain"])
        m.bias.copy_(p["bias"])
    return m


def lin(*terms):
    """terms: (係数, パラメータ辞書) の和"""
    return {k: sum(c * p[k] for c, p in terms) for k in terms[0][1]}


def auroc(pos, neg):
    s = torch.cat([pos, neg])
    rank = s.argsort().argsort().float() + 1
    n1, n0 = len(pos), len(neg)
    return ((rank[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


@torch.no_grad()
def evaluate(logit_fn, test, enc):
    lg = {c: logit_fn(enc(test[c])) for c in range(4)}
    p_app = {c: torch.softmax(lg[c], 1)[:, 0] for c in range(4)}
    three = lambda c: torch.tensor([0, 2, 3], device=DEV)[lg[c][:, [0, 2, 3]].argmax(1)]
    return dict(
        auc=round(auroc(p_app[0], p_app[1]), 4),
        fp_tom=round((three(1) == 0).float().mean().item(), 4),
        rec_app=round((three(0) == 0).float().mean().item(), 4),
        acc_bg=round(sum((lg[c].argmax(1) == c).float().mean().item() for c in (2, 3)) / 2, 4),
        acc4=round(sum((lg[c].argmax(1) == c).float().mean().item() for c in range(4)) / 4, 4),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--base_steps", type=int, default=1500)
    ap.add_argument("--ft_steps", type=int, default=1000)
    ap.add_argument("--ft_lr", type=float, default=0.005)
    a = ap.parse_args()

    keys, tr, te = make_data(a.share, 500, 250, a.seed)
    cfg = Config(L=a.L, C=4, far=True, bias=True, slot=a.S, blur="mean")
    enc = lambda xs: encode(xs, cfg.L, cfg.M, DEV, a.S, "freq")
    R = dict(share=a.share, seed=a.seed, keys={k: "".join(v) for k, v in keys.items()})

    base = fit(new_model(cfg, a.seed, a.P, a.D), tr, [2, 3], a.base_steps, 0.01, enc)
    mA = fit(copy.deepcopy(base), tr, [0, 2, 3], a.ft_steps, a.ft_lr, enc)
    mT = fit(copy.deepcopy(base), tr, [1, 2, 3], a.ft_steps, a.ft_lr, enc)
    p0, pA, pT = params_of(base), params_of(mA), params_of(mT)
    dA = lin((1, pA), (-1, p0))
    dT = lin((1, pT), (-1, p0))

    R["apple_skill"] = evaluate(mA, te, enc)
    R["tomato_skill"] = evaluate(mT, te, enc)
    for lam in (0.25, 0.5, 1.0, 1.5):
        R[f"A_minus_{lam}T"] = evaluate(with_params(base, lin((1, p0), (1, dA), (-lam, dT))), te, enc)
    R["A_plus_T"] = evaluate(with_params(base, lin((1, p0), (1, dA), (1, dT))), te, enc)
    # 並列光路の出力どうしの差（パラメータは足し引きしない）: リンゴの確度 = A のリンゴ logit − λ·T のトマト logit
    for lam in (0.5, 1.0):
        def out_diff(X, lam=lam):
            la, lt = mA(X), mT(X)
            lg = la.clone()
            lg[:, 0] = la[:, 0] - lam * lt[:, 1]
            return lg
        R[f"output_diff_{lam}"] = evaluate(out_diff, te, enc)
    # 並列光路の融合（参考）
    R["parallel_merge"] = evaluate(lambda X: (mA(X) + mT(X)) / 2, te, enc)
    # 対照: ゼロから別々に学んだリンゴとトマトの単純な引き算
    sA = fit(new_model(cfg, a.seed + 100, a.P, a.D), tr, [0, 2, 3], a.base_steps, 0.01, enc)
    sT = fit(new_model(cfg, a.seed + 200, a.P, a.D), tr, [1, 2, 3], a.base_steps, 0.01, enc)
    R["scratch_apple"] = evaluate(sA, te, enc)
    for lam in (0.25, 1.0):
        R[f"scratch_A_minus_{lam}T"] = evaluate(with_params(sA, lin((1, params_of(sA)), (-lam, params_of(sT)))), te, enc)
    print(json.dumps(R, ensure_ascii=False), flush=True)
