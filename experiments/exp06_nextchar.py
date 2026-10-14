"""実験06: 実テキスト (CC-100 日本語 60万字) で次の1文字を当てる。

入力: 直前 n 文字（レイヤ長まで繰り返す）。出力: 頻出 V-1 字 + 「その他」の V クラス。
文書の前 85% を訓練、後ろ 15% をテスト（文書単位）。
比較: n-gram（後退あり）。
並列学習: 訓練文書を 2 つに分けて別光路で学習 → 融合。全訓練文書で 1 光路を学習したものと比べる。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from exp02_modular import DEV, new_model
from sinwave.model import Config, Parallel, encode


def load(path, n, V, max_train, max_test, seed=0):
    docs = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    k = int(len(docs) * 0.85)
    tr_docs, te_docs = docs[:k], docs[k:]
    freq = Counter("".join(tr_docs))
    vocab = [c for c, _ in freq.most_common(V - 1)]
    idx = {c: i for i, c in enumerate(vocab)}
    other = V - 1

    def pairs(ds):
        out = []
        for d in ds:
            for i in range(n, len(d)):
                out.append((d[i - n:i], idx.get(d[i], other), len(out)))
        return out

    r = random.Random(seed)
    tr, te = pairs(tr_docs), pairs(te_docs)
    # 並列学習用に文書の前半/後半で分ける印をつける
    half = sum(len(d) - n for d in tr_docs[:len(tr_docs) // 2] if len(d) > n)
    tr = [(x, y, j < half) for x, y, j in tr]
    r.shuffle(tr)
    r.shuffle(te)
    return tr[:max_train], te[:max_test], vocab, tr_docs, idx


def ngram_baseline(tr_docs, te, idx, V, order):
    """後退つき n-gram（加算スムージング）。テストの正解率とクロスエントロピー(nats)。"""
    other = V - 1
    tab = [defaultdict(Counter) for _ in range(order)]
    for d in tr_docs:
        ys = [idx.get(c, other) for c in d]
        for i in range(len(d)):
            for k in range(order):
                if i - k < 0:
                    break
                tab[k][d[i - k:i]][ys[i]] += 1
    acc = ce = 0.0
    for x, y in te:
        for k in range(order - 1, -1, -1):
            ctx = x[len(x) - k:] if k else ""
            c = tab[k].get(ctx)
            if c and sum(c.values()) >= 2:
                break
        tot = sum(c.values())
        acc += c.most_common(1)[0][0] == y
        ce += -math.log((c[y] + 0.1) / (tot + 0.1 * V))
    return acc / len(te), ce / len(te)


def fit(m, X, Y, steps, lr=0.01, bs=256):
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(len(Y), (bs,), device=DEV)
        loss = F.cross_entropy(m(X[i]), Y[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


@torch.no_grad()
def score(m, X, Y, gain_fit=None):
    lg = torch.cat([m(X[i:i + 1024]) for i in range(0, len(Y), 1024)])
    return (lg.argmax(1) == Y).float().mean().item(), F.cross_entropy(lg, Y).item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cc100_ja.txt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--V", type=int, default=64)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--incoh", type=int, default=0)
    ap.add_argument("--detlog", action="store_true")
    ap.add_argument("--norepeat", action="store_true")
    ap.add_argument("--pre", type=int, default=0)
    ap.add_argument("--freeze_w", action="store_true")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--dw", type=int, default=4, help="検出窓の幅")
    ap.add_argument("--pat_w", default="")
    ap.add_argument("--readout", default="win")
    ap.add_argument("--k0", type=int, default=1)
    ap.add_argument("--dk", type=int, default=1)
    ap.add_argument("--blur", default="")
    ap.add_argument("--input_ft", action="store_true")
    ap.add_argument("--blur_mix", action="store_true")
    ap.add_argument("--illum", default="random")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tr, te, vocab, tr_docs, idx = load(a.data, a.n, a.V, a.train, a.test, a.seed)
    res = dict(vars(a))
    for o in (1, 2, 3, 4):
        acc, ce = ngram_baseline(tr_docs, [(x, y) for x, y, _ in te], idx, a.V, o)
        res[f"ngram{o}"] = (round(acc, 4), round(ce, 3))
    print(json.dumps(res, ensure_ascii=False), flush=True)

    cfg = Config(L=a.L, M=a.M, C=a.V, far=True, bias=a.bias, incoh=a.incoh, slot=a.S, illum=a.illum, detlog=a.detlog, pre_layers=a.pre, blur=a.blur, blur_mix=a.blur_mix, det_width=a.dw, pat_w=a.pat_w, readout=a.readout, readout_k0=a.k0, readout_dk=a.dk, input_ft=a.input_ft)
    enc = lambda xs: encode(xs, cfg.L, cfg.M, DEV, a.S, "freq", not a.norepeat)
    X, Y = enc([x for x, _, _ in tr]), torch.tensor([y for _, y, _ in tr], device=DEV)
    Xt, Yt = enc([x for x, _, _ in te]), torch.tensor([y for _, y, _ in te], device=DEV)
    m = new_model(cfg, a.D, a.P, "all", a.seed)
    if a.freeze_w:
        for l in m.layers:
            l.w.requires_grad_(False)
    m.gain.requires_grad_(True)
    m.bias.requires_grad_(a.bias)
    fit(m, X, Y, a.steps)
    res["sinwave_all"] = tuple(round(v, 4) for v in score(m, Xt, Yt))
    res["params"] = sum(p.numel() for p in m.layers.parameters() if p.requires_grad)
    if a.calib:
        # 確信度の尺度だけを合わせ直す: 学習後に logit 全体を 1 つの温度 τ で割り、
        # 訓練データの一部で CE が最小になる τ を選ぶ。正解率は変わらない。CE が大きく下がれば「尺度のずれ」が原因
        with torch.no_grad():
            lg = torch.cat([m(X[i:i + 1024]) for i in range(0, 10000, 1024)])[:10000]
            best = min((F.cross_entropy(lg / t, Y[:len(lg)]).item(), t) for t in [0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 3])
            t = best[1]
            lgt = torch.cat([m(Xt[i:i + 1024]) for i in range(0, len(Yt), 1024)])
            res["calib_tau"] = t
            res["sinwave_calib"] = (round((lgt.argmax(1) == Yt).float().mean().item(), 4),
                                    round(F.cross_entropy(lgt / t, Yt).item(), 4))
    if a.parallel:
        h = torch.tensor([f for _, _, f in tr], device=DEV)
        paths = []
        for k, mask in enumerate((h, ~h)):
            p = new_model(cfg, a.D, a.P, f"h{k}", a.seed + 10 + k)
            p.gain.requires_grad_(True)
            p.bias.requires_grad_(a.bias)
            fit(p, X[mask], Y[mask], a.steps)
            res[f"sinwave_half{k}"] = tuple(round(v, 4) for v in score(p, Xt, Yt))
            paths.append(p)
        par = Parallel(paths)
        res["sinwave_merged"] = tuple(round(v, 4) for v in score(par, Xt, Yt))
    print(json.dumps(res, ensure_ascii=False), flush=True)
