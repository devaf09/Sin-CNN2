"""実験33c: 文字列 X → 文字列 Y（ユーザー案、2026-10-16）。

ユーザー案: 文字列を 1 つの記号として入力すれば（「あいう」= X）、対応する文字列（Y =「えおか」）を出力できるのでは。
16 種類の文字、X も Y も長さ 4。出力は 1 つの出力面に 4×16 個の窓を並べ、位置 k を k 番目の 16 個で読む。

(1) 言い回し帳の容量 (--part phrase): ランダムな X → Y の組を N 組覚えさせ、同じ組で思い出せるかを測る（暗記）。
    X の入れ方 --enc
      atom: 文字列を丸ごと 1 つの記号にする。組ごとに固有の向き（周波数）の平面波 1 本（A: 辞書型。未知の文字列には使えない）
      pair: 文字と位置の組ごとの向きの平面波を重ね合わせる（B: 並び順つきの組み合わせ）
(2) 規則の汎化 (--part rule): 2000 個の X で学習し、見たことのない 1000 個で試す（pair のみ）。
      reverse: Y = X を逆順に / map: Y_k = 決まった置き換え(X_k) / copy: Y = X
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from exp33b_seqout import CHARS, N, V, waves
from sinwave.model import Config, SinWave

PERM = [(5 * i + 3) % V for i in range(V)]   # map 課題の置き換え（固定）


def make_rule(n, seed, rule):
    r = random.Random(seed)
    X = torch.tensor([[r.randrange(V) for _ in range(N)] for _ in range(n)])
    if rule == "reverse":
        Y = X.flip(1)
    elif rule == "map":
        Y = torch.tensor(PERM)[X]
    else:
        Y = X.clone()
    return X, Y


def make_phrase(n, seed):
    r = random.Random(seed)
    seen, X = set(), []
    while len(X) < n:                         # X は重複させない
        x = tuple(r.randrange(V) for _ in range(N))
        if x not in seen:
            seen.add(x)
            X.append(x)
    Y = torch.tensor([[r.randrange(V) for _ in range(N)] for _ in range(n)])
    return torch.tensor(X), Y


def atom_field(idx, n_atoms, L, dev, seed=7):
    """組の番号 → 固有の向きの平面波 1 本 (B, L)。向きは周波数の帯域全体に散らす。"""
    g = torch.Generator().manual_seed(seed)
    f = torch.randperm(L // 2, generator=g)[:n_atoms].to(dev)          # 組ごとに重ならない周波数
    th = (2 * math.pi * torch.rand(n_atoms, generator=g)).to(dev)
    x = torch.arange(L, device=dev, dtype=torch.float32)
    return torch.exp(1j * (2 * math.pi * f[idx.to(dev), None].float() * x / L + th[idx.to(dev), None]))


def scores(model, field, L):
    u = model.field(field[:, None, None])[:, 0, 0]
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    I = u.abs().pow(2)
    return model.detect(I / (I.mean(-1, keepdim=True) + 1e-12)).reshape(-1, N, V)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="phrase")
    ap.add_argument("--enc", default="pair")
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--rule", default="reverse")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    if a.part == "phrase":
        Xtr, Ytr = make_phrase(a.pairs, a.seed + 1)
        Xte, Yte = Xtr, Ytr                    # 暗記: 同じ組で思い出せるか
    else:
        Xtr, Ytr = make_rule(2000, a.seed + 1, a.rule)
        Xte, Yte = make_rule(1000, a.seed + 2, a.rule)
    idx_tr = torch.arange(len(Xtr))

    def field(sel, X):
        if a.enc == "atom":
            return atom_field(sel, a.pairs, a.L, dev)
        return waves(X[sel], a.L, dev, True).sum(1)

    model = SinWave(Config(L=a.L, C=N * V, far=True)).to(dev)
    model.add_layers(a.D, 1, "phase", seed=a.seed, P=a.P)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(a.steps):
        i = torch.randint(len(Ytr), (min(256, len(Ytr)),), generator=gen)
        loss = F.cross_entropy(scores(model, field(i, Xtr), a.L).reshape(-1, V), Ytr[i].to(dev).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = torch.cat([scores(model, field(torch.arange(j, min(j + 250, len(Yte))), Xte), a.L).argmax(-1).cpu()
                          for j in range(0, len(Yte), 250)])
    params = sum(p.numel() for p in model.layers.parameters())
    ok = (pred == Yte)
    print(json.dumps(dict(vars(a), char_acc=round(ok.float().mean().item(), 4), string_acc=round(ok.all(1).float().mean().item(), 4),
                          params=params, bits_stored=round(ok.all(1).sum().item() * N * math.log2(V)),
                          sec=round(time.time() - t0, 1)), ensure_ascii=False), flush=True)
