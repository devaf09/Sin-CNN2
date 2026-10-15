"""実験30: 文字を「レイヤ全体を照らす光」として入れる（ユーザー案 1・2、2026-10-14）。

これまで: 1 文字に S 画素の区画（スロット）を割り当てて並べていた。使えるのは先頭 L/S 文字だけで、
          実テキストで文脈を伸ばすと悪化した（実験06・24）。
今回: 各文字の光がレイヤ全体を照らす。文字の種類と順番を、光の向き・位相・波面の曲がりで表す。

位置の表し方 (--pos)
  pair  : 文字と位置の組ごとに一意な向き。周波数 = 文字番号 × 位置数 + 位置（案2 の ii）
  phase : 文字の向き × 位置ごとの一定の位相ずれ（案2 の i）
  sphere: 文字の向き × 位置ごとの球面波の曲がり exp(iπ(x−L/2)²/(λ d))、d = d0·L·(位置+1)（案1: 点光源を距離 d に置く）
文字どうしの合わせ方 (--mix)
  coh   : 全部干渉させる
  adj   : 隣の文字どうしだけ干渉させる（照明パターン j で位置 m に位相 2πjm/n を与え、重み 1+2a·cos(2πj/n) で平均。実験25 の方法を文字の位置に使う）
  color : 文字ごとに別の色（干渉なし。案1 の a）
文字の番号 (--code): ord = 文字コード mod V / rank = 訓練データでの出現順位（上位 V−1 字、残りは共通の番号。固定の対応表で学習しない）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # 同じフォルダの実験コード

import argparse
import json
import math
import time
from collections import Counter

import torch
import torch.nn.functional as F

from sinwave.model import Config, SinWave, char_phase

GOLD = 0.6180339887
STRIDE = 0   # 0: 周波数をそのまま使う。>0: f·STRIDE mod L（--stride で設定）


def coprime_stride(L):
    """L と約数を共有しない（互いに素な）、L の約 0.618 倍の歩幅。"""
    g = int(L * GOLD) | 1
    while math.gcd(g, L) != 1:
        g += 2
    return g


class Encoder:
    """文字列 → 文字番号・位置の表。光の場はミニバッチごとに作る。"""

    def __init__(self, n_pos, V, code="ord", train_texts=None, from_end=False):
        self.n, self.V, self.code, self.from_end = n_pos, V, code, from_end
        if code == "rank":
            freq = Counter("".join(train_texts))
            self.rank = {c: i for i, (c, _) in enumerate(freq.most_common(V - 1))}

    def char_code(self, c):
        if self.code == "ord":
            return ord(c) % self.V
        return self.rank.get(c, self.V - 1)

    def table(self, texts):
        N, n = len(texts), self.n
        codes = torch.zeros(N, n, dtype=torch.long)
        theta = torch.zeros(N, n)
        pos = torch.zeros(N, n, dtype=torch.long)
        mask = torch.zeros(N, n)
        for b, t in enumerate(texts):
            t = t[-n:] if self.from_end else t[:n]
            for i, c in enumerate(t):
                p = (len(t) - 1 - i) if self.from_end else i      # 次文字予測では「直前の文字」が位置 0
                codes[b, i], theta[b, i], pos[b, i], mask[b, i] = self.char_code(c), char_phase(c, 0), p, 1.0
        return codes, theta, pos, mask


def make_fields(codes, theta, pos, mask, L, n, mode, mix, d0, dev):
    """(B, n) の文字の表 → 光の場 (B, T, L)"""
    codes, theta, pos, mask = codes.to(dev), theta.to(dev), pos.to(dev).float(), mask.to(dev)
    x = torch.arange(L, device=dev, dtype=torch.float32)
    f = codes.float() * n + (pos if mode == "pair" else 0)                      # 向き（空間周波数の番号）
    if STRIDE:   # 実験35: 小さな L でも重ならないよう、L と約数を共有しない歩幅で周波数を並べ直す（f·g mod L）
        f = torch.remainder(f * STRIDE, L)
    ph = 2 * math.pi * f[..., None] * x / L + theta[..., None]                   # (B, n, L)
    if mode == "phase":
        ph = ph + (2 * math.pi * GOLD * pos)[..., None]
    elif mode == "sphere":
        d = d0 * L * (pos + 1)
        ph = ph + math.pi * (x - L / 2) ** 2 / d[..., None]                      # 点光源からの球面波（波長 1）
    amp = mask / mask.sum(1, keepdim=True).clamp(min=1).sqrt()                   # 全体の明るさを揃える
    w = torch.polar(amp[..., None].expand_as(ph), ph)                            # (B, n, L)
    if mix == "coh":
        return w.sum(1, keepdim=True)
    if mix == "color":
        return w
    if mix == "adj":   # パターン j: 位置 m に位相 2πjm/n
        j = torch.arange(n, device=dev, dtype=torch.float32)
        code = torch.exp(1j * (2 * math.pi * j[None, :, None] * pos[:, None, :] / n))          # (B, T, n)
        return torch.einsum("btn,bnl->btl", code, w)
    raise ValueError(mix)


def forward(model, U, mix, n, a=0.5):
    u = model.field(U[:, :, None, :])                                           # (B, T, 1, L)
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)              # レンズ
    I = u.abs().pow(2)[:, :, 0]
    if mix == "adj":
        j = torch.arange(I.shape[1], device=I.device, dtype=torch.float32)
        wt = 1 + 2 * a * torch.cos(2 * math.pi * j / n)                          # 隣どうしだけ干渉が残る重み
        I = (I * (wt / wt.sum())[None, :, None]).sum(1)
    else:
        I = I.mean(1)
    I = I / (I.mean(-1, keepdim=True) + 1e-12)
    return model.detect(I)


def load_task(name, n_text):
    if name == "order":
        from sinwave.tasks import task_order
        (xtr, ytr), (xte, yte) = task_order()
        return xtr, ytr, xte, yte, 2, 8, dict(code="ord", V=256), False
    if name == "skills4":
        from exp01_learnability import task_skills4
        (xtr, ytr), (xte, yte) = task_skills4()
        return xtr, ytr, xte, yte, 16, 10, dict(code="ord", V=256), False
    if name == "text":
        from exp06_nextchar import load
        tr, te, *_ = load(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "cc100_ja.txt"), n_text, 64, 60000, 10000)
        return ([x for x, _, _ in tr], [y for _, y, _ in tr], [x for x, _, _ in te], [y for _, y, _ in te],
                64, n_text, dict(code="rank", V=512), True)
    raise ValueError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="order")
    ap.add_argument("--n", type=int, default=2, help="実テキストの文脈の文字数")
    ap.add_argument("--pos", default="pair")
    ap.add_argument("--mix", default="coh")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--d0", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=0, help="0 なら課題ごとの既定（合成 1500、実テキスト 3000）")
    ap.add_argument("--nl", default="")
    ap.add_argument("--freeze_w", action="store_true")
    ap.add_argument("--focus", action="store_true", help="レイヤごとに学習できる焦点の強さを足す")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0, help="動作確認用に訓練例を減らす")
    ap.add_argument("--stride", action="store_true", help="実験35: 周波数を f·g mod L に並べ直す（g は L と互いに素）")
    ap.add_argument("--paths", type=int, default=1, help="実験35: 光路を何本並べるか（幅）")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    xtr, ytr, xte, yte, C, n, cd, is_text = load_task(a.task, a.n)
    if a.max_train:
        xtr, ytr, xte, yte = xtr[:a.max_train], ytr[:a.max_train], xte[:a.max_train // 2], yte[:a.max_train // 2]
    steps = a.steps or (3000 if is_text else 1500)
    enc = Encoder(n, cd["V"], cd["code"], xtr, from_end=is_text)
    Ttr, Tte = enc.table(xtr), enc.table(xte)
    Ytr, Yte = torch.tensor(ytr), torch.tensor(yte)
    if a.stride:
        STRIDE = coprime_stride(a.L)
    elif (cd["V"] - 1) * n + n > a.L:
        raise SystemExit(f"L={a.L} では周波数が足りない（必要 {cd['V'] * n}）。--stride で並べ直せる")
    # 使われている（文字, 位置）の周波数のうち、別のものと重なってしまった割合
    codes_, _, pos_, mask_ = Ttr
    raw = (codes_ * n + (pos_ if a.pos == "pair" else 0))[mask_ > 0].unique()
    fq = torch.remainder(raw * STRIDE, a.L) if STRIDE else raw
    collided = round(1 - fq.unique().numel() / max(raw.numel(), 1), 4)

    # 光路を a.paths 本並べる（幅）。点数はその平均（明るさの平均）
    models = []
    for k in range(a.paths):
        cfg = Config(L=a.L, C=C, far=True, bias=is_text, nl=a.nl, focus=a.focus)
        model = SinWave(cfg).to(dev)
        model.add_layers(a.D, 1, "phase", seed=a.seed + 100 * k, P=a.P)
        model.bias.requires_grad_(is_text)
        if a.freeze_w:
            for l in model.layers:
                l.w.requires_grad_(False)
        models.append(model)
    run = lambda U: sum(forward(m, U, a.mix, n) for m in models) / len(models)
    params = [p for m in models for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(steps):
        i = torch.randint(len(Ytr), (min(256, len(Ytr)),), generator=gen)
        U = make_fields(*(t[i] for t in Ttr), a.L, n, a.pos, a.mix, a.d0, dev)
        loss = F.cross_entropy(run(U), Ytr[i].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    train_sec = time.time() - t0

    @torch.no_grad()
    def evaluate(T, Y):
        lg = torch.cat([run(make_fields(*(t[j:j + 256] for t in T), a.L, n, a.pos, a.mix, a.d0, dev))
                        for j in range(0, len(Y), 256)]).cpu()
        return round((lg.argmax(1) == Y).float().mean().item(), 4), round(F.cross_entropy(lg, Y).item(), 4)

    res = dict(vars(a), n_pos=n, stride_g=STRIDE, collided=collided, train=evaluate(Ttr, Ytr)[0], test=evaluate(Tte, Yte),
               params=sum(p.numel() for m in models for p in m.layers.parameters() if p.requires_grad),
               sec=round(train_sec, 1), ms_per_step=round(1000 * train_sec / steps, 2))
    print(json.dumps(res, ensure_ascii=False), flush=True)
