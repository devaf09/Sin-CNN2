"""実験45: 単位（よく出る文字列）を「部品の組」で表し、答えの窓を減らす（実験34 の語根の考え方を言語の単位に使う）。

実験43 の問題: 単位ごとに窓を置くと窓が約 3000 個になり、限られた光を分け合って確信度が下がった（光の記憶 4.25 → 4.40）。
今回: 単位 u に部品の組 (a_u, b_u) を割り当てる（a は A 種、b は B 種、A·B ≥ 単位の数）。窓は部品ごとに 1 つ（A+B 個）。
  点数(u) = log P_a(a_u) + log P_b(b_u) + 単位ごとの補正 c_u      P_a, P_b は部品の窓の明るさの softmax
  学習・評価は、全単位についての softmax（実験43 と同じ 1 文字あたりの CE）。
部品の割り当て --assign:
  ctx    : 「次に来る文字の分布」が似た単位を同じ a にまとめる（容量つき k-means）。b は仲間の中での番号（出現の多い順）
  random : でたらめ（窓を減らす効果だけを測る基準）
入力 --input:
  unit  : 単位ごとに固有の周波数の平面波（実験43 と同じ。L ≥ 単位の数が必要）
  parts : 単位の光 = 部品 a の光 + 部品 b の光（2 本の平面波）。L が単位の数より小さくてよい。
          出力の窓は部品の光がレンズの後で集まる位置なので、出力の光がそのまま次の入力と同じ形になる
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import math
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp43_tokens import build_vocab, load_docs, tokenize
from sinwave.model import Config, SinWave


def coprime_stride(L):
    g = int(L * 0.6180339887) | 1
    while math.gcd(g, L) != 1:
        g += 2
    return g


def assign_parts(vocab, tr_stream, A, B, how, seed):
    """単位ごとの部品の組 (a, b) を返す。"""
    Vt = len(vocab)
    freq = Counter(t for s in tr_stream for t in s)
    order = sorted(range(Vt), key=lambda u: -freq[u])               # 出現の多い順
    if how == "random":
        perm = torch.randperm(A * B, generator=torch.Generator().manual_seed(seed))[:Vt]
        return perm // B, perm % B
    # ctx: 次に来る単位の「先頭の文字」の分布（頻出 64 字 + その他）を特徴にする
    first = Counter(vocab[t][0] for s in tr_stream for t in s if vocab[t] != "<不明>")
    top = {c: i for i, (c, _) in enumerate(first.most_common(63))}
    fid = lambda t: top.get(vocab[t][0], 63) if vocab[t] != "<不明>" else 63
    M = torch.full((Vt, 64), 0.1)
    for s in tr_stream:
        for x, y in zip(s, s[1:]):
            M[x, fid(y)] += 1
    X = (M / M.sum(1, keepdim=True)).sqrt()                         # ヘリンガー距離で比べられる形
    w = torch.tensor([freq[u] + 1.0 for u in range(Vt)]).log()      # 出現の多い単位を重く
    g = torch.Generator().manual_seed(seed)
    C = X[torch.tensor(order[:A])].clone()                          # 頻出 A 単位から始める
    for _ in range(30):                                             # 重みつき k-means
        lab = torch.cdist(X, C).argmin(1)
        for k in range(A):
            m = lab == k
            C[k] = (X[m] * w[m, None]).sum(0) / w[m].sum() if m.any() else X[torch.randint(Vt, (1,), generator=g)][0]
    # 容量つきの割り当て: 出現の多い順に、空きのある一番近い中心へ（各 a に B 個まで）
    D = torch.cdist(X, C)
    cap = [0] * A
    a = torch.zeros(Vt, dtype=torch.long)
    b = torch.zeros(Vt, dtype=torch.long)
    for u in order:
        for k in D[u].argsort().tolist():
            if cap[k] < B:
                a[u], b[u] = k, cap[k]
                cap[k] += 1
                break
    return a, b


class PartsMemory(nn.Module):
    def __init__(self, Vt, pa, pb, A, B, L, D, P, alpha, inp, seed, dev):
        super().__init__()
        g = coprime_stride(L)
        x = torch.arange(L, device=dev, dtype=torch.float32)
        th = torch.rand(max(Vt, A + B), generator=torch.Generator().manual_seed(1)).to(dev) * 2 * math.pi
        pf = torch.tensor([(i * g) % L for i in range(A + B)], device=dev)           # 部品の周波数（a: 0..A-1、b: A..A+B-1）
        pw = torch.exp(1j * (2 * math.pi * pf[:, None] * x / L + th[:A + B, None]))
        self.pa, self.pb = pa.to(dev), pb.to(dev)
        if inp == "unit":
            if Vt > L:
                raise SystemExit(f"単位 {Vt} が L={L} を超えている（--input parts を使う）")
            uf = torch.tensor([(i * g) % L for i in range(Vt)], device=dev)
            waves = torch.exp(1j * (2 * math.pi * uf[:, None] * x / L + th[:Vt, None]))
            # 部品の窓は、どの単位の周波数とも重ならない位置に置く（空いている周波数から等間隔に選ぶ）
            free = sorted(set(range(L)) - set(uf.tolist()))
            if len(free) < A + B:
                raise SystemExit("部品の窓を置く空きの周波数が足りない")
            pf = torch.tensor([free[i * len(free) // (A + B)] for i in range(A + B)], device=dev)
            self.bins = (pf + L // 2) % L
        else:
            waves = (pw[self.pa] + pw[A + self.pb]) / math.sqrt(2)
            self.bins = (pf + L // 2) % L                                             # 部品の光が集まる位置 = 出力の窓
        self.register_buffer("waves", waves)
        self.A = A
        self.sw = SinWave(Config(L=L, C=1, far=True))
        self.sw.add_layers(D, 1, "phase", seed=seed, P=P)
        for l in self.sw.layers:
            l.w.requires_grad_(False)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias_p = nn.Parameter(torch.zeros(A + B))
        self.bias_u = nn.Parameter(torch.zeros(Vt))                                   # 単位ごとの補正（光の外、Vt 個）
        self.alpha = alpha

    def forward(self, X):
        """X (B, T) → 各時刻の全単位の対数確率 (T, B, Vt)"""
        prev, outs = None, []
        for t in range(X.shape[1]):
            u = self.waves[X[:, t]]
            if prev is not None and self.alpha > 0:
                u = (1 - self.alpha) * u + self.alpha * prev
            v = self.sw.field(u[:, None, None, :])[:, 0, 0]
            I = torch.fft.fftshift(torch.fft.fft(v, norm="ortho"), dim=-1).abs().pow(2)
            I = I / (I.mean(-1, keepdim=True) + 1e-12)
            s = self.gain * (I[:, self.bins] + 1e-3).log() + self.bias_p     # P ∝ 明るさ^gain（集光で明るさが数百倍になっても点数が暴れない）
            la = F.log_softmax(s[:, :self.A], -1)
            lb = F.log_softmax(s[:, self.A:], -1)
            outs.append(F.log_softmax(la[:, self.pa] + lb[:, self.pb] + self.bias_u, -1))
            prev = v / (v.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
        return torch.stack(outs)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--assign", default="ctx", help="ctx / random")
    ap.add_argument("--input", default="parts", help="unit / parts")
    ap.add_argument("--A", type=int, default=56)
    ap.add_argument("--B", type=int, default=56)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--P", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tr_docs, te_docs = load_docs(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "cc100_ja.txt"))
    vocab = build_vocab(tr_docs, a.K)
    vid = {s: i for i, s in enumerate(vocab)}
    vset, maxlen = set(vocab), max(len(s) for s in vocab if s != "<不明>")
    tok = lambda d: [vid[s] for s in tokenize(d, vset, maxlen)]
    tr_stream = [tok(d) for d in tr_docs]
    te_stream = [tok(d) for d in te_docs]
    Vt = len(vocab)
    if a.A * a.B < Vt:
        raise SystemExit(f"A·B={a.A * a.B} が単位の数 {Vt} より小さい")
    pa, pb = assign_parts(vocab, tr_stream, a.A, a.B, a.assign, a.seed)
    lens = torch.tensor([1 if s == "<不明>" else len(s) for s in vocab], dtype=torch.float32)
    R = dict(vars(a), vocab=Vt, windows=a.A + a.B)
    # 割り当ての例（ctx で同じ a にまとまった単位）
    if a.assign == "ctx":
        freq = Counter(t for s in tr_stream for t in s)
        R["examples"] = {k: [vocab[u] for u in sorted((pa == k).nonzero().flatten().tolist(), key=lambda u: -freq[u])[:8]] for k in range(3)}

    def windows(stream, step):
        W = []
        for s in stream:
            for i in range(0, len(s) - a.T, step):
                W.append(s[i:i + a.T + 1])
        return torch.tensor(W)

    Wtr, Wte = windows(tr_stream, 1), windows(te_stream, a.T)
    model = PartsMemory(Vt, pa, pb, a.A, a.B, a.L, a.D, a.P, a.alpha, a.input, a.seed, dev).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(a.steps):
        w = Wtr[torch.randint(len(Wtr), (128,), generator=gen)].to(dev)
        out = model(w[:, :-1])
        loss = F.nll_loss(out[1:].reshape(-1, Vt), w[:, 2:].T.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        ce_sum, ch_sum, correct, count = 0.0, 0.0, 0, 0
        for j in range(0, len(Wte), 256):
            w = Wte[j:j + 256].to(dev)
            out = model(w[:, :-1])[1:]
            y = w[:, 2:].T
            ce_sum += F.nll_loss(out.reshape(-1, Vt), y.reshape(-1), reduction="sum").item()
            ch_sum += lens[y.cpu()].sum().item()
            correct += (out.argmax(-1) == y).sum().item()
            count += y.numel()
    R.update(params=model.n_params(), token_acc=round(correct / count, 4), ce_per_token=round(ce_sum / count, 4),
             ce_per_char=round(ce_sum / ch_sum, 4), sec=round(time.time() - t0, 1))
    print(json.dumps(R, ensure_ascii=False), flush=True)
