"""実験43: よく出る文字列を 1 つの単位（トークン）にまとめ、光の記憶で次の単位を予測する（ユーザー案「認知の圧縮」）。

単位の作り方: 学習データで出現回数の多い 2〜6 文字の文字列を、(出現回数 × 短縮できる文字数) の大きい順に K 個選ぶ。
文章は最長一致で単位に区切る（K=0 なら文字単位）。訓練データで 3 回以上出る文字はすべて単位に含める（それ以外は「不明」）。
光: 単位ごとに固有の周波数（番号·g mod L、g は L と互いに素）の平面波。出力の窓は各単位の周波数の位置（全語彙をスクリーンに並べる）。
記憶: 実験41 と同じ（入力 = (1−α)·今の単位の光 + α·前の出力の光、光を戻す回路ごと学習）。

評価: 単位の長さが違っても比べられるよう、「1 文字あたりのクロスエントロピー」(nats/文字) = Σ 単位ごとの CE / Σ 単位の文字数。
比較: 同じ物差しの文字 n-gram（補間つき）、同じ構成の GRU（記憶つきの小型ニューラルネット。重みの表を持つ）。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import json
import math
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sinwave.model import Config, SinWave


def load_docs(path):
    docs = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    k = int(len(docs) * 0.85)
    return docs[:k], docs[k:]


def build_vocab(train_docs, K, min_char=3):
    chars = Counter("".join(train_docs))
    vocab = ["<不明>"] + sorted(c for c, n in chars.items() if n >= min_char)
    if K > 0:
        sub = Counter()
        for d in train_docs:
            for n in range(2, 7):
                for i in range(len(d) - n + 1):
                    sub[d[i:i + n]] += 1
        cand = sorted(((cnt * (len(s) - 1), s) for s, cnt in sub.items() if cnt >= 5), reverse=True)
        vocab += [s for _, s in cand[:K]]
    return vocab


def tokenize(doc, vocab_set, maxlen):
    out, i = [], 0
    while i < len(doc):
        for n in range(min(maxlen, len(doc) - i), 0, -1):
            piece = doc[i:i + n]
            if piece in vocab_set:
                out.append(piece)
                i += n
                break
        else:
            out.append("<不明>")
            i += 1
    return out


def ngram_char_ce(train_docs, test_docs, order=3, k=0.01):
    """文字 n-gram（補間、加算スムージング）の 1 文字あたり CE (nats)。"""
    V = len(set("".join(train_docs + test_docs))) + 1
    tabs = [defaultdict(Counter) for _ in range(order)]
    for d in train_docs:
        for i, c in enumerate(d):
            for o in range(order):
                if i - o >= 0:
                    tabs[o][d[i - o:i]][c] += 1
    lam = {1: [1.0], 2: [0.3, 0.7], 3: [0.1, 0.3, 0.6], 4: [0.05, 0.15, 0.3, 0.5]}[order]
    tot, n = 0.0, 0
    for d in test_docs:
        for i, c in enumerate(d):
            p = 0.0
            for o in range(order):
                ctx = d[max(i - o, 0):i] if o else ""
                cnt = tabs[o].get(ctx) if i - o >= 0 else None
                po = ((cnt[c] + k) / (sum(cnt.values()) + k * V)) if cnt else 1.0 / V
                p += lam[o] * po
            tot -= math.log(p)
            n += 1
    return tot / n


class OpticalMemory(nn.Module):
    def __init__(self, Vt, L, D, P, alpha, seed, dev):
        super().__init__()
        g = int(L * 0.6180339887) | 1
        while math.gcd(g, L) != 1:
            g += 2
        self.freq = torch.tensor([(i * g) % L for i in range(Vt)], device=dev)
        th = torch.rand(Vt, generator=torch.Generator().manual_seed(1)).to(dev) * 2 * math.pi
        x = torch.arange(L, device=dev, dtype=torch.float32)
        self.register_buffer("waves", torch.exp(1j * (2 * math.pi * self.freq[:, None] * x / L + th[:, None])))
        self.sw = SinWave(Config(L=L, C=1, far=True))
        self.sw.add_layers(D, 1, "phase", seed=seed, P=P)
        for l in self.sw.layers:
            l.w.requires_grad_(False)
        self.bins = (self.freq + L // 2) % L                 # 単位の光がレンズの後で集まる位置 = 出力の窓
        self.gain = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.zeros(Vt))
        self.alpha = alpha

    def forward(self, X):
        """X (B, T) → 各時刻の点数 (T, B, Vt)"""
        prev, outs = None, []
        for t in range(X.shape[1]):
            u = self.waves[X[:, t]]
            if prev is not None and self.alpha > 0:
                u = (1 - self.alpha) * u + self.alpha * prev
            v = self.sw.field(u[:, None, None, :])[:, 0, 0]
            I = torch.fft.fftshift(torch.fft.fft(v, norm="ortho"), dim=-1).abs().pow(2)
            I = I / (I.mean(-1, keepdim=True) + 1e-12)
            outs.append(self.gain * I[:, self.bins] + self.bias)
            prev = v / (v.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
        return torch.stack(outs)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRULM(nn.Module):
    """比較用の記憶つき小型ニューラルネット（埋め込み・重みの表を持つ）。"""

    def __init__(self, Vt, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(Vt, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, Vt)

    def forward(self, X):
        h, _ = self.gru(self.emb(X))
        return self.out(h).transpose(0, 1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024, help="まとめる文字列の数（0 なら文字単位）")
    ap.add_argument("--model", default="optical", help="optical / gru / ngram")
    ap.add_argument("--T", type=int, default=8, help="1 回に学習する単位の並びの長さ")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--P", type=int, default=256)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tr_docs, te_docs = load_docs(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "cc100_ja.txt"))
    R = dict(vars(a))
    if a.model == "ngram":
        for o in (1, 2, 3, 4):
            R[f"char_{o}gram_ce_per_char"] = round(ngram_char_ce(tr_docs, te_docs, o), 4)
        print(json.dumps(R, ensure_ascii=False), flush=True)
        sys.exit()

    vocab = build_vocab(tr_docs, a.K)
    vid = {s: i for i, s in enumerate(vocab)}
    vset, maxlen = set(vocab), max(len(s) for s in vocab if s != "<不明>")
    tok = lambda d: [vid[s] for s in tokenize(d, vset, maxlen)]
    tr_stream = [tok(d) for d in tr_docs]
    te_stream = [tok(d) for d in te_docs]
    lens = torch.tensor([1 if s == "<不明>" else len(s) for s in vocab], dtype=torch.float32)
    n_chars = sum(len(d) for d in te_docs)
    n_tok = sum(len(s) for s in te_stream)
    R.update(vocab=len(vocab), chars_per_token_test=round(n_chars / n_tok, 3))

    # 学習用の窓: 文書ごとの単位の列から、長さ T+1 の連続した窓を作る
    def windows(stream, step):
        W = []
        for s in stream:
            for i in range(0, len(s) - a.T, step):
                W.append(s[i:i + a.T + 1])
        return torch.tensor(W)

    Wtr = windows(tr_stream, 1)
    Wte = windows(te_stream, a.T)                            # 重ならない窓で全体を評価
    Vt = len(vocab)
    if a.model == "optical":
        if Vt > a.L:
            raise SystemExit(f"語彙 {Vt} が L={a.L} を超えている")
        model = OpticalMemory(Vt, a.L, a.D, a.P, a.alpha, a.seed, dev).to(dev)
    else:
        model = GRULM(Vt, a.emb, a.hid).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(a.steps):
        i = torch.randint(len(Wtr), (128,), generator=gen)
        w = Wtr[i].to(dev)
        out = model(w[:, :-1])                                # (T, B, Vt)
        loss = F.cross_entropy(out[1:].reshape(-1, Vt), w[:, 2:].T.reshape(-1))   # 最初の 1 単位は立ち上がり
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        ce_sum, ch_sum, correct, count = 0.0, 0.0, 0, 0
        for j in range(0, len(Wte), 256):
            w = Wte[j:j + 256].to(dev)
            out = model(w[:, :-1])[1:]                        # (T-1, B, Vt)
            y = w[:, 2:].T
            ce = F.cross_entropy(out.reshape(-1, Vt), y.reshape(-1), reduction="sum").item()
            ce_sum += ce
            ch_sum += lens[y.cpu()].sum().item()
            correct += (out.argmax(-1) == y).sum().item()
            count += y.numel()
    R.update(params=model.n_params(), token_acc=round(correct / count, 4),
             ce_per_token=round(ce_sum / count, 4), ce_per_char=round(ce_sum / ch_sum, 4),
             sec=round(time.time() - t0, 1))
    print(json.dumps(R, ensure_ascii=False), flush=True)
