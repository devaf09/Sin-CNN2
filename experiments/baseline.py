"""比較相手: 文字頻度 (unigram) と巡回バイグラムのロジスティック回帰。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import sys

import torch
import torch.nn.functional as F

from sinwave.tasks import task_order, task_topic, task_words


def feats(xs, vocab, bigram):
    idx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    X = torch.zeros(len(xs), V * V if bigram else V)
    for b, s in enumerate(xs):
        for i, c in enumerate(s):
            if bigram:
                X[b, idx[c] * V + idx[s[(i + 1) % len(s)]]] += 1
            else:
                X[b, idx[c]] += 1
    return X


def run(task, bigram):
    (xtr, ytr), (xte, yte) = task
    vocab = sorted(set("".join(xtr + xte)))
    Xtr, Xte = feats(xtr, vocab, bigram), feats(xte, vocab, bigram)
    Ytr, Yte = torch.tensor(ytr), torch.tensor(yte)
    C = max(ytr) + 1
    W = torch.zeros(Xtr.shape[1], C, requires_grad=True)
    b = torch.zeros(C, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(500):
        loss = F.cross_entropy(Xtr @ W + b, Ytr) + 1e-3 * (W**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    acc = lambda X, Y: ((X @ W + b).argmax(1) == Y).float().mean().item()
    return acc(Xtr, Ytr), acc(Xte, Yte), W.numel() + C


if __name__ == "__main__":
    for name, t in [("words", task_words()), ("topic", task_topic()), ("order", task_order())]:
        for bg in (False, True):
            tr, te, p = run(t, bg)
            print(f"{name:6s} {'bigram ' if bg else 'unigram'} params={p:6d} train={tr:.3f} test={te:.3f}")
