"""実験28: 典型の模様で判定し、新しいクラスを板の学習なしで追加する（ユーザー案）。

着想: 1 クラス = 1 光点ではなく、出力面全体の明るさの模様（光点群）でクラスを表す。
似た物（赤いボール・リンゴ・トマト）は光点の一部を共有するので、新しい物（ザクロ）を
既存の光点の組み合わせとして、板を学習し直さずに覚えられるのではないか。間違えても大きく外れにくいのではないか。

課題: 特徴 8 個（各 3 文字）。クラス = 特徴 3 個の組み合わせ。文字は確率 0.7 でクラスの特徴の文字、残りは雑音の文字。
  基本 8 クラス: 板の学習に使う
  新しい 4 クラス: 既存の特徴の新しい組み合わせ。板の学習には使わず、k 個の例の模様を平均して登録するだけ

判定の方法
  window : 検出器の窓の明るさ（従来。基本クラスのみ）
  proto  : 出力面の明るさの模様 (L 次元) と、各クラスの典型（例の模様の平均）の相関
  fourier: 模様をフーリエ変換した振幅の低い周波数 64 個（コンパクト、位置のずれに強い）の相関
  bag    : 比較相手。文字の出現回数のベクトルで同じ「典型との相関」（マップのない定番の方法）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う

import argparse
import itertools
import json
import random
import warnings

import torch

import sinwave_minimal as sw

warnings.filterwarnings("ignore", message=".*文字を超えている")

FEAT_CHARS = "あいうえおかきくけこさしすせそたちつてとなにぬね"   # 8 特徴 × 3 文字
NOISE = "はひふへほまみむめもやゆよらりるれろわを"


def make_task(seed, n_train=300, n_test=150):
    r = random.Random(seed)
    feats = [FEAT_CHARS[3 * i:3 * i + 3] for i in range(8)]
    combos = list(itertools.combinations(range(8), 3))
    r.shuffle(combos)
    # 基本クラスで全特徴が使われるように選ぶ
    while True:
        base = combos[:8]
        if len({f for c in base for f in c}) == 8:
            break
        r.shuffle(combos)
    novel = [c for c in combos[8:] if c not in base][:4]
    classes = base + novel

    def sample(rr, ci):
        chars = "".join(feats[f] for f in classes[ci])
        return "".join(rr.choice(chars) if rr.random() < 0.7 else rr.choice(NOISE) for _ in range(rr.randint(4, 8)))

    r1, r2 = random.Random(seed * 7 + 1), random.Random(seed * 7 + 2)
    train = {ci: [sample(r1, ci) for _ in range(n_train)] for ci in range(12)}
    test = {ci: [sample(r2, ci) for _ in range(n_test)] for ci in range(12)}
    return classes, train, test


def corr_classify(feats, protos):
    """中心化した相関で、一番似た典型の番号を返す。feats (N, d)、protos (K, d)。"""
    f = feats - feats.mean(1, keepdim=True)
    p = protos - protos.mean(1, keepdim=True)
    f = f / (f.norm(dim=1, keepdim=True) + 1e-9)
    p = p / (p.norm(dim=1, keepdim=True) + 1e-9)
    return (f @ p.T).argmax(1)


@torch.no_grad()
def patterns(net, texts):
    out = []
    for i in range(0, len(texts), 512):
        out.append(net.intensity(net.encode(texts[i:i + 512])).cpu())
    return torch.cat(out)


def fourier(P, K=64):
    return torch.fft.rfft(P, dim=-1).abs()[:, 1:K + 1]


def bag(texts):
    idx = {c: i for i, c in enumerate(FEAT_CHARS + NOISE)}
    X = torch.zeros(len(texts), len(idx))
    for b, t in enumerate(texts):
        for c in t:
            X[b, idx[c]] += 1
    return X


def overlap(classes, a, b):
    return len(set(classes[a]) & set(classes[b])) / 3


def evaluate(feat_fn, train, test, classes, cls_ids, k, rng):
    """cls_ids のクラスについて、train から k 個（基本クラスは全部）で典型を作り、test で当てる。"""
    protos = []
    for ci in cls_ids:
        ex = train[ci] if ci < 8 else rng.sample(train[ci], k)
        protos.append(feat_fn(ex).mean(0))
    protos = torch.stack(protos)
    res = {}
    for name, ids in (("base", [c for c in cls_ids if c < 8]), ("novel", [c for c in cls_ids if c >= 8])):
        if not ids:
            continue
        xs = sum([test[c] for c in ids], [])
        ys = torch.tensor(sum([[cls_ids.index(c)] * len(test[c]) for c in ids], []))
        pred = corr_classify(feat_fn(xs), protos)
        res[name] = (pred == ys).float().mean().item()
        wrong = (pred != ys).nonzero()[:, 0]
        if len(wrong):   # 間違えたとき、正解と何割の特徴を共有するクラスに間違えたか
            res[name + "_err_overlap"] = sum(overlap(classes, cls_ids[ys[i]], cls_ids[pred[i]]) for i in wrong) / len(wrong)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patterns", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500, help="板の学習ステップ数（0 なら学習しないランダムな板）")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes, train, test = make_task(a.seed)
    R = dict(seed=a.seed, patterns=a.patterns, steps=a.steps)
    # 基準: 無作為に選んだ別クラスとの特徴の共有率（間違え方が「似たものへ」偏っているかの比較用）
    R["chance_overlap_base"] = sum(overlap(classes, i, j) for i in range(8) for j in range(8) if i != j) / 56
    R["chance_overlap_all"] = sum(overlap(classes, i, j) for i in range(12) for j in range(12) if i != j) / 132

    # 基本 8 クラスで板を学習（窓での判定）
    net = sw.SinWaveNet(8, list(range(8)), patterns=a.patterns, seed=a.seed).to(dev)
    if a.steps > 0:
        sw.train(net, sum([train[c] for c in range(8)], []), sum([[c] * len(train[c]) for c in range(8)], []),
                 steps=a.steps, seed=a.seed)
    xs = sum([test[c] for c in range(8)], [])
    ys = sum([[c] * len(test[c]) for c in range(8)], [])
    pred = sw.predict(net, xs)
    R["window_base"] = sum(p == y for p, y in zip(pred, ys)) / len(ys)
    wrong = [(y, p) for p, y in zip(pred, ys) if p != y]
    R["window_base_err_overlap"] = sum(overlap(classes, y, p) for y, p in wrong) / max(len(wrong), 1)

    feat_fns = {"proto": lambda t: patterns(net, t), "fourier": lambda t: fourier(patterns(net, t)), "bag": bag}
    for name, fn in feat_fns.items():
        R[f"{name}_base8"] = evaluate(fn, train, test, classes, list(range(8)), 0, random.Random(0))
        for k in (1, 5, 20):
            R[f"{name}_add4_k{k}"] = evaluate(fn, train, test, classes, list(range(12)), k, random.Random(a.seed + k))

    if a.steps == 0:
        print(json.dumps(R, ensure_ascii=False), flush=True)
        sys.exit()
    # 上限の目安: 12 クラス全部を最初から板で学習（窓での判定）
    full = sw.SinWaveNet(12, list(range(12)), patterns=a.patterns, seed=a.seed).to(dev)
    sw.train(full, sum([train[c] for c in range(12)], []), sum([[c] * len(train[c]) for c in range(12)], []), seed=a.seed)
    for name, ids in (("base", range(8)), ("novel", range(8, 12))):
        xs = sum([test[c] for c in ids], [])
        ys = sum([[c] * len(test[c]) for c in ids], [])
        R[f"full12_window_{name}"] = sum(p == y for p, y in zip(sw.predict(full, xs), ys)) / len(ys)
    print(json.dumps(R, ensure_ascii=False), flush=True)
