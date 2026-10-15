"""実験34: 語根（部品）の検出器で、未知の物を学習なしで認識する（ユーザー案、2026-10-16）。

ユーザー案: サンスクリット語の語根のように、先に共通の部品を作っておけば、新しい概念を部品の組み合わせ（造語）として
扱え、学習の加算・融合にも有利なのでは。未知の物に対応していく仕組みは今後必ず必要になる。

課題は実験28 と同じ: 特徴 8 個（各 3 文字）。クラス = 特徴 3 個の組み合わせ。基本 8 クラスで学習し、新しい 4 クラスは見せない。
実験28 は「クラスごとの検出器」で学習し、新クラスは例の模様の平均で覚えた（例 20 個で 52%、文字の出現回数なら 70%）。
今回は「特徴（語根）ごとの検出器」を学習する（どの部品が含まれるかの多ラベル学習、BCE）。
新クラスは例を 1 つも見せず、部品の組み合わせとして定義するだけ（ゼロショット）。
クラス c の点数 = Σ_{f∈c} log σ(z_f) + Σ_{f∉c} log(1 − σ(z_f))（部品が「あるべき所にあり、ないべき所にない」度合い）。

比較
  bag_roots : 文字の出現回数のロジスティック回帰で同じ「部品の検出 → 組み合わせ」（重み行列を持つ定番の方法）
  class_head: クラスごとの検出器（従来）。新クラスは原理的に答えられない（基本 8 クラスの判定の比較用）
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
import json
import warnings

import torch
import torch.nn.functional as F

import sinwave_minimal as sw
from exp28_proto import bag, make_task

warnings.filterwarnings("ignore", message=".*文字を超えている")


def class_scores(z, classes):
    """部品の点数 z (N, 8) → クラスの点数 (N, len(classes))"""
    lp, ln = F.logsigmoid(z), F.logsigmoid(-z)
    M = torch.zeros(len(classes), 8)
    for i, c in enumerate(classes):
        M[i, list(c)] = 1
    return lp @ M.T + ln @ (1 - M).T


def gzsl(z_fn, test, classes, prior, gammas=(0, 1, 2, 3, 4)):
    """実験36: 全 12 クラスから選ぶとき、新しい物が既知の物に引き寄せられる偏りを直す。
    prior: 部品の出やすさ p_f（学習データでの割合）。z から logit(p_f) を引いて「出やすさの偏り」を除く。
    gamma: 既知のクラスの点数から γ を引く（較正つきの積み重ね）。
    S = 既知の正答、U = 新しい物の正答、H = 調和平均 2SU/(S+U)（一般化ゼロショット学習の標準的な指標）"""
    z = {c: z_fn(test[c]) for c in range(12)}
    out = {}
    for adj in ("raw", "prior"):
        shift = torch.logit(prior.clamp(0.02, 0.98)) if adj == "prior" else torch.zeros(8)
        for g in gammas:
            acc = {}
            for name, ids in (("S", range(8)), ("U", range(8, 12))):
                correct = total = 0
                for c in ids:
                    s = class_scores(z[c] - shift, classes)
                    s[:, :8] -= g
                    correct += (s.argmax(1) == c).sum().item()
                    total += len(s)
                acc[name] = correct / total
            H = 2 * acc["S"] * acc["U"] / (acc["S"] + acc["U"] + 1e-9)
            out[f"{adj}_g{g}"] = dict(S=round(acc["S"], 4), U=round(acc["U"], 4), H=round(H, 4))
    return out


def evaluate(z_fn, test, classes):
    res = {}
    xs = {c: test[c] for c in range(12)}
    z = {c: z_fn(xs[c]) for c in range(12)}
    feat = torch.stack([torch.zeros(8).index_fill_(0, torch.tensor(classes[c]), 1) for c in range(12)])
    res["root_acc"] = sum(((z[c] > 0).float() == feat[c]).float().mean().item() for c in range(12)) / 12
    for name, cand, ids in (("base8", range(8), range(8)), ("all12_base", range(12), range(8)),
                            ("all12_novel", range(12), range(8, 12)), ("novel4", range(8, 12), range(8, 12))):
        cand = list(cand)
        correct = total = 0
        for c in ids:
            s = class_scores(z[c], [classes[k] for k in cand])
            correct += (s.argmax(1) == cand.index(c)).sum().item()
            total += len(s)
        res[name] = round(correct / total, 4)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patterns", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes, train, test = make_task(a.seed)
    xs = sum([train[c] for c in range(8)], [])
    feat = torch.stack([torch.zeros(8).index_fill_(0, torch.tensor(classes[c]), 1) for c in range(8)])
    Yf = torch.cat([feat[c].expand(len(train[c]), 8) for c in range(8)])
    R = dict(seed=a.seed, patterns=a.patterns, classes=[list(c) for c in classes])

    # SinWave: 部品ごとの検出器（8 個の窓）を多ラベルで学習
    net = sw.SinWaveNet(8, list(range(8)), patterns=a.patterns, seed=a.seed).to(dev)
    X = net.encode(xs, device="cpu")
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    for s in range(a.steps):
        i = torch.randint(len(Yf), (256,), generator=gen)
        loss = F.binary_cross_entropy_with_logits(net(X[i].to(dev)), Yf[i].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    @torch.no_grad()
    def z_sinwave(texts):
        return torch.cat([net(net.encode(texts[j:j + 512])) for j in range(0, len(texts), 512)]).cpu()

    R["sinwave_roots"] = evaluate(z_sinwave, test, classes)
    prior = Yf.mean(0)
    R["sinwave_gzsl"] = gzsl(z_sinwave, test, classes, prior)

    # 比較: 文字の出現回数のロジスティック回帰で部品を検出
    Xb = bag(xs)
    W = torch.zeros(Xb.shape[1], 8, requires_grad=True)
    b = torch.zeros(8, requires_grad=True)
    o = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(500):
        l = F.binary_cross_entropy_with_logits(Xb @ W + b, Yf) + 1e-3 * (W ** 2).sum()
        o.zero_grad(); l.backward(); o.step()
    R["bag_roots"] = evaluate(lambda t: (bag(t) @ W + b).detach(), test, classes)
    R["bag_gzsl"] = gzsl(lambda t: (bag(t) @ W + b).detach(), test, classes, prior)

    # 比較: クラスごとの検出器（従来、基本 8 クラスのみ）
    head = sw.train(sw.SinWaveNet(8, list(range(8)), patterns=a.patterns, seed=a.seed).to(dev), xs,
                    sum([[c] * len(train[c]) for c in range(8)], []), steps=a.steps, seed=a.seed)
    xt = sum([test[c] for c in range(8)], [])
    yt = sum([[c] * len(test[c]) for c in range(8)], [])
    R["class_head_base8"] = round(sum(p == y for p, y in zip(sw.predict(head, xt), yt)) / len(yt), 4)
    print(json.dumps(R, ensure_ascii=False), flush=True)
