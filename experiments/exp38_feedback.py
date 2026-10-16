"""実験38: 出力の光をそのまま入力へ戻す（ユーザー案、2026-10-17）。

文字は平面波 exp(i(2π f x/L + θ))。レンズを通すと、その文字の周波数 f の位置の 1 点に集まる。
そこで検出器の窓を「各文字の周波数の位置」に置けば、出力の光そのものが次の文字の入力の光と同じ形になる
（レンズを逆にたどれば、その点は元の平面波に戻る）。つまり、レンズの手前の場をそのまま入力へ戻せる。

(a) ピンポン (--part pingpong): 「あ→い, い→あ」（--K 2）や「あ→い→う→え→お→あ」（--K 5）を 1 ステップずつ学習し、
    「あ」を入れて出た光を、加工せずにそのまま入力へ戻すことを繰り返す（raw）。毎回きれいな文字の光に作り直す場合（clean）と比べる。
(b) 記憶 (--part memory): 「あいうえおあきくけこ」の循環を学習する。入力 = (1−α)·今の文字の光 + α·前の出力の光（案分）。
    2 回目の「あ」（直前が「お」）では「い」ではなく「き」を出すべき。α=0（記憶なし）では区別できないはず。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import argparse
import json
import math

import torch
import torch.nn.functional as F

from sinwave.model import Config, SinWave, char_phase

VOCAB = list("あいうえおかきくけこ")


def coprime_stride(L):
    g = int(L * 0.6180339887) | 1
    while math.gcd(g, L) != 1:
        g += 2
    return g


class Loop:
    """文字の光を入れ、レンズの手前の場（= 次の入力にできる光）と、各文字の窓の点数を返す。"""

    def __init__(self, L, D, P, seed, dev):
        self.L, self.dev = L, dev
        g = coprime_stride(L)
        self.freq = torch.tensor([((ord(c) % 256) * g) % L for c in VOCAB], device=dev)
        theta = torch.tensor([char_phase(c, 0) for c in VOCAB], device=dev)
        x = torch.arange(L, device=dev, dtype=torch.float32)
        self.waves = torch.exp(1j * (2 * math.pi * self.freq[:, None] * x / L + theta[:, None]))   # (V, L)
        cfg = Config(L=L, C=len(VOCAB), far=True, det_width=3)
        self.model = SinWave(cfg).to(dev)
        self.model.add_layers(D, 1, "phase", seed=seed, P=P)
        det = torch.zeros(len(VOCAB), L, device=dev)            # 窓を各文字の周波数の位置（レンズの後）に置く
        for k, f in enumerate(self.freq.tolist()):
            c = (f + L // 2) % L
            det[k, [(c - 1) % L, c, (c + 1) % L]] = 1
        self.model.det = det

    def step(self, u_in):
        """u_in (B, L) → (点数 (B, V), レンズの手前の場 (B, L))"""
        u = self.model.field(u_in[:, None, None, :])[:, 0, 0]
        far = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
        I = far.abs().pow(2)
        I = I / (I.mean(-1, keepdim=True) + 1e-12)
        return self.model.detect(I), u

    def purity(self, u_out, k):
        """出力の光のうち、文字 k の窓に入った割合"""
        far = torch.fft.fftshift(torch.fft.fft(u_out, norm="ortho"), dim=-1)
        I = far.abs().pow(2)
        return ((I * (self.model.det[k] > 0)).sum(-1) / I.sum(-1)).mean().item()


def train(loop, inputs_fn, steps, lr=0.01):
    opt = torch.optim.Adam([p for p in loop.model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        loss = inputs_fn()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="pingpong")
    ap.add_argument("--K", type=int, default=2, help="ピンポン: 循環させる文字数")
    ap.add_argument("--alpha", type=float, default=0.5, help="記憶: 前の出力の光を混ぜる割合")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    loop = Loop(a.L, a.D, a.P, a.seed, dev)
    R = dict(vars(a))

    if a.part == "pingpong":
        K = a.K
        nxt = torch.tensor([(k + 1) % K for k in range(K)], device=dev)

        def loss_fn():
            idx = torch.randint(K, (64,), device=dev)
            lg, _ = loop.step(loop.waves[idx])
            return F.cross_entropy(lg, nxt[idx])
        train(loop, loss_fn, a.steps)
        with torch.no_grad():
            for mode in ("raw", "clean"):
                u = loop.waves[:K].clone()                     # K 本の連鎖を同時に（各文字から開始）
                cur = torch.arange(K, device=dev)
                acc, pur = [], []
                for t in range(20):
                    lg, u_out = loop.step(u)
                    cur = nxt[cur]
                    acc.append(round((lg.argmax(1) == cur).float().mean().item(), 3))
                    pur.append(round(sum(loop.purity(u_out[i:i + 1], cur[i].item()) for i in range(K)) / K, 3))
                    u = u_out if mode == "raw" else loop.waves[lg.argmax(1)]
                R[f"{mode}_acc_by_step"] = acc
                R[f"{mode}_purity_by_step"] = pur
    else:
        seq = [VOCAB.index(c) for c in "あいうえおあきくけこ"]
        S = torch.tensor(seq, device=dev)
        n = len(seq)
        W = n                                                    # 1 周ぶん展開して学習（時間をさかのぼる学習）

        def run(start, steps_, alpha, grad=True):
            """start (B,) から steps_ 文字ぶん回す。戻り値: 点数のリスト、各時刻の正解"""
            B = len(start)
            prev = None
            out, tgt = [], []
            for t in range(steps_):
                pos = (start + t) % n
                x = loop.waves[S[pos]]
                if prev is not None and alpha > 0:
                    x = (1 - alpha) * x + alpha * prev           # 案分: 今の文字の光と前の出力の光を混ぜる
                lg, u_out = loop.step(x)
                prev = u_out / (u_out.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
                out.append(lg)
                tgt.append(S[(pos + 1) % n])
            return out, tgt

        def loss_fn():
            start = torch.randint(n, (32,), device=dev)
            out, tgt = run(start, W + 2, a.alpha)
            # 最初の 2 文字は記憶の立ち上がりとして点数に入れない
            return sum(F.cross_entropy(o, y) for o, y in zip(out[2:], tgt[2:])) / W
        train(loop, loss_fn, a.steps)
        with torch.no_grad():
            out, tgt = run(torch.zeros(1, dtype=torch.long, device=dev), 3 * n, a.alpha)  # 「あ1」から 3 周回す
            preds = [o.argmax(1).item() for o in out]
            ok = [p == y.item() for p, y in zip(preds, tgt)]
            R["acc_after_warmup"] = round(sum(ok[n:]) / len(ok[n:]), 3)   # 1 周目は立ち上がりとして除く
            i_, ki = VOCAB.index("い"), VOCAB.index("き")
            ratio = {}
            for name, p in (("あ1", 0), ("あ2", 5)):         # 2 周目以降の「あ1」「あ2」の時刻
                ts = [t for t in range(n, 3 * n) if t % n == p]
                pr = [torch.softmax(out[t], 1)[0] for t in ts]
                ratio[name] = dict(pred=[VOCAB[preds[t]] for t in ts],
                                   p_い=round(sum(q[i_].item() for q in pr) / len(pr), 3),
                                   p_き=round(sum(q[ki].item() for q in pr) / len(pr), 3))
            R["at_a"] = ratio
            R["pred_cycle2"] = "".join(VOCAB[p] for p in preds[n:2 * n])
    print(json.dumps(R, ensure_ascii=False), flush=True)
