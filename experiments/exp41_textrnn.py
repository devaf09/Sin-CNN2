"""実験41: 光の記憶で実テキストの次の文字を予測する（実験38b の方式を CC-100 日本語に使う）。

これまで: 文脈の文字列を一度に全部入れていた。文字が増えるほど、直前の文字の信号が他の文字の光に埋もれ、
          文脈を 3 文字以上に伸ばしても 2 文字を超えなかった（実験30・32・40）。
今回: 1 文字ずつ順に入れ、入力 = (1−α)·今の文字の光 + α·前の出力の光（光の帰還）。直前の文字は常に一番強く入り、
      古い文字は少しずつ薄まって持ち越される。光を戻す回路ごと、並びを通して学習する。
各時刻 t で「t+1 文字の文脈から次の文字」を予測させ、文脈が伸びるほど正答率が上がるかを見る。

文字は平面波（出現順位の上位 V−1 字 + その他、周波数 = 番号·g mod L）。出力は 64 クラス（頻出 63 字 + その他）で、
窓は各クラスの文字の周波数の位置（レンズの後でその文字の光が集まる場所）に置く（出力の光が次の入力と同じ形になる）。
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
import torch.nn.functional as F

from exp06_nextchar import load
from sinwave.model import Config, SinWave, char_phase


def coprime_stride(L):
    g = int(L * 0.6180339887) | 1
    while math.gcd(g, L) != 1:
        g += 2
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="文脈の文字数（並びの長さ）")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mem", default="mix", help="記憶の方式（実験42）: mix（混ぜるだけ）/ shift（前の光の周波数をずらしてから混ぜる）/ band（帯域ごとに混ぜる割合を学習）")
    ap.add_argument("--shift", type=int, default=7, help="shift: 1 ステップでずらす周波数の本数")
    ap.add_argument("--bands", type=int, default=8, help="band: 帯域の数")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--V", type=int, default=512, help="入力の文字の番号の数")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    data = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "cc100_ja.txt")
    tr, te, vocab, tr_docs, idx = load(data, a.n, 64, 60000, 10000, a.seed)
    other = 63
    # 入力の文字の番号（出現順位）
    rank = {c: i for i, (c, _) in enumerate(Counter("".join(tr_docs)).most_common(a.V - 1))}
    code = lambda c: rank.get(c, a.V - 1)

    def table(items):
        """(文脈, 次の文字のクラス) → 入力の番号 (N, n)、各時刻の正解 (N, n)"""
        X = torch.tensor([[code(c) for c in x] for x, _, _ in items])
        Y = torch.tensor([[idx.get(x[t + 1], other) for t in range(a.n - 1)] + [y] for x, y, _ in items])
        return X, Y

    Xtr, Ytr = table(tr)
    Xte, Yte = table(te)
    L = a.L
    g = coprime_stride(L)
    freq = torch.tensor([(k * g) % L for k in range(a.V)], device=dev)
    chars_by_code = {v: k for k, v in rank.items()}
    theta = torch.tensor([char_phase(chars_by_code.get(k, "〓"), 0) for k in range(a.V)], device=dev)
    x = torch.arange(L, device=dev, dtype=torch.float32)
    waves = torch.exp(1j * (2 * math.pi * freq[:, None] * x / L + theta[:, None]))       # (V, L)

    model = SinWave(Config(L=L, C=64, far=True, bias=True, det_width=3)).to(dev)
    model.add_layers(a.D, 1, "phase", seed=a.seed, P=a.P)
    for l in model.layers:
        l.w.requires_grad_(False)
    det = torch.zeros(64, L, device=dev)                       # クラス k の窓 = その文字の周波数の位置
    for k in range(64):
        ch = vocab[k] if k < 63 else None
        f = freq[code(ch)].item() if ch is not None else freq[a.V - 1].item()
        c = (f + L // 2) % L
        det[k, [(c - 1) % L, c, (c + 1) % L]] = 1
    model.det = det

    def run(X):
        """X (B, n) → 各時刻の点数 (n, B, 64)"""
        prev, outs = None, []
        for t in range(a.n):
            u = waves[X[:, t].to(dev)]
            if prev is not None and a.mem == "band":
                # 帯域ごとに混ぜる割合 α_b を学習（速く忘れる帯域と、長く覚える帯域を同時に持てる）
                al = torch.sigmoid(band_logit)[band_of]                                   # (L,) 周波数ごとの α
                U = (1 - al) * torch.fft.fft(u) + al * torch.fft.fft(prev)
                u = torch.fft.ifft(U)
            elif prev is not None and a.alpha > 0:
                if a.mem == "shift":   # 古い文字ほど遠い帯域へ移す（周波数で並ぶ記憶、シフトレジスタ）
                    prev = prev * torch.exp(1j * (2 * math.pi * a.shift * x / L))
                u = (1 - a.alpha) * u + a.alpha * prev
            v = model.field(u[:, None, None, :])[:, 0, 0]
            far = torch.fft.fftshift(torch.fft.fft(v, norm="ortho"), dim=-1)
            I = far.abs().pow(2)
            outs.append(model.detect(I / (I.mean(-1, keepdim=True) + 1e-12)))
            prev = v / (v.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
        return torch.stack(outs)

    # band: 帯域ごとの混ぜる割合（a.alpha で初期化）
    band_logit = torch.nn.Parameter(torch.full((a.bands,), math.log(a.alpha / (1 - a.alpha)) if 0 < a.alpha < 1 else 0.0, device=dev))
    band_of = (torch.fft.fftfreq(L, device=dev).abs() * 2 * a.bands).long().clamp(max=a.bands - 1)    # 周波数 → 帯域
    extra = [band_logit] if a.mem == "band" else []
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad] + extra, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for s in range(a.steps):
        i = torch.randint(len(Ytr), (256,), generator=gen)
        out = run(Xtr[i])
        loss = F.cross_entropy(out.reshape(-1, 64), Ytr[i].T.reshape(-1).to(dev))   # 全時刻で次の文字を予測
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        out = torch.cat([run(Xte[j:j + 512]) for j in range(0, len(Yte), 512)], 1).cpu()   # (n, N, 64)
    acc = [round((out[t].argmax(1) == Yte[:, t]).float().mean().item(), 4) for t in range(a.n)]
    ce = [round(F.cross_entropy(out[t], Yte[:, t]).item(), 3) for t in range(a.n)]
    if a.mem == "band":
        acc_band = [round(v, 3) for v in torch.sigmoid(band_logit).tolist()]
    print(json.dumps(dict(vars(a), band_alpha=acc_band if a.mem == "band" else None, acc_by_context=acc, ce_by_context=ce, sec=round(time.time() - t0, 1),
                          note="acc_by_context[t] = 文脈 t+1 文字で次の文字を当てた正答率"), ensure_ascii=False), flush=True)
