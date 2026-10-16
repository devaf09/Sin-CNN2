"""SinWave の学習の様子を、絵で見るためのコード（sinwave_minimal.py の説明用）。

    python examples/visualize.py                     # 画面に出しながら学習（一定ステップごとに絵を更新）
    python examples/visualize.py --save out.png      # 最後の絵を保存
    python examples/visualize.py --gif out.gif       # 学習の途中の絵をつないだアニメーションを保存
    python examples/visualize.py --no-show --gif docs/images/training.gif   # 画面を出さずに保存だけ

画面の 6 つの図
  ① 入力の光        : 1 つの文字列を光にしたときの「周波数の成分」。文字ごとに違う向き（周波数）の光が並ぶ
  ② 板の模様        : 各 Sin の板が、位置ごとに光の位相をどれだけ遅らせるか（学習で変わるのはここだけ。先頭の一部を拡大）
  ③ 光の集まり方    : 板を 1 枚通るたびに、各クラスの窓へ向かう光の割合。
                      学習が進むと、板を通るたびに正解のクラスの線だけが上がっていく（光が正解の窓へ曲げられていく）
  ④ スクリーン      : レンズで集めた後の明るさ。四角が各クラスの検出器の窓。一番明るい窓が答え
  ⑤ 学習の進み具合  : 損失（小さいほど良い）と、学習に使っていない問題での正解率
  ⑥ 答えの例        : いくつかの文字列と、各クラスの点数（確率）

依存: PyTorch、matplotlib（GIF の保存には Pillow）。CPU でも動く（数分）。
"""
import argparse
import pathlib
import random
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave_minimal を使う

import matplotlib
import torch
import torch.nn.functional as F

import sinwave_minimal as sw

warnings.filterwarnings("ignore", message=".*文字を超えている")

# 合成データ: 4 つのクラス。各クラスは好む文字を持ち、文字列にその文字が多く含まれる
KEYS = {0: "あいうえ", 1: "かきくけ", 2: "さしすせ", 3: "たちつて"}
NAMES = ["あ行", "か行", "さ行", "た行"]   # クラスの名前（その行の文字が多い文字列）
POOL = "なにぬねのはひふへほまみむめもやゆよらりるれろわを"


def sample(r, y):
    return "".join(r.choice(KEYS[y]) if r.random() < 0.6 else r.choice(POOL) for _ in range(r.randint(4, 8)))


def japanese_font():
    """日本語を表示できるフォントを選ぶ（見つからなければ既定のまま）。"""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic", "TakaoGothic"):
        if name in have:
            matplotlib.rcParams["font.family"] = name
            return
    print("日本語のフォントが見つからないので、文字が四角になるかもしれない")


def draw(fig, axes, net, sample_text, sample_label, history, test, step, log_x=True):
    import numpy as np
    for ax in axes.flat:
        ax.clear()
    L = net.L
    u = net.encode([sample_text])
    tr = net.trace(u)

    # ① 入力の光（周波数の成分）
    ax = axes[0, 0]
    spec = torch.fft.fftshift(torch.fft.fft(u[0].reshape(-1, L), norm="ortho"), dim=-1).abs().pow(2).mean(0).cpu()
    ax.plot(np.arange(L) - L // 2, spec.numpy(), lw=0.8, color="tab:blue")
    ax.set_title(f"① 入力の光「{sample_text}」の周波数の成分")
    ax.set_xlabel("周波数（光の向き）")
    ax.set_yticks([])

    # ② 板の模様（位相の遅れ）
    ax = axes[0, 1]
    zoom = min(256, L)
    ax.imshow(tr["masks"].numpy()[:, :zoom], aspect="auto", cmap="twilight", interpolation="nearest")
    ax.set_title(f"② 板の模様（位相の遅れ、先頭 {zoom} 画素）")
    ax.set_xlabel("位置")
    ax.set_ylabel("板の番号")

    # ③ 光の伝わり方
    ax = axes[0, 2]
    sp = tr["spectra"].numpy()                                   # (板の数, L) 各板の後の光の向き
    share = []
    for slot in net.own:
        m = net.det[slot].cpu().numpy() > 0
        idx = np.nonzero(m)[0]
        wide = np.zeros(L, bool)
        wide[max(idx.min() - 6, 0):idx.max() + 7] = True              # 窓のまわりを少し広めに
        share.append(sp[:, wide].sum(1) / sp.sum(1))
    for k, sh in enumerate(share):
        ax.plot(range(1, len(sh) + 1), sh * 100, marker="o", ms=3, color=f"C{k}",
                lw=2.5 if k == sample_label else 1, label=NAMES[k] + ("（正解）" if k == sample_label else ""))
    ax.set_title("③ 板を通るたびに、各窓へ向かう光の割合")
    ax.set_xlabel("通った板の数")
    ax.set_ylabel("光の割合 (%)")
    ax.legend(fontsize=8, loc="upper left")

    # ④ スクリーンと検出器の窓
    ax = axes[1, 0]
    scr = tr["screen"].numpy()
    ax.plot(np.arange(L), scr, lw=0.8, color="tab:gray")
    probs = torch.softmax(sw.scores(net, [sample_text])[0][net.own], 0).numpy()
    for k, slot in enumerate(net.own):
        idx = net.det[slot].nonzero()[:, 0].cpu().numpy()
        lo, hi = idx.min(), idx.max()
        ax.axvspan(lo - 6, hi + 6, color=f"C{k}", alpha=0.25 + 0.6 * probs[k])
        ax.text((lo + hi) / 2, scr.max() * 0.95, NAMES[k], ha="center", fontsize=9, color=f"C{k}")
    ax.set_title("④ スクリーン（帯 = 検出器の窓、濃いほど高い点数）", fontsize=10)
    ax.set_xlabel("スクリーン上の位置")
    ax.set_yticks([])

    # ⑤ 学習の進み具合
    ax = axes[1, 1]
    if history["step"]:
        ax.plot(history["step"], history["loss"], color="tab:red", label="損失")
        ax.set_ylabel("損失", color="tab:red")
        ax2 = ax.twinx()
        ax2.plot(history["step"], history["acc"], color="tab:green", label="正解率")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("正解率（学習に使っていない問題）", color="tab:green")
        if log_x:
            ax.set_xscale("symlog", linthresh=1)   # 対数の目盛り（0 も描けるよう 0〜1 だけは等間隔）
            ax.set_xlim(0, max(history["step"][-1], 1) * 1.1)
    ax.set_title(f"⑤ 学習の進み具合（{step} ステップ）")
    ax.set_xlabel("ステップ")

    # ⑥ 答えの例
    ax = axes[1, 2]
    xs = [t for t, _ in test[:6]]
    P = torch.softmax(sw.scores(net, xs)[:, net.own], 1).numpy()
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(xs)))
    ax.set_yticklabels([f"{t}（正解: {NAMES[y]}）" for t, y in test[:6]], fontsize=8)
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels(NAMES, fontsize=9)
    for i in range(len(xs)):
        for k in range(len(NAMES)):
            ax.text(k, i, f"{P[i, k]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("⑥ 答えの例（各クラスの確率）")
    fig.suptitle("SinWave — 正弦の板と回折だけで学ぶ（重み行列なし）", fontsize=13)
    fig.tight_layout()


def draw_steps(steps, every):
    """絵を更新するステップ。"log" なら 1, 1.5, 2, 3, 5, 7 × 10^k（学習は最初の 100 ステップでほぼ決まるので、そこを細かく見る）。"""
    if every == "log":
        marks = {0, steps}
        k = 1
        while k <= steps:
            for m in (1, 1.5, 2, 3, 5, 7):
                if m * k <= steps:
                    marks.add(int(m * k))
            k *= 10
        return marks
    return set(range(0, steps, int(every))) | {steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", default="log", help="絵を更新する間隔: log（最初は細かく、だんだん粗く: 0,1,2,3,5,7,10,15,20,30,50,…）か、ステップ数（例 50）")
    ap.add_argument("--mix", default="coh", help="文字どうしの合わせ方 coh / rainbow / color（rainbow, color は計算が重い）")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--save", default="", help="最後の絵を保存するファイル名 (.png)")
    ap.add_argument("--gif", default="", help="学習の途中の絵をつないだアニメーション (.gif)")
    ap.add_argument("--no-show", action="store_true", help="画面に出さない（保存だけ）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    japanese_font()

    r = random.Random(0)
    train_set = [(sample(r, y), y) for y in KEYS for _ in range(200)]
    test = [(sample(r, y), y) for y in KEYS for _ in range(50)]
    r.shuffle(test)
    net = sw.SinWaveNet(n_slots=4, own=[0, 1, 2, 3], L=a.L, mix=a.mix, max_chars=8).to(a.device)
    X = net.encode([t for t, _ in train_set], device="cpu")
    Y = torch.tensor([y for _, y in train_set])
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen = torch.Generator().manual_seed(0)
    history = dict(step=[], loss=[], acc=[])
    frames = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    if not a.no_show:
        plt.ion()
        plt.show()
    sample_text, sample_label = test[0]

    marks = draw_steps(a.steps, a.every)
    log_x = a.every == "log"
    for step in range(a.steps + 1):
        if step in marks:
            acc = sum(p == y for p, (_, y) in zip(sw.predict(net, [t for t, _ in test]), test)) / len(test)
            history["step"].append(step)
            history["acc"].append(acc)
            history["loss"].append(loss.item() if step else float("nan"))
            draw(fig, axes, net, sample_text, sample_label, history, test, step, log_x)
            if not a.no_show:
                plt.pause(0.01)
            if a.gif:
                fig.canvas.draw()
                import numpy as np
                from PIL import Image
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            print(f"step {step:4d}  正解率 {acc:.3f}", flush=True)
        if step == a.steps:
            break
        i = torch.randint(len(Y), (64,), generator=gen)
        loss = F.cross_entropy(net(X[i].to(a.device)), Y[i].to(a.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    if a.save:
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    if a.gif and frames:
        frames[0].save(a.gif, save_all=True, append_images=frames[1:] + [frames[-1]] * 4, duration=600, loop=0)
        print("保存:", a.gif)
    if not a.no_show:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
