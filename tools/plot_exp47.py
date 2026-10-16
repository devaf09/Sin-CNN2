"""実験47 の図: 学習する数（パラメータ）と次の文字の確信度 CE（文脈 8 文字）。python tools/plot_exp47.py"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
from visualize import japanese_font

root = pathlib.Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(root / "results" / "exp47.txt", encoding="utf-8") if l.startswith("{")]
# 実験41b の D32 P256（16449 個、6000 ステップ、α=0.3）
optical = [(16449, 1.957, "D32 P256")]
gru = []
for d in rows:
    if d["model"] == "gru":
        gru.append((d["params"], min(d["ce_by_context"]), f"GRU {d['hid']}"))
    elif d["L"] == 1024 and d["steps"] == 6000:
        optical.append((d["params"], min(d["ce_by_context"]), f"D{d['D']} P{d['P']}"))
japanese_font()
fig, ax = plt.subplots(figsize=(7, 4.5))
for pts, c, name in ((sorted(optical), "tab:blue", "光の記憶（SinWave）"), (sorted(gru), "tab:orange", "GRU（重みの表を持つ）")):
    ax.plot([p for p, _, _ in pts], [v for _, v, _ in pts], "o-", color=c, label=name)
    for p, v, t in pts:
        ax.annotate(t, (p, v), textcoords="offset points", xytext=(4, 4), fontsize=8, color=c)
ax.axhline(1.80, color="gray", ls="--", lw=1)
ax.text(11000, 1.805, "文字 3-gram 1.80", color="gray", fontsize=8, va="bottom")
ax.axhline(2.10, color="gray", ls=":", lw=1)
ax.text(11000, 2.095, "2-gram 2.10", color="gray", fontsize=8, va="top")
ax.set_xscale("log")
ax.set_xlabel("学習する数（パラメータ、対数）")
ax.set_ylabel("次の文字の確信度 CE（小さいほど良い）")
ax.set_title("実験47: 大きくすると伸びるか（頻出 63 字 + その他、seed 0）")
ax.legend()
fig.tight_layout()
fig.savefig(root / "docs" / "images" / "exp47_scaling.png", dpi=110)
print("保存: docs/images/exp47_scaling.png")
