"""SinWave の到達度を「ポイント」で表す（合計 100 点）。

    python tools/scoreboard.py            # 表を表示
    python tools/scoreboard.py --md       # docs/scoreboard.md を書き出す

考え方
  各項目に「ゼロ点の基準（何もしない場合: 偶然・最頻の答えなど）」と「目標（最低限の指標: 定番の手法の値）」を置き、
  今の値がその間のどこにあるかで採点する（目標に届けば満点、越えても満点）。
    点 = 配点 × clip((今の値 − ゼロ点の基準) / (目標 − ゼロ点の基準), 0, 1)
  小さいほど良い項目（確信度 CE、時間など）も、同じ式でそのまま扱える（基準と目標の大小が逆になるだけ）。
  時間のように桁で変わるものは、対数（log10）で採点する。
新しい結果が出たら、下の ITEMS の「今の値」と「出典」を書き換えて再実行する。
"""
import math
import sys

# (軸, 項目, 配点, ゼロ点の基準, 目標, 今の値, 対数で採点, 目標の根拠, 今の値の出典)。今の値が None なら未測定（0 点）
ITEMS = [
    ("言語", "次の文字の正解率 (%)", 8, 36.3, 53.6, 51.1, False, "3-gram（直前 2 文字の表）", "実験41b 光の記憶"),
    ("言語", "次の文字の確信度 CE（小さいほど良い）", 8, 3.09, 1.80, 1.96, False, "3-gram", "実験41b"),
    ("言語", "文脈の活用（文脈 1 文字→最良の伸び、点）", 5, 0.0, 12.9, 9.8, False, "n-gram の伸び 42.5→55.4", "実験41b 41.3→51.1"),
    ("言語", "文字列の分類（16 クラス、%）", 5, 6.25, 95.1, 95.2, False, "ロジスティック回帰", "実験30・32（色ごと）"),
    ("言語", "文字の並び（順序課題、%）", 4, 50.0, 100.0, 100.0, False, "隣り合う 2 文字の回帰", "実験30・33"),
    ("言語", "文字列→文字列の規則（未知の文字列、%）", 5, 6.25, 100.0, 99.9, False, "完全に正しい変換", "実験37 逆順"),
    ("画像", "手書き数字の認識 MNIST（%）", 10, 10.0, 92.5, None, False, "ロジスティック回帰", "未測定（SinWave-Tree で実施中）"),
    ("画像", "画像の継続学習 Split-MNIST（クラスを順に追加、%）", 10, 20.0, 80.0, None, False, "少量の過去データを使う既存手法", "未測定（SinWave-Tree で実施中）"),
    ("得意分野", "2 技能の融合（同時学習との差、点。小さいほど良い）", 7, 10.0, 0.0, 1.2, False, "差なし", "実験21"),
    ("得意分野", "4 技能の融合（同時学習との差、点）", 7, 20.0, 0.0, 4.3, False, "差なし", "実験21"),
    ("得意分野", "取消の厳密さ（外して元に戻るか）", 3, 0.0, 1.0, 1.0, False, "完全に戻る", "実験02・03・テスト"),
    ("得意分野", "未知の物（例 0 個）の認識（既知と未知の調和平均、%）", 7, 0.0, 80.0, 72.7, False, "既知・未知とも 80%", "実験36"),
    ("得意分野", "少数例で新しいクラスを追加（例 5 個、%）", 6, 8.3, 55.1, 21.2, False, "文字の出現回数で典型と照合", "実験28"),
    ("得意分野", "短い記憶（同じ文字を直前の文字で出し分け、%）", 2.5, 50.0, 100.0, 100.0, False, "完全に出し分け", "実験38b"),
    ("得意分野", "長い記憶（文脈 2→5 文字の伸び、点）", 2.5, 0.0, 1.8, 1.2, False, "n-gram 3→4-gram の伸び", "実験41b 49.9→51.1"),
    ("効率", "1 技能の大きさ（KB、小さいほど良い）", 3, 1000.0, 50.0, 10.1, False, "50KB 以下", "最小版 v1.1"),
    ("効率", "推論の速さ（µs/件、GPU、小さいほど良い）", 4, 1000.0, 1.0, 61.0, True, "1µs（n-gram は 0.1µs）", "最小版 v1.1 虹"),
    ("効率", "1 技能の学習時間（秒、小さいほど良い）", 3, 3600.0, 10.0, 106.0, True, "10 秒", "L=1024 色ごと 71ms × 1500"),
]


def score(lo, hi, cur, use_log):
    if cur is None:
        return 0.0
    if use_log:
        lo, hi, cur = math.log10(lo), math.log10(hi), math.log10(cur)
    r = (cur - lo) / (hi - lo)
    return max(0.0, min(1.0, r))


def table():
    rows, axes = [], {}
    for axis, name, w, lo, hi, cur, lg, why, src in ITEMS:
        r = score(lo, hi, cur, lg)
        rows.append((axis, name, w, lo, hi, cur, r * w, why, src))
        a = axes.setdefault(axis, [0.0, 0.0, 0.0])
        a[0] += r * w
        a[1] += w
        a[2] += w if cur is not None else 0
    return rows, axes


def markdown():
    rows, axes = table()
    total = sum(v[0] for v in axes.values())
    measured = sum(v[2] for v in axes.values())
    out = ["# SinWave の到達度（ポイント）", "",
           "各項目に「ゼロ点の基準」（何もしない場合）と「目標」（最低限の指標＝定番の手法の値）を置き、今の値がその間のどこにあるかで採点しています。",
           "目標に届けば満点です。計算式と数値は `tools/scoreboard.py` にあり、新しい結果が出たら書き換えて `python tools/scoreboard.py --md` で更新します。", "",
           f"## 合計: **{total:.1f} / 100 点**（測定済みの項目だけなら {total:.1f} / {measured:.0f} 点）", "",
           "| 軸 | 点 | 配点 | 達成率 |", "|---|---|---|---|"]
    for axis, (got, w, mw) in axes.items():
        out.append(f"| {axis} | {got:.1f} | {w:g} | {100 * got / w:.0f}% |")
    out += ["", "## 項目ごと", "",
            "| 軸 | 項目 | 配点 | ゼロ点の基準 | 目標 | 今の値 | 点 | あと | 目標の根拠 | 出典 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for axis, name, w, lo, hi, cur, got, why, src in rows:
        c = "未測定" if cur is None else f"{cur:g}"
        out.append(f"| {axis} | {name} | {w:g} | {lo:g} | {hi:g} | {c} | {got:.1f} | {w - got:.1f} | {why} | {src} |")
    rest = sorted(rows, key=lambda r: -(r[2] - r[6]))[:5]
    out += ["", "## 伸びしろが大きい項目（残りの点が多い順）", ""]
    for axis, name, w, lo, hi, cur, got, why, src in rest:
        out.append(f"- **{name}**（{axis}）: あと {w - got:.1f} 点")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    md = markdown()
    if "--md" in sys.argv:
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "docs" / "scoreboard.md"
        p.write_text(md, encoding="utf-8")
        print("書き出し:", p)
    else:
        print(md)
