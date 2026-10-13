"""
prepare_corpus.py : 日本語コーパスの整形と診断 (numpy だけ)

やること
  1. 文字コードを自動判別 (UTF-8 / Shift_JIS(cp932) / EUC-JP)
  2. 青空文庫形式のゴミを除去: 冒頭の記号説明、ルビ《...》と｜、注記［＃...］、末尾の底本情報
     HTMLタグ、制御文字、ゼロ幅文字も除去。NFKC で全角英数・半角カナなどを正規化
  3. 空行と行頭の空白を除去
  4. 診断 (本文は書き換えない):
       - 文字種の内訳、学習に使えない超低頻度文字の数
       - 『末尾15%を検証にすると偏るか』: 末尾と全体の文字分布のずれ、学習側と重複する文章の割合
         (08 の split=tail と split=scatter のどちらが信頼できるかの目安)
       - 10分割した各区間の文体のずれ (途中で別の作品や章が混ざっていないか)
       - 何度も繰り返される行 (ヘッダ・フッタ・広告などの疑い)

注意: ルビ(読み仮名)が本文に混ざっているかは自動では判定できません。
  青空文庫をブラウザからコピーすると『吾輩わがはいは猫である』のように読みが本文に入ります。
  《》付きの .txt (青空文庫のダウンロード版) を使えば、このスクリプトが正しく除去します。
  手元のファイルは、先頭を目で見て『漢字の直後に読みが重複していないか』を確認してください。

使い方
  python prepare_corpus.py 入力.txt [出力.txt] [--drop-repeats] [--drop-nonjp] [--min-count N]
    出力を省略すると 入力_clean.txt に書く
    --drop-repeats : 10文字以上で3回以上繰り返す行を削除
    --drop-nonjp   : 日本語(かな・漢字)が3割未満の行を削除 (英文ヘッダや URL など)
    --min-count N  : 出現が N 回未満の文字を〓に置き換える (語彙を絞りたいとき)
"""
import re
import sys
import unicodedata
from collections import Counter

import numpy as np

JP = re.compile(r"[぀-ヿ㐀-鿿]")


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp932", "euc_jp"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace"), "utf-8(置換あり)"


def clean(text, drop_repeats=False, drop_nonjp=False):
    log = {}
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    n0 = len(t)
    t, c = re.subn(r"\A.*?-{20,}\n【テキスト中に現れる記号について】.*?-{20,}\n", "", t, count=1, flags=re.S)
    log["青空文庫の冒頭(タイトル・作者・記号説明)"] = c
    m = re.search(r"\n底本[：:]", t)
    if m:
        log["末尾の底本情報(文字数)"] = len(t) - m.start()
        t = t[:m.start()]
    t, c = re.subn(r"《[^》\n]*》", "", t)
    log["ルビ《》"] = c
    t = t.replace("｜", "")
    t, c = re.subn(r"［＃[^］\n]*］", "", t)
    log["注記［＃］"] = c
    t, c = re.subn(r"<[^>\n]{1,200}>", "", t)
    log["HTMLタグ"] = c
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if ch == "\n" or (unicodedata.category(ch) not in ("Cc", "Cf") and ch != "﻿"))
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    if drop_repeats:
        rep = Counter(ln for ln in lines if len(ln) >= 10)
        bad = {ln for ln, k in rep.items() if k >= 3}
        log["繰り返し行の削除(行数)"] = sum(1 for ln in lines if ln in bad)
        lines = [ln for ln in lines if ln not in bad]
    if drop_nonjp:
        keep = [ln for ln in lines if len(JP.findall(ln)) >= 0.3 * len(ln)]
        log["日本語が3割未満の行の削除(行数)"] = len(lines) - len(keep)
        lines = keep
    log["整形前の文字数"] = n0
    return lines, log


def jsd(p, q):
    m = (p + q) / 2
    def f(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log(a[k] / b[k])))
    return 0.5 * f(p, m) + 0.5 * f(q, m)


def dist(s, index):
    v = np.zeros(len(index))
    for ch, k in Counter(s).items():
        if ch in index:
            v[index[ch]] = k
    return v / max(v.sum(), 1)


def diagnose(text, block=500, every=7, shingle=12, max_chars=3_000_000):
    if len(text) > max_chars:
        print(f"  (診断は先頭 {max_chars} 文字で行います)")
        text = text[:max_chars]
    cnt = Counter(text)
    index = {c: i for i, c in enumerate(cnt)}
    n = len(text)
    kinds = Counter()
    for ch, k in cnt.items():
        o = ord(ch)
        key = ("ひらがな" if 0x3041 <= o <= 0x309f else "カタカナ" if 0x30a0 <= o <= 0x30ff
               else "漢字" if 0x3400 <= o <= 0x9fff else "英数字" if ch.isascii() and ch.isalnum()
               else "記号・その他")
        kinds[key] += k
    print(f"  文字数 {n} | 文字の種類 {len(cnt)}")
    print("  内訳: " + " / ".join(f"{k} {100 * v / n:.1f}%" for k, v in kinds.most_common()))
    print("  頻出: " + " ".join(f"{c}:{k}" for c, k in cnt.most_common(8)))
    rare1 = sum(1 for k in cnt.values() if k == 1)
    rare2 = sum(1 for k in cnt.values() if k <= 2)
    print(f"  超低頻度の文字: 1回だけ {rare1}種 / 2回以下 {rare2}種 (計 {sum(k for k in cnt.values() if k <= 2)}文字分)"
          f"  -> 学習しても予測できない。多いなら --min-count 3 などで〓に置換")

    blocks = [text[i:i + block] for i in range(0, n, block)]
    val_s = "".join(b for i, b in enumerate(blocks) if i % every == every // 2)
    tr_s = "".join(b for i, b in enumerate(blocks) if i % every != every // 2)
    cut = int(n * 0.85)
    tail_tr, tail_va = text[:cut], text[cut:]

    def report(name, tr, va):
        d = jsd(dist(tr, index), dist(va, index))
        trset = set(tr)
        unseen = 100 * sum(1 for ch in va if ch not in trset) / max(len(va), 1)
        hs = {hash(tr[i:i + shingle]) for i in range(len(tr) - shingle)}
        leak = 100 * sum(1 for i in range(len(va) - shingle) if hash(va[i:i + shingle]) in hs) / max(len(va) - shingle, 1)
        print(f"  {name:<22} 文字分布のずれ(JSD) {d:.4f} | 学習に無い文字 {unseen:.2f}% | 学習側と重複する{shingle}文字列 {leak:.1f}%")
        return d, leak

    print("\n  [検証データの取り方による違い]")
    d_t, l_t = report("末尾15%を検証(tail)", tail_tr, tail_va)
    d_s, l_s = report("散らして検証(scatter)", tr_s, val_s)

    print("\n  [10分割した各区間と全体の文字分布のずれ (大きい区間は別の文体・別の作品の疑い)]")
    whole = dist(text, index)
    ds = []
    for i in range(10):
        seg = text[i * n // 10:(i + 1) * n // 10]
        ds.append(jsd(dist(seg, index), whole))
    med = float(np.median(ds))
    print("  " + " ".join(f"{i + 1}:{d:.3f}" for i, d in enumerate(ds)) + f"   (中央値 {med:.3f})")
    odd = [i + 1 for i, d in enumerate(ds) if d > 2.0 * med and d > 0.03]
    print("  所見:")
    if d_t > 1.8 * d_s and d_t - d_s > 0.01:
        print(f"   * 末尾は全体と文体がかなり違います(tail のずれ {d_t:.3f} vs scatter {d_s:.3f})。"
              "末尾15%を検証にすると精度が実力より低く出ます -> 08 は split=scatter (既定) を使ってください")
    else:
        print("   * 末尾と全体の文体の差は小さめです (tail でも scatter でも大きくは変わらない見込み)")
    if odd:
        print(f"   * 区間 {odd} が全体と違います。別の作品・章・解説文が混ざっていないか確認を")
    if l_t > 5 or l_s > 5:
        print(f"   * 検証側に学習側と同じ{shingle}文字列が多い({max(l_t, l_s):.1f}%)。同じ文章の重複があり、検証が甘くなります")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if not args:
        print(__doc__)
        return
    min_count = 0
    if "--min-count" in sys.argv:
        i = sys.argv.index("--min-count")
        min_count = int(sys.argv[i + 1])
        args = [a for a in args if a != sys.argv[i + 1]]
    src = args[0]
    dst = args[1] if len(args) > 1 else re.sub(r"\.txt$", "", src) + "_clean.txt"
    text, enc = read_text(src)
    print(f"読み込み: {src} (文字コード {enc})")
    lines, log = clean(text, "--drop-repeats" in flags, "--drop-nonjp" in flags)
    out = "\n".join(lines)
    if min_count > 0:
        cnt = Counter(out.replace("\n", ""))
        rare = {c for c, k in cnt.items() if k < min_count}
        out = "".join("〓" if ch in rare else ch for ch in out)
        log[f"出現{min_count}回未満の文字を〓へ置換(種類)"] = len(rare)
    print("\n[整形]")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print(f"  整形後の文字数: {len(out.replace(chr(10), ''))} (行数 {len(lines)})")
    rep = Counter(ln for ln in lines if len(ln) >= 10)
    top = [(ln, k) for ln, k in rep.most_common(5) if k >= 3]
    if top:
        print("  何度も繰り返される行 (ヘッダ/フッタ/広告の疑い。--drop-repeats で削除):")
        for ln, k in top:
            print(f"    {k}回: {ln[:40]}")
    nonjp = sum(1 for ln in lines if len(JP.findall(ln)) < 0.3 * len(ln))
    if nonjp:
        print(f"  日本語が3割未満の行: {nonjp}行 (英文・URL・記号の疑い。--drop-nonjp で削除)")
    print("\n[診断]")
    diagnose(out.replace("\n", ""))
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\n書き出し: {dst}")


if __name__ == "__main__":
    main()
