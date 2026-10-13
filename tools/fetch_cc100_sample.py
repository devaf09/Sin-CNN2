"""
fetch_cc100_sample.py : CC-100 日本語コーパスの「先頭の一部だけ」を取って、学習用に整形する

なぜ自前のストリーム処理か
  * ja.txt.xz は約15GB。HuggingFace の load_dataset("cc100") は全体をダウンロード・展開する作りで、
    さらに datasets 4.0 以降は読み込みスクリプト方式が使えない。
  * このスクリプトは xz を HTTP で少しずつ受け取りながら展開し、必要な文字数に達したら通信を止める。
    必要なのは数MB程度の通信だけ。標準ライブラリだけで動く (numpy も不要)。

出力: 1行 = 1文書 (08 の split=doc がこの形式を使う)。文書の途中の改行は除いてある。

使い方
  python fetch_cc100_sample.py [出力.txt] [--chars 600000] [--min-doc 200] [--max-mb 80]
                               [--file ローカルの.txt.xz または .txt] [--url URL] [--keep-nonjis]
    --chars   集める文字数の目標 (既定 60万)
    --min-doc 文書の最小文字数 (既定 200。短い文書はメニューや広告が多い)
    --max-mb  ダウンロードする圧縮データの上限MB (既定 80)
    --file    手元にある ja.txt.xz などから読む (ネット不要)
    --keep-nonjis  Shift_JIS(JIS X 0208) に無い文字も残す (既定は除去。絵文字・簡体字・異体字が消えて語彙が小さくなる)

フィルタ (どれも『なぜ除くか』の理由つき。結果の内訳は最後に表示される)
  * 日本語らしさ: かな・漢字が6割以上、ひらがなが15%以上 (中国語だけの文書・英文・コードを除く)
  * URL / メールを含む文書 (広告・リンク集)
  * 同じ段落の再登場 (サイト共通のフッタ・メニュー) と、冒頭または末尾が同じ文書 (重複)
  * JIS外の文字が3%を超える文書 (絵文字だらけ・外国語混じり)。3%以下なら該当文字だけ取り除く

注意 (ライセンスと内容)
  * CC-100 のライセンスは明記されておらず、Common Crawl の利用規約に従う。元は Web ページの本文。
    個人の実験用にとどめ、取ったテキストそのものを GitHub などで公開しないこと。
  * Web 由来なので、内容に不適切な文章や宣伝が混ざりうる。先頭部分だけなので話題や出所が偏る可能性もある
    (整形後に prepare_corpus.py の『10分割のずれ』で確認できる)。
"""
import codecs
import hashlib
import lzma
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter

DEFAULT_URL = "https://data.statmt.org/cc-100/ja.txt.xz"
URL_RE = re.compile(r"https?:|www\.|\w@\w|\.(com|net|jp|org)\b", re.I)
KANA_KANJI = re.compile(r"[぀-ヿ㐀-鿿]")
HIRA = re.compile(r"[ぁ-ゟ]")


def open_stream(url=None, path=None, chunk=1 << 20, max_bytes=None):
    """圧縮データを chunk ずつ返すイテレータ。max_bytes を超えたら止まる。"""
    got = 0
    if path:
        f = open(path, "rb")
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (corpus sampler)"})
        f = urllib.request.urlopen(req, timeout=60)
    try:
        while True:
            b = f.read(chunk)
            if not b:
                return
            got += len(b)
            yield b, got
            if max_bytes and got >= max_bytes:
                return
    finally:
        f.close()


def text_lines(chunks, is_xz):
    """圧縮データ(または平文)のチャンク列 -> 行のイテレータ。不完全な末尾行は次のチャンクと連結する。"""
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ) if is_xz else None
    udec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    for b, got in chunks:
        raw = dec.decompress(b) if dec else b
        buf += udec.decode(raw)
        *lines, buf = buf.split("\n")
        for ln in lines:
            yield ln, got
        if dec is not None and dec.eof:
            break
    if buf:
        yield buf, got


def iter_docs(lines):
    """空行で区切られた文書を、段落のリストとして返す。"""
    cur, last = [], 0
    for ln, got in lines:
        last = got
        ln = ln.strip()
        if ln:
            cur.append(ln)
        elif cur:
            yield cur, got
            cur = []
    if cur:
        yield cur, last


def is_jis(ch):
    try:
        ch.encode("shift_jis")
        return True
    except UnicodeEncodeError:
        return False


def main():
    argv = sys.argv[1:]
    opt = lambda name, default: argv[argv.index(name) + 1] if name in argv else default
    target = int(opt("--chars", 600000))
    min_doc = int(opt("--min-doc", 200))
    max_mb = int(opt("--max-mb", 80))
    src_file = opt("--file", None)
    url = opt("--url", DEFAULT_URL)
    keep_nonjis = "--keep-nonjis" in argv
    skip = {"--chars", "--min-doc", "--max-mb", "--file", "--url"}
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in skip)]
    out = pos[0] if pos else "cc100_ja_sample.txt"

    is_xz = (src_file or url).endswith(".xz")
    print(f"取得元: {src_file or url} | 目標 {target} 文字 | 上限 {max_mb}MB(圧縮) | 出力 {out}")
    t0 = time.time()
    stats = Counter()
    seen_par, seen_edge = set(), set()
    h = lambda s: hashlib.md5(s.encode("utf-8")).digest()[:8]
    kept, total = [], 0
    got = 0
    try:
        stream = open_stream(url=None if src_file else url, path=src_file, max_bytes=max_mb << 20)
        for paras, got in iter_docs(text_lines(stream, is_xz)):
            stats["読んだ文書"] += 1
            # 段落の重複(サイト共通のフッタやメニュー)を除く
            fresh = []
            for p in paras:
                p = unicodedata.normalize("NFKC", p)
                k = h(p)
                if len(p) >= 15 and k in seen_par:
                    stats["重複段落を除去(段落数)"] += 1
                    continue
                seen_par.add(k)
                fresh.append(p)
            doc = "".join(fresh)
            if len(doc) < min_doc:
                stats[f"短い文書(<{min_doc}文字)"] += 1
                continue
            if URL_RE.search(doc):
                stats["URL/メールを含む"] += 1
                continue
            jp = len(KANA_KANJI.findall(doc)) / len(doc)
            if jp < 0.6 or len(HIRA.findall(doc)) / len(doc) < 0.15:
                stats["日本語らしくない(かな漢字<60% or ひらがな<15%)"] += 1
                continue
            if not keep_nonjis:
                bad = sum(1 for ch in doc if not is_jis(ch))
                if bad / len(doc) > 0.03:
                    stats["JIS外の文字が3%超"] += 1
                    continue
                if bad:
                    doc = "".join(ch for ch in doc if is_jis(ch))
                    stats["JIS外の文字を取り除いた文書"] += 1
            e1, e2 = h(doc[:40]), h(doc[-40:])
            if e1 in seen_edge or e2 in seen_edge:
                stats["冒頭/末尾が既出の重複文書"] += 1
                continue
            seen_edge.update((e1, e2))
            kept.append(doc)
            total += len(doc)
            if total >= target:
                break
    except (OSError, lzma.LZMAError) as e:
        print(f"\n取得に失敗しました: {type(e).__name__}: {e}")
        print("  * ネットに繋がらない/ブロックされている場合は、ja.txt.xz を別の方法で取得して --file で指定してください")
        if not kept:
            sys.exit(1)

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    cnt = Counter("".join(kept))
    print(f"\n採用 {len(kept)} 文書 / {total} 文字 / 文字の種類 {len(cnt)} | 通信した圧縮データ {got / 1e6:.1f}MB | {time.time() - t0:.0f}秒")
    print("除外の内訳:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    if total < target:
        print(f"\n目標の {target} 文字に届きませんでした。--max-mb を増やすか --min-doc を下げてください。")
    print(f"\n書き出し: {out}")
    print("次は:  python tools/prepare_corpus.py " + out + " 出力.txt --min-count 1000   (整形と偏りの診断)")


if __name__ == "__main__":
    main()
