# リポジトリ直下で実行する: bash experiments/sweeps/sweep33.sh
# 実験33: 位置に合わせて色を少しずつ変える「虹」（コヒーレンス・分散）と、順番つきの出力
R="python experiments/exp33_rainbow.py"
for rb in 0.25 0.5 1.0; do for t in order skills4; do
  PYTHONIOENCODING=utf-8 $R --task $t --pos phase --mix rainbow --rb $rb 2>&1 | tail -1
done; done
for sp in 0.05 0.2; do for t in order skills4; do
  PYTHONIOENCODING=utf-8 $R --task $t --pos phase --mix disp --spread $sp 2>&1 | tail -1
done; done
for t in copy shift; do for r in shared color code; do
  PYTHONIOENCODING=utf-8 python experiments/exp33b_seqout.py --task $t --read $r 2>&1 | tail -1
done; done
for rb in 0.25 0.5; do
  PYTHONIOENCODING=utf-8 $R --task text --n 8 --pos sphere --mix rainbow --rb $rb --freeze_w 2>&1 | tail -1
done
