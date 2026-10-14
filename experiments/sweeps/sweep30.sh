# リポジトリ直下で実行する: bash experiments/sweeps/sweep30.sh
# 実験30: 文字をレイヤ全体を照らす光として入れる（位置の表し方 × 文字どうしの合わせ方）
for t in order skills4; do for p in pair phase sphere; do for m in coh adj color; do
  PYTHONIOENCODING=utf-8 python experiments/exp30_fullaperture.py --task $t --pos $p --mix $m 2>&1 | tail -1
done; done; done
# 実テキスト: 文脈を伸ばすと良くなるか（スロット方式は n=3 で悪化した）
for n in 2 3 5 8; do for m in coh adj; do
  PYTHONIOENCODING=utf-8 python experiments/exp30_fullaperture.py --task text --n $n --pos pair --mix $m --freeze_w 2>&1 | tail -1
done; done
for args in "--pos sphere --mix adj" "--pos pair --mix color"; do
  PYTHONIOENCODING=utf-8 python experiments/exp30_fullaperture.py --task text --n 8 --freeze_w $args 2>&1 | tail -1
done
