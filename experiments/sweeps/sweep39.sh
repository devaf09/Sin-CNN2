# リポジトリ直下で実行する: bash experiments/sweeps/sweep39.sh
# 実験39: 1 文字を周波数 K 本の「和音」にする（区別できる文字の数を組み合わせで増やす）
X="python experiments/exp30_fullaperture.py --stride"
for K in 1 2 3; do
  PYTHONIOENCODING=utf-8 $X --task skills4 --pos phase --mix color --L 1024 --chord $K 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --task skills4 --pos sphere --mix coh --L 1024 --chord $K 2>&1 | tail -1
done
PYTHONIOENCODING=utf-8 $X --task order --pos sphere --mix coh --L 1024 --chord 2 2>&1 | tail -1
for K in 1 2; do
  PYTHONIOENCODING=utf-8 $X --task text --n 2 --pos pair --mix coh --freeze_w --L 512 --chord $K 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --task text --n 2 --pos pair --mix coh --freeze_w --L 1024 --V 2048 --chord $K 2>&1 | tail -1
done
