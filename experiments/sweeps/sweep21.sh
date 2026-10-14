# リポジトリ直下で実行する: bash experiments/sweeps/sweep21.sh
# 実験21: 平均の滲み + tilt 8 の並列融合を 3 seed に（seed 0 は実験20）
for seed in 1 2; do for n in 2 4; do
  PYTHONIOENCODING=utf-8 python experiments/exp05_scaling.py --n $n --joint --blur mean --incoh 8 --illum tilt --seed $seed 2>&1 | tail -1
done; done
