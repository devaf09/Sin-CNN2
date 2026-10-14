# リポジトリ直下で実行する: bash experiments/sweeps/sweep17.sh
# 実験17: tilt 照明での並列融合を 3 seed で（avg と gate の二つの融合方式）
for seed in 0 1 2; do for n in 2 4; do
  PYTHONIOENCODING=utf-8 python experiments/exp05_scaling.py --n $n --joint --incoh 8 --illum tilt --seed $seed 2>&1 | tail -1
done; done
