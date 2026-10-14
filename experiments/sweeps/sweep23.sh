# リポジトリ直下で実行する: bash experiments/sweeps/sweep23.sh
# 実験23: 光路の引き算（タスク算術）
for share in 2 3; do for seed in 0 1 2; do
  PYTHONIOENCODING=utf-8 python experiments/exp23_arith.py --share $share --seed $seed 2>&1 | tail -1
done; done
