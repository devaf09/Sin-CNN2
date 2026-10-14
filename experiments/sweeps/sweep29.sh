# リポジトリ直下で実行する: bash experiments/sweeps/sweep29.sh
# 実験29: 学習した板を敷き詰めて大きな板へ移す
for seed in 0 1 2; do
  PYTHONIOENCODING=utf-8 python experiments/exp29_tile.py --seed $seed 2>&1 | tail -1
done
