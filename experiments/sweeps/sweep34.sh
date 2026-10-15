# リポジトリ直下で実行する: bash experiments/sweeps/sweep34.sh
# 実験34: 語根（部品）の検出器で、未知の物を学習なしで認識する
for seed in 0 1 2; do for p in 0 8; do
  PYTHONIOENCODING=utf-8 python experiments/exp34_roots.py --seed $seed --patterns $p 2>&1 | tail -1
done; done
