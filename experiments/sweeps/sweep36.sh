# リポジトリ直下で実行する: bash experiments/sweeps/sweep36.sh
# 実験36: 語根の検出器で、新しい物が既知の物に引き寄せられる偏りを直す（出やすさの補正、較正つきの積み重ね）
for seed in 0 1 2; do for p in 0 8; do
  PYTHONIOENCODING=utf-8 python experiments/exp34_roots.py --seed $seed --patterns $p 2>&1 | tail -1
done; done
