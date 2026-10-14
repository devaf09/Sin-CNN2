# リポジトリ直下で実行する: bash experiments/sweeps/sweep11.sh
# 実験11: 照明の効果を 3 seed で確認
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
for seed in 1 2 3; do
  for args in "" "--incoh 8 --illum random" "--incoh 8 --illum tilt" "--incoh 8 --illum hadamard" "--incoh 8 --illum tilt --feedback field"; do
    PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B --seed $seed $args 2>&1 | tail -1
  done
done
