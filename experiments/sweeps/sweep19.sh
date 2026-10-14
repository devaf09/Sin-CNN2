# リポジトリ直下で実行する: bash experiments/sweeps/sweep19.sh
# 実験19: レイヤの角速度 ω の初期配置（乱数 / 格子 / 素数 / ゼータのゼロ点）
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far --incoh 8 --illum tilt"
for seed in 0 1; do for w in logrand rand grid prime zeta; do
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B --w_init $w --seed $seed 2>&1 | tail -1
done; done
