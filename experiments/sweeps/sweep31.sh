# リポジトリ直下で実行する: bash experiments/sweeps/sweep31.sh
# 実験31: 閾値の増幅曲線（弱い光を絞る）と、マイナスの明るさ（差動検出の単純な差）。従来の入力方式で
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far --blur mean"
O="--task order --enc freq --S 128 --L 2048 --P 64 --far"
for args in "" "--nl thresh --nl_theta 0.5" "--nl thresh --nl_theta 1.0" "--diff --diff_mode sub" "--nl thresh --diff --diff_mode sub"; do
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B $args 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $O $args 2>&1 | tail -1
done
