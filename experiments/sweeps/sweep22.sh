# リポジトリ直下で実行する: bash experiments/sweeps/sweep22.sh
# 実験22: 実テキストの確信度 (CE) は温度のずれか / ω 固定の軽量版
for args in "--calib" "--calib --freeze_w" "--calib --D 32 --P 128 --L 4096"; do
  PYTHONIOENCODING=utf-8 python experiments/exp06_nextchar.py --bias --n 2 --M 1 --S 1024 --L 2048 $args 2>&1 | tail -1
done
