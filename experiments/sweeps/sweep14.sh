# リポジトリ直下で実行する: bash experiments/sweeps/sweep14.sh
# 実験14: 実テキスト。対数応答の検出器、大きいモデルでの並列学習→融合
for args in "--S 1024 --L 2048 --detlog" "--S 1024 --L 4096 --D 32 --P 128 --detlog --parallel"; do
  PYTHONIOENCODING=utf-8 python experiments/exp06_nextchar.py --bias --n 2 --M 1 $args 2>&1 | tail -1
done
