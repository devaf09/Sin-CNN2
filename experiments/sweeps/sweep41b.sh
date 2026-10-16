# リポジトリ直下で実行する: bash experiments/sweeps/sweep41b.sh
# 実験41b: 光の記憶 × 大きいモデル（比較: 文字列を一度に入れる方式の大きいモデル 49.1%、3-gram 53.6%）
for al in 0.2 0.3; do
  PYTHONIOENCODING=utf-8 python experiments/exp41_textrnn.py --n 8 --alpha $al --D 32 --P 256 --steps 6000 2>&1 | tail -1
done
