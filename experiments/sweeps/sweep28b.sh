# リポジトリ直下で実行する: bash experiments/sweeps/sweep28b.sh
# 実験28b: フーリエの指紋に距離測定用の参照波を加える
for seed in 0 1 2; do for p in 0 8; do
  PYTHONIOENCODING=utf-8 python experiments/exp28b_reference.py --seed $seed --patterns $p 2>&1 | tail -1
done; done
