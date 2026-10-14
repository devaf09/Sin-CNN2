# リポジトリ直下で実行する: bash experiments/sweeps/sweep27.sh
# 実験27: 実テキストでモデルを大きくする（ω 固定）
B="--bias --n 2 --M 1 --S 1024 --L 2048 --freeze_w"
for args in "--D 32 --P 128 --steps 12000" "--D 32 --P 256 --steps 8000" "--D 64 --P 128 --steps 8000"; do
  PYTHONIOENCODING=utf-8 python experiments/exp06_nextchar.py $B $args 2>&1 | tail -1
done
