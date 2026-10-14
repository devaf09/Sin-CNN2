# 実験24: 実テキストの確信度 (CE) を改善する（ω 固定の軽量版を基準に）
B="--bias --M 1 --S 1024 --L 2048 --freeze_w"
for args in "--n 2 --dw 16" "--n 2 --dw 32" "--n 2 --steps 8000" "--n 3 --L 4096" "--n 2 --D 32 --P 128 --steps 6000"; do
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py $B $args 2>&1 | tail -1
done
