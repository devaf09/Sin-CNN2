# 実験10b: 差動検出・層ごとのレンズ
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
for args in "--diff" "--lens_every 1" "--lens_every 2" "--lens_every 4" "--diff --lens_every 2"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $B $args 2>&1 | tail -1
done
