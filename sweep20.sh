# 実験20: 平均の滲み層で並列融合（3 seed）、平均 + tilt (seed 0)
for seed in 0 1 2; do for n in 2 4; do
  PYTHONIOENCODING=utf-8 python exp05_scaling.py --n $n --joint --blur mean --seed $seed 2>&1 | tail -1
done; done
for n in 2 4; do
  PYTHONIOENCODING=utf-8 python exp05_scaling.py --n $n --joint --blur mean --incoh 8 --illum tilt --seed 0 2>&1 | tail -1
done
