# 実験25: 照明パターンの重み付き平均（逆拡散の重み付け）で「どのスロットの組を干渉させるか」を選ぶ
O="--task order --enc freq --S 128 --L 2048 --P 64 --far --illum tilt"
for args in "--incoh 16 --pat_w adj" "--incoh 16 --pat_w learn"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $O $args 2>&1 | tail -1
done
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far --illum tilt --incoh 8"
for args in "--pat_w adj" "--pat_w learn" "--pat_w adj --blur mean"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $B $args 2>&1 | tail -1
done
T="--bias --n 2 --M 1 --S 512 --L 2048 --freeze_w --illum tilt --incoh 4"
for args in "" "--pat_w adj"; do
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py $T $args 2>&1 | tail -1
done
