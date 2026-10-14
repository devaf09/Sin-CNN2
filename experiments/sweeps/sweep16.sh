# 実験16: 背景光 → 前 Sin レイヤ（前処理光）→ 入力 → 後 Sin レイヤ
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
for args in "--pre 4" "--pre 8 --D 8" "--pre 16" "--incoh 8 --illum tilt --pre 4" "--incoh 8 --illum tilt --pre 8 --D 8" "--incoh 8 --illum tilt --pre 16"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $B $args 2>&1 | tail -1
done
PYTHONIOENCODING=utf-8 python exp01_learnability.py --task order --enc freq --S 128 --L 2048 --P 64 --far --pre 4 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python exp06_nextchar.py --bias --n 2 --M 1 --S 1024 --L 2048 --pre 4 2>&1 | tail -1
