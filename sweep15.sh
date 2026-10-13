# 実験15: 入力の繰り返しは必要か（同じ L で、繰り返す / 1回だけ置いて残りは暗く）
for rep in "" "--norepeat"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py --task skills4 --enc freq --S 256 --L 2048 --P 64 --far --incoh 8 --illum tilt $rep 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 python exp01_learnability.py --task skills4 --enc freq --S 256 --L 2048 --P 64 --far $rep 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 python exp01_learnability.py --task order --enc freq --S 128 --L 2048 --P 64 --far $rep 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py --bias --n 2 --M 1 --S 1024 --L 4096 $rep 2>&1 | tail -1
done
