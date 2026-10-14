# 実験18b: 滲みの別方式（因果的なばらつき、差分、元の入力と滲みの2色重ね）
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
O="--task order --enc freq --S 128 --L 2048 --P 64 --far"
for args in "--blur cstd" "--blur diff" "--blur std --blur_mix --M 2"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $B $args 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $O $args 2>&1 | tail -1
done
for args in "--M 1 --blur cstd" "--M 1 --blur diff" "--M 2 --blur std --blur_mix" "--M 2 --blur cstd --blur_mix"; do
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py --bias --n 2 --S 1024 --L 2048 $args 2>&1 | tail -1
done
