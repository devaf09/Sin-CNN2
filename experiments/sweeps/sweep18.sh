# リポジトリ直下で実行する: bash experiments/sweeps/sweep18.sh
# 実験18: 滲み層（前後の文字とのばらつき）と、入力のフーリエ変換
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
for args in "--blur std" "--blur ustd" "--blur mean" "--blur std --blur_r 2" "--blur std --incoh 8 --illum tilt" \
            "--norepeat --input_ft" "--input_ft" "--norepeat --input_ft --enc code" "--norepeat --enc code"; do
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B $args 2>&1 | tail -1
done
O="--task order --enc freq --S 128 --L 2048 --P 64 --far"
for args in "--blur std" "--norepeat --input_ft"; do
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $O $args 2>&1 | tail -1
done
for args in "--blur std" "--blur ustd" "--norepeat --input_ft"; do
  PYTHONIOENCODING=utf-8 python experiments/exp06_nextchar.py --bias --n 2 --M 1 --S 1024 --L 2048 $args 2>&1 | tail -1
done
