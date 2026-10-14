# リポジトリ直下で実行する: bash experiments/sweeps/sweep18c.sh
# 実験18c: 平均の滲みはなぜ効くか（順序・実テキスト・半径・照明との組み合わせ）
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
O="--task order --enc freq --S 128 --L 2048 --P 64 --far"
for args in "--blur mean --blur_r 2" "--blur mean --seed 1" "--seed 1"; do
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B $args 2>&1 | tail -1
done
PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $O --blur mean 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B --blur mean --incoh 8 --illum tilt 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python experiments/exp06_nextchar.py --bias --n 2 --M 1 --S 1024 --L 2048 --blur mean 2>&1 | tail -1
