# 実験12: tilt 照明で技能融合と実テキストをやり直す
for n in 2 4; do PYTHONIOENCODING=utf-8 python exp05_scaling.py --n $n --joint --incoh 8 --illum tilt 2>&1 | tail -1; done
for args in "--n 3" "--n 2" "--n 3 --D 32 --P 128"; do
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py --parallel --bias --incoh 8 --illum tilt $args 2>&1 | tail -1
done
