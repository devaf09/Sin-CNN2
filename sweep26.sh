# 実験26: 出力面全体を周波数として読む（縞模様の感度 fcos / 明るさの模様の振幅 fabs）
B="--task skills4 --enc freq --S 256 --L 2048 --P 64 --far"
for args in "--readout fcos" "--readout fabs" "--readout fabs --dk 8" "--blur mean --readout fcos" "--blur mean --readout fabs"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $B $args 2>&1 | tail -1
done
T="--bias --n 2 --M 1 --S 1024 --L 2048 --freeze_w"
for args in "--readout fcos" "--readout fabs" "--readout fabs --dk 4"; do
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py $T $args 2>&1 | tail -1
done
