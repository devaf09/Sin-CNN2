# 実験10: 光学的な改良の候補を同じ条件 (skills4, D=16, P=64, seed 0) で比べる
B="--task skills4 --enc freq --S 256 --L 2048 --P 64"
for args in "--far" "--far --z_sched geom" "--far --z_sched learn" "--far --feedback field" "--far --feedback phase" \
            "--out_pad 4 --z_out 2048" "--out_pad 4 --z_out 4096" "--far --out_pad 2 --z_out 1024" "--far --detlog" \
            "--far --incoh 8 --illum random" "--far --incoh 8 --illum tilt" "--far --incoh 8 --illum hadamard" "--far --incoh 4 --illum tilt"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py $B $args 2>&1 | tail -1
done
# 順序課題で、完全に干渉を消す照明が順序の情報を壊さないか (S=128 → 16 スロット)
for args in "--incoh 8 --illum tilt" "--incoh 16 --illum tilt"; do
  PYTHONIOENCODING=utf-8 python exp01_learnability.py --task order --enc freq --S 128 --L 2048 --P 64 --far $args 2>&1 | tail -1
done
