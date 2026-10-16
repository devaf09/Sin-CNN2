# リポジトリ直下で実行する: bash experiments/sweeps/sweep44.sh
# 実験44: 答えが同じ入力を重ねて 1 回の計算で学習する。評価は 1 件ずつ
X="python experiments/exp44_superpose.py"
for seed in 0 1 2; do
  for st in 100 300 1000; do
    for m in "single" "coh --k 2" "coh --k 4" "coh --k 8" "rphase --k 2" "rphase --k 4" "rphase --k 8" \
             "coh --k 4 --fixed" "rphase --k 4 --fixed" "incoh --k 4"; do
      PYTHONIOENCODING=utf-8 $X --mode $m --steps $st --seed $seed 2>&1 | tail -1
    done
  done
done
