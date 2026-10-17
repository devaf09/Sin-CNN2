# リポジトリ直下で実行する: bash experiments/sweeps/sweep51.sh
# 実験51: うなり読み。比較 = 今の虹（rb 0.5, T 8, 平均）、同じ計算量の平均（rb 1, T 20）、全部干渉（coh: rb 0, T 1）
X="python experiments/exp51_beat.py"
for task in dist skills4; do
  for seed in 0 1 2; do
    PYTHONIOENCODING=utf-8 $X --task $task --seed $seed --read beat 2>&1 | tail -1
    PYTHONIOENCODING=utf-8 $X --task $task --seed $seed --read mean --rb 1 --T 20 2>&1 | tail -1
    PYTHONIOENCODING=utf-8 $X --task $task --seed $seed --read mean --rb 0.5 --T 8 2>&1 | tail -1
    PYTHONIOENCODING=utf-8 $X --task $task --seed $seed --read mean --rb 0 --T 1 2>&1 | tail -1
  done
done
