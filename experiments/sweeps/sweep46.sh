# リポジトリ直下で実行する: bash experiments/sweeps/sweep46.sh
# 実験46: 部品の組の b にも意味を持たせる（左右の文脈）。実験45 と同じ条件
X="python experiments/exp45_unitparts.py"
for m in left lr rl; do
  PYTHONIOENCODING=utf-8 $X --assign $m --input parts --L 1024 2>&1 | tail -1
done
