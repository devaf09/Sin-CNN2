# リポジトリ直下で実行する: bash experiments/sweeps/sweep43.sh
# 実験43: よく出る文字列を単位にまとめる × 光の記憶。物差しは 1 文字あたりの CE（全語彙）
X="python experiments/exp43_tokens.py"
PYTHONIOENCODING=utf-8 $X --model ngram 2>&1 | tail -1
for K in 0 1024; do
  PYTHONIOENCODING=utf-8 $X --model gru --K $K 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --model optical --K $K --L 4096 2>&1 | tail -1
done
PYTHONIOENCODING=utf-8 $X --model gru --K 1024 --emb 64 --hid 128 2>&1 | tail -1
