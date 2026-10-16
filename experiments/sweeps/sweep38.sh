# リポジトリ直下で実行する: bash experiments/sweeps/sweep38.sh
# 実験38: 出力の光をそのまま入力へ戻す（ピンポン、記憶）
for K in 2 5 10; do
  PYTHONIOENCODING=utf-8 python experiments/exp38_feedback.py --part pingpong --K $K 2>&1 | tail -1
done
for al in 0 0.3 0.5 0.7; do
  PYTHONIOENCODING=utf-8 python experiments/exp38_feedback.py --part memory --alpha $al 2>&1 | tail -1
done
