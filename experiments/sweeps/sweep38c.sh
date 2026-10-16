# リポジトリ直下で実行する: bash experiments/sweeps/sweep38c.sh
# 実験38c: 出力の周波数成分をそろえる学習（fid）と、戻す前の補正（空間フィルタ・強調する増幅）
X="python experiments/exp38_feedback.py"
for fid in 0 1 5; do for K in 2 5 10; do
  PYTHONIOENCODING=utf-8 $X --part pingpong --K $K --fid $fid 2>&1 | tail -1
done; done
for fb in filter sharpen; do for al in 0.5 0.7; do
  PYTHONIOENCODING=utf-8 $X --part memory --alpha $al --fb $fb --fid 1 2>&1 | tail -1
done; done
