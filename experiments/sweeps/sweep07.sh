# リポジトリ直下で実行する: bash experiments/sweeps/sweep07.sh
# 実験07: 16クラス課題 (skills4) で表現力の天井を探る
B="--task skills4 --enc freq --far --S 256 --L 2048 --P 64"
for args in "" "--steps 4000" "--lr 0.03" "--lr 0.003 --steps 4000" "--z 16" "--z 256" "--z 1024" "--D 32" "--P 256" "--dw 16" "--K 3" "--mode amp"; do
  PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py $B $args 2>&1 | tail -1
done
