# リポジトリ直下で実行する: bash experiments/sweeps/sweep42.sh
# 実験42: 記憶が早く薄まる対策（比較: 混ぜるだけ α=0.3 → 文脈 8 文字で 45.4%、実験41）
X="python experiments/exp41_textrnn.py --n 8"
for args in "--mem shift --shift 7 --alpha 0.3" "--mem shift --shift 7 --alpha 0.5" "--mem shift --shift 1 --alpha 0.5" "--mem band --alpha 0.3" "--mem band --alpha 0.5"; do
  PYTHONIOENCODING=utf-8 $X $args 2>&1 | tail -1
done
