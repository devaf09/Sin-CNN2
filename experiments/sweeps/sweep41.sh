# リポジトリ直下で実行する: bash experiments/sweeps/sweep41.sh
# 実験41: 光の記憶（前の出力の光を混ぜる）で実テキストの次の文字を予測。文脈が伸びると良くなるか
for al in 0 0.3 0.5 0.7; do
  PYTHONIOENCODING=utf-8 python experiments/exp41_textrnn.py --n 8 --alpha $al 2>&1 | tail -1
done
