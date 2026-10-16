# リポジトリ直下で実行する: bash experiments/sweeps/sweep50.sh
# 実験50: 訓練データを増やすと、光の記憶と GRU のどちらが伸びるか（実験47 は 6 万件。全部で約 53 万件）
X="python experiments/exp41_textrnn.py --alpha 0.3 --steps 6000"
for N in 200000 530000; do
  for h in 32 64 128 256; do PYTHONIOENCODING=utf-8 $X --model gru --hid $h --max_train $N 2>&1 | tail -1; done
done
PYTHONIOENCODING=utf-8 $X --D 32 --P 1024 --max_train 530000 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --D 64 --P 1024 --max_train 530000 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --D 64 --P 1024 --max_train 200000 2>&1 | tail -1
