# リポジトリ直下で実行する: bash experiments/sweeps/sweep47.sh
# 実験47: 規模を大きくすると伸びるか（実験41 の課題: 頻出 63 字 + その他、文脈 8 文字）。光の記憶 α=0.3 と GRU
X="python experiments/exp41_textrnn.py --alpha 0.3 --steps 6000"
for h in 16 32 64 128 256; do PYTHONIOENCODING=utf-8 $X --model gru --hid $h 2>&1 | tail -1; done
PYTHONIOENCODING=utf-8 $X --D 32 --P 1024 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --D 64 --P 256 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --D 32 --P 256 --L 2048 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python experiments/exp41_textrnn.py --alpha 0.3 --steps 12000 --D 32 --P 256 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --D 64 --P 1024 2>&1 | tail -1
