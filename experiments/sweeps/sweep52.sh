# リポジトリ直下で実行する: bash experiments/sweeps/sweep52.sh
# 実験52: うなり読みで実テキストの次の文字（文脈 8 文字を虹で一度に入れる）。小さいモデル D16 P64 で比較 → 大きいモデル
X="python experiments/exp52_beat_text.py"
PYTHONIOENCODING=utf-8 $X --read beat 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read mean --rb 0.5 --T 8 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read mean --rb 0 --T 1 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read beat --n 4 --dmax 3 --T 10 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read beat --D 32 --P 256 --steps 6000 2>&1 | tail -1
