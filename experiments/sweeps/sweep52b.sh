# リポジトリ直下で実行する: bash experiments/sweeps/sweep52b.sh
# 実験52b: 直前の文字ほど明るく入れる（D16 P64）
X="python experiments/exp52_beat_text.py"
PYTHONIOENCODING=utf-8 $X --read beat --decay 0.5 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read beat --decay 0.5 --learn_amp 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read mean --rb 0.5 --T 8 --decay 0.5 --learn_amp 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read beat --n 2 --dmax 1 --T 4 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --read mean --n 2 --rb 0 --T 1 2>&1 | tail -1
