# リポジトリ直下で実行する: bash experiments/sweeps/sweep33c.sh
# 実験33c: 文字列 X → 文字列 Y（言い回し帳の容量、規則の汎化）
P="python experiments/exp33c_phrase.py"
for n in 16 64 256 1024; do for e in atom pair; do
  PYTHONIOENCODING=utf-8 $P --part phrase --enc $e --pairs $n 2>&1 | tail -1
done; done
for r in copy map reverse; do
  PYTHONIOENCODING=utf-8 $P --part rule --rule $r 2>&1 | tail -1
done
