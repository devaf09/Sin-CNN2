# リポジトリ直下で実行する: bash experiments/sweeps/sweep13.sh
# 実験13: 実テキスト。周波数の数 S を増やして符号化の天井を上げる（n=2、M=1、背景光あり）
for args in "--S 1024 --L 2048" "--S 1024 --L 4096" "--S 1024 --L 4096 --incoh 2 --illum tilt" "--S 512 --L 2048" "--S 1024 --L 4096 --D 32 --P 128"; do
  PYTHONIOENCODING=utf-8 python experiments/exp06_nextchar.py --bias --n 2 --M 1 $args 2>&1 | tail -1
done
