# リポジトリ直下で実行する: bash experiments/sweeps/sweep08.sh
# 実験08: 非コヒーレント光源
for t in skills4 topic order; do
  S=256; [ $t != skills4 ] && S=128
  for args in "" "--incoh 8" "--incoh 32"; do
    PYTHONIOENCODING=utf-8 python experiments/exp01_learnability.py --task $t --enc freq --far --S $S --L 2048 --P 64 $args 2>&1 | tail -1
  done
done
