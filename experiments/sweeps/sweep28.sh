# リポジトリ直下で実行する: bash experiments/sweeps/sweep28.sh
# 実験28: 典型の模様で判定し、新しいクラスを板の学習なしで追加する
for seed in 0 1 2; do
  for args in "--steps 1500" "--steps 0" "--steps 1500 --patterns 8" "--steps 0 --patterns 8" "--steps 300"; do
    PYTHONIOENCODING=utf-8 python experiments/exp28_proto.py --seed $seed $args 2>&1 | tail -1
  done
done
