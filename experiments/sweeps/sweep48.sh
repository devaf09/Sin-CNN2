# リポジトリ直下で実行する: bash experiments/sweeps/sweep48.sh
# 実験48: アナログな装置で作ったときの強さ。理想の装置で学習 / 雑音を入れて学習（σ=0.2）
for seed in 0 1 2; do
  for tn in 0 0.2; do
    PYTHONIOENCODING=utf-8 python experiments/exp48_analog.py --seed $seed --train_noise $tn 2>&1 | tail -1
  done
done
