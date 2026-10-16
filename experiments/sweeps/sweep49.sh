# リポジトリ直下で実行する: bash experiments/sweeps/sweep49.sh
# 実験49: 位置合わせ・板の間隔の誤差に強くする（粗い縞 wmax、ずれ・誤差を入れて学習）
X="python experiments/exp48_analog.py"
R="--train_noise 0.2 --train_shift 1 --train_zerr 0.03"
for seed in 0 1 2; do
  PYTHONIOENCODING=utf-8 $X --seed $seed --wmax 0.4 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --seed $seed $R 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --seed $seed $R --wmax 0.4 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --seed $seed $R --wmax 0.1 2>&1 | tail -1
done
