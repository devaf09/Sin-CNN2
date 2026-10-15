# リポジトリ直下で実行する: bash experiments/sweeps/sweep37.sh
# 実験37: 取り出しの符号を学習して、位置をまたぐ出力（ずらし・逆順）を作る
for t in copy shift reverse; do
  PYTHONIOENCODING=utf-8 python experiments/exp33b_seqout.py --task $t --read mix 2>&1 | tail -1
done
PYTHONIOENCODING=utf-8 python experiments/exp33b_seqout.py --task reverse --read shared 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python experiments/exp33b_seqout.py --task reverse --read code 2>&1 | tail -1
