# リポジトリ直下で実行する: bash experiments/sweeps/sweep40.sh
# 実験40: 漢字を部品（KRADFILE）の和音で表す。比較: 部品なし L=2048 → 44.6（実験35）、大きいモデル L=1024 → 49.1
X="python experiments/exp30_fullaperture.py --task text --pos pair --mix coh --freeze_w --stride --L 2048 --radicals"
PYTHONIOENCODING=utf-8 $X --n 2 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --n 2 --rad_beta 0.5 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --n 2 --V_id 128 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --n 3 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python experiments/exp30_fullaperture.py --task text --pos pair --mix coh --freeze_w --stride --L 2048 --n 3 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --n 2 --D 32 --P 256 --steps 6000 2>&1 | tail -1
