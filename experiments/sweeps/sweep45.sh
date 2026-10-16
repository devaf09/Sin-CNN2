# リポジトリ直下で実行する: bash experiments/sweeps/sweep45.sh
# 実験45: 単位を部品の組で表し、答えの窓を 3096 → 112 個に減らす。物差しは 1 文字あたりの CE（全語彙）
X="python experiments/exp45_unitparts.py"
PYTHONIOENCODING=utf-8 $X --assign ctx --input parts --L 1024 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --assign ctx --input unit --L 4096 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --assign random --input parts --L 1024 2>&1 | tail -1
