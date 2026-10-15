# リポジトリ直下で実行する: bash experiments/sweeps/sweep32.sh
# 実験32: 実験30 の確認（seed 1,2）、レイヤの焦点の強さ（不採用案の再検討）、実テキストでの容量
X="python experiments/exp30_fullaperture.py"
for seed in 1 2; do
  PYTHONIOENCODING=utf-8 $X --task skills4 --pos phase --mix color --seed $seed 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --task skills4 --pos sphere --mix coh --seed $seed 2>&1 | tail -1
done
PYTHONIOENCODING=utf-8 $X --task skills4 --pos sphere --mix coh --focus 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --task order --pos sphere --mix coh --focus 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --task text --n 8 --pos sphere --mix adj --freeze_w --focus 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --task text --n 2 --pos pair --mix coh --freeze_w --D 32 --P 256 --steps 6000 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --task text --n 3 --pos sphere --mix adj --freeze_w --D 32 --P 256 --steps 6000 2>&1 | tail -1
