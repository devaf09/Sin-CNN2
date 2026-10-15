# リポジトリ直下で実行する: bash experiments/sweeps/sweep35.sh
# 実験35: 間引き（L を小さくし、周波数を L と互いに素な歩幅で並べ直す）と、幅（小さな光路を並べる）
X="python experiments/exp30_fullaperture.py"
for L in 1024 512 256; do
  PYTHONIOENCODING=utf-8 $X --task skills4 --pos phase --mix color --stride --L $L 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 $X --task skills4 --pos sphere --mix coh --stride --L $L 2>&1 | tail -1
done
for L in 1024 256; do
  PYTHONIOENCODING=utf-8 $X --task order --pos sphere --mix coh --stride --L $L 2>&1 | tail -1
done
for L in 2048 1024; do
  PYTHONIOENCODING=utf-8 $X --task text --n 2 --pos pair --mix coh --freeze_w --stride --L $L 2>&1 | tail -1
done
# 幅と大きさ: 実テキスト n=2、6000 steps（比較: L=4096 の 1 本、D=32 P=256 → 48.5、実験32）
PYTHONIOENCODING=utf-8 $X --task text --n 2 --pos pair --mix coh --freeze_w --stride --L 1024 --D 32 --P 256 --steps 6000 2>&1 | tail -1
PYTHONIOENCODING=utf-8 $X --task text --n 2 --pos pair --mix coh --freeze_w --stride --L 1024 --D 32 --P 64 --paths 4 --steps 6000 2>&1 | tail -1
