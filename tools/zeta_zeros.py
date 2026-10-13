"""リーマンゼータ関数の非自明なゼロ点 1/2 + i t_n の高さ t_n を計算して保存する（mpmath）。"""
import sys

import mpmath

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
with open("sinwave/zeta_zeros.txt", "w") as f:
    for k in range(1, n + 1):
        f.write(f"{float(mpmath.zetazero(k).imag):.9f}\n")
print("saved", n)
