"""SinWave の使い方: 2 つの技能を別々に学習し、足すだけで融合し、外して取り消す。

    python examples/quickstart.py            (GPU があれば GPU、なければ CPU。CPU では数分かかる)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

from sinwave import SinWaveClassifier, merge
from sinwave.tasks import task_skills

# 合成データ: 2 技能 × 4 クラス。各クラスは好む文字（ひらがな・カタカナ 4 字）を持つ
skills = task_skills(2)
(xa, ya), (xa_t, ya_t) = skills[0]
(xb, yb), (xb_t, yb_t) = skills[1]

# 技能 A, B を別々に学習する（互いのデータは見ない）。検出器は 8 個のうち A が 0-3、B が 4-7 を担当
a = SinWaveClassifier(classes=[0, 1, 2, 3], slots=[0, 1, 2, 3], n_slots=8, name="A").fit(xa, ya)
b = SinWaveClassifier(classes=[4, 5, 6, 7], slots=[4, 5, 6, 7], n_slots=8, name="B").fit(xb, yb)
print(a, f"学習するパラメータ {a.n_params()} 個")
print("A 単独:", round(a.score(xa_t, ya_t), 3), " B 単独:", round(b.score(xb_t, yb_t), 3))

# 融合: 強度を足すだけ。どちらの技能の入力かも当てる（8 クラス）
m = merge([a, b])
print("融合 (8 クラス):", round(m.score(xa_t + xb_t, ya_t + yb_t), 3))
print("例:", xa_t[0], "→", m.predict([xa_t[0]])[0], "(正解", ya_t[0], ")")

# 取消: B を外すと A 単独と完全に同じに戻る
m.remove("B")
print("B を外した後の A:", round(m.score(xa_t, ya_t), 3))

# 保存と読み込み
a.save("skill_A.pt")
a2 = SinWaveClassifier.load("skill_A.pt")
print("読み込んだ A:", round(a2.score(xa_t, ya_t), 3))
