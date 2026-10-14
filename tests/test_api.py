"""高水準 API の動作確認（CPU、小さな設定で数十秒）。  python -m pytest tests -q  または  python tests/test_api.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下の sinwave を使う

import os
import tempfile
import warnings

import torch

from sinwave import SinWaveClassifier, merge
from sinwave.tasks import task_skills

SMALL = dict(L=512, S=128, D=4, P=16)
warnings.filterwarnings("ignore", message=".*文字を超えている")   # 小さな設定では 4 文字で切れる（意図どおり）


def _skill(i, sk, seed=0):
    (xtr, ytr), _ = sk[i]
    classes = sorted(set(ytr))
    clf = SinWaveClassifier(classes, slots=classes, n_slots=8, preset="fast", name=f"s{i}",
                            seed=seed, device="cpu", **SMALL)
    return clf.fit(xtr, ytr, steps=150)


def test_fit_merge_remove_save():
    torch.manual_seed(0)
    sk = task_skills(2, n_train=200, n_test=100)
    a, b = _skill(0, sk), _skill(1, sk)
    (xa, ya), (xb, yb) = sk[0][1], sk[1][1]
    acc_a = a.score(xa, ya)
    assert acc_a > 0.4, acc_a                        # 4 クラスの偶然は 0.25
    m = merge([a, b])
    assert m.classes == a.classes + b.classes
    assert m.score(xa + xb, ya + yb) > 0.3           # 8 クラスの偶然は 0.125
    m.remove("s1")                                   # 取消: 外せば a 単独と完全に同じ
    assert torch.allclose(m.logits(xa), a.logits(xa)[:, a.slots])
    with tempfile.TemporaryDirectory() as d:         # 保存と読み込み
        path = os.path.join(d, "a.pt")
        a.save(path)
        a2 = SinWaveClassifier.load(path, device="cpu")
        assert torch.allclose(a2.logits(xa), a.logits(xa))
    assert merge([a, b], mode="gate").predict(xa[:3])


def test_combine():
    from sinwave import combine
    torch.manual_seed(0)
    sk = task_skills(2, n_train=200, n_test=100)
    (x0, y0), _ = sk[0]
    (x1, y1), (x1t, y1t) = sk[1]
    base = SinWaveClassifier([0, 1], slots=[0, 1], n_slots=8, device="cpu", **SMALL)
    base.fit([x for x, y in zip(x0, y0) if y < 2], [y for y in y0 if y < 2], steps=100)
    a = base.derive([0, 1, 2, 3], slots=[0, 1, 2, 3], name="a").fit(x0, y0, steps=100)
    b = base.derive([4, 5, 6, 7], slots=[4, 5, 6, 7], name="b").fit(x1, y1, steps=100)
    ab = combine(base, [(1, a), (1, b)])
    assert ab.classes == a.classes + b.classes
    same = combine(base, [(1, a)])                     # 係数 1 の 1 項なら a と同じ
    assert torch.allclose(same.logits(x1t[:5]), a.logits(x1t[:5]), atol=1e-4)
    assert ab.n_params() == a.n_params()               # 1 本の光路のまま


def test_checks_and_reproducibility():
    sk = task_skills(1, n_train=100, n_test=50)
    (x, y), _ = sk[0]
    mk = lambda: SinWaveClassifier([0, 1, 2, 3], n_slots=8, device="cpu", seed=3, **SMALL)
    a, b = mk().fit(x, y, steps=30), mk().fit(x, y, steps=30)
    assert torch.equal(a.logits(x[:5]), b.logits(x[:5]))          # 同じ seed なら同じ結果
    for bad in (lambda: SinWaveClassifier([0, 0]), lambda: SinWaveClassifier([0, 1], slots=[0, 0]),
                lambda: mk().fit(x, [9] * len(x)), lambda: mk().fit(["", "あ"], [0, 1]),
                lambda: merge([a, b]), lambda: merge([a]).remove("x")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError になるはず")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a.logits(["あ" * (a.max_chars + 1)])
        assert any("超えている" in str(m.message) for m in w)


if __name__ == "__main__":
    test_fit_merge_remove_save()
    test_combine()
    test_checks_and_reproducibility()
    print("ok")
