"""sinwave_minimal.py の確認: 実験で使ってきた本体 (sinwave/model.py) と同じ出力になるか、融合・取消・保存。
python tests/test_minimal.py  または  python -m pytest tests -q"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # リポジトリ直下を使う

import torch

import sinwave_minimal as sw
from sinwave.model import Config, SinWave, encode

TEXTS = ["こんにちは", "あいうえおかき", "カタカナ", "ん"]


def _reference(patterns, seed):
    """実験コード側で同じ構成を組む（ω 固定は値が同じなので関係しない）。"""
    cfg = Config(L=512, C=8, far=True, bias=True, slot=64, blur="mean", incoh=patterns, illum="tilt")
    m = SinWave(cfg)
    m.add_layers(4, 1, "phase", seed=seed, P=16)
    return m


def test_same_as_reference():
    for patterns in (0, 8):
        net = sw.SinWaveNet(n_slots=8, L=512, S=64, layers=4, patches=16, patterns=patterns, seed=3)
        ref = _reference(patterns, 3)
        with torch.no_grad():
            net.bias.copy_(torch.randn(8))
            ref.bias.copy_(net.bias)
        for d, layer in enumerate(ref.layers):                     # 初期値が一致すること
            assert torch.allclose(layer.A[:, 0], net.A[d]) and torch.allclose(layer.w[:, 0], net.w[d])
        x_min = sw.encode(TEXTS, 512, 64)
        x_ref = encode(TEXTS, 512, 1, "cpu", 64, "freq")[:, 0]
        assert torch.allclose(x_min, x_ref, atol=1e-6)               # 符号化が一致
        # 出力が一致（計算の順序の違いで単精度の丸め誤差が出るので、相対誤差で比べる）
    assert torch.allclose(net(x_min), ref(x_ref[:, None]), rtol=1e-4, atol=1e-3), patterns


def test_merge_remove_combine_save():
    a = sw.SinWaveNet(n_slots=4, own=[0, 1], L=512, S=64, layers=4, patches=16)
    b = sw.SinWaveNet(n_slots=4, own=[2, 3], L=512, S=64, layers=4, patches=16, seed=1)
    sw.train(a, ["あいう", "かきく"] * 20, [0, 1] * 20, steps=30)
    sw.train(b, ["さしす", "たちつ"] * 20, [2, 3] * 20, steps=30)
    m = sw.merge([a, b])
    assert set(sw.predict(m, TEXTS)) <= {0, 1, 2, 3}
    assert torch.equal(sw.scores(sw.merge([a]), TEXTS), sw.scores(a, TEXTS))   # 取消 = 元どおり
    a2 = sw.train(sw.derive(a, [0, 1, 2]), ["あいう", "かきく", "さしす"] * 20, [0, 1, 2] * 20, steps=30)
    c = sw.combine(a, [(1, a2)])                                     # 係数 1 の 1 項なら a2 と同じ光路
    x = sw.encode(TEXTS, 512, 64)
    assert torch.allclose(c(x), a2(x), atol=1e-5) and c.own == [0, 1, 2]
    try:
        sw.combine(a, [(1, b)])                                      # 共通の出発点でないものは拒む
    except ValueError:
        pass
    else:
        raise AssertionError("ω が違う光路の足し引きはエラーになるはず")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.pt")
        sw.save(a, path)
        assert torch.equal(sw.scores(sw.load(path), TEXTS), sw.scores(a, TEXTS))
    try:
        sw.merge([a, a])
    except ValueError:
        pass
    else:
        raise AssertionError("担当が重なる融合はエラーになるはず")


def test_wave_and_roots():
    """v1.1: 文字をレイヤ全体に照らす入力（虹・色ごと・全部干渉）と、語根の検出器。"""
    texts = ["あいう", "かきく", "あかさ", "いきし"] * 10
    for mix, T in (("rainbow", 8), ("color", 6), ("coh", 1)):
        net = sw.SinWaveNet(4, [0, 1], L=512, layers=2, patches=16, input="wave", mix=mix, max_chars=6)
        assert tuple(net.encode(texts[:2]).shape) == (2, T, 512)
        sw.train(net, texts, [0, 1] * 20, steps=5)
    a = sw.SinWaveNet(4, [0, 1], L=512, layers=2, patches=16, input="wave")
    b = sw.SinWaveNet(4, [2, 3], L=512, layers=2, patches=16, input="wave", seed=1)
    m = sw.merge([a, b])                                             # wave どうしの融合
    assert set(sw.predict(m, texts[:4])) <= {0, 1, 2, 3}
    try:
        sw.merge([a, sw.SinWaveNet(4, [2, 3], L=512, layers=2, patches=16)])   # slot と wave は混ぜられない
    except ValueError:
        pass
    else:
        raise AssertionError("入力の設定が違う融合はエラーになるはず")
    r = sw.train_roots(sw.SinWaveNet(3, L=512, layers=2, patches=16, input="wave"), texts, [[0], [1], [0, 2], [1, 2]] * 10, steps=5)
    out = sw.compose(r, texts[:4], {"x": [0], "y": [1], "新": [0, 1]}, seen=["x", "y"], gamma=1.0)
    assert len(out) == 4 and set(out) <= {"x", "y", "新"}
    with tempfile.TemporaryDirectory() as d:                         # 部品の出やすさも保存される
        path = os.path.join(d, "r.pt")
        sw.save(r, path)
        assert torch.equal(sw.load(path).root_prior, r.root_prior)


if __name__ == "__main__":
    test_same_as_reference()
    test_merge_remove_combine_save()
    test_wave_and_roots()
    print("ok")
