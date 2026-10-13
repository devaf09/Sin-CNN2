"""合成タスク。外部データを使わずに「何が学べるか」を切り分ける。"""
from __future__ import annotations

import random

HIRA = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")

WORDS = ["こんにちは", "こんばんは", "おはよう", "ありがとう", "さようなら", "すみません", "いただきます",
         "ごちそうさま", "おやすみ", "はじめまして", "よろしく", "おめでとう", "いってきます", "ただいま",
         "おかえり", "ごめんなさい", "だいじょうぶ", "がんばって", "おつかれさま", "いらっしゃい",
         "もしもし", "どうぞ", "どうも", "またね", "げんき", "さくら", "やま", "かわ", "うみ", "そら",
         "ほし", "つき"]


def task_words(n: int = 32):
    """T1: 単語の識別（暗記）。n クラス、各1サンプル。"""
    w = WORDS[:n]
    return (w, list(range(n))), (w, list(range(n)))


def task_topic(C: int = 8, n_train: int = 2000, n_test: int = 1000, seed: int = 0, p_key: float = 0.6):
    """T2: 文字分布による分類（汎化）。クラスごとに好む文字集合があり、確率 p_key でそこから引く。
    文字の並び順は無関係 → 文字の出現頻度だけで解ける。"""
    rng = random.Random(seed)
    pool = HIRA[:]
    rng.shuffle(pool)
    keys = [pool[i * 5:(i + 1) * 5] for i in range(C)]

    def sample(r):
        y = r.randrange(C)
        n = r.randint(4, 10)
        s = "".join(r.choice(keys[y]) if r.random() < p_key else r.choice(HIRA) for _ in range(n))
        return s, y

    r1, r2 = random.Random(seed + 1), random.Random(seed + 2)
    tr = [sample(r1) for _ in range(n_train)]
    te = [sample(r2) for _ in range(n_test)]
    return tuple(map(list, zip(*tr))), tuple(map(list, zip(*te)))


def task_order(n_train: int = 2000, n_test: int = 1000, seed: int = 0):
    """T3: 順序依存。文字列に あ と い が1回ずつ入る。い が あ の直後（巡回）なら 1。
    文字の出現頻度では解けない（両クラスで同じ）。"""
    others = HIRA[2:12]

    def sample(r):
        n = r.randint(4, 8)
        y = r.randrange(2)
        body = [r.choice(others) for _ in range(n - 2)]
        if y == 1:
            i = r.randrange(n - 1)
            body[i:i] = ["あ", "い"]
        else:
            while True:
                s = body[:]
                i, j = sorted(r.sample(range(n), 2))
                s.insert(i, "あ" if r.random() < 0.5 else "い")
                s.insert(j, "い" if s[i] == "あ" else "あ")
                txt = "".join(s)
                if "あい" not in txt + txt[0]:
                    body = s
                    break
            return "".join(body), 0
        return "".join(body), 1

    r1, r2 = random.Random(seed + 1), random.Random(seed + 2)
    tr = [sample(r1) for _ in range(n_train)]
    te = [sample(r2) for _ in range(n_test)]
    return tuple(map(list, zip(*tr))), tuple(map(list, zip(*te)))


KATA = list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")


def task_skills(n_skills: int, per: int = 4, n_keys: int = 4, n_train: int = 500, n_test: int = 250,
                seed: int = 0, p_key: float = 0.6):
    """T2 を n_skills 個の技能に分けた版。技能 i はクラス i*per .. i*per+per-1 を持つ。
    キー文字はひらがな+カタカナから重ならないように割り当てる。
    戻り値: skills[i] = ((xtr, ytr), (xte, yte))（ラベルは全体通し番号）"""
    rng = random.Random(seed)
    pool = HIRA + KATA
    rng.shuffle(pool)
    C = n_skills * per
    assert C * n_keys <= len(pool)
    keys = [pool[i * n_keys:(i + 1) * n_keys] for i in range(C)]

    def sample(r, y):
        n = r.randint(4, 10)
        return "".join(r.choice(keys[y]) if r.random() < p_key else r.choice(pool) for _ in range(n))

    skills = []
    for i in range(n_skills):
        r1, r2 = random.Random(seed * 1000 + i * 2 + 1), random.Random(seed * 1000 + i * 2 + 2)
        tr = [(sample(r1, y), y) for y in (r1.randrange(i * per, (i + 1) * per) for _ in range(n_train * per // 4))]
        te = [(sample(r2, y), y) for y in (r2.randrange(i * per, (i + 1) * per) for _ in range(n_test * per // 4))]
        skills.append((tuple(map(list, zip(*tr))), tuple(map(list, zip(*te)))))
    return skills
