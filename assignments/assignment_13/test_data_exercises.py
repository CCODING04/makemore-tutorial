#!/usr/bin/env python3
"""Part 13 作业测试。独立运行：python test_data_exercises.py；或 pytest。"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_exercises import *  # noqa: F401,F403


def test_ex1_jaccard():
    assert shingles is not None, "shingles 未实现"
    s = shingles("The quick brown fox!", k=2)
    assert s is not None and isinstance(s, (set, frozenset)) and len(s) == 3
    assert "quick brown" in s and "brown fox" in s, f"2-gram 集合: {s}"
    assert jaccard is not None
    assert abs(jaccard({"a", "b"}, {"a", "b"}) - 1.0) < 1e-12
    assert abs(jaccard({"a"}, {"b"})) < 1e-12
    assert abs(jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-12


def test_ex2_signature():
    assert signature_agreement is not None, "signature_agreement 未实现"
    assert signature_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert signature_agreement([1, 2, 3], [9, 9, 9]) == 0.0
    assert abs(signature_agreement([1, 2, 3, 4], [1, 9, 3, 9]) - 0.5) < 1e-12


def test_ex3_lsh_prob():
    assert lsh_hit_probability is not None, "lsh_hit_probability 未实现"
    # J=1 → 必命中；J=0 → 概率 0
    assert abs(lsh_hit_probability(1.0, 16, 4) - 1.0) < 1e-12
    assert lsh_hit_probability(0.0, 16, 4) == 0.0
    # 手算：J=0.5, r=2, b=3 → 1-(1-0.25)^3 = 1-0.421875
    assert abs(lsh_hit_probability(0.5, 3, 2) - (1 - 0.75 ** 3)) < 1e-12
    assert choose_bands_for_recall is not None
    assert choose_bands_for_recall(1.0, 4, 0.99) == 1
    b = choose_bands_for_recall(0.9, 4, 0.99)
    assert b is not None and lsh_hit_probability(0.9, b, 4) >= 0.99
    assert b == 1 or lsh_hit_probability(0.9, b - 1, 4) < 0.99, "应返回【最小】b"


def test_ex4_cluster():
    assert keep_first_per_cluster is not None, "keep_first_per_cluster 未实现"
    kept, dropped = keep_first_per_cluster(
        ["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert (kept, dropped) == (["a", "d"], ["b", "c"]), f"传递闭包: got {kept},{dropped}"
    kept, dropped = keep_first_per_cluster(["x", "a", "b"], [("a", "x")])
    assert (kept, dropped) == (["x", "b"], ["a"]), "保留原顺序最前的"


_TESTS = [("题1 Jaccard/shingles", test_ex1_jaccard), ("题2 签名一致率", test_ex2_signature),
          ("题3 LSH 概率", test_ex3_lsh_prob), ("题4 簇消解", test_ex4_cluster)]


def main():
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 data_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
