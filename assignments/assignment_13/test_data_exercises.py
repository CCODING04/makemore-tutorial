#!/usr/bin/env python3
"""Part 13 作业测试。独立运行：python test_data_exercises.py；或 pytest。"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_exercises import *  # noqa: F401,F403


def test_ex1_jaccard():
    s = shingles("The quick brown fox!", k=2)
    assert s is not None, "shingles 未实现"
    assert isinstance(s, (set, frozenset)) and len(s) == 3, f"2-gram 集合: {s}"
    assert "quick brown" in s and "brown fox" in s, f"2-gram 集合: {s}"
    j = jaccard({"a", "b"}, {"a", "b"})
    assert j is not None, "jaccard 未实现"
    assert abs(j - 1.0) < 1e-12
    assert abs(jaccard({"a"}, {"b"})) < 1e-12
    assert abs(jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-12


def test_ex2_signature():
    r = signature_agreement([1, 2, 3], [1, 2, 3])
    assert r is not None, "signature_agreement 未实现"
    assert r == 1.0
    assert signature_agreement([1, 2, 3], [9, 9, 9]) == 0.0
    assert abs(signature_agreement([1, 2, 3, 4], [1, 9, 3, 9]) - 0.5) < 1e-12


def test_ex3_lsh_prob():
    p = lsh_hit_probability(1.0, 16, 4)
    assert p is not None, "lsh_hit_probability 未实现"
    # J=1 → 必命中；J=0 → 概率 0
    assert abs(p - 1.0) < 1e-12
    assert lsh_hit_probability(0.0, 16, 4) == 0.0
    # 手算：J=0.5, r=2, b=3 → 1-(1-0.25)^3 = 1-0.421875
    assert abs(lsh_hit_probability(0.5, 3, 2) - (1 - 0.75 ** 3)) < 1e-12
    b = choose_bands_for_recall(1.0, 4, 0.99)
    assert b is not None, "choose_bands_for_recall 未实现"
    assert b == 1
    b = choose_bands_for_recall(0.9, 4, 0.99)
    assert lsh_hit_probability(0.9, b, 4) >= 0.99
    assert b == 1 or lsh_hit_probability(0.9, b - 1, 4) < 0.99, "应返回【最小】b"


def test_ex4_cluster():
    r = keep_first_per_cluster(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert r is not None, "keep_first_per_cluster 未实现"
    kept, dropped = r
    assert (kept, dropped) == (["a", "d"], ["b", "c"]), f"传递闭包: got {kept},{dropped}"
    kept, dropped = keep_first_per_cluster(["x", "a", "b"], [("a", "x")])
    assert (kept, dropped) == (["x", "b"], ["a"]), "保留原顺序最前的"


def test_ex5_stretch():
    """🌟 Stretch：未实现（返回 None）时优雅 SKIP，不判 FAIL。"""
    result = keep_best_per_cluster(
        ["a", "b", "c", "d"], {"a": 10, "b": 5, "c": 20, "d": 1},
        [("a", "b"), ("b", "c")])
    if result is None:  # 未实现 → SKIP（pytest 下 pytest.skip，独立运行下返回标记）
        if "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip("🌟 Stretch 未实现（返回 None）——实现后此测试自动生效")
        return "SKIP"
    # 传递闭包 + 每簇保留最长：簇 {a,b,c} 长度 c=20 最长 → 保留 c；d 是单例
    kept, dropped = result
    assert kept == ["c", "d"] and dropped == ["a", "b"], \
        f"簇{{a,b,c}} 应保留最长的 c: got kept={kept}, dropped={dropped}"
    # 并列长度 → 保留原顺序最先（x 在 a 之前，长度同为 8）
    kept, dropped = keep_best_per_cluster(
        ["x", "a", "b"], {"x": 8, "a": 8, "b": 3}, [("a", "x")])
    assert (kept, dropped) == (["x", "b"], ["a"]), "并列长度保留原顺序最前的 x"
    # 无重复对 → 全部保留
    kept, dropped = keep_best_per_cluster(["p", "q"], {"p": 1, "q": 2}, [])
    assert (kept, dropped) == (["p", "q"], []), "无重复对应全保留"


_TESTS = [("题1 Jaccard/shingles", test_ex1_jaccard), ("题2 签名一致率", test_ex2_signature),
          ("题3 LSH 概率", test_ex3_lsh_prob), ("题4 簇消解", test_ex4_cluster),
          ("题5 🌟 Stretch 簇消解(keep-longest)", test_ex5_stretch)]


def main():
    p = f = s = 0
    for name, fn in _TESTS:
        try:
            r = fn()
            if r == "SKIP":
                print(f"  ⏭️  {name} — SKIP（未实现，不扣分）"); s += 1
            else:
                print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    msg = f"\n  通过: {p}/{p + f}" + (f"（另 {s} 项 SKIP ⏭️）" if s else "")
    print(msg + ("  🎉" if f == 0 else "  💡 先实现 data_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
