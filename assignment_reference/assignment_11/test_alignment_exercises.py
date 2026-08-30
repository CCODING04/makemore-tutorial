#!/usr/bin/env python3
"""Part 11 作业测试。独立运行：python test_alignment_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_exercises import *  # noqa: F401,F403


def test_ex1_reward():
    assert math_reward is not None, "math_reward 未实现"
    cases = [
        ("\\\\boxed{42}", "42", 1.0), ("\\\\boxed{ 42 }", "42", 1.0),
        ("#### 3.5", "3.5", 1.0), ("The answer is 100, no wait, 7.", "7", 1.0),
        ("\\\\boxed{41}", "42", 0.0), ("I don't know", "42", 0.0),
        ("There are 1,234 ways.", "1234", 1.0), ("value is 42.", "42", 1.0),
    ]
    for resp, gt, want in cases:
        got = math_reward(resp, gt)
        assert got is not None and abs(got - want) < 1e-9, f"{resp!r} gt={gt}: {got} != {want}"


def test_ex2_group():
    assert group_advantages is not None, "group_advantages 未实现"
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert adv is not None and abs(sum(adv)) < 1e-6
    assert abs(adv[0] - 1.0) < 1e-6 and abs(adv[1] + 1.0) < 1e-6
    # 全同组 → 全 0 而非 NaN
    adv_same = group_advantages([1.0, 1.0, 1.0, 1.0])
    assert all(abs(a) < 1e-6 for a in adv_same), "全同组应 eps 兜底为全 0"
    assert zero_gradient_groups is not None and zero_gradient_groups(
        [[1.0, 1.0], [0.0, 1.0], [2.0, 2.0]]) == [0, 2]


def test_ex3_kl():
    assert k3_kl is not None, "k3_kl 未实现"
    v = k3_kl([math.log(0.5)], [math.log(0.5)])
    assert abs(v) < 1e-9, "同分布 KL 应为 0"
    v2 = k3_kl([math.log(0.4), math.log(0.6)], [math.log(0.5), math.log(0.5)])
    assert v2 > 0, "KL 应恒非负"
    assert kl_budget_ok is not None
    assert kl_budget_ok([math.log(0.5)], [math.log(0.5)]) is True
    assert kl_budget_ok([0.0, 0.0], [math.log(2.0), 0.0], budget=0.05) is False


_TESTS = [("题1 奖励函数", test_ex1_reward), ("题2 组内优势", test_ex2_group),
          ("题3 KL 预算", test_ex3_kl)]


def main():
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 alignment_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
