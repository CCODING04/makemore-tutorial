#!/usr/bin/env python3
"""
Part 11 作业测试。

独立运行：python test_alignment_exercises.py
或 pytest：pytest test_alignment_exercises.py -v

测试原则：
- 测试数学性质（不变量），不测试精确值
- 处理 None 返回值（未实现的函数）
- 提供 informative 错误信息
"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_exercises import *  # noqa: F401,F403


# ═══════════════════════════════════════════════════════════════════════════════
# 题 1: 奖励函数测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_ex1_reward():
    """测试奖励函数的正确性和健壮性。"""
    assert math_reward is not None, "math_reward 未实现"

    # 基本功能测试
    cases = [
        # (response, ground_truth, expected_reward, description)
        ("\\\\boxed{42}", "42", 1.0, "基本 \\boxed{} 格式"),
        ("\\\\boxed{ 42 }", "42", 1.0, "\\boxed{} 带空格"),
        ("#### 3.5", "3.5", 1.0, "#### 格式"),
        ("The answer is 100, no wait, 7.", "7", 1.0, "自我纠正场景"),
        ("\\\\boxed{41}", "42", 0.0, "错误答案"),
        ("I don't know", "42", 0.0, "无数字"),
        ("There are 1,234 ways.", "1234", 1.0, "千分位逗号"),
        ("value is 42.", "42", 1.0, "尾随小数点"),
    ]

    for resp, gt, want, desc in cases:
        got = math_reward(resp, gt)
        assert got is not None, f"{desc}: 返回 None（未实现）"
        assert abs(got - want) < 1e-9, f"{desc}: {resp!r} gt={got}: got={got} != want={want}"


# ═══════════════════════════════════════════════════════════════════════════════
# 题 2: 组内优势测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_ex2_group():
    """测试组内优势的数学性质。"""
    assert group_advantages is not None, "group_advantages 未实现"

    # 测试 1: 基本功能
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert adv is not None, "返回 None（未实现）"

    # 性质 1: 优势之和应为 0
    assert abs(sum(adv)) < 1e-6, f"优势之和应为 0，实际为 {sum(adv)}"

    # 性质 2: 高奖励应有正优势，低奖励应有负优势
    assert adv[0] > 0, f"奖励 1.0 应有正优势，实际为 {adv[0]}"
    assert adv[1] < 0, f"奖励 0.0 应有负优势，实际为 {adv[1]}"

    # 测试 2: 全同组 → 全 0 而非 NaN
    adv_same = group_advantages([1.0, 1.0, 1.0, 1.0])
    assert adv_same is not None, "全同组返回 None"
    assert all(abs(a) < 1e-6 for a in adv_same), \
        f"全同组应 eps 兜底为全 0，实际为 {adv_same}"

    # 测试 3: 零梯度组检测
    assert zero_gradient_groups is not None, "zero_gradient_groups 未实现"
    result = zero_gradient_groups([[1.0, 1.0], [0.0, 1.0], [2.0, 2.0]])
    assert result == [0, 2], f"应检测出全同组 [0, 2]，实际为 {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 题 3: KL 惩罚测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_ex3_kl():
    """测试 KL 散度估计器的数学性质。"""
    assert k3_kl is not None, "k3_kl 未实现"

    # 性质 1: 同分布 KL 应为 0
    v = k3_kl([math.log(0.5)], [math.log(0.5)])
    assert v is not None, "k3_kl 返回 None（未实现）"
    assert abs(v) < 1e-9, f"同分布 KL 应为 0，实际为 {v}"

    # 性质 2: KL 应恒非负
    v2 = k3_kl([math.log(0.4), math.log(0.6)], [math.log(0.5), math.log(0.5)])
    assert v2 is not None, "k3_kl 返回 None（未实现）"
    assert v2 > 0, f"KL 应恒非负，实际为 {v2}"

    # 性质 3: KL 不对称（通常）
    v3 = k3_kl([math.log(0.5), math.log(0.5)], [math.log(0.4), math.log(0.6)])
    assert v3 is not None, "k3_kl 返回 None（未实现）"
    # 注意：k3 估计器在某些情况下可能对称，这里只检查非负
    assert v3 >= 0, f"KL 应非负，实际为 {v3}"

    # 测试预算护栏
    assert kl_budget_ok is not None, "kl_budget_ok 未实现"

    # 同分布应在预算内
    result1 = kl_budget_ok([math.log(0.5)], [math.log(0.5)])
    assert result1 is not None, "kl_budget_ok 返回 None（未实现）"
    assert result1 is True, "同分布应在预算内"

    # 大 KL 应超预算
    result2 = kl_budget_ok([0.0, 0.0], [math.log(2.0), 0.0], budget=0.05)
    assert result2 is not None, "kl_budget_ok 返回 None（未实现）"
    assert result2 is False, "大 KL 应超预算"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试运行器
# ═══════════════════════════════════════════════════════════════════════════════

_TESTS = [
    ("题1 奖励函数", test_ex1_reward),
    ("题2 组内优势", test_ex2_group),
    ("题3 KL 预算", test_ex3_kl),
]


def main():
    """运行所有测试并报告结果。"""
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"  ✅ {name}")
            p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}")
            f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}")
            f += 1

    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 alignment_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
