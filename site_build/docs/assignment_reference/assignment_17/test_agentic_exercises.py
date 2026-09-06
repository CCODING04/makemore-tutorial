#!/usr/bin/env python3
"""Assignment 17 测试。独立运行：python test_agentic_exercises.py；或 pytest。
题 4 为 🌟 stretch：未实现（返回 None）时优雅 SKIP ⏭️，不算失败。"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentic_exercises import *  # noqa: F401,F403


class SkipTest(Exception):
    """🌟 stretch 题未实现（返回 None）时的优雅跳过信号（独立运行模式）。"""


def _skip(msg):
    """优雅跳过：pytest 环境用 pytest.skip；独立运行抛 SkipTest 由 main 捕获。"""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise SkipTest(msg)


def test_ex1_mask():
    assert build_trajectory_mask is not None, "build_trajectory_mask 未实现"
    m = build_trajectory_mask([(4, 9), (13, 20)], 24)
    assert m is not None, "未实现（返回 None）"
    assert len(m) == 24
    assert sum(m) == (9 - 4) + (20 - 13), "mask 的 1 的数量应等于各段长度和"
    assert m[4] == 1 and m[8] == 1 and m[9] == 0 and m[12] == 0 and m[13] == 1
    assert all(x == 0 for x in m[:4]) and all(x == 0 for x in m[20:])
    # 两段各 8 个 assistant token：16/32 = 0.5
    assert assistant_token_fraction([(4, 12), (16, 24)], 32) == 0.5, \
        "两段各 8 个 token：16/32 = 0.5"


def test_ex2_traj_grpo():
    assert trajectory_advantages is not None, "trajectory_advantages 未实现"
    adv = trajectory_advantages([[1.0, 0.0, 1.0, 0.0]])
    assert adv is not None, "未实现（返回 None）"
    # 组内标准化：全同组 → 全 0
    same = trajectory_advantages([[2.0, 2.0], [2.0, 2.0]])
    assert all(abs(x) < 1e-9 for grp in same for x in grp), "全同组优势应为全 0"
    # 区分组：0/1 组 → 优势应为一正一负
    adv01 = trajectory_advantages([[0.0, 1.0]])[0]
    assert abs(adv01[0] + 1) < 1e-6 and abs(adv01[1] - 1) < 1e-6


def test_ex3_parse():
    assert parse_tool_calls is not None, "parse_tool_calls 未实现"
    text = "think... <tool_call> multiply 3 2 </tool_call> obs <tool_call> add 6 1 </tool_call>"
    calls = parse_tool_calls(text)
    assert calls is not None and len(calls) == 2, \
        "未实现（返回 None）或解析数量不对：应按序解析出 2 个调用"
    assert calls[0] == {"name": "multiply", "args": [3, 2]}
    assert calls[1] == {"name": "add", "args": [6, 1]}
    assert parse_tool_calls("no calls here") == []


def test_ex4_echo():
    # 🌟 stretch：骨架返回 None → 优雅 SKIP（不算失败）
    score = echo_trap_score([[1, 2], [1, 2], [1, 2]])
    if score is None:
        _skip("🌟 未实现（echo_trap_score 返回 None），先完成题 1-3 再挑战")
    assert abs(score - 1 / 3) < 1e-9, "3 条相同轨迹多样性 = 1/3"
    b = [[1, 2], [3, 4], [5, 6]]
    assert abs(echo_trap_score(b) - 1.0) < 1e-9, "全不同 → 多样性 1.0"
    # Echo Trap 检测：多样性得分低 = 探索坍缩（教程 02 章引用 RAGEN 论文）


_TESTS = [("题1 轨迹 mask", test_ex1_mask), ("题2 轨迹级 GRPO", test_ex2_traj_grpo),
          ("题3 工具调用解析", test_ex3_parse), ("题4 🌟 Echo Trap", test_ex4_echo)]


def main():
    p = f = s = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except SkipTest as e:
            print(f"  ⏭️ {name} — SKIP: {e}"); s += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    summary = f"\n  通过: {p}/{p + f}"
    if s:
        summary += f"（另 {s} 题 SKIP ⏭️）"
    print(summary + ("  🎉" if f == 0 else "  💡 先实现 agentic_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
