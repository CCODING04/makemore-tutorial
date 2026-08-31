#!/usr/bin/env python3
"""Assignment 17 测试。独立运行：python test_agentic_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentic_exercises import *  # noqa: F401,F403


def test_ex1_mask():
    assert build_trajectory_mask is not None, "build_trajectory_mask 未实现"
    m = build_trajectory_mask([(4, 9), (13, 20)], 24)
    assert m is not None, "未实现（返回 None）"
    assert len(m) == 24
    assert sum(m) == (9 - 4) + (20 - 13), "mask 的 1 的数量应等于各段长度和"
    assert m[4] == 1 and m[8] == 1 and m[9] == 0 and m[12] == 0 and m[13] == 1
    assert all(x == 0 for x in m[:4]) and all(x == 0 for x in m[20:])
    assert assistant_token_fraction([(4, 12), (16, 24)], 32) == 0.5, "12/32 = 0.375… 需按实现核对"


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
    assert calls is not None and len(calls) == 2, "应解析出 2 个调用"
    assert calls[0] == {"name": "multiply", "args": [3, 2]}
    assert calls[1] == {"name": "add", "args": [6, 1]}
    assert parse_tool_calls("no calls here") == []


def test_ex4_echo():
    assert echo_trap_score is not None, "echo_trap_score 未实现"
    a = [[1, 2], [1, 2], [1, 2]]
    assert abs(echo_trap_score(a) - 1 / 3) < 1e-9, "3 条相同轨迹多样性 = 1/3"
    b = [[1, 2], [3, 4], [5, 6]]
    assert abs(echo_trap_score(b) - 1.0) < 1e-9, "全不同 → 多样性 1.0"
    # Echo Trap 检测：多样性得分低 = 探索坍缩（教程 02 章引用 RAGEN 论文）


_TESTS = [("题1 轨迹 mask", test_ex1_mask), ("题2 轨迹级 GRPO", test_ex2_traj_grpo),
          ("题3 工具调用解析", test_ex3_parse), ("题4 🌟 Echo Trap", test_ex4_echo)]


def main():
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 agentic_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
