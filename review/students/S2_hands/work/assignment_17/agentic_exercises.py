#!/usr/bin/env python3
"""Assignment 17 作业骨架：Agentic RL。在此文件实现四道题，用同目录
test_agentic_exercises.py 验证（python 直跑或 pytest 均可）。纯 CPU 可完成；
题 4 为 🌟 弹性题——不实现保持 return None 即可，测试会优雅 SKIP ⏭️。
题目说明与验收清单见同目录 assignment.md。"""

import math
import random
import re


# ── 题 1：多轮轨迹的 loss mask（30 分）──────────────────────
def build_trajectory_mask(turn_spans, total_len):
    """给定各 assistant 段的 (start, end) 闭开区间切片，构造整条轨迹的 loss mask。

    Args:
        turn_spans: list[(start, end)]，assistant 段的闭开区间
        total_len: 轨迹总长
    Returns:
        list[int]：长度 total_len 的 0/1 mask（assistant 段为 1，观测/padding 为 0）
    """
    mask = [0] * total_len
    for start, end in turn_spans:
        for i in range(start, end):
            mask[i] = 1
    return mask


def assistant_token_fraction(turn_spans, total_len):
    """assistant token 占比 = mask 中 1 的数量 / total_len。"""
    mask = build_trajectory_mask(turn_spans, total_len)
    return sum(mask) / total_len


# ── 题 2：轨迹级 GRPO（30 分）────────────────────────────────
def trajectory_advantages(reward_matrix, eps=1e-6):
    """轨迹级 GRPO 优势：逐组标准化后广播到该轨迹全部 assistant token。

    Args:
        reward_matrix: list[list[float]]，外层 prompt、内层 G 条轨迹的奖励
        eps: std 兜底
    Returns:
        list[list[float]]：与输入同形状；组内全同时应全 0（eps 兜底）
    """
    out = []
    for rewards in reward_matrix:
        n = len(rewards)
        mean = sum(rewards) / n
        std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
        denom = std if std > 0 else eps   # eps 只在 std=0（全同组）时兜底
        out.append([(r - mean) / denom for r in rewards])
    return out


# ── 题 3：工具调用解析器（25 分）─────────────────────────────
def parse_tool_calls(text):
    """解析全部工具调用（空格分隔协议）：
        <tool_call> name arg1 arg2 ... </tool_call>
    正则参考：r"<tool_call>\\s*([a-z_]+)((?:\\s+-?\\d+)*)\\s*</tool_call>"
    （args 可能为空——group(2) 为空串时返回空列表）

    Returns:
        list[dict]：[{"name": str, "args": [int, ...]}, ...]（按出现顺序）
    """
    calls = []
    for m in re.finditer(r"<tool_call>\s*([a-z_]+)((?:\s+-?\d+)*)\s*</tool_call>", text):
        args_str = m.group(2).strip()
        args = [int(x) for x in args_str.split()] if args_str else []
        calls.append({"name": m.group(1), "args": args})
    return calls


# ── 题 4（🌟）：Echo Trap 检测（15 分）──────────────────────
def echo_trap_score(trajectories):
    """多样性得分 = 不同轨迹数 / 总轨迹数（1=全部不同，→0=Echo Trap）。"""
    if not trajectories:
        return 0.0
    return len({tuple(t) for t in trajectories}) / len(trajectories)
