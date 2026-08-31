#!/usr/bin/env python3
"""Assignment 17 参考答案：Agentic RL。纯 CPU 可验证。"""

import math
import re


def build_trajectory_mask(turn_spans, total_len):
    mask = [0] * total_len
    for start, end in turn_spans:
        for i in range(start, min(end, total_len)):
            mask[i] = 1
    return mask


def assistant_token_fraction(turn_spans, total_len):
    mask = build_trajectory_mask(turn_spans, total_len)
    return sum(mask) / total_len


def trajectory_advantages(reward_matrix, eps=1e-6):
    out = []
    for group in reward_matrix:
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        std = max(var ** 0.5, eps)
        out.append([(r - mean) / std for r in group])
    return out


def parse_tool_calls(text):
    calls = []
    for m in re.finditer(
            r"<tool_call>\s*([a-z_]+)((?:\s+-?\d+)*)\s*</tool_call>", text):
        args = [int(x) for x in m.group(2).split()] if m.group(2).strip() else []
        calls.append({"name": m.group(1), "args": args})
    return calls


def echo_trap_score(trajectories):
    seen = {tuple(t) for t in trajectories}
    return len(seen) / len(trajectories)
