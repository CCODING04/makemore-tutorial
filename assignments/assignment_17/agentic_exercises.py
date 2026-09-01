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
    # TODO: 初始化全 0 mask，遍历各段置 1
    return None


def assistant_token_fraction(turn_spans, total_len):
    """assistant token 占比 = mask 中 1 的数量 / total_len。"""
    # TODO: 调用 build_trajectory_mask 后求均值
    return None


# ── 题 2：轨迹级 GRPO（30 分）────────────────────────────────
def trajectory_advantages(reward_matrix, eps=1e-6):
    """轨迹级 GRPO 优势：逐组标准化后广播到该轨迹全部 assistant token。

    Args:
        reward_matrix: list[list[float]]，外层 prompt、内层 G 条轨迹的奖励
        eps: std 兜底
    Returns:
        list[list[float]]：与输入同形状；组内全同时应全 0（eps 兜底）
    """
    # TODO: 逐组 mean/std → (r - mean)/std
    return None


# ── 题 3：工具调用解析器（25 分）─────────────────────────────
def parse_tool_calls(text):
    """解析全部工具调用（空格分隔协议）：
        <tool_call> name arg1 arg2 ... </tool_call>
    正则参考：r"<tool_call>\\s*([a-z_]+)((?:\\s+-?\\d+)*)\\s*</tool_call>"
    （args 可能为空——group(2) 为空串时返回空列表）

    Returns:
        list[dict]：[{"name": str, "args": [int, ...]}, ...]（按出现顺序）
    """
    # TODO: finditer 遍历，args 按空格 split 转 int
    return None


# ── 题 4（🌟）：Echo Trap 检测（15 分）──────────────────────
def echo_trap_score(trajectories):
    """多样性得分 = 不同轨迹数 / 总轨迹数（1=全部不同，→0=Echo Trap）。"""
    # TODO: set(tuple(t)) 去重后除以总数
    return None
