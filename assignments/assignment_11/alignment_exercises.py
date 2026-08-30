#!/usr/bin/env python3
"""
Part 11 作业：对齐实战（verl）

纯 Python/纸笔可完成的三道题（verl 概念的"手写侧"）。实现后运行
test_alignment_exercises.py 验证。
"""

import math
import re


# ── 题 1：稳健的数学答案奖励函数（40 分）────────────────────
def math_reward(response: str, ground_truth: str) -> float:
    """RLVR 奖励函数（比 Part 11 脚本 01 多两个工程要求）：
      ① 抽取链：\\boxed{} → '#### x' → 最后一个数字（不做 'answer is' 特判——
         "answer is 100, no wait, 7" 这类自我纠正应取最后的 7）
      ② 千分位/尾随小数健壮性："1,234"、"42."、"3.50" 都要能对上
    Returns:
        1.0（对）/ 0.0（错或抽不到）
    """
    # TODO:
    #   1. 按顺序尝试三种模式（正则），取第一个命中组
    #   2. 归一化：去千分位逗号、去尾部 '.'，float() 转（异常→0.0）
    #   3. |pred - gt| < 1e-4 → 1.0
    return None


# ── 题 2：GRPO 组内优势 + 全同组处理（30 分）────────────────
def group_advantages(rewards, eps=1e-6):
    """A_i = (r_i - mean)/max(std, eps)。
    注意：全同组（如全 1.0）std=0 → 应返回全 0（eps 兜底），而不是 NaN。
    Returns:
        list[float]，且 sum≈0
    """
    # TODO: Part 11 脚本 01 同款，注意 eps 位置
    return None


def zero_gradient_groups(reward_matrix):
    """找出"全同奖励"的组下标（这些组本轮没有梯度，浪费的 rollout）。

    Args:
        reward_matrix: list[list[float]]
    Returns:
        list[int]（组下标，升序）
    """
    # TODO: max==min 的组
    return None


# ── 题 3：KL 惩罚预算（30 分）────────────────────────────────
def k3_kl(logp_ref, logp_new):
    """k3 估计器：mean(exp(d) - d - 1)，d = logp_ref - logp_new（恒 ≥0）。"""
    # TODO: 纯 math 实现
    return None


def kl_budget_ok(logp_ref, logp_new, budget=0.05):
    """RL 常见护栏：估计 KL 超 budget 视为"策略漂移过大"，应提前停止/降 lr。
    Returns:
        bool：kl <= budget 时 True（在预算内）
    """
    # TODO: 调用 k3_kl 比较
    return None
