#!/usr/bin/env python3
"""Part 11 作业参考实现（4 核心 + 1 stretch，与作业骨架签名一一对应）。"""

import math
import re


# ═══════════════════════════════════════════════════════════════════════════════
# 题 1：稳健的数学答案奖励函数（25 分）
# ═══════════════════════════════════════════════════════════════════════════════

def math_reward(response: str, ground_truth: str) -> float:
    """RLVR 奖励函数：抽取链 \\boxed{} → '#### x' → 最后一个数字。"""
    for pat in (r"\\boxed\{(-?[\d,\.]+)\}",      # 1. \boxed{} 优先
                r"####\s*(-?[\d,\.]+)",           # 2. '#### 42' 标记
                r"(-?\d[\d,]*(?:\.\d+)?)"):       # 3. 最后一个数字
        m = re.findall(pat, response, flags=re.I)
        if m:
            try:
                # 归一化：去千分位逗号、去尾部 '.'，转 float（异常→继续下一级兜底）
                pred = float(m[-1].replace(",", "").rstrip("."))
                return 1.0 if abs(pred - float(ground_truth)) < 1e-4 else 0.0
            except ValueError:
                continue
    return 0.0  # 一个数字都抽不到 → 0.0（不是 None、不抛异常）


# ═══════════════════════════════════════════════════════════════════════════════
# 题 2：GRPO 组内优势（25 分，单组语义）
# ═══════════════════════════════════════════════════════════════════════════════

def group_advantages(rewards, eps=1e-6):
    """A_i = (r_i - mean) / max(std, eps)；全同组 → 全 0（无梯度）。"""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((x - mean) ** 2 for x in rewards) / n   # 总体方差（除以 G）
    std = max(var ** 0.5, eps)                        # eps 兜底防除零
    return [(x - mean) / std for x in rewards]


# ═══════════════════════════════════════════════════════════════════════════════
# 题 3：k3 KL 估计器（25 分）
# ═══════════════════════════════════════════════════════════════════════════════

def k3_kl(logp_ref, logp_new):
    """mean(exp(d) - d - 1)，d = logp_ref - logp_new（恒 ≥ 0）。"""
    total = sum(math.exp(a - b) - (a - b) - 1 for a, b in zip(logp_ref, logp_new))
    return total / len(logp_ref)   # 平均，不是求和


# ═══════════════════════════════════════════════════════════════════════════════
# 题 4：KL 预算护栏（25 分）
# ═══════════════════════════════════════════════════════════════════════════════

def kl_budget_ok(logp_ref, logp_new, budget=0.05):
    """kl <= budget → True（在预算内）。内部自己算 KL，参数是两个 logp 列表。"""
    return k3_kl(logp_ref, logp_new) <= budget


# ═══════════════════════════════════════════════════════════════════════════════
# 🌟 题 5：零梯度组检测（Stretch）
# ═══════════════════════════════════════════════════════════════════════════════

def zero_gradient_groups(reward_matrix):
    """返回全同奖励组（无梯度组）的下标，升序。"""
    eps = 1e-9
    return [i for i, g in enumerate(reward_matrix)
            if abs(max(g) - min(g)) < eps]
