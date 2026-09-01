#!/usr/bin/env python3
"""
Part 11 作业：对齐实战（verl）

纯 Python 可完成的四道核心题 + 一道 stretch 选做（verl 概念的"手写侧"）。
实现后运行 test_alignment_exercises.py 验证（stretch 未实现会优雅跳过）。

对应教程：courses/Part11_alignment_verl/tutorial/
运行：python alignment_exercises.py（CPU 即可，<5 秒）
"""

import math
import re


# ═══════════════════════════════════════════════════════════════════════════════
# 题 1：稳健的数学答案奖励函数（25 分）
# ═══════════════════════════════════════════════════════════════════════════════

def math_reward(response: str, ground_truth: str) -> float:
    """RLVR 奖励函数（比 Part 11 脚本 01 多两个工程要求）。

    抽取链：\\boxed{} → '#### x' → 最后一个数字
    （不做 'answer is' 特判——"answer is 100, no wait, 7" 这类自我纠正应取最后的 7）

    Args:
        response: 模型生成的回答（字符串）
        ground_truth: 标准答案（字符串，如 "42" 或 "3.5"）

    Returns:
        float: 1.0（对）/ 0.0（错或抽不到）

    Steps:
        1. 用正则抽取 \\boxed{} 中的内容
        2. 如果没有，尝试抽取 '#### 42' 格式
        3. 如果都没有，抽取最后一个数字
        4. 处理千分位逗号和尾点
        5. 比较预测值和真实值

    常见陷阱：
        - response 为空字符串 → 返回 0.0
        - ground_truth 包含千分位逗号 "1,234" → 需要处理
        - 浮点数精度问题 → 用 abs(pred - gt) < 1e-4 判断
        - 自我纠正场景 → 取最后一个数字
    """
    # TODO:
    #   1. 按顺序尝试三种模式（正则），取第一个命中组
    #   2. 归一化：去千分位逗号、去尾部 '.'，float() 转（异常→0.0）
    #   3. |pred - gt| < 1e-4 → 1.0
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 题 2：GRPO 组内优势（25 分，单组语义）
# ═══════════════════════════════════════════════════════════════════════════════

def group_advantages(rewards, eps=1e-6):
    """GRPO 组内优势计算：A_i = (r_i - mean) / max(std, eps)。

    注意：单组语义——rewards 是**一个 prompt** 的 G 个回答的奖励（list[float]），
    不是脚本 01 里的 (n_prompts, n_responses) 二维批量。

    数学推导：
        mean = (1/G) * Σ r_i
        std = sqrt((1/G) * Σ (r_i - mean)^2)
        A_i = (r_i - mean) / max(std, eps)

        性质：
        - Σ A_i = 0（优势之和为零）
        - 如果所有 r_i 相同，则 std = 0，所有 A_i = 0
          → "太简单的题没有梯度"

    Args:
        rewards: list[float]，一个 prompt 的 G 个回答的奖励
        eps: 防止除零的小常数

    Returns:
        list[float]，长度与输入相同，且 sum≈0（数值精度允许 1e-6 误差）

    Steps:
        1. 计算组内均值 mean
        2. 计算组内标准差 std
        3. 如果 std < eps，返回全 0（全同组）
        4. 否则计算 A_i = (r_i - mean) / std
        5. 验证 Σ A_i = 0

    常见陷阱：
        - 全对/全错组 → std = 0 → 优势全 0（无梯度）
        - eps 位置要正确：max(std, eps) 而不是 std + eps
    """
    # TODO: Part 11 脚本 01 同款公式，注意 eps 位置
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 题 3：k3 KL 估计器（25 分）
# ═══════════════════════════════════════════════════════════════════════════════

def k3_kl(logp_ref, logp_new):
    """k3 估计器：mean(exp(d) - d - 1)，d = logp_ref - logp_new（恒 ≥0）。

    数学推导：
        KL(q || p) = E_q[log(q/p)] = E_q[log q - log p]
        令 d = log p_ref - log p_new
        则 KL = E[exp(d) - d - 1]

        性质：
        - exp(d) - d - 1 ≥ 0 对所有 d 成立（因为 e^x ≥ x + 1）
        - 当 d = 0 时取等号（两个分布相同）

    Args:
        logp_ref: 参考策略的对数概率列表（list[float]）
        logp_new: 新策略的对数概率列表（list[float]，与 ref 同一批 token）

    Returns:
        float: KL 散度估计值（恒非负）

    Steps:
        1. 遍历每个 token 的对数概率
        2. 计算 d = logp_ref - logp_new
        3. 累加 exp(d) - d - 1
        4. 返回平均值

    常见陷阱：
        - logp_ref 和 logp_new 必须是同一批 token 的对数概率
        - 除以列表长度取平均，不是直接返回累加和
    """
    # TODO: 纯 math 实现
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 题 4：KL 预算护栏（25 分）
# ═══════════════════════════════════════════════════════════════════════════════

def kl_budget_ok(logp_ref, logp_new, budget=0.05):
    """RL 常见护栏：估计 KL 超 budget 视为"策略漂移过大"，应提前停止/降 lr。

    Args:
        logp_ref: 参考策略的对数概率列表
        logp_new: 新策略的对数概率列表
        budget: KL 预算阈值（默认 0.05）

    Returns:
        bool：kl <= budget 时 True（在预算内）

    Steps:
        1. 调用 k3_kl(logp_ref, logp_new) 计算 KL 散度
        2. 比较 KL 与 budget
        3. 返回是否在预算内

    常见陷阱：
        - 参数是两个 logp 列表，不是 KL 值（函数内部自己算 KL）
        - budget 的选择很重要：太小会频繁停止，太大会允许策略偏离太远
        - 通常在 0.01-0.1 之间
    """
    # TODO: 调用 k3_kl 比较
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 🌟 题 5：零梯度组检测（Stretch，选做，未实现测试优雅跳过）
# ═══════════════════════════════════════════════════════════════════════════════

def zero_gradient_groups(reward_matrix):
    """🌟 Stretch：找出"全同奖励"的组下标（这些组本轮没有梯度，浪费的 rollout）。

    Args:
        reward_matrix: list[list[float]]
            外层是 prompt 列表，内层是每个 prompt 的 G 个回答的奖励
            形状：(n_prompts, n_responses)

    Returns:
        list[int]（组下标，升序）

    Steps:
        1. 遍历每个 prompt 的奖励列表
        2. 如果 max(rewards) == min(rewards)，则为全同组
        3. 返回全同组的下标列表

    常见陷阱：
        - 浮点数比较要用 abs(max - min) < eps
        - 返回的下标要升序排列
    """
    # TODO: max==min 的组（未实现返回 None，测试会 ⏭️ 跳过）
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 自检（可选）：实现完成后直接运行本文件做冒烟测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("骨架文件：请实现上面的 TODO 后运行 test_alignment_exercises.py")
    print(f"math_reward 未实现? {math_reward('x', 'x') is None}")
    print(f"group_advantages 未实现? {group_advantages([1.0, 0.0]) is None}")
    print(f"k3_kl 未实现? {k3_kl([0.0], [0.0]) is None}")
    print(f"kl_budget_ok 未实现? {kl_budget_ok([0.0], [0.0]) is None}")
    print(f"zero_gradient_groups (stretch) 未实现? "
          f"{zero_gradient_groups([[1.0, 1.0]]) is None}")
