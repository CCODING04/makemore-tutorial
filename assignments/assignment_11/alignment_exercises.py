#!/usr/bin/env python3
"""
Part 11 作业：对齐实战（verl）

纯 Python/纸笔可完成的三道题（verl 概念的"手写侧"）。实现后运行
test_alignment_exercises.py 验证。

对应教程：courses/Part11_alignment_verl/tutorial/
运行：python alignment_exercises.py（CPU 即可，<5 秒）
"""

import math
import re


# ═══════════════════════════════════════════════════════════════════════════════
# 题 1：稳健的数学答案奖励函数（40 分）
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
# 题 2：GRPO 组内优势 + 全同组处理（30 分）
# ═══════════════════════════════════════════════════════════════════════════════

def group_advantages(rewards, eps=1e-6):
    """GRPO 组内优势计算：A_i = (r_i - mean) / max(std, eps)。

    数学推导：
        mean = (1/G) * Σ r_i
        std = sqrt((1/G) * Σ (r_i - mean)^2)
        A_i = (r_i - mean) / std

        性质：
        - Σ A_i = 0（优势之和为零）
        - 如果所有 r_i 相同，则 std = 0，所有 A_i = 0
          → "太简单的题没有梯度"

    Args:
        rewards: list[float]，一个 prompt 的 G 个回答的奖励
        eps: 防止除零的小常数

    Returns:
        list[float]，且 sum≈0（数值精度允许 1e-6 误差）

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
    # TODO: Part 11 脚本 01 同款，注意 eps 位置
    return None


def zero_gradient_groups(reward_matrix):
    """找出"全同奖励"的组下标（这些组本轮没有梯度，浪费的 rollout）。

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
    # TODO: max==min 的组
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 题 3：KL 惩罚预算（30 分）
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
        logp_ref: 参考策略的对数概率列表
        logp_new: 新策略的对数概率列表

    Returns:
        float: KL 散度估计值（恒非负）

    Steps:
        1. 遍历每个 token 的对数概率
        2. 计算 d = logp_ref - logp_new
        3. 累加 exp(d) - d - 1
        4. 返回平均值

    常见陷阱：
        - logp_ref 和 logp_new 必须是同一批 token 的对数概率
        - 如果概率为 0，log 会变成 -inf，需要特殊处理
    """
    # TODO: 纯 math 实现
    return None


def kl_budget_ok(logp_ref, logp_new, budget=0.05):
    """RL 常见护栏：估计 KL 超 budget 视为"策略漂移过大"，应提前停止/降 lr。

    Args:
        logp_ref: 参考策略的对数概率列表
        logp_new: 新策略的对数概率列表
        budget: KL 预算阈值（默认 0.05）

    Returns:
        bool：kl <= budget 时 True（在预算内）

    Steps:
        1. 调用 k3_kl 计算 KL 散度
        2. 比较 KL 与 budget
        3. 返回是否在预算内

    常见陷阱：
        - budget 的选择很重要：太小会频繁停止，太大会允许策略偏离太远
        - 通常在 0.01-0.1 之间
    """
    # TODO: 调用 k3_kl 比较
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 🌟 练习 4: 奖励函数设计（Stretch Goal）
# ═══════════════════════════════════════════════════════════════════════════════

def anti_hacking_reward(response: str, ground_truth: str) -> float:
    """设计一个防作弊的奖励函数，用于评估数学推理质量。

    Args:
        response: 模型生成的回答
        ground_truth: 标准答案

    Returns:
        float: 0.0-1.0 之间的分数

    Steps:
        1. 检查答案正确性（使用 math_reward 函数）
        2. 检查推理过程（是否有逻辑连接词）
        3. 检查回答长度（惩罚过短）
        4. 综合计算分数

    常见陷阱：
        - 只看答案不够，模型可能学会"钻空子"
        - 推理过程检查要宽松（模型可能用不同的词）
        - 长度检查要合理（太短可能没有推理过程）
    """
    # TODO: Implement
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 🌟 练习 5: Rollout 成本计算器（Stretch Goal）
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_rollout_cost(
    model_params_B: float,  # 模型参数量（单位：B）
    n: int,                 # 组大小
    num_prompts: int,       # prompt 数量
    num_gpus: int = 1,      # GPU 数量
) -> float:
    """估算 rollout 时间（秒）。

    Args:
        model_params_B: 模型参数量（单位：B）
        n: 组大小
        num_prompts: prompt 数量
        num_gpus: GPU 数量

    Returns:
        float: 预估的 rollout 时间（秒）

    经验公式：
        - 每个 token 的生成时间 ≈ 0.1ms * model_params_B
        - 每个回答平均 100 tokens
        - 总时间 = prompts * n * tokens * time_per_token / num_gpus

    Steps:
        1. 计算每个 token 的生成时间
        2. 计算总 token 数
        3. 计算总时间
        4. 考虑 GPU 并行度

    常见陷阱：
        - 这是粗略估计，实际时间受多种因素影响
        - 小模型的生成时间可能比估计的快
        - 大模型的生成时间可能比估计的慢
    """
    # TODO: Implement
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 🌟 练习 6: KL 预算护栏（Stretch Goal）
# ═══════════════════════════════════════════════════════════════════════════════

def kl_budget_guard(
    logp_ref: list,
    logp_new: list,
    budget: float = 0.05,
) -> tuple:
    """KL 预算护栏：监控 KL 散度并在超预算时发出警告。

    Args:
        logp_ref: 参考策略的对数概率列表
        logp_new: 新策略的对数概率列表
        budget: KL 预算阈值（默认 0.05）

    Returns:
        tuple: (is_ok, kl_value, warning_msg)
            - is_ok: bool，是否在预算内
            - kl_value: float，当前 KL 值
            - warning_msg: str，警告信息（如果超预算）

    Steps:
        1. 调用 k3_kl 计算 KL 散度
        2. 比较 KL 与 budget
        3. 如果超预算，生成警告信息
        4. 返回结果

    常见陷阱：
        - budget 的选择很重要：太小会频繁停止，太大会允许策略偏离太远
        - 警告信息要详细，便于调试
    """
    # TODO: Implement
    return None
