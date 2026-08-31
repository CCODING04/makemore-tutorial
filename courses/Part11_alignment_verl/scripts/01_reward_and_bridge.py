#!/usr/bin/env python3
"""
Part 11 - 脚本 01: 奖励函数手写 + verl 概念桥接（纯 Python，无需 GPU/Docker）

目标：
  verl quickstart 的第一步就是"写一个可验证奖励函数"（RLVR）。本脚本把
  GSM8K 的规则奖励（官方 quickstart 同款语义：抽取 \\boxed{} / 最后数字 → 对错 0/1）
  手写并按"批量→组内优势"的管线跑通——这正是 02 章 verl 配置里
  reward_model.reward_manager 与 adv_estimator=grpo 背后的代码语义。

对应教程：tutorial/01_handwritten_to_verl.md
运行：python 01_reward_and_bridge.py（CPU 即可，<5 秒）

数据流追踪：
  rewards_per_prompt: list[list[float]]  # (n_prompts, n_responses)
    ↓ group_advantages()
  advantages: list[list[float]]          # (n_prompts, n_responses), 每组均值=0
    ↓ 用于 GRPO loss
  loss = -Σ(log_prob * advantage)        # 标量
"""

import os
import re
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 可验证奖励函数（RLVR 的核心；verl quickstart 的 custom reward 同语义）
# ═══════════════════════════════════════════════════════════════════════════════

def gsm8k_reward(response: str, ground_truth: str) -> float:
    """从模型回答里抽取数字，对则 1 分否则 0 分。

    抽取顺序（工程惯例）：\\boxed{} 优先 → '#### 42' 标记 → 最后一个数字。

    Args:
        response: 模型生成的回答（字符串）
        ground_truth: 标准答案（字符串，如 "42" 或 "3.5"）

    Returns:
        float: 1.0 表示正确，0.0 表示错误

    数学原理：
        RLVR (Reinforcement Learning with Verifiable Rewards) 的核心思想是
        用规则而非模型来评判答案——不可作弊、无 RM 偏差。

    常见陷阱：
        - response 为空字符串 → 返回 0.0
        - ground_truth 包含千分位逗号 "1,234" → 需要处理
        - 浮点数精度问题 → 用 abs(pred - gt) < 1e-4 判断
    """
    # Step 1: 尝试抽取 \boxed{} 中的内容
    m = re.findall(r"\\boxed\{(-?[\d,\.]+)\}", response)

    # Step 2: 如果没有 \boxed{}，尝试抽取 '#### 42' 格式
    if not m:
        m = re.findall(r"####\s*(-?[\d,\.]+)", response)

    # Step 3: 如果都没有，抽取最后一个数字
    if not m:
        m = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))

    # Step 4: 没有找到任何数字 → 错误
    if not m:
        return 0.0

    # Step 5: 取最后一个匹配的数字（工程惯例：最后的数字通常是最终答案）
    pred = m[-1].replace(",", "").rstrip(".")

    # Step 6: 比较预测值和真实值
    try:
        return 1.0 if abs(float(pred) - float(ground_truth)) < 1e-4 else 0.0
    except ValueError:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 批量奖励 → 组内优势（GRPO；Part 8 04 章手写逻辑的工具侧再现）
# ═══════════════════════════════════════════════════════════════════════════════

def group_advantages(rewards_per_prompt, eps=1e-6):
    """每个 prompt 采 G 个回答 → A_i = (r_i - mean) / std。

    数学推导（GRPO 核心公式）：
        给定一个 prompt 的 G 个回答的奖励 r_1, r_2, ..., r_G
        mean = (1/G) * Σ r_i
        std = sqrt((1/G) * Σ (r_i - mean)^2)
        A_i = (r_i - mean) / std

        性质：
        - Σ A_i = 0（优势之和为零，因为减去了均值）
        - 如果所有 r_i 相同（全对或全错），则 std = 0，所有 A_i = 0
          → "太简单的题没有梯度"，GRPO 天然跳过已掌握样本

    Args:
        rewards_per_prompt: list[list[float]]
            外层是 prompt 列表，内层是每个 prompt 的 G 个回答的奖励
            形状：(n_prompts, n_responses)
        eps: 防止除零的小常数

    Returns:
        list[list[float]]: 每个 prompt 的 G 个回答的优势值
            形状：(n_prompts, n_responses)

    常见陷阱：
        - 组内全对（全 1.0）→ std = 0 → 优势全 0（无梯度）
        - 组内全错（全 0.0）→ std = 0 → 优势全 0（无梯度）
        - 这两种情况在 RL 训练中很常见，需要通过课程学习缓解
    """
    advs = []
    for group in rewards_per_prompt:
        r = group
        n = len(r)

        # Step 1: 计算组内均值
        mean = sum(r) / n  # shape: scalar

        # Step 2: 计算组内标准差
        var = sum((x - mean) ** 2 for x in r) / n  # shape: scalar
        std = max(var ** 0.5, eps)  # shape: scalar, 防止除零

        # Step 3: 计算优势值
        group_adv = [(x - mean) / std for x in r]  # shape: (n_responses,)
        advs.append(group_adv)

    return advs


# ═══════════════════════════════════════════════════════════════════════════════
# 3. k3 KL 估计器（Part 8 04 章同款；verl 的 KL 惩罚即此形态）
# ═══════════════════════════════════════════════════════════════════════════════

def k3_kl(logp_ref, logp_new):
    """KL(π_new || π_ref) 的低方差估计：exp(d) - d - 1, d = logp_ref - logp_new

    数学推导：
        KL(q || p) = E_q[log(q/p)] = E_q[log q - log p]

        令 d = log p_ref - log p_new
        则 KL = E[exp(d) - d - 1]

        性质：
        - exp(d) - d - 1 ≥ 0 对所有 d 成立（因为 e^x ≥ x + 1）
        - 当 d = 0 时取等号（两个分布相同）
        - 这是一个低方差估计器，比直接计算 KL 更稳定

    Args:
        logp_ref: 参考策略的对数概率列表
        logp_new: 新策略的对数概率列表

    Returns:
        float: KL 散度估计值（恒非负）

    常见陷阱：
        - logp_ref 和 logp_new 必须是同一批 token 的对数概率
        - 如果概率为 0，log 会变成 -inf，需要特殊处理
    """
    kl = 0.0
    for lr, ln in zip(logp_ref, logp_new):
        d = lr - ln  # shape: scalar
        # exp(d) - d - 1，使用 math.exp 处理标量
        kl += (math.exp(d) if isinstance(d, (int, float)) else d.exp()) - d - 1
    return kl / len(logp_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 主函数：验证所有组件
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═══ 奖励函数 + GRPO 管线（verl 概念桥接）═══\n")

    # ── 1) 奖励函数单测（RLVR 的"考官"）───
    print("[1] GSM8K 规则奖励函数（verl quickstart 的 custom reward 同语义）:")
    print("    数据流: response(str) → reward(float)")
    print()

    cases = [
        ("答案是 \\boxed{42}。", "42", 1.0),
        ("#### 3.5", "3.5", 1.0),
        ("我觉得是 100，不对，是 7。", "7", 1.0),          # 最后一个数字
        ("答案 \\boxed{41}", "42", 0.0),
        ("我不会。", "42", 0.0),                            # 无数字
        ("1,234 个。", "1234", 1.0),                        # 千分位
    ]

    for resp, gt, want in cases:
        got = gsm8k_reward(resp, gt)
        assert abs(got - want) < 1e-9, f"FAIL: {resp!r} got={got}, want={want}"
        print(f"    {resp[:22]!r:26} gt={gt:>5} → reward={got}")

    print()

    # ── 2) 批量 rollout 打分 → 组内优势 ──
    print("[2] 组内优势（adv_estimator=grpo 的语义）:")
    print("    数据流: rewards(n_prompts, n_responses) → advantages(n_prompts, n_responses)")
    print()

    # 模拟：3 个 prompt × 每个采 4 个回答的奖励（真实中来自 rollout 引擎生成后打分）
    batch = [
        [1.0, 0.0, 1.0, 0.0],     # 组内对半 → 优势两正两负
        [1.0, 1.0, 1.0, 1.0],     # 全对 → 优势全 0（GRPO 的著名边界：无区分度）
        [0.0, 0.0, 1.0, 0.0],     # 一个对 → 它拿全部正优势
    ]

    advs = group_advantages(batch)

    for i, (grp, adv) in enumerate(zip(batch, advs)):
        # 验证：优势之和应为 0（数值精度允许 1e-6 误差）
        assert abs(sum(adv)) < 1e-6, f"FAIL: sum(adv) = {sum(adv)} != 0"
        print(f"    prompt{i} rewards={grp} → adv={[round(a, 2) for a in adv]}")

    print()
    print("    ⚠️ 关键观察：prompt1 全对 → 优势全 0：'太简单的题没有梯度'")
    print("    这是 GRPO 的天然特性：已掌握的样本不会产生梯度更新")

    # ── 3) KL 惩罚 ──
    print()
    print("[3] k3 KL（verl 的 KL 惩罚形态）:")
    print("    数据流: logp_ref(list), logp_new(list) → kl(scalar)")
    print()

    kl = k3_kl([math.log(0.4), math.log(0.6)], [math.log(0.5), math.log(0.5)])
    print(f"    KL(π_new || π_ref) = {kl:.4f}")
    print(f"    性质: ≥ 0（恒非负，估计器保证）")

    # ── 4) 总结：从手写到 verl 的映射 ──
    print("""
═══ 从手写到 verl：概念映射 ═══

┌─────────────────────────────────────────────────────────────────────┐
│  你手写的（本脚本）              │  verl 里的对应                    │
├─────────────────────────────────────────────────────────────────────┤
│  gsm8k_reward()                 │  custom_reward_function           │
│  group_advantages()             │  algorithm.adv_estimator=grpo     │
│  k3_kl()                        │  algorithm.kl_penalty             │
│  单进程 for 循环                │  Ray 单控制器数据流               │
│  玩具模型同时干生成+训练        │  actor_rollout_ref 三个角色       │
└─────────────────────────────────────────────────────────────────────┘

到 verl 只剩三步（02 章）：
  ① 我们手写的"生成 G 个回答 → 打分 → 组内优势 → KL 惩罚 → 更新"
     在 verl 里 = actor_rollout_ref 三角色 + adv_estimator=grpo 配置
  ② 我们的手写版在玩具模型上同步执行；工业版 rollout(vLLM/SGLang) 与
     训练(FSDP2) 是两个引擎，靠【权重回同步】衔接（3D-HybridEngine 省这一步的开销）
  ③ 奖励函数 = 你在 quickstart 里唯一必须自己写的代码（本脚本 [1] 的工程化版本）

💡 面试要点：
   - GRPO 什么情况下优势全零？→ 组内奖励全相同（题已全对/全错），天然跳过
   - 这也是 GRPO 比 PPO 稳定的原因之一（基线来自组内而非学习出的 Value）
   - RLVR vs RM：规则奖励不可作弊，但只适用于可形式化验证的任务
""")


if __name__ == '__main__':
    main()
