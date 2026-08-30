#!/usr/bin/env python3
"""
Part 11 - 脚本 01: 奖励函数手写 + verl 概念桥接（纯 Python，无需 GPU/Docker）
目标：verl quickstart 的第一步就是"写一个可验证奖励函数"（RLVR）。本脚本把
      GSM8K 的规则奖励（官方 quickstart 同款语义：抽取 \\boxed{} / 最后数字 → 对错 0/1）
      手写并按"批量→组内优势"的管线跑通——这正是 02 章 verl 配置里
      reward_modelReward 函数与 adv_estimator=grpo 背后的代码语义。

对应教程：tutorial/01_handwritten_to_verl.md
运行：python 01_reward_and_bridge.py（CPU 即可，<5 秒）
"""

import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 1. 可验证奖励函数（RLVR 的核心；verl quickstart 的 custom reward 同语义）───
def gsm8k_reward(response: str, ground_truth: str) -> float:
    """从模型回答里抽取数字，对则 1 分否则 0 分。
    抽取顺序（工程惯例）：\\boxed{} 优先 → '#### 42' 标记 → 最后一个数字。"""
    m = re.findall(r"\\boxed\{(-?[\d,\.]+)\}", response)
    if not m:
        m = re.findall(r"####\s*(-?[\d,\.]+)", response)
    if not m:
        m = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))
    if not m:
        return 0.0
    pred = m[-1].replace(",", "").rstrip(".")
    try:
        return 1.0 if abs(float(pred) - float(ground_truth)) < 1e-4 else 0.0
    except ValueError:
        return 0.0


# ─── 2. 批量奖励 → 组内优势（GRPO；Part 8 04 章手写逻辑的工具侧再现）───
def group_advantages(rewards_per_prompt, eps=1e-6):
    """每个 prompt 采 G 个回答 → A_i = (r_i - mean) / std。
    rewards_per_prompt: list[list[float]]（外层 prompt，内层 G 个回答）"""
    advs = []
    for group in rewards_per_prompt:
        r = group
        mean = sum(r) / len(r)
        var = sum((x - mean) ** 2 for x in r) / len(r)
        std = max(var ** 0.5, eps)
        advs.append([(x - mean) / std for x in r])
    return advs


# ─── 3. k3 KL 估计器（Part 8 04 章同款；verl 的 KL 惩罚即此形态）───
def k3_kl(logp_ref, logp_new):
    """KL(π_new || π_ref) 的低方差估计：exp(d) - d - 1, d = logp_ref - logp_new"""
    kl = 0.0
    for lr, ln in zip(logp_ref, logp_new):
        d = lr - ln
        kl += (d.exp() if hasattr(d, 'exp') else __import__('math').exp(d)) - d - 1
    return kl / len(logp_ref)


def main():
    print("═══ 奖励函数 + GRPO 管线（verl 概念桥接）═══\n")

    # ── 1) 奖励函数单测（RLVR 的"考官"）───
    cases = [
        ("答案是 \\boxed{42}。", "42", 1.0),
        ("#### 3.5", "3.5", 1.0),
        ("我觉得是 100，不对，是 7。", "7", 1.0),          # 最后一个数字
        ("答案 \\boxed{41}", "42", 0.0),
        ("我不会。", "42", 0.0),                            # 无数字
        ("1,234 个。", "1234", 1.0),                        # 千分位
    ]
    print("[1] GSM8K 规则奖励函数（verl quickstart 的 custom reward 同语义）:")
    for resp, gt, want in cases:
        got = gsm8k_reward(resp, gt)
        assert abs(got - want) < 1e-9, (resp, got, want)
        print(f"    {resp[:22]!r:26} gt={gt:>5} → reward={got}")

    # ── 2) 批量 rollout 打分 → 组内优势 ──
    # 模拟：3 个 prompt × 每个采 4 个回答的奖励（真实中来自 rollout 引擎生成后打分）
    batch = [
        [1.0, 0.0, 1.0, 0.0],     # 组内对半 → 优势两正两负
        [1.0, 1.0, 1.0, 1.0],     # 全对 → 优势全 0（GRPO 的著名边界：无区分度）
        [0.0, 0.0, 1.0, 0.0],     # 一个对 → 它拿全部正优势
    ]
    advs = group_advantages(batch)
    print("\n[2] 组内优势（adv_estimator=grpo 的语义）:")
    for i, (grp, adv) in enumerate(zip(batch, advs)):
        assert abs(sum(adv)) < 1e-6
        print(f"    prompt{i} rewards={grp} → adv={[round(a, 2) for a in adv]}")
    print("    注意 prompt1 全对 → 优势全 0：'太简单的题没有梯度'，GRPO 天然跳过已掌握样本")

    # ── 3) KL 惩罚 ──
    import math
    kl = k3_kl([math.log(0.4), math.log(0.6)], [math.log(0.5), math.log(0.5)])
    print(f"\n[3] k3 KL（verl 的 KL 惩罚形态）: {kl:.4f} ≥ 0（恒非负，估计器性质）")

    print("""
═══ 到 verl 只剩三步（02 章）═══
  ① 我们手写的"生成 G 个回答 → 打分 → 组内优势 → KL 惩罚 → 更新"
     在 verl 里 = actor_rollout_ref 三角色 + adv_estimator=grpo 配置
  ② 我们的手写版在玩具模型上同步执行；工业版 rollout(vLLM/SGLang) 与
     训练(FSDP2) 是两个引擎，靠【权重回同步】衔接（3D-HybridEngine 省这一步的开销）
  ③ 奖励函数 = 你在 quickstart 里唯一必须自己写的代码（本脚本 [1] 的工程化版本）
  💡 面试：GRPO 什么情况下优势全零？→ 组内奖励全相同（题已全对/全错），天然跳过；
     这也是 GRPO 比 PPO 稳定的原因之一（基线来自组内而非学习出的 Value）。""")


if __name__ == '__main__':
    main()
