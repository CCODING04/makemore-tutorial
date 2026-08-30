#!/usr/bin/env python3
"""
Part 8 作业：从零训练 LLM —— 后训练全流程

本作业带你从零实现 LLM 后训练全流程的八个关键组件：

  题 1. Causal Self-Attention Head（因果自注意力头）
  题 2. Pre-LN Transformer Block（Pre-LN 残差块）
  题 3. Prompt-Masked SFT Loss（提示词遮罩的 SFT 损失）
  题 4. Bradley-Terry Reward Loss（Bradley-Terry 奖励损失）
  题 5. DPO Loss（直接偏好优化损失）
  题 6.（🌟 拓展）GAE Advantage Estimation（广义优势估计）
  题 7.（🌟 拓展）PPO Clipped Loss（PPO 裁剪策略损失）
  题 8.（🌟 拓展）GRPO Group Advantage（GRPO 组相对优势）

所有函数/类定义在编写后应该能用下面的测试脚本验证：
  python test_post_training_exercises.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
#  题 1：Causal Self-Attention Head（基础）
# ═══════════════════════════════════════════════════════════════════

class Head(nn.Module):
    """单头因果注意力。

    Q/K 各自投影后算内积 → scale → causal mask（上三角 -inf）→ softmax → 加权 V。

    Args:
        head_size: 每个注意力头的维度（通常 n_embed // n_head）。
        n_embed: 输入 embedding 维度。
        context_length: 最大序列长度（用于预分配 causal mask）。

    Attributes:
        key: K 投影线性层（无 bias）。
        query: Q 投影线性层（无 bias）。
        value: V 投影线性层（无 bias）。
        tril: 因果 mask buffer（下三角为 1，上三角为 0）。

    提示:
      - 用 register_buffer 注册 tril（不算参数，但随 .to(device) 移动）
      - torch.tril(torch.ones(T, T)) 生成下三角矩阵
      - masked_fill(mask == 0, float('-inf')) 遮罩未来位置
    """

    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        """前向传播。

        Args:
            x: (B, T, n_embed) 输入张量。

        Returns:
            (B, T, head_size) 注意力输出。

        步骤:
            1. 计算 K, Q, V
            2. Q @ K^T / sqrt(d_k) 得到注意力权重
            3. 应用 causal mask
            4. softmax 归一化
            5. 加权 V
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (head_size_dk if False else k.shape[-1]) ** 0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


# ═══════════════════════════════════════════════════════════════════
#  题 2：Pre-LN Transformer Block（基础）
# ═══════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """Pre-LN 残差块：x = x + Attn(LN(x)); x = x + MLP(LN(x))

    Pre-LN vs Post-LN：
      Pre-LN（本题）：先 LayerNorm 再进子层，训练更稳定
      Post-LN（原始论文）：先子层再 LayerNorm，需要 warmup

    Args:
        n_head: 注意力头数。
        n_embed: embedding 维度。
        context_length: 最大序列长度。

    提示:
      - head_size = n_embed // n_head
      - MLP: Linear(n_embed, 4*n_embed) → ReLU → Linear(4*n_embed, n_embed)
      - 残差连接：x = x + sublayer(LN(x))
    """

    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_heads = nn.ModuleList([Head(head_size, n_embed, context_length)
                                       for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
                                 nn.Linear(4 * n_embed, n_embed))

    def forward(self, x):
        """前向传播。

        Args:
            x: (B, T, n_embed) 输入张量。

        Returns:
            (B, T, n_embed) 输出张量（shape 与输入一致）。
        """
        x = x + self.proj(torch.cat([h(self.ln1(x)) for h in self.sa_heads], dim=-1))
        x = x + self.mlp(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════
#  题 3：Prompt-Masked SFT Loss（基础）
# ═══════════════════════════════════════════════════════════════════

def sft_loss(logits, tokens, loss_mask):
    """带 prompt 遮罩的 SFT 损失。

    普通预训练在所有 token 上算 loss，但 SFT 只应在 assistant 回复上训练，
    不应学习"复述提示词"。

    Args:
        logits: (B, T, V) 模型输出 logits。
        tokens: (B, T) token id 序列。
        loss_mask: (B, T) 遮罩，1 表示 assistant token（需要计算 loss），
                   0 表示 prompt token（忽略）。

    Returns:
        标量 loss。

    公式:
        L = sum(CE(logits[:, :-1], tokens[:, 1:]) * mask[:, 1:]) / sum(mask[:, 1:])

    提示:
      - 需要 shift：logits[:, :-1] 预测 tokens[:, 1:]
      - mask 也要相应 shift
      - 用 F.cross_entropy(..., reduction='none') 得到逐 token 的 loss
      - 最终除以 mask 中 1 的数量（clamp 防除零）
    """
    shift_logits = logits[:, :-1, :]
    shift_tokens = tokens[:, 1:]
    shift_mask = loss_mask[:, 1:]
    ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]),
                         shift_tokens.reshape(-1), reduction='none').view(shift_tokens.shape)
    return (ce * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)


# ═══════════════════════════════════════════════════════════════════
#  题 4：Bradley-Terry Reward Loss（基础）
# ═══════════════════════════════════════════════════════════════════

def reward_loss(r_chosen, r_rejected):
    """Bradley-Terry 偏好模型的奖励损失。

    假设人类偏好遵循 P(A > B) = sigmoid(r(A) - r(B))，
    最大化似然等价于最小化 -log sigmoid(r_chosen - r_rejected)。

    Args:
        r_chosen: (B,) chosen 回答的奖励分数。
        r_rejected: (B,) rejected 回答的奖励分数。

    Returns:
        标量 loss。

    公式:
        L = -log sigmoid(r_chosen - r_rejected).mean()

    提示:
      - F.logsigmoid(x) 数值稳定
      - 当 r_chosen = r_rejected 时，loss = ln(2) ≈ 0.693
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


# ═══════════════════════════════════════════════════════════════════
#  题 5：DPO Loss（基础）
# ═══════════════════════════════════════════════════════════════════

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO（Direct Preference Optimization）损失。

    核心洞察：把 Bradley-Terry 模型中的 reward 用 policy log-prob 表示后，
    reward function 被消掉，得到只依赖 policy 和 reference 的闭式解。

    Args:
        policy_chosen_logps: (B,) policy 对 chosen 的 log-prob（已求和）。
        policy_rejected_logps: (B,) policy 对 rejected 的 log-prob。
        ref_chosen_logps: (B,) reference 对 chosen 的 log-prob。
        ref_rejected_logps: (B,) reference 对 rejected 的 log-prob。
        beta: 温度参数，控制偏离 reference 的惩罚强度。

    Returns:
        (loss, chosen_reward, rejected_reward)

    公式:
        logits = (pi_ch - pi_rej) - (ref_ch - ref_rej)
        loss = -logsigmoid(beta * logits).mean()
        chosen_reward = beta * (pi_ch - ref_ch)
        rejected_reward = beta * (pi_rej - ref_rej)

    提示:
      - 当 pi == ref（policy 还没训练）时，logits = 0，loss = ln(2)
      - chosen_reward 和 rejected_reward 是 detached 的诊断指标
    """
    logits = (policy_chosen_logps - ref_chosen_logps) - \
             (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(beta * logits).mean()
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward


# ═══════════════════════════════════════════════════════════════════
#  题 6：GAE Advantage Estimation（🌟 拓展）
# ═══════════════════════════════════════════════════════════════════

def gae(rewards, values, gamma=1.0, lam=0.95):
    """GAE（Generalized Advantage Estimation）。

    在 TD error 的低方差和 Monte Carlo 的低 bias 之间折中。
    λ=0 退化为单步 TD error，λ=1 退化为 Monte Carlo。

    Args:
        rewards: (B, T) 每步奖励。
        values: (B, T+1) 每步状态价值（包含 bootstrap 的 V(s_{T+1})）。
        gamma: 折扣因子。
        lam: GAE lambda（bias-variance 折中参数）。

    Returns:
        (B, T) advantage 估计。

    公式:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + γ * λ * A_{t+1}

    提示:
      - 从后往前递推（reversed(range(T))）
      - 用 lastgae 变量累积
      - λ=0 时 A_t = δ_t（单步 TD error）
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, 1)
    for t in reversed(range(T)):
        delta = rewards[:, t:t + 1] + gamma * values[:, t + 1:t + 2] - values[:, t:t + 1]
        lastgae = delta + gamma * lam * lastgae
        advantages[:, t:t + 1] = lastgae
    return advantages


# ═══════════════════════════════════════════════════════════════════
#  题 7：PPO Clipped Loss（🌟 拓展）
# ═══════════════════════════════════════════════════════════════════

def ppo_loss(logp_new, logp_old, advantages, eps=0.2):
    """PPO 裁剪策略损失。

    用 ratio 裁剪防止策略更新幅度过大——这是 PPO 的核心创新。

    Args:
        logp_new: (B, T) 新策略的 log-prob。
        logp_old: (B, T) 旧策略的 log-prob。
        advantages: (B, T) 优势估计。
        eps: 裁剪范围 ε，ratio 被限制在 [1-ε, 1+ε]。

    Returns:
        标量 loss。

    公式:
        ratio = exp(logp_new - logp_old)
        surr1 = ratio * advantages
        surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
        loss = -mean(min(surr1, surr2))

    提示:
      - ratio 在 [1-ε, 1+ε] 内时，surr2 = surr1，loss 不变
      - ratio 超出范围时被裁剪，防止过大更新
      - logp_new == logp_old 时，ratio=1，loss = -mean(advantages)
    """
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(surr1, surr2).mean()


# ═══════════════════════════════════════════════════════════════════
#  题 8：GRPO Group Advantage（🌟 拓展）
# ═══════════════════════════════════════════════════════════════════

def group_advantages(rewards, group_size, eps=1e-4):
    """GRPO 组相对优势。

    用同一 prompt 下多个回答的组内均值和标准差来标准化奖励。
    不需要价值网络——baseline 就是组均值。

    Args:
        rewards: (num_prompts * group_size,) 所有回答的奖励，按组连续排列。
                 例如 [p0_a0, p0_a1, p0_a2, p1_a0, p1_a1, p1_a2, ...]
        group_size: 每个 prompt 采样的回答数 G。
        eps: 防除零的小常数。

    Returns:
        (num_prompts * group_size,) 组相对优势。

    公式:
        reshape 成 (num_prompts, group_size)
        adv = (r - group_mean) / (group_std + eps)
        reshape 回一维

    提示:
      - 每个 prompt 的 G 个回答共享同一个 baseline（组均值）
      - 组内标准差为 0 时（所有回答奖励相同），advantage 全为 0
    """
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    adv = (r - mean) / (std + eps)
    return adv.view(-1)
