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
    """

    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer(
            'tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        d_k = q.size(-1)
        wei = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, T, head_size)
        return out


# ═══════════════════════════════════════════════════════════════════
#  题 2：Pre-LN Transformer Block（基础）
# ═══════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """Pre-LN 残差块：x = x + Attn(LN(x)); x = x + MLP(LN(x))"""

    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        head_size = n_embed // n_head
        self.ln1 = nn.LayerNorm(n_embed)
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, context_length) for _ in range(n_head)])
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        attn_out = torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════
#  题 3：Prompt-Masked SFT Loss（基础）
# ═══════════════════════════════════════════════════════════════════

def sft_loss(logits, tokens, loss_mask):
    """带 prompt 遮罩的 SFT 损失。

    L = sum(CE(logits[:, :-1], tokens[:, 1:]) * mask[:, 1:]) / sum(mask[:, 1:])
    """
    shift_logits = logits[:, :-1, :]   # (B, T-1, V)
    shift_tokens = tokens[:, 1:]       # (B, T-1)
    shift_mask = loss_mask[:, 1:]      # (B, T-1)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_tokens.reshape(-1),
        reduction='none',
    ).reshape(shift_tokens.shape)
    loss = (ce * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
    return loss


# ═══════════════════════════════════════════════════════════════════
#  题 4：Bradley-Terry Reward Loss（基础）
# ═══════════════════════════════════════════════════════════════════

def reward_loss(r_chosen, r_rejected):
    """Bradley-Terry：L = -log sigmoid(r_chosen - r_rejected).mean()"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


# ═══════════════════════════════════════════════════════════════════
#  题 5：DPO Loss（基础）
# ═══════════════════════════════════════════════════════════════════

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO：logits = (pi_ch - pi_rej) - (ref_ch - ref_rej)"""
    logits = (policy_chosen_logps - policy_rejected_logps) \
        - (ref_chosen_logps - ref_rejected_logps)
    loss = -F.logsigmoid(beta * logits).mean()
    chosen_reward = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_reward = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
    return loss, chosen_reward, rejected_reward


# ═══════════════════════════════════════════════════════════════════
#  题 6：GAE Advantage Estimation（🌟 拓展）
# ═══════════════════════════════════════════════════════════════════

def gae(rewards, values, gamma=1.0, lam=0.95):
    """GAE：δ_t = r_t + γ*V(s_{t+1}) - V(s_t)；A_t = δ_t + γλ*A_{t+1}"""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        lastgae = delta + gamma * lam * lastgae
        adv[:, t] = lastgae
    return adv


# ═══════════════════════════════════════════════════════════════════
#  题 7：PPO Clipped Loss（🌟 拓展）
# ═══════════════════════════════════════════════════════════════════

def ppo_loss(logp_new, logp_old, advantages, eps=0.2):
    """PPO：ratio=exp(logp_new-logp_old); loss=-mean(min(surr1, surr2))"""
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
    loss = -(torch.min(surr1, surr2)).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════
#  题 8：GRPO Group Advantage（🌟 拓展）
# ═══════════════════════════════════════════════════════════════════

def group_advantages(rewards, group_size, eps=1e-4):
    """GRPO：组内标准化 adv = (r - group_mean) / (group_std + eps)"""
    grouped = rewards.view(-1, group_size)                    # (num_prompts, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    adv = (grouped - mean) / (std + eps)
    return adv.view(-1)
