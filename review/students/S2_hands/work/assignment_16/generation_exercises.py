#!/usr/bin/env python3
"""Assignment 16 作业实现（S2）：图像/视频生成（扩散数学 + 对齐机制）。
四题全部纯 CPU 可完成；🌟 题 4 为选做 stretch（未实现返回 None 时测试优雅 SKIP）。
实现后运行：python test_generation_exercises.py"""

import math

import torch
import torch.nn.functional as F


# ── 题 1：DDPM 前向闭式（30 分）────────────────────────────
def q_sample(x0, alphas_cumprod, t, noise):
    """前向闭式：x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·noise（DDPM 式 4）。
    Args:
        x0: (B, D)；alphas_cumprod: (T,)；t: (B,) 长整型；noise: (B, D)
    """
    s = alphas_cumprod[t].view(-1, 1)          # 按行广播
    return s.sqrt() * x0 + (1 - s).sqrt() * noise


def signal_ratio(t, betas):
    """返回 t 时刻的信号保留比例 √ᾱ_t（标量）——β 线性 schedule。"""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod[t].sqrt().item()


# ── 题 2：CFG 公式（25 分）──────────────────────────────────
def cfg(eps_uncond, eps_cond, w):
    """无分类器引导：eps = uncond + w·(cond − uncond)。"""
    return eps_uncond + w * (eps_cond - eps_uncond)


# ── 题 3：img2img 的 strength→起始步（25 分）────────────────
def img2img_start_step(strength, num_inference_steps):
    """strength ∈ (0,1] → 起始步 t₀ = floor(steps × strength)。
    strength=1 → t₀ = steps（纯文生图）；→0 几乎照抄参考图。
    Returns:
        int
    """
    return int(num_inference_steps * strength)


# ── 题 4（🌟 Stretch，选做）：IP-Adapter 解耦交叉注意力（20 分）──
def decoupled_cross_attn(Q, K_txt, V_txt, K_ref, V_ref, scale):
    """IP-Adapter 解耦注入：out = attn(Q,K_txt,V_txt) + scale·attn(Q,K_ref,V_ref)
    其中 attn(X,K,V) = softmax(X@K.T/√d)@V。
    Args:
        Q: (B, Tq, D)；K_txt/V_txt: (B, Tt, D)；K_ref/V_ref: (B, Tr, D)
    Returns:
        (B, Tq, D)；🌟 未实现保持 return None → 测试优雅 SKIP，不判 FAIL
    """
    d = Q.shape[-1]
    attn_txt = F.softmax(Q @ K_txt.transpose(-2, -1) / d ** 0.5, dim=-1) @ V_txt
    attn_ref = F.softmax(Q @ K_ref.transpose(-2, -1) / d ** 0.5, dim=-1) @ V_ref
    return attn_txt + scale * attn_ref
