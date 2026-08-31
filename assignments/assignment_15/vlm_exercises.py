#!/usr/bin/env python3
"""Assignment 15 参考答案：多模态理解（VLM）。纯 CPU 可验证。"""

import math

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


# ── 题 1：patch 数量与形状（25 分）──────────────────────────
def patch_tokens(h, w, patch):
    """ViT 切 patch：token 数 = ceil(H/p) × ceil(W/p)（向上取整，边缘补齐）。"""
    # TODO: 两行
    return None


def vit_out_shape(n_tokens, embed_dim, n_layers=1):
    """ViT 不改变 token 数与维度：输出 = (n_tokens, embed_dim)（元组）。"""
    # TODO: 返回 (n_tokens, embed_dim)
    return None


# ── 题 2：InfoNCE 对比损失（30 分）──────────────────────────
def infonce_loss(f_img, f_txt, scale):
    """对称 InfoNCE：logits = scale·f_img@f_txt.T，标签=对角线，
    loss = 0.5*(CE(logits, labels) + CE(logits.T, labels))。"""
    # TODO: 三行（labels 用 torch.arange）
    return None


# ── 题 3：投影器参数量（25 分）──────────────────────────────
def mlp2x_params(vision_dim, llm_dim):
    """LLaVA mlp2x_gelu：Linear(vision→llm) + Linear(llm→llm)，均含 bias。
    Returns:
        int 参数总数"""
    # TODO: 两个 Linear 的 w+b 参数和
    return None


# ── 题 4（🌟）：动态分辨率 token 估算（20 分）───────────────
def dynamic_tokens(h, w, patch=14, compress=4, max_tokens=2560):
    """Qwen-VL 风格：按 patch 网格切分后经压缩率 ÷compress（pixel shuffle 类），
    并做 token 预算控制：超过 max_tokens 时按比例缩小分辨率重算。
    Returns:
        int：最终视觉 token 数（不超过 max_tokens）
    """
    # TODO:
    #   1. raw = ceil(h/patch) * ceil(w/patch)
    #   2. tokens = raw // compress
    #   3. while tokens > max_tokens: 缩小 h/w（各乘 0.8 取整）重算
    return None
