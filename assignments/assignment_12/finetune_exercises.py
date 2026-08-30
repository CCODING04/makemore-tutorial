#!/usr/bin/env python3
"""
Part 12 作业：微调实战（LLaMA-Factory）

三道纯数学/纯 Python 题（LoRA 的账与行为）+ 一道观测题。
实现后运行 test_finetune_exercises.py 验证。
"""

import math

try:
    import torch
except ImportError:
    torch = None


# ── 题 1：LoRA 参数账（30 分）──────────────────────────────
def lora_params(layer_dims, r):
    """计算把 LoRA(r) 注入若干层后的可训练参数量。

    Args:
        layer_dims: list[(out_features, in_features)]，被注入的各 Linear
        r: LoRA 秩
    Returns:
        int：全部 A/B 的参数总数 = Σ r*(out+in)
    """
    # TODO: 一行求和
    return None


def lora_ratio(layer_dims, r, base_params):
    """LoRA 可训练参数占"全参"的比例 = lora_params / base_params。"""
    # TODO: 调用 lora_params，除以 base_params
    return None


# ── 题 2：合并数学（30 分）──────────────────────────────────
def merged_weight(W, A, B, alpha, r):
    """LoRA 合并：W' = W + (alpha/r)·B@A（B: out×r, A: r×in）。

    Args:
        W: (out, in) 张量; A: (r, in); B: (out, r)
    Returns:
        合并后的 (out, in) 张量（不要原地修改 W）
    """
    # TODO: 一行（注意 (alpha/r) 的位置）
    return None


def merge_changes_output(W, A, B, alpha, r, x, tol=1e-6):
    """验证"合并前后前向一致"：y_before = Wx + (alpha/r)·B(Ax)，y_after = W'x。
    Returns:
        bool（max abs 差 < tol）
    """
    # TODO:
    #   1. y_before = W@x + (alpha/r)*(B@(A@x))
    #   2. W2 = merged_weight(...); y_after = W2@x
    #   3. 返回 torch.allclose(y_before, y_after, atol=tol)
    return None


# ── 题 3：B 零初始化的意义（20 分）──────────────────────────
def initial_delta_norm(A, B):
    """训练起点 ΔW = B@A 的 Frobenius 范数。B 零初始化时它应该是多少？

    Returns:
        float（B 为全零时 = 0.0 —— "起点无损"的数学表述）
    """
    # TODO: (B @ A) 的 frobenius 范数；torch 没有就手算平方和开根
    return None


# ── 题 4：显存账（20 分）────────────────────────────────────
def qlora_vram_gb(base_params_billion, quant_bits=4, lora_params=20_000_000):
    """QLoRA 训练显存的粗估（GB）：
        底座：base_params × quant_bits/8 字节（量化存储）
        可训练：lora_params × 12 字节（参数+梯度+AdamW，fp32 口径，见 Part 10）
    Returns:
        float
    """
    # TODO: 两项相加 / 1e9
    return None
