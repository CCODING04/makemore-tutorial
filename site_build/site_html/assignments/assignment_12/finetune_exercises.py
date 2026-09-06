#!/usr/bin/env python3
"""
Part 12 作业：微调实战（LLaMA-Factory）

四道核心题（纯数学/纯 Python）：LoRA 参数账 / 合并数学 / B 零初始化 / QLoRA 显存账，
外加一道 🌟 stretch（多 rank 对比实验，torch 实现，可选——未实现返回 None 时测试 SKIP ⏭️）。
观测型实验题（跑 02 章工具链）见 assignment.md。
实现后运行 test_finetune_exercises.py 验证。
"""

import math

try:
    import torch
except ImportError:
    torch = None


# ── 题 1：LoRA 参数账（25 分）──────────────────────────────
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


# ── 题 2：合并数学（25 分）──────────────────────────────────
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


# ── 题 3：B 零初始化的意义（25 分）──────────────────────────
def initial_delta_norm(A, B):
    """训练起点 ΔW = B@A 的 Frobenius 范数。B 零初始化时它应该是多少？

    Returns:
        float（B 为全零时 = 0.0 —— "起点无损"的数学表述）
    """
    # TODO: (B @ A) 的 frobenius 范数；torch 没有就手算平方和开根
    return None


# ── 题 4：显存账（25 分）────────────────────────────────────
def qlora_vram_gb(base_params_billion, quant_bits=4, lora_params=20_000_000):
    """QLoRA 训练显存的粗估（GB）：
        底座：base_params × quant_bits/8 字节（量化存储）
        可训练：lora_params × 12 字节（参数+梯度+AdamW，fp32 口径，见 Part 10）
    Returns:
        float
    """
    # TODO: 两项相加 / 1e9
    return None


# ── 题 5：🌟 多 rank 对比实验（stretch，加分 20 分，不计入 100 基础分）──
def lora_rank_sweep(out_f=32, in_f=16, ranks=(1, 2, 4, 8), steps=300, lr=5e-2,
                    alpha=8.0, seed=0):
    """🌟 stretch：rank 买到的"表达能力"到底值不值——多 rank 对比实验。

    场景：底座 W 冻结，真实的权重更新是秩 4 的固定矩阵 ΔW* = B*@A*。
    对每个 r ∈ ranks：注入 A(高斯)/B(零) 旁路，Adam 只训 [A, B]，
    记录起点/终点 loss——r ≥ 4 能完整表达 ΔW*（loss → 0），
    r < 4 只能学到它的最优低秩近似（loss 卡在更高的平台）。

    Args:
        out_f, in_f: 玩具 Linear 的输出/输入维度
        ranks: 要对比的 rank 列表
        steps / lr / alpha / seed: 每个 rank 的训练配置（Adam，固定 seed 可复现）

    Returns:
        dict: {r: {"params": int, "loss_start": float, "loss_end": float}}
        未实现返回 None（测试将 SKIP ⏭️，不扣分）

    Steps:
        1. 固定 seed 构造：W(out_f,in_f)、秩 4 目标 Bt(out_f,4)/At(4,in_f)、
           输入 X(128,in_f)、目标 Y = X@W.T + (X@At.T)@Bt.T
        2. 对每个 r：A ~ N(0,1)/√r 形状 (r,in_f)、B = 0 形状 (out_f,r)，
           torch.optim.Adam([A, B], lr=lr)
        3. 每步前向 Yh = X@W.T + (alpha/r)·(X@A.T)@B.T，损失 F.mse_loss(Yh, Y)
        4. 记录第 1 步 / 最后一步的 loss 和 params = r·(out_f+in_f)

    Hint:
        B 零初始化 ⇒ 起点 Yh = X@W.T 与 r 无关，各 rank 的 loss_start 应完全相同。
        需要 torch；函数内 import 即可（骨架顶部已有 try/except 兜底）。

    Acceptance Criteria:
        - 返回 dict 的 keys 恰为 ranks
        - params[r] == r·(out_f+in_f)（精确整数）
        - 各 r 的 loss_start 相同（B=0 ⇒ 起点即纯底座输出）
        - 每个 r 的 loss_end < loss_start
        - r=8 的 loss_end ≤ r=1 的 loss_end（rank 越大表达力越强）
        - r=4（≥ 目标秩）的 loss_end < 0.1 × loss_start
    """
    # TODO: stretch（可选）
    return None
