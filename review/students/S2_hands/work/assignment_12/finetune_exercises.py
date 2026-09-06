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
    # Σ r·(out+in)：每层 A 为 r×out、B 为 r×in
    return sum(r * (out_f + in_f) for out_f, in_f in layer_dims)


def lora_ratio(layer_dims, r, base_params):
    """LoRA 可训练参数占"全参"的比例 = lora_params / base_params。"""
    return lora_params(layer_dims, r) / base_params


# ── 题 2：合并数学（25 分）──────────────────────────────────
def merged_weight(W, A, B, alpha, r):
    """LoRA 合并：W' = W + (alpha/r)·B@A（B: out×r, A: r×in）。

    Args:
        W: (out, in) 张量; A: (r, in); B: (out, r)
    Returns:
        合并后的 (out, in) 张量（不要原地修改 W）
    """
    # 不用 += / add_，保持纯函数式，避免原地改 W
    return W + (alpha / r) * (B @ A)


def merge_changes_output(W, A, B, alpha, r, x, tol=1e-6):
    """验证"合并前后前向一致"：y_before = Wx + (alpha/r)·B(Ax)，y_after = W'x。
    Returns:
        bool（max abs 差 < tol）
    """
    y_before = W @ x + (alpha / r) * (B @ (A @ x))
    y_after = merged_weight(W, A, B, alpha, r) @ x
    return bool(torch.allclose(y_before, y_after, atol=tol))


# ── 题 3：B 零初始化的意义（25 分）──────────────────────────
def initial_delta_norm(A, B):
    """训练起点 ΔW = B@A 的 Frobenius 范数。B 零初始化时它应该是多少？

    Returns:
        float（B 为全零时 = 0.0 —— "起点无损"的数学表述）
    """
    delta = B @ A                       # (out, in)，起点 ΔW
    return float(delta.norm(p="fro"))   # Frobenius 范数 = 元素平方和开根


# ── 题 4：显存账（25 分）────────────────────────────────────
def qlora_vram_gb(base_params_billion, quant_bits=4, lora_params=20_000_000):
    """QLoRA 训练显存的粗估（GB）：
        底座：base_params × quant_bits/8 字节（量化存储）
        可训练：lora_params × 12 字节（参数+梯度+AdamW，fp32 口径，见 Part 10）
    Returns:
        float
    """
    base_bytes = base_params_billion * 1e9 * quant_bits / 8   # 4bit → 0.5 B/参数
    train_bytes = lora_params * 12                            # fp32 参数+梯度+两个动量
    return (base_bytes + train_bytes) / 1e9


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
    if torch is None:
        return None
    import torch.nn.functional as F

    # 1. 固定 seed 构造世界：冻结底座 W + 秩 4 的真实更新 ΔW* = Bt@At
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(out_f, in_f, generator=g)
    At = torch.randn(4, in_f, generator=g) * 0.1   # 目标 A: (4, in)
    Bt = torch.randn(out_f, 4, generator=g) * 0.1  # 目标 B: (out, 4)
    X = torch.randn(128, in_f, generator=g)
    Y = X @ W.T + (X @ At.T) @ Bt.T                # 目标输出（含真实低秩更新）
    base_out = X @ W.T                              # 纯底座输出（B=0 时的起点）

    res = {}
    for r in ranks:
        # 2. 注入旁路：A 高斯 / B 零，只训 [A, B]
        A = (torch.randn(r, in_f, generator=g) / math.sqrt(r)).requires_grad_(True)
        B = torch.zeros(out_f, r).requires_grad_(True)
        opt = torch.optim.Adam([A, B], lr=lr)
        loss_start = None
        for step in range(steps):
            opt.zero_grad()
            # 3. 前向 + MSE
            Yh = base_out + (alpha / r) * (X @ A.T) @ B.T
            loss = F.mse_loss(Yh, Y)
            if step == 0:
                loss_start = float(loss.item())   # 第 1 步（更新前）的 loss
            loss.backward()
            opt.step()
        # 4. 记录起点/终点与参数量
        res[r] = {"params": r * (out_f + in_f),
                  "loss_start": loss_start,
                  "loss_end": float(loss.item())}
    return res
