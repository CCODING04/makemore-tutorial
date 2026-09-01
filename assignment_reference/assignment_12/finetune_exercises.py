import math


def lora_params(layer_dims, r): return sum(r * (o + i) for o, i in layer_dims)
def lora_ratio(layer_dims, r, base_params): return lora_params(layer_dims, r) / base_params


def merged_weight(W, A, B, alpha, r): return W + (alpha / r) * (B @ A)


def merge_changes_output(W, A, B, alpha, r, x, tol=1e-6):
    import torch
    y0 = W @ x + (alpha / r) * (B @ (A @ x)); W2 = merged_weight(W, A, B, alpha, r)
    return torch.allclose(y0, W2 @ x, atol=tol)


def initial_delta_norm(A, B): return float((B @ A).pow(2).sum().sqrt())


def qlora_vram_gb(base_params_billion, quant_bits=4, lora_params=20_000_000):
    return (base_params_billion * 1e9 * quant_bits / 8 + lora_params * 12) / 1e9


# ── 题 5：🌟 多 rank 对比实验（stretch）──────────────────────────
def lora_rank_sweep(out_f=32, in_f=16, ranks=(1, 2, 4, 8), steps=300, lr=5e-2,
                    alpha=8.0, seed=0):
    """🌟 stretch 参考实现：底座 W 冻结，目标更新是秩 4 的 ΔW*=Bt@At；
    对每个 r 注入 A(高斯)/B(零) 旁路，Adam 只训 A/B，对比起点/终点 loss。"""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    W = torch.randn(out_f, in_f)                      # 冻结底座 (32, 16)
    Bt = torch.randn(out_f, 4) * 0.1                  # 秩 4 目标 ΔW* = Bt@At
    At = torch.randn(4, in_f) * 0.1
    X = torch.randn(128, in_f)                        # 输入 (128, 16)
    Y = X @ W.T + (X @ At.T) @ Bt.T                   # 目标 = (W + ΔW*)x, (128, 32)

    results = {}
    for r in ranks:
        torch.manual_seed(seed + 1)                   # 各 rank 同起点分布
        A = (torch.randn(r, in_f) / math.sqrt(r)).requires_grad_(True)   # (r, 16)
        B = torch.zeros(out_f, r).requires_grad_(True)                    # (32, r)
        opt = torch.optim.Adam([A, B], lr=lr)
        loss_start = loss_end = None
        for step in range(steps):
            Yh = X @ W.T + (alpha / r) * (X @ A.T) @ B.T                  # (128, 32)
            loss = F.mse_loss(Yh, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 0:
                loss_start = loss.item()
            loss_end = loss.item()
        results[r] = {"params": r * (out_f + in_f),
                      "loss_start": loss_start, "loss_end": loss_end}
    return results
