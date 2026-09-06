#!/usr/bin/env python3
"""
Part 7 - 脚本 2: RMSNorm 与 RoPE（旋转位置编码）
目标：实现现代 LLM 取代 LayerNorm / 可学习位置编码的两个基础组件，
并从数学上验证它们的性质：
  - RMSNorm：无需均值中心化、无 bias，输出均方根 ≈ weight
  - RoPE：在 q/k 上做"旋转"，是正交变换（保范数），只编码相对位置

覆盖知识点：
  - RMSNorm（Zhang & Sennrich 2019）:
      RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
      对比 LayerNorm：LayerNorm 先减均值再除标准差（把 mean 归 0、std 归 1），
      RMSNorm 只除"均方根"，不做均值中心化，也没有 bias。效果接近但更省。
  - RoPE（Rotary Position Embedding, Su et al. 2021）:
      对 q/k 的每个维度对 (x1,x2) 乘一个角度为 m*theta_i 的旋转矩阵，
      theta_i = 1/(theta^(2i/dim))。频率预先算好存成复数表。
      <q_m, k_n> = <q_0, k_{n-m}> 只依赖相对位置 m-n（绝对位置不产生作用）
  - 与 Part 6 可学习位置编码对比：Part 6 把位置信息"加"进输入 embedding
    （绝对位置），RoPE 把位置信息"转"进 q/k（相对位置），且可外推到更长序列
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 模式选择 ──────────────────────────────────────────────
# CPU 模式: 小模型，<30s 跑完，用于学习验证
# GPU 模式: 完整规模，匹配 minimind 架构，需 GPU
CPU_MODE = not torch.cuda.is_available()
if CPU_MODE:
    vocab_size = 256
    hidden_size = 64
    n_layers = 2
    n_heads = 4
    n_kv_heads = 2
else:
    vocab_size = 6400
    hidden_size = 768
    n_layers = 8
    n_heads = 8
    n_kv_heads = 4
# ─── ───────────────────────────────────────────────────────

torch.manual_seed(1337)


# ─── RMSNorm ───────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    - 无均值中心化（LayerNorm 先减均值）
    - 无 bias（LayerNorm 有 gamma/beta）
    - 只有 1 个可学习缩放 weight，初始为 1
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x^2 的均值 → 开方 → 加 eps → 相除。注意是平方的均值（RMS），不是 std
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def demo_rmsnorm_vs_layernorm():
    print("\n═══ RMSNorm vs LayerNorm ═══")
    dim = 8
    x = torch.randn(4, 5, dim)  # (B, T, C)，每一行是一个 token 的特征

    ln = nn.LayerNorm(dim)
    rms = RMSNorm(dim)
    y_ln, y_rms = ln(x), rms(x)

    print(f"  输入 x: shape={tuple(x.shape)}，每个 token 一行特征")
    print(f"  LayerNorm: mean={y_ln.mean(-1)[0, 0].item():+.4f} (→0), "
          f"std={y_ln.std(-1)[0, 0].item():.4f} (→1), bias=beta 存在")
    print(f"  RMSNorm  : 均方根 rms={torch.sqrt((y_rms ** 2).mean(-1))[0, 0].item():.4f} (→weight=1), "
          f"无 bias")
    # 固定 weight 后，RMSNorm 输出的均方根应精确等于 weight
    rms.weight.data.fill_(2.0)
    out = rms(x)
    print(f"  设 weight=2.0 后，输出均方根 = "
          f"{torch.sqrt((out ** 2).mean(-1))[0, 0].item():.4f} ≈ 2.0 ✅")
    rms.weight.data.fill_(1.0)
    print("  结论：RMSNorm 只归一化“均方根”，不做均值中心化，省掉 beta。"
          "LayerNorm 的 mean≈0 恰好是 RMSNorm 不保证的，但对深层网络影响很小。")


# ─── RoPE ──────────────────────────────────────────────────
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 的复数旋转因子。

    对 dim 维分成 dim/2 个二维组，第 i 组的旋转角频率：
        freq_i = 1 / (theta^(2i/dim))
    位置 m 的旋转角 = m * freq_i。
    返回 (max_seq_len, dim/2) 的复数张量 cis = cos + i*sin。

    torch API 速查：
      torch.outer(a, b)  — 外积，a[i]*b[j] → 矩阵
      torch.polar(abs, angle) — 构造复数 abs * e^{i*angle}，即 abs*(cos(angle) + i*sin(angle))
    """
    assert dim % 2 == 0, "RoPE 需要 dim 为偶数"
    # i = 0, 2, 4, ..., dim-2，对应 freq_i = theta^(-2i/dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    m = torch.arange(max_seq_len)                     # 位置
    angles = torch.outer(m.float(), freqs)            # (max_seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # cos+i*sin
    return freqs_cis


def apply_rotary_pos_emb(q, k, freqs_cis):
    """对 q、k 施加旋转位置编码。

    q, k: (B, T, num_heads, head_dim)
    freqs_cis: (T, head_dim/2) 复数，位置 t 的旋转因子
    把 q/k 的最后一维按 (x0, x1) 两个一组看成复数，乘上旋转因子，再拼回实数。

    torch API 速查：
      torch.view_as_complex(tensor) — 把 (..., 2) 的实数对变成复数，例 [3,4] → 3+4i
      torch.view_as_real(tensor)   — 反向，复数 → (..., 2) 实数对，例 3+4i → [3,4]
      .float()  — bf16 不支持 view_as_complex，需先转 fp32，末尾 .type_as() 还原
    """
    B, T, num_heads, head_dim = q.shape
    half = head_dim // 2
    freqs_cis = freqs_cis[:T].view(T, 1, half)  # (T, 1, half) 广播到各 head
    # (B, T, H, half, 2) → 复数 (B, T, H, half)
    # view_as_complex 不支持 bf16（autocast 下会报错），先转 fp32，末尾 type_as 还原
    q_ = torch.view_as_complex(q.float().reshape(B, T, num_heads, half, 2))
    k_ = torch.view_as_complex(k.float().reshape(B, T, num_heads, half, 2))
    q_out = torch.view_as_real(q_ * freqs_cis).reshape(B, T, num_heads, head_dim)
    k_out = torch.view_as_real(k_ * freqs_cis).reshape(B, T, num_heads, head_dim)
    return q_out.type_as(q), k_out.type_as(k)


def demo_rope():
    print("\n═══ RoPE 演示 ═══")
    B, T, H, head_dim = 1, 8, 4, 8
    theta = 10000.0
    freqs_cis = precompute_freqs_cis(head_dim, T, theta)
    print(f"  构造: B={B}, T={T}, num_heads={H}, head_dim={head_dim}")
    print(f"  precompute_freqs_cis({head_dim}, {T}, theta={theta}) → "
          f"shape {tuple(freqs_cis.shape)}（复数张量）")
    # 展示前几个频率值：低频组 i=0 旋转慢，高频组旋转快
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    print(f"  频率 freq_i = theta^(-2i/dim): {[f'{f:.3f}' for f in freqs.tolist()]}")
    print("  位置 m 的旋转角 = m*freq_i，位置越靠后转的角度越大")

    # 关键：为了展示"只依赖相对位置"，让每个位置都放相同的 token 内容，
    # 这样位置间的差异完全来自 RoPE 的旋转，而不是内容本身。
    x_q = torch.randn(head_dim)                          # 固定的 query 内容
    x_k = torch.randn(head_dim)                          # 固定的 key 内容
    q = x_q.unsqueeze(0).unsqueeze(0).expand(B, T, H, head_dim)
    k = x_k.unsqueeze(0).unsqueeze(0).expand(B, T, H, head_dim)
    q_r, k_r = apply_rotary_pos_emb(q, k, freqs_cis)

    # 1) 保范数：旋转是正交变换
    print("\n  [1] 旋转保范数（正交变换）:")
    n_before = q.norm(dim=-1).mean()
    n_after = q_r.norm(dim=-1).mean()
    print(f"      旋转前 q 范数均值: {n_before.item():.4f}")
    print(f"      旋转后 q 范数均值: {n_after.item():.4f}")
    print(f"      ✅ 范数不变（误差 {abs(n_before - n_after).item():.2e}）"
          f"—— 旋转不改变向量的“大小”，只改变“方向”")

    # 2) 相对位置编码性质：<R(m)x_q, R(n)x_k> = <x_q, R(n-m)x_k> 只依赖相对位置
    w0 = (q_r[:, 0, 0, :] * k_r[:, 3, 0, :]).sum()   # 相对偏移 3
    w1 = (q_r[:, 4, 0, :] * k_r[:, 7, 0, :]).sum()   # 相对偏移 3
    w2 = (q_r[:, 1, 0, :] * k_r[:, 4, 0, :]).sum()   # 相对偏移 3
    w_abs = (q_r[:, 0, 0, :] * k_r[:, 7, 0, :]).sum()  # 相对偏移 7
    print("\n  [2] 相对位置编码性质（内容相同，内积只依赖相对位置）:")
    print(f"      <q@0, k@3> = {w0.item():+.4f}  （相对偏移 3）")
    print(f"      <q@4, k@7> = {w1.item():+.4f}  （相对偏移 3）")
    print(f"      <q@1, k@4> = {w2.item():+.4f}  （相对偏移 3）")
    print(f"      <q@0, k@7> = {w_abs.item():+.4f}  （相对偏移 7）")
    err = abs(w0.item() - w1.item()) + abs(w0.item() - w2.item())
    print(f"      ✅ 相同相对偏移的内积一致（最大偏差 {err:.2e}），"
          f"不同相对偏移则不同 —— 绝对位置不影响注意力，相对位置才影响")

    # 3) 与 Part 6 可学习位置编码对比：加性位置编码"绝对位置"不平移不变
    print("\n  [3] 与 Part 6 可学习位置编码对比:")
    pos_emb = torch.randn(T, head_dim)                    # 模拟 Part 6 的可学习位置向量
    q_add = x_q.unsqueeze(0).repeat(T, 1) + pos_emb       # 内容 + 绝对位置
    k_add = x_k.unsqueeze(0).repeat(T, 1) + pos_emb
    A_add = q_add @ k_add.T                               # (T, T) 内积矩阵
    A_rope = q_r[:, :, 0, :].squeeze(0) @ k_r[:, :, 0, :].squeeze(0).T  # head0
    inv_add = max(abs(A_add[i, j].item() - A_add[i - 1, j - 1].item())
                  for i in range(1, T) for j in range(1, T))
    inv_rope = max(abs(A_rope[i, j].item() - A_rope[i - 1, j - 1].item())
                   for i in range(1, T) for j in range(1, T))
    print(f"      加性位置编码（Part 6）平移前后内积最大偏差: {inv_add:.3f}（不平移不变 ❌）")
    print(f"      RoPE 平移前后内积最大偏差:               {inv_rope:.2e}（平移不变 ✅）")
    print(f"      Part 6 把位置“加”进 embedding（绝对位置，外推差）；"
          f"RoPE 把位置“转”进 q/k（相对位置，可外推）")


def demo_extra_rope():
    # 额外演示：整条序列上的平移不变性——每个位置都是相同的 token 内容，
    # 内积矩阵 A[i,j] 平移后对应元素应完全相等（Toeplitz 结构）。
    print("\n═══ RoPE 平移不变性验证（整条序列） ═══")
    T = 8
    freqs_cis = precompute_freqs_cis(8, T + 4, 10000.0)
    x_q = torch.randn(2, 8)  # H=2 个 head，内容处处相同
    x_k = torch.randn(2, 8)
    q = x_q.unsqueeze(0).unsqueeze(0).expand(1, T, 2, 8)
    k = x_k.unsqueeze(0).unsqueeze(0).expand(1, T, 2, 8)
    qr, kr = apply_rotary_pos_emb(q, k, freqs_cis)
    # 每 head 的 位置×位置 内积矩阵 A[i,j,h] = <q_i,h, k_j,h>
    A = torch.einsum('bihd,bjhd->bijh', qr, kr)[0]  # (T, T, H)
    c = 2  # 平移量
    max_err = 0.0
    # 平移不变：把整条序列整体平移 c 后，相对偏移不变 → A[i,j] 应等于 A[i-c,j-c]
    for i in range(c, T):
        for j in range(c, T):
            for h in range(2):
                max_err = max(max_err, abs(A[i, j, h].item() - A[i - c, j - c, h].item()))
    print(f"  平移 c={c}：比较 A[i,j,h] 与 A[i-c,j-c,h]，最大偏差 = {max_err:.2e}")
    print(f"  ✅ 平移不变性成立 —— 这就是“只依赖相对位置”的严格形式")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    # 本脚本为纯组件演示，只需确认数据文件存在
    if not os.path.exists(data_path):
        print(f"⚠️  数据文件不存在: {data_path}")
        return
    print("═══ 组件演示：RMSNorm + RoPE ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}（hidden={hidden_size}, heads={n_heads}）")

    demo_rmsnorm_vs_layernorm()
    demo_rope()
    demo_extra_rope()

    # 顺带：在随机数据上跑一次 RoPE，确认能放进后面脚本的注意力里
    print("\n═══ RoPE 集成冒烟测试 ═══")
    B, T, H, hd = 2, 6, 4, 8
    q = torch.randn(B, T, H, hd)
    k = torch.randn(B, T, H, hd)
    fc = precompute_freqs_cis(hd, T)
    qr, kr = apply_rotary_pos_emb(q, k, fc)
    attn = torch.softmax((qr @ kr.transpose(-1, -2)) / (hd ** 0.5), dim=-1)
    print(f"  旋转后 attention 权重 shape={tuple(attn.shape)}, "
          f"每行和为 {attn.sum(-1)[0, 0, 0].item():.3f} ✅")
    print("""
═══ 总结 ═══

RMSNorm：x / sqrt(mean(x^2)+eps) * weight，省掉均值中心化和 bias，
         与 LayerNorm 效果相当但实现更简单、计算更省（无减法/无 beta）。
RoPE：   把 q/k 的每个维度对旋转 m*theta_i 角度，正交变换保范数，
         注意力内积只依赖相对位置（平移不变），可外推到更长序列。

下一个脚本：GQA（分组查询注意力）+ KV Cache —— 现代 LLM 推理加速两大件。""")


if __name__ == '__main__':
    main()
