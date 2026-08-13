#!/usr/bin/env python3
"""
Part 7 - 脚本 4: SwiGLU 前馈 + Mixture of Experts（MoE）
目标：实现现代 LLM 的 FFN（SwiGLU 激活）与稀疏专家网络（MoE + 路由 +
负载均衡损失），展示它们与 Part 6 的 ReLU FFN 的区别。

覆盖知识点：
  - SwiGLU（Shazeer 2020）：gate_proj(n→4n) 过 SiLU 当作"门"，与 up_proj(n→4n)
    逐元素相乘，再 down_proj(4n→n) 投影回原维度：
        FFN(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
    对比 Part 6 的 ReLU FFN（Linear→ReLU→Linear）：ReLU 把负值硬截断为 0，
    SiLU 平滑、可微、不硬截断，门控思想更"软"。
  - MoE（Mixture of Experts）：FFN 复制成 E 个 expert，router 输出 E 个概率，
    top-k 路由只激活 k 个 expert（minimind 用 8 experts / top-2，本脚本 4/2）。
    参数量 ×E（更大），但每个 token 只算 k 个 → 稀疏激活，计算量不增太多。
  - 负载均衡损失（Switch Transformer, Fedus et al. 2021）：
        aux_loss = E * sum(f_i * P_i)
        f_i = expert i 被路由到 的 token 频率，P_i = expert i 的平均路由概率
    鼓励所有 expert 均匀分担 token，避免"路由坍缩"到少数几个 expert。
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

# minimind 的 FFN 中间维度公式：把 hidden 放大到 4 倍附近的"64 的倍数"
def calc_intermediate_size(hidden):
    return int((math.pi * hidden / 64) + 0.5) * 64

torch.manual_seed(1337)


# ─── SwiGLU FFN ───────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络：gate 门控 + up 投影 + down 合并。"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))     # 门：决定"放行多少"
        up = self.up_proj(x)                  # 值：要写入的内容
        return self.down_proj(gate * up)     # 门控后投影回原维度


class ReLUFFN(nn.Module):
    """Part 6 的旧式 FFN：Linear→ReLU→Linear，作为对比。"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def demo_swish_vs_relu():
    print("\n═══ Swish(SiLU) vs ReLU 激活对比 ═══")
    xs = torch.linspace(-4, 4, 9)
    print(f"  {'x':>6} {'silu(x)=x·sigmoid(x)':>20} {'relu(x)':>10}")
    for x in xs:
        print(f"  {x.item():>6.1f} {F.silu(x).item():>20.4f} {F.relu(x).item():>10.3f}")
    print("  ReLU: x<0 硬截断为 0（导数在 0 处跳变，负区无梯度）")
    print("  SiLU: 平滑、可微、负区仍有微弱非零梯度（不硬截断，门控更软）")


def demo_ffn_param_compare():
    print("\n═══ SwiGLU FFN vs ReLU FFN（参数量） ═══")
    hidden = 64
    intermediate = calc_intermediate_size(hidden)
    print(f"  hidden={hidden}, intermediate_size={intermediate}（minimind 公式）")
    swiglu_params = sum(p.numel() for p in SwiGLUFFN(hidden, intermediate).parameters())
    relu_params = sum(p.numel() for p in ReLUFFN(hidden, intermediate).parameters())
    print(f"  ReLU  FFN 参数: {relu_params:>6,}  = 2 × n×4n（含 bias）")
    print(f"  SwiGLU FFN 参数: {swiglu_params:>6,}  = 3 × n×4n（无 bias）")
    print(f"  ✅ SwiGLU 比 ReLU 多一个 gate_proj（表达力更强），无 bias 抵消部分开销")


# ─── MoE ──────────────────────────────────────────────────
class MoE(nn.Module):
    """Mixture of Experts：top-k 稀疏路由 + 负载均衡损失。

    - router: Linear(hidden, num_experts) → softmax 得路由概率
    - top-k：每个 token 只激活概率最高的 k 个 expert
    - 输出 = 被选中的 expert 输出 × 对应概率 之和
    - aux_loss（负载均衡）：E * sum(f_i * P_i)，f_i=选中频率，P_i=平均路由概率
    """

    def __init__(self, hidden_size, intermediate_size, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, intermediate_size) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        flat = x.view(N, C)
        # ── 路由：给每个 token 打分 → softmax → top-k 选出专家 ──
        router_logits = self.router(flat)                 # (N, E)
        probs = F.softmax(router_logits, dim=-1)          # (N, E) 路由概率
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)   # 归一化权重
        # ── 每个 expert 处理被路由给它的 token，并按概率加权累加 ──
        out = torch.zeros_like(flat)
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.long)
        for e in range(self.num_experts):
            mask = (topk_idx == e)                        # (N, k) 哪些 slot 选了 e
            rows = mask.any(dim=-1)
            if rows.sum() == 0:
                continue
            w = topk_probs[rows][mask[rows]]              # 权重（每个 token 至多 1 个 slot 选 e）
            out[rows] += self.experts[e](flat[rows]) * w.unsqueeze(-1)
            tokens_per_expert[e] = rows.sum()
        # ── 负载均衡损失：鼓励专家均衡使用 ──
        f_i = tokens_per_expert.float() / N               # 每个 expert 被选中的 token 频率
        p_i = probs.mean(dim=0)                           # 每个 expert 的平均路由概率
        aux_loss = self.num_experts * (f_i * p_i).sum()   # E * sum(f_i * P_i)
        return out.view(B, T, C), aux_loss


def demo_moe():
    print("\n═══ MoE 稀疏路由演示 ═══")
    hidden = 64
    num_experts = 4
    top_k = 2
    moe = MoE(hidden, calc_intermediate_size(hidden), num_experts, top_k)
    B, T = 3, 4
    x = torch.randn(B, T, hidden)

    with torch.no_grad():
        out, aux_loss = moe(x)
        N = B * T
        flat = x.view(N, hidden)
        probs = F.softmax(moe.router(flat), dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)

    print(f"  输入 {B}×{T} = {N} 个 token，num_experts={num_experts}，top_k={top_k}")
    print("\n  每个 token 路由到哪些 expert（Top-2）:")
    names = ["EXPERT_0", "EXPERT_1", "EXPERT_2", "EXPERT_3"]
    for n in range(N):
        s = ", ".join(f"{names[i]}({topk_probs[n, k].item():.2f})"
                      for k, i in enumerate(topk_idx[n].tolist()))
        print(f"    token {n:2d} → {s}")
    counts = [int((topk_idx == e).any(dim=-1).sum()) for e in range(num_experts)]
    print(f"\n  每个 expert 处理的 token 数: {counts}（共 {sum(counts)} = {N}×{top_k} 次路由）")
    active = N * top_k / (N * num_experts)
    print(f"  稀疏度: 每个 token 只激活 {top_k}/{num_experts} = {active:.0%} 的 expert")
    print(f"  ⚠️  但 MoE 的参数量 = {num_experts}×单个 FFN（更大），"
          f"换来的是单次前向算力不变（只算 {top_k} 个）")
    print(f"\n  负载均衡损失 aux_loss = E * sum(f_i * P_i) = {aux_loss.item():.4f}")
    print(f"  💡 若所有 token 都路由到同一个 expert（坍缩），aux_loss 会升高；")
    print(f"     训练时把 aux_loss 加进总 loss，可促使专家均匀分工。")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    if not os.path.exists(data_path):
        print(f"⚠️  数据文件不存在: {data_path}")
        return
    print("═══ SwiGLU + MoE 演示 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}（hidden={hidden_size}, "
          f"intermediate={calc_intermediate_size(hidden_size)}）")

    demo_swish_vs_relu()
    demo_ffn_param_compare()
    demo_moe()

    print("""
═══ 总结 ═══

SwiGLU：gate_proj + up_proj + down_proj 三段式，用 SiLU 当软门控。
         对比 ReLU 硬截断，SwiGLU 平滑可微、负区仍有梯度，已是 LLAMA/Mistral
         等现代模型的标准 FFN（minimind 也用 SwiGLU）。
MoE：    FFN 复制成 E 个专家，top-k 稀疏路由只激活 k 个。
         参数量更大但单次前向计算量不增，负载均衡损失防止专家"旱的旱死涝的涝死"。

下一个脚本：组装完整的 MiniMind 模型（RMSNorm + GQA + SwiGLU + RoPE）。""")


if __name__ == '__main__':
    main()
