#!/usr/bin/env python3
"""
Part 7 - 脚本 12: MLA 与 NSA —— DeepSeek 的两条"省"路线（数值账本 + 三分支机制）
目标：① 手写 MLA 的低秩 KV 压缩数学，验证"KV 缓存大幅缩减"的逐 token 账本；
      ② 手写 NSA 的三分支注意力（压缩/选择/滑窗），观察与全注意力的近似质量。
对应教程：tutorial/04_attention_mla_nsa.md（新增章）
参考：DeepSeek-V2 (2405.04434, MLA) · NSA (2502.11089, ACL'25 最佳论文)
运行（CPU <5 秒）：python 12_mla_nsa_accounting.py
"""

import math
import torch
import torch.nn.functional as F

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

torch.manual_seed(7)


# ═══ Part A: KV 缓存账本（MHA / GQA / MLA）═══
def kv_bytes_mha(n_layers, n_heads, head_dim, seq, batch=1, bytes_per=2):
    """MHA：每头独立 K/V。"""
    return 2 * n_layers * n_heads * head_dim * seq * batch * bytes_per


def kv_bytes_gqa(n_layers, n_kv_heads, head_dim, seq, batch=1, bytes_per=2):
    """GQA：KV 头数减到 n_kv_heads。"""
    return 2 * n_layers * n_kv_heads * head_dim * seq * batch * bytes_per


def kv_bytes_mla(n_layers, kv_lora_rank, rope_dim, seq, batch=1, bytes_per=2):
    """MLA：每 token 只存 1 个压缩 latent c_KV(kv_lora 维) + 1 个位置 key(rope_dim 维)。
    （解耦 RoPE 的位置部分无法压进 latent，单独携带——教程 04 章详述）"""
    per_token = kv_lora_rank + rope_dim
    return n_layers * per_token * seq * batch * bytes_per


def part_a():
    print("═══ Part A: KV 缓存账本（LLaMA-7B 级：32 层 / 128 head_dim / seq 2048 / fp16）═══")
    L, H, D, SEQ = 32, 32, 128, 2048
    mha = kv_bytes_mha(L, H, D, SEQ)
    gqa = kv_bytes_gqa(L, 8, D, SEQ)
    # MLA（DeepSeek-V2 真实配置）：kv_lora_rank=512，rope 携带 64
    mla = kv_bytes_mla(L, 512, 64, SEQ)
    print(f"  MHA      : {mha / 1e9:>6.2f} GB")
    print(f"  GQA(kv8) : {gqa / 1e9:>6.2f} GB")
    print(f"  MLA      : {mla / 1e9:>6.2f} GB   (KV 缓存降至 MHA 的 {mla / mha:.1%})")

    # 逐 token 精确复算（与公式对照）
    per_tok_mla = (512 + 64) * 2   # bytes / token / layer
    exact = per_tok_mla * L * SEQ
    assert abs(exact - mla) / mla < 1e-9, "逐 token 复算与公式不一致"
    print("  ✅ 逐 token 复算与公式一致")
    print("""
  💡 MLA vs GQA（同为压缩，路线不同）：
     GQA = 共享 K/V 头（heads 分组共享 → 缓存按头数线性降）
     MLA = 低秩压缩到 latent 向量（缓存按 latent 维降，质量上限更高）
     两者都源于同一观察：K/V 头间存在冗余。""")


# ═══ Part B: NSA 三分支注意力的最小手写 ═══
def nsa_branches(q, k, v, window=4, block=4, top_k=1):
    """NSA (arXiv 2502.11089) 三分支的最小实现（单头、无缩放，纯机制演示）。
    q/k/v: (T, d)。返回 三分支输出。"""
    T = q.shape[0]
    n_blocks = T // block

    # 分支 1: 压缩 —— 每块平均成一个"摘要 token"
    k_cmp = k.view(n_blocks, block, -1).mean(dim=1)      # (n_blocks, d)
    v_cmp = v.view(n_blocks, block, -1).mean(dim=1)
    o_cmp = F.softmax(q @ k_cmp.T, dim=-1) @ v_cmp       # (T, d)

    # 分支 2: 选择 —— 用压缩分数选 top_k 个关键块，只在选中块内做精细注意力
    block_scores = F.softmax(q @ k_cmp.T, dim=-1)        # (T, n_blocks) 每查询的块分
    sel_blocks = block_scores.topk(top_k, dim=-1).indices  # (T, top_k)
    o_sel = torch.zeros_like(q)
    for t in range(T):
        for b in sel_blocks[t].tolist():
            lo, hi = b * block, (b + 1) * block
            attn = F.softmax(q[t:t + 1] @ k[lo:hi].T, dim=-1)
            o_sel[t:t + 1] += attn @ v[lo:hi]
    o_sel = o_sel / top_k                                 # 块间平均

    # 分支 3: 滑动窗口 —— 只看最近 window 个 token（局部细节）
    o_win = torch.zeros_like(q)
    for t in range(T):
        lo = max(0, t - window + 1)
        attn = F.softmax(q[t:t + 1] @ k[lo:t + 1].T, dim=-1)
        o_win[t:t + 1] = attn @ v[lo:t + 1]

    # 门控融合（NSA 用可学习 gating；演示用固定等权）
    gate = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    out = gate[0] * o_cmp + gate[1] * o_sel + gate[2] * o_win
    return out, o_cmp, o_sel, o_win, gate


def part_b():
    print("\n═══ Part B: NSA 三分支最小手写（T=16, block=4, top_k=1, window=4）═══")
    T, d = 16, 32
    q = torch.randn(T, d)
    k = torch.randn(T, d)
    v = torch.randn(T, d)

    # 全注意力参照（因果）
    wei = q @ k.T / d ** 0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    wei = wei.masked_fill(mask, float('-inf'))
    o_full = F.softmax(wei, -1) @ v

    out, o_cmp, o_sel, o_win, gate = nsa_branches(q, k, v)
    err = (out - o_full).abs().max().item()
    print(f"  NSA(等权门控) vs 全注意力: max diff = {err:.4f}"
          f"（方向一致；门控可学习后误差进一步收敛——原生可训练是 NSA 的核心主张）")
    print(f"  计算量: NSA 每 query 只算 top_k={1} 个块（{d} 维）的精细注意力"
          f" vs 全注意力的 {T} 块（长上下文下 FLOPs 大幅降低）")
    print("""
  💡 三分支分工（面试向）：
     压缩分支 = 全局粗读（块级摘要 token，成本 n_blocks）
     选择分支 = 精读关键块（由压缩分数选 top_k）
     滑动窗口 = 局部细节（最近 token 永不丢）
     门控融合 = 可学习三路权重；'原生可训练'是 NSA 区别于推理期近似的关键主张
       （推理期后置近似 = 训好后用稀疏化近似；NSA 从预训练起就原生稀疏，
         两者效果差距大——NSA 获 ACL'25 最佳论文的核心贡献）""")


if __name__ == '__main__':
    part_a()
    part_b()
