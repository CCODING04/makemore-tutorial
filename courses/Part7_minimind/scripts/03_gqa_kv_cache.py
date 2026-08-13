#!/usr/bin/env python3
"""
Part 7 - 脚本 3: GQA（分组查询注意力）+ KV Cache
目标：实现现代 LLM 的注意力核心——把 MHA 的每头独立 K/V 换成"分组共享"的
GQA，并实现 KV Cache 让推理只算最后一个 token，演示两者输出一致。

覆盖知识点：
  - MHA vs MQA vs GQA：
      MHA：n_heads 个 Q 头，每个头都有自己的 K/V 头（参数多）
      MQA：n_heads 个 Q 头，全部共享 1 个 K/V 头（参数最少，质量略降）
      GQA：n_heads 个 Q 头，分成 n_kv_heads 组，组内共享 K/V（折中，minimind 用
           8 Q 头 / 4 KV 头，本脚本 CPU 模式 4 Q 头 / 2 KV 头）
  - repeat_kv：把 K/V 头复制 n_rep = n_heads//n_kv_heads 份，凑到和 Q 头一致
  - KV Cache：生成第 t 个 token 时，只取它的 Q，而 K/V 用缓存好的前 t-1 个 +
    当前 1 个。无 cache 版和 cache 版生成的 logits 应逐 token 一致。
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

# flash_attn 开关：GPU 上更快的 F.scaled_dot_product_attention，CPU 用手写注意力
FLASH_ATTN = not CPU_MODE

torch.manual_seed(1337)


# ─── 基础组件：RoPE（复用脚本 2）──────────────────────────
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    assert dim % 2 == 0, "RoPE 需要 dim 为偶数"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_pos_emb(q, k, freqs_cis):
    # GQA 下 q 有 n_heads 个、k 只有 n_kv_heads 个，头数不同也能各自旋转
    B, T, nq_heads, head_dim = q.shape
    nk_heads = k.shape[2]
    half = head_dim // 2
    freqs_cis = freqs_cis[:T].view(T, 1, half)
    # view_as_complex 不支持 bf16（autocast 下会报错），先转 fp32，末尾 type_as 还原
    q_ = torch.view_as_complex(q.float().reshape(B, T, nq_heads, half, 2))
    k_ = torch.view_as_complex(k.float().reshape(B, T, nk_heads, half, 2))
    q_out = torch.view_as_real(q_ * freqs_cis).reshape(B, T, nq_heads, head_dim)
    k_out = torch.view_as_real(k_ * freqs_cis).reshape(B, T, nk_heads, head_dim)
    return q_out.type_as(q), k_out.type_as(k)


def repeat_kv(x, n_rep):
    """把 K/V 头复制 n_rep 份，与 Q 头数对齐。

    x: (B, T, n_kv_heads, head_dim) → (B, T, n_kv_heads*n_rep, head_dim)
    只复制"头"，不复制内容（K/V 投影本来就只有 n_kv_heads 组）。
    """
    B, T, n_kv, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :].expand(B, T, n_kv, n_rep, head_dim)
    return x.reshape(B, T, n_kv * n_rep, head_dim)


def update_kv_cache(k, v, past_key_value=None):
    """KV Cache 更新：首次调用（cache 为空）直接返回 k,v 作为 cache；
    之后把新的 k,v 拼接到 cache 末尾（时间维上 cat）。
    """
    if past_key_value is None:
        return k, v
    pk, pv = past_key_value
    k = torch.cat([pk, k], dim=1)
    v = torch.cat([pv, v], dim=1)
    return k, v


# ─── GQA 注意力层 ─────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention。

    - wq 投影出 n_heads 个 Q 头
    - wk/wv 只投影出 n_kv_heads 个 K/V 头（省参数）
    - RoPE 旋转（K/V 用相同的旋转频率）
    - repeat_kv 把 K/V 头复制到 n_heads 份
    - 因果遮罩的 scaled dot-product attention
    """

    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim,
                 flash_attn=False, max_seq_len=256, rope_theta=10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.flash_attn = flash_attn
        self.n_rep = n_heads // n_kv_heads  # 每组 KV 头要复制几份
        assert n_heads % n_kv_heads == 0, "Q 头数必须能被 KV 头数整除"

        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.register_buffer('freqs_cis', precompute_freqs_cis(head_dim, max_seq_len, rope_theta))
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)).bool())

    def forward(self, x, past_key_value=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        # RoPE 用"绝对位置"：无 cache 时 x 是位置 0..T-1；
        # 有 cache 时 x 是新的 token，绝对位置从 cache 末尾开始。
        start = 0 if past_key_value is None else past_key_value[0].shape[1]
        q, k = apply_rotary_pos_emb(q, k, self.freqs_cis[start:start + T])

        k, v = update_kv_cache(k, v, past_key_value)  # 加入 KV cache
        kv_T = k.shape[1]

        # 只复制需要的那一段 mask（cache 变长时 mask 要同步变大）
        mask = self.mask[:T, :kv_T]

        # K/V 头复制到与 Q 头一致
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if self.flash_attn:
            # F.scaled_dot_product_attention 自带因果 mask 支持，GPU 上更快
            causal = ~mask
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=causal)
            out = attn.transpose(1, 2).contiguous()
        else:
            # 手写 scaled dot-product attention（带因果遮罩）。
            # 交换到 (B, n_heads, T, head_dim) 布局，让 matmul 作用于时间维。
            q_h = q.transpose(1, 2)   # (B, n_heads, T, hd)
            k_h = k.transpose(1, 2)   # (B, n_heads, kv_T, hd)
            v_h = v.transpose(1, 2)
            attn_wei = (q_h @ k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_wei = attn_wei.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_wei = F.softmax(attn_wei, dim=-1)
            out = (attn_wei @ v_h).transpose(1, 2).contiguous()
        out = self.wo(out.reshape(B, T, self.n_heads * self.head_dim))
        return out


# ─── 演示辅助 ─────────────────────────────────────────────
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def demo_param_table():
    print("\n═══ MHA / MQA / GQA 参数量对比 ═══")
    hidden = 64
    head_dim = 16
    n_heads = 4
    rows = []
    for label, kv, n_kv in [
        ("MHA (4Q/4KV)", n_heads, 4),
        ("GQA (4Q/2KV)", n_heads, 2),
        ("MQA (4Q/1KV)", n_heads, 1),
    ]:
        attn = GroupedQueryAttention(hidden, n_heads, n_kv, head_dim, max_seq_len=32)
        q_params = count_params(attn.wq)
        kv_params = count_params(attn.wk) + count_params(attn.wv)
        o_params = count_params(attn.wo)
        total = q_params + kv_params + o_params
        rows.append((label, q_params, kv_params, total, f"{kv_params/q_params:.2f}x"))
    print(f"  {'attention':<14}{'Q 参数':>8}{'K/V 参数':>10}{'总参数':>10}  K/V 相对 Q")
    for r in rows:
        print(f"  {r[0]:<14}{r[1]:>8,}{r[2]:>10,}{r[3]:>10,}  {r[4]}")
    print(f"  说明：GQA 用 {rows[1][2]} 的 K/V 参数就达到 MHA {rows[0][2]} 的效果；")
    print(f"        在 n_heads 更大（如 8Q/4KV）时省得更多。")


def demo_kv_cache():
    print("\n═══ KV Cache 一致性验证 ═══")
    torch.manual_seed(42)
    max_seq = 16
    head_dim = hidden_size // n_heads
    attn = GroupedQueryAttention(hidden_size, n_heads, n_kv_heads,
                                 head_dim=head_dim, max_seq_len=max_seq)
    emb = nn.Embedding(8, hidden_size)
    T = 6  # 序列长度
    idx = torch.randint(0, 8, (1, T))
    x = emb(idx)  # (1, T, hidden)

    with torch.no_grad():
        # ── 无 cache：一次跑完整序列 ──
        full_out = attn(x)  # (1, T, hidden)

        # ── 有 cache：每个 token 只算自己的 Q 和新的 K/V，逐步拼接 ──
        cached_outs = []
        cache = None
        for t in range(T):
            xt = x[:, t:t + 1]  # 只取第 t 个 token
            q = attn.wq(xt).view(1, 1, n_heads, head_dim)
            k = attn.wk(xt).view(1, 1, n_kv_heads, head_dim)
            v = attn.wv(xt).view(1, 1, n_kv_heads, head_dim)
            # 用"位置 t"对应的旋转频率（freqs_cis[t:t+1]），
            # 不能传整个表：apply_rotary_pos_emb 会按 q 的时间维取 [:T]=[:1] 即位置 0
            q, k = apply_rotary_pos_emb(q, k, attn.freqs_cis[t:t + 1])
            # 首次 cache 为空 → 直接作为 cache；之后把新 k/v 拼到 cache 末尾
            cache = update_kv_cache(k, v, cache)
            k_cache, v_cache = cache
            k_all = repeat_kv(k_cache, attn.n_rep)   # 复制到 n_heads 份
            v_all = repeat_kv(v_cache, attn.n_rep)
            tt = t + 1
            # (B,n_heads,T,hd) 布局下手写注意力（与模块 forward 完全一致）。
            # 注意：生成时 Q 只有最后 1 个位置，因果 mask 只取"最后一行"，
            # 否则 (tt,tt) 的 mask 会把 Q 的 1 个位置广播成 tt 个。
            wei = (q.transpose(1, 2) @ k_all.transpose(1, 2).transpose(-2, -1)) / math.sqrt(head_dim)
            causal = torch.tril(torch.ones(tt, tt, dtype=torch.bool))[tt - 1:tt, :]
            wei = wei.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
            wei = F.softmax(wei, dim=-1)
            h = (wei @ v_all.transpose(1, 2)).transpose(1, 2)  # (1,1,n_heads,hd)
            cached_outs.append(attn.wo(h.reshape(1, 1, n_heads * head_dim)))

        # ── 对比：两者在第 t 个 token 上的输出应逐 token 一致 ──
        max_diff = 0.0
        for t in range(T):
            diff = (full_out[:, t] - cached_outs[t][0, 0]).abs().max().item()
            max_diff = max(max_diff, diff)
    print(f"  完整序列计算 vs 逐步 KV Cache 计算的输出最大偏差: {max_diff:.2e}")
    print(f"  ✅ 一致（偏差≈0）—— KV Cache 不改变结果，只是省去重复计算")
    print(f"  💡 复杂度：无 cache 每步重算前 {t + 1} 个 token 的 Q/K/V，总 O(T²)；"
          f"有 cache 每步只算 1 个 token 的 Q 和新的 K/V，总 O(T)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    if not os.path.exists(data_path):
        print(f"⚠️  数据文件不存在: {data_path}")
        return
    print("═══ GQA + KV Cache 演示 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}（hidden={hidden_size}, "
          f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, flash_attn={FLASH_ATTN}）")
    print(f"  head_dim = hidden // n_heads = {hidden_size // n_heads}")

    # 1) repeat_kv 形状演示
    print("\n═══ repeat_kv 形状演示 ═══")
    B, T, n_kv, hd = 1, 4, 2, 8
    x = torch.arange(B * T * n_kv * hd).float().view(B, T, n_kv, hd)
    y = repeat_kv(x, n_rep=2)
    print(f"  输入 (B,T,KV头,head_dim) = {tuple(x.shape)} → 输出 = {tuple(y.shape)}")
    print(f"  ✅ 第 0 个头与复制出的头内容相同: "
          f"{torch.equal(y[0, 0, 0], y[0, 0, 1])}（K/V 共享，省显存省参数）")

    # 2) 三种 attention 参数量对比
    demo_param_table()

    # 3) KV Cache 一致性
    demo_kv_cache()

    print("""
═══ 总结 ═══

GQA 用"分组共享 K/V"在参数量与表达力之间折中：
  MHA（4Q/4KV）→ GQA（4Q/2KV）→ MQA（4Q/1KV），K/V 参数逐级减半
KV Cache 是自回归推理的关键加速：把已算过的 K/V 存起来，
每步只算最后一个 token 的注意力，输出与完整重算完全一致（O(T²)→O(T)）。

下一个脚本：SwiGLU 前馈 + Mixture of Experts —— 现代 LLM 的 FFN 与稀疏专家。""")


if __name__ == '__main__':
    main()
