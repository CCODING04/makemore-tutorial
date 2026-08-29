#!/usr/bin/env python3
"""
Part 8 - 脚本 1: 从零构建 GPT-2（经典 Transformer）
目标：用纯 PyTorch 实现一个完整的 decoder-only Transformer，
架构对齐 train-llm-from-scratch 仓库：LayerNorm + learned PE + MHA + ReLU FFN。

覆盖知识点：
  - Pre-LN Transformer Block（先归一化再进子层，比 Post-LN 更稳定）
  - 单头因果注意力 Head：Q/K/V 投影 + scaled dot-product + causal mask
  - 多头并行 MultiHeadAttention：拼接 + 输出投影
  - MLP 前馈：4x 扩展 + ReLU + 投影回（不用 SwiGLU，这是经典款）
  - token embedding + learned absolute position embedding（不用 RoPE）
  - forward_hidden()：返回 final LN 之后、lm_head 之前的 hidden state
    （供奖励模型 / 价值头使用——后训练的关键 hook 点）
  - 参数量计算

架构对比：
  Part 6:  教学版（字符级 tokenizer，小模型）
  Part 7:  minimind（RMSNorm + RoPE + GQA + SwiGLU）
  Part 8:  GPT-2 经典款（LayerNorm + learned PE + MHA + ReLU）← 本脚本
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 模式选择 ──────────────────────────────────────────────
CPU_MODE = not torch.cuda.is_available()
if CPU_MODE:
    vocab_size = 256
    n_embed = 64
    n_head = 4
    n_blocks = 2
    context_length = 64
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
# ───────────────────────────────────────────────────────────

torch.manual_seed(1337)


# ─── 单头因果注意力 ────────────────────────────────────────
class Head(nn.Module):
    """单头因果注意力。

    Q/K 各自投影后算内积 → scale → causal mask（上三角 -inf）→ softmax → 加权 V。

    torch API 速查：
      nn.Linear(in, out, bias=False) — 线性投影（无 bias 是现代惯例）
      register_buffer('name', tensor) — 注册为 buffer（不算参数，但会随 model.to(device) 移动）
      torch.tril(torch.ones(T,T)) — 下三角矩阵，用于 causal mask
      masked_fill(mask==0, -inf) — 把 mask 为 0 的位置填 -inf（softmax 后变 0）
    """

    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)       # (B, T, head_size)
        q = self.query(x)     # (B, T, head_size)
        head_size = k.shape[-1]
        # scaled dot-product: Q @ K^T / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)     # (B, T, head_size)
        return wei @ v        # (B, T, head_size)


# ─── 多头注意力 ────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """多个 Head 并行 → 拼接 → 输出投影。

    n_head 个 Head 各自独立算注意力（不同子空间），
    拼接后用 proj 投影回 n_embed 维（为残差连接做准备）。
    """

    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(head_size, n_embed, context_length)
                                    for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(x)


# ─── MLP 前馈网络 ──────────────────────────────────────────
class MLP(nn.Module):
    """经典 FFN：4x 扩展 + ReLU + 投影回。

    与 Part 7 的 SwiGLU 对比：这里用 ReLU（硬截断）+ 单层扩展，
    SwiGLU 用 SiLU（软门控）+ 双分支 gate/up。经典款更简单但表达力稍弱。
    """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)


# ─── Transformer Block ─────────────────────────────────────
class Block(nn.Module):
    """Pre-LN 残差块：x = x + Attn(LN(x)); x = x + MLP(LN(x))

    Pre-LN vs Post-LN：
      Pre-LN（本脚本）：先 LayerNorm 再进子层，训练更稳定，是 GPT-2/Llama 的默认选择
      Post-LN（原始论文）：先子层再 LayerNorm，需要 warmup，否则容易训崩
    """

    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─── 完整 GPT-2 ────────────────────────────────────────────
class GPT(nn.Module):
    """Decoder-only Transformer（对齐 train-llm-from-scratch 仓库的架构）。

    组件：
      token_embed + position_embed（learned，相加）→ N × Block → LayerNorm → lm_head

    forward_hidden()：返回 LN 之后、lm_head 之前的 hidden state。
    这是后训练的关键 hook——奖励模型和价值头都在这里接入。
    """

    def __init__(self, n_head, n_embed, context_length, vocab_size, n_blocks):
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.blocks = nn.ModuleList([Block(n_head, n_embed, context_length)
                                     for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def forward_hidden(self, idx):
        """backbone 前向：返回 final LN 之后的 hidden state (B, T, n_embed)。
        后训练的奖励头/价值头接入点。"""
        B, T = idx.shape
        tok_emb = self.token_embed(idx)                        # (B, T, n_embed)
        pos_emb = self.position_embed(self.pos_idxs[:T])      # (T, n_embed)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx, targets=None):
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)    # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("═══ 从零构建 GPT-2 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}")
    print(f"  架构: embed={n_embed}, heads={n_head}, blocks={n_blocks}, "
          f"vocab={vocab_size}, ctx={context_length}")

    model = GPT(n_head, n_embed, context_length, vocab_size, n_blocks)
    n_params = count_params(model)
    print(f"\n═══ 参数量 ═══")
    print(f"  总参数: {n_params:,} = {n_params / 1e6:.2f}M")

    # 参数量分解
    tok_params = vocab_size * n_embed
    pos_params = context_length * n_embed
    head_params = n_head * (n_embed // n_head) * n_embed * 3 + n_embed * n_embed
    mlp_params = n_embed * 4 * n_embed * 2 + 5 * n_embed
    block_params = (head_params + mlp_params + 4 * n_embed)  # + LN
    lm_head_params = n_embed * vocab_size + vocab_size
    print(f"  token embedding: {tok_params:,}")
    print(f"  position embedding: {pos_params:,}")
    print(f"  per block (attn+mlp+ln): {block_params:,}")
    print(f"  {n_blocks} blocks: {block_params * n_blocks:,}")
    print(f"  lm_head: {lm_head_params:,}")
    print(f"  💡 参数量 ≈ vocab*embed + n_blocks*(12*embed²) + embed*vocab")

    # forward shape 验证
    print(f"\n═══ Forward Shape 验证 ═══")
    B, T = 2, 16
    idx = torch.randint(0, vocab_size, (B, T))
    logits, loss = model(idx, targets=idx)
    print(f"  输入 idx: {tuple(idx.shape)} → logits: {tuple(logits.shape)}, loss={loss.item():.4f}")
    print(f"  loss ≈ ln(vocab) = {math.log(vocab_size):.4f}（随机初始化的期望值）")

    # forward_hidden 验证
    hidden = model.forward_hidden(idx)
    print(f"  forward_hidden: {tuple(hidden.shape)}（LM head 之前的 hidden state）")

    # 生成验证
    print(f"\n═══ 生成验证 ═══")
    prompt = torch.randint(0, vocab_size, (1, 4))
    gen = model.generate(prompt, max_new_tokens=16, temperature=1.0, top_k=10)
    print(f"  prompt shape: {tuple(prompt.shape)} → 生成后: {tuple(gen.shape)}")
    print(f"  ✅ 自回归生成正常（未训练，输出随机）")

    # 参数量随规模增长
    print(f"\n═══ 参数量随规模增长 ═══")
    configs = [
        ("CPU 缩小版", 64, 4, 2, 256, 64),
        ("~1M 中规模", 128, 4, 4, 1000, 128),
        ("~10M", 256, 8, 6, 50304, 256),
        ("~100M", 512, 8, 12, 50304, 512),
        ("~406M（标准）", 1024, 16, 24, 50304, 1024),
    ]
    for name, ne, nh, nb, vs, cl in configs:
        n = count_params(GPT(nh, ne, cl, vs, nb))
        print(f"  {name:<16} {n / 1e6:>8.2f}M  (embed={ne}, heads={nh}, blocks={nb})")

    print(f"""
═══ 总结 ═══

本脚本构建了 GPT-2 的完整架构（经典款）：
  - token embedding + learned position embedding（相加，不用 RoPE）
  - N × Pre-LN Block（LayerNorm → MHA → 残差 → LayerNorm → MLP → 残差）
  - MHA：多头并行 + 输出投影（不用 GQA）
  - MLP：4x 扩展 + ReLU（不用 SwiGLU）
  - lm_head：hidden → vocab logits
  - forward_hidden()：后训练的 hook 点（奖励头/价值头接入处）

与 Part 7 架构对比：
  Part 7 (minimind): RMSNorm + RoPE + GQA + SwiGLU → 现代 LLM
  Part 8 (本脚本):   LayerNorm + learned PE + MHA + ReLU → 经典 GPT-2

下一个脚本：预训练（数据加载 + AdamW + cosine schedule + bf16 + checkpoint）。""")

if __name__ == '__main__':
    main()
