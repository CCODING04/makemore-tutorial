#!/usr/bin/env python3
"""
Part 6 - Script 6: LayerNorm(pre-norm) 与完整 Transformer Block
目标：引入 LayerNorm（pre-norm 结构）、Block 模块与最终 ln_f，用多层 Block
组出深层网络。2 层 Block + LayerNorm 在本脚本约 1200 步时 val loss ≈ 2.2
（原视频 1 层网络 2.08→2.06，LayerNorm 的主要价值是让更深网络可优化）。

覆盖知识点：
  - LayerNorm 与 Part3 BatchNorm 的关系：
      BatchNorm 归一化"列"（跨 batch 维），LayerNorm 归一化"行"（per-token 特征）
      LayerNorm 无 running buffer、训练/推理无区别、保留 gamma/beta
  - Pre-norm 结构（先 LayerNorm 后 attention/ffwd，区别于原论文 post-norm）：
      x = x + sa(ln1(x));  x = x + ffwd(ln2(x))
  - nn.LayerNorm(n_embd)；末尾 ln_f（lm_head 之前）
  - n_layer=2 的深层网络
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使训练更快更稳定
torch.set_num_threads(1)

# ─── 超参数 ────────────────────────────────────────────────────────
batch_size = 16
block_size = 8
max_iters = 1200
eval_interval = 400
eval_iters = 15
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
n_head = 4
head_size = n_embd // n_head
n_layer = 2          # 2 个 Block 叠起来（深层网络）
# ─── ───────────────────────────────────────────────────────────────

torch.manual_seed(1337)


def main():
    # ─── 数据路径 ───────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # ─── DataLoader + 评估（复用 Script 5）─────────────────────────
    def get_batch(split):
        data_local = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_local) - block_size, (batch_size,))
        x = torch.stack([data_local[i:i + block_size] for i in ix])
        y = torch.stack([data_local[i + 1:i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # ─── LayerNorm vs BatchNorm 演示 ────────────────────────────────
    print("═══ LayerNorm vs BatchNorm ═══")
    x = torch.randn(4, 5)                     # B=4, C=5
    print(f"  原始 x 每行 mean: {x.mean(dim=1)}")
    print(f"  原始 x 每行 std:  {x.std(dim=1)}")
    y_ln = nn.LayerNorm(5)(x)                 # 归一化"行"（per-token 特征）
    print(f"  LayerNorm 后每行 mean: {y_ln.mean(dim=1)} (≈0)")
    print(f"  LayerNorm 后每行 std:  {y_ln.std(dim=1)} (≈1)")
    print("  BatchNorm 归一化'列'（跨 batch），LayerNorm 归一化'行'（每个 token）。")
    print("  LayerNorm 无 running buffer、训练/推理无区别，保留 gamma/beta。")

    # ─── 组件定义（在 Script 5 基础上 + LayerNorm）────────────────
    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            v = self.value(x)
            out = wei @ v
            return out

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.proj(out)
            return out

    class FeedForward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """Transformer block：通信(attention) + 计算(ffwd)，pre-norm 结构。
        先 LayerNorm 再进入子模块（区别于原论文 post-norm）"""

        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))   # pre-norm：先 LN 再 attention
            x = x + self.ffwd(self.ln2(x)) # pre-norm：先 LN 再 ffwd
            return x

    class BigramLanguageModel(nn.Module):
        """完整 decoder-only Transformer（缩小版）"""

        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)
                                          for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)   # 最终 LayerNorm（lm_head 之前）
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    # ─── 训练 ─────────────────────────────────────────────────────
    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"\n═══ 训练 (n_layer={n_layer}, AdamW, lr={learning_rate}, {max_iters} 步) ═══")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"  step {iter:4d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ─── 生成 ─────────────────────────────────────────────────────
    print("\n═══ 生成 (400 token) ═══")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))

    print(f"""
═══ 总结 ═══

加入 LayerNorm（pre-norm）+ 最终 ln_f + 多层 Block 后，本脚本 2 层网络
val loss ≈ {losses['val']:.2f}。LayerNorm 的主要价值不是"降 loss"，
而是让更深的网络也能稳定优化（对比 Script 5 单层残差块的 ~2.2）。
至此 decoder-only Transformer 的主要部件已齐备：
  token 编码 + 位置编码 → N×Block(残差+多头注意力+前馈+LayerNorm) → ln_f → lm_head

loss 演进回顾（原视频数字）：
  2.5 (bigram) → 2.4 (单头) → 2.28/2.24/2.08 (多头/前馈/残差) → 2.06 (+LayerNorm)

下一步：加 Dropout、初始化技巧、scale up + 大规模生成 → Script 7。
""")


if __name__ == '__main__':
    main()
