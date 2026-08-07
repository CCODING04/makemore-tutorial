#!/usr/bin/env python3
"""
Part 6 - 脚本 4: 单头 Self-Attention + 位置编码
目标：实现 Transformer 的核心——单头自注意力（query/key/value），
加上位置编码让 attention 有"空间"概念。训练后 loss 从 2.5 降到 ~2.4。

覆盖知识点：
  - 代码清理：去掉 vocab_size 参数、引入 n_embd、token embedding + lm_head
  - 位置编码：position_embedding_table + 广播相加 (B,T,C)+(T,C)
  - Self-attention 单头（Head）：key/query/value、q@k.T 亲和力、
    masked_fill + softmax + wei@v、scaled 除以 sqrt(head_size)、register_buffer
  - 6 条 attention 笔记中的可演示部分：
      ① 通信机制  ② 无空间概念（需要位置编码）  ③ batch 间不通信
      ④ decoder 遮罩  ⑤ self vs cross  ⑥ scaled attention 控方差
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
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 500
eval_iters = 50
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32          # embedding 维度（把 token 表示得更丰富）
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

    # ─── DataLoader（复用 Script 2）────────────────────────────────
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

    # ─── 单头自注意力 Head ─────────────────────────────────────────
    class Head(nn.Module):
        """一个自注意力头：每个 token 发出 query（找什么）、key（有什么）、
        value（若有趣就传达什么），按亲和力聚合过去的信息"""

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            # tril 不是可训练参数，用 register_buffer 注册（会随模型移动设备）
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)    # (B,T,head_size)
            q = self.query(x)  # (B,T,head_size)
            # 亲和力 = query 与所有 key 的内积 → 数据依赖
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,T)，scaled
            # 遮罩：未来不能看向过去（decoder 三角遮罩）
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)  # 每行归一化成概率 (B,T,T)
            v = self.value(x)             # (B,T,head_size)
            out = wei @ v                 # 加权聚合 → (B,T,head_size)
            return out

    # ─── Self-Attention 亲和力演示（数据依赖）────────────────────
    print("═══ Self-Attention 亲和力演示 ═══")
    demo_head = Head(n_embd).to(device)
    with torch.no_grad():
        xd = torch.randn(1, block_size, n_embd, device=device)
        k = demo_head.key(xd)
        q = demo_head.query(xd)
        wei_demo = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei_demo = wei_demo.masked_fill(demo_head.tril[:block_size, :block_size] == 0, float('-inf'))
        wei_demo = F.softmax(wei_demo, dim=-1)
    print("  亲和力矩阵（每行 = 该 token 对过去各 token 的注意力权重）:")
    print(wei_demo[0])
    print("  不再是全 0/全平均 → 权重数据依赖，这就是 attention 的核心！")

    # ─── 为什么需要 scaled attention 的演示 ────────────────────────
    print("\n═══ 为什么除以 sqrt(head_size) ═══")
    vals = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2])
    print(f"  接近 0 的小值 softmax（扩散）: {F.softmax(vals, dim=-1).tolist()}")
    print(f"  放大 8 倍后 softmax（尖锐/one-hot）: {F.softmax(vals * 8, dim=-1).tolist()}")
    print("  若 unit gaussian 输入，wei 方差 ≈ head_size；除以 sqrt(head_size) 使方差≈1，")
    print("  避免初始化时 softmax 太尖锐（每个 token 只聚合一个 token）。")

    # ─── 语言模型（Bigram → +位置编码 → +单头 Self-Attention）─────
    class BigramLanguageModel(nn.Module):
        """在 Bigram 基础上：引入位置编码 + 单头自注意力"""

        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            # 位置编码：每个位置一个可学习的向量（attention 无空间概念！）
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.sa_head = Head(n_embd)       # 单头，head_size = n_embd
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)          # (B,T,C)
            pos_emb = self.position_embedding_table(
                torch.arange(T, device=device))                # (T,C)
            x = tok_emb + pos_emb                              # 广播相加 (B,T,C)
            x = self.sa_head(x)                                # 自注意力
            logits = self.lm_head(x)                           # (B,T,vocab_size)
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
                # 裁剪到 block_size：位置表只有 block_size 个位置，超了会越界
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    model = BigramLanguageModel().to(device)

    # ─── 训练 ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"\n═══ 训练 (AdamW, lr={learning_rate}, {max_iters} 步) ═══")
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
    print("\n═══ 生成 (300 token) ═══")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))

    print(f"""
═══ 总结 ═══

单头 Self-Attention：token 之间开始"交流"（按数据依赖的亲和力聚合），
loss 从 bigram 的 ~2.5 降到 ~2.4，生成文本开始有少量结构。

关于 attention 的 6 条笔记：
  1. 通信机制：有向图上按权重聚合
  2. 无空间概念：对集合操作 → 必须加位置编码（对比卷积的空间性）
  3. batch 间不通信：批矩阵乘法各自独立
  4. decoder 块用三角遮罩（未来不看过去）；encoder 块删除遮罩全连通
  5. self-attention 的 K/Q/V 都来自 X；cross-attention 的 K/V 来自外部源
  6. scaled attention：除以 sqrt(head_size) 控方差，避免 softmax 太尖锐

下一步：多头并行 + 前馈网络 + 残差连接 → Script 5。
""")


if __name__ == '__main__':
    main()
