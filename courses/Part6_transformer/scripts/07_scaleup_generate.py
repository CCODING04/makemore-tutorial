#!/usr/bin/env python3
"""
Part 6 - Script 7: 完整 GPT（scale up + Dropout + 参数统计 + 生成）
目标：把前面所有组件组装成与 gpt.py 一致的 decoder-only GPTLanguageModel，
加 Dropout 与更好的初始化，用缩小版超参在 CPU 上训练并生成文本。

覆盖知识点：
  - Dropout（Srivastava 2014）：随机置零部分神经元/注意力，训练子网络集成
    放置位置：残差连接前（attention 输出、feedforward 输出）、softmax 后
  - _init_weights：用 std=0.02 的高斯初始化 Linear/Embedding
  - 参数数量统计：sum(p.numel()) / 1e6（百万为单位）
  - 缩小版超参适配 CPU；生成时 0=换行符作为起始上下文

GPU 版完整超参（原视频，A100 上约 15 分钟，val loss 可达 1.48，~10M 参数）：
  batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6,
  dropout=0.2, lr=3e-4, max_iters=5000
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

# ─── 超参数（缩小版，适配 CPU，<30s 可跑）────────────────────────
batch_size = 16
block_size = 64
max_iters = 150
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.2
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

    # ─── DataLoader + 评估（复用 Script 6）─────────────────────────
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

    # ─── 组件定义（与 gpt.py 完全一致）─────────────────────────────
    class Head(nn.Module):
        """单头自注意力，带 dropout（softmax 后随机屏蔽部分注意力）"""

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)      # 随机阻止一些节点通信
            v = self.value(x)
            out = wei @ v
            return out

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))   # 残差连接前 dropout
            return out

    class FeedForward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),      # 残差连接前 dropout
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """Transformer block：通信 + 计算，pre-norm 结构"""

        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPTLanguageModel(nn.Module):
        """decoder-only Transformer（完整 GPT，与 gpt.py 收敛一致）"""

        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)
                                          for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            # 更好的初始化：std=0.02 的高斯（视频后续补充，实践中重要）
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    # ─── 模型与参数统计 ────────────────────────────────────────────
    model = GPTLanguageModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("═══ 模型 ═══")
    print(f"  缩小版超参: batch={batch_size}, block={block_size}, "
          f"n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, dropout={dropout}")
    print(f"  参数量: {n_params:,} = {n_params / 1e6:.3f} M")
    print(f"  原视频 GPU 版约 10M 参数（~300K tokens 预训练，对比 GPT-3 175B 参数/300B tokens）")

    # ─── 训练 ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"\n═══ 训练 (AdamW, lr={learning_rate}, {max_iters} 步) ═══")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"  step {iter:3d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ─── 生成 ─────────────────────────────────────────────────────
    print("\n═══ 生成 (500 token，0=换行符作为起始上下文) ═══")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    print("""
═══ 总结 ═══

完整的 decoder-only Transformer（GPT）构建完成，约 200 行代码：
  token 编码 + 位置编码 → N×Block(残差 + 多头注意力 + 前馈 + LayerNorm)
  → ln_f → lm_head，训练于 tiny Shakespeare。

缩小型在 CPU 上快速演示；若在 GPU 上用完整超参
（batch=64, block=256, n_embd=384, n_head=6, n_layer=6, lr=3e-4, 5000 步），
val loss 可达 1.48（A100 约 15 分钟），生成更接近真实的莎士比亚文本。

ChatGPT = 预训练（文档补全器，我们所做的）→ 微调（SFT → 奖励模型 → RLHF/PPO）。
""")


if __name__ == '__main__':
    main()
