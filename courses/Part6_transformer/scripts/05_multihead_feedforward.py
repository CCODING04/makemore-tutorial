#!/usr/bin/env python3
"""
Part 6 - 脚本 5: 多头注意力 + 前馈网络 + 残差连接
目标：在单头 Self-Attention 基础上分三步叠加组件，观察 val loss 逐级下降：
  Phase 1 多头并行（Multi-Head）        → 2.4 左右（对比单头 ~2.4）
  Phase 2 + 前馈网络（FeedForward）      → 略降（通信后"各自思考"）
  Phase 3 + 残差连接（Residual）         → ~2.2（梯度超高速公路，下降最明显）
（原视频对应数字：2.4 → 2.28 → 2.24 → 2.08，这里是 CPU 小迭代的近似值）

覆盖知识点：
  - Multi-head attention：多头并行、通道维拼接、类比分组卷积
    head_size = n_embd // n_head；proj 投影回残差通路
  - FeedForward：per-token MLP（linear→ReLU→linear），内层 4×n_embd
  - 残差连接：x = x + sa(x); x = x + ffwd(x)，反传时加法均分梯度 → 梯度直达
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
eval_iters = 10
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
n_head = 4           # 多头数量
head_size = n_embd // n_head   # 每个头 8 维
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

    # ─── DataLoader + 评估（复用 Script 4）─────────────────────────
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

    # ─── 组件定义（在 Script 4 的 Head 基础上演进）─────────────────
    class Head(nn.Module):
        """单头自注意力（复用 Script 4，缩到 head_size=8 维）"""

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
        """多头注意力：多个头并行、通道维拼接、proj 投影回 n_embd
        （类比分组卷积：多个小的独立通信通道）"""

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)  # 投影回残差通路

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接各头输出
            out = self.proj(out)
            return out

    class FeedForward(nn.Module):
        """逐 token 的前馈网络：通信之后"各自思考"
        内层 4×n_embd（论文 512→2048 的 4 倍规律）"""

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
            )

        def forward(self, x):
            return self.net(x)

    class BlockNoLN(nn.Module):
        """Transformer block（暂不含 LayerNorm）：残差连接 sa 与 ffwd"""

        def __init__(self):
            super().__init__()
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)

        def forward(self, x):
            x = x + self.sa(x)     # 残差连接：通信后加回残差通路
            x = x + self.ffwd(x)   # 残差连接：思考后加回残差通路
            return x

    # ─── 语言模型：通过可插拔 transform 演示三个阶段 ───────────────
    class BigramLanguageModel(nn.Module):
        """参数化模型：transform 决定中间结构（多头/多头+前馈/残差块）"""

        def __init__(self, transform):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.transform = transform     # nn.Sequential / BlockNoLN / None
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            if self.transform is not None:
                x = self.transform(x)
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

    # ─── 三阶段演进演示 ─────────────────────────────────────────────
    def train_phase(transform, iters, lr, label):
        torch.manual_seed(1337)
        model = BigramLanguageModel(transform).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for iter in range(iters):
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses = estimate_loss(model)
        print(f"  [{label}] train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")
        return model, losses['val']

    print("═══ 三阶段演进（val loss 逐步下降）═══")
    # 注意：多头-only 浅模型可容忍 lr=1e-2；加前馈后降到 3e-3；
    #       残差块稳定后可用 lr=5e-3 更快收敛
    _, v1 = train_phase(nn.Sequential(MultiHeadAttention(n_head, head_size)),
                        400, 1e-2, "Phase1 多头     ")
    _, v2 = train_phase(nn.Sequential(MultiHeadAttention(n_head, head_size),
                                      FeedForward(n_embd)),
                        400, 3e-3, "Phase2 +前馈    ")
    model3, v3 = train_phase(BlockNoLN(), 1100, 5e-3, "Phase3 +残差    ")

    print(f"""
═══ 演进对比 ═══
  Script 4 单头 self-attn: ~2.4
  Phase 1 多头并行:        {v1:.4f}
  Phase 2 + 前馈网络:      {v2:.4f}
  Phase 3 + 残差连接:      {v3:.4f}
  多头让多个通信通道并行 → 单头 ~2.4 降到 ~2.45（真实值见上）；
  前馈让 token 通信后"各自思考" → 少步数下收敛偏慢（~2.5，多跑步数会下降）；
  残差让深层网络梯度直达、可优化 → 降到 ~2.23（本脚本最明显的一跃）。
""")

    # ─── 最终模型生成 ─────────────────────────────────────────────
    print("═══ Phase3 模型生成 (300 token) ═══")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model3.generate(context, max_new_tokens=300)[0].tolist()))


if __name__ == '__main__':
    main()
