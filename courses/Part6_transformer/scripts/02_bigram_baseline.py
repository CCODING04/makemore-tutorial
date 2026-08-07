#!/usr/bin/env python3
"""
Part 6 - 脚本 2: Dataloader 与 Bigram 基线
目标：搭建语言建模的完整训练管线，用最简单的 Bigram 模型作为基线：
  - block_size（上下文长度）与 batch_size（并行序列数）概念
  - 一个 chunk 含多个 (x,y) 样本（T, T+1 偏移）
  - get_batch(split)：随机 offset 采样 + torch.stack
  - Bigram 模型：nn.Embedding(vocab_size, vocab_size) 直接当 logits（token 间无交流）
  - 交叉熵损失（PyTorch 要求 reshape 成 B*T, C）
  - generate：softmax + torch.multinomial 采样
  - 训练后 loss ≈ 2.5（无上下文交流的基线）

覆盖知识点：
  - 不把整篇文本喂入 Transformer，只采样 chunk
  - 优化器用 AdamW（比之前 makemore 的 SGD 更先进），小网络 lr 可较大
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
batch_size = 32      # 每次并行处理的独立序列数
block_size = 8       # 最大上下文长度（context length）
max_iters = 1500
eval_interval = 500
eval_iters = 80
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ─── ───────────────────────────────────────────────────────────────

torch.manual_seed(1337)


def main():
    # ─── 数据路径 ───────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # tokenizer（复用 Script 1）
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # train/val 划分（复用 Script 1）
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # ─── 一个 chunk 内含多个样本的演示 ─────────────────────────────
    print("═══ 一个 chunk 内含多个样本 ═══")
    print(f"  取前 {block_size}+1 = {block_size+1} 个字符，它们按顺序偏移形成 {block_size} 个训练样本：")
    chunk = train_data[:block_size + 1]
    for t in range(block_size):
        context = chunk[:t + 1].tolist()
        target = chunk[t + 1].item()
        print(f"    x={str(context):28s} → y={target}")

    # ─── DataLoader：get_batch ─────────────────────────────────────
    def get_batch(split):
        # 从 train/val 中随机采样 batch_size 个长度为 block_size 的 chunk
        data_local = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_local) - block_size, (batch_size,))
        x = torch.stack([data_local[i:i + block_size] for i in ix])        # (B,T)
        y = torch.stack([data_local[i + 1:i + block_size + 1] for i in ix])  # (B,T) 偏移 1
        x, y = x.to(device), y.to(device)
        return x, y

    xb, yb = get_batch('train')
    print("\n═══ 一个 batch ═══")
    print(f"  X shape: {xb.shape} (B={batch_size}, T={block_size})，Y shape: {yb.shape}")
    print(f"  共 {xb.shape[0] * xb.shape[1]} 个独立样本打包在一个 batch 中")

    # ─── 评估函数（对多个 batch 求平均，降低噪声）─────────────────
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

    # ─── Bigram 语言模型 ──────────────────────────────────────────
    class BigramLanguageModel(nn.Module):
        """最简单基线：每个 token 只看"我是谁"，token 之间不交流"""

        def __init__(self, vocab_size):
            super().__init__()
            # 直接用 vocab_size×vocab_size 的 embedding 表当 logits：
            # 输入一个 token 索引，直接查表得到下一个 token 的分数
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx, targets=None):
            logits = self.token_embedding_table(idx)  # (B,T,C)
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                # 交叉熵要求 (B*T, C) 与 (B*T)，把批量/时间合并
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

        def generate(self, idx, max_new_tokens):
            # 取最后位置的 logits → softmax → multinomial 采样 → 拼接
            for _ in range(max_new_tokens):
                logits, loss = self(idx)                  # 只看最后一个 token 就够（bigram）
                logits = logits[:, -1, :]                 # (B, C)
                probs = F.softmax(logits, dim=-1)         # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1)   # (B, T+1)
            return idx

    model = BigramLanguageModel(vocab_size).to(device)

    # ─── 初始 loss ────────────────────────────────────────────────
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    print("\n═══ 初始 loss ═══")
    print(f"  初始 loss: {loss.item():.4f}")
    print(f"  理论下限: -ln(1/65) = ln65 ≈ 4.17（完全均匀分布时的损失）")

    # ─── 训练前生成（随机垃圾文本）────────────────────────────────
    print("\n═══ 训练前生成 (100 token，纯随机) ═══")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

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

    # ─── 训练后生成 ───────────────────────────────────────────────
    print("\n═══ 训练后生成 (200 token) ═══")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

    print("""
═══ 总结 ═══

Bigram 基线：token 之间完全不交流，只看"我是谁"来预测下一个字符。
训练后 loss ≈ 2.5，明显低于初始 ~4.8，但离好的语言模型还很远。
生成结果只有零星英文碎片（没有真正的词汇/语法结构），因为缺少上下文。

关键局限：预测下一个字符只用了最后一个 token → 上下文信息完全被浪费。
解决思路：让 token 之间相互交流，根据上下文做更好的预测 → Self-Attention！
""")

    print(f"最终 train/val loss: {losses['train']:.4f} / {losses['val']:.4f}")


if __name__ == '__main__':
    main()
