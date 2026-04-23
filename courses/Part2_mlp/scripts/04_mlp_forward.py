#!/usr/bin/env python3
"""
04 - MLP 完整前向传播
构建一个简单的多层感知机:
  Embedding -> 拼接 -> 隐藏层(tanh) -> 输出层 -> CrossEntropy Loss
打印各层形状，理解数据流动。
"""

import os
import torch
import torch.nn.functional as F
import random

# ── 数据路径 ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

# ── 超参数 ────────────────────────────────────────────────
BLOCK_SIZE = 3
N_EMBD = 2          # embedding 维度
N_HIDDEN = 100      # 隐藏层神经元数


def build_dataset(words: list[str], stoi: dict, block_size: int = BLOCK_SIZE):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(2147483647)

    # 读取数据并构建映射
    with open(data_path, 'r') as f:
        words = f.read().splitlines()
    chars = sorted(set(''.join(words)))
    chars = ['.'] + chars
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)

    # 构建训练集
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    X_train, Y_train = build_dataset(words[:n1], stoi)

    print(f"词汇表大小: {vocab_size}")
    print(f"训练样本数: {X_train.shape[0]}")
    print()

    # ── 初始化参数 ────────────────────────────────────────
    # Embedding 矩阵
    C = torch.randn((vocab_size, N_EMBD))

    # 隐藏层: 输入维度 = block_size * n_embd = 3 * 2 = 6
    W1 = torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN))
    b1 = torch.randn(N_HIDDEN)

    # 输出层: 隐藏层 -> 词汇表大小
    W2 = torch.randn((N_HIDDEN, vocab_size))
    b2 = torch.randn(vocab_size)

    print("=== 参数形状 ===")
    print(f"  C (Embedding):    {C.shape}")
    print(f"  W1 (隐藏层权重):  {W1.shape}")
    print(f"  b1 (隐藏层偏置):  {b1.shape}")
    print(f"  W2 (输出层权重):  {W2.shape}")
    print(f"  b2 (输出层偏置):  {b2.shape}")
    total_params = C.numel() + W1.numel() + b1.numel() + W2.numel() + b2.numel()
    print(f"  总参数量: {total_params}")
    print()

    # ── 前向传播（全部样本） ──────────────────────────────
    # 第 1 步: Embedding lookup
    emb = C[X_train]                                   # (N, 3, 2)
    print(f"1. Embedding lookup:  {emb.shape}")

    # 第 2 步: 拼接上下文中的 embedding
    emb_cat = emb.view(emb.shape[0], -1)               # (N, 6)
    print(f"2. 拼接后:           {emb_cat.shape}")

    # 第 3 步: 隐藏层 + tanh 激活
    h = torch.tanh(emb_cat @ W1 + b1)                  # (N, 100)
    print(f"3. 隐藏层 (tanh):    {h.shape}")

    # 第 4 步: 输出层 (logits)
    logits = h @ W2 + b2                               # (N, 27)
    print(f"4. Logits:           {logits.shape}")

    # 第 5 步: CrossEntropy Loss
    loss = F.cross_entropy(logits, Y_train)
    print(f"5. CrossEntropy Loss: {loss.item():.4f}")
    print()
    print("前向传播完成！这个 loss 就是我们的优化目标。")
