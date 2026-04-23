#!/usr/bin/env python3
"""
07 - 从训练好的模型采样
训练完成后，从模型中采样新的名字。
用 torch.multinomial 从 softmax 概率分布中采样，
每次生成直到遇到终止符 '.'。
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
N_EMBD = 2
N_HIDDEN = 100
MAX_STEPS = 20000
BATCH_SIZE = 32
LR_INIT = 0.1
LR_DECAY = 0.01
LR_DECAY_STEP = 15000
N_SAMPLES = 20  # 采样名字数量


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

    # ── 数据准备 ──────────────────────────────────────────
    with open(data_path, 'r') as f:
        words = f.read().splitlines()
    chars = sorted(set(''.join(words)))
    chars = ['.'] + chars
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)

    random.shuffle(words)
    n1 = int(0.8 * len(words))
    X_train, Y_train = build_dataset(words[:n1], stoi)

    # ── 初始化并训练模型 ──────────────────────────────────
    C = torch.randn((vocab_size, N_EMBD))
    W1 = torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN))
    b1 = torch.randn(N_HIDDEN)
    W2 = torch.randn((N_HIDDEN, vocab_size))
    b2 = torch.randn(vocab_size)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    print("训练中...")
    for step in range(MAX_STEPS):
        ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
        Xb, Yb = X_train[ix], Y_train[ix]

        emb = C[Xb]
        h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = LR_INIT if step < LR_DECAY_STEP else LR_DECAY
        for p in parameters:
            p.data += -lr * p.grad

    print(f"训练完成，最终 loss: {loss.item():.4f}")
    print()

    # ── 采样新名字 ────────────────────────────────────────
    g = torch.Generator()
    g.manual_seed(2147483647 + 10)

    print(f"=== 采样 {N_SAMPLES} 个名字 ===")
    with torch.no_grad():
        for i in range(N_SAMPLES):
            out = []
            context = [0] * BLOCK_SIZE  # 从 '...' 开始

            while True:
                # 前向传播
                emb = C[torch.tensor([context])]       # (1, 3, 2)
                h = torch.tanh(emb.view(1, -1) @ W1 + b1)  # (1, 100)
                logits = h @ W2 + b2                        # (1, 27)

                # Softmax 转概率
                probs = F.softmax(logits, dim=1)

                # 按概率采样下一个字符
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()

                # 滑动窗口
                context = context[1:] + [ix]

                if ix == 0:  # 遇到 '.' 则结束
                    break
                out.append(itos[ix])

            print(f"  {i + 1:2d}. {''.join(out)}")
