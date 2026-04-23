#!/usr/bin/env python3
"""
06 - 可视化 Embedding
训练模型后，将 Embedding 矩阵 C 的前两维画在 2D 平面上，
每个点标注对应字符，保存为 embedding_visualization.png。
"""

import os
import torch
import torch.nn.functional as F
import random

# Matplotlib 兼容：优先用 Agg 后端（无 GUI 也能保存）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置中文字体
_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'Noto Sans CJK SC']
_available = {f.name for f in fm.fontManager.ttflist}
for _f in _candidates:
    if _f in _available:
        plt.rcParams['font.sans-serif'] = [_f]
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 数据路径 ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

# ── 超参数 ────────────────────────────────────────────────
BLOCK_SIZE = 3
N_EMBD = 2       # 2 维 embedding，可以直接画在平面上
N_HIDDEN = 100
MAX_STEPS = 20000
BATCH_SIZE = 32
LR_INIT = 0.1
LR_DECAY = 0.01
LR_DECAY_STEP = 15000


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

    # ── 可视化 Embedding ─────────────────────────────────
    print("生成 Embedding 可视化...")
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].detach().numpy(), C[:, 1].detach().numpy(), s=200)

    for i in range(vocab_size):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
                 ha='center', va='center', color='white',
                 fontsize=10, fontweight='bold')

    plt.title("Embedding 可视化 (前两维)", fontsize=14)
    plt.xlabel("维度 1")
    plt.ylabel("维度 2")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(script_dir, 'embedding_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")
