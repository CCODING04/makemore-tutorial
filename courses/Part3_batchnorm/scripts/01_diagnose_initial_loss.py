"""
01_diagnose_initial_loss.py - 诊断初始 Loss 过高问题

本脚本构建 Part 2 的 MLP 基线，观察初始化后的 loss。
理论上，在 27 个字符上的均匀分布 → loss ≈ -ln(1/27) ≈ 3.29
但默认初始化的 loss 远高于此，说明网络初始输出过于自信（分布尖锐）。

解决方案：缩小输出层权重 W2 和偏置 b2，使初始输出接近均匀分布。
"""

import os
import torch
import torch.nn.functional as F

# ─── 固定随机种子 ───────────────────────────────────────────────
torch.manual_seed(42)

# ─── 数据加载 ───────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

with open(data_path, 'r') as f:
    words = f.read().splitlines()

print(f"数据集大小: {len(words)} 个名字")
print(f"示例: {words[:3]}")

# ─── 构建词汇表 ─────────────────────────────────────────────────
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
print(f"词汇表大小: {vocab_size}")

# ─── 构建数据集 ─────────────────────────────────────────────────
block_size = 3  # 上下文长度：用 3 个字符预测第 4 个


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(f"训练集: {Xtr.shape}, 验证集: {Xdev.shape}, 测试集: {Xte.shape}")


# ─── 构建模型（基线：默认初始化）───────────────────────────────
n_embd = 10    # 嵌入维度
n_hidden = 200  # 隐藏层大小

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

total_params = sum(p.nelement() for p in parameters)
print(f"\n模型参数总量: {total_params}")

# ─── 基线：初始化后立即计算 loss ────────────────────────────────
emb = C[Xtr]
h = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss_baseline = F.cross_entropy(logits, Ytr)
print(f"\n═══ 基线初始化 Loss ═══")
print(f"实际初始 loss: {loss_baseline.item():.4f}")
print(f"理论最优 loss:  {-torch.tensor(1/27).log().item():.4f}")
print(f"差距:           {loss_baseline.item() - (-torch.tensor(1/27).log().item()):.4f}")

# ─── 修正后的初始化 ─────────────────────────────────────────────
# 问题解释：
# - 默认初始化 W2 较大，logits 方差大 → softmax 输出尖锐 → loss 高
# - 理想情况：初始时 logits 接近 0 → softmax 接近均匀 → loss ≈ -ln(1/27)
#
# 解决方案：
# - W2 缩小 100 倍：让 logits 接近 0
# - b2 设为 0：去掉偏置的影响

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01  # 缩小！
b2 = torch.zeros(vocab_size)                                    # 清零！

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

emb = C[Xtr]
h = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss_fixed = F.cross_entropy(logits, Ytr)
print(f"\n═══ 修正后初始化 Loss ═══")
print(f"实际初始 loss: {loss_fixed.item():.4f}")
print(f"理论最优 loss:  {-torch.tensor(1/27).log().item():.4f}")
print(f"差距:           {loss_fixed.item() - (-torch.tensor(1/27).log().item()):.4f}")

# ─── 训练对比 ───────────────────────────────────────────────────
print(f"\n═══ 训练 1000 步对比 ═══")


def train_model(C, W1, b1, W2, b2, n_steps=1000, lr=0.1):
    """训练模型并返回 loss 历史"""
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    losses = []
    for i in range(n_steps):
        # 小批量
        ix = torch.randint(0, Xtr.shape[0], (32,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # 前向传播
        emb = C[Xb]
        h = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        # 反向传播
        for p in parameters:
            p.grad = None
        loss.backward()

        # 更新（简单 SGD）
        for p in parameters:
            p.data += -lr * p.grad

        losses.append(loss.item())
        if (i + 1) % 200 == 0:
            print(f"  step {i+1:4d} | loss = {loss.item():.4f}")

    return losses


# 基线训练
print("\n--- 基线初始化 ---")
torch.manual_seed(42)
g = torch.Generator().manual_seed(42)
C0 = torch.randn((vocab_size, n_embd), generator=g)
W1_0 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1_0 = torch.randn(n_hidden, generator=g)
W2_0 = torch.randn((n_hidden, vocab_size), generator=g)
b2_0 = torch.randn(vocab_size, generator=g)
losses_baseline = train_model(C0, W1_0, b1_0, W2_0, b2_0, n_steps=1000)

# 修正训练
print("\n--- 修正后初始化 ---")
torch.manual_seed(42)
g = torch.Generator().manual_seed(42)
C1 = torch.randn((vocab_size, n_embd), generator=g)
W1_1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1_1 = torch.randn(n_hidden, generator=g)
W2_1 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2_1 = torch.zeros(vocab_size)
losses_fixed = train_model(C1, W1_1, b1_1, W2_1, b2_1, n_steps=1000)

# ─── 总结 ───────────────────────────────────────────────────────
print(f"\n═══ 总结 ═══")
print(f"基线初始 loss: {losses_baseline[0]:.4f} → 最终: {losses_baseline[-1]:.4f}")
print(f"修正初始 loss: {losses_fixed[0]:.4f} → 最终: {losses_fixed[-1]:.4f}")
print(f"\n关键教训：")
print(f"  1. 初始化 loss 应接近 -ln(1/vocab_size)")
print(f"  2. 输出层 W 缩小、b 清零 → logits 接近 0 → softmax 接近均匀")
print(f'  3. 好的初始化让训练从正确的起点开始，不浪费前几步在"纠错"上')
