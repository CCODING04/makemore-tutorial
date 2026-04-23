"""
06_manual_training.py - 手动反向传播训练完整网络

把前面的知识串起来！不用 loss.backward()，完全手动计算梯度训练网络。
这是对反向传播理解的终极验证 —— 如果能跑通，说明你真的懂了 💪

网络结构：
  Embedding → Linear → BatchNorm → Tanh → Linear → CrossEntropy Loss

手动梯度包括：
  1. CrossEntropy 简化反传（3 行）
  2. 线性层反传（矩阵乘法）
  3. Tanh 反传（1 - tanh²）
  4. BatchNorm 简化反传（一行公式）
  5. Embedding 反传（scatter 操作）
"""

import os
import math
import torch
import torch.nn.functional as F

# ─── 固定随机种子 ───────────────────────────────────────────────
torch.manual_seed(42)

# ─── 数据加载 ───────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

with open(data_path, 'r') as f:
    words = f.read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# ─── 网络参数 ──────────────────────────────────────────────────
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(42)

C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / math.sqrt(n_embd * block_size)
# b1 被 BatchNorm 的 beta 取代，不需要单独的 bias
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.zeros(vocab_size)

# BN 的 running mean/var（推理时用，训练时更新）
bnmean_running = torch.zeros((1, n_hidden))
bnvar_running = torch.ones((1, n_hidden))

parameters = [C, W1, bngain, bnbias, W2, b2]
total_params = sum(p.nelement() for p in parameters)
print(f"网络参数总量: {total_params:,}")
print(f"结构: Emb({n_embd}) → Linear({n_embd * block_size}, {n_hidden}) → BN → Tanh → Linear({n_hidden}, {vocab_size})")
print()

# ─── 训练超参数 ────────────────────────────────────────────────
max_steps = 200000
batch_size = 32
learning_rate = 0.1
lossi = []

print("=" * 60)
print("🏋️ 开始手动梯度训练！")
print("=" * 60)
print()

for step in range(max_steps):
    # ── Mini-batch ──────────────────────────────────────────
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    n = batch_size

    # ════════════════════════════════════════════════════════
    # 前向传播（展开所有中间变量）
    # ════════════════════════════════════════════════════════
    emb = C[Xb]                              # (B, 3, 10)
    embcat = emb.view(n, -1)                 # (B, 30)

    # Linear 1
    hprebn = embcat @ W1                     # (B, 200)

    # BatchNorm
    bnmeani = hprebn.mean(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = bndiff2.mean(0, keepdim=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # 更新 running stats
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnvar_running = 0.999 * bnvar_running + 0.001 * bnvar

    # Tanh
    h = torch.tanh(hpreact)                  # (B, 200)

    # Linear 2
    logits = h @ W2 + b2                     # (B, 27)

    # CrossEntropy loss
    loss = F.cross_entropy(logits, Yb)

    # ════════════════════════════════════════════════════════
    # 手动反向传播 🪄
    # ════════════════════════════════════════════════════════

    # 1️⃣ CrossEntropy 简化反传
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n

    # 2️⃣ Linear 2 反传: logits = h @ W2 + b2
    dh = dlogits @ W2.T                      # (B, 200)
    dW2 = h.T @ dlogits                      # (200, 27)
    db2 = dlogits.sum(0)                     # (27,)

    # 3️⃣ Tanh 反传: h = tanh(hpreact)
    dhpreact = dh * (1.0 - h ** 2)           # (B, 200)

    # 4️⃣ BatchNorm 简化反传（一行公式）
    dhprebn = (bngain * bnvar_inv / n) * (
        n * dhpreact
        - dhpreact.sum(0)
        - (n / (n - 1)) * bnraw * (dhpreact * bnraw).sum(0)
    )
    # BN 参数梯度
    dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)

    # 5️⃣ Linear 1 反传: hprebn = embcat @ W1
    dembcat = dhprebn @ W1.T                 # (B, 30)
    dW1 = embcat.T @ dhprebn                 # (30, 200)

    # 6️⃣ Embedding 反传
    demb = dembcat.view(emb.shape)            # (B, 3, 10)
    dC = torch.zeros_like(C)
    for i in range(n):
        for j in range(block_size):
            dC[Xb[i, j]] += demb[i, j]

    # ════════════════════════════════════════════════════════
    # 参数更新（SGD）
    # ════════════════════════════════════════════════════════
    lr = learning_rate if step < 100000 else learning_rate * 0.1  # step decay

    C.data -= lr * dC
    W1.data -= lr * dW1
    bngain.data -= lr * dbngain
    bnbias.data -= lr * dbnbias
    W2.data -= lr * dW2
    b2.data -= lr * db2

    # ── 日志 ────────────────────────────────────────────────
    lossi.append(loss.log10().item())
    if step % 10000 == 0:
        print(f"  Step {step:>7d} | loss = {loss.item():.4f}")

print()
print("=" * 60)
print("🏁 训练完成！")
print("=" * 60)
print()

# ─── 评估 Train / Dev / Test loss ─────────────────────────────
@torch.no_grad()
def split_loss(split_name, X, Y):
    """用 running stats 做 BN 推理"""
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1
    hpreact = bngain * (hprebn - bnmean_running) / torch.sqrt(bnvar_running + 1e-5) + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(f"  {split_name:6s} loss: {loss.item():.4f}")


print("📊 最终评估：")
split_loss("Train", Xtr, Ytr)
split_loss("Dev",   Xdev, Ydev)
split_loss("Test",  Xte, Yte)
print()

# ─── 采样生成名字 ─────────────────────────────────────────────
print("=" * 60)
print("🎲 采样生成名字（手动梯度训练的模型）")
print("=" * 60)
print()

g_sample = torch.Generator().manual_seed(42 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        embcat = emb.view(1, -1)
        hprebn = embcat @ W1
        hpreact = bngain * (hprebn - bnmean_running) / torch.sqrt(bnvar_running + 1e-5) + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print("  " + "".join(out))

print()
print("✅ 完全手动反向传播训练成功！没有用 loss.backward() ！")
print("   这证明你理解了反向传播的每一个细节 🎉")
