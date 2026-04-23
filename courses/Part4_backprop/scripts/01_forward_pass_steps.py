"""
01_forward_pass_steps.py - 逐步展开前向传播

把 MLP+BN 的前向传播拆成一步步，保存每个中间变量。
这是手动反向传播的基础 —— 你得先知道前向是怎么走的。

网络结构：
    Embedding → Linear → BatchNorm → Tanh → Linear → CrossEntropy Loss

关键点：
  1. 每一步都保存中间变量（后续反向传播要用）
  2. 打印每步 tensor 的形状
  3. 理解数据在网络中是怎么流动的
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

# ─── 小网络参数 ─────────────────────────────────────────────────
n_embd = 10     # embedding 维度
n_hidden = 64   # 隐藏层大小（比 Part 3 小，方便调试）

g = torch.Generator().manual_seed(42)

# Embedding 矩阵
C = torch.randn((vocab_size, n_embd), generator=g)

# 第一层线性（Embedding → Hidden）
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / math.sqrt(n_embd * block_size)
b1 = torch.zeros(n_hidden)  # BN 有 beta，b1 不需要，但保留便于调试

# BatchNorm 参数
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

# 第二层线性（Hidden → Output）
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.zeros(vocab_size)

# 收集参数
parameters = [C, W1, b1, bngain, bnbias, W2, b2]
for p in parameters:
    p.requires_grad = True

total_params = sum(p.nelement() for p in parameters)
print(f"参数总量: {total_params:,}")
print(f"网络: Emb({n_embd}) → Linear({n_embd * block_size}, {n_hidden}) → BN → Tanh → Linear({n_hidden}, {vocab_size})")
print()

# ─── 取一个 mini-batch ─────────────────────────────────────────
batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

print(f"输入 Xb: {Xb.shape}   标签 Yb: {Yb.shape}")
print(f"Xb 前 3 行:\n{Xb[:3]}")
print()

# ─── 逐步前向传播 ───────────────────────────────────────────────
print("=" * 60)
print("逐步前向传播")
print("=" * 60)

# Step 1: Embedding lookup
emb = C[Xb]                          # (32, 3, 10)
print(f"Step 1 — emb = C[Xb]")
print(f"  形状: {tuple(emb.shape)}")
print(f"  说明: 每个字符用 {n_embd} 维向量表示，{block_size} 个上下文字符 → ({batch_size}, {block_size}, {n_embd})")
print()

# Step 2: 拼接成行向量
embcat = emb.view(emb.shape[0], -1)  # (32, 30)
print(f"Step 2 — embcat = emb.view(B, -1)")
print(f"  形状: {tuple(embcat.shape)}")
print(f"  说明: 把 {block_size} 个 embedding 拼成一个向量 → ({batch_size}, {n_embd * block_size})")
print()

# Step 3: 第一层线性变换
hprebn = embcat @ W1 + b1            # (32, 64)
print(f"Step 3 — hprebn = embcat @ W1 + b1")
print(f"  形状: {tuple(hprebn.shape)}")
print(f"  说明: 线性变换到隐藏空间 ({n_embd * block_size} → {n_hidden})")
print()

# Step 4: BatchNorm（训练模式）
bnmeani = hprebn.mean(dim=0, keepdim=True)            # (1, 64)
bndiff = hprebn - bnmeani                              # (32, 64)
bndiff2 = bndiff ** 2                                  # (32, 64)
bnvar = bndiff2.mean(dim=0, keepdim=True)              # (1, 64)
bnvar_inv = (bnvar + 1e-5) ** -0.5                     # (1, 64)
bnraw = bndiff * bnvar_inv                             # (32, 64)
hpreact = bngain * bnraw + bnbias                      # (32, 64)
print(f"Step 4 — BatchNorm")
print(f"  bnmeani:   {tuple(bnmeani.shape)}   (batch 均值)")
print(f"  bndiff:    {tuple(bndiff.shape)}   (减均值)")
print(f"  bnvar:     {tuple(bnvar.shape)}   (batch 方差)")
print(f"  bnvar_inv: {tuple(bnvar_inv.shape)}   (标准差倒数)")
print(f"  bnraw:     {tuple(bnraw.shape)}   (标准化结果)")
print(f"  hpreact:   {tuple(hpreact.shape)}   (缩放+平移后)")
print(f"  说明: x̂ = (x - μ) / √(σ² + ε),  y = γ·x̂ + β")
print()

# Step 5: Tanh 激活
h = torch.tanh(hpreact)               # (32, 64)
print(f"Step 5 — h = tanh(hpreact)")
print(f"  形状: {tuple(h.shape)}")
print(f"  范围: [{h.min():.4f}, {h.max():.4f}]")
print(f"  饱和比例 (|h| > 0.99): {(h.abs() > 0.99).float().mean():.4f}")
print()

# Step 6: 第二层线性变换（输出 logits）
logits = h @ W2 + b2                  # (32, 27)
print(f"Step 6 — logits = h @ W2 + b2")
print(f"  形状: {tuple(logits.shape)}")
print(f"  说明: 从隐藏空间映射到 {vocab_size} 个字符的分数")
print()

# Step 7: CrossEntropy Loss
# F.cross_entropy 内部做了：logits → softmax → -log(prob) → mean
# 手动展开以便反向传播
logit_maxes = logits.max(dim=1, keepdim=True).values   # (32, 1)
norm_logits = logits - logit_maxes                       # (32, 27) 减最大值防溢出
counts = norm_logits.exp()                               # (32, 27)
counts_sum = counts.sum(dim=1, keepdim=True)             # (32, 1)
counts_sum_inv = counts_sum ** -1                        # (32, 1)
probs = counts * counts_sum_inv                          # (32, 27)
logprobs = probs.log()                                   # (32, 27)
loss = -logprobs[torch.arange(batch_size), Yb].mean()    # scalar

print(f"Step 7 — CrossEntropy Loss 展开")
print(f"  logit_maxes:   {tuple(logit_maxes.shape)}   (每行最大值)")
print(f"  norm_logits:   {tuple(norm_logits.shape)}   (减最大值)")
print(f"  counts:        {tuple(counts.shape)}   (exp)")
print(f"  counts_sum:    {tuple(counts_sum.shape)}   (行求和)")
print(f"  counts_sum_inv:{tuple(counts_sum_inv.shape)}   (求和倒数)")
print(f"  probs:         {tuple(probs.shape)}   (softmax)")
print(f"  logprobs:      {tuple(logprobs.shape)}   (log)")
print(f"  loss:          {loss.shape}  = {loss.item():.4f}")
print()

# ─── 验证与 F.cross_entropy 一致 ────────────────────────────────
loss_ref = F.cross_entropy(logits, Yb)
print(f"验证: 手动 loss = {loss.item():.6f}")
print(f"      F.cross_entropy = {loss_ref.item():.6f}")
print(f"      差异 = {abs(loss.item() - loss_ref.item()):.2e}")
print()

# ─── 前向传播流程图 ─────────────────────────────────────────────
print("=" * 60)
print("前向传播流程图")
print("=" * 60)
print(f"""
  Xb ({tuple(Xb.shape)})          Yb ({tuple(Yb.shape)})
   │                            │
   ▼                            │
  emb = C[Xb]                   │
   ({tuple(emb.shape)})                │
   │                            │
   ▼                            │
  embcat = view(-1)             │
   ({tuple(embcat.shape)})              │
   │                            │
   ▼                            │
  hprebn = embcat @ W1 + b1     │
   ({tuple(hprebn.shape)})              │
   │                            │
   ▼                            │
  ┌─ BatchNorm ─┐               │
  │ bnmeani     │               │
  │ bndiff      │               │
  │ bnvar       │               │
  │ bnraw       │               │
  └──────┬──────┘               │
         ▼                      │
  hpreact = γ·bnraw + β         │
   ({tuple(hpreact.shape)})              │
   │                            │
   ▼                            │
  h = tanh(hpreact)             │
   ({tuple(h.shape)})                  │
   │                            │
   ▼                            │
  logits = h @ W2 + b2          │
   ({tuple(logits.shape)})              │
   │                            │
   ▼                            │
  ┌─ Softmax + NLL ─┐           │
  │ norm_logits      │           │
  │ counts           │           │
  │ probs            │           │
  │ logprobs         │           │
  └───────┬─────────┘           │
          ▼                     ▼
        loss = -logprobs[range(B), Yb].mean()
         ({loss.shape} = {loss.item():.4f})
""")

print("✅ 前向传播逐步展开完成！每个中间变量都保存好了，接下来可以反向传播了。")
