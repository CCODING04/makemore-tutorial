"""
05_batchnorm_backward.py - 简化版 BatchNorm 反向传播

BatchNorm 的逐步反传要 5-6 步，但其实也有一个简化公式！🎯

一行公式：
  dhprebn = bngain * bnvar_inv / n * (
      n * dhpreact
    - dhpreact.sum(0)
    - n / (n-1) * bnraw * (dhpreact * bnraw).sum(0)
  )

这行公式做了什么？
  1. 处理均值减法的梯度传播
  2. 处理方差归一化的梯度传播
  3. 处理 BatchNorm 缩放 (bngain) 的梯度传播
  4. 用 Bessel 校正 (n/(n-1)) 处理偏差

来验证它和逐步版本是不是完全一致！
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
Xtr, Ytr = build_dataset(words[:n1])

# ─── 网络参数 ──────────────────────────────────────────────────
n_embd = 10
n_hidden = 64

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / math.sqrt(n_embd * block_size)
b1 = torch.zeros(n_hidden)
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, bngain, bnbias, W2, b2]
for p in parameters:
    p.requires_grad = True

# ─── Mini-batch ────────────────────────────────────────────────
batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# ═══════════════════════════════════════════════════════════════
# 前向传播
# ═══════════════════════════════════════════════════════════════
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1

# BatchNorm
bnmeani = hprebn.mean(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = bndiff2.mean(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)

print(f"Loss = {loss.item():.4f}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 1：逐步 BatchNorm 反向传播（5-6 步）
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 1：逐步 BatchNorm 反向传播")
print("=" * 60)

# 先拿到 dhpreact（从 loss 到 hpreact 的梯度）
for p in parameters:
    p.grad = None
loss.backward()
dhpreact_auto = hpreact.grad.clone()

# 手动逐步反传 BN
# hpreact = bngain * bnraw + bnbias → dbngain, dbnbias, dbnraw
dbngain = (dhpreact_auto * bnraw).sum(0, keepdim=True)
dbnbias = dhpreact_auto.sum(0, keepdim=True)
dbnraw = dhpreact_auto * bngain

# bnraw = bndiff * bnvar_inv
dbndiff = dbnraw * bnvar_inv
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)

# bnvar_inv = (bnvar + eps)^{-0.5}
dbnvar = dbnvar_inv * (-0.5) * (bnvar + 1e-5) ** -1.5

# bnvar = bndiff2.mean(0)
dbndiff2 = torch.ones_like(bndiff2) * (dbnvar / batch_size)

# bndiff2 = bndiff^2
dbndiff += dbndiff2 * 2.0 * bndiff

# bndiff = hprebn - bnmeani → dbnmeani
dbnmeani = -dbndiff.sum(0, keepdim=True)

# bnmeani = hprebn.mean(0)
dhprebn_step = dbndiff + torch.ones_like(hprebn) * (dbnmeani / batch_size)

print(f"  逐步 dhprebn: shape = {tuple(dhprebn_step.shape)}")
print(f"  前 3 行前 5 列:\n{dhprebn_step[:3, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 2：简化版 —— 一行公式 🪄
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 2：简化版 BatchNorm 反向传播（一行公式）")
print("=" * 60)
print()

n = batch_size

# 🪄 一行魔法公式：
dhprebn_simple = (bngain * bnvar_inv / n) * (
    n * dhpreact_auto
    - dhpreact_auto.sum(0)
    - (n / (n - 1)) * bnraw * (dhpreact_auto * bnraw).sum(0)
)

print(f"  简化 dhprebn: shape = {tuple(dhprebn_simple.shape)}")
print(f"  前 3 行前 5 列:\n{dhprebn_simple[:3, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 3：Autograd 直接对 hprebn 求梯度
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 3：PyTorch Autograd 参考值")
print("=" * 60)

# 重新前向，这次 hprebn 保留梯度
for p in parameters:
    p.grad = None

emb2 = C[Xb]
embcat2 = emb2.view(emb2.shape[0], -1)
hprebn2 = embcat2 @ W1 + b1
hprebn2.retain_grad()

bnmeani2 = hprebn2.mean(0, keepdim=True)
bndiff2_2 = hprebn2 - bnmeani2
bndiff2_sq = bndiff2_2 ** 2
bnvar2 = bndiff2_sq.mean(0, keepdim=True)
bnvar_inv2 = (bnvar2 + 1e-5) ** -0.5
bnraw2 = bndiff2_2 * bnvar_inv2
hpreact2 = bngain * bnraw2 + bnbias

h2 = torch.tanh(hpreact2)
logits2 = h2 @ W2 + b2
loss2 = F.cross_entropy(logits2, Yb)
loss2.backward()

dhprebn_auto = hprebn2.grad.clone()
print(f"  Autograd dhprebn: shape = {tuple(dhprebn_auto.shape)}")
print(f"  前 3 行前 5 列:\n{dhprebn_auto[:3, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 三种方法对比
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("🔍 三种方法对比")
print("=" * 60)

diff_step_vs_auto = (dhprebn_step - dhprebn_auto).abs().max().item()
diff_simple_vs_auto = (dhprebn_simple - dhprebn_auto).abs().max().item()
diff_step_vs_simple = (dhprebn_step - dhprebn_simple).abs().max().item()

print(f"  逐步 vs Autograd:  max diff = {diff_step_vs_auto:.2e}  {'✅' if diff_step_vs_auto < 1e-5 else '❌'}")
print(f"  简化 vs Autograd:  max diff = {diff_simple_vs_auto:.2e}  {'✅' if diff_simple_vs_auto < 1e-5 else '❌'}")
print(f"  逐步 vs 简化:      max diff = {diff_step_vs_simple:.2e}  {'✅' if diff_step_vs_simple < 1e-5 else '❌'}")
print()

# ═══════════════════════════════════════════════════════════════
# 原理解释
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("💡 公式拆解")
print("=" * 60)
print("""
BatchNorm 前向：
  μ = mean(x, dim=0)           ← batch 均值
  σ² = var(x, dim=0)           ← batch 方差
  x̂ = (x - μ) / √(σ² + ε)    ← 标准化
  y = γ · x̂ + β               ← 缩放平移

反向传播（简化公式）：
  dhprebn = (γ / √(σ²+ε)) / n * (
      n · dhpreact              ← 直接传播
    - Σ(dhpreact)               ← 均值减法的修正
    - n/(n-1) · x̂ · Σ(dhpreact · x̂)  ← 方差归一化的修正
  )

三个项的含义：
  📌 n · dhpreact: 直接把梯度传回来（因为 x̂ 包含了 x）
  📌 -Σ(dhpreact): 减去均值 μ 导致的修正（μ 依赖于所有 x）
  📌 -n/(n-1) · x̂ · Σ(dhpreact · x̂): 除以标准差 σ 导致的修正
     其中 n/(n-1) 是 Bessel 校正（无偏估计）

关键洞察：
  - BatchNorm 的梯度依赖于整个 batch 的统计量
  - 这就是为什么 BatchNorm 的行为和 batch size 有关
  - n/(n-1) 项在 n 很大时约等于 1，所以大 batch 时影响小
""")
print()

# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("🎉 简化版 BatchNorm 反向传播验证完成！")
print()
print("   记住这个一行公式（面试利器）：")
print("   dhprebn = bngain * bnvar_inv / n * (")
print("       n * dhpreact - dhpreact.sum(0)")
print("       - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))")
print("   )")
print("=" * 60)
