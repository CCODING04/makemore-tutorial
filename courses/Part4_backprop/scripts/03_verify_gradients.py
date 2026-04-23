"""
03_verify_gradients.py - 梯度验证工具箱

手写梯度到底对不对？用 autograd 当标准答案来比对！
这是调试反向传播的神器 —— 你能立刻看到哪一步算错了。

核心思路：
  1. 前向传播（展开所有中间变量）
  2. 手动计算梯度
  3. 同时跑 loss.backward() 拿到 autograd 梯度
  4. cmp() 函数对比两者，打印最大误差
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
# cmp() 工具函数 —— 梯度验证的核心
# ═══════════════════════════════════════════════════════════════

def cmp(name: str, dt: torch.Tensor, t: torch.Tensor) -> bool:
    """
    比较手动梯度 dt 和 autograd 梯度 t。

    参数:
        name: 参数名称（用于打印）
        dt:   手动计算的梯度
        t:    PyTorch autograd 计算的梯度

    返回:
        bool: 是否在容忍范围内（atol=1e-5）
    """
    exact = torch.allclose(dt, t, atol=1e-5)
    maxdiff = (dt - t).abs().max().item()
    mean_diff = (dt - t).abs().mean().item()
    status = "✅" if exact else "❌"
    print(f"  {status} {name:15s} | max diff = {maxdiff:.2e} | mean diff = {mean_diff:.2e}")
    return exact


# ═══════════════════════════════════════════════════════════════
# 前向传播（完全展开）
# ═══════════════════════════════════════════════════════════════
emb = C[Xb]                            # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)    # (32, 30)
hprebn = embcat @ W1 + b1              # (32, 64)

# BatchNorm
bnmeani = hprebn.mean(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = bndiff2.mean(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

# Tanh
h = torch.tanh(hpreact)

# Output
logits = h @ W2 + b2

# CrossEntropy (展开)
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[torch.arange(batch_size), Yb].mean()

print(f"Loss = {loss.item():.4f}")
print()

# ═══════════════════════════════════════════════════════════════
# Autograd 参考梯度
# ═══════════════════════════════════════════════════════════════
for p in parameters:
    p.grad = None
loss.backward()

# ═══════════════════════════════════════════════════════════════
# 手动反向传播（完整 12 步 + Extra）
# ═══════════════════════════════════════════════════════════════

# Step 1: loss → dlogprobs
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[torch.arange(batch_size), Yb] = -1.0 / batch_size

# Step 2: dlogprobs → dprobs
dprobs = dlogprobs * (1.0 / probs)

# Step 3: probs = counts * counts_sum_inv
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts = dprobs * counts_sum_inv

# Step 4: counts_sum_inv = counts_sum ** -1
dcounts_sum = dcounts_sum_inv * (-counts_sum ** -2)

# Step 5: counts_sum = counts.sum(1) → 补充 dcounts
dcounts += torch.ones_like(counts) * dcounts_sum

# Step 6: counts = norm_logits.exp()
dnorm_logits = dcounts * counts

# Step 7: norm_logits = logits - logit_maxes
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)

# Step 8: logit_maxes = logits.max(1)
max_indices = logits.argmax(dim=1)
dlogit_maxes_grad = torch.zeros_like(logits)
dlogit_maxes_grad[torch.arange(batch_size), max_indices] = dlogit_maxes.squeeze()
dlogits += dlogit_maxes_grad

# Step 9: logits = h @ W2 + b2
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

# Step 10: h = tanh(hpreact)
dhpreact = dh * (1.0 - h ** 2)

# Step 11: hpreact = bngain * bnraw + bnbias
dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw = dhpreact * bngain

# Step 12: BatchNorm 反向传播
dbndiff = dbnraw * bnvar_inv
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
dbnvar = dbnvar_inv * (-0.5) * (bnvar + 1e-5) ** -1.5
dbndiff2 = torch.ones_like(bndiff2) * (dbnvar / batch_size)
dbndiff += dbndiff2 * 2.0 * bndiff
dbnmeani = -dbndiff.sum(0, keepdim=True)
dhprebn = dbndiff + torch.ones_like(hprebn) * (dbnmeani / batch_size)

# Extra: dhprebn → dembcat → dC
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for i in range(batch_size):
    for j in range(block_size):
        dC[Xb[i, j]] += demb[i, j]

# ═══════════════════════════════════════════════════════════════
# 逐参数验证
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("🔍 梯度验证：手动梯度 vs Autograd 梯度")
print("=" * 60)
print()

all_ok = True

print("📐 网络参数梯度：")
all_ok &= cmp("C  (Embedding)", dC, C.grad)
all_ok &= cmp("W1 (Linear1)",  dW1, W1.grad)
all_ok &= cmp("b1 (bias1)",    db1, b1.grad)
all_ok &= cmp("bngain (BN γ)", dbngain, bngain.grad)
all_ok &= cmp("bnbias (BN β)", dbnbias, bnbias.grad)
all_ok &= cmp("W2 (Linear2)",  dW2, W2.grad)
all_ok &= cmp("b2 (bias2)",    db2, b2.grad)

print()
print("🧩 中间变量梯度：")
all_ok &= cmp("dhprebn",  dhprebn, None if not hasattr(hprebn, 'grad') else hprebn.grad)
print("  (中间变量需要 .retain_grad() 才有 autograd 梯度)")

print()
print("=" * 60)
if all_ok:
    print("🎉 全部 7 个参数梯度验证通过！最大误差 < 1e-5")
    print("   你手写的反向传播是正确的！✨")
else:
    print("⚠️  有些梯度不匹配，回去检查推导过程")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 额外：带 retain_grad 的完整验证
# ═══════════════════════════════════════════════════════════════
print()
print("🔬 完整中间变量验证（带 retain_grad）：")
print("-" * 40)

# 重新跑一遍，这次 retain_grad
for p in parameters:
    p.grad = None

# 重新前向（使用新的变量名避免覆盖）
emb2 = C[Xb]; emb2.retain_grad()
embcat2 = emb2.view(emb2.shape[0], -1); embcat2.retain_grad()
hprebn2 = embcat2 @ W1 + b1; hprebn2.retain_grad()

bnmeani2 = hprebn2.mean(0, keepdim=True)
bndiff2_2 = hprebn2 - bnmeani2; bndiff2_2.retain_grad()
bndiff2_sq = bndiff2_2 ** 2
bnvar2 = bndiff2_sq.mean(0, keepdim=True)
bnvar_inv2 = (bnvar2 + 1e-5) ** -0.5
bnraw2 = bndiff2_2 * bnvar_inv2; bnraw2.retain_grad()
hpreact2 = bngain * bnraw2 + bnbias; hpreact2.retain_grad()

h2 = torch.tanh(hpreact2); h2.retain_grad()
logits2 = h2 @ W2 + b2; logits2.retain_grad()

loss2 = F.cross_entropy(logits2, Yb)
loss2.backward()

# 验证关键中间变量
print()
print("中间变量梯度对比：")
cmp("dembcat",  dembcat,  embcat2.grad)
cmp("dhprebn",  dhprebn,  hprebn2.grad)
cmp("dhpreact", dhpreact, hpreact2.grad)
cmp("dh",       dh,       h2.grad)
cmp("dlogits",  dlogits,  logits2.grad)
cmp("dbnraw",   dbnraw,   bnraw2.grad)
cmp("dbndiff",  dbndiff,  bndiff2_2.grad)

print()
print("✅ 验证完成！cmp() 函数已就绪，可以用于任何梯度调试场景。")
