"""
02_backprop_step_by_step.py - 12 步手动反向传播

接 01_forward_pass_steps.py，从 loss 反向推导每一层的梯度。
关键思想：链式法则 —— 每步梯度 = 上游梯度 × 局部梯度。

反向传播顺序（12 步）：
  loss → logprobs → probs → counts_sum_inv → counts_sum → counts → norm_logits → logit_maxes → logits
  → h → hpreact → bnraw/bngain/bnbias → bnvar_inv → bnvar → bndiff2 → bndiff → bnmeani → hprebn
  → embcat → W1/b1 → emb → C

每步都保存手动梯度，最后与 PyTorch autograd 对比验证。
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

# ─── 网络参数（同 01）───────────────────────────────────────────
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

# ─── 取 mini-batch ─────────────────────────────────────────────
batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# ═══════════════════════════════════════════════════════════════
# 前向传播（完全展开，保存所有中间变量）
# ═══════════════════════════════════════════════════════════════
emb = C[Xb]                            # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)    # (32, 30)
hprebn = embcat @ W1 + b1              # (32, 64)

# BatchNorm
bnmeani = hprebn.mean(0, keepdim=True)        # (1, 64)
bndiff = hprebn - bnmeani                      # (32, 64)
bndiff2 = bndiff ** 2                          # (32, 64)
bnvar = bndiff2.mean(0, keepdim=True)          # (1, 64)
bnvar_inv = (bnvar + 1e-5) ** -0.5             # (1, 64)
bnraw = bndiff * bnvar_inv                     # (32, 64)
hpreact = bngain * bnraw + bnbias              # (32, 64)

# Tanh
h = torch.tanh(hpreact)                         # (32, 64)

# Output layer
logits = h @ W2 + b2                            # (32, 27)

# CrossEntropy (展开)
logit_maxes = logits.max(1, keepdim=True).values  # (32, 1)
norm_logits = logits - logit_maxes                # (32, 27)
counts = norm_logits.exp()                        # (32, 27)
counts_sum = counts.sum(1, keepdim=True)          # (32, 1)
counts_sum_inv = counts_sum ** -1                  # (32, 1)
probs = counts * counts_sum_inv                    # (32, 27)
logprobs = probs.log()                             # (32, 27)
loss = -logprobs[torch.arange(batch_size), Yb].mean()

print(f"前向传播完成，loss = {loss.item():.4f}")
print()

# ═══════════════════════════════════════════════════════════════
# PyTorch autograd（作为参考）
# ═══════════════════════════════════════════════════════════════
for p in parameters:
    p.grad = None
loss.backward()

# ═══════════════════════════════════════════════════════════════
# 手动反向传播（12 步）
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("手动反向传播 12 步")
print("=" * 60)

# ── Step 1: loss → dlogprobs ──────────────────────────────────
# loss = -logprobs[arange(B), Yb].mean()
# mean 除以了 N，所以梯度要乘 1/N
# 只有 Yb 位置有非零梯度
dlogprobs = torch.zeros_like(logprobs)           # (32, 27)
dlogprobs[torch.arange(batch_size), Yb] = -1.0 / batch_size
print(f"Step 1  dlogprobs: {tuple(dlogprobs.shape)}")

# ── Step 2: dlogprobs → dprobs ───────────────────────────────
# logprobs = log(probs),  d/dx log(x) = 1/x
dprobs = dlogprobs * (1.0 / probs)               # (32, 27)
print(f"Step 2  dprobs:    {tuple(dprobs.shape)}")

# ── Step 3: dprobs → dcounts_sum_inv + dcounts ──────────────
# probs = counts * counts_sum_inv
# dcounts_sum_inv: counts 贡献了一部分梯度
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)  # (32, 1)
# dcounts: counts_sum_inv 贡献了一部分梯度
dcounts = dprobs * counts_sum_inv                          # (32, 27)
print(f"Step 3  dcounts_sum_inv: {tuple(dcounts_sum_inv.shape)}")
print(f"        dcounts:         {tuple(dcounts.shape)}")

# ── Step 4: dcounts_sum_inv → dcounts_sum ───────────────────
# counts_sum_inv = counts_sum ** -1,  d/dx x^-1 = -x^-2
dcounts_sum = dcounts_sum_inv * (-counts_sum ** -2)        # (32, 1)
print(f"Step 4  dcounts_sum: {tuple(dcounts_sum.shape)}")

# ── Step 5: dcounts_sum → dcounts（补充）─────────────────────
# counts_sum = counts.sum(1, keepdim=True)
# 每个 counts 元素贡献 1 到 sum
dcounts += torch.ones_like(counts) * dcounts_sum            # (32, 27)
print(f"Step 5  dcounts (合计): {tuple(dcounts.shape)}")

# ── Step 6: dcounts → dnorm_logits ──────────────────────────
# counts = norm_logits.exp(),  d/dx exp(x) = exp(x)
dnorm_logits = dcounts * counts                             # (32, 27)
print(f"Step 6  dnorm_logits: {tuple(dnorm_logits.shape)}")

# ── Step 7: dnorm_logits → dlogits + dlogit_maxes ───────────
# norm_logits = logits - logit_maxes
dlogits = dnorm_logits.clone()                              # (32, 27)
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)         # (32, 1)
print(f"Step 7  dlogits:       {tuple(dlogits.shape)}")
print(f"        dlogit_maxes:  {tuple(dlogit_maxes.shape)}")

# ── Step 8: dlogit_maxes → dlogits（补充）────────────────────
# logit_maxes = logits.max(1),  只在最大值位置有梯度
# 需要找最大值位置
max_indices = logits.argmax(dim=1)
dlogit_maxes_grad = torch.zeros_like(logits)
dlogit_maxes_grad[torch.arange(batch_size), max_indices] = dlogit_maxes.squeeze()
dlogits += dlogit_maxes_grad
print(f"Step 8  dlogits (合计): {tuple(dlogits.shape)}")

# ── Step 9: dlogits → dh + dW2 + db2 ────────────────────────
# logits = h @ W2 + b2
dh = dlogits @ W2.T                                        # (32, 64)
dW2 = h.T @ dlogits                                        # (64, 27)
db2 = dlogits.sum(0)                                       # (27,)
print(f"Step 9  dh:  {tuple(dh.shape)}")
print(f"        dW2: {tuple(dW2.shape)}")
print(f"        db2: {tuple(db2.shape)}")

# ── Step 10: dh → dhpreact ──────────────────────────────────
# h = tanh(hpreact),  d/dx tanh(x) = 1 - tanh(x)^2
dhpreact = dh * (1.0 - h ** 2)                             # (32, 64)
print(f"Step 10 dhpreact: {tuple(dhpreact.shape)}")

# ── Step 11: dhpreact → dbngain + dbnbias + dbnraw ──────────
# hpreact = bngain * bnraw + bnbias
dbngain = (dhpreact * bnraw).sum(0, keepdim=True)          # (1, 64)
dbnbias = dhpreact.sum(0, keepdim=True)                     # (1, 64)
dbnraw = dhpreact * bngain                                 # (32, 64)
print(f"Step 11 dbngain:  {tuple(dbngain.shape)}")
print(f"        dbnbias:  {tuple(dbnbias.shape)}")
print(f"        dbnraw:   {tuple(dbnraw.shape)}")

# ── Step 12: dbnraw → dhprebn (BatchNorm 反向传播) ──────────
# BN 前向：
#   bnraw = bndiff * bnvar_inv
#   bndiff = hprebn - bnmeani
#   bnvar = bndiff^2.mean(0)
#   bnvar_inv = (bnvar + eps)^{-0.5}
#
# BN 反向（按公式推导）：
dbndiff = dbnraw * bnvar_inv                               # (32, 64)
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)        # (1, 64)
dbnvar = dbnvar_inv * (-0.5) * (bnvar + 1e-5) ** -1.5      # (1, 64)
dbndiff2 = torch.ones_like(bndiff2) * (dbnvar / batch_size) # (32, 64)
dbndiff += dbndiff2 * 2.0 * bndiff                          # (32, 64) 补充
dbnmeani = -dbndiff.sum(0, keepdim=True)                    # (1, 64)
dhprebn = dbndiff + torch.ones_like(hprebn) * (dbnmeani / batch_size)  # (32, 64)
print(f"Step 12 dhprebn: {tuple(dhprebn.shape)}")

# ── 继续：dhprebn → dW1, db1, dembcat, demb, dC ─────────────
# hprebn = embcat @ W1 + b1
dembcat = dhprebn @ W1.T                                   # (32, 30)
dW1 = embcat.T @ dhprebn                                   # (30, 64)
db1 = dhprebn.sum(0)                                       # (64,)
print(f"  Extra dembcat: {tuple(dembcat.shape)}")
print(f"  Extra dW1:     {tuple(dW1.shape)}")
print(f"  Extra db1:     {tuple(db1.shape)}")

# embcat = emb.view(B, -1)
demb = dembcat.view(emb.shape)                              # (32, 3, 10)
print(f"  Extra demb:    {tuple(demb.shape)}")

# emb = C[Xb]
dC = torch.zeros_like(C)
for i in range(batch_size):
    for j in range(block_size):
        dC[Xb[i, j]] += demb[i, j]
print(f"  Extra dC:      {tuple(dC.shape)}")
print()

# ═══════════════════════════════════════════════════════════════
# 验证：手动梯度 vs autograd 梯度
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("梯度验证：手动 vs autograd")
print("=" * 60)

# 比较函数
def cmp(name, dt, t):
    """比较手动梯度 dt 和 autograd 梯度 t"""
    exact = torch.allclose(dt, t, atol=1e-5)
    maxdiff = (dt - t).abs().max().item()
    status = "✅" if exact else "❌"
    print(f"  {status} {name:15s} | max diff = {maxdiff:.2e}")
    return exact

all_ok = True
all_ok &= cmp("dC",       dC,       C.grad)
all_ok &= cmp("dW1",      dW1,      W1.grad)
all_ok &= cmp("db1",      db1,      b1.grad)
all_ok &= cmp("dbngain",  dbngain,  bngain.grad)
all_ok &= cmp("dbnbias",  dbnbias,  bnbias.grad)
all_ok &= cmp("dW2",      dW2,      W2.grad)
all_ok &= cmp("db2",      db2,      b2.grad)

print()
if all_ok:
    print("🎉 全部梯度验证通过！手动推导和 autograd 完全一致。")
else:
    print("⚠️  有些梯度不匹配，检查推导过程。")
