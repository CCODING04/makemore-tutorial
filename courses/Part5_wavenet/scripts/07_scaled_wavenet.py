#!/usr/bin/env python3
"""
07_scaled_wavenet.py - 放大 WaveNet 并训练到 loss < 2.0

把 n_embd 从 10 增大到 24，n_hidden 从 68 增大到 128。
更大的网络容量 + 层次化融合 = 更好的性能。

目标：验证 loss < 2.0
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

block_size = 8

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))


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


Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


# ─── 层定义 ─────────────────────────────────────────────────────
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim_reduce = 0
            else:
                dim_reduce = (0, 1)
            xmean = x.mean(dim=dim_reduce, keepdim=True)
            xvar = x.var(dim=dim_reduce, keepdim=True, unbiased=False)
            with torch.no_grad():
                if x.ndim == 2:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze(0)
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze(0)
                else:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze((0, 1))
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze((0, 1))
        else:
            if x.ndim == 2:
                xmean = self.running_mean.unsqueeze(0)
                xvar = self.running_var.unsqueeze(0)
            else:
                xmean = self.running_mean.unsqueeze(0).unsqueeze(0)
                xvar = self.running_var.unsqueeze(0).unsqueeze(0)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ═══════════════════════════════════════════════════════════════════
#  放大的 WaveNet
# ═══════════════════════════════════════════════════════════════════

n_embd = 24    # 从 10 增大到 24
n_hidden = 128  # 从 68 增大到 128

print("═══ 放大 WaveNet ═══\n")
print(f"  n_embd={n_embd}, n_hidden={n_hidden}, block_size={block_size}")

model = Sequential([
    Embedding(vocab_size, n_embd),
    # ---- 层 1：8→4 ----
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # ---- 层 2：4→2 ----
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # ---- 层 3：2→1 ----
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # ---- 输出层 ----
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

for p in model.parameters():
    p.requires_grad = True

total_params = sum(p.nelement() for p in model.parameters())
print(f"  参数总量: {total_params:,}")

# ─── 打印模型结构 ───────────────────────────────────────────────
print(f"\n═══ 模型结构 ═══")
for i, layer in enumerate(model.layers):
    name = layer.__class__.__name__
    if isinstance(layer, (Linear, Embedding)):
        print(f"  [{i:2d}] {name}: weight {layer.weight.shape}")
    elif isinstance(layer, BatchNorm1d):
        print(f"  [{i:2d}] {name}: dim={layer.gamma.shape[0]}")
    elif isinstance(layer, FlattenConsecutive):
        print(f"  [{i:2d}] {name}: n={layer.n}")
    else:
        print(f"  [{i:2d}] {name}")

# ─── 训练 ───────────────────────────────────────────────────────
import sys
QUICK = "--quick" in sys.argv
max_steps = 1000 if QUICK else 50000
batch_size = 128  # 更大的 batch size 加速训练

if QUICK:
    print("⚡ Quick 模式：只训练 1000 步（完整训练去掉 --quick）")

print(f"\n═══ 训练 ({max_steps} 步, batch_size={batch_size}) ═══")

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    # 学习率衰减
    lr = 0.1 if i < 30000 else (0.05 if i < 40000 else 0.01)
    for p in model.parameters():
        p.data += -lr * p.grad

    if (i + 1) % 5000 == 0:
        # 快速评估
        for layer in model.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        with torch.no_grad():
            dev_logits = model(Xdev)
            dev_loss = F.cross_entropy(dev_logits.view(-1, dev_logits.shape[-1]), Ydev)
        for layer in model.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        print(f"  step {i+1:5d} | train loss = {loss.item():.4f} | dev loss = {dev_loss.item():.4f}")

# ─── 最终评估 ───────────────────────────────────────────────────
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False

print(f"\n═══ 最终评估 ═══")
with torch.no_grad():
    for name, X, Y in [("训练集", Xtr, Ytr), ("验证集", Xdev, Ydev), ("测试集", Xte, Yte)]:
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Y)
        print(f"  {name}: {loss.item():.4f}")

# ─── 采样 ───────────────────────────────────────────────────────
print(f"\n═══ 采样生成 ═══")
g_gen = torch.Generator().manual_seed(42)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        with torch.no_grad():
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits.view(-1, logits.shape[-1]), dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_gen).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

print(f"""
═══ 总结 ═══

放大的 WaveNet：
  n_embd=24, n_hidden=128
  参数量: {total_params:,}
  
  层次融合：8 chars → 4 bigrams → 2 fourgrams → 1 eightgram
  
  对比：
  - Part 2 MLP (block_size=3):         dev loss ≈ 2.10
  - Part 3 深层 (block_size=3):         dev loss ≈ 2.07
  - WaveNet 小模型 (block_size=8):      dev loss ≈ 2.07
  - WaveNet 放大 (block_size=8):        dev loss ≈ 1.99

  关键提升来源：
  1. 更长的上下文 (8 vs 3)
  2. 层次化融合（WaveNet 结构）
  3. 更大的模型容量
""")
