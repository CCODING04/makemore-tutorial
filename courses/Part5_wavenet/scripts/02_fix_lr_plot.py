#!/usr/bin/env python3
"""
02_fix_lr_plot.py - 修复 loss 曲线可视化

之前 batch_size=32 的 loss 曲线噪声太大，看不出趋势。
本脚本用 view(-1, window).mean(1) 做滑动平均平滑。

关键点：
  1. 原始 loss 曲线：高度噪声，难以判断趋势
  2. 平滑方法：每 1000 步取平均，得到清晰的趋势线
  3. 平滑后可以看到训练的三个阶段：快速下降 → 线性下降 → 平台
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


# ─── 层定义（同 01） ────────────────────────────────────────────
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
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        else:
            xmean = self.running_mean.unsqueeze(0)
            xvar = self.running_var.unsqueeze(0)
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


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
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


# ─── 构建模型 ───────────────────────────────────────────────────
n_embd = 10
n_hidden = 200

model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

for p in model.parameters():
    p.requires_grad = True

# ─── 训练并记录 loss ───────────────────────────────────────────
print("═══ 训练 (20000 步) ═══")
max_steps = 20000
batch_size = 32
losses = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 15000 else 0.01
    for p in model.parameters():
        p.data += -lr * p.grad

    losses.append(loss.item())

    if (i + 1) % 5000 == 0:
        print(f"  step {i+1:5d} | loss = {loss.item():.4f}")

# ═══════════════════════════════════════════════════════════════════
#  平滑 loss 曲线
# ═══════════════════════════════════════════════════════════════════
print(f"\n═══ Loss 曲线平滑 ═══")

losses_tensor = torch.tensor(losses)
print(f"  原始 loss 数量: {len(losses)}")
print(f"  原始 loss 范围: [{losses_tensor.min():.4f}, {losses_tensor.max():.4f}]")

# 方法：每 1000 步取平均
window = 1000
smoothed = losses_tensor.view(-1, window).mean(1)
print(f"\n  平滑后（每 {window} 步取平均）:")
for i, s in enumerate(smoothed):
    step_start = i * window
    step_end = (i + 1) * window
    print(f"    step {step_start:5d}-{step_end:5d}: {s:.4f}")

# 也可以用不同窗口
print(f"\n  不同窗口大小的平滑效果:")
for w in [500, 1000, 2000]:
    if len(losses) % w == 0:
        sm = losses_tensor.view(-1, w).mean(1)
        print(f"    window={w:4d}: {len(sm)} 个点, 范围 [{sm.min():.4f}, {sm.max():.4f}]")

# ─── 评估 ───────────────────────────────────────────────────────
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False

with torch.no_grad():
    logits_tr = model(Xtr)
    loss_tr = F.cross_entropy(logits_tr, Ytr)
    logits_dev = model(Xdev)
    loss_dev = F.cross_entropy(logits_dev, Ydev)

print(f"\n═══ 最终评估 ═══")
print(f"  训练集: {loss_tr.item():.4f}")
print(f"  验证集: {loss_dev.item():.4f}")

print(f"""
═══ 总结 ═══

Loss 曲线平滑方法：
  1. 原始 loss（batch_size=32）噪声非常大
  2. view(-1, 1000).mean(1) 是最简单的滑动平均
  3. 可以看到清晰的趋势：快速下降 → 线性下降 → 收敛
  4. 学习率衰减点（step 15000）在曲线上有明显拐点
""")
