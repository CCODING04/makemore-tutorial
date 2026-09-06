#!/usr/bin/env python3
"""
03_increase_context.py - 扩大上下文窗口 block_size 3→8

之前用 3 个字符预测下一个，现在扩展到 8 个字符。
这是 WaveNet 的前提——更多上下文意味着更多信息可以利用。

关键观察（seed=42 实测，CPU，20K 步）：
  1. block_size=3 → 验证 loss ≈ 2.17（02 脚本同配置实测）
  2. block_size=8 → 验证 loss ≈ 2.11（本脚本实测；更多上下文确实有帮助）
  3. 视频里更长训练（200K 步）能到 ~2.05；本仓库 20K 步档复现不到
  4. 展平 8 个字符给 Linear 层并不吃亏——层次化融合的收益要放大
     模型/加长训练才明显（见 05/07 脚本）

运行时长预期（CPU）：默认档 20000 步约 5-8 分钟；
自定义步数用环境变量 STEPS（如 STEPS=2000，约 1 分钟）。
默认档（不设 STEPS）行为与输出和旧版完全一致。
"""

import os
import math
import functools
import torch
import torch.nn.functional as F

# 所有 print 实时刷新；不改变输出内容
print = functools.partial(print, flush=True)

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

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))


# ─── 关键变化：block_size = 8 ──────────────────────────────────
block_size = 8


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

print(f"═══ 数据集 (block_size={block_size}) ═══")
print(f"  训练集: {Xtr.shape}")
print(f"  验证集: {Xdev.shape}")
print(f"  测试集: {Xte.shape}")


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

print(f"\n═══ 模型配置 ═══")
print(f"  n_embd={n_embd}, n_hidden={n_hidden}, block_size={block_size}")
print(f"  Linear 输入维度: {n_embd * block_size} = {n_embd} × {block_size}")

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

total_params = sum(p.nelement() for p in model.parameters())
print(f"  参数总量: {total_params:,}")

# ─── 训练 ───────────────────────────────────────────────────────
# 档位：环境变量 STEPS=N → N 步；默认 20000 步（行为与旧版一致）
STEPS_ENV = os.environ.get("STEPS")
max_steps = max(1, int(STEPS_ENV)) if STEPS_ENV else 20000
batch_size = 32

# 打印间隔：默认档每 5000 步（与旧版一致）；短程档按 max_steps//5
log_every = 5000 if max_steps >= 5000 else max(1, max_steps // 5)
# lr 衰减点：默认档在 15000 步（与旧版一致）；短程档按 75% 步数等比缩放
lr_decay_at = 15000 if max_steps >= 20000 else int(max_steps * 0.75)

if STEPS_ENV:
    print(f"⚡ STEPS 短程档：只训练 {max_steps} 步（完整训练去掉 STEPS 环境变量）")
print(f"\n═══ 训练 ({max_steps} 步) ═══")

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    lr = 0.1 if i < lr_decay_at else 0.01
    for p in model.parameters():
        p.data += -lr * p.grad

    if (i + 1) % log_every == 0:
        print(f"  step {i+1:5d} | loss = {loss.item():.4f}")

# ─── 评估 ───────────────────────────────────────────────────────
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False

print(f"\n═══ 评估 ═══")
with torch.no_grad():
    for name, X, Y in [("训练集", Xtr, Ytr), ("验证集", Xdev, Ydev), ("测试集", Xte, Yte)]:
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        print(f"  {name}: {loss.item():.4f}")

# ─── 采样 ───────────────────────────────────────────────────────
print(f"\n═══ 采样生成 ═══")
g_gen = torch.Generator().manual_seed(42)

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        with torch.no_grad():
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_gen).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

print(f"""
═══ 总结 ═══

block_size 从 3 增大到 8：
  - 更多上下文 → 更好的预测能力（实测 dev ≈2.17 → ≈2.11，20K 步档）
  - 展平并不吃亏：展平 MLP 与 WaveNet 小模型同预算下几乎打平（03 vs 05 脚本）
  - Linear 层直接处理 80 维输入，信息是"一次性"混合的
  - 下一步：WaveNet 的层次化融合，逐步从 2-gram → 4-gram → 8-gram
    （首层参数 20×200，只有展平 80×200 的 1/4；收益在放大后显现）
""")
