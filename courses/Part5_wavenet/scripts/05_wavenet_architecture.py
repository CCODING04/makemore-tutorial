#!/usr/bin/env python3
"""
05_wavenet_architecture.py - 构建 WaveNet 层次化架构

把前面学到的所有组件组合起来，构建一个层次化的 WaveNet 模型：
  Embedding → [FlattenConsecutive(2) → Linear → BN → Tanh] × 3 → Linear → 输出

关键架构：
  - block_size = 8（8 个字符上下文）
  - 3 层 FlattenConsecutive(2)：8→4→2→1
  - 每层融合两个相邻位置的 embedding
  - 最终得到一个 8-gram 的表征

性能（seed=42 实测，CPU，20K 步）：参数量 22,397，dev loss ≈ 2.10；
与同预算展平 MLP（03 脚本，dev ≈ 2.11）几乎打平——层次化的收益要放大
模型/加长训练才明显（见 07 脚本）。

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
                # 3D 输入时，在 batch 和 time 维度上同时 reduce
                dim_reduce = (0, 1)
            xmean = x.mean(dim=dim_reduce, keepdim=True)
            xvar = x.var(dim=dim_reduce, keepdim=True, unbiased=False)
            with torch.no_grad():
                if x.ndim == 2:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
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
        assert T % self.n == 0, f"T={T} 不能被 n={self.n} 整除（block_size 必须是 n 的倍数）"
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
#  构建 WaveNet
# ═══════════════════════════════════════════════════════════════════

n_embd = 10
n_hidden = 68  # 控制参数量约 22K

print("═══ WaveNet 架构 ═══\n")
print(f"  block_size={block_size}, n_embd={n_embd}, n_hidden={n_hidden}")
print()

# 层次化结构（B=批大小）：
# Embedding → (B, 8, 10)
# FC(2) → (B, 4, 20) → Linear(20, 68) → BN → Tanh → (B, 4, 68)
# FC(2) → (B, 2, 136) → Linear(136, 68) → BN → Tanh → (B, 2, 68)
# FC(2) → (B, 1, 136) → Linear(136, 68) → BN → Tanh → (B, 1, 68)
# Linear(68, 27) → (B, 1, 27)；训练时用 view(-1, 27) 收成 (B, 27) 算交叉熵
# （末级 T=1，Linear 直接作用在 (B, 1, 68) 上，无需再接 Flatten 层）

model = Sequential([
    Embedding(vocab_size, n_embd),
    # ---- 层 1：8 chars → 4 bigrams ----
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # ---- 层 2：4 bigrams → 2 fourgrams ----
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # ---- 层 3：2 fourgrams → 1 eightgram ----
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    # ---- 输出层 ----
    Linear(n_hidden, vocab_size),
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

for p in model.parameters():
    p.requires_grad = True

total_params = sum(p.nelement() for p in model.parameters())
print(f"参数总量: {total_params:,}")

# ─── 打印每层 tensor 形状 ──────────────────────────────────────
print(f"\n═══ 每层 Tensor 形状 ═══")
ix = torch.randint(0, Xtr.shape[0], (4,))
x_sample = Xtr[ix]
x = model.layers[0](x_sample)  # Embedding
print(f"  Embedding:     {x.shape}")

for i in range(1, len(model.layers)):
    x = model.layers[i](x)
    name = model.layers[i].__class__.__name__
    print(f"  {name:>18s}: {x.shape}")

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
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Yb)

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
    for name, X, Y in [("训练集", Xtr, Ytr), ("验证集", Xdev, Ydev)]:
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Y)
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
            probs = F.softmax(logits.view(-1, logits.shape[-1]), dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_gen).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

print(f"""
═══ 总结 ═══

WaveNet 架构：
  Embedding → [FC(2) → Linear → BN → Tanh] × 3 → Linear → 输出

  层次融合过程：
    8 chars → 4 bigrams → 2 fourgrams → 1 eightgram
    
  参数量: {total_params:,}
  验证 loss: ≈2.10（seed=42 / 20K 步实测；同预算展平 MLP ≈2.11，几乎打平）

  下一步：放大网络 n_embd=24, n_hidden=128、训练 50K 步 → 07 脚本
""")
