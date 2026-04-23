#!/usr/bin/env python3
"""
01_pytorchify_layers.py - PyTorch 化代码：用模块化的层重构网络

Part 3 的代码用了字典手动管理层，本脚本把各层抽象成类：
  - Embedding：词嵌入层
  - Flatten：展平层（view 的封装）
  - Sequential：顺序容器（替代手动 for 循环）

然后用 Sequential 重构 Part 3 的网络，验证 forward pass 结果一致。

关键洞察：
  1. nn.Module 风格的 parameters() 接口统一参数收集
  2. Sequential 让 forward 逻辑更清晰
  3. BatchNorm 的 training 标志需要显式切换
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


# ═══════════════════════════════════════════════════════════════════
#  PyTorch 化的层
# ═══════════════════════════════════════════════════════════════════

class Linear:
    """线性层: y = x @ W + b"""

    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming 初始化
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
    """一维批归一化"""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # 可训练参数
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # running statistics
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
    """Tanh 激活函数"""

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    """词嵌入层"""

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    """展平层：把最后两个维度合并（view 的封装）"""

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class Sequential:
    """顺序容器：依次调用各层"""

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
#  用 Sequential 重构网络
# ═══════════════════════════════════════════════════════════════════

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

# 参数初始化（最后一层缩小权重）
with torch.no_grad():
    model.layers[-1].weight *= 0.1  # 输出层小权重

# 启用梯度
for p in model.parameters():
    p.requires_grad = True

total_params = sum(p.nelement() for p in model.parameters())
print(f"模型参数总量: {total_params:,}")

# ─── 验证 forward pass ─────────────────────────────────────────
print("\n═══ 验证 forward pass ═══")
ix = torch.randint(0, Xtr.shape[0], (4,))
Xb = Xtr[ix]
logits = model(Xb)
print(f"  输入 shape:  {Xb.shape}")
print(f"  输出 shape:  {logits.shape}")
print(f"  期望输出 shape: (4, {vocab_size})")

# 验证每层输出
print("\n  各层输出形状:")
x = Xb
for i, layer in enumerate(model.layers):
    x = layer(x)
    name = layer.__class__.__name__
    print(f"    Layer {i} ({name:>12s}): {x.shape}")

# ─── 快速训练测试 ───────────────────────────────────────────────
print("\n═══ 快速训练测试 (2000 步) ═══")
max_steps = 2000
batch_size = 32

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 1000 else 0.01
    for p in model.parameters():
        p.data += -lr * p.grad

    if (i + 1) % 500 == 0:
        print(f"  step {i+1:5d} | loss = {loss.item():.4f}")

# ─── 评估 ───────────────────────────────────────────────────────
print("\n═══ 评估 ═══")
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False

with torch.no_grad():
    for name, X, Y in [("训练集", Xtr, Ytr), ("验证集", Xdev, Ydev)]:
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        print(f"  {name}: {loss.item():.4f}")

# ─── 采样 ───────────────────────────────────────────────────────
print("\n═══ 采样生成 ═══")
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

PyTorch 化重构要点：
  1. 每个层是一个类，有 __call__ 和 parameters() 方法
  2. Embedding 层封装了 C[IX] 操作
  3. Flatten 层封装了 view 操作
  4. Sequential 容器统一了 forward 逻辑
  5. 参数收集只需 model.parameters() 一行

接下来要做的是：
  → 把网络从"展平所有上下文"改成"层次化融合"（WaveNet！）
""")
