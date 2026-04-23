"""
05_deep_network.py - 构建更深的网络

之前是 1 层隐藏层的浅网络，本脚本构建多层深度网络：
    Embedding → [Linear → BatchNorm → Tanh] × N → Linear → Output

使用 Python 列表动态管理层，构建灵活的深层网络。

关键点：
  1. 用列表管理各层参数
  2. 每层用 Kaiming 初始化
  3. 每个线性层后接 BatchNorm → Tanh
  4. 输出层（最后一层）不加 BN 和激活
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


# ─── BatchNorm1d（同 Part 04）───────────────────────────────────
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


# ─── 网络配置 ───────────────────────────────────────────────────
n_embd = 10
n_hidden = 100  # 每层 100 个神经元
layers_config = [n_embd * block_size, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, vocab_size]
# 层大小: [30, 100, 100, 100, 100, 100, 27]

print(f"网络层数配置: {layers_config}")
print(f"隐藏层数: {len(layers_config) - 2}")

# ─── 构建层 ─────────────────────────────────────────────────────
g = torch.Generator().manual_seed(42)

C = torch.randn((vocab_size, n_embd), generator=g)

layers = []  # 用列表存放所有层
for i in range(len(layers_config) - 1):
    fan_in_i = layers_config[i]
    fan_out_i = layers_config[i + 1]

    # 线性层
    W = torch.randn((fan_in_i, fan_out_i), generator=g) * (5 / 3) / math.sqrt(fan_in_i)
    b = torch.zeros(fan_out_i)

    layers.append({'type': 'linear', 'W': W, 'b': b})

    # 除最后一层外，加 BatchNorm + Tanh
    if i < len(layers_config) - 2:
        bn = BatchNorm1d(fan_out_i)
        layers.append({'type': 'batchnorm', 'bn': bn})
        layers.append({'type': 'activation', 'act': 'tanh'})

# 收集所有参数
parameters = [C]
for layer in layers:
    if layer['type'] == 'linear':
        parameters.append(layer['W'])
        parameters.append(layer['b'])
    elif layer['type'] == 'batchnorm':
        parameters.extend(layer['bn'].parameters())

for p in parameters:
    p.requires_grad = True

total_params = sum(p.nelement() for p in parameters)
print(f"模型参数总量: {total_params:,}")

# ─── 前向传播函数 ───────────────────────────────────────────────


def forward(X, training=True):
    """完整前向传播"""
    emb = C[X]  # (B, block_size, n_embd)
    x = emb.view(emb.shape[0], -1)  # (B, fan_in)

    for layer in layers:
        if layer['type'] == 'linear':
            x = x @ layer['W'] + layer['b']
        elif layer['type'] == 'batchnorm':
            layer['bn'].training = training
            x = layer['bn'](x)
        elif layer['type'] == 'activation':
            if layer['act'] == 'tanh':
                x = torch.tanh(x)

    return x


# ─── 训练 ───────────────────────────────────────────────────────
max_steps = 20000
batch_size = 32

print("\n═══ 训练深层网络 ═══")
losses = []
update_data_ratios = []  # 记录更新/数据比率

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = forward(Xb, training=True)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    # 学习率调度
    lr = 0.1 if i < 10000 else 0.01

    # 更新参数
    for p in parameters:
        p.data += -lr * p.grad

    losses.append(loss.item())
    if (i + 1) % 2000 == 0:
        print(f"  step {i+1:5d} | loss = {loss.item():.4f} | lr = {lr}")

# ─── 评估 ───────────────────────────────────────────────────────


def eval_split(X, Y, split_name):
    with torch.no_grad():
        logits = forward(X, training=False)
        loss = F.cross_entropy(logits, Y)
    print(f"  {split_name}: {loss.item():.4f}")
    return loss.item()


print("\n═══ 最终评估 ═══")
train_loss = eval_split(Xtr, Ytr, "训练集")
dev_loss = eval_split(Xdev, Ydev, "验证集")
test_loss = eval_split(Xte, Yte, "测试集")

# ─── 采样生成 ───────────────────────────────────────────────────
print("\n═══ 采样生成 ═══")
g_gen = torch.Generator().manual_seed(42 + 10)

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        with torch.no_grad():
            logits = forward(torch.tensor([context]), training=False)
            probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_gen).item()
        context = context[1:] + [ix]
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# ─── 总结 ───────────────────────────────────────────────────────
print(f"""
═══ 总结 ═══

深层网络架构：
  Embedding → [Linear → BN → Tanh] × {len(layers_config)-2} → Linear

  - 使用列表动态管理层，灵活配置深度
  - 每层 Kaiming 初始化 + BatchNorm → 训练稳定
  - 参数总量: {total_params:,}

为什么 BatchNorm 使深层网络成为可能：
  1. 每层输出被归一化 → 梯度不会爆炸/消失
  2. 网络对初始化更鲁棒
  3. 可以用更大的学习率加速训练
  4. 深层网络不再"难训练"
""")
