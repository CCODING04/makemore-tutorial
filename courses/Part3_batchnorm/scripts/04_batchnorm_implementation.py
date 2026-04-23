"""
04_batchnorm_implementation.py - 从零实现 BatchNorm

BatchNorm 的核心思想：在训练时，对小批量数据归一化激活值，
使其均值≈0、方差≈1，然后通过可学习的 γ(增益) 和 β(偏移) 恢复表达能力。

公式：
    hpreact_norm = (hpreact - mean) / sqrt(var + eps)
    out = γ * hpreact_norm + β

训练时：使用当前批量的均值和方差，同时更新 running_mean/running_var
推理时：使用训练时积累的 running_mean/running_var

本脚本：
  1. 从零实现 BatchNorm1d
  2. 在 MLP 中插入 BN 层（在 hpreact 之后、tanh 之前）
  3. 训练并评估 train/dev loss
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


# ─── BatchNorm1d 实现 ───────────────────────────────────────────
class BatchNorm1d:
    """
    从零实现的 1D BatchNorm

    参数：
        dim: 特征维度
        eps: 防止除零的小常数
        momentum: running stats 更新速率

    可学习参数：
        gamma (增益): 初始为 1，控制归一化后的缩放
        beta (偏移): 初始为 0，控制归一化后的平移

    非学习参数：
        running_mean: 训练时积累的全局均值
        running_var: 训练时积累的全局方差
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True  # 训练/推理模式标志

        # 可学习参数（用 nn.Parameters 的简化版）
        self.gamma = torch.ones(dim)    # 增益
        self.beta = torch.zeros(dim)    # 偏移

        # 运行时统计量（不是梯度更新的，是指数移动平均）
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # 前向传播：根据 training 标志选择计算方式
        if self.training:
            # 训练模式：用当前批量计算均值和方差
            xmean = x.mean(dim=0, keepdim=True)       # (1, dim)
            xvar = x.var(dim=0, keepdim=True, unbiased=False)  # (1, dim)

            # 更新 running stats（指数移动平均）
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        else:
            # 推理模式：用 running stats
            xmean = self.running_mean.unsqueeze(0)     # (1, dim)
            xvar = self.running_var.unsqueeze(0)       # (1, dim)

        # 归一化
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # 标准化
        self.out = self.gamma * xhat + self.beta            # 缩放平移

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ─── 构建含 BatchNorm 的 MLP ────────────────────────────────────
n_embd = 10
n_hidden = 200
fan_in = n_embd * block_size

g = torch.Generator().manual_seed(42)

C = torch.randn((vocab_size, n_embd), generator=g)

# 隐藏层
W1 = torch.randn((fan_in, n_hidden), generator=g) * (5 / 3) / math.sqrt(fan_in)
b1 = torch.randn(n_hidden, generator=g) * 0.01  # BN 后 b1 会被吸收，但保留

# BatchNorm 层
bn1 = BatchNorm1d(n_hidden)

# 输出层
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, W2, b2] + bn1.parameters()
for p in parameters:
    p.requires_grad = True

total_params = sum(p.nelement() for p in parameters)
print(f"模型参数总量: {total_params}")

# ─── 训练 ───────────────────────────────────────────────────────
max_steps = 20000
batch_size = 32

print("\n═══ 训练含 BatchNorm 的 MLP ═══")
losses = []
for i in range(max_steps):
    # 小批量采样
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # 前向传播
    emb = C[Xb]                                          # (32, 3, 10)
    hpreact = emb.view(-1, fan_in) @ W1 + b1             # (32, 200)
    hpreact = bn1(hpreact)                               # ← BatchNorm 在这里！
    h = torch.tanh(hpreact)                              # (32, 200)
    logits = h @ W2 + b2                                 # (32, 27)
    loss = F.cross_entropy(logits, Yb)

    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()

    # 学习率调度
    lr = 0.1 if i < 10000 else 0.01

    # 参数更新
    for p in parameters:
        p.data += -lr * p.grad

    losses.append(loss.item())
    if (i + 1) % 2000 == 0:
        print(f"  step {i+1:5d} | loss = {loss.item():.4f} | lr = {lr}")


# ─── 评估（切换到推理模式）─────────────────────────────────────
bn1.training = False  # 切换到推理模式！


def eval_split(X, Y, split_name):
    with torch.no_grad():
        emb = C[X]
        hpreact = emb.view(-1, fan_in) @ W1 + b1
        hpreact = bn1(hpreact)
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)
    print(f"  {split_name}: {loss.item():.4f}")
    return loss.item()


print("\n═══ 最终评估 ═══")
train_loss = eval_split(Xtr, Ytr, "训练集")
dev_loss = eval_split(Xdev, Ydev, "验证集")
test_loss = eval_split(Xte, Yte, "测试集")

# ─── 采样生成名字 ───────────────────────────────────────────────
print("\n═══ 采样生成 ═══")
g_gen = torch.Generator().manual_seed(42 + 10)

for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        hpreact = emb.view(1, -1) @ W1 + b1
        hpreact = bn1(hpreact)
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
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

BatchNorm 的作用：
  1. 稳定每层输入的分布（均值≈0，方差≈1）
  2. 使得网络对初始化不那么敏感
  3. 允许使用更大的学习率
  4. 有轻微的正则化效果（批量噪声）

BatchNorm 在网络中的位置：
  Linear → BatchNorm → Activation（tanh/relu）
  hpreact = x @ W + b
  hpreact = BN(hpreact)     ← 在激活函数之前
  h = tanh(hpreact)

注意：
  - 推理时必须切换 training=False，使用 running stats
  - BN 中的 gamma/beta 是可学习参数，通过反向传播更新
  - running_mean/running_var 不是梯度更新的，是指数移动平均
""")
