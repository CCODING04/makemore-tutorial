#!/usr/bin/env python3
"""Part 3 作业：训练诊断与 BatchNorm —— S2 作答"""

import os
import math
import random
import torch
import torch.nn.functional as F


def _build_dataset(words, block_size=3):
    stoi = {s: i + 1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
    stoi['.'] = 0
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, dtype=torch.int64), torch.tensor(Y, dtype=torch.int64)


def diagnose_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    X, Y = _build_dataset(words, block_size)
    C = torch.randn(27, n_embd, generator=g)
    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn(n_hidden, 27, generator=g)
    b2 = torch.randn(27, generator=g)
    with torch.no_grad():
        emb = C[X]
        emb_cat = emb.view(emb.shape[0], -1)
        h = torch.tanh(emb_cat @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)
    return loss.item()


def fix_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    X, Y = _build_dataset(words, block_size)
    C = torch.randn(27, n_embd, generator=g)
    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn(n_hidden, 27, generator=g)
    b2 = torch.randn(27, generator=g)
    # 关键修正
    W2 = W2 * 0.01
    b2 = torch.zeros(27)
    with torch.no_grad():
        emb = C[X]
        emb_cat = emb.view(emb.shape[0], -1)
        h = torch.tanh(emb_cat @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)
    return loss.item()


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True, unbiased=False)
            xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
            self.out = self.gamma * xhat + self.beta
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        else:
            xhat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


def diagnose_tanh_saturation(hpreact):
    h = torch.tanh(hpreact)
    saturated = (h.abs() > 0.99).sum().item()
    total = h.numel()
    return saturated / total


def train_deep_bn(words, block_size=3, n_embd=10, n_hidden=200,
                  steps=200000, seed=2147483647):
    # 1. 划分数据
    random.seed(seed)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    Xtr, Ytr = _build_dataset(words[:n1], block_size)
    Xdev, Ydev = _build_dataset(words[n1:n2], block_size)

    # 2. 初始化参数（最佳实践）
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(27, n_embd, generator=g)
    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g) * (5 / 3) / (block_size * n_embd) ** 0.5
    b1 = torch.zeros(n_hidden)
    W2 = torch.randn(n_hidden, 27, generator=g) * 0.01
    b2 = torch.zeros(27)
    bn = BatchNorm1d(n_hidden)
    params = [C, W1, W2, b2] + bn.parameters()
    for p in params:
        p.requires_grad = True

    # 3. 训练
    batch_size = 32
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        emb = C[Xb].view(Xb.shape[0], -1)
        hpreact = emb @ W1 + b1
        hpreact = bn(hpreact)
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        for p in params:
            p.grad = None
        loss.backward()

        lr = 0.1 if i < 100000 else 0.01
        for p in params:
            p.data -= lr * p.grad

        if (i + 1) % 10000 == 0:
            print(f"  step {i+1:6d} | train loss = {loss.item():.4f}")

    # 4. eval 模式评估 dev loss
    bn.training = False
    with torch.no_grad():
        emb = C[Xdev].view(Xdev.shape[0], -1)
        hpreact = emb @ W1 + b1
        hpreact = bn(hpreact)
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        dev_loss = F.cross_entropy(logits, Ydev)

    return dev_loss.item(), {'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'bn': bn}


if __name__ == '__main__':
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')
    words = open(_DATA_PATH, 'r').read().splitlines()

    loss1 = diagnose_initial_loss(words)
    print(f"题 1 — 未经修正的初始 loss: {loss1:.4f}  (预期 >> 3.298)")

    loss2 = fix_initial_loss(words)
    print(f"题 2 — 修正后的初始 loss:   {loss2:.4f}  (预期 ≈ 3.298)")

    torch.manual_seed(42)
    bn = BatchNorm1d(10)
    x = torch.randn(4, 10)
    y = bn(x)
    print(f"题 3 — BatchNorm1d 输出 shape: {y.shape}")
    print(f"       参数数量: {len(bn.parameters())}")

    torch.manual_seed(42)
    hpreact = torch.randn(100, 200) * 3
    ratio = diagnose_tanh_saturation(hpreact)
    print(f"题 4 — Tanh 饱和比例: {ratio:.4f}")
