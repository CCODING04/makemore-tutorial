#!/usr/bin/env python3
"""Part 3 作业：训练诊断与 BatchNorm"""

import os
import math
import random
import torch
import torch.nn.functional as F


def _build_dataset(words, block_size=3):
    """
    辅助函数：构建数据集（与作业 2 相同）

    Args:
        words: 名字列表 (list of str)
        block_size: 上下文长度
    Returns:
        X: (N, block_size) int64
        Y: (N,) int64
    """
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
    """
    题 1：诊断初始 loss

    构建标准 MLP，用默认随机初始化，forward 一次返回 loss。
    预期 loss >> ln(27) ≈ 3.298，说明初始化有问题。

    Args:
        words: 名字列表
        block_size: 上下文长度
        n_embd: Embedding 维度
        n_hidden: 隐藏层大小
        seed: 随机种子
    Returns:
        loss: 初始 loss 值 (float)
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 固定随机种子
    #    g = torch.Generator().manual_seed(seed)
    #
    # 2. 用 _build_dataset 构建全部数据的 X, Y
    #
    # 3. 初始化参数（全部用标准正态随机初始化，不做任何缩放）：
    #    C  = torch.randn(27, n_embd, generator=g)
    #    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g)
    #    b1 = torch.randn(n_hidden, generator=g)
    #    W2 = torch.randn(n_hidden, 27, generator=g)
    #    b2 = torch.randn(27, generator=g)
    #
    # 4. Forward（不需要梯度）：
    #    with torch.no_grad():
    #        emb = C[X]                        # (N, block_size, n_embd)
    #        emb_cat = emb.view(emb.shape[0], -1)  # (N, block_size*n_embd)
    #        h = torch.tanh(emb_cat @ W1 + b1)     # (N, n_hidden)
    #        logits = h @ W2 + b2                   # (N, 27)
    #        loss = F.cross_entropy(logits, Y)
    #
    # 5. 返回 loss.item()
    pass


def fix_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647):
    """
    题 2：修正初始 loss

    在题 1 的基础上对输出层做初始化修正：
    - W2 *= 0.01（缩小权重，让 logits 趋近零）
    - b2 = torch.zeros(27)（偏置归零）
    预期 loss ≈ ln(27) ≈ 3.298

    Args:
        words: 名字列表
        block_size: 上下文长度
        n_embd: Embedding 维度
        n_hidden: 隐藏层大小
        seed: 随机种子
    Returns:
        loss: 修正后的初始 loss (float)
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 固定随机种子（与题 1 相同）
    #
    # 2. 用 _build_dataset 构建全部数据的 X, Y
    #
    # 3. 初始化参数（与题 1 相同的初始化顺序），然后修正输出层：
    #    g = torch.Generator().manual_seed(seed)
    #    C  = torch.randn(27, n_embd, generator=g)
    #    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g)
    #    b1 = torch.randn(n_hidden, generator=g)
    #    W2 = torch.randn(n_hidden, 27, generator=g)
    #    b2 = torch.randn(27, generator=g)
    #    # 👇 关键修正
    #    W2 = W2 * 0.01
    #    b2 = torch.zeros(27)
    #
    # 4. Forward（与题 1 相同）
    #
    # 5. 返回 loss.item()
    pass


class BatchNorm1d:
    """
    题 3：从零实现 BatchNorm1d

    一维批归一化层，支持 training 和 eval 两种模式。

    Args:
        dim: 特征维度（即 n_hidden）
        eps: 防止除零的小常数
        momentum: running statistics 的更新速率
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        # TODO: 初始化以下属性
        #
        # self.eps = eps
        # self.momentum = momentum
        # self.training = True  # 默认训练模式
        #
        # 可训练参数（用 requires_grad=True 的 tensor）：
        # self.gamma = torch.ones(dim, requires_grad=True)   # 缩放因子
        # self.beta = torch.zeros(dim, requires_grad=True)   # 平移因子
        #
        # Running statistics（不需要梯度）：
        # self.running_mean = torch.zeros(dim)
        # self.running_var = torch.ones(dim)
        pass

    def __call__(self, x):
        """
        前向传播

        Args:
            x: (N, dim) 输入 tensor
        Returns:
            out: (N, dim) 归一化后的输出
        """
        # TODO: 根据 self.training 选择不同路径
        #
        # if self.training:
        #     # === Training 模式 ===
        #     # 1. 计算 batch 均值和方差
        #     #    xmean = x.mean(dim=0, keepdim=True)          # (1, dim)
        #     #    xvar = x.var(dim=0, keepdim=True, unbiased=False)  # (1, dim)
        #     #
        #     # 2. 标准化
        #     #    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # (N, dim)
        #     #
        #     # 3. 缩放和平移
        #     #    self.out = self.gamma * xhat + self.beta            # (N, dim)
        #     #
        #     # 4. 更新 running statistics（不要计算梯度！）
        #     #    with torch.no_grad():
        #     #        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        #     #        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        #
        # else:
        #     # === Eval 模式 ===
        #     # 直接使用 running statistics
        #     xhat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        #     self.out = self.gamma * xhat + self.beta
        #
        # return self.out
        pass

    def parameters(self):
        """返回可训练参数列表"""
        # TODO: 返回 [self.gamma, self.beta]
        pass


def diagnose_tanh_saturation(hpreact):
    """
    题 4：诊断 tanh 饱和

    计算隐藏层 tanh 输出中，饱和（|h| > 0.99）的比例。

    Args:
        hpreact: (N, n_hidden) 隐藏层线性输出（tanh 之前）
    Returns:
        saturation_ratio: 饱和比例 (float)，范围 [0, 1]
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 计算 tanh 输出：
    #    h = torch.tanh(hpreact)  # (N, n_hidden)
    #
    # 2. 统计 |h| > 0.99 的元素数：
    #    saturated = (h.abs() > 0.99).sum().item()
    #    total = h.numel()
    #
    # 3. 计算比例并返回：
    #    return saturated / total
    pass


def train_deep_bn(words, block_size=3, n_embd=10, n_hidden=200,
                  steps=200000, seed=2147483647):
    """
    题 5（拓展）：含 BatchNorm 的深层网络

    构建含 BN 的 MLP，训练到 dev loss < 2.1。

    Args:
        words: 名字列表
        block_size: 上下文长度
        n_embd: Embedding 维度
        n_hidden: 隐藏层大小
        steps: 训练步数
        seed: 随机种子
    Returns:
        dev_loss: 验证集 loss (float)
        params: dict，包含所有训练好的参数和 BN 层
    """
    # TODO: 实现这个函数（拓展题）
    #
    # 步骤：
    # 1. 划分数据集（80/10/10）：
    #    random.seed(seed)
    #    random.shuffle(words)
    #    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    #    Xtr, Ytr = _build_dataset(words[:n1], block_size)
    #    Xdev, Ydev = _build_dataset(words[n1:n2], block_size)
    #
    # 2. 初始化参数（应用最佳实践）：
    #    g = torch.Generator().manual_seed(seed)
    #    C  = torch.randn(27, n_embd, generator=g)
    #    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g) * (5/3) / (block_size * n_embd) ** 0.5
    #    b1 = torch.zeros(n_hidden)  # BN 有 beta，不需要偏置
    #    W2 = torch.randn(n_hidden, 27, generator=g) * 0.01
    #    b2 = torch.zeros(27)
    #    bn = BatchNorm1d(n_hidden)
    #    params = [C, W1, W2, b2] + bn.parameters()
    #    for p in params:
    #        p.requires_grad = True
    #
    # 3. 训练循环：
    #    batch_size = 32
    #    for i in range(steps):
    #        # Mini-batch 采样
    #        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        #        Xb, Yb = Xtr[ix], Ytr[ix]
    #
    #        # Forward
    #        emb = C[Xb].view(Xb.shape[0], -1)
    #        hpreact = emb @ W1 + b1
    #        hpreact = bn(hpreact)     # ← BatchNorm 在 tanh 之前！
    #        h = torch.tanh(hpreact)
    #        logits = h @ W2 + b2
    #        loss = F.cross_entropy(logits, Yb)
    #
    #        # Backward
    #        for p in params:
    #            p.grad = None
    #        loss.backward()
    #
    #        # 学习率衰减
    #        lr = 0.1 if i < 100000 else 0.01
    #
    #        # Update
    #        for p in params:
    #            p.data -= lr * p.grad
    #
    # 4. 评估 dev loss（eval 模式！）：
    #    bn.training = False  # 切换到 eval 模式
    #    with torch.no_grad():
    #        emb = C[Xdev].view(Xdev.shape[0], -1)
    #        hpreact = emb @ W1 + b1
    #        hpreact = bn(hpreact)
    #        h = torch.tanh(hpreact)
    #        logits = h @ W2 + b2
    #        dev_loss = F.cross_entropy(logits, Ydev)
    #
    # 5. 返回 dev_loss.item() 和参数字典
    pass


if __name__ == '__main__':
    # 加载数据（基于脚本位置解析路径）
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')
    words = open(_DATA_PATH, 'r').read().splitlines()

    # 题 1：初始 loss 诊断
    loss1 = diagnose_initial_loss(words)
    print(f"题 1 — 未经修正的初始 loss: {loss1:.4f}  (预期 >> 3.298)")

    # 题 2：修正后的初始 loss
    loss2 = fix_initial_loss(words)
    print(f"题 2 — 修正后的初始 loss:   {loss2:.4f}  (预期 ≈ 3.298)")

    # 题 3：BatchNorm1d
    torch.manual_seed(42)
    bn = BatchNorm1d(10)
    x = torch.randn(4, 10)
    y = bn(x)
    print(f"题 3 — BatchNorm1d 输出 shape: {y.shape}")
    print(f"       参数数量: {len(bn.parameters())}")

    # 题 4：Tanh 饱和诊断
    torch.manual_seed(42)
    hpreact = torch.randn(100, 200) * 3  # 乘 3 增加饱和
    ratio = diagnose_tanh_saturation(hpreact)
    print(f"题 4 — Tanh 饱和比例: {ratio:.4f}")
