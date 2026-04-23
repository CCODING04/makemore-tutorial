#!/usr/bin/env python3
"""Part 5 作业：WaveNet — 层次化融合"""

import os
import math
import torch
import torch.nn.functional as F


def _build_dataset(words, block_size=8):
    """
    辅助函数：构建数据集

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


# ═══════════════════════════════════════════════════════════════════
#  基础层定义（题 2 需要用到）
# ═══════════════════════════════════════════════════════════════════

class Linear:
    """线性层: y = x @ W + b"""

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
#  题 1：FlattenConsecutive 层
# ═══════════════════════════════════════════════════════════════════

class FlattenConsecutive:
    """
    题 1：把连续 n 个位置的 embedding 拼接成一个。

    输入:  (B, T, C)
    输出:  (B, T//n, C*n)
    """

    def __init__(self, n):
        # TODO: 保存 n
        # self.n = n
        pass

    def __call__(self, x):
        """
        前向传播：reshape (B, T, C) → (B, T//n, C*n)

        Args:
            x: (B, T, C) 输入 tensor
        Returns:
            (B, T//n, C*n) 输出 tensor
        """
        # TODO: 实现这个方法
        #
        # 步骤：
        # 1. 获取 B, T, C = x.shape
        # 2. assert T % self.n == 0
        # 3. x = x.view(B, T // self.n, C * self.n)
        # 4. self.out = x
        # 5. return self.out
        pass

    def parameters(self):
        """返回空列表（无可训练参数）"""
        # TODO: return []
        pass


# ═══════════════════════════════════════════════════════════════════
#  题 3：支持 3D 输入的 BatchNorm1d
# ═══════════════════════════════════════════════════════════════════

class BatchNorm1d3D:
    """
    题 3：支持 2D 和 3D 输入的 BatchNorm1d

    2D 输入 (B, C): 在 dim=0 上 reduce
    3D 输入 (B, T, C): 在 dim=(0,1) 上 reduce
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        # TODO: 初始化以下属性
        #
        # self.eps = eps
        # self.momentum = momentum
        # self.training = True
        # self.gamma = torch.ones(dim)
        # self.beta = torch.zeros(dim)
        # self.running_mean = torch.zeros(dim)
        # self.running_var = torch.ones(dim)
        pass

    def __call__(self, x):
        """
        前向传播，支持 2D 和 3D 输入。

        Args:
            x: (B, C) 或 (B, T, C)
        Returns:
            归一化后的输出，shape 同输入
        """
        # TODO: 根据 self.training 和 x.ndim 选择不同路径
        #
        # if self.training:
        #     if x.ndim == 2:
        #         dim_reduce = 0
        #     else:
        #         dim_reduce = (0, 1)  # 同时在 batch 和 time 上归一化
        #     xmean = x.mean(dim=dim_reduce, keepdim=True)
        #     xvar = x.var(dim=dim_reduce, keepdim=True, unbiased=False)
        #     # 更新 running stats
        #     with torch.no_grad():
        #         if x.ndim == 2:
        #             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze(0)
        #             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze(0)
        #         else:
        #             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze((0, 1))
        #             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze((0, 1))
        # else:
        #     if x.ndim == 2:
        #         xmean = self.running_mean.unsqueeze(0)
        #         xvar = self.running_var.unsqueeze(0)
        #     else:
        #         xmean = self.running_mean.unsqueeze(0).unsqueeze(0)
        #         xvar = self.running_var.unsqueeze(0).unsqueeze(0)
        # xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # self.out = self.gamma * xhat + self.beta
        # return self.out
        pass

    def parameters(self):
        """返回可训练参数"""
        # TODO: return [self.gamma, self.beta]
        pass


# ═══════════════════════════════════════════════════════════════════
#  题 2：构建 WaveNet 模型
# ═══════════════════════════════════════════════════════════════════

def build_wavenet(vocab_size, n_embd=10, n_hidden=68, block_size=8):
    """
    题 2：构建 WaveNet 模型

    结构：Embedding → [FC(2) → Linear → BN → Tanh] × 3 → Linear → 输出

    Args:
        vocab_size: 词汇表大小 (27)
        n_embd: Embedding 维度
        n_hidden: 隐藏层大小
        block_size: 上下文长度（必须是 2 的幂）
    Returns:
        model: Sequential 对象
    """
    # TODO: 构建并返回 WaveNet 模型
    #
    # 步骤：
    # 1. 创建 Sequential，包含：
    #    - Embedding(vocab_size, n_embd)
    #    - 3 组 [FlattenConsecutive(2), Linear(?, n_hidden, bias=False), BatchNorm1d3D(n_hidden), Tanh()]
    #    - 最后一层 Linear(n_hidden, vocab_size)
    #
    # 2. 注意每层 Linear 的输入维度：
    #    - 第 1 组: n_embd * 2 (FC 把两个 embedding 拼起来)
    #    - 第 2 组: n_hidden * 2
    #    - 第 3 组: n_hidden * 2
    #
    # 3. 最后一层 weight *= 0.1
    #
    # 4. 所有参数 requires_grad = True
    #
    # 5. 返回 model
    pass


# ═══════════════════════════════════════════════════════════════════
#  题 4：Tensor 形状流转验证
# ═══════════════════════════════════════════════════════════════════

def verify_shapes(model, vocab_size=27, block_size=8, batch_size=4):
    """
    题 4：验证模型每层的输出 shape

    Args:
        model: Sequential 对象
        vocab_size: 词汇表大小
        block_size: 上下文长度
        batch_size: 测试 batch 大小
    Returns:
        shapes: list of (layer_name, output_shape) 元组
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 创建随机输入 x = torch.randint(0, vocab_size, (batch_size, block_size))
    # 2. 逐层 forward，记录每层的 name 和 output shape
    # 3. 验证最终 shape == (batch_size, vocab_size)
    # 4. 返回 shapes 列表
    pass


# ═══════════════════════════════════════════════════════════════════
#  题 5：训练 WaveNet
# ═══════════════════════════════════════════════════════════════════

def train_wavenet(words, n_embd=24, n_hidden=128, block_size=8,
                  steps=50000, seed=42):
    """
    题 5（拓展）：训练 WaveNet 到 dev loss < 2.0

    Args:
        words: 名字列表
        n_embd: Embedding 维度
        n_hidden: 隐藏层大小
        block_size: 上下文长度
        steps: 训练步数
        seed: 随机种子
    Returns:
        dev_loss: 验证集 loss (float)
        model: 训练好的模型
    """
    # TODO: 实现这个函数（拓展题）
    #
    # 步骤：
    # 1. 划分数据集 80/10/10
    # 2. 用 build_wavenet 构建模型（替换 BN 为 BatchNorm1d3D）
    # 3. 训练循环：mini-batch + lr decay
    # 4. eval 模式评估 dev loss
    # 5. 返回 (dev_loss.item(), model)
    pass


if __name__ == '__main__':
    # 加载数据
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')
    words = open(_DATA_PATH, 'r').read().splitlines()

    # 题 1：FlattenConsecutive
    fc = FlattenConsecutive(2)
    x = torch.randn(4, 8, 10)
    y = fc(x)
    print(f"题 1 — FlattenConsecutive(2): {x.shape} → {y.shape}")

    # 题 3：BatchNorm1d3D
    bn = BatchNorm1d3D(10)
    x2d = torch.randn(32, 10)
    x3d = torch.randn(32, 4, 10)
    print(f"题 3 — BN 2D 输出 shape: {bn(x2d).shape}")
    print(f"题 3 — BN 3D 输出 shape: {bn(x3d).shape}")
    print(f"       running_mean shape: {bn.running_mean.shape}")

    # 题 2：build_wavenet
    model = build_wavenet(27)
    if model is not None:
        print(f"题 2 — WaveNet 参数量: {sum(p.nelement() for p in model.parameters()):,}")

    # 题 4：verify_shapes
    if model is not None:
        shapes = verify_shapes(model)
        if shapes:
            print(f"题 4 — 各层形状:")
            for name, shape in shapes:
                print(f"  {name}: {shape}")
