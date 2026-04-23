#!/usr/bin/env python3
"""Part 2 作业：MLP 字符级语言模型"""

import os
import torch
import torch.nn.functional as F


def build_dataset(words, block_size=3):
    """
    构建训练数据集

    Args:
        words: 名字列表 (list of str)
        block_size: 上下文长度（默认 3）
    Returns:
        X: (N, block_size) 输入上下文 tensor (int64)
        Y: (N,) 目标字符 tensor (int64)
    """
    # TODO: 实现这个函数
    #
    # 字符映射：'.'=0, 'a'=1, ..., 'z'=26
    #
    # 步骤：
    # 1. 创建 stoi 映射
    #    stoi = {s: i+1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
    #    stoi['.'] = 0
    #
    # 2. 初始化空列表 X, Y
    #
    # 3. 遍历每个名字：
    #    a. 创建初始上下文 context = [0] * block_size
    #    b. 遍历 w + '.' 中的每个字符 ch：
    #       - 将 stoi[ch] 记为 ix
    #       - 将 context 的拷贝加入 X
    #       - 将 ix 加入 Y
    #       - 更新 context：滑动窗口，去掉最左，加入 ix
    #
    # 4. 将 X, Y 转为 torch.tensor（dtype=torch.int64）
    pass


def mlp_forward(X, C, W1, b1, W2, b2):
    """
    MLP 前向传播

    Args:
        X: (N, block_size) 输入上下文索引
        C: (27, n_embd) Embedding 矩阵
        W1: (block_size * n_embd, n_hidden) 第一层权重
        b1: (n_hidden,) 第一层偏置
        W2: (n_hidden, 27) 第二层权重
        b2: (27,) 第二层偏置
    Returns:
        logits: (N, 27) 未经 softmax 的输出
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. Embedding 查表：emb = C[X]  → (N, block_size, n_embd)
    #
    # 2. 拼接：emb_cat = emb.view(emb.shape[0], -1)  → (N, block_size * n_embd)
    #
    # 3. 隐藏层（带 tanh 激活）：
    #    h = torch.tanh(emb_cat @ W1 + b1)  → (N, n_hidden)
    #
    # 4. 输出层：
    #    logits = h @ W2 + b2  → (N, 27)
    #
    # 5. 返回 logits
    pass


def train_step(X, Y, C, W1, b1, W2, b2, lr=0.1):
    """
    执行一次训练步骤：forward → loss → backward → update

    Args:
        X, Y: 训练数据
        C, W1, b1, W2, b2: 模型参数（将被原地更新）
        lr: 学习率
    Returns:
        loss: 当前 loss 值 (float)
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. Forward：调用 mlp_forward 得到 logits
    #
    # 2. 计算 loss：F.cross_entropy(logits, Y)
    #
    # 3. 清零所有参数的梯度：
    #    for p in [C, W1, b1, W2, b2]:
    #        p.grad = None
    #
    # 4. 反向传播：loss.backward()
    #
    # 5. 更新参数：
    #    for p in [C, W1, b1, W2, b2]:
    #        p.data -= lr * p.grad
    #
    # 6. 返回 loss.item()
    pass


def evaluate(X, Y, C, W1, b1, W2, b2):
    """
    在给定数据上评估 loss（不计算梯度，不更新参数）

    Args:
        X, Y: 评估数据
        C, W1, b1, W2, b2: 模型参数
    Returns:
        loss: 评估 loss (float)
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 使用 torch.no_grad() 上下文
    # 2. 调用 mlp_forward 得到 logits
    # 3. 计算 F.cross_entropy(logits, Y)
    # 4. 返回 loss.item()
    pass


def tuning_experiment(words, block_size=3, n_embd=10, n_hidden=200,
                      steps=200000, lr=0.1, seed=2147483647):
    """
    （拓展题）调参实验

    尝试不同的超参数组合，找到验证 loss < 2.2 的配置。

    Args:
        words: 名字列表
        block_size: 上下文长度
        n_embd: Embedding 维度
        n_hidden: 隐藏层大小
        steps: 训练步数
        lr: 初始学习率
        seed: 随机种子
    Returns:
        val_loss: 验证集 loss (float)
        C, W1, b1, W2, b2: 训练好的参数
    """
    # TODO: 实现这个函数（拓展题）
    #
    # 步骤：
    # 1. 划分数据集：80% 训练, 10% 验证, 10% 测试
    #    import random
    #    random.seed(42)
    #    random.shuffle(words)
    #    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    #
    # 2. 用 build_dataset 分别构建三个数据集
    #
    # 3. 初始化参数：
    #    g = torch.Generator().manual_seed(seed)
    #    C = torch.randn(27, n_embd, generator=g, requires_grad=True)
    #    W1 = torch.randn(block_size * n_embd, n_hidden, generator=g, requires_grad=True)
    #    b1 = torch.randn(n_hidden, generator=g, requires_grad=True)
    #    W2 = torch.randn(n_hidden, 27, generator=g, requires_grad=True)
    #    b2 = torch.randn(27, generator=g, requires_grad=True)
    #
    # 4. 训练循环：
    #    for i in range(steps):
    #        # Mini-batch（可选，提升训练速度）
    #        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    #        current_lr = lr if i < steps // 2 else lr * 0.1
    #        loss = train_step(Xtr[ix], Ytr[ix], C, W1, b1, W2, b2, lr=current_lr)
    #
    # 5. 评估验证集 loss 并返回
    pass


if __name__ == '__main__':
    # 加载数据（基于脚本位置解析路径）
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')
    words = open(_DATA_PATH, 'r').read().splitlines()

    # 测试你的实现
    X, Y = build_dataset(words[:1], block_size=3)
    print(f"build_dataset 输出: X.shape={X.shape}, Y.shape={Y.shape}")

    # 初始化参数（小模型测试）
    torch.manual_seed(2147483647)
    n_embd, n_hidden, block_size = 10, 200, 3
    C = torch.randn(27, n_embd, requires_grad=True)
    W1 = torch.randn(block_size * n_embd, n_hidden, requires_grad=True)
    b1 = torch.randn(n_hidden, requires_grad=True)
    W2 = torch.randn(n_hidden, 27, requires_grad=True)
    b2 = torch.randn(27, requires_grad=True)

    logits = mlp_forward(X, C, W1, b1, W2, b2)
    print(f"logits shape: {logits.shape}")

    loss = train_step(X, Y, C, W1, b1, W2, b2, lr=0.1)
    print(f"train_step loss: {loss:.4f}")

    val_loss = evaluate(X, Y, C, W1, b1, W2, b2)
    print(f"evaluate loss: {val_loss:.4f}")
