#!/usr/bin/env python3
"""Part 1 作业：Bigram 语言模型"""

import torch
import torch.nn.functional as F


def build_bigram_matrix(words):
    """
    构建 bigram 计数矩阵

    Args:
        words: 名字列表 (list of str)
    Returns:
        N: (27, 27) 的计数 tensor (int32)
    """
    # 注意：必须用固定的 26 字母表建映射，不能从 words 动态推导！
    # 否则传入 ['emma','olivia','ava'] 时字符集不足 27 个，索引会错位。
    stoi = {s: i + 1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
    stoi['.'] = 0

    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def compute_probabilities(N, smoothing=1):
    """
    从计数矩阵计算概率矩阵

    Args:
        N: (27, 27) 计数矩阵
        smoothing: 平滑系数（默认 1，即 Laplace Smoothing）
    Returns:
        P: (27, 27) 概率矩阵，每行和为 1
    """
    # (N + smoothing) 先平滑，再转 float 按行归一化
    # keepdims=True 保住 (27,1) 形状才能正确按行广播
    P = (N + smoothing).float()
    P = P / P.sum(1, keepdims=True)
    return P


def generate_names(P, n=5, seed=2147483647):
    """
    从概率矩阵采样生成名字

    Args:
        P: (27, 27) 概率矩阵
        n: 生成名字的数量
        seed: 随机种子
    Returns:
        names: 生成的名字列表 (list of str)
    """
    g = torch.Generator().manual_seed(seed)
    itos = {0: '.'}
    itos.update({i + 1: s for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')})

    names = []
    for _ in range(n):
        out = []
        ix = 0  # 从起始符 '.' 开始
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:  # 采样到 '.' → 名字结束
                break
            out.append(itos[ix])
        names.append(''.join(out))
    return names


def compute_nll_loss(P, words):
    """
    计算数据集的平均 NLL 损失

    Args:
        P: (27, 27) 概率矩阵
        words: 名字列表
    Returns:
        loss: 平均 NLL (float)
    """
    stoi = {s: i + 1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
    stoi['.'] = 0

    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return (-log_likelihood / n).item()


def train_bigram_nn(words, epochs=100, lr=50, seed=2147483647):
    """
    （拓展题）用梯度下降训练 bigram 神经网络

    Args:
        words: 名字列表
        epochs: 训练轮数
        lr: 学习率
        seed: 随机种子
    Returns:
        W: 训练好的权重矩阵 (27, 27)
        final_loss: 最终 loss 值
    """
    stoi = {s: i + 1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
    stoi['.'] = 0

    # 构建 (xs, ys) 训练对
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    g = torch.Generator().manual_seed(seed)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    loss = None
    for _ in range(epochs):
        # forward: one-hot → logits → softmax → NLL
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(len(xs)), ys].log().mean()
        # backward + 更新
        W.grad = None
        loss.backward()
        W.data += -lr * W.grad
    return W, loss.item()


if __name__ == '__main__':
    # 加载数据（基于脚本位置解析路径）
    import os
    _data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'names.txt')
    words = open(_data_path, 'r').read().splitlines()

    # 测试你的实现
    N = build_bigram_matrix(words)
    print(f"Bigram 矩阵形状: {N.shape}")

    P = compute_probabilities(N)
    print(f"概率矩阵行和: {P.sum(1)[:3]}")  # 应该都是 1.0

    names = generate_names(P, n=5)
    print(f"生成的名字: {names}")

    loss = compute_nll_loss(P, words)
    print(f"平均 NLL: {loss:.4f}")

    W, final_loss = train_bigram_nn(words, epochs=100)
    print(f"神经网络最终 loss: {final_loss:.4f}")
