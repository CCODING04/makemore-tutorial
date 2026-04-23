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
    # TODO: 实现这个函数
    # 提示：stoi 映射 '.'=0, 'a'=1, ..., 'z'=26
    #
    # 步骤：
    # 1. 创建 stoi 和 itos 映射
    # 2. 创建全零 (27, 27) tensor
    # 3. 遍历每个名字，在首尾加 '.'
    # 4. 统计每个 (ch1, ch2) 对的出现次数
    pass


def compute_probabilities(N, smoothing=1):
    """
    从计数矩阵计算概率矩阵

    Args:
        N: (27, 27) 计数矩阵
        smoothing: 平滑系数（默认 1，即 Laplace Smoothing）
    Returns:
        P: (27, 27) 概率矩阵，每行和为 1
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 给 N 的每个元素加上 smoothing
    # 2. 按行求和
    # 3. 按行归一化，得到概率
    #
    # 注意：广播机制！P[i,j] = (N[i,j] + s) / sum(N[i,:] + s)
    pass


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
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 创建 Generator 并设置 seed
    # 2. 创建 itos 反向映射 {0: '.', 1: 'a', ..., 26: 'z'}
    # 3. 循环 n 次：
    #    a. 从 ix=0（'.'）开始
    #    b. 用 P[ix] 作为分布，torch.multinomial 采样下一个字符
    #    c. 如果采样到 0（'.'），结束这个名字
    #    d. 否则把字符加入当前名字
    pass


def compute_nll_loss(P, words):
    """
    计算数据集的平均 NLL 损失

    Args:
        P: (27, 27) 概率矩阵
        words: 名字列表
    Returns:
        loss: 平均 NLL (float)
    """
    # TODO: 实现这个函数
    #
    # 步骤：
    # 1. 创建 stoi 映射
    # 2. 遍历所有名字的所有 bigram
    # 3. 累加 log(P[ch1_idx, ch2_idx])
    # 4. 返回 -平均值
    pass


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
    # TODO: 实现这个函数（拓展题）
    #
    # 步骤：
    # 1. 构建 stoi 映射
    # 2. 构建 training set：xs（输入字符索引列表）, ys（目标字符索引列表）
    # 3. 初始化 W = randn(27, 27)，设置 requires_grad=True
    # 4. 训练循环：
    #    a. Forward: one_hot(xs) @ W → softmax → NLL loss
    #    b. Backward: loss.backward()
    #    c. Update: W.data -= lr * W.grad
    # 5. 返回 W 和最终 loss
    pass


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
