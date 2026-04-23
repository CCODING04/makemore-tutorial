#!/usr/bin/env python3
"""
Part 1 - 脚本 06: 神经网络版 Bigram
目标：用单层神经网络实现 bigram 语言模型。
网络结构：one-hot 编码 → 线性层 (W) → softmax → 概率
这与计数版 bigram 是等价的，但用梯度下降来优化。
"""

import os
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    # 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    # 构建字符映射
    chars = sorted(set(''.join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    # ============ 构建训练数据集 ============
    # xs: 输入字符的索引，ys: 目标字符的索引
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    print(f"训练集大小: {xs.shape[0]} 个 bigram")

    # ============ 初始化权重矩阵 ============
    # W 是一个 27x27 的矩阵，等价于 bigram 计数矩阵
    # 初始化为随机值（接近均匀分布）
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    # ============ 前向传播 ============
    # one-hot 编码: (N, 27)
    xenc = F.one_hot(xs, num_classes=27).float()
    print(f"one-hot 编码形状: {xenc.shape}")

    # 线性层: xenc @ W → logits (N, 27)
    logits = xenc @ W

    # softmax → 概率
    # 先取 exp 得到 "计数"，再归一化得到概率
    counts = logits.exp()  # 等价于 N 矩阵
    probs = counts / counts.sum(1, keepdims=True)

    # ============ 计算 NLL Loss ============
    # 对每个 bigram，取其对应位置的概率，求负对数
    loss = -probs[torch.arange(xs.shape[0]), ys].log().mean()
    print(f"\n初始 loss（未训练）: {loss.item():.4f}")

    # 解释等价性
    print("\n" + "=" * 50)
    print("等价性说明：")
    print("  计数版: P = N / N.sum(1)，其中 N 是手动统计的计数矩阵")
    print("  神经网络版: logits = xenc @ W，P = softmax(logits)")
    print("  当 W 的最优值 = log(N) 时，两者完全等价！")
    print("  神经网络通过梯度下降自动找到这个最优 W。")
    print("=" * 50)

    # ============ 手动计算几个 bigram 的 NLL ============
    print(f"\n前 5 个 bigram 的概率和 NLL：")
    for i in range(5):
        input_char = itos[xs[i].item()]
        target_char = itos[ys[i].item()]
        prob = probs[i, ys[i]].item()
        nll_i = -torch.log(probs[i, ys[i]]).item()
        print(f"  '{input_char}' → '{target_char}': P = {prob:.4f}, NLL = {nll_i:.4f}")
