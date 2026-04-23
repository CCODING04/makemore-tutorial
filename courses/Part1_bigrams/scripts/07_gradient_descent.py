#!/usr/bin/env python3
"""
Part 1 - 脚本 07: 梯度下降训练
目标：用梯度下降优化神经网络版 bigram 模型。
完整流程：构建数据集 → 前向传播 → 计算损失 → 反向传播 → 更新权重。
训练完成后从模型采样生成名字。
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
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.shape[0]
    print(f"训练集大小: {num} 个 bigram")

    # ============ 初始化权重 ============
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    # ============ 梯度下降训练循环 ============
    print("\n开始训练（100 步，学习率 50，L2 正则化 0.01）：")
    for step in range(100):
        # 前向传播
        xenc = F.one_hot(xs, num_classes=27).float()  # (N, 27)
        logits = xenc @ W  # (N, 27)
        counts = logits.exp()  # 等价于计数矩阵
        probs = counts / counts.sum(1, keepdims=True)  # softmax

        # 计算损失：平均 NLL + L2 正则化
        # L2 正则化等价于计数版中的模型平滑（N+1）
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()

        # 反向传播
        W.grad = None  # 清零梯度
        loss.backward()

        # 更新权重
        W.data += -50 * W.grad  # 梯度下降，学习率 50

        if step % 10 == 0:
            print(f"  步骤 {step:3d}: loss = {loss.item():.4f}")

    print(f"  步骤  99: loss = {loss.item():.4f}")
    print(f"\n训练完成！最终 loss ≈ 2.47（应接近计数版的平均 NLL）")

    # ============ 从训练好的模型采样 ============
    print("\n从模型采样生成 5 个名字：")
    g_sample = torch.Generator().manual_seed(2147483647 + 10)

    for i in range(5):
        out = []
        ix = 0  # 从起始符开始
        while True:
            # 前向传播（单个字符）
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)

            # 采样
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g_sample).item()
            if ix == 0:
                break
            out.append(itos[ix])
        print(f"  {i + 1}. {''.join(out)}")

    print("\n总结：")
    print("  1. 计数版 bigram：直接统计频率，简单直观")
    print("  2. 神经网络版：用梯度下降优化权重 W，自动学到同样的结果")
    print("  3. 神经网络版的优势：可以扩展到更复杂的架构（RNN, Transformer 等）")
