#!/usr/bin/env python3
"""
Part 1 - 脚本 04: 概率采样生成名字
目标：将 bigram 计数转为概率分布，通过采样生成新名字。
使用模型平滑（N+1）避免零概率。
"""

import os
import torch

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

    # 构建 bigram 计数矩阵
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1

    # 模型平滑：N+1 避免零概率（拉普拉斯平滑）
    P = (N + 1).float()
    # 按行归一化：P[i] 表示给定字符 i 后，下一个字符的概率分布
    P = P / P.sum(1, keepdims=True)

    print("概率矩阵 P 的大小:", P.shape)
    print("每行概率之和（应全为 1.0）:", P[0].sum().item())

    # 固定随机种子，确保结果可重复
    g = torch.Generator().manual_seed(2147483647)

    # 从 bigram 模型采样生成名字
    print("\n采样生成的 5 个名字：")
    for i in range(5):
        out = []
        ix = 0  # 从起始符 '.' 开始
        while True:
            # 获取当前字符的概率分布
            p = P[ix]
            # 按概率采样下一个字符
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:
                # 采样到终止符 '.'，名字结束
                break
            out.append(itos[ix])
        print(f"  {i + 1}. {''.join(out)}")
