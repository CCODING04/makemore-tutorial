#!/usr/bin/env python3
"""
Part 1 - 脚本 03: 可视化 Bigram 矩阵
目标：用 matplotlib 绘制 bigram 计数矩阵的热力图。
"""

import os

import matplotlib
matplotlib.use('Agg')  # 无头模式，不需要 GUI
import matplotlib.pyplot as plt
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

    # 绘制热力图
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=8)
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray', fontsize=8)

    plt.axis('off')
    plt.title('Bigram Count Matrix', fontsize=16)

    # 保存到当前目录
    output_path = os.path.join(script_dir, 'bigram_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bigram 矩阵已保存到: {output_path}")
