#!/usr/bin/env python3
"""
Part 1 - 脚本 02: Bigram 计数
目标：统计所有 bigram（相邻字符对）的出现频率，构建 27x27 计数矩阵。
"""

import os
import torch

if __name__ == '__main__':
    # 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    # 构建字符映射表
    # '.' 作为特殊起始/终止符，索引 0；'a' 到 'z' 索引 1 到 26
    chars = sorted(set(''.join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}  # a=1, b=2, ..., z=26
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    print(f"字符映射（共 {len(stoi)} 个）：")
    print(f"  stoi = {{'.': 0, 'a': 1, ..., 'z': 26}}")

    # 构建 27x27 的 bigram 计数矩阵
    N = torch.zeros((27, 27), dtype=torch.int32)

    for w in words:
        # 在名字前后加特殊字符 '.'，表示起始和终止
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    print(f"\nBigram 计数矩阵大小: {N.shape}")
    print(f"总 bigram 数量: {N.sum().item()}")

    # 找出 top 10 最常见的 bigram
    bigram_counts = []
    for i in range(27):
        for j in range(27):
            bigram_counts.append((itos[i] + itos[j], N[i, j].item()))

    bigram_counts.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 最常见的 bigram：")
    for bg, count in bigram_counts[:10]:
        print(f"  '{bg}' -> {count}")
