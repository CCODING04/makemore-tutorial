#!/usr/bin/env python3
"""
01 - 探索数据集
读取 names.txt，构建字符到索引的双向映射 (stoi / itos)，
打印字符集与统计信息。
"""

import os

# ── 数据路径 ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

if __name__ == '__main__':
    # 读取所有名字
    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    print(f"名字总数: {len(words)}")
    print(f"前 10 个名字: {words[:10]}")
    print()

    # 构建字符集（包含特殊起始/终止符 '.'）
    chars = sorted(set(''.join(words)))
    chars = ['.'] + chars  # '.' 作为起始和终止标记
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}

    print(f"字符集: {''.join(chars)}")
    print(f"字符总数（含 '.'）: {len(chars)}")
    print()
    print("stoi 映射:")
    for s, i in stoi.items():
        print(f"  '{s}' -> {i}")
