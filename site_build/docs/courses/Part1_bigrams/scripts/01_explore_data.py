#!/usr/bin/env python3
"""
Part 1 - 脚本 01: 探索数据集
目标：读取 names.txt，了解数据的基本统计信息。
"""

import os

if __name__ == '__main__':
    # 读取数据（相对于 scripts/ 目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    # 打印前 10 个名字
    print("前 10 个名字：")
    for w in words[:10]:
        print(f"  {w}")

    # 统计信息
    print(f"\n总数: {len(words)}")
    print(f"最短名字: {min(words, key=len)} (长度 {min(len(w) for w in words)})")
    print(f"最长名字: {max(words, key=len)} (长度 {max(len(w) for w in words)})")

    # 字符集
    chars = sorted(set(''.join(words)))
    print(f"\n字符集大小: {len(chars)}")
    print(f"字符集: {''.join(chars)}")
