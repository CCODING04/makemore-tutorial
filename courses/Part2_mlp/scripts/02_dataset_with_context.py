#!/usr/bin/env python3
"""
02 - 构建上下文数据集
用 block_size=3 的上下文窗口把名字列表转成 (X, Y) 张量对，
展示 "emma" 展开过程，并做 train/dev/test 80/10/10 划分。
"""

import os
import torch
import random

# ── 数据路径 ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

# ── 超参数 ────────────────────────────────────────────────
BLOCK_SIZE = 3   # 用 3 个字符预测第 4 个


def build_dataset(words: list[str], stoi: dict, itos: dict,
                  block_size: int = BLOCK_SIZE):
    """把名字列表转成输入 X 和标签 Y 的张量。"""
    X, Y = [], []
    for w in words:
        # 初始上下文全是 '.'（索引 0）
        context = [0] * block_size
        for ch in w + '.':  # 名字末尾加 '.' 表示结束
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # 滑动窗口：去掉最左边，加入新字符
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(2147483647)

    # 读取数据
    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    # 构建字符映射
    chars = sorted(set(''.join(words)))
    chars = ['.'] + chars
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}

    print(f"名字总数: {len(words)}")
    print(f"字符集大小: {len(stoi)}")
    print()

    # ── 展示 "emma" 的展开过程 ────────────────────────────
    print("=== 'emma' 展开后的输入输出对 ===")
    demo_word = 'emma'
    context = [0] * BLOCK_SIZE
    for ch in demo_word + '.':
        ix = stoi[ch]
        input_str = ' '.join(itos[i] for i in context)
        print(f"  输入: {input_str}  -->  输出: '{ch}' (索引 {ix})")
        context = context[1:] + [ix]
    print()

    # ── 划分数据集 ────────────────────────────────────────
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    train_words = words[:n1]
    dev_words = words[n1:n2]
    test_words = words[n2:]

    X_train, Y_train = build_dataset(train_words, stoi, itos)
    X_dev, Y_dev = build_dataset(dev_words, stoi, itos)
    X_test, Y_test = build_dataset(test_words, stoi, itos)

    print("=== 数据集划分 (80/10/10) ===")
    print(f"  训练集: {len(train_words)} 个名字 -> X {X_train.shape}, Y {Y_train.shape}")
    print(f"  验证集: {len(dev_words)} 个名字 -> X {X_dev.shape}, Y {Y_dev.shape}")
    print(f"  测试集: {len(test_words)} 个名字 -> X {X_test.shape}, Y {Y_test.shape}")
