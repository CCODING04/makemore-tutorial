#!/usr/bin/env python3
"""
03 - Embedding 层
创建 embedding 矩阵 C，将离散字符索引映射到连续低维向量，
展示 C[X] 的形状和 lookup 效果。
"""

import os
import torch
import random

# ── 数据路径 ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

# ── 超参数 ────────────────────────────────────────────────
BLOCK_SIZE = 3
N_EMBD = 2       # embedding 维度（先选小的方便可视化）


def build_dataset(words: list[str], stoi: dict, block_size: int = BLOCK_SIZE):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(2147483647)

    # 读取数据并构建映射
    with open(data_path, 'r') as f:
        words = f.read().splitlines()
    chars = sorted(set(''.join(words)))
    chars = ['.'] + chars
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)

    print(f"词汇表大小: {vocab_size}")
    print(f"Embedding 维度: {N_EMBD}")
    print()

    # ── 构建 Embedding 矩阵 ──────────────────────────────
    C = torch.randn((vocab_size, N_EMBD))
    print(f"Embedding 矩阵 C 的形状: {C.shape}")
    print(f"  每一行对应一个字符的 {N_EMBD} 维向量")
    print()

    # ── 展示单个字符的 embedding ─────────────────────────
    print("=== 单字符 embedding lookup ===")
    for ch in 'abcd':
        idx = stoi[ch]
        print(f"  '{ch}' (索引 {idx}) -> {C[idx].tolist()}")
    print()

    # ── 构建训练集并展示 C[X] ────────────────────────────
    random.shuffle(words)
    X, Y = build_dataset(words, stoi)

    print(f"X 的形状: {X.shape}  (样本数={X.shape[0]}, 上下文长度={X.shape[1]})")
    print(f"Y 的形状: {Y.shape}")
    print()

    # 核心：C[X] 一次性查找所有样本的 embedding
    emb = C[X]
    print(f"=== C[X] Embedding Lookup ===")
    print(f"  C[X] 形状: {emb.shape}")
    print(f"  含义: ({emb.shape[0]} 个样本) × ({emb.shape[1]} 个上下文字符) × ({emb.shape[2]} 维 embedding)")
    print()

    # 展示第一个样本
    print(f"第一个样本 X[0] = {X[0].tolist()} -> 字符: {[itos[i] for i in X[0].tolist()]}")
    print(f"对应的 embedding:")
    for i, idx in enumerate(X[0]):
        print(f"  位置 {i}: '{itos[idx.item()]}' -> {emb[0, i].tolist()}")
