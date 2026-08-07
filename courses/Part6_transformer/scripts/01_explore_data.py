#!/usr/bin/env python3
"""
Part 6 - Script 1: 数据读取与字符级 Tokenizer
目标：了解语言建模任务，读取 tiny Shakespeare 数据集，构建最简单的字符级
tokenizer（encode/decode），并划分训练/验证集。这是整个 Transformer 教程的数据基础。

覆盖知识点：
  - 语言模型 = 预测序列中的下一个 token；GPT = Generative Pretrained Transformer
  - tiny Shakespeare：~1MB、~1M 字符、65 个唯一字符
  - chars / stoi / itos / encode / decode（字符级 tokenizer）
  - 其它 tokenizer 对比：sentencepiece(Google) / tiktoken-BPE(OpenAI, ~50K tokens)
  - train/val 90/10 划分（用于检测过拟合）
"""

import os
import sys
import torch

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使运行更快更稳定
torch.set_num_threads(1)

torch.manual_seed(1337)


def main():
    # ─── 数据路径（相对脚本位置，绝不依赖 cwd）──────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')

    # ─── 读取数据 ───────────────────────────────────────────────────
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("═══ 数据集统计 ═══")
    print(f"  总字符数: {len(text):,}")

    print("\n═══ 前 1000 字符预览 ═══")
    print(repr(text[:1000]))

    # ─── 构建词汇表（所有唯一字符）────────────────────────────────
    # 语言模型只能输出它见过的字符：vocab = 数据集中出现过的所有字符
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("\n═══ 词汇表 ═══")
    print(f"  唯一字符数 (vocab_size): {vocab_size}")
    print(f"  字符列表: {''.join(chars)}")

    # ─── 字符级 tokenizer：stoi / itos / encode / decode ──────────
    stoi = {ch: i for i, ch in enumerate(chars)}   # char → int
    itos = {i: ch for i, ch in enumerate(chars)}   # int → char
    encode = lambda s: [stoi[c] for c in s]        # 字符串 → 整数列表
    decode = lambda l: ''.join([itos[i] for i in l])  # 整数列表 → 字符串

    print("\n═══ Tokenizer 演示 ═══")
    s = "hi there"
    encoded = encode(s)
    print(f"  encode('{s}') = {encoded}")
    print(f"  decode({encoded}) = '{decode(encoded)}'")
    print(f"  往返一致: {decode(encode(s)) == s}")

    # 字符 0 通常是换行符（字幕里有点不确定，取决于数据）
    print(f"  索引 0 对应字符: {repr(itos[0])}（换行符）")
    print(f"  索引 1 对应字符: {repr(itos[1])}")

    # ─── 其它 tokenizer 对比（仅说明，不运行）────────────────────
    # 词表大小 vs 序列长度互为 trade-off：
    #   - 字符级：65 个 token，序列很长（我们采用，最简单）
    #   - sentencepiece（Google）：subword，实践中常见
    #   - tiktoken/BPE（OpenAI, GPT-2 用）：~50K tokens，序列更短

    # ─── 训练/验证划分（前 90% 训练，后 10% 验证）─────────────────
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print("\n═══ Train/Val 划分 (90/10) ═══")
    print(f"  data shape: {data.shape}, dtype: {data.dtype}")
    print(f"  train: {len(train_data):,} 字符 ({len(train_data) / len(data) * 100:.1f}%)")
    print(f"  val:   {len(val_data):,} 字符 ({len(val_data) / len(data) * 100:.1f}%)")
    print(f"  验证集用途：隐藏数据，检测模型是否过拟合（而非死记硬背）")

    print("""
═══ 总结 ═══

字符级 tokenizer 是最简单的方案（65 个 token），代价是序列很长；
工业界常用 subword tokenizer（sentencepiece / tiktoken-BPE）来压缩序列长度。
接下来我们要把这段长整数序列喂给神经网络 → 但不会一次喂全部，
而是每次采样一个 chunk（block_size × batch_size）来训练。

下一步：Script 2 — Dataloader 与 Bigram 基线。
""")


if __name__ == '__main__':
    main()
