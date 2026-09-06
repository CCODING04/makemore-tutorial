#!/usr/bin/env python3
"""
Part 7 - 脚本 1: 用 HuggingFace tokenizers 训练 BPE 分词器
目标：在 tiny Shakespeare（input.txt）上从零训练 BPE 分词器，演示
encode/decode 往返验证、压缩率对比（字符级 vs BPE）、学到的常见子词
（"th", "ing", "the" 等），并把 tokenizer 保存到 temp 目录。
这是 modern LLM 流水线（minimind）的第一步：字符级 → BPE 子词级。

覆盖知识点：
  - BPE（Byte Pair Encoding，Sennrich et al. 2016）：从字符/字节出发，
    反复合并出现频率最高的相邻对，直到词表达到目标大小
  - byte-level BPE（GPT-2 同款）：基础字母表是 256 个字节，无 <unk>，
    任何文本都可无损还原（不会出现未知 token）
  - 特殊 token：<|im_start|> / <|im_end|>（minimind 的 chat 格式标记）
  - 压缩率：同样的文本，BPE 需要的 token 数远少于字符级
"""

import os
import sys
from collections import Counter

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# tokenizers 库可能未安装：先尝试 import，未安装时给出清晰提示
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
    _HAVE_TOKENIZERS = True
except ImportError:
    _HAVE_TOKENIZERS = False

import torch  # 仅用于模式选择（是否 GPU）

# ─── 模式选择 ──────────────────────────────────────────────
# CPU 模式: 小模型，<30s 跑完，用于学习验证
# GPU 模式: 完整规模，匹配 minimind 架构，需 GPU
CPU_MODE = not torch.cuda.is_available()
if CPU_MODE:
    vocab_size = 256
    hidden_size = 64
    n_layers = 2
    n_heads = 4
    n_kv_heads = 2
else:
    vocab_size = 6400
    hidden_size = 768
    n_layers = 8
    n_heads = 8
    n_kv_heads = 4
# ─── ───────────────────────────────────────────────────────

# minimind 的 chat 格式特殊 token（SFT/DPO 脚本也会用到）
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]


def train_bpe(data_path, vocab, special_tokens, show_progress=False):
    """用 HuggingFace tokenizers 训练一个 byte-level BPE。"""
    tokenizer = Tokenizer(BPE(unk_token=None))
    # add_prefix_space=False：不给句首词加前缀空格，保证 encode/decode 无损往返
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()  # 必须配套，decode 才能无损还原原文本
    trainer = BpeTrainer(
        vocab_size=vocab,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),  # 256 个字节作为初始字母表
        show_progress=show_progress,
    )
    tokenizer.train([data_path], trainer)
    return tokenizer


def demo_roundtrip(tokenizer, text):
    print("\n═══ encode/decode 往返验证 ═══")
    sample = text[1500:1580].replace('\n', ' ')  # 取一段英文
    ids = tokenizer.encode(sample).ids
    decoded = tokenizer.decode(ids)
    ok = (decoded == sample)
    print(f"  原始文本: {sample!r}")
    print(f"  token ids: {ids[:25]} ...（共 {len(ids)} 个 token）")
    print(f"  解码还原: {decoded!r}")
    print(f"  ✅ 往返无损: {ok}（byte-level BPE 天然无损，不会出现 <unk>）")


def demo_special_tokens(tokenizer):
    print("\n═══ 特殊 token（chat 格式）═══")
    chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"
    ids = tokenizer.encode(chat).ids
    print(f"  聊天文本: {chat!r}")
    print(f"  token ids: {ids}")
    # 特殊 token 应各自是"单个"id
    stoi = tokenizer.get_vocab()
    start_id = stoi["<|im_start|>"]
    end_id = stoi["<|im_end|>"]
    print(f"  <|im_start|> → id {start_id}（占 1 个 token）")
    print(f"  <|im_end|>   → id {end_id}（占 1 个 token）")
    print(f"  ✅ 特殊 token 是一个整体，不会被拆散（这与字符级不同）")


def demo_compression(tokenizer, text):
    print("\n═══ 压缩率对比：字符级 vs BPE ═══")
    char_tokens = len(text)                              # 字符级：1 字符 = 1 token
    bpe_ids = tokenizer.encode(text).ids                 # BPE：合并成子词
    bpe_tokens = len(bpe_ids)
    ratio = char_tokens / bpe_tokens
    print(f"  字符级 token 数: {char_tokens:,}")
    print(f"  BPE   token 数:  {bpe_tokens:,}")
    print(f"  压缩率: {ratio:.3f}x（即平均 {ratio:.2f} 个字符合并成 1 个 token）")
    print(f"  💡 压缩率越高 → 每个 token 携带更多信息 → 相同计算量下看到更长上下文")


def demo_subwords(tokenizer, text, n_show=30):
    print("\n═══ BPE 学到的常见子词 ═══")
    freq = Counter(tokenizer.encode(text).ids)
    vocab = tokenizer.get_vocab()
    print("  频率 Top-%d token（展示真实学到的子词）:" % n_show)
    for i, (tid, cnt) in enumerate(freq.most_common(n_show)):
        token = tokenizer.decode([tid])
        print(f"    {i + 1:2d}. {token!r:16s}  次数 {cnt:>7,}")
    # byte-level 里空格会显示为 'Ġ' 前缀，比较子词时去掉它
    cleaned = {t.lstrip('Ġ') for t in vocab}
    print("  典型子词是否被学到:")
    for t in ['th', 'ing', 'the', 'and', 'tion']:
        hit = t in cleaned
        print(f"    {t!r:8s} {'✅ 在词表中' if hit else '❌ 不在（可能被拆成更小子词）'}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("═══ 数据 ═══")
    print(f"  数据文件: {data_path}")
    print(f"  字符数: {len(text):,}，行数: {text.count(chr(10)):,}")

    if not _HAVE_TOKENIZERS:
        print("""
⚠️  未检测到 tokenizers 库，无法训练 BPE。请先安装：
      pip install tokenizers
    （安装后重新运行本脚本即可）""")
        return

    # ─── 训练"正式" tokenizer ──────────────────────────────
    # byte-level BPE 的初始字母表是 256 个字节，再加 2 个特殊 token，
    # 所以词表至少要 258；若配置值更小则自动抬升并说明。
    effective_vocab = max(vocab_size, len(SPECIAL_TOKENS) + 256)
    if effective_vocab > vocab_size:
        print(f"\n  ⚠️  配置 vocab_size={vocab_size} 对 byte-level BPE 过小，"
              f"抬升为 {effective_vocab}（256 字节 + 2 特殊 token）")
    print(f"\n═══ 训练 BPE (vocab_size={effective_vocab}) ═══")
    tokenizer = train_bpe(data_path, effective_vocab, SPECIAL_TOKENS)
    print(f"  训练完成，实际词表大小: {tokenizer.get_vocab_size()}")

    demo_roundtrip(tokenizer, text)
    demo_special_tokens(tokenizer)
    demo_compression(tokenizer, text)

    # ─── 子词展示 ──────────────────────────────────────────
    # CPU 模式下 258 词表 = 256 字节，几乎无 merge（=字节级），
    # 所以额外训练一个 vocab=2000 的"演示版"来展示真正的子词。
    if CPU_MODE:
        print("\n  💡 CPU 模式 258 词表≈字节级（0 次 merge），"
              "下面用 vocab=2000 的演示版展示子词与真实压缩率：")
        demo_vocab = 2000
    else:
        demo_vocab = vocab_size
    demo_tok = train_bpe(data_path, demo_vocab, SPECIAL_TOKENS)
    print(f"  演示版词表: {demo_tok.get_vocab_size()}（{'vocab_size=6400 主词表' if not CPU_MODE else 'vocab=2000 演示用'}）")
    demo_compression(demo_tok, text)
    demo_subwords(demo_tok, text)

    # ─── 保存 tokenizer ────────────────────────────────────
    save_path = os.path.join(temp_dir, 'bpe_tokenizer.json')
    tokenizer.save(save_path)
    print("\n═══ 保存 ═══")
    print(f"  ✅ 已保存 tokenizer → {save_path}")
    print(f"  下次可用 Tokenizer.from_file('{save_path}') 直接加载（脚本 5 会用到）")
    print("""
═══ 总结 ═══

BPE 把"频繁出现的相邻字符/字节"合并成词表中的一个子词：
  - 词表越大，子词越接近"单词"，压缩率越高，但对稀有词覆盖变差
  - 词表太小（如 256）→ 几乎不 merge，退化成字节级，压缩率≈1.0
  - byte-level BPE 无损：任何文本都能被 encode/decode 精确还原
  - minimind 用 6400 词表 + <|im_start|> / <|im_end|> 两个 chat 特殊 token

下一个脚本：RMSNorm + RoPE —— 现代 LLM 的两个基础组件。""")


if __name__ == '__main__':
    main()
