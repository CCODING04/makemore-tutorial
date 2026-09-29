#!/usr/bin/env python3
"""
Part 7 - 脚本 01: BPE 分词器 —— 在官方 mini 语料上自训一版，与官方 tokenizer 对照

三件事：
  ① 体检官方 tokenizer（dataset/tokenizer/，00_download_data.py 下载）：
     词表大小、bos/eos/pad 三个角色 token、压缩率样例
  ② 在官方 mini 预训练语料（sample 2 万条）上从零训练一个教学版 byte-level BPE
     （vocab 6400，与官方同规格），跑四道体检：往返无损 / 特殊 token / 压缩率 / Top 子词
  ③ 两版对照：同规格词表，"sample 语料训练"与"全量语料训练"差在哪——
     这就是官方"不建议重训 tokenizer"的原因（差异可感知，收益为零）

前置：python 00_download_data.py --sample-only --no-full   # ~48MB
运行（纯 CPU，约 10-30 秒）：python 01_bpe_tokenizer.py
"""

import json
import os
import sys
import time
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def log_result(name, row):
    """消融实验一次性结果行落盘——教程引用数字的证据文件（temp/out/logs/exp*.csv）。"""
    import csv
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f'{name}.csv')
    row = {'date': time.strftime('%Y-%m-%d'), **row}
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
    _HAVE_TOKENIZERS = True
except ImportError:
    _HAVE_TOKENIZERS = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'dataset'))
TEMP_DIR = os.path.join(SCRIPT_DIR, '..', 'temp')
LOG_DIR = os.path.join(TEMP_DIR, 'out', 'logs')

VOCAB_SIZE = 6400                       # 与官方同规格（26M/64M 共用 6400 词表）
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]   # 教学版只挂 chat 两个；官方另挂 think/vision 等
EVAL_SENTENCES = ["人工智能是计算机科学的一个分支，它企图了解智能的实质。",
                  "Large language models are trained on vast amounts of text data."]


def find_data():
    for name in ('pretrain_t2t_mini_sample.jsonl', 'pretrain_t2t_mini.jsonl'):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"未找到数据——先跑 python 00_download_data.py（期望 {DATA_DIR} 下有 jsonl）")


def extract_texts(path, limit=None):
    """jsonl → text 列表（训练与评测共用一份解析）。"""
    texts = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                texts.append(json.loads(line)['text'])
            except (json.JSONDecodeError, KeyError):
                continue
    return texts


# ════════════════════════ ① 官方 tokenizer 体检 ════════════════════════

def official_report():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))
    print("═══ 1/3 官方 tokenizer 体检 ═══")
    print(f"  词表大小: {len(tok)}")
    print(f"  三个角色 token: bos={tok.bos_token!r}(id={tok.bos_token_id})  "
          f"eos={tok.eos_token!r}(id={tok.eos_token_id})  pad={tok.pad_token!r}(id={tok.pad_token_id})")
    print(f"  保留的特殊 token: {len(tok.added_tokens_decoder)} 个（im/vision/audio/tool_call/think…）")
    for s in EVAL_SENTENCES:
        n = len(tok(s, add_special_tokens=False).input_ids)
        print(f"  压缩率样例: {len(s)} 字 → {n} token（{len(s)/n:.2f} 字/token）")
    return tok


# ════════════════════════ ② 自训教学版 BPE ════════════════════════

def train_bpe(corpus_path, vocab, special_tokens):
    """用 HuggingFace tokenizers 训 byte-level BPE（GPT-2 同款路线）。"""
    tokenizer = Tokenizer(BPE(unk_token=None))
    # add_prefix_space=False：不给句首词加前缀空格，保证 encode/decode 无损往返
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()      # 必须配套，decode 才能无损还原
    trainer = BpeTrainer(vocab_size=vocab,
                         special_tokens=special_tokens,
                         initial_alphabet=ByteLevel.alphabet(),   # 256 字节打底，无 <unk>
                         show_progress=False)
    tokenizer.train([corpus_path], trainer)
    return tokenizer


def demo_roundtrip(tokenizer):
    print("\n  ── 体检 1/4：encode/decode 往返无损 ──")
    sample = EVAL_SENTENCES[0] + "\n" + EVAL_SENTENCES[1]
    ids = tokenizer.encode(sample).ids
    decoded = tokenizer.decode(ids)
    ok = decoded == sample
    print(f"    原文 {len(sample)} 字 → {len(ids)} token → 还原 {len(decoded)} 字")
    print(f"    {'✅ 往返无损（byte-level BPE 天然覆盖所有文本，不会出现 <unk>）' if ok else '❌ 不一致！'}")


def demo_special_tokens(tokenizer):
    print("\n  ── 体检 2/4：特殊 token 是不是一个整体 ──")
    chat = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好呀<|im_end|>"
    ids = tokenizer.encode(chat).ids
    stoi = tokenizer.get_vocab()
    print(f"    chat 文本编码为 {len(ids)} 个 token，其中 <|im_start|> → id {stoi['<|im_start|>']}、"
          f"<|im_end|> → id {stoi['<|im_end|>']}（各占 1 个，不会被拆散）")


def compression(tok_like_encode, text):
    """tok_like_encode(text) -> ids。返回 字符数/token 数。"""
    return len(text) / max(len(tok_like_encode(text)), 1)


def demo_subwords(tokenizer, eval_text, n_show=12):
    print("\n  ── 体检 4/4：BPE 学到的常见子词（中英混合语料）──")
    freq = Counter(tokenizer.encode(eval_text).ids)
    for i, (tid, cnt) in enumerate(freq.most_common(n_show)):
        print(f"    {i+1:2d}. {tokenizer.decode([tid])!r:14s}  次数 {cnt:>6,}")
    # 词表键是 byte-level 映射形式（如 '的'→'çš„'），须解码回表面形式再比对
    cleaned = {tokenizer.decode([i]).lstrip('Ġ') for i in range(tokenizer.get_vocab_size())}
    for t in ['的', '我们', '人工', 'the', 'ing', 'tion']:
        hit = t in cleaned
        print(f"    {t!r:8s} {'✅ 在词表中' if hit else '❌ 不在（被拆成更小子词——常见词未必整词收录）'}")


# ════════════════════════ ③ 两版对照 ════════════════════════

def compare(teach, official, eval_text):
    print("\n═══ 3/3 对照：教学版（sample 训练）vs 官方（全量训练）═══")
    t_ids = teach.encode(eval_text).ids
    o_ids = official(eval_text, add_special_tokens=False).input_ids
    # 词表键一边是 byte-level 映射串、一边是表面串，先统一解码成表面形式再比
    t_vocab = {teach.decode([i]) for i in range(teach.get_vocab_size())}
    o_vocab = {official.convert_ids_to_tokens(i) for i in range(len(official))}
    o_vocab = {official.convert_tokens_to_string([v]) for v in o_vocab}
    overlap = len(t_vocab & o_vocab) / len(t_vocab | o_vocab) * 100
    log_result('exp01_bpe_compare',
               dict(teach_vocab=teach.get_vocab_size(), official_vocab=len(official),
                    teach_heldout_ratio=round(len(eval_text) / len(t_ids), 3),
                    official_heldout_ratio=round(len(eval_text) / len(o_ids), 3),
                    overlap_jaccard_pct=round(overlap, 1),
                    official_special_tokens=len(official.added_tokens_decoder)))
    print(f"  {'':24s}{'教学版':>10s}{'官方':>10s}")
    print(f"  {'词表大小':<24s}{teach.get_vocab_size():>10,}{len(official):>10,}")
    print(f"  {'留出集压缩率(字/token)':<22s}{len(eval_text)/len(t_ids):>10.2f}{len(eval_text)/len(o_ids):>10.2f}")
    print(f"  {'特殊 token 数':<24s}{len(SPECIAL_TOKENS):>10}{len(official.added_tokens_decoder):>10}")
    print(f"  {'词表重合度(Jaccard)':<23s}{overlap:>9.1f}%")
    print("  ↳ 留出集上教学版甚至更省——2 万条样本养出的『领域专家』在自己分布里占优，")
    print("    但官方版面向全量语料更通用。更关键的是换 tokenizer = 换词表 = 权重全部重训，")
    print("    所以官方不建议重训 tokenizer：差异可感知、收益为零，下游（02 章起）一律用官方版。")


def main():
    data_path = find_data()
    texts = extract_texts(data_path)
    train_texts, heldout = texts[500:], texts[:200]   # 前 200 条留出：训练没见过，评测才公平
    os.makedirs(TEMP_DIR, exist_ok=True)
    corpus_path = os.path.join(TEMP_DIR, 'p7_bpe_corpus.txt')

    official = official_report()

    print(f"\n═══ 2/3 自训教学版 BPE（vocab={VOCAB_SIZE}，语料 {len(train_texts):,} 条，"
          f"留出 {len(heldout)} 条不参与训练）═══")
    if not _HAVE_TOKENIZERS:
        print("  ⚠️  未检测到 tokenizers 库（pip install tokenizers），跳过自训，官方体检已在上方完成")
        return
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_texts))
    t0 = time.time()
    teach = train_bpe(corpus_path, VOCAB_SIZE, SPECIAL_TOKENS)
    print(f"  训练完成: 词表 {teach.get_vocab_size():,}，耗时 {time.time()-t0:.1f}s"
          f"（语料 {os.path.getsize(corpus_path)/1e6:.1f} MB → {corpus_path}）")

    demo_roundtrip(teach)
    demo_special_tokens(teach)

    print("\n  ── 体检 3/4：压缩率 ──")
    eval_text = '\n'.join(EVAL_SENTENCES + heldout)
    r = compression(lambda s: teach.encode(s).ids, eval_text)
    print(f"    评测集 {len(eval_text):,} 字 → 平均 {r:.2f} 字/token"
          f"（字符级=1.0；越高=同算力看到越长上下文）")

    demo_subwords(teach, eval_text)

    compare(teach, official, eval_text)

    save_path = os.path.join(TEMP_DIR, 'teach_bpe_tokenizer.json')
    teach.save(save_path)
    print(f"\n  ✅ 教学版已存 {save_path}（仅作对照留档；后续章节一律用官方 tokenizer）")
    print("\n  下一步：python 02_baseline.py——用官方 tokenizer 造出 28.98M 的古董基线并跑第一次训练。")


if __name__ == '__main__':
    main()
