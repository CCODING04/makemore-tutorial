#!/usr/bin/env python3
"""
Part 13 - 脚本 01: 手写 MinHash + LSH 去重（FineWeb/Data-Juicer 同款思想的最小实现）
目标：在 ~100 篇玩具文档上完整走一遍工业去重的四个阶段：
      shingling → MinHash 签名 → 分带 LSH → Jaccard 验证，
      并与暴力 O(n²) Jaccard 对照：召回的候选对、去重结果、LSH 概率性质。
对应教程：tutorial/01_dedup_from_scratch.md（Data-Juicer 就是这条管线的工业版）
运行（CPU 即可，<10 秒）：python 01_minhash_dedup.py
"""

import os
import re
import sys
import random
from itertools import combinations

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 0. 玩具语料：30 篇"新闻"，其中手工埋 3 组近似重复 ───
BASE_DOCS = [
    "the quick brown fox jumps over the lazy dog near the river bank",
    "deep learning models are trained on large text corpora with deduplication",
    "the president announced a new policy on climate change yesterday",
    "researchers found that data quality matters more than model size",
    "the stock market rallied after the central bank cut interest rates",
    "a new study shows that sleep improves memory consolidation in adults",
    "the football team won the championship after a dramatic final match",
    "engineers optimized the inference engine to reduce latency by half",
    "the museum unveiled a rare manuscript from the twelfth century",
    "quantization reduces model memory while keeping accuracy almost intact",
]


def make_corpus():
    random.seed(42)
    docs = []
    for i, d in enumerate(BASE_DOCS):
        docs.append((f"doc{i:02d}", d))
    # 埋重复：同义改写（真实网络里"近似重复"的形态）
    docs.append(("dupA", BASE_DOCS[0] + " under the morning sun"))
    docs.append(("dupB", "the quick brown fox jumped over a lazy dog near the riverbank"))
    docs.append(("dupC", "researchers found data quality matters more than model size overall"))
    docs.append(("dupD", BASE_DOCS[9] + " quantization is useful for deployment"))
    return docs


# ─── 1. shingling：文档 → k-gram 集合（Jaccard 的比较单位）───
def shingles(text, k=3):
    words = re.findall(r"[a-z]+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


# ─── 2. MinHash：用最小哈希值集合近似 Jaccard ───
NUM_HASHES = 64


def minhash_signature(shingle_set, num_hashes=NUM_HASHES, seed=7):
    """h_j(sh) = (a_j * hash(sh) + b_j) mod P 的最小值。签名维度 = num_hashes。"""
    rng = random.Random(seed)
    P = (1 << 31) - 1
    params = [(rng.randint(1, P - 1), rng.randint(0, P - 1)) for _ in range(num_hashes)]
    hashes = [hash(s) & 0x7FFFFFFF for s in shingle_set]
    sig = []
    for a, b in params:
        sig.append(min((a * h + b) % P for h in hashes))
    return sig


# ─── 3. 分带 LSH：b bands × r rows → 只比较"某一带完全相同"的候选对 ───
def lsh_candidates(signatures, bands=16, rows=4):
    """签名切 bands 段，每段 rows 维；任何一带相等 → 候选对。
    概率性质：P(成为候选) = 1 - (1 - J^r)^b —— 相似度越高越必然命中。"""
    buckets = {}
    for name, sig in signatures.items():
        for band in range(bands):
            key = (band, tuple(sig[band * rows:(band + 1) * rows]))
            buckets.setdefault(key, []).append(name)
    pairs = set()
    for members in buckets.values():
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
    return sorted(pairs)


def main():
    docs = make_corpus()
    sh = {name: shingles(text) for name, text in docs}
    print("═══ 手写 MinHash + LSH 去重 ═══")
    print(f"  文档数={len(docs)}, shingles(3-gram), 签名 {NUM_HASHES} 维, LSH 16 bands × 4 rows\n")

    # ── 真值：暴力 O(n²) Jaccard（n 大时不可行——这正是 MinHash 存在的理由）──
    truth_pairs = [(a, b) for (a, _), (b, _) in combinations(docs, 2)
                   if jaccard(sh[a], sh[b]) >= 0.5]

    # ── MinHash 签名 → LSH 候选 → Jaccard 验证 ──
    sigs = {name: minhash_signature(sh[name]) for name, _ in docs}
    cands = lsh_candidates(sigs)
    confirmed = [(a, b) for a, b in cands if jaccard(sh[a], sh[b]) >= 0.5]

    print(f"[1] 暴力 Jaccard（真值）      : {len(truth_pairs)} 对: {truth_pairs}")
    print(f"[2] LSH 候选对                : {len(cands)} 对: {cands}")
    print(f"[3] LSH 候选 + Jaccard≥0.5    : {len(confirmed)} 对: {confirmed}")

    # ── 性质验证 ──
    assert set(confirmed) == set(truth_pairs), "LSH 漏掉了真重复！"
    # MinHash 近似性质：签名一致比例 ≈ Jaccard（大样本下）
    a, b = "dupA", "doc00"
    agree = sum(1 for x, y in zip(sigs[a], sigs[b]) if x == y) / NUM_HASHES
    exact = jaccard(sh[a], sh[b])
    print(f"\n[4] 性质: 签名一致率 {agree:.2f} ≈ 真实 Jaccard {exact:.2f}"
          f"   （P[minhash 相等] = Jaccard，64 维采样）")
    # 去重执行：保留每簇第一个
    seen, keep = set(), []
    for name, _ in docs:
        if not any(name in c and (c[0] == name or c[1] == name) and
                   (c[0] in keep or c[1] in keep) for c in confirmed):
            keep.append(name)
    # 简单策略：confirmed 里的第二个元素丢弃
    drop = {b for _, b in confirmed}
    keep = [name for name, _ in docs if name not in drop]
    print(f"[5] 去重结果: {len(docs)} → {len(keep)} 篇（丢弃 {sorted(drop)}）")
    print(f"""
═══ 与 Data-Juicer 的对照（02 章）═══
  手写版这 60 行，对应 data-juicer 的 document_minhash_deduplicator：
    shingling  → 内置分词+Cython 加速
    签名       → C++/矢量化计算（百万文档级）
    分带 LSH   → Ray 分布式分桶
    验证去重   → 簇消解策略（keep-first）+ 逐 op 追踪审计
  相同的数学，差 4 个数量级的工程。这就是"手写 → 工具"的学习闭环。
  💡 面试：LSH 为什么能不漏掉高相似对？→ P(候选|J) = 1-(1-J^r)^b，J 高时概率趋近 1；
     代价是低相似对有少量误报（被 [3] 的 Jaccard 验证挡住）。""")


if __name__ == '__main__':
    main()
