#!/usr/bin/env python3
"""
Part 13 作业：数据工程。纯标准库可完成。实现后运行 test_data_exercises.py。
"""

# ── 题 1：Jaccard 与 shingling（25 分）──────────────────────
def shingles(text, k=3):
    """小写化 → 提取连续单词（只留字母词）→ k-gram 集合。"""
    # TODO: re 提取 [a-z]+ 词 → "w1 w2 w3" 形式的 k-gram 集合
    return None


def jaccard(s1, s2):
    """|A∩B|/|A∪B|；两个都空 = 1.0。"""
    # TODO: 一行
    return None


# ── 题 2：MinHash 签名性质（25 分）──────────────────────────
def signature_agreement(sig1, sig2):
    """两个签名的一致比例 = 相等维数 / 总维数（无 torch 依赖）。"""
    # TODO: 一行
    return None


# ── 题 3：分带 LSH 概率（25 分）─────────────────────────────
def lsh_hit_probability(j, bands, rows):
    """P(成为候选) = 1 - (1 - j**rows) ** bands。"""
    # TODO: 一行
    return None


def choose_bands_for_recall(j, rows, target=0.99):
    """给定目标召回（对 J=j 的文档对命中概率 ≥ target），求最小 bands。
    Returns:
        int：最小的 b 使 1-(1-j^r)^b >= target；j<=0 或 j>=1 时返回 1
    """
    # TODO: 从 1 起枚举 b（上限 10000）
    return None


# ── 题 4：去重簇消解（25 分）────────────────────────────────
def keep_first_per_cluster(doc_names, duplicate_pairs):
    """给定文档名列表与重复对（(a,b) 表示 a/b 重复），保留每簇"列表顺序最先"的，
    其余丢弃。重复关系可能传递（a-b, b-c → 簇 {a,b,c}）。
    Returns:
        (kept: list[str], dropped: list[str])——kept 保持原列表顺序
    """
    # TODO:
    #   1. 并查集或邻接 BFS 求连通簇
    #   2. 每簇保留"原顺序最先出现"的文档
    return None
