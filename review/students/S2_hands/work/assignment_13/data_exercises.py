#!/usr/bin/env python3
"""
Part 13 作业：数据工程。纯标准库可完成。实现后运行 test_data_exercises.py。
"""

import re

# ── 题 1：Jaccard 与 shingling（25 分）──────────────────────
def shingles(text, k=3):
    """小写化 → 提取连续单词（只留字母词）→ k-gram 集合。"""
    words = re.findall(r"[a-z]+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(s1, s2):
    """|A∩B|/|A∪B|；两个都空 = 1.0。"""
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


# ── 题 2：MinHash 签名性质（25 分）──────────────────────────
def signature_agreement(sig1, sig2):
    """两个签名的一致比例 = 相等维数 / 总维数（无 torch 依赖）。"""
    same = sum(1 for x, y in zip(sig1, sig2) if x == y)
    return same / len(sig1)


# ── 题 3：分带 LSH 概率（25 分）─────────────────────────────
def lsh_hit_probability(j, bands, rows):
    """P(成为候选) = 1 - (1 - j**rows) ** bands。"""
    return 1 - (1 - j ** rows) ** bands


def choose_bands_for_recall(j, rows, target=0.99):
    """给定目标召回（对 J=j 的文档对命中概率 ≥ target），求最小 bands。
    Returns:
        int：最小的 b 使 1-(1-j^r)^b >= target；j<=0 或 j>=1 时返回 1
    """
    if j <= 0 or j >= 1:
        return 1
    for b in range(1, 10001):
        if 1 - (1 - j ** rows) ** b >= target:
            return b
    return 10000  # 到上限仍未达标，返回上限（避免死循环）


# ── 题 4：去重簇消解（25 分）────────────────────────────────
def keep_first_per_cluster(doc_names, duplicate_pairs):
    """给定文档名列表与重复对（(a,b) 表示 a/b 重复），保留每簇"列表顺序最先"的，
    其余丢弃。重复关系可能传递（a-b, b-c → 簇 {a,b,c}）。
    Returns:
        (kept: list[str], dropped: list[str])——kept 保持原列表顺序
    """
    # 思考题 Q2 的反例提醒我不能"逐对丢第二个"（会漏删同簇文档），先求连通分量
    adj = {}
    for a, b in duplicate_pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited = set()
    keep = set()
    for name in doc_names:          # 按原顺序遍历：每簇最先遇到的就是要保留的
        if name in visited:
            continue
        stack = [name]
        while stack:                # DFS 求整个连通簇（含传递闭包）
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(adj.get(cur, ()))
        keep.add(name)
    kept = [n for n in doc_names if n in keep]
    dropped = [n for n in doc_names if n not in keep]
    return kept, dropped


# ── 🌟 题 5（Stretch，附加 10 分，未实现返回 None → 测试 SKIP ⏭️）──
def keep_best_per_cluster(doc_names, doc_lengths, duplicate_pairs):
    """带偏好的簇消解：每簇保留"长度最长"的文档（Data-Juicer 的
    "保留文本最长的"策略，02 章对照表第 4 行）。并列长度 → 保留原顺序最先的。

    Args:
        doc_names: list[str]——文档名列表（定义"原顺序"）
        doc_lengths: dict[str, int]——文档名 → 长度（如词数）
        duplicate_pairs: 重复对列表，(a, b) 表示 a/b 重复（关系可传递，
                         与题 4 相同：a-b, b-c → 簇 {a,b,c}）
    Returns:
        (kept: list[str], dropped: list[str])——两者都保持 doc_names 的原顺序
    """
    # S2 本轮未做（Stretch），保持 None → 测试 SKIP
    return None
