#!/usr/bin/env python3
"""
Assignment 18: RAG 全链路（对应 Part 18）。
四题必做 + 一题 🌟 Stretch。纯标准库可完成（不需要 torch/模型）。
实现后运行：python test_rag_exercises.py（或 pytest test_rag_exercises.py）

四个必做函数与课程脚本同名同签名（courses/Part18_rag/scripts/01_minimal_rag.py
与 03_rag_eval.py）——写完可以直接回课程脚本里对照。
"""

import math
import re


# ── 已提供的辅助：中英混合分词（英文 [a-z0-9]+ 词 + 中文单字与相邻二元）──
def _tokens(text):
    """题 2 会用到；题 4 的 mock judge 不依赖它。不要修改。"""
    t = text.lower()
    words = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', t)
    cjk = [w for w in words if '\u4e00' <= w <= '\u9fff']
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return words + bigrams


# ── 已提供的辅助：answer → 原子 claims（题 4 用，与课程脚本 03 同款）──
def split_claims(answer):
    """按句切分，过滤列表符号等残片。不要修改。"""
    parts = re.split(r'(?<=[。！？!?])|\n', answer)
    claims = []
    for p in parts:
        p = re.sub(r'^[\s\d\-*•·.、）)]+', '', p).strip()
        if len(p) >= 6:
            claims.append(p)
    return claims


# ── 题 1：递归分块（25 分）──────────────────────────────────
def recursive_chunk(text, size=512, overlap=64):
    """递归分块：按 ['\\n\\n', '\\n', '。', ' '] 逐级切出原子段，贪心装进
    <= size 字符的 chunk；相邻 chunk 共享前一块的最后 overlap 个字符。

    必须满足的三条不变量（测试就测这三条）：
      ① len(chunk) <= size（对所有 chunk）
      ② 除首块外，每块以上一块尾部 min(overlap, len(prev)) 个字符开头
      ③ 把每块去掉开头 overlap 段后按序拼接，去空白后与原文去空白一致（不丢字符）

    Args:
        text (str): 原始文档
        size (int): chunk 最大长度（字符，含 overlap 部分）
        overlap (int): 相邻 chunk 的重叠字符数（约定 overlap < size）
    Returns:
        list[str]：空文本返回 []；否则至少返回 1 块
    """
    # TODO:
    #   1. 递归下钻分隔符列表把文本切成"原子段"（段落→行→句→词，越切越细；
    #      所有分隔符都切不动、单段仍超长时按固定步长硬切）
    #   2. 贪心装箱：装不下当前原子段就结算，新块以旧块尾部 overlap 字符开头
    #   3. 注意给 overlap 前缀和连接空格预留长度，保证不变量 ①
    #   4. ⚠️ '。' 是内容字符：直接 s.split('。') 会把句号丢掉、破坏不变量 ③——
    #      用捕获组 re.split('(。)', s) 把它保留成独立原子段
    return None


# ── 题 2：手写 BM25（25 分）────────────────────────────────
def bm25_scores(query, chunks, k1=1.2, b=0.75):
    """Okapi BM25：
    score(q, d) = Σ_{t∈q(按出现次数)} IDF(t) · tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))
    IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   —— 恒正

    Args:
        query (str): 查询（用 _tokens 分词）
        chunks (list[str]): 全部 chunk
        k1, b: BM25 超参（默认经典值）
    Returns:
        list[float]：与 chunks 等长、位置对齐的分数（越大越相关；不含词则 0.0）
    """
    # TODO:
    #   1. 对所有 chunk 分词，统计 df（每个词出现在几个 chunk）与 avgdl
    #   2. 对每个 chunk 统计 tf，按公式累加查询词得分
    #   3. 查询里出现 2 次的词要累加 2 次（标准 Okapi 形态）
    return None


# ── 题 3：RRF 融合（25 分）─────────────────────────────────
def rrf_fuse(list_a, list_b, k=60):
    """Reciprocal Rank Fusion：score(item) = Σ_{两个榜单} 1/(k + rank)，rank 从 1 起。
    只融合名次不融合分值；并列时按"list_a 先出现、再 list_b 先出现"稳定排序。

    Args:
        list_a, list_b: 两个排名列表（最优在前；元素任意可哈希对象）
        k (int): 平滑常数（论文默认 60）
    Returns:
        list：融合后的排名（最优在前）。两个输入都空返回 []。
        元素只在一个榜单出现也保留（按单榜名次参与融合）。
    """
    # TODO:
    #   1. 对两个列表逐名累加 1/(k+rank)；同时记录每个元素首次出现的顺序
    #   2. 按 分数降序、并列按首次出现序 排序输出
    #   3. k→∞ 时应退化为"入选榜单数优先、名次和次之"的计数排序（测试验证）
    return None


# ── 题 4：手写 faithfulness（25 分）────────────────────────
def faithfulness(answer, contexts, judge):
    """RAGAS faithfulness 的手写版：
    answer 拆成原子 claims（用已提供的 split_claims），逐条问 judge
    "根据以下上下文，该陈述是否被支持？"，分数 = 支持 claims 数 / 总 claims 数。

    计分口径（文档化，测试按此校验）：
      judge 输出解析出 yes  → 计 1
      解析出 no 或 unsure   → 计 0（unsure 一律按"不支持"——保守口径）
      解析规则：输出文本里含 'unsure' → unsure；含 'yes' → yes；含 'no' → no；
      都不含 → unsure

    Args:
        answer (str): 待评回答
        contexts (list[str]): 检索上下文（作为唯一判定依据）
        judge (Callable[[str], str]): 输入 prompt 字符串、返回文本的可调用对象。
            约定：prompt 中应包含当前 claim 的文本（逐条判定的自然设计）。
    Returns:
        float：支持数 / 总 claims；无有效 claims（空答案）返回 None
    """
    # TODO:
    #   1. split_claims(answer) 拆 claims；为空返回 None
    #   2. 每条 claim 构造 prompt（含上下文与该 claim），调 judge 拿到输出
    #   3. 解析 yes/no/unsure，统计支持数，返回比值
    return None


# ── 🌟 题 5（Stretch，附加 10 分，未实现返回 None → 测试 SKIP ⏭️）──
def hybrid_weight_sweep(dense_scores, sparse_scores, relevant_sets,
                        weights=None, k=5):
    """网格搜索加权混合 score = w·dense + (1-w)·sparse 的最优权重 w。

    两路分数量纲不同（cosine ∈ [-1,1]，BM25 无界正数），必须先对**每个 query
    的每路分数**做 min-max 归一化到 [0,1] 再加权，否则 sparse 会统治融合。

    Args:
        dense_scores (list[list[float]]): 每个 query 的 dense 分数（与 chunk 对齐）
        sparse_scores (list[list[float]]): 同形的 BM25 分数
        relevant_sets (list[set[int]]): 每个 query 的相关 chunk 下标集合
        weights (list[float] | None): w 网格，默认 [0, 0.1, ..., 1.0]
        k (int): recall@k 的 k
    Returns:
        (best_w, curve)：
          best_w (float)——平均 recall@k 最高的 w（并列取网格中先出现者）
          curve (list[tuple[float, float]])——[(w, mean_recall), ...]，按 weights 顺序
        未实现返回 None。
    """
    # TODO:
    #   1. 逐 query 对两路分数 min-max 归一化
    #   2. 对每个 w：融合 → 排名 → recall@k = |topk ∩ rel| / min(|rel|, k)，
    #      对全部 query 取平均
    #   3. 返回最优 w 与完整曲线（可另用 matplotlib 画 recall-α 曲线，不参与测试）
    return None
