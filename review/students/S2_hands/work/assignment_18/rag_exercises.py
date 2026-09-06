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
    max_atom = size - overlap - 2

    def split_atoms(s, seps):
        if len(s) <= max_atom:
            return [s]
        if not seps:
            return [s[i:i + max_atom] for i in range(0, len(s), max_atom)]
        sep, rest = seps[0], seps[1:]
        # '。' 是内容字符：用捕获组保留，否则 split 会丢句号、破坏不变量③
        parts = re.split(f'({re.escape(sep)})', s) if sep == '。' else s.split(sep)
        pieces = []
        for part in parts:
            pieces.extend(split_atoms(part, rest))
        return pieces

    atoms = [a for a in split_atoms(text.strip(), ['\n\n', '\n', '。', ' ']) if a.strip()]
    chunks, cur = [], ''
    for atom in atoms:
        if cur and len(cur) + len(atom) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:]
        cur = atom if not cur else cur + ' ' + atom
    if cur.strip():
        chunks.append(cur)
    return chunks


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
    n = len(chunks)
    if n == 0:
        return []
    doc_toks = [_tokens(c) for c in chunks]
    avgdl = sum(len(d) for d in doc_toks) / n
    df = {}
    for dt in doc_toks:
        for term in set(dt):
            df[term] = df.get(term, 0) + 1
    scores = []
    for dt in doc_toks:
        tf = {}
        for term in dt:
            tf[term] = tf.get(term, 0) + 1
        s = 0.0
        for qt in _tokens(query):
            if qt not in tf:
                continue
            idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
            s += idf * tf[qt] * (k1 + 1) / (
                tf[qt] + k1 * (1 - b + b * len(dt) / avgdl))
        scores.append(s)
    return scores


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
    scores, order = {}, {}
    for lst in (list_a, list_b):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores, key=lambda it: (-scores[it], order[it]))


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
    claims = split_claims(answer)
    if not claims:
        return None
    ctx = '\n'.join(contexts)
    supported = 0
    for claim in claims:
        prompt = (f'根据以下上下文判断陈述是否被支持，只回答 yes/no/unsure。\n'
                  f'上下文：{ctx}\n陈述：{claim}')
        out = (judge(prompt) or '').lower()
        if 'yes' in out:
            supported += 1
    return supported / len(claims)


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
