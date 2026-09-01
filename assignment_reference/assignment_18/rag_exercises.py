#!/usr/bin/env python3
"""
Assignment 18 参考实现（教师版）：与作业骨架 rag_exercises.py 同名同签名。
四题必做 + 🌟 题 5 全部实现；与课程脚本 courses/Part18_rag/scripts/ 的实现一致。
"""

import math
import re


# ── 已提供的辅助：中英混合分词（英文 [a-z0-9]+ 词 + 中文单字与相邻二元）──
def _tokens(text):
    t = text.lower()
    words = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', t)
    cjk = [w for w in words if '\u4e00' <= w <= '\u9fff']
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return words + bigrams


# ── 已提供的辅助：answer → 原子 claims ──
def split_claims(answer):
    parts = re.split(r'(?<=[。！？!?])|\n', answer)
    claims = []
    for p in parts:
        p = re.sub(r'^[\s\d\-*•·.、）)]+', '', p).strip()
        if len(p) >= 6:
            claims.append(p)
    return claims


# ── 题 1：递归分块 ──────────────────────────────────────────
def recursive_chunk(text, size=512, overlap=64):
    """三条不变量：① len(chunk)<=size ② 相邻块共享 overlap 尾巴 ③ 不丢字符。"""
    if not text or not text.strip():
        return []
    max_atom = size - overlap - 2  # 给 overlap 前缀 + 连接空格留位，保证 ①

    def split_atoms(s, seps):  # 递归下钻：优先大分隔符，切不动换小分隔符
        if len(s) <= max_atom:
            return [s]
        if not seps:            # 都切不动 → 硬切
            return [s[i:i + max_atom] for i in range(0, len(s), max_atom)]
        sep, rest = seps[0], seps[1:]
        # '。'是内容字符：捕获组保留（split 丢分隔符会破坏不变量③）
        parts = re.split(f'({re.escape(sep)})', s) if sep == '。' else s.split(sep)
        pieces = []
        for part in parts:
            pieces.extend(split_atoms(part, rest))
        return pieces

    atoms = [a for a in split_atoms(text.strip(), ['\n\n', '\n', '。', ' ']) if a.strip()]
    chunks, cur = [], ''
    for atom in atoms:
        if cur and len(cur) + len(atom) + 1 > size:  # 装不下 → 结算
            chunks.append(cur)
            cur = cur[-overlap:]   # 上下文桥
        cur = atom if not cur else cur + ' ' + atom
    if cur.strip():
        chunks.append(cur)
    return chunks


# ── 题 2：手写 BM25 ─────────────────────────────────────────
def bm25_scores(query, chunks, k1=1.2, b=0.75):
    """Okapi BM25：IDF 恒正 + TF 饱和 + 长度归一。"""
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
        for qt in _tokens(query):  # 查询词按出现次数累加
            if qt not in tf:
                continue
            idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
            denom = tf[qt] * (k1 + 1)
            norm = tf[qt] + k1 * (1 - b + b * len(dt) / avgdl)
            s += idf * denom / norm
        scores.append(s)
    return scores


# ── 题 3：RRF 融合 ──────────────────────────────────────────
def rrf_fuse(list_a, list_b, k=60):
    """score(item) = Σ 1/(k+rank)；并列按首次出现序稳定输出。"""
    scores, order = {}, {}
    for lst in (list_a, list_b):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores, key=lambda it: (-scores[it], order[it]))


# ── 题 4：手写 faithfulness ─────────────────────────────────
def _parse_verdict(text):
    t = text.lower()
    for kw in ('unsure', '无法确定', '不确定'):
        if kw in t:
            return 'unsure'
    if 'yes' in t or t.startswith('支持'):
        return 'yes'
    if 'no' in t or t.startswith('不支持'):
        return 'no'
    return 'unsure'


def faithfulness(answer, contexts, judge):
    """claims 逐条 entailment 判定；yes 计 1，no/unsure 计 0；无 claims 返回 None。"""
    claims = split_claims(answer)
    if not claims:
        return None
    ctx = '\n'.join(contexts)[:4000]
    supported = 0
    for c in claims:
        verdict = _parse_verdict(judge(
            f'请只依据下面的【上下文】判断【陈述】是否被支持。\n'
            f'严格三选一回答：yes（被支持）/ no（与上下文矛盾）/ unsure（没有相关信息）。\n'
            f'【上下文】\n{ctx}\n【陈述】{c}\n回答：'))
        if verdict == 'yes':
            supported += 1
    return supported / len(claims)


# ── 🌟 题 5：hybrid 权重网格搜索 ─────────────────────────────
def _minmax(xs):
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def hybrid_weight_sweep(dense_scores, sparse_scores, relevant_sets,
                        weights=None, k=5):
    """min-max 归一化两路分数后，对每个 w 计算 mean recall@k，返回 (best_w, curve)。"""
    if weights is None:
        weights = [i / 10 for i in range(11)]
    curve = []
    for w in weights:
        recs = []
        for ds, ss, rel in zip(dense_scores, sparse_scores, relevant_sets):
            nd, ns = _minmax(ds), _minmax(ss)
            fused = [w * a + (1.0 - w) * b_ for a, b_ in zip(nd, ns)]
            order = sorted(range(len(fused)), key=lambda i: -fused[i])
            hits = len(set(order[:k]) & set(rel))
            recs.append(hits / max(1, min(len(rel), k)))
        curve.append((w, sum(recs) / len(recs)))
    best_w, best_r = curve[0]
    for w, r in curve[1:]:
        if r > best_r:  # 严格大于 → 并列取网格中先出现者
            best_w, best_r = w, r
    return best_w, curve
