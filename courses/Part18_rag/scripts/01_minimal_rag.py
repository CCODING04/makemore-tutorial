#!/usr/bin/env python3
"""
Part 18 - 脚本 01: 手写 RAG 五件套（递归分块 → 嵌入 → BM25 → RRF 混合 → 重排 → 生成）
目标：在本仓库 docs/ 的 8 篇真实 Markdown 上，不借助任何 RAG 框架，从零拼出一条
      最小但"五脏俱全"的检索增强生成管线，并用 3 个带 ground truth 的查询对比
      dense / BM25 / hybrid / +rerank 四种检索形态的 recall@5。
对应教程：tutorial/01_naive_to_hybrid.md（五件套逐行讲解）
运行（GPU 约几分钟 / 纯 CPU 可接受；模型缺失时自动降级，脚本永不崩）：
      CUDA_VISIBLE_DEVICES=0 python 01_minimal_rag.py
强制体验降级路径：RAG18_FORCE_FALLBACK=1 python 01_minimal_rag.py
"""

import os
import re
import sys
import math
import time
import hashlib

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────── 0. 配置 ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'docs')  # 语料：本仓库 8 篇 md

CORPUS_FILES = [  # 挑内容相对稳定的文档；不用 gap_closure_plan.md（滚动更新）
    'course_roadmap_v3.md', 'llm_interview_guide.md', 'paper_reading_guide.md',
    'references_by_part.md', 'datasets.md', 'part8_post_training_plan.md',
    'part9_cuda_kernels_plan.md', 'part11_14_tutorial_design.md',
]
EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B'   # 检索嵌入（last-token pooling）
GEN_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'    # 生成
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'    # cross-encoder 重排

CHUNK_SIZE, CHUNK_OVERLAP = 512, 64  # 字符级
BM25_K1, BM25_B = 1.2, 0.75          # BM25 经典默认值
RRF_K = 60                            # RRF 论文默认
TOP_K, CAND_K = 5, 10                 # 最终取 top5，混合/重排在 top10 候选池上做
FORCE_FALLBACK = os.environ.get('RAG18_FORCE_FALLBACK') == '1'  # 强制降级演示

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════ 五件套之一：递归分块 ══════════════════════════
def recursive_chunk(text, size=512, overlap=64):
    """递归分块：按 ['\\n\\n', '\\n', '。', ' '] 逐级切出原子段，再贪心装进
    <= size 字符的 chunk；相邻 chunk 共享前一块的最后 overlap 个字符。

    Args:
        text (str): 原始文档
        size (int): chunk 最大长度（字符，含 overlap 部分）；要求 overlap < size
        overlap (int): 相邻 chunk 的重叠字符数
    Returns:
        list[str]：chunk 列表。空文本返回 []。
        不变量：① len(chunk) <= size；② 除首块外每块以上一块尾部 overlap 个
        字符开头；③ 把每块去掉开头 overlap 段后按序拼接，去空白后与原文
        去空白完全一致（不丢字符）。
    """
    if not text or not text.strip():
        return []
    max_atom = size - overlap - 2  # 给 overlap 前缀 + 连接空格留位，保证 ①

    def split_atoms(s, seps):  # 递归下钻：优先大分隔符，切不动就换小分隔符
        if len(s) <= max_atom:
            return [s]
        if not seps:            # 所有分隔符都切不动 → 硬切
            return [s[i:i + max_atom] for i in range(0, len(s), max_atom)]
        sep, rest = seps[0], seps[1:]
        # '。'是内容字符：用捕获组保留（split 会丢分隔符，破坏"不丢字符"不变量）
        parts = re.split(f'({re.escape(sep)})', s) if sep == '。' else s.split(sep)
        pieces = []
        for part in parts:
            pieces.extend(split_atoms(part, rest))
        return pieces

    atoms = [a for a in split_atoms(text.strip(), ['\n\n', '\n', '。', ' ']) if a.strip()]
    chunks, cur = [], ''
    for atom in atoms:
        # 装不下就结算当前 chunk；新 chunk 以旧 chunk 尾部 overlap 字符开头
        if cur and len(cur) + len(atom) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:]  # 上下文桥：跨块语义靠这 64 个字符续命
        cur = atom if not cur else cur + ' ' + atom
    if cur.strip():
        chunks.append(cur)
    return chunks


# ══════════════════ 五件套之二：BM25（词法检索）══════════════════
def _tokens(text):
    """中英混合分词：英文取 [a-z0-9]+ 词，中文取单字 + 相邻二字 bigram。
    中文没有空格分隔，字符 n-gram 是 BM25 处理中文的经典做法。"""
    t = text.lower()
    words = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', t)
    cjk = [w for w in words if '\u4e00' <= w <= '\u9fff']
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return words + bigrams


def bm25_scores(query, chunks, k1=1.2, b=0.75):
    """手写 Okapi BM25。
    score(q, d) = Σ_{t∈q} IDF(t) · tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))
    IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   —— 恒正，避免负 IDF
    哲学与 Part 13 的 LSH 一脉相承：都是"精确算不动就设计可算的近似"——
    LSH 用分带哈希近似 Jaccard，BM25 用 TF 饱和 + 文档长度归一近似"词项重要性"。
    Returns: list[float]，与 chunks 等长、位置对齐的分数（越大越相关）。
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
        for qt in _tokens(query):  # 查询词按出现次数累加（标准 Okapi 形态）
            if qt not in tf:
                continue
            idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
            denom = tf[qt] * (k1 + 1)
            norm = tf[qt] + k1 * (1 - b + b * len(dt) / avgdl)
            s += idf * denom / norm
        scores.append(s)
    return scores


# ══════════════════ 五件套之三：RRF 混合融合 ══════════════════
def rrf_fuse(list_a, list_b, k=60):
    """Reciprocal Rank Fusion：score(item) = Σ_lists 1/(k + rank)，rank 从 1 起。
    只融合"名次"不融合"分值"——天然免尺度标定，这是它取代加权混合的工业原因。
    并列时按 (list_a 中先出现, 再 list_b 中先出现) 稳定排序。
    Returns: list——融合后的排名（最优在前）。空输入返回 []。
    """
    scores, order = {}, {}
    for lst in (list_a, list_b):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores, key=lambda it: (-scores[it], order[it]))


# ══════════════════ 五件套之四：嵌入（含降级路径）══════════════════
def hash_embed(text, dim=256):
    """降级嵌入：hashing trick（特征哈希）。
    对每个 token 与 2-gram 做 md5 → (桶下标, 符号)，累加成定长向量再归一化。
    完全确定性（不依赖 PYTHONHASHSEED）、可复现、零模型依赖——
    语义为零、字面可用，正好用来体会"嵌入模型到底贡献了什么"。"""
    vec = torch.zeros(dim)
    toks = _tokens(text)
    grams = toks + [toks[i] + toks[i + 1] for i in range(len(toks) - 1)]
    for g in grams:
        h = hashlib.md5(g.encode('utf-8')).digest()
        idx = int.from_bytes(h[:4], 'little') % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return F.normalize(vec, dim=0)


def load_embedder():
    """加载 Qwen3-Embedding-0.6B；不可用则返回 None（调用方走 hash_embed 降级）。"""
    if FORCE_FALLBACK:
        print('⚠️  RAG18_FORCE_FALLBACK=1 —— 强制使用 hashing trick 降级嵌入')
        return None
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side='left')
        model = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float32).to(DEVICE).eval()
        return tok, model
    except Exception as e:  # 模型未下载 / 无网络 / 无 transformers
        print(f'⚠️  嵌入模型不可用（{type(e).__name__}: {str(e)[:80]}）→ 降级：hashing trick 向量')
        print('    安装/下载指引：huggingface-cli download ' + EMBED_MODEL)
        return None


def last_token_pool(last_hidden, attention_mask):
    """Qwen3-Embedding 官方 pooling：取序列最后一个有效 token 的隐状态。
    （对比 BERT 式 mean-pooling：因果模型的有效语义集中在最后一个 token。）"""
    left = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left:
        return last_hidden[:, -1]                       # 左填充 → 直接取末列
    seq_len = attention_mask.sum(dim=1) - 1             # 右填充 → 逐样本取最后有效位
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_len]


QUERY_INSTR = ('Instruct: Given a web search query, retrieve relevant passages that answer the query\n'
               'Query: ')  # Qwen3-Embedding 官方查询侧指令前缀


@torch.no_grad()
def embed_texts(texts, embedder, is_query=False):
    """批量嵌入。embedder=None → hash_embed 降级。Returns: (N, dim) 已 L2 归一化。"""
    if embedder is None:
        return torch.stack([hash_embed(('' if is_query else '') + t) for t in texts])
    tok, model = embedder
    inp = [QUERY_INSTR + t for t in texts] if is_query else list(texts)
    out = []
    for i in range(0, len(inp), 16):  # 小 batch，省显存
        batch = tok(inp[i:i + 16], padding=True, truncation=True,
                    max_length=896, return_tensors='pt').to(DEVICE)
        h = model(**batch).last_hidden_state          # (B, L, hidden=1024)
        v = last_token_pool(h, batch['attention_mask'])  # (B, 1024)
        out.append(F.normalize(v, p=2, dim=-1))       # L2 归一化 → 点积即 cosine
    return torch.cat(out)


def dense_search(q_vec, chunk_mat, top_k):
    """暴力 cosine：q_vec (D,) · chunk_mat (N, D)^T → (N,) 一次广播算完。
    注：这里刻意不手写 ANN（HNSW/IVF）——8 篇文档的规模下暴力即最优；
    真正上百万级语料时换 FAISS 一行 faiss.IndexFlatIP 起步，思想不变。"""
    sims = chunk_mat @ q_vec  # (N,)
    ranked = torch.argsort(sims, descending=True).tolist()
    return ranked[:top_k], sims


# ══════════════════ 五件套之五：重排 + 生成（均含降级）══════════════════
def load_reranker():
    if FORCE_FALLBACK:
        print('⚠️  RAG18_FORCE_FALLBACK=1 —— 跳过 cross-encoder 重排（降级路径）')
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, dtype=torch.float32).to(DEVICE).eval()
        return tok, model
    except Exception as e:
        print(f'⚠️  重排模型不可用（{type(e).__name__}: {str(e)[:80]}）→ 跳过重排，保留 hybrid 排序')
        print('    安装/下载指引：huggingface-cli download ' + RERANK_MODEL)
        return None


@torch.no_grad()
def rerank(query, cand_texts, reranker):
    """cross-encoder 打分：把 (query, chunk) 拼成一对一起进模型，输出相关性 logit。
    bi-encoder 是"两人各自写简介再比对"，cross-encoder 是"当面逐词对质"——
    精度高一个量级、算贵一个量级，所以只对 top10 候选做。"""
    if reranker is None:
        return list(range(len(cand_texts)))  # 降级：保持原序
    tok, model = reranker
    pairs = [(query, t[:1600]) for t in cand_texts]
    batch = tok(pairs, padding=True, truncation=True, max_length=768,
                return_tensors='pt').to(DEVICE)
    logits = model(**batch).logits.squeeze(-1)  # (B,)
    return torch.argsort(logits, descending=True).tolist()


def load_generator():
    if FORCE_FALLBACK:
        print('⚠️  RAG18_FORCE_FALLBACK=1 —— 生成走抽取式降级')
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL, dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
        ).to(DEVICE).eval()
        # 贪心解码用不到采样参数，置空避免 transformers 的无效 flag 警告
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        return tok, model
    except Exception as e:
        print(f'⚠️  生成模型不可用（{type(e).__name__}: {str(e)[:80]}）→ 抽取式降级')
        print('    安装/下载指引：huggingface-cli download ' + GEN_MODEL)
        return None


@torch.no_grad()
def generate_answer(query, hits, chunks, generator):
    """带引用生成：把 top 片段编号为 [1][2][3] 喂给 0.5B-Instruct，
    要求句末标 [编号]——小模型对"数字编号引用"的遵从度远高于长格式引用。"""
    ctx = '\n\n'.join(f'[{k}] {src}: {chunks[i][:400]}' for k, (i, src) in enumerate(hits, 1))
    if generator is None:  # 抽取式降级：挑含查询关键词的句子
        kws = [w for w in _tokens(query) if len(w) >= 2]  # 英文词 + 中文 bigram
        sent = ''
        for k, (i, src) in enumerate(hits, 1):
            for s in re.split(r'(?<=[。！？\n])', chunks[i]):
                if s.strip() and any(kw in s.lower() for kw in kws):
                    sent += s.strip() + f' [{k}:{src}]\n'
                    break
        return sent.strip() or '(抽取式降级也未找到相关句)'
    tok, model = generator
    msgs = [{'role': 'user', 'content':
             f'你是严谨的问答助手。只依据下面的资料回答问题，每句话末尾用 [编号] 标注出处；'
             f'资料不足以回答就直说"资料不足"。\n\n{ctx}\n\n问题：{query}'}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').to(DEVICE)
    out = model.generate(**ids, max_new_tokens=220, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()


# ─────────────────────── ground truth 与主流程 ───────────────────────
# 每个 query 人工标注"相关 chunk 判定规则"：chunk 命中任一关键词组（组内全部出现）即相关。
# 三个查询分别代表三种典型形态：
#   Q1 语义型（词汇不匹配：问"出自哪篇论文/哪个框架"，BM25 被中文常用 bigram 淹没 → dense 赢）
#   Q2 词典型（问"清单叫什么"：dense 摸不着头脑，BM25 靠稀有词 TOP8 一击命中 → BM25 赢）
#   Q3 综合型（"去重"+"质量"两路都只抓到一部分 → 融合互补）
QUERIES = [
    ('GRPO 出自哪篇论文？课程里用哪个框架跑它的实战？', [('GRPO', 'DeepSeekMath')]),
    ('面试要能默写的手写代码清单叫什么？', [('TOP8',)]),
    ('预训练数据去重为什么能提升模型质量？', [('去重',)]),
]


def relevant_chunks(chunks, groups):
    rel = set()
    for i, c in enumerate(chunks):
        cl = c.lower()
        if any(all(k.lower() in cl for k in g) for g in groups):
            rel.add(i)
    return rel


def recall_at_k(ranked, rel, k):
    hits = len(set(ranked[:k]) & rel)
    return hits / max(1, min(len(rel), k))  # 分母取 min(|rel|, k)：可达到的上限是 1


def main():
    t0 = time.time()
    print('=' * 72)
    print('Part 18 - 01: 手写 RAG 五件套（dense vs BM25 vs hybrid vs +rerank）')
    print(f'设备: {DEVICE}' + (f'（{torch.cuda.get_device_name(0)}）' if DEVICE == 'cuda' else ''))
    print('=' * 72)

    # ── Step 0: 语料 ──
    print('\n[Step 0] 语料加载：docs/ 下 8 篇 Markdown')
    docs = {}
    for f in CORPUS_FILES:
        with open(os.path.join(DOCS_DIR, f), encoding='utf-8') as fh:
            docs[f] = fh.read()
        print(f'  {f:36s} {len(docs[f]):6d} chars')

    # ── Step 1: 递归分块 ──
    print(f'\n[Step 1] 递归分块：size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars')
    chunks, chunk_src = [], []
    for name, text in docs.items():
        for j, ch in enumerate(recursive_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(ch)
            chunk_src.append(f'{name}#c{j:02d}')
    lens = [len(c) for c in chunks]
    print(f'  共 {len(chunks)} 个 chunk；长度 min/mean/max = '
          f'{min(lens)}/{sum(lens)//len(lens)}/{max(lens)}')
    print(f'  示例 chunk[0] 前 80 字符: {chunks[0][:80]!r}')

    # ── Step 2: 嵌入 ──
    print(f'\n[Step 2] 嵌入（{EMBED_MODEL}，last-token pooling）')
    embedder = load_embedder()
    t = time.time()
    chunk_mat = embed_texts(chunks, embedder)          # (N, 1024) 或降级 (N, 256)
    print(f'  chunk 矩阵: {tuple(chunk_mat.shape)}，耗时 {time.time()-t:.1f}s，'
          f'设备 {chunk_mat.device}')
    if embedder is None:
        print('  [降级模式] 当前向量为 hashing trick：只有字面碰撞信号，无语义——'
              '看后面 dense 列 recall 掉多少，即嵌入模型的贡献。')

    # ── Step 3: 四种检索形态对比 ──
    print(f'\n[Step 3] 检索对比：dense / BM25 / hybrid(RRF k={RRF_K}) / +rerank，'
          f'指标 recall@{TOP_K}')
    reranker = load_reranker()
    rows = []
    for qi, (query, groups) in enumerate(QUERIES):
        rel = relevant_chunks(chunks, groups)
        q_vec = embed_texts([query], embedder, is_query=True)[0]   # (D,)
        dense_top, sims = dense_search(q_vec, chunk_mat, CAND_K)
        bm25_s = bm25_scores(query, chunks)
        bm25_top = sorted(range(len(chunks)), key=lambda i: -bm25_s[i])[:CAND_K]
        hybrid_top = rrf_fuse(dense_top, bm25_top, k=RRF_K)
        cand_texts = [chunks[i] for i in hybrid_top[:CAND_K]]
        rr = rerank(query, cand_texts, reranker)
        rerank_top = [hybrid_top[:CAND_K][j] for j in rr][:TOP_K]

        r_dense = recall_at_k(dense_top, rel, TOP_K)
        r_bm25 = recall_at_k(bm25_top, rel, TOP_K)
        r_hyb = recall_at_k(hybrid_top, rel, TOP_K)
        r_rr = recall_at_k(rerank_top, rel, TOP_K)
        rows.append((r_dense, r_bm25, r_hyb, r_rr))

        print(f'\n  Q{qi+1}: {query}')
        print(f'    ground truth: {len(rel)} 个相关 chunk → {[chunk_src[i] for i in sorted(rel)]}')
        print(f'    dense  top{TOP_K}: {[chunk_src[i] for i in dense_top[:TOP_K]]}  recall={r_dense:.2f}')
        print(f'    bm25   top{TOP_K}: {[chunk_src[i] for i in bm25_top[:TOP_K]]}  recall={r_bm25:.2f}')
        print(f'    hybrid top{TOP_K}: {[chunk_src[i] for i in hybrid_top[:TOP_K]]}  recall={r_hyb:.2f}')
        print(f'    +rr    top{TOP_K}: {[chunk_src[i] for i in rerank_top]}  recall={r_rr:.2f}')
        if qi == 0:
            top_sim = float(sims[dense_top[0]])
            print(f'    [debug] Q1 与 dense 第一名的 cosine = {top_sim:.4f}（查询侧带官方 Instruct 前缀）')

    print('\n  ' + '=' * 60)
    print(f'  {"query":>6s} | {"dense":>6s} | {"bm25":>6s} | {"hybrid":>6s} | {"+rerank":>7s}')
    for i, r in enumerate(rows):
        print(f'  {"Q"+str(i+1):>6s} | {r[0]:6.2f} | {r[1]:6.2f} | {r[2]:6.2f} | {r[3]:7.2f}')
    avg = [sum(c) / len(rows) for c in zip(*rows)]
    print(f'  {"mean":>6s} | {avg[0]:6.2f} | {avg[1]:6.2f} | {avg[2]:6.2f} | {avg[3]:7.2f}')
    print('  ' + '=' * 60)

    # ── Step 4: 生成带引用回答 ──
    print(f'\n[Step 4] 生成（{GEN_MODEL}，贪心解码，句末标 [编号] 引用）')
    generator = load_generator()
    for qi, (query, _) in enumerate(QUERIES):
        q_vec = embed_texts([query], embedder, is_query=True)[0]
        dense_top, _ = dense_search(q_vec, chunk_mat, CAND_K)
        bm25_s = bm25_scores(query, chunks)
        bm25_top = sorted(range(len(chunks)), key=lambda i: -bm25_s[i])[:CAND_K]
        hybrid_top = rrf_fuse(dense_top, bm25_top, k=RRF_K)
        cand_texts = [chunks[i] for i in hybrid_top[:CAND_K]]
        rr = rerank(query, cand_texts, reranker)
        final = [hybrid_top[:CAND_K][j] for j in rr][:3]
        hits = [(i, chunk_src[i]) for i in final]
        ans = generate_answer(query, hits, chunks, generator)
        print(f'\n  Q{qi+1}: {query}')
        print(f'  证据: {[chunk_src[i] for i in final]}')
        print(f'  回答: {ans[:500]}')

    print(f'\n总耗时 {time.time()-t0:.1f}s（含模型加载与嵌入）')
    print('下一步 → 02_contextual_retrieval.py：给每个 chunk 生成"全文定位"上下文前缀，'
          '看 recall 还能再涨多少。')


if __name__ == '__main__':
    main()
