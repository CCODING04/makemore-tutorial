#!/usr/bin/env python3
"""
Part 18 - 脚本 02: 复刻 Anthropic 的 Contextual Retrieval（上下文前置检索）+ 上下文质量消融
思想：chunk 脱离原文后"身份模糊"（一个孤立的表格/片段嵌不进任何查询语义），
      于是在嵌入前给每个 chunk 拼一段"全文定位"上下文前缀。
实验一（Anthropic 复刻主链，官方同构四阶梯）：
      plain 嵌入 → +LLM 上下文前缀（Qwen2.5-0.5B 生成 ~50 token 定位句）
      → +BM25 混合（RRF）→ +cross-encoder 重排
实验二（上下文质量消融，回答"本机为什么没增益"）：
      plain → +章节路径前缀（确定性结构信息）→ +章节路径&LLM 句
官方对照（Anthropic, 2024-09, 语料 248M chunk）：
      检索失败率：嵌入 5.7% → +上下文 3.7% → +BM25 混合 2.9% → +rerank 1.9%（累计 -67%）
      来源：https://www.anthropic.com/engineering/contextual-retrieval
附：late chunking（arXiv 2409.04701，Jina AI）对照——见文末打印与教程 02 章。
运行（GPU 约 2 分钟）：CUDA_VISIBLE_DEVICES=0 python 02_contextual_retrieval.py
共享件：五件套（分块/BM25/RRF/嵌入/重排）直接 import 自 01_minimal_rag.py（累积式脚本）。
"""

import os
import re
import sys
import time
import importlib.util

import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'docs')

# 累积式设计：直接复用 01 的五件套实现（模块名以数字开头，用 spec 方式加载）
_spec = importlib.util.spec_from_file_location('part18_s01', os.path.join(SCRIPT_DIR, '01_minimal_rag.py'))
m01 = importlib.util.module_from_spec(_spec)
sys.modules['part18_s01'] = m01
_spec.loader.exec_module(m01)

CORPUS_FILES = m01.CORPUS_FILES
GEN_MODEL = m01.GEN_MODEL
CTX_MAX_TOKENS = 64          # ~50 token 的定位句
DOC_HEAD_CHARS = 600         # 喂给生成器的"全文"开头
FUSE_POOL = 100              # RRF 融合的候选池宽度（官方融合全量排名；20+20 太窄会丢候选）
EVAL_K = 20                  # 官方口径 recall@20 + top-20 失败率
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

QUERIES = [  # 比 01 更"刁"：词汇错位重，@20 上单路仍只有部分命中，留出抬升空间
    ('组内相对策略梯度是哪篇论文提出的？', [('GRPO', 'DeepSeekMath')]),
    ('SGLang 和 vLLM 各自适合什么场景？', [('SGLang', 'vLLM')]),
    ('推理服务里 KV cache 显存碎片的问题是怎么解决的？', [('PagedAttention',)]),
]


def doc_outline(text, k=10):
    """文档大纲 = 前 k 个 markdown 标题行，给 0.5B 提供定位锚点。"""
    return '\n'.join(re.findall(r'^#{1,3} .+$', text, re.MULTILINE)[:k])


def chunk_section_paths(text, chunks):
    """确定性结构上下文：每个 chunk 归属的"章节路径"（H1 › H2 › H3）。
    做法：逐行扫原文记录 (行起始偏移, 当前章节路径)；再用 chunk 首个原子段
    在原文中定位，取不晚于它的最后一个章节路径。零模型、零随机性。"""
    sec_by_pos, cur, pos = [], '', 0
    for ln in text.split('\n'):
        m = re.match(r'^(#{1,4})\s+(.*)$', ln)
        if m:
            depth, title = len(m.group(1)), m.group(2).strip()
            parts = cur.split(' › ') if cur else []
            cur = ' › '.join((parts[:depth - 1] if depth > 1 else []) + [title])
        sec_by_pos.append((pos, cur))
        pos += len(ln) + 1
    paths, offset = [], 0
    for ch in chunks:
        probe = ch[64:].strip()[:25] if len(ch) > 64 else ch.strip()[:25]  # 跳过 overlap 前缀
        idx = text.find(probe, max(0, offset - 100))
        if idx < 0:
            idx = offset
        sec = ''
        for p, s in sec_by_pos:  # 线性扫足够（每文档行数 < 700）
            if p <= idx:
                sec = s
            else:
                break
        paths.append(sec)
        offset = idx
    return paths


def load_context_generator():
    """加载 0.5B-Instruct 做"chunk 定位句"生成。不可用时返回 None → 降级为
    空字符串前缀（实验退化为 plain），打印警告，脚本不崩。"""
    if m01.FORCE_FALLBACK:
        print('⚠️  RAG18_FORCE_FALLBACK=1 —— LLM 定位句降级为空串（实验一退化为 plain 对照）')
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL, dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
        ).to(DEVICE).eval()
        model.generation_config.temperature = None  # 贪心解码，清掉无效采样 flag 警告
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        return tok, model
    except Exception as e:
        print(f'⚠️  上下文生成模型不可用（{type(e).__name__}: {str(e)[:80]}）→ LLM 前缀降级为空串')
        print('    安装/下载指引：huggingface-cli download ' + GEN_MODEL)
        return None


CTX_PROMPT = (  # Anthropic 原版 prompt 的 0.5B 汉化 + 反指代约束（0.5B 爱写"这段内容是第 X 块"）
    '文档《{name}》的目录大纲：\n{outline}\n\n其中一段内容：\n{chunk}\n\n'
    '请用一句话概括这段内容讨论的具体主题（可提及大纲中的相关章节名）。'
    '禁止出现「第X块」「这段」「其中」等指代词。只输出这一句话。')


@torch.no_grad()
def make_context(doc_name, outline, chunk_head, generator):
    if generator is None:  # 降级：空前缀（等价于 plain 嵌入）
        return ''
    tok, model = generator
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': CTX_PROMPT.format(
            name=doc_name, outline=outline, chunk=chunk_head)}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors='pt', truncation=True, max_length=4096).to(DEVICE)
    out = model.generate(**ids, max_new_tokens=CTX_MAX_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:],
                      skip_special_tokens=True).strip()


def recall_row(mat, queries, rels, embedder):
    """给定嵌入矩阵，返回每个 query 的 recall@EVAL_K 排名列表。"""
    rows = []
    for (q, _), rel in zip(queries, rels):
        qv = m01.embed_texts([q], embedder, is_query=True)[0]  # 查询侧不做上下文化
        top, _ = m01.dense_search(qv, mat, EVAL_K)
        rows.append(top)
    return rows


def _rec(stage_rows, qi, rels):
    """第 qi 个 query 在给定排名列表上的 recall@EVAL_K。"""
    return len(set(stage_rows[qi]) & rels[qi]) / max(1, min(len(rels[qi]), EVAL_K))


def main():
    t0 = time.time()
    print('=' * 72)
    print('Part 18 - 02: Contextual Retrieval 复刻 + 上下文质量消融')
    print(f'设备: {DEVICE}' + (f'（{torch.cuda.get_device_name(0)}）' if DEVICE == 'cuda' else ''))
    print('=' * 72)

    # ── Step 0: 语料 + 分块（与 01 相同配置，保证可比）──
    docs = {}
    for f in CORPUS_FILES:
        with open(os.path.join(DOCS_DIR, f), encoding='utf-8') as fh:
            docs[f] = fh.read()
    chunks, chunk_doc, sec_paths = [], [], []
    for name, text in docs.items():
        cs = m01.recursive_chunk(text, m01.CHUNK_SIZE, m01.CHUNK_OVERLAP)
        chunks.extend(cs)
        chunk_doc.extend([name] * len(cs))
        sec_paths.extend(chunk_section_paths(text, cs))
    rels = [m01.relevant_chunks(chunks, groups) for _, groups in QUERIES]
    print(f'\n[Step 0] 语料 {len(docs)} 篇 → {len(chunks)} 个 chunk（size=512/overlap=64，与 01 一致）')
    print(f'  章节路径示例: [{sec_paths[0][:52]}...] / [{sec_paths[len(chunks)//2][:52]}...]')

    # ── Step 1: 三套嵌入矩阵 ──
    print('\n[Step 1] 嵌入：① plain ② +LLM 定位句(Anthropic 复刻) ③ +章节路径(结构化对照)')
    embedder = m01.load_embedder()
    plain_mat = m01.embed_texts(chunks, embedder)                    # (N, 1024)

    generator = load_context_generator()
    llm_ctxs, t = [], time.time()
    for i, ch in enumerate(chunks):
        llm_ctxs.append(make_context(chunk_doc[i], doc_outline(docs[chunk_doc[i]]),
                                     ch[:350], generator))
        if (i + 1) % 60 == 0:
            print(f'  LLM 定位句已生成 {i + 1}/{len(chunks)}（{time.time() - t:.0f}s）')
    print(f'  完成 {len(llm_ctxs)} 条，耗时 {time.time() - t:.0f}s。示例：')
    for i in (0, len(chunks) // 2, len(chunks) - 1):
        print(f'    [{chunk_doc[i]}] {llm_ctxs[i][:88]}')

    ctx_texts = [f'{c} {ch}' for c, ch in zip(llm_ctxs, chunks)]       # 实验一：LLM 前缀拼接
    ctx_mat = m01.embed_texts(ctx_texts, embedder)
    print(f'  plain / +LLM 前缀两套矩阵均为 {tuple(plain_mat.shape)}（实验二的变体矩阵在 Step 3 现算）')

    # ── Step 2: 实验一（Anthropic 复刻主链）──
    print(f'\n[Step 2] 实验一：plain → +LLM 前缀 → +BM25 混合(RRF k={m01.RRF_K}) → +rerank')
    reranker = m01.load_reranker()
    plain_rows = recall_row(plain_mat, QUERIES, rels, embedder)
    ctx_rows, hyb_rows, rr_rows = [], [], []
    for (q, _), rel in zip(QUERIES, rels):
        qv = m01.embed_texts([q], embedder, is_query=True)[0]
        ctx_top, _ = m01.dense_search(qv, ctx_mat, FUSE_POOL)
        bm25_s = m01.bm25_scores(q, ctx_texts)            # BM25 也在"带前缀"文本上算（官方同款）
        bm25_top = sorted(range(len(chunks)), key=lambda i: -bm25_s[i])[:FUSE_POOL]
        hyb = m01.rrf_fuse(ctx_top, bm25_top, k=m01.RRF_K)
        rr = m01.rerank(q, [ctx_texts[i] for i in hyb[:EVAL_K]], reranker)
        ctx_rows.append(ctx_top[:EVAL_K])
        hyb_rows.append(hyb[:EVAL_K])
        rr_rows.append([hyb[j] for j in rr])

    print(f'\n  {"":2s}     query                          recall@{EVAL_K}（实验一四阶梯）')
    stages1 = [plain_rows, ctx_rows, hyb_rows, rr_rows]
    for qi, (q, _) in enumerate(QUERIES):
        cells = '  '.join(f'{_rec(s, qi, rels):4.2f}' for s in stages1)
        print(f'  Q{qi + 1}  {q[:26]:28s} {cells}')
    mean1 = [sum(_rec(s, qi, rels) for qi in range(len(QUERIES))) / len(QUERIES)
             for s in stages1]
    fail1 = [sum(1 for qi in range(len(QUERIES)) if not (set(s[qi]) & rels[qi])) / len(QUERIES)
             for s in stages1]
    official = ['5.7%', '3.7%', '2.9%', '1.9%']
    print('  ' + '-' * 78)
    print(f'  {"mean":4s} {"":28s} ' + '  '.join(f'{v:4.2f}' for v in mean1))
    print(f'  {"失败率":4s} {"":26s} ' + '  '.join(f'{v:4.2f}' for v in fail1)
          + '   （官方: ' + ' / '.join(official) + '，累计 -67%）')

    # ── Step 3: 实验二（上下文质量与"格式噪声"消融）──
    print('\n[Step 3] 实验二：同样的信息、不同的前缀，recall 会怎么摆？')
    print('  （章节路径 A：`文档名 · 章节 原文`；B：`《文档名》章节：原文`——信息完全相同，仅排版不同）')
    secA_texts = [f'{chunk_doc[i]} · {sec_paths[i]} {ch}' for i, ch in enumerate(chunks)]
    secB_texts = [f'《{chunk_doc[i]}》{sec_paths[i]}：{ch}' for i, ch in enumerate(chunks)]
    secL_texts = [f'{chunk_doc[i]} · {sec_paths[i]} —— {llm_ctxs[i]} {ch}'
                  for i, ch in enumerate(chunks)]
    rowsA = recall_row(m01.embed_texts(secA_texts, embedder), QUERIES, rels, embedder)
    rowsB = recall_row(m01.embed_texts(secB_texts, embedder), QUERIES, rels, embedder)
    rowsL = recall_row(m01.embed_texts(secL_texts, embedder), QUERIES, rels, embedder)
    stages2 = [plain_rows, rowsA, rowsB, rowsL]
    for qi, (q, _) in enumerate(QUERIES):
        cells = '  '.join(f'{_rec(s, qi, rels):4.2f}' for s in stages2)
        print(f'  Q{qi + 1}  {q[:26]:26s} {cells}')
    mean2 = [sum(_rec(s, qi, rels) for qi in range(len(QUERIES))) / len(QUERIES)
             for s in stages2]
    print('  ' + '-' * 78)
    print(f'  {"mean":4s} {"":26s} ' + '  '.join(f'{v:4.2f}' for v in mean2))

    print(f"""
[结论与讨论]（详细展开见教程 02 章）
  实验一（Anthropic 复刻主链）：本机四阶梯 {mean1[0]:.2f} → {mean1[1]:.2f} → {mean1[2]:.2f} → {mean1[3]:.2f}，
  没有复现官方 5.7%→1.9% 的方向。逐条归因：
  1) 上下文生成质量：官方用 Claude 看整篇文档写定位句；本机 0.5B 只看大纲 + 前 350 字，
     定位句偶有跑题（Step 1 示例可自查）——噪声前缀会把嵌入拉离查询语义。
  2) 语料规模：官方 248M chunk 跨百万文档，"chunk 脱离文档就认不出"的问题普遍存在；
     本机 8 篇文档 234 个 chunk，plain 嵌入本来就不太缺上下文，增益空间小。
  3) 评测粒度：官方指标是"top-20 一无所获"的失败率（亿级查询平均）；本机 3 个查询上
     实验一失败率为 {fail1[0]:.0%}——{'该指标在主模式下已饱和，recall 微差纯属小样本噪声。'
     if fail1[0] == 0 else '（降级模式下不饱和：hashing 向量连 top-20 都摸不到相关 chunk。）'}
  4) 混合口径：+BM25 行在"带前缀文本"上算 BM25，前缀引入文档级高频词（df 被抬高），
     稀有词判别力被稀释——小语料上尤其明显；官方靠加权组合 + 调参绕开了这一点。
  实验二（信息相同、排版不同）：章节路径 A 版 mean {mean2[1]:.2f}、B 版 {mean2[2]:.2f}——
  信息一字不差，仅换标点排版，recall 就摆动 {abs(mean2[1] - mean2[2]):.2f}；再叠 0.5B
  定位句到 {mean2[3]:.2f}。这说明小语料 + 通用嵌入模型上，"格式噪声"与"技术增益"
  同量级——任何 contextual 改造必须配 A/B 评测与多样本查询，单点数字不可信。
  这正是 Anthropic 用 248M chunk、按检索失败率在亿级查询上平均的原因。
  工程启示：① 先上确定性结构前缀（文档名/章节路径/元数据），零成本且方向对；
  ② LLM 定位句是"语料大、生成模型强、有 prompt caching 摊成本"时才划算的选项；
  ③ 上下文工程的收益来自信息量，不是"加前缀"这个动作本身。

[对照] Late Chunking（arXiv 2409.04701，Jina AI）——同一问题的另一端解法：
  contextual retrieval = 先切块、后为每块补上下文（每块一次 LLM 调用，贵但模型无关）；
  late chunking = 先让长上下文嵌入模型把全文一次编码，再在 token 序列上切块、
  对每块做池化（零 LLM 调用，但要求嵌入模型支持长上下文且逐 token 输出）。
  一句话：一个是"给块补上下文"，一个是"让上下文先于切块发生"。展开见教程 02 章。""")

    print(f'\n总耗时 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
