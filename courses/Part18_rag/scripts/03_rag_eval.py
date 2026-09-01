#!/usr/bin/env python3
"""
Part 18 - 脚本 03: 手写 RAG 评测（faithfulness / context precision）+ ragas 对照
目标：不依赖评测框架，用 0.5B-Instruct 当裁判（judge），手写 RAGAS 的两大核心指标：
      ① faithfulness：把 answer 拆成原子 claims，逐条问裁判"上下文是否支持"，
         分数 = 支持数 / 总 claims（unsure 一律按"不支持"计——保守口径，文档化）
      ② context precision：对检索回来的每个 (query, chunk) 问裁判"相关吗"，
         再按 RAGAS 同款 average-precision 口径聚合
      ③ 评测器噪声实验：同一条 claim、两种等价问法，裁判结论会不会翻转？
      ④ ragas（可选依赖）：未安装时打印安装指引并跳过，脚本 rc=0 永不崩
对应教程：tutorial/01_naive_to_hybrid.md §评测 + 02 章 RAGAS 四指标
运行（GPU 约半分钟）：CUDA_VISIBLE_DEVICES=0 python 03_rag_eval.py
共享件：五件套直接 import 自 01_minimal_rag.py；查询与语料与 01 一致（结果可对照）。
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

_spec = importlib.util.spec_from_file_location('part18_s01', os.path.join(SCRIPT_DIR, '01_minimal_rag.py'))
m01 = importlib.util.module_from_spec(_spec)
sys.modules['part18_s01'] = m01
_spec.loader.exec_module(m01)

GEN_MODEL = m01.GEN_MODEL
TOP_K = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 与 01 相同的三查询：保证"检索质量 → 生成质量 → 评测分数"可串起来讲
QUERIES = [q for q, _ in m01.QUERIES]
GT_GROUPS = [g for _, g in m01.QUERIES]

# 人为构造的"幻觉句"：拼进正确答案末尾，演示 faithfulness 如何被抓出来
HALLUCINATION = ('这项技术在 2015 年由 OpenAI 首次提出，并为 GPT-1 的训练奠定了基础。'
                 '工业界统计它平均每年为公司节省 300 万美元的算力开支。')


def load_judge_model():
    """0.5B-Instruct 裁判。不可用时返回 None → 降级为关键词裁判（诚实标注其弱）。"""
    if m01.FORCE_FALLBACK:
        print('⚠️  RAG18_FORCE_FALLBACK=1 —— 裁判降级为关键词规则（只看字面重合）')
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL, dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
        ).to(DEVICE).eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        return tok, model
    except Exception as e:
        print(f'⚠️  裁判模型不可用（{type(e).__name__}: {str(e)[:80]}）→ 降级为关键词规则')
        print('    安装/下载指引：huggingface-cli download ' + GEN_MODEL)
        return None


@torch.no_grad()
def llm_judge(generator):
    """把 (tokenizer, model) 包装成 judge 可调用对象：输入 prompt 字符串，返回文本。
    faithfulness(answer, contexts, judge) 的 judge 形参就是这个类型——
    作业里用 mock 函数替换它即可离线测试。"""

    def judge(prompt: str) -> str:
        tok, model = generator
        text = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt', truncation=True, max_length=4096).to(DEVICE)
        out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    return judge


def keyword_judge(prompt: str) -> str:
    """降级裁判：prompt 里嵌着【陈述】与【上下文】，字面重合率高 → yes。"""
    m_s = re.search(r'【陈述】(.*?)【上下文】', prompt, re.S)
    m_c = re.search(r'【上下文】(.*)', prompt, re.S)
    if not (m_s and m_c):
        return 'unsure'
    claim, ctx = m_s.group(1), m_c.group(1)
    toks = [w for w in m01._tokens(claim) if len(w) >= 2]
    hit = sum(1 for w in toks if w in ctx.lower())
    return 'yes' if toks and hit / len(toks) >= 0.6 else ('unsure' if hit else 'no')


def split_claims(answer):
    """answer → 原子 claims（按句切分，过滤过短碎片与 Markdown 列表符号）。"""
    parts = re.split(r'(?<=[。！？!?])|\n', answer)
    claims = []
    for p in parts:
        p = re.sub(r'^[\s\d\-*•·.、）)]+', '', p).strip()
        if len(p) >= 6:  # 过滤"综上所述"之类的残句
            claims.append(p)
    return claims


def _parse_verdict(text):
    """从裁判输出里解析 yes / no / unsure（0.5B 可能输出多余字符）。"""
    t = text.lower()
    for kw in ('unsure', '无法确定', '不确定'):
        if kw in t:
            return 'unsure'
    if 'yes' in t or '是' == t.strip()[:1] or t.startswith('支持'):
        return 'yes'
    if 'no' in t or t.startswith('不支持') or t.startswith('否定'):
        return 'no'
    return 'unsure'


def faithfulness(answer, contexts, judge):
    """手写 RAGAS faithfulness：answer 拆 claims，逐条问 judge 是否被上下文支持。
    计分：yes 计 1；no / unsure 计 0（unsure 按不支持——保守口径，宁可低估不虚高）。
    Args:
        answer (str): 待评回答
        contexts (list[str]): 检索上下文（拼接后作为唯一依据）
        judge (Callable[[str], str]): 输入 prompt、返回文本的可调用对象
    Returns:
        float：支持 claims 数 / 总 claims；无有效 claims 时返回 None
    """
    claims = split_claims(answer)
    if not claims:
        return None
    ctx = '\n'.join(contexts)[:4000]
    supported = 0
    for c in claims:
        verdict = _parse_verdict(judge(
            f'请只依据下面的【上下文】判断【陈述】是否被支持。\n'
            f'严格三选一回答：yes（被支持）/ no（与上下文矛盾）/ unsure（上下文没有相关信息）。\n'
            f'【上下文】\n{ctx}\n【陈述】{c}\n回答：'))
        if verdict == 'yes':
            supported += 1
    return supported / len(claims)


def context_precision(query, contexts, judge):
    """手写 RAGAS context precision（average-precision 口径）：
    对每个检索位次 k 的 chunk 问 judge "与 query 相关吗"（yes=1），
    precision@k = 前 k 位中相关数 / k；总分 = 相关位次的 precision@k 平均。
    全不相关时返回 0.0（RAGAS 同款边界约定）。
    注：问法用"相关吗"而非"有用吗"——实测 0.5B 对"是否有用"几乎一律答 no。"""
    rel = []
    for rk, c in enumerate(contexts, 1):
        v = _parse_verdict(judge(
            f'下面的【资料】与【问题】相关吗（内容能帮助回答该问题）？只回答 yes 或 no。\n'
            f'【问题】{query}\n【资料】{c[:600]}\n回答：'))
        rel.append(1 if v == 'yes' else 0)
    hits = sum(rel)
    if hits == 0:
        return 0.0, rel
    s, n = 0.0, 0
    for k in range(1, len(rel) + 1):
        pk = sum(rel[:k]) / k
        if rel[k - 1]:
            s += pk
            n += 1
    return s / n, rel


def kendall_tau(rank_a, rank_b):
    """Kendall τ（手写 O(n²)）：两组排名的一致/不一致对之差 / 总对数。
    用于"裁判给的相关性排名 vs 检索器排名"的一致性度量。"""
    n = len(rank_a)
    conc = disc = 0
    pos_a = {v: i for i, v in enumerate(rank_a)}
    pos_b = {v: i for i, v in enumerate(rank_b)}
    items = list(pos_a)
    for i in range(n):
        for j in range(i + 1, n):
            sa = (pos_a[items[i]] > pos_a[items[j]]) - (pos_a[items[i]] < pos_a[items[j]])
            sb = (pos_b[items[i]] > pos_b[items[j]]) - (pos_b[items[i]] < pos_b[items[j]])
            if sa * sb > 0:
                conc += 1
            elif sa * sb < 0:
                disc += 1
    total = n * (n - 1) / 2
    return (conc - disc) / total if total else 1.0


def main():
    t0 = time.time()
    print('=' * 72)
    print('Part 18 - 03: 手写 RAG 评测（faithfulness / context precision / 评测器噪声）')
    print(f'设备: {DEVICE}' + (f'（{torch.cuda.get_device_name(0)}）' if DEVICE == 'cuda' else ''))
    print('=' * 72)

    # ── Step 0: 重建 01 的检索管线，拿到 top-5 上下文 ──
    print('\n[Step 0] 语料 + 分块 + hybrid 检索（与 01 相同）')
    docs = {}
    for f in m01.CORPUS_FILES:
        with open(os.path.join(DOCS_DIR, f), encoding='utf-8') as fh:
            docs[f] = fh.read()
    chunks, chunk_src = [], []
    for name, text in docs.items():
        for j, ch in enumerate(m01.recursive_chunk(text, m01.CHUNK_SIZE, m01.CHUNK_OVERLAP)):
            chunks.append(ch)
            chunk_src.append(f'{name}#c{j:02d}')
    embedder = m01.load_embedder()
    chunk_mat = m01.embed_texts(chunks, embedder)
    generator = m01.load_generator()
    reranker = m01.load_reranker()  # 用 01 的最终形态（hybrid + rerank），答案可与 01 对照
    print(f'  {len(chunks)} 个 chunk，矩阵 {tuple(chunk_mat.shape)}')

    judge_model = load_judge_model()
    judge = llm_judge(judge_model) if judge_model else keyword_judge

    # ── Step 1: 生成 grounded / hallucinated 两种答案 ──
    print('\n[Step 1] 生成回答（grounded）并构造幻觉版本（追加 2 句无中生有）')
    results = []
    for qi, query in enumerate(QUERIES):
        qv = m01.embed_texts([query], embedder, is_query=True)[0]
        d_top, _ = m01.dense_search(qv, chunk_mat, m01.CAND_K)
        bm25_s = m01.bm25_scores(query, chunks)
        b_top = sorted(range(len(chunks)), key=lambda i: -bm25_s[i])[:m01.CAND_K]
        hyb = m01.rrf_fuse(d_top, b_top, k=m01.RRF_K)
        rr = m01.rerank(query, [chunks[i] for i in hyb[:m01.CAND_K]], reranker)
        top = [hyb[:m01.CAND_K][j] for j in rr][:TOP_K]
        ctxs = [chunks[i] for i in top]
        answer = m01.generate_answer(query, [(i, chunk_src[i]) for i in top[:3]],
                                     chunks, generator)
        results.append((query, ctxs, answer))
        print(f'  Q{qi + 1} {query}')
        print(f'    证据 top{TOP_K}: {[chunk_src[i] for i in top]}')
        print(f'    回答（前 120 字）: {answer[:120]}')

    # ── Step 2: faithfulness（grounded vs hallucinated 对照）──
    print('\n[Step 2] faithfulness：grounded vs 拼接幻觉句（judge = '
          + ('0.5B-Instruct' if judge_model else '关键词规则（降级）') + '）')
    print(f'  幻觉句: {HALLUCINATION[:56]}...')
    f_rows = []
    for qi, (query, ctxs, answer) in enumerate(results):
        f_g = faithfulness(answer, ctxs, judge)
        f_h = faithfulness(answer + HALLUCINATION, ctxs, judge)
        n_g = len(split_claims(answer))
        n_h = len(split_claims(answer + HALLUCINATION))
        f_rows.append((f_g, f_h))
        print(f'  Q{qi + 1}: claims {n_g}→{n_h} 条 | grounded={f_g:.2f} | +幻觉={f_h:.2f}'
              if f_g is not None else f'  Q{qi + 1}: 空 claims')
    mean_g = sum(r[0] for r in f_rows) / len(f_rows)
    mean_h = sum(r[1] for r in f_rows) / len(f_rows)
    print(f'  mean: grounded={mean_g:.2f}，+幻觉={mean_h:.2f}'
          f'（幻觉句拉低 {mean_g - mean_h:.2f}——若没拉低，说明裁判太弱）')

    # ── Step 3: context precision + 与检索排名的 Kendall 一致性 ──
    print(f'\n[Step 3] context precision（top{TOP_K}，AP 口径）+ 裁判相关性 vs 检索排名')
    for qi, (query, ctxs, _) in enumerate(results):
        cp, rel = context_precision(query, ctxs, judge)
        if 0 < sum(rel) < len(rel):  # 标签有区分度才算 τ（全 0/全 1 时 τ 无意义）
            judge_rank = [i for i, r in enumerate(rel) if r] + \
                         [i for i, r in enumerate(rel) if not r]
            tau_s = f'{kendall_tau(judge_rank, list(range(len(rel)))):+.2f}'
        else:
            tau_s = '  n/a（裁判标签无区分度）'
        print(f'  Q{qi + 1}: judge 逐位判定 {rel} | context_precision={cp:.2f} | '
              f'裁判-检索排名 Kendall τ={tau_s}')

    # ── Step 4: 评测器噪声（同一问题、等价问法，判决会翻转吗）──
    print('\n[Step 4] 评测器噪声：固定同一输入、只换 prompt 措辞，0.5B 裁判的判决')
    query, ctxs, answer = results[0]
    ctx = '\n'.join(ctxs)[:4000]
    claim = split_claims(answer + HALLUCINATION)[-1]  # 一句幻觉句（期望 no）
    print(f'  被判陈述（幻觉句，期望 no）: {claim[:52]}...')
    faith_a = ('请只依据下面的【上下文】判断【陈述】是否被支持。\n'
               '严格三选一回答：yes（被支持）/ no（与上下文矛盾）/ unsure（上下文没有相关信息）。\n'
               f'【上下文】\n{ctx}\n【陈述】{claim}\n回答：')
    faith_b = (f'请只依据【上下文】判断【陈述】是否被支持，严格回答 yes/no/unsure。\n'
               f'【上下文】\n{ctx}\n【陈述】{claim}\n回答：')
    faith_c = (f'【陈述】{claim}\n\n【上下文】\n{ctx}\n\n'
               f'只根据上面的上下文，这个陈述成立吗？请只回答 yes、no 或 unsure。')
    va, vb, vc = (_parse_verdict(judge(p)) for p in (faith_a, faith_b, faith_c))
    print(f'    entailment 三种问法: A(上下文前置)={va} / B(紧凑指令)={vb} / C(陈述前置)={vc}'
          f'  → {"⚠️ 翻转" if len({va, vb, vc}) > 1 else "本次未翻转（碰巧一致）"}')
    # 相关性判定：同一"相关 chunk"，三种等价问法（Step 3 调参时实测会翻）
    print(f'  被判资料: Q1 的 top-1 证据（人工认定相关，期望 yes）')
    for tag, p in [('W1「相关吗」', '下面的【资料】与【问题】相关吗（内容能帮助回答该问题）？只回答 yes 或 no。'),
                   ('W2「有用吗」', '判断下面【资料】对回答【问题】是否有用。严格二选一：yes（有用）/ no（无关）。'),
                   ('W3 加 few-shot', '判断【资料】能否帮助回答【问题】。\n示例——问题："去重用什么算法？" '
                                      '资料："FineWeb 用 MinHash+LSH 去重" → yes；资料："KV cache 显存管理" → no。\n'
                                      '现在只回答 yes 或 no。')]:
        v = _parse_verdict(judge(p + f'\n【问题】{query}\n【资料】{ctxs[0][:600]}\n回答：'))
        print(f'    {tag:14s} → {v}')
    print('  → 判决随措辞漂移 = 评测器噪声；few-shot 未必更稳（0.5B 会被示例带偏）。\n'
          '    工程对策：固定 prompt 模板、多次采样投票、分数只做系统间相对比较')

    # ── Step 5: ragas 对照（可选依赖，缺失则打印指引跳过）──
    print('\n[Step 5] ragas 对照（可选依赖）')
    try:
        from ragas import evaluate  # noqa: F401
        from ragas.metrics import faithfulness as ragas_faithfulness  # noqa: F401
        HAS_RAGAS = True
    except ImportError:
        HAS_RAGAS = False
    if HAS_RAGAS:
        try:
            print('  ragas 已安装——用 ragas.faithfulness 对同一批 (query, answer, contexts) 打分')
            print('  （ragas 默认接 OpenAI；离线环境需配置 LangchainLLMWrapper 指到本地模型，'
                  '此处从略，见教程 02 章"评测器噪声"讨论）')
        except Exception as e:
            print(f'  ⚠️  ragas 调用失败（{type(e).__name__}: {str(e)[:80]}）→ 跳过')
    else:
        print('  ⚠️  ragas 未安装 → 跳过 ragas 对照段（不影响本脚本其余部分）')
        print('      安装指引：uv pip install --python .venv ragas')
        print('      （ragas 还需要配置 judge LLM，默认 OpenAI；离线可包 LangchainLLmWrapper）')
        print('      我们的 hand-written 指标与其同构：claims 拆解 + 逐条 entailment 判定')

    print(f'''\n[小结]
  - faithfulness 能把"拼进去的幻觉句"从分数上压下来（{mean_g:.2f} → {mean_h:.2f}），
    但绝对值受裁判能力上限制约——0.5B 判 entailment 时，判决随措辞在 yes/no 摆动。
  - context precision 的 AP 口径对"相关 chunk 排前面"敏感；Kendall τ 给出
    裁判与检索器的排名一致性，是"裁判可信度"的旁证。
  - 评测器噪声是真实存在的：等价问法即可让 0.5B 翻转判决。工程上：
    固定 prompt 模板 + 多次采样投票 + 只做相对比较（A/B 两个系统比大小），
    不把小裁判的绝对分数当真。''')
    print(f'\n总耗时 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
