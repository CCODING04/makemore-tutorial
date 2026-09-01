# Part 18: RAG 全链路 — 手写五件套、上下文增强与"何时不该用 RAG"

> 🧭 应用线 A1 的第一站。算法主线（Part 1-17）教会模型"会算"，应用线教
> "会用"：**RAG（检索增强生成）是 LLM 应用工程师赛道的第一块硬通货**——
> 面试指南簇 D+E（RAG/Agent，★★★★）的核心。本部分不调 LangChain、不配
> 向量数据库，在本仓库 docs/ 的 8 篇真实 Markdown 上**从零手写**工业 RAG
> 管线的每一层：递归分块 → 稠密嵌入（Qwen3-Embedding，官方 last-token
> pooling）→ 手写 BM25 → RRF 混合 → cross-encoder 重排 → 0.5B 生成带引用回答
> → 手写 RAGAS 评测。所有模型缺失时自动降级（hashing trick / 抽取式 /
> 关键词裁判），**脚本永不崩**。
> 与算法主线的互文：检索的"近似哲学"接 [Part 13 数据工程](../../Part13_data_engineering/tutorial/README.md)，
> 生成的服务化接 [Part 14 推理部署](../../Part14_inference_vllm/tutorial/README.md)，
> "检索变成工具"预告 Part 19（Agent）。

## 学习目标

完成本部分后，你将能够：

- ✅ **手写** RAG 五件套并用 recall@k 消融表量化每件套的贡献（本机实测
  单路 0.65/0.60 → 混合 0.85 → +重排 0.92）
- ✅ **复现** Anthropic contextual retrieval 四阶梯实验，并对照官方数字
  （失败率 5.7%→1.9%）解释本机量级差异——包括"格式噪声与增益同量级"这一反
  直觉实测
- ✅ **手写** RAGAS 的 faithfulness / context precision（0.5B 当裁判），
  并演示与防御"评测器噪声"
- ✅ **决策** RAG vs 微调 vs 长上下文（LaRA 无银弹 / Self-Route 23× 成本 /
  装得下就直塞的 context engineering 共识）
- ✅ **设计** 模型缺失时的降级路径，让整条管线在任何环境 rc=0

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [从朴素 RAG 到混合检索：手写五件套](01_naive_to_hybrid.md) | 递归分块/BM25 推导/RRF/cross-encoder/last-token pooling/降级路径；recall@5 四级消融实测 | [`01_minimal_rag.py`](../scripts/01_minimal_rag.py) |
| 02 | [高级 RAG：上下文增强、结构化检索与"何时不该用 RAG"](02_advanced_rag.md) | contextual retrieval 复刻 + late chunking；GraphRAG/HippoRAG 2/RAPTOR（认知）；RAGAS 四指标手写与评测器噪声 | [`02_contextual_retrieval.py`](../scripts/02_contextual_retrieval.py) · [`03_rag_eval.py`](../scripts/03_rag_eval.py) |

## 🧰 前置知识

- **必须掌握**：
  - [Part 6 Transformer](../../Part6_transformer/tutorial/README.md)——嵌入模型与
    cross-encoder 都是 Transformer 编码器；cosine = 归一化点积
  - [Part 8 SFT](../../Part8_post_training/tutorial/README.md)——instruct 模型与
    chat template（生成/裁判环节全靠它）
- **建议掌握**：
  - [Part 13 数据工程](../../Part13_data_engineering/tutorial/01_dedup_from_scratch.md)——
    "精确算不动就设计可算的近似"的检索哲学（LSH ↔ BM25 互文）；语料就来自
    本仓库经过 Part 13 思想清洗的 docs/
- **可选**：
  - [Part 14 推理部署](../../Part14_inference_vllm/tutorial/README.md)——生产 RAG
    的生成侧要架在 vLLM/SGLang 上；长上下文的 KV cache 成本结构是
    02 章"何时不该用 RAG"的物理基础

## 🔗 在 LLM 链路中的位置

```
Part 13（数据工程：语料从哪来）─┐
Part 6/8（模型：会读会写）────┼→ 【本部分: 给模型外挂一个可检索的记忆】
Part 14（推理部署：跑得快）───┘        ↓
                              Part 19（Agent: 检索变成模型手里的工具，待开）
```

RAG 是应用线的地基：Agent 的"查资料"动作、长上下文应用的"知识底座"、
面试的"系统设计题"（面试指南跨方向高频考点 TOP15 第 13 条），全都从这条
管线讲起。

## 📦 环境

```bash
# 模型（首次运行自动从 HF 缓存读取；缺失时自动降级并打印下载指引）
#   Qwen/Qwen3-Embedding-0.6B   检索嵌入（fp32）
#   Qwen/Qwen2.5-0.5B-Instruct  生成/上下文生成/裁判（fp16）
#   BAAI/bge-reranker-v2-m3     cross-encoder 重排（fp32）
cd courses/Part18_rag/scripts
CUDA_VISIBLE_DEVICES=0 python 01_minimal_rag.py        # ~20s（RTX 4090，共享 GPU 有波动）
CUDA_VISIBLE_DEVICES=0 python 02_contextual_retrieval.py  # ~60-85s
CUDA_VISIBLE_DEVICES=0 python 03_rag_eval.py           # ~20s

# 体验降级路径（零模型、纯 CPU，约 3s，验证"永不崩"设计）
RAG18_FORCE_FALLBACK=1 python 01_minimal_rag.py
```

- GPU 与其他任务共享时先 `nvidia-smi` 挑空卡；语料仅 8 篇 md，纯 CPU 也可接受
  （0.5B 生成约慢 10 倍，嵌入/重排更慢，但全部有降级路径兜底）
- ragas 为**可选依赖**（未装时脚本 03 打印 `uv pip install --python .venv ragas`
  指引后跳过该段，rc=0）

## 📈 学习地图

```
五件套（分块→嵌入→BM25→RRF→重排→生成）      ← 点：每一件都能单独消融
   ↓ recall@5 消融：0.65/0.60 → 0.85 → 0.92（01 章实测）
上下文增强（contextual retrieval / late chunking / 结构化前缀） ← 线：chunk 失语境问题
   ↓ 格式噪声与增益同量级（02 章实验二实测 ±0.19）
评测（RAGAS 四指标手写 + 评测器噪声）        ← 线：答案质量 ≠ 检索质量
   ↓
边界决策（RAG vs 微调 vs 长上下文）          ← 面：什么时候根本不该用 RAG
   ↓ Part 19：检索变成工具，Agentic RAG
```

## 📝 课后作业

👉 [Assignment 18](../../../assignments/assignment_18/)——手写
`recursive_chunk` / `bm25_scores` / `rrf_fuse` / `faithfulness` 四件核心
（与课程脚本同名同签名），🌟 题 5 网格搜 hybrid 权重画 recall-α 曲线。

## 🔗 相关资源

- 📄 Lewis et al. 2020《RAG for Knowledge-Intensive NLP Tasks》（arXiv [2005.11401](https://arxiv.org/abs/2005.11401)）
- 📄 Anthropic《Introducing Contextual Retrieval》([engineering blog](https://www.anthropic.com/engineering/contextual-retrieval)) · Late Chunking（arXiv [2409.04701](https://arxiv.org/abs/2409.04701)）
- 📄 Agentic RAG 综述（[2501.09136](https://arxiv.org/abs/2501.09136)）· GraphRAG（[2404.16130](https://arxiv.org/abs/2404.16130)）· HippoRAG 2（[2502.14802](https://arxiv.org/abs/2502.14802)）· RAPTOR（[2401.18059](https://arxiv.org/abs/2401.18059)）
- 📄 LaRA（[2502.09977](https://arxiv.org/abs/2502.09977)）· Self-Route（[2310.03052](https://arxiv.org/abs/2310.03052)）· MTEB 维护性研究（[2506.21182](https://arxiv.org/abs/2506.21182)）· [RTEB](https://github.com/NovaSearch-Team/RTEB)
- 🐙 [RAGAS](https://github.com/explodinggradients/ragas) · [Qwen3-Embedding 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

---

[← 上一部分：Part 17 Agentic RL](../../Part17_agentic_rl/tutorial/README.md) | [返回课程总览](../../../README.md)

🚀 **下一站 Part 19（Agent，待开）**：本部分的检索管线将成为模型的**工具**——
什么时候查、查什么、查完够不够，都交给模型决策（Agentic RAG），训练方法接
[Part 17 的多轮轨迹 RL](../../Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md)。
