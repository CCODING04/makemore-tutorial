# 测验 · Part 18（RAG 全链路：手写五件套、上下文增强与何时不该用 RAG）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（数字）** 本机 recall@5 四级消融（dense / BM25 / hybrid(RRF) / +rerank）的均值各是多少？哪个查询暴露 BM25 的盲区、哪个暴露 dense 的盲区？强制降级（hashing trick）时 dense 列变成多少、说明了什么？

- **答案**：均值 0.58 / 0.60 / 0.85 / 0.92（8 篇 md → 238 chunk 语料，RTX 4090 实测）。Q1（语义型："GRPO 出自哪篇论文"）BM25=0.00——被"论文/出自/框架"等中文常用 bigram 淹没；dense 0.75 靠"组内相对策略梯度 ≈ GRPO"的语义近邻赢。Q2（词典型："手写代码清单叫什么"）dense=0.40 抓瞎，BM25=1.00 靠稀有词 TOP8 的 IDF 一击命中。hybrid(0.85) > 两个单路 = RRF 把互补名单叠起来；但 Q2 上 hybrid 反而 1.00→0.80——融合不是免费的，弱路会挤掉强路的好名次；+rerank 在 10 个候选里精排把被挤掉的捞回 top-5（重排只重排不召回）。降级实测：dense 0.58→0.20，这 0.38 的差就是嵌入模型买到的东西（字面碰撞 vs 语义泛化）——降级本身就是一次消融。
- **锚点**：教程 01_naive_to_hybrid.md §实测输出 + §降级路径实测

**Q2（概念）** RRF 为什么只融合名次、不融合分值？写出公式，说明 k 的作用与 k→∞ 的退化行为。

- **答案**：dense 给 cosine ∈ [-1, 1]，BM25 给无界正分——两把尺子的数字不可直接加，加权融合要先做尺度标定且标定错全错。RRF 丢掉分值只看名次：$\mathrm{score}(item)=\sum_{\mathrm{list}} 1/(k+\mathrm{rank})$，rank 从 1 起、k=60（论文默认）。第 1 名得 1/61、第 2 名 1/62……k 把相邻名次的得分差压平，于是"两个榜单都进前 10"轻松赢过"单榜第 1"，且抗单榜名次抖动。k→∞ 时退化为"入选榜单数优先、名次和次之"的计数排序。免标定免调参是工业默认，但不等于最优——网格搜 α 仍能挤出最后几个点。
- **锚点**：教程 01_naive_to_hybrid.md §数学推导 ② RRF：只融合名次，不融合分值

**Q3（数字）** Anthropic 官方 contextual retrieval 的失败率四阶梯是多少？本机复刻（8 篇 238 chunk）为什么失败率全 0、只看到 ±0.08 的"增益摆动"？这个反直觉实测的结论是什么？

- **答案**：官方（248M chunk）：5.7% →（+BM25 混合）3.7% →（+rerank）2.9% → 1.9%，累计 −67%。本机四阶梯 mean 0.67/0.61/0.61/0.61、失败率全 0（3 个查询上指标已饱和）。逐条归因：① 0.5B 只看大纲+前 350 字写定位句，质量远不如 Claude 读全文；② 语料太小，plain 嵌入本来不缺上下文、增益空间小；③ 小样本噪声；④ BM25 在带前缀文本上被文档级高频词抬高 df、稀释稀有词判别力。实验二：信息完全相同、仅换前缀排版（A vs B），mean 就从 0.72 摆到 0.64——**格式噪声与技术增益同量级（±0.08）**。结论：小语料 + 通用嵌入模型上单点数字不可信，任何 contextual 改造必须配 A/B 评测与多样本查询；先上确定性结构前缀（零成本），LLM 定位句要"语料大、模型强、有 prompt caching 摊成本"才划算。
- **锚点**：教程 02_advanced_rag.md §一 实测：复刻 Anthropic 四阶梯 + 官方 vs 本机逐条归因

**Q4（诊断）** 手写 faithfulness 实测：grounded=0.56、拼接幻觉句后 0.40；同一幻觉句换三种等价问法，判决在 yes/no 间翻转；问"有用吗"时相关性标签几乎全 no。诊断这是什么问题、给出工程对策。

- **答案**：评测器噪声（0.5B LLM-as-judge）。三条读数：① faithfulness 有判别力但不完美——幻觉句把分数拉低 0.15（方向对），但 Q2 grounded=0.00 是裁判误杀（正确短答案被判不支持），绝对分数不可信；② 裁判 prompt 本身是最大的超参数——"有用吗"全 no、"相关吗"立刻恢复正常，few-shot 反而把小裁判带偏；③ 噪声是结构性的（等价问法翻转判决）。对策：固定 prompt 模板 + 多次采样投票 + 只做系统间相对比较（A/B 谁高谁低可信，绝对值 0.56 vs 0.72 不可信）；口径上把 unsure 单独报列（supported/refuted/unknown 三数齐报）；离线回归可用确定性指标、抽样审计才用 LLM 裁判。
- **锚点**：教程 02_advanced_rag.md §四 RAGAS 实测 + 三条读数

**Q5（对比）** RAG、微调、长上下文三选一怎么决策？给出决策表要点与论文数字依据。

- **答案**：知识 < 200k token 且稳定 → 直接塞 prompt（context engineering 共识：装得下就别绕路，零检索误差零运维）；知识量大 / 高频更新 / 需要引用出处 → RAG（重嵌入比重训便宜、出处可审计）；要改模型的"风格/能力/语言" → 微调（RAG 改不了行为模式）；既要事实又要风格 → 微调 + RAG 叠加（参数管能力、检索管事实）。数字依据：Self-Route（2407.16833）实测 RAG(k=5) 的 token 消耗只约为长上下文直塞的 17%（≈1/6，长 prompt 的 KV cache 线性膨胀），让模型自己路由"要不要全文"后效果接近纯长上下文、token 省 39%-65%；LaRA（2502.09977）证明两者各有胜负域、没有银弹——选型必须回到任务分布。复杂多跳/全局性问题再上模块化/Agentic RAG。
- **锚点**：教程 02_advanced_rag.md §三 什么时候不该用 RAG + 决策表

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 手写 RAG 五件套并用 recall@k 消融表量化每件套的贡献 | Q1、Q2 | Q1 四级 0.58/0.60/0.85/0.92 与两级盲区；Q2 RRF 公式（BM25/分块公式见闪卡） |
| 复现 Anthropic contextual retrieval 四阶梯并对照官方数字（含格式噪声反直觉实测） | Q3 | 官方 5.7→1.9%；本机失败率饱和、±0.08 格式噪声 |
| 手写 RAGAS 的 faithfulness / context precision 并演示与防御评测器噪声 | Q4 | 0.56→0.40、问法翻转、"有用吗"全 no 与三条对策 |
| 决策 RAG vs 微调 vs 长上下文 | Q5 | 决策表 + Self-Route 17%/39-65% + LaRA 无银弹 |
| 设计模型缺失时的降级路径，让管线在任何环境 rc=0 | Q1 | 降级实测 dense 0.58→0.20 = 嵌入贡献可视化（降级路径机制见闪卡） |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| RAG 解决的三个痛点 | 知识截止（参数知识冻结）、私有数据不可见、幻觉无追溯；微调=背书（风格与能力），RAG=开卷（事实与出处） |
| BM25 公式 | $\mathrm{IDF}=\ln(1+(N-df+0.5)/(df+0.5))$ 恒正平滑；TF 项 $\mathrm{tf}\,(k_1+1)/(\mathrm{tf}+k_1(1-b+b\,|d|/\mathrm{avgdl}))$；k1=1.2、b=0.75 |
| BM25 参数 b | 长度归一强度：b=0 完全不看长度（长文占优）、b=1 完全按长度缩放（短文档浮上来）、|d|=avgdl 时因子为 1；结构化短条目语料调小 b |
| 中文 BM25 分词 | 单字 + 相邻二字 bigram；查询词按出现次数累加 |
| 递归分块三条不变量 | 所有 chunk 长度 ≤ size；除首块外每块以上一块尾部 overlap 字符开头；去重叠段拼接后不丢任何非空白字符（坑：s.split('。') 丢句号，要用捕获组） |
| 配置起点（中文通用） | chunk 512 字符 / overlap 64；BM25 k1=1.2 b=0.75；RRF k=60；候选池 top-10 重排取 top-5；chunk 太碎实测（size=180）：plain recall@20 0.65→0.53 |
| Qwen3-Embedding 官方用法 | 因果（decoder-only）嵌入模型 → last-token pooling（注意左右填充分支）；查询侧拼 Instruct 前缀、文档侧不拼；用 mean-pooling 是"静默劣化"（不报错只掉召回） |
| bi-encoder vs cross-encoder | bi 各自编码 → 可离线、可 ANN、便宜但精度低；cross 成对 (query, doc) 过模型 → 精度高一个量级、贵一个量级、无法离线；定式：bi/BM25 从 10⁶ 捞 top-10/20，cross 精排 |
| 降级路径设计 | 嵌入→hashing trick（md5 特征哈希、确定性零依赖）；重排→跳过；生成→抽取式；RAG18_FORCE_FALLBACK=1 实测 dense 0.58→0.20；教程脚本铁律：降级不崩、rc=0 |
| contextual retrieval vs late chunking | 先切后补（每 chunk 拼 LLM 定位句再嵌入；任意嵌入模型可用、每 chunk 一次 LLM 调用）vs 先嵌后切（整篇编码成 token 向量序列再按边界切、逐块池化；零 LLM 调用但绑定长上下文模型）；前者因兼容性先大规模落地 |
| Anthropic 成本数字 | prompt caching 把 248M chunk 的上下文生成成本压到约 1.02 美元/百万 token |
| RAG 四代演进 | Naive（平面 chunk）→ Advanced（混合/重排/查询改写/上下文增强）→ Modular（RAPTOR 摘要树 / GraphRAG 图 / HippoRAG 2 海马体记忆）→ Agentic（检索成为工具，模型决定查几轮查什么 → Part 19） |
| RAGAS 四指标 | faithfulness（claims 逐条判支持，抓幻觉）、answer relevancy（反向生成问题，抓跑题）、context precision（逐条判有用，AP 口径，抓噪声）、context recall（ground truth 逐句找依据，抓漏检） |
| 手写 faithfulness 口径 | 拆 claims → 逐条带上下文问 judge → yes 计 1，no/unsure 计 0（保守口径）；无有效 claims 返回 None；judge 可注入便于 mock 测试 |
| 评测器噪声对策 | 固定 prompt 模板 + 多次采样投票 + 只做系统间相对比较；"全 yes 裁判"下幻觉版与 grounded 版同分 = 无判别力 |
| 暴力 cosine vs ANN | 238 chunk 一次矩阵乘 <0.01s；N 小于几万时暴力最快且无损，不要提前上 FAISS；百万级才换 IndexFlatIP/HNSW |
