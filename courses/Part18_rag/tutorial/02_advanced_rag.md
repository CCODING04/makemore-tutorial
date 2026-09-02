# 02 — 高级 RAG：上下文增强、结构化检索与"什么时候不该用 RAG"

> 🧭 01 章的五件套把 recall@5 从单路 0.58/0.60 抬到混合+重排 0.92——但检索的
> 天花板卡在一个结构性问题上：**chunk 一旦切出来，就脱离了它的原文语境**。
> 本章先复刻 Anthropic 的 contextual retrieval 与 Jina 的 late chunking（贴
> [scripts/02_contextual_retrieval.py](../scripts/02_contextual_retrieval.py) 本机
> 实测），再鸟瞰 GraphRAG / HippoRAG 2 / RAPTOR 三个结构化思想（认知章，不实现），
> 然后用 [scripts/03_rag_eval.py](../scripts/03_rag_eval.py) 手写 RAGAS 指标——
> 最后回答一个更根本的问题：**什么时候根本不该用 RAG**。

## 学习目标

完成本章后，你将能够：

- ✅ **解释** chunk 失上下文问题的两派解法：contextual retrieval（先切后补）与
  late chunking（先嵌后切）的本质差异
- ✅ **复现** contextual retrieval 四阶梯实验，并对照 Anthropic 官方数字解释
  本机量级差异（含"格式噪声与增益同量级"这一反直觉实测）
- ✅ **画出** naive → advanced → modular → agentic RAG 的演进图谱，说出
  GraphRAG / HippoRAG 2 / RAPTOR 各自解决什么
- ✅ **手写** RAGAS 的 faithfulness 与 context precision，并演示评测器噪声
- ✅ **决策** RAG vs 微调 vs 长上下文三选一（带论文依据与成本量级）

## 前置知识

**必须掌握：**
- [01 章五件套](01_naive_to_hybrid.md)——本章所有实验都跑在同一条管线上

**建议掌握：**
- [Part 8 06 章 PPO/GRPO](../../Part8_post_training/tutorial/README.md)——02 章
  结尾"RAG vs 微调"需要知道微调买的是什么
- [Part 14 推理部署](../../Part14_inference_vllm/tutorial/README.md)——长上下文
  的成本结构（KV cache 随长度线性涨）是"什么时候不该用 RAG"的物理基础

**可选：**
- 图算法基础（/PageRank）——只影响 GraphRAG/HippoRAG 小节的阅读深度

## 一、上下文增强双雄

### 问题引入：chunk 是"断章取义"的最小单位

```
原文：《论文阅读指南》§4 逐 Part 论文实战
      …… | ### Part 13 — Penedo et al. 2024《FineWeb Datasets》 | ……
      （一个 512 字符的 chunk 被切出来后，只剩孤零零的技术名词和半张表格）
```

查询"去重用什么算法"来的时候，这个 chunk 里的"14 band × 8 row"没有任何
"我是讲去重的"信号——**语义靠语境，而语境在切块时被扔掉了**。两派解法：

| | Contextual Retrieval（Anthropic, 2024-09） | Late Chunking（Jina AI, arXiv [2409.04701](https://arxiv.org/abs/2409.04701)） |
|---|---|---|
| 口径 | **先切块、后补上下文**：每个 chunk 前面拼一段 LLM 生成的"全文定位句"再嵌入 | **先嵌后切**：长上下文嵌入模型把**整篇文档**一次编码成 token 向量序列，再按边界切、对每块池化 |
| 成本 | 每 chunk 一次 LLM 调用（Anthropic 用 prompt caching 把 248M chunk 的成本压到 1.02 美元/百万 token） | 零 LLM 调用；但要求嵌入模型支持长上下文 + 输出逐 token 向量 |
| 通用性 | 任意嵌入模型可用 | 绑定长上下文嵌入模型（Jina 自家 jina-embeddings-v2-base-en 等） |
| 一句话 | 给块补上下文 | 让上下文先于切块发生 |

> 💡 类比：contextual retrieval 像给每张从相册撕下来的照片**手写备注**
> "这是 2019 年京都之行第 3 天"；late chunking 像要求看照片的人**先把整本
> 相册翻一遍**再看单张——备注要一张张写（贵但通用），整本翻要记性好
>（便宜但挑模型）。

### 实测：复刻 Anthropic 四阶梯（[脚本 02](../scripts/02_contextual_retrieval.py)）

> 📊 环境标注：与 01 章相同（RTX 4090 / torch 2.6.0+cu124 / transformers 4.57.6）；
> 定位句由 Qwen2.5-0.5B-Instruct 贪心生成（约 64 token，输入=文档大纲 + chunk 前 350 字）；
> 238 chunk 全量，脚本总耗时 50-55s（实测 52-54s，其中 LLM 定位句 ~28s；共享 GPU 上有波动）。

```
[Step 2] 实验一：plain → +LLM 前缀 → +BM25 混合(RRF k=60) → +rerank
         query                                recall@20
  Q1  组内相对策略梯度是哪篇论文提出的？       0.75  0.75  0.75  0.75
  Q2  SGLang 和 vLLM 各自适合什么场景？        0.58  0.42  0.50  0.50
  Q3  推理服务里 KV cache 显存碎片……怎么解    0.67  0.67  0.58  0.58
  --------------------------------------------------------------
  mean                                        0.67  0.61  0.61  0.61
  失败率                                      0.00  0.00  0.00  0.00
        （官方: 5.7% / 3.7% / 2.9% / 1.9%，累计 -67%）

[Step 3] 实验二：同样的信息、不同的前缀，recall 会怎么摆？
  （章节路径 A：`文档名 · 章节 原文`；B：`《文档名》章节：原文`——信息完全相同，仅排版不同）
  Q1                                0.75  1.00  0.75  1.00
  Q2                                0.58  0.50  0.50  0.33
  Q3                                0.67  0.67  0.67  0.67
  --------------------------------------------------------------
  mean                              0.67  0.72  0.64  0.67
```

（四列分别为：plain / +章节路径A / +章节路径B / +章节路径&LLM 句）

**官方 vs 本机，逐条归因**（官方数据见
[Anthropic 工程博客](https://www.anthropic.com/engineering/contextual-retrieval)，
248M chunk 语料）：

1. **上下文生成质量**：官方用 Claude 读整篇文档写定位句；本机 0.5B 只看大纲 +
   前 350 字，定位句偶有跑题（跑脚本看 Step 1 打印的示例即可自查）——噪声前缀
   会把嵌入拉离查询语义。**这不是实现 bug，是复刻条件的天花板**
2. **语料规模**：官方 248M chunk 跨百万文档，"chunk 脱离文档就认不出"的问题
   普遍存在；本机 8 篇文档 238 个 chunk，plain 嵌入本来就不太缺上下文，
   增益空间小
3. **评测粒度**：官方指标是"top-20 一无所获"的失败率（亿级查询平均）；本机
   3 个查询的失败率全为 0——指标已饱和，recall 微差纯属小样本噪声
4. **混合口径**：+BM25 行在"带前缀文本"上算 BM25，前缀引入文档级高频词
   （df 被抬高），稀有词判别力被稀释；官方用加权组合 + 调参绕开

> 🔑 **本章最重要的一张表是实验二**：信息一字不差、只换排版（A vs B），
> mean 就从 0.72 摆到 0.64（±0.08）——**在小语料 + 通用嵌入模型上，"格式噪声"
> 与"技术增益"同量级**。任何 contextual 改造必须配 A/B 评测与多样本查询，
> 单点数字不可信。这正是 Anthropic 要用 248M chunk、按失败率在亿级查询上
> 平均的原因：不是炫富，是被噪声逼的。

> ⚠️ **降级模式实测**（`RAG18_FORCE_FALLBACK=1`）：hashing 向量 + 空定位句下
> 四阶梯 mean 仅 0.11→0.11→0.17→0.17、失败率 33%——连 top-20 都摸不到大部分相关
> chunk。上下文工程救不了烂嵌入。

**工程启示**（按性价比排序）：
1. 先上**确定性结构前缀**（文档名/章节路径/元数据）——零成本、方向对、可复现
2. LLM 定位句是"语料大、生成模型强、有 prompt caching 摊成本"时才划算的选项
3. 上下文工程的收益来自**信息量**，不是"加前缀"这个动作本身

### Late Chunking 的展开（选读）

```
传统:  文档 → 切块 → 逐块嵌入          每块独立编码，语境归零
Late:  文档 → 整篇编码(token 级向量序列) → 按边界切 → 逐块 mean-pool
                                   第 i 块的每个 token 向量都"看过"全文，
                                   池化出来的块向量自带长程语境
```

Jina 的实测（[论文 2409.04701](https://arxiv.org/abs/2409.04701) /
[博客](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)）：
在长文档检索基准上，late chunking 显著优于传统"先切后嵌"，且不需要任何 LLM
调用。代价是嵌入模型必须支持长上下文并暴露 token 级输出——Qwen3-Embedding-0.6B
的 32k 上下文理论上可行（本课程留作扩展思考，未实现）。

## 二、结构化检索思想（认知小节，不实现）

RAG 的演进不是零件替换，是**检索结构的代际跃迁**（综述见 arXiv
[2501.09136](https://arxiv.org/abs/2501.09136) Agentic RAG Survey）：

| 代际 | 代表 | 检索结构 | 解决什么 |
|---|---|---|---|
| Naive RAG | Lewis 2020 | 平面 chunk 列表 | 有没有资料可查 |
| Advanced RAG | 混合/重排/contextual（01 章 + 本章§一） | 平面 + 查询改写 + 上下文增强 | 查得准不准 |
| Modular RAG | RAPTOR / GraphRAG / HippoRAG 2 | **树 / 图** | 多跳问题、全局性问题 |
| Agentic RAG | Search-R1、ReAct 式（→ Part 19） | 检索成为**工具**，模型决定查几轮查什么 | 检索策略本身 |

### RAPTOR：把语料组织成一棵"摘要树"（arXiv [2401.18059](https://arxiv.org/abs/2401.18059)）

```
                ┌────────────┐
                │ 全文档摘要  │  ← 层 3（最抽象：整本文档讲什么）
                └─────┬──────┘
          ┌───────────┴───────────┐
     ┌────┴─────┐           ┌────┴─────┐
     │ 章节摘要  │           │ 章节摘要  │  ← 层 2（聚类 + 摘要，自底向上）
     └────┬─────┘           └────┬─────┘
     ┌────┴──────────────────────┴────┐
     │ chunk  chunk  chunk  chunk …   │  ← 层 1（原始 chunk）
     └────────────────────────────────┘
```

- **痛点**：细节问题要底层 chunk，"这本书的主线论点是什么"要上层概括——
  平面检索只能命中其一
- **做法**：对 chunk 做向量聚类 → 每簇生成摘要 → 摘要再聚类再摘要，
  形成树；检索时可在多层上并行取证据
- 一句话：**把"局部细节"和"全局要义"放进同一棵可检索的树里**

### GraphRAG：先建知识图谱，再检索（arXiv [2404.16130](https://arxiv.org/abs/2404.16130)）

```
原文 --LLM 抽取--> 实体/关系三元组 --聚类--> 社区(community) --逐社区摘要--> 图摘要
查询来了：
  局部查询 → 图邻域扩展（种子实体的多跳邻居）
  全局查询 → 遍历社区摘要（"整个语料对 X 的态度"这类问题）
```

- **痛点**：平面 RAG 对"全局性/多跳"问题天然残废（答案分散在几百个 chunk 里，
  没有单块能命中）
- 代价：建图阶段的 LLM 调用成本高（微软原版对语料做多轮实体抽取）
- 一句话：**把检索从"相似度匹配"升级成"图上的推理"**

### HippoRAG 2：检索即记忆（arXiv [2502.14802](https://arxiv.org/abs/2502.14802)）

```
离线:  chunk → LLM 抽取三元组 → 知识图谱(节点=实体) + Personalized PageRank 备好
在线:  query → 抽实体作为种子 → 图上跑 Personalized PageRank → 激活的节点带回 chunk
```

- **思想**：模仿海马体记忆索引——用图上的随机游走做"联想"，一次查询能带出
  多跳之外的知识；参数化的 LLM 负责读，非参数化的图负责记（"From RAG to Memory"）
- 一句话：**把"非参数化持续学习"做成图上的联想检索**

> 📝 三者共同的底层判断：**当问题不再是一段话能回答的时候，检索结构本身
> 要升级**。树（RAPTOR）管抽象层级，图（GraphRAG/HippoRAG 2）管多跳关联。
> 工程取舍：结构化索引的构建成本（LLM 调用）vs 平面索引的召回上限——
> 私有语料小、问题多跳多时才值得上结构。

## 三、什么时候不该用 RAG

> ⚠️ 这一节在面试里的价值不亚于"会用 RAG"——说得出边界才证明真懂。

**1. LaRA（arXiv [2502.09977](https://arxiv.org/abs/2502.09977)）：没有银弹。**
在多任务、多模型维度上系统对比 RAG 与长上下文 LLM，结论是两者各有胜负域，
不存在全面占优的一方——选型必须回到任务分布。

**2. Self-Route（arXiv [2407.16833](https://arxiv.org/abs/2407.16833)，"Retrieval Augmented Generation or Long-Context LLMs?"）：成本
差数倍，让模型自己路由。**
论文实测：长上下文 LLM 平均效果更好，但 RAG(k=5) 的 token 消耗只约为长上下文直塞的
**17%**（约 1/6——长 prompt 的 KV cache 线性膨胀，→ [Part 14](../../Part14_inference_vllm/tutorial/README.md)）；
提出的 Self-Route 让模型先判断"这题需要全文吗"，只把真正需要长上下文的
查询路由给全文模式——效果接近纯长上下文，token 成本比纯长上下文省 39%~65%
（Self-Route 的 token 占长上下文的 38.6%~61%，仍高于纯 RAG 的 17%）。

**3. Context engineering 共识：装得下就别绕路。**
当上下文预算（现代模型 128k-1M token）轻松装下全部相关知识（<200k token 的
私有文档、一两次会话的记忆），直接把材料塞进 prompt 是更简单、更可靠、
更易调试的方案——RAG 引入的每个组件（分块/嵌入/检索/重排）都是新的误差源
与运维面。RAG 的真正战场是：**知识量超出上下文预算**、**知识高频更新**
（重嵌入比重训便宜亿万倍）、**需要引用出处**（可审计性）。

一张决策表：

| 场景 | 首选 | 理由 |
|---|---|---|
| 知识 < 200k token 且稳定 | 直接塞 prompt | 零检索误差、零运维 |
| 知识大 / 高频更新 / 要引用 | RAG | 重嵌入 ≪ 重训；出处可审计 |
| 要改变模型的"风格/能力/语言" | 微调（→ [Part 8](../../Part8_post_training/tutorial/README.md)、[Part 12](../../Part12_finetune_llamafactory/tutorial/README.md)） | RAG 改不了行为模式 |
| 要事实 + 要风格 | 微调 + RAG 叠加 | 两者正交：参数管能力，检索管事实 |
| 复杂多跳/全局性问题 | 模块化/Agentic RAG | 平面检索召回不了分散证据 |

> 💡 类比：长上下文 = 把整本百科全书搬进考场（贵但全）；RAG = 考场配图书管理员
>（便宜但要赌他找得对）；微调 = 让学生变成领域专家（最贵但改变的是人不是书）。

## 四、RAGAS：让"答案质量"可测量

01 章止步于 recall@k（检索指标）——但用户感知的是**答案**。RAGAS（业界最常用的
RAG 评测框架）用四个 LLM-as-judge 指标补上这条链路：

| 指标 | 定义 | 判什么 |
|---|---|---|
| **Faithfulness**（忠实度） | answer 拆成原子 claims，逐条判"是否被 contexts 支持"；分数 = 支持/总数 | 幻觉（答案有没有编） |
| **Answer Relevancy**（答案相关性） | 从 answer 反向生成问题，与原 query 算相似度 | 跑题（答非所问） |
| **Context Precision**（上下文精确率） | 检索回的 contexts 逐条判"对回答有用吗"，按 AP 口径聚合 | 噪声（检索塞没塞无关材料） |
| **Context Recall**（上下文召回率） | ground truth answer 逐句判"能否在 contexts 里找到依据" | 漏检（该查的查到没有） |

### 实测：手写 faithfulness / context precision（[脚本 03](../scripts/03_rag_eval.py)）

> 📊 环境标注：同前；裁判 = Qwen2.5-0.5B-Instruct 贪心解码（max 8 token）；
> 评测对象 = 01 章同款管线（hybrid+rerank top-5 证据）的生成答案；
> "幻觉版" = 在正确答案后拼两句无中生有的话。总耗时实测 17s。

```
[Step 2] faithfulness：grounded vs 拼接幻觉句
  Q1: claims 6→8 条 | grounded=0.67 | +幻觉=0.50
  Q2: claims 1→3 条 | grounded=0.00 | +幻觉=0.00
  Q3: claims 5→7 条 | grounded=1.00 | +幻觉=0.71
  mean: grounded=0.56，+幻觉=0.40（幻觉句拉低 0.15——若没拉低，说明裁判太弱）

[Step 3] context precision（top-5，AP 口径）+ 裁判相关性 vs 检索排名
  Q1: judge 逐位判定 [1, 1, 1, 1, 0] | context_precision=1.00 | Kendall τ=+1.00
  Q2: judge 逐位判定 [1, 1, 1, 1, 1] | context_precision=1.00 | τ=n/a（标签无区分度）
  Q3: judge 逐位判定 [1, 1, 1, 1, 1] | context_precision=1.00 | τ=n/a

[Step 4] 评测器噪声：固定同一输入、只换 prompt 措辞
  幻觉句（期望 no）三种 entailment 问法: A=no / B=yes / C=yes  → ⚠️ 翻转
  相关 chunk（期望 yes）三种相关性问法: 「相关吗」=yes /「有用吗」=no / few-shot=no
```

**三条读数**：

1. **faithfulness 有判别力但不完美**：幻觉句把分数从 0.56 压到 0.40——方向对、
   幅度被裁判能力封顶。Q2 的 grounded=0.00 是"裁判误杀"（正确短答案
   "手写代码清单叫 TOP8"被判不支持）：**0.5B 裁判的绝对分数不可信**
2. **context precision 的措辞陷阱**：问"有用吗"时 0.5B 几乎一律答 no（全 0 标签），
   换成"相关吗"立刻恢复正常——**裁判 prompt 本身是最大的超参数**
3. **评测器噪声是结构性的**：同一陈述、等价问法，判决在 yes/no 之间翻转；
   few-shot 示例反而把 0.5B 带偏。工程对策：固定 prompt 模板 + 多次采样投票
   + 只做**系统间相对比较**（A/B 谁高谁低可信，绝对值 0.56 vs 0.72 不可信）

> 📝 ragas 本身在本环境未安装——脚本 03 的处理方式就是教程要教的姿势：
> `import` 失败 → 打印 `uv pip install --python .venv ragas` 指引 → 跳过该段，
> rc=0。手写指标与 RAGAS 同构（claims 拆解 + 逐条 entailment），装不装框架
> 都能跑通同一条评测链路。

## 练习与思考

### 概念检验

**Q1：contextual retrieval 和 late chunking 都在解决"chunk 失语境"，为什么
工程界先大规模落地了前者？**

<details>
<summary>💡 答案</summary>

因为**兼容性**。contextual retrieval 只是"改了送进嵌入模型的文本"，下游
（嵌入模型/向量库/检索/重排）零改动，任何存量系统加一层 LLM 前缀生成就能上线；
late chunking 要求嵌入模型本身支持长上下文并暴露 token 级输出——这把技术
选型绑死在特定模型家族上。工程里"在哪一层打补丁"往往比"哪个补丁更优雅"
更决定落地速度。（成本上 contextual retrieval 有了 prompt caching 之后也不再
是障碍——Anthropic 把 248M chunk 的上下文生成成本做到了约 1.02 美元/百万文档
token。）
</details>

**Q2：RAGAS 四个指标里，哪两个可以不用 LLM 裁判？怎么做？**

<details>
<summary>💡 答案</summary>

严格说四个都可以换实现，但最自然"去 LLM 化"的是 **context precision** 和
**context recall**：有标注的相关性标签（01 章的关键词规则 ground truth 就是
一种）时，context precision = 检索列表前 k 位中相关的比例（AP 口径手算），
context recall = ground truth 证据被检索列表覆盖的比例——纯确定性计算。
而 faithfulness / answer relevancy 本质是语义判断（"这句话算不算被支持"），
规则替代的误差大（03 章降级模式的关键词裁判就抓不出幻觉句，实测
grounded=1.00、+幻觉=1.00——降级运行日志留档可复现）。所以生产上常见组合：离线回归用确定性指标，
抽样审计用 LLM 裁判。
</details>

**Q3：老板说"把产品手册全量塞进 2M 上下文的模型，撤掉 RAG 省事"——
你会怎么回应？**

<details>
<summary>💡 畅所欲答版</summary>

分四步算账：① **延迟与成本**：长 prompt 的 KV cache 随长度线性涨，
Self-Route（2407.16833）实测 RAG 的 token 消耗仅约为长上下文的 1/6，且长上下文
每次提问都要付全量 token 钱；② **效果的迷失**：LaRA（2502.09977）显示长上下文并非全面
占优；超长上下文还存在"lost in the middle"现象（中间段信息利用率下降）；
③ **更新频率**：手册改一版就要重发全量 prompt（或重造缓存），RAG 只需重嵌入
改动的 chunk；
④ **可审计性**：产品场景常要"这句话出处是哪页"，RAG 天然带引用。
合理方案是 Self-Route 式路由：简单查询走检索，确需全局比对时才放长上下文。
</details>

### 动手实践

**练习 1：量化你的裁判**

给 [脚本 03](../scripts/03_rag_eval.py) 的 `judge` 写一个"全 yes 裁判"
（`lambda p: 'yes'`）和一个"诚实裁判"（按关键词重合度），对比 faithfulness
分数差异。
验收标准：
- [ ] 全 yes 裁判下幻觉版与 grounded 版分数相同（=裁判无判别力）
- [ ] 能解释为什么"裁判的绝对分数要校准、相对比较才可信"

**练习 2：late chunking 思想最小复现（进阶）**

不用长上下文模型，用"整篇文档嵌入 + chunk 与文档向量的凸组合"近似
"chunk 自带全文语境"，测 recall@20 相比 plain 的变化。
验收标准：
- [ ] 与 02 章实验二的结构化前缀对照（同为"给块加全局信息"的廉价近似）
- [ ] 得出你自己的结论：凸组合权重 α 的敏感性如何

**练习 3：跑一遍 Anthropic 官方博客的数字**

读 [Introducing Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)，
把官方实验设置（语料、指标、各阶段数字）整理成表，与脚本 02 输出逐行对照，
列出每个"不可比因素"。
验收标准：
- [ ] 表格覆盖：语料规模 / 指标口径 / 上下文生成模型 / 融合方式
- [ ] 用自己的话解释"为什么复刻方向正确≠复现幅度"

### 扩展思考

- HippoRAG 2 的 Personalized PageRank 与 Part 13 的 LSH 都在"用随机化换
  可算性"——这个哲学还能不能在 RAG 里找到第三处应用？
- Agentic RAG（→ Part 19）把"查不查、查什么、查几轮"交给模型决策——
  这会把评测从"单次检索质量"变成什么形态？
- RAGAS 的裁判换成 70B 模型，评测器噪声会消失吗？设计实验回答。

## 参考资源

- 📄 Anthropic《Introducing Contextual Retrieval》[工程博客](https://www.anthropic.com/engineering/contextual-retrieval)
- 📄 Late Chunking: Long-Context Embedding Models（arXiv [2409.04701](https://arxiv.org/abs/2409.04701) / [Jina 博客](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)）
- 📄 Agentic RAG 综述（arXiv [2501.09136](https://arxiv.org/abs/2501.09136)）
- 📄 GraphRAG: From Local to Global（arXiv [2404.16130](https://arxiv.org/abs/2404.16130)）· HippoRAG 2: From RAG to Memory（arXiv [2502.14802](https://arxiv.org/abs/2502.14802)）· RAPTOR（arXiv [2401.18059](https://arxiv.org/abs/2401.18059)）
- 📄 LaRA 基准（arXiv [2502.09977](https://arxiv.org/abs/2502.09977)）· Self-Route（arXiv [2407.16833](https://arxiv.org/abs/2407.16833)）
- 📄 MTEB 维护性研究（arXiv [2506.21182](https://arxiv.org/abs/2506.21182)）· [RTEB](https://github.com/NovaSearch-Team/RTEB)
- 🐙 [RAGAS 官方仓库](https://github.com/explodinggradients/ragas)

## 学完本章你能...

- [ ] 说清 contextual retrieval / late chunking / 结构化前缀各自的位置
- [ ] 用"格式噪声与增益同量级"的实测结论解释为什么评测要多样本
- [ ] 画出 naive→advanced→modular→agentic 演进图并给出选型判据
- [ ] 手写 faithfulness / context precision 并校准裁判噪声
- [ ] 面对任何需求先回答"该不该用 RAG"

---

[← 上一章：01 手写五件套](01_naive_to_hybrid.md) | [返回 Part 18 目录](README.md)

> 🚀 下一站 **Part 19（Agent）**：检索从"管线的一个阶段"变成"模型的一个工具"——
> 模型自己决定什么时候查、查什么、查完够不够，Agentic RAG 把本章的检索结构
> 交给策略来学（与 [Part 17 Agentic RL](../../Part17_agentic_rl/tutorial/README.md)
> 的多轮轨迹训练衔接）。
