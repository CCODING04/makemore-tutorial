# LLM 算法岗面试备战指南（JD 对齐版）

> **版本**：v2（2026-08-30 晚修订，见文末修订记录）。依据：Boss 直聘在招 JD 聚类（大模型算法工程师 / LLM+RAG / 多智能体 /
> 预训练方向）+ 牛客 2025-26 面经热点（阿里系整理帖、RLHF 八股、推理优化专题）+ 本课程实测数据。
> **用法**：先看 §2 映射表定位自己的缺口 → 按 §3 讲故事链复习 → 用 §6 的项目故事包装简历 →
> 按 §8 路线补齐 ✗ 项。
>
> 主要来源：[Boss直聘-大模型算法工程师](https://www.zhipin.com/job_detail/f25df87216b4806b1HV939m-E1dQ.html)、
> [Boss直聘-LLM+RAG方向](https://www.zhipin.com/job_detail/b43c8104d8d512130nB_29W5GFNW.html)、
> [牛客-2025-26 阿里系面试题整理](https://www.nowcoder.com/discuss/848942791164981248)、
> [牛客-RLHF 八股总结](https://www.nowcoder.com/feed/main/detail/20e8f456d0c5418cad2b46b39c0d0f61)、
> [美团-大模型应用算法实习](https://zhaopin.meituan.com/web/position/detail?jobUnionId=4215619700&highlightType=campus)、
> [字节-大模型算法工程师](https://jobs.bytedance.com/experienced/position/7366887909082859803/detail)

---

## 1. 岗位需求扫描结论（2025-26 JD 聚类）

对 Boss 直聘/美团/字节/滴滴/宇树等 JD 做聚类，技术要求落在 **8 个簇**：

| # | 需求簇 | JD 原文高频词 | 出现频率 |
|---|---|---|:---:|
| A | **Transformer 与现代 LLM 架构** | Transformer、attention 变体、Decoder-only、LLaMA 结构 | ★★★★★ |
| B | **SFT / 指令微调** | SFT、Instruction Tuning、LoRA/PEFT、数据构造 | ★★★★★ |
| C | **对齐 / 强化学习** | RLHF、DPO、PPO、**GRPO**、奖励模型 | ★★★★★ |
| D | **RAG 与知识增强** | RAG、向量检索、文档处理、知识库问答 | ★★★★ |
| E | **Agent / 多智能体** | Agent、function calling、多智能体协作（阿里专门设岗） | ★★★★ |
| F | **推理优化与部署** | vLLM、量化、KV Cache、推理加速（2025 面经"明星话题"） | ★★★★ |
| G | **预训练与数据工程** | 预训练、数据清洗/去重/配比、分布式训练、scaling law | ★★★ |
| H | **评估与落地** | 效果评估、benchmark、A/B、幻觉/安全 | ★★★ |

面经热点补充（牛客 2025-26 归纳）：Transformer 考察深度创新高；**推理优化与 GRPO 是新增热点**；
"为什么 Decoder-only 成为主流"、"GPT vs BERT"、"RLHF 效果评估维度（生成质量/安全性/对齐性）"
是高频题。

### ⚠️ 赛道澄清（2026-08 修订：v1 版此处有分类错误）

簇 D（RAG）/E（Agent）**主要属于"大模型应用工程师/Agent 工程师"赛道，不属于本指南默认的
"大模型算法工程师"赛道**。v1 把它们列为算法岗 ★★★★ 需求是检索词混入了应用开发岗所致。
证据与边界：

- 知乎《大模型应用算法岗和开发岗的界线在哪里？》：**"2024 年 RAG 还是算法工程师在做，
  现在基本都由开发岗承担"**，趋势是归入 Agent 工程师/AI 应用工程师；
- 真·算法岗 JD（腾讯混元后训练 RM 方向、上海 AI Lab 训练算法、高校算法岗）的职责是
  **预训练 / SFT / RLHF / DPO / RM / 数据清洗**——正是本课程主线，几乎不出现 RAG/Agent；
- 交叉地带是"大模型应用算法工程师"（如美团生成式搜索岗）：需要懂 RAG/Agent 的**算法侧**
  （召回排序、评测、领域模型调优），但模型训练仍是主线。

因此本指南 §2 映射表中簇 D/E 的状态对**算法岗求职**参考意义有限；若目标是应用工程师赛道，
请配合 §7 的赛道表与 §8 的"应用线"课程。

---

## 2. 需求 × 课程映射总表

> ✅ 已覆盖（可直接答题）/ 🟡 部分覆盖（概念有、缺实战或深度）/ ❌ 未覆盖（见 §8 补课路线）

### 簇 A：架构（★★★★★）

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| attention/除以√d/因果 mask | Part 6 02 章 | ✅ |
| Decoder-only vs Encoder-Decoder | Part 6 04 章 | ✅ |
| LayerNorm/RMSNorm、pre-norm | Part 3 + Part 7 02 章 | ✅ |
| 位置编码：learned PE → RoPE（相对位置性质、外推） | Part 7 02 章 | ✅ |
| GQA/MQA、KV 头数与显存 | Part 7 03 章 + Part 8 06 章 | ✅ |
| SwiGLU / MoE / 负载均衡 | Part 7 03 章 + 脚本 10 | ✅ |
| 长上下文（PI/NTK/YaRN） | Part 7 05 章进阶小节 | 🟡（有公式与路线，无实验） |
| 多模态 VLM | **Part 15（多模态理解）** | 🟡（拼接式手写+三大方案+CLIP/SigLIP；Q-Former/评估/幻觉见 Part 15 章"进阶与缺口"） |

### 簇 B：SFT / 微调（★★★★★）

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| 预训练→SFT 全流程（真实数据复现） | Part 7 04/05 章（minimind 毕业指南） | ✅ |
| Chat template / prompt masking | Part 8 02 章 | ✅ |
| GPT-2 生命周期的完整实现 | Part 8 01-02 章 | ✅ |
| **LoRA / PEFT 参数高效微调** | — | ❌ |
| 微调框架实战（LLaMA-Factory 等） | — | ❌ |

### 簇 C：对齐 / RL（★★★★★）

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| Bradley-Terry / 奖励模型 | Part 8 03 章 | ✅ |
| DPO（完整推导）/ ORPO / KTO | Part 8 03 章 | ✅ |
| PPO（GAE + clip）/ **GRPO / RLVR** | Part 8 04 章 | ✅ |
| RLHF 效果评估（质量/安全/对齐） | Part 8 07 章 | ✅ |

### 簇 D+E：RAG / Agent（★★★★）——⚠️ 属应用工程师赛道，非算法岗核心需求

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| RAG 全链路（分块/embedding/检索/重排/生成） | — | ❌ |
| 向量库与检索（ANN、BM25 混合） | — | ❌ |
| Agent / function calling / 多智能体 | — | ❌ |
| Prompt Engineering（系统化） | 散落在各章生成参数处 | 🟡 |

### 簇 F：推理优化与部署（★★★★）

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| decode 是 memory-bound（roofline 直觉） | Part 9 02 章 | ✅ |
| KV Cache 显存公式 / GQA 量化省多少 | Part 8 06 章 + 脚本 09 | ✅ |
| 量化 int8/int4（RTN）/ GPTQ / AWQ 思想 | Part 8 06 章 + 脚本 09 | ✅（概念+toy 实测）|
| PagedAttention / 连续批处理 | Part 8 06 章 + 脚本 09 | ✅（模拟实测） |
| 投机解码（判据+期望公式+实测 α） | Part 8 06 章 + 脚本 09 | ✅ |
| vLLM / SGLang / TensorRT-LLM 生产实战 | Part 8 06 章（10 行最小实操） | 🟡 |
| CUDA 内核与优化方法论 | Part 9 全部 | ✅ |
| Flash Attention 原理（tiling/online softmax） | Part 7/9（原理）+教程 | 🟡（未写内核） |

### 簇 G：预训练与数据工程（★★★）

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| 训练全流程（优化器/调度/混合精度/checkpoint） | Part 7/8 | ✅ |
| **分布式 DDP / ZeRO / FSDP / TP / PP** | **Part 10 全部** | ✅（双卡实测） |
| 预训练数据工程（清洗/去重/配比） | roadmap T8 归档 | ❌ |
| scaling law | — | ❌ |
| Tokenizer（BPE 训练与选型） | Part 7 01 章 + 05 章对照 | ✅ |

### 簇 H：评估与落地（★★★）

| 需求点 | 课程位置 | 状态 |
|---|---|:---:|
| 评估方法论（规则/人工/LLM-judge、污染） | Part 8 07 章 | ✅ |
| GSM8K 风格评估、lm-eval-harness 语义 | Part 8 05/07 章 | 🟡（harness 未实操） |
| 幻觉/安全对齐专题 | — | ❌ |

**总评**：簇 A/B/C/F/G 的核心问答与代码能力已全部覆盖且带实测；系统性缺口按严重度排序为
**RAG（D）> Agent（E）> LoRA 实战（B）> 数据工程（G）> 多模态 > 幻觉安全**。

---

## 3. 讲故事链复习法（面试官说"串一下"时用）

每条链 = 一串追问 = 一段你可以讲 10 分钟的故事。复习时对着链上每一环自问"为什么"。

1. **训练链**（Part 1→3→7→8）：初始 loss=ln(vocab) 说明什么 → loss 不降怎么诊断（Part 3 四件
   体检工具）→ 为什么 RMSNorm 替代 LayerNorm、放哪 → AdamW+cosine+warmup 各管什么 →
   梯度累积/裁剪解决什么 → 混合精度为什么省一半显存。
2. **架构链**（Part 6→7）：除以 √d → pre-norm vs post-norm → RoPE 相对位置性质怎么证 →
   GQA 省的是什么（KV 头数 × KV cache 显存，配公式）→ SwiGLU/MoE 取舍 → MoE 为什么会
   塌缩、aux loss 公式每一项是什么。
3. **对齐链**（Part 8 03-04）：SFT 为什么要 prompt masking → Bradley-Terry 是什么 → DPO
   推导哪一步消掉奖励模型 → clip 和 GAE 各防什么 → GRPO 为什么能去掉 Value Network →
   RLVR 什么时候可用（配 minimind DPO lr=4e-8 的实例）。
4. **推理系统链**（Part 9→Part 8 06）：decode 为什么 memory-bound → KV cache 省的是访存不是
   计算 → Flash Attention 快在哪（tiling+fused，少一次 HBM 往返）→ bf16 为什么省一半带宽 →
   PagedAttention 解决 60-80%→<4% → 投机解码判据与期望公式 → 给你一个 kernel 怎么判断优化
   方向（算术强度 + ncu SOL）。
5. **分布式链**（Part 10，新增）：多卡为什么不是插上就行 → DDP all-reduce 是平均 → 16Ψ 账本 →
   ZeRO 三阶段各切什么、通信代价 → TP 的 f/g 算子 → PP bubble 公式 → 给 32 台 8 卡机训 70B
   的配置推理链。
6. **应用链**（❌ 课程缺口，需自补）：RAG 全链路 → chunk 策略与召回率 → 混合检索与重排 →
   幻觉抑制 → Agent 的 function calling 循环 → 多智能体分工。**面试前若没有项目支撑，
   建议按 §8 应用线 A1/A2 快速做一个。**

## 4. 答题框架与硬数字清单

**答题框架：直觉 → 公式 → 数字。** 例：RMSNorm——"LayerNorm 砍掉均值中心化和 bias"
（直觉）→ `x/√(mean(x²)+ε)·γ`（公式）→"省一次减均值和一次求均值，26M 模型上权重绑定另省
6400×512≈12% 参数"（数字）。只背公式或只讲直觉都拿不到满分。

**本课实测硬数字清单**（面试直接引用，均可在对应脚本复现）：

| 数字 | 出处 |
|---|---|
| 初始 loss ≈ ln(vocab)；char 级 Shakespeare 训好后 ppl≈9-11 | Part 6 / 脚本 09 |
| 4090 fp32 峰值 ~82 TFLOPS；cuBLAS 实测 ~22 TFLOPS；手写阶梯 553→8795 GFLOPS | Part 9 脚本 04/06 |
| 合并访存做错 vs 做对差 ~9 倍；树形归约 vs 朴素 atomic 差 ~77 倍 | Part 9 脚本 04/05 |
| warp=32；block 最大 1024 线程；SMEM 48KB/block（可配置更大） | Part 9 01 章 |
| PagedAttention：碎片浪费 41%（简化模拟）→ 论文语境 60-80% → <4% | Part 8 脚本 09 |
| 投机解码 α≈0.65 → 实测 2.81 tokens/cycle vs 理论 2.53 | Part 8 脚本 09 |
| KV 显存公式：LLaMA-7B fp16 seq2048 = 1.07GB；GQA(kv=8) → 0.27GB | Part 8 脚本 09 |
| MoE：α=0 时 gini 0.73、5/8 专家死亡；α=0.01 拉平但任务 loss 翻倍；minimind 用 5e-4 | Part 7 脚本 10 |
| ZeRO：16Ψ → 4Ψ+12Ψ/N（zero1）；zero1 vs zero2 谁省取决于 N | Part 10 脚本 03 |
| TP 前向误差 6e-07、PP 流水线 loss 与单进程完全一致（4.380254） | Part 10 脚本 05/06 |
| minimind 全流程：3090 单卡 ≈2.3h/¥3；DPO lr=4e-8、β=0.15；26M=hidden512/kv头2 | Part 7 05 章 |

## 5. 手写代码 TOP8（写到能默写）

| # | 题目 | 课程出处 |
|---|---|---|
| 1 | 多头因果自注意力（causal mask 位置） | Part 6 / Part 8 01 |
| 2 | GQA 的 repeat_kv | Part 7 03 章 |
| 3 | prompt-masked CE（ignore_index=-100） | Part 8 02 章 |
| 4 | DPO loss（-logsigmoid(β·(Δπ−Δref))） | Part 8 03 章 |
| 5 | GRPO 组内标准化优势 | Part 8 04 章 |
| 6 | RoPE 的 precompute_freqs_cis + apply | Part 7 02 章 |
| 7 | absmax int8 逐通道量化 + 反量化 | Part 8 脚本 09 |
| 8 | speculative decoding 接受判据（u < pt/pd） | Part 8 脚本 09 |

## 6. 用本课程包装简历（三个项目故事模板）

1. **"从零复现 minimind"**：自实现 BPE/RMSNorm/RoPE/GQA/SwiGLU + Pretrain→SFT→DPO 三阶段；
   验收用 lm-eval-harness 语义的自评脚本对比三阶段行为。对应 Part 7 全部 + 05 章毕业指南。
2. **"手写推理优化实验"**：int8/int4 量化 ppl 对比、PagedAttention 碎片模拟（41%→5%）、
   投机解码 α 实测对齐理论公式。对应 Part 8 脚本 09 + 06 章。
3. **"双卡分布式训练"**：DDP/ZeRO 记账/FSDP/手写 TP 与 GPipe，流水线 loss 与单进程完全一致
   的等价性验证。对应 Part 10。

> ⭐ 面试叙事公式：**"我跑过，现象是 X，原因是 Y"**（引用上面实测数字）比"论文说"高一个档位。

## 7. 按岗位方向的备战侧重（2026-08 修订：区分两条赛道）

**先选赛道，再选补课**——两类岗位的 JD 差异比想象中大：

| | 算法赛道（本课程主线） | 应用赛道（姊妹系列） |
|---|---|---|
| 典型岗位名 | 大模型算法工程师 / 后训练算法 / 预训练算法 | 大模型应用工程师 / Agent 工程师 / AI 应用开发 |
| 核心职责 | 预训练、SFT/RLHF/DPO、RM、数据清洗、评估 | RAG、Agent、选型评估、Prompt、部署调优 |
| 技术栈 | 训练框架、分布式、CUDA | LangGraph/厂商 SDK、向量库、API 编排 |
| 课程覆盖 | Part 1-10 基本全覆盖 | 仅概念层；应用工程需应用线课程 |

| 算法赛道内方向 | 重点章节 | 优先补齐（§8 主线） |
|---|---|---|
| 预训练岗 | Part 7/10 全部 + 9 | 数据工程（Part 13）、scaling law |
| 对齐/后训练岗 | Part 8 全部 + 07 章 | **RL 实战（verl/TRL，主线 Part 11，最急）**、LoRA 实战（Part 12） |
| 推理 infra 岗 | Part 9 + 8 06 章 + 10 | vLLM/SGLang 源码级（主线 Part 14） |

| 应用赛道方向（若选它） | 依赖 | 补齐（§8 应用线） |
|---|---|---|
| RAG 工程师 | 本课程概念层即可 | 应用线 A1：手写最小 RAG → 框架认知 → ragflow 平台 |
| Agent 工程师 | 同上 + 06 章生成参数 | 应用线 A2：agent loop 手写 → smolagents/pi/OpenAI Agents SDK → LangGraph 认知 |

## 8. 差距 → 优质仓库 → 拟开课程（star 数 2026-08 实查）

### 算法主线（承接 Part 1-10，按缺口优先级）

| 拟开 | 主题 | 核心仓库（stars，2026-08 实查） | 备注 |
|---|---|---|---|
| **Part 11（最急）** | 对齐实战：RLHF/GRPO 训真模型 | verl（23.2k，字节，GRPO 事实标准） | OpenRLHF（10.0k）、trl（19.2k）——补簇 C 的"实战"缺口 |
| **Part 12** | 微调实战（LoRA/QLoRA/全参） | LLaMA-Factory（74.4k） | unsloth（75.2k）；补簇 B 的 LoRA ❌ |
| **Part 13** | 预训练数据工程 | Data-Juicer（7.0k，阿里） | FineWeb 博客、MinHash 去重自实现；补簇 G ❌ |
| **Part 14** | 推理部署实战 | vLLM（90.5k） | llama.cpp（126.3k 端侧）、SGLang；把 🟡 变 ✅ |
| Part 15/16（已建） | 多模态理解 + 图像/视频生成 | 课程内（nanoVLM/minimind-v；diffusers/CogVideoX/Wan2.1） | 🟡 手写机制与概念已覆盖；Q-Former/评估/幻觉/扩散训练侧见各章"进阶与缺口" |
| 面试八股参考（不入课程） | 题库 | wdndev/llm_interview_note（15.0k） | mlabonne/llm-course（82.1k） |
| 源码精读（毕业读物） | 训练内核 | llm.c（30.9k，Karpathy） | rasbt/LLMs-from-scratch（104.0k） |

### 应用线（姊妹系列，"大模型应用工程师"赛道；与算法主线相互独立）

> 定位修正（2026-08）：v1 曾把 "RAG 与 Agent" 列为算法主线 Part 11 且默认推荐 LangChain——
> 两点都不成立。RAG/Agent 属应用赛道（§1 赛道澄清）；框架选型上 LangChain 因过度封装
> 在生产界口碑分化（有"45% 团队在用、仅 12% 留在生产"的说法；$125M 融资主要靠 LangSmith
> 而非框架本身），官方教程偏生态营销。教学载体改为"**先手写、后 harness**"：

| 拟开 | 主题 | 载体与顺序 |
|---|---|---|
| **应用线 A1** | 最小 RAG：手写分块/embedding/检索/重排/生成（~200 行，呼应本课程"从零"哲学） | 先裸写 → 再对比 ragflow（89.6k）平台化能力与 llama_index（51.9k）抽象 |
| **应用线 A2** | Agent：手写 agent loop（模型 + 工具表 + while 循环） | ① 手写 loop → ② 读 **pi**（earendil-works/pi，Mario Zechner/badlogic 的极简 harness，4 个工具 + <1000 token 系统提示词，驱动 OpenClaw，"工程减法"哲学与本课程同源）→ ③ **smolagents**（HF，~1000 行，CodeAgent 代码即动作）→ ④ **OpenAI Agents SDK**（Swarm 升级，Agent/Handoff/Guardrail 三原语）与 **Google ADK 2.0**（GCP 生态）按厂商栈选学 → ⑤ **LangGraph** 只学其图/状态机/checkpoint 概念（生产中真正被保留的部分），LangChain 本体仅作生态认知 |
| 应用线 A3（可选） | 评测与可观测 | Langfuse 等观测栈、A/B 与线上评估 |

## 9. 面试资源索引

> 📋 逐 Part 的参考仓库/教程清单（含 star/最后推送时间实查与四维评级：可靠性/详实/实践/时间性）
> 见 [references_by_part.md](references_by_part.md)，供审查与更新。

- 牛客：[2025-26 阿里系算法面试题](https://www.nowcoder.com/discuss/848942791164981248)、
  [大模型面试专题](https://www.nowcoder.com/creation/subject/278ae3e75eca413fa6f96d70cd03ae57)、
  [RLHF 八股](https://www.nowcoder.com/feed/main/detail/20e8f456d0c5418cad2b46b39c0d0f61)、
  [备战路径经验帖](https://www.nowcoder.com/feed/main/detail/63aededb536044d783161a109f70c2fd)
- Datawhale：[LLM & VLM & Agent 面试题总结](https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra01-%25E9%259D%25A2%25E8%25AF%2595%25E9%2597%25AE%25E9%25A2%2598%25E6%2580%25BB%25E7%25BB%2593.md)
- CSDN：[快手大模型面经精选](https://blog.csdn.net/ZHHHHH15/article/details/160495637)、
  知乎：[大模型算法岗面试题含答案](https://zhuanlan.zhihu.com/p/1921990540699869821)

> ⚠️ 时效声明：JD 聚类与 star 数核对于 2026-08-30；上游仓库与 JD 风向变化快，使用前建议复查。

## 修订记录

- **v2（2026-08-30 晚）**：应审阅人质询修订三处——① 簇 D/E（RAG/Agent）重新归类为**应用工程师
  赛道**，不再作为算法岗核心需求（v1 的 ★★★★ 是检索混入应用岗 JD 所致，引知乎"2024 年 RAG
  还是算法岗在做，现在基本归开发岗"）；② 主线 Part 11 从"RAG 与 Agent"改为**对齐实战
  （verl/GRPO）**，RAG/Agent 移入独立**应用线 A1/A2**；③ 应用线技术选型弃用 LangChain 作为
  默认载体（生产留存争议 + 过度封装批评），改为"手写 loop → pi/smolagents/OpenAI Agents
  SDK/ADK → LangGraph 概念认知"的顺序，Pi（badlogic 的 pi-mono）定位为极简 harness 参照。
- **v1（2026-08-30）**：初版（8 簇 JD 聚类 + 课程映射 + Part 11-16 建议）。
