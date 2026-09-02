# 逐 Part 参考仓库与教程清单（供审查）

> **生成**：2026-08-30。**star / 最后推送时间均为 GitHub API 当日实查**（非估计）。
> 评级维度：可靠性（维护方/权威性）、详实性（内容深度）、实践性（可运行）、时间性（是否贴合 SOTA）。
> 供审查人逐条核对；风险项在"时间性"列明确标注。

## 图例

- ✅ 时间性无虞：持续维护或内容属于"不过时的经典"
- 🟡 有时效风险：内容仍可靠但部分技术非当前 SOTA，或 API 变动快（需锁版本/补阅读）
- ❌ 建议替换或仅作历史参考

---

## Part 1-5（Bigrams → WaveNet）

| 参考 | 链接 | star / 最后推送 | 可靠 | 详实 | 实践 | 时间性 |
|---|---|---|:--:|:--:|:--:|---|
| Karpathy Neural Networks: Zero to Hero（视频课） | https://karpathy.ai/zero-to-hero.html | — | ✅权威 | ✅ | ✅ | 🟡 经典教学（2023 前技术），非 SOTA——但作为教学基底足够，现代组件由 Part 7-10 补 |
| karpathy/nn-zero-to-hero（课程仓库） | https://github.com/karpathy/nn-zero-to-hero | 24.2k / 2024-08 | ✅ | ✅ | ✅ | 🟡 同上（官方停更但内容自洽） |
| karpathy/makemore（源项目） | https://github.com/karpathy/makemore | — | ✅ | 🟡 | ✅ | 🟡 同上 |

> 审查结论：这 5 部分的教学价值依赖"Karpathy 第一性原理讲解"，不追求 SOTA；**不建议替换**，
> 建议在 Part 1 README 顶部加一句"本部分是 2022 年技术基线，现代组件见 Part 7-10"（现有 README 已有类似说明）。

## Part 6（Transformer/GPT）

| 参考 | 链接 | star / 最后推送 | 可靠 | 详实 | 实践 | 时间性 |
|---|---|---|:--:|:--:|:--:|---|
| karpathy/ng-video-lecture（"Let's build GPT"） | https://github.com/karpathy/ng-video-lecture | 4.9k / 2024-01 | ✅权威 | ✅ | ✅ | 🟡 GPT-2 基线（教学经典，非 SOTA） |
| 视频 "Let's build GPT" | https://www.youtube.com/watch?v=kCc8FmEb1nY | — | ✅ | ✅ | ✅ | 🟡 同上 |
| rasbt/LLMs-from-scratch（补充：更现代的从零实现） | https://github.com/rasbt/LLMs-from-scratch | 104.0k / 活跃 | ✅ | ✅ | ✅ | ✅ 建议列为"延伸读物" |

## Part 7（minimind 复现）

| 参考 | 链接 | star / 最后推送 | 可靠 | 详实 | 实践 | 时间性 |
|---|---|---|:--:|:--:|:--:|---|
| jingyaogong/minimind（主源） | https://github.com/jingyaogong/minimind | 55.2k / **2026-08-29** | ✅ | ✅ | ✅ | ✅ 昨日仍在推送；GRPO/工具调用/蒸馏均已跟进，SOTA 贴合度极高 |
| minimind 数据集（ModelScope） | https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files | — | ✅ | ✅ | ✅ | ✅ |
| 论文组（RMSNorm/RoPE/GQA/SwiGLU/DPO） | 见教程 README 引用 | — | ✅ | ✅ | — | ✅ 原理类论文不过时 |

> 审查结论：**Part 7 的时间性是全课程最好的**（主源日更）。风险仅一处：课程 05 章毕业指南
> 锁定在 2026-08 快照，上游迭代快（已有 train_ppo/train_grpo/train_agent 等新脚本）——
> 指南已写"以原仓库为准"声明，建议每学期复核一次。

## Part 8（后训练全流程）

| 参考 | 链接 | star / 最后推送 | 可靠 | 详实 | 实践 | 时间性 |
|---|---|---|:--:|:--:|:--:|---|
| FareedKhan-dev/train-llm-from-scratch（主源） | https://github.com/FareedKhan-dev/train-llm-from-scratch | 9.5k / 2026-08-17 | ✅ | ✅ | ✅ | 🟡 **时间线核查结论（2026-08-30 commit log）**：结构详实且强连贯（每阶段加载上一阶段 ckpt，GSM8K 单指标贯穿 Base/SFT/DPO/PPO/GRPO）；2026-06-04 一次性 squash 提交全部现代化（GRPO/RLVR/评估 harness/UI），非冻结遗物、也非 SOTA 跟踪（GRPO 晚 R1 约 17 个月，8 月仅外观提交）；主管线刻意用经典架构（MHA+learned PE，RoPE/GQA 仅在旁支 notebook）。**定性：高质量自包含教学仓库，含 GRPO，但非前沿追踪** |
| 原理论文（InstructGPT/DPO/ORPO/KTO/PPO/GRPO） | 见教程 README 引用 | — | ✅ | ✅ | — | ✅ 方法论论文不过时 |
| huggingface/trl（补充：工业级对齐库的写法） | https://github.com/huggingface/trl | 19.2k / **2026-08-30** | ✅ | ✅ | ✅ | ✅ 建议列为"对照读物"：看同样的算法在 production 库里怎么写 |
| rasbt/LLMs-from-scratch（**延伸与拓展**，2026-08 评估） | https://github.com/rasbt/LLMs-from-scratch | 104.0k / 2026-08-29 | ✅ | ✅ | ✅ | ✅ 评估：对已修完本课程的学生难度 **2.5/10**（ch01-05 ≈90% 复习）；真增量=ch06 分类微调、ch07 指令数据工程+LLM-as-judge、附录 E 从零 LoRA、ch04 KV 变体显存估算；**续作 reasoning-from-scratch（RLVR-GRPO/进阶 GRPO/蒸馏）与 Part 11 双视角对照**。定位：巩固资料库，不作主线下一章 |

> 审查结论与建议（2026-08-30 修订）：主源时间线核查后撤销"2024 基线"表述——正确说法是
> "2026-06 现代化（含 GRPO/RLVR/评估），架构为教学经典款"。**给 Part 8 README 加时效声明**
> （措辞见上表）；rasbt 定位为"延伸与拓展"（见 Part 8 表第二行）而非换源候选。

## Part 9（CUDA 内核）

| 参考 | 链接 | star / 最后推送 | 可靠 | 详实 | 实践 | 时间性 |
|---|---|---|:--:|:--:|:--:|---|
| Infatoshi/cuda-course（主源） | https://github.com/Infatoshi/cuda-course | 4.0k / 2026-03 | ✅ | 🟡（笔记式 README，需教程补全——本课已补） | ✅ | ✅ 基础内核教学不过时；🟡 未覆盖 Blackwell/TMA 等新硬件特性 |
| Simon Boehm 矩阵乘法优化博客 + SGEMM_CUDA | https://siboehm.com/articles/22/CUDA-MMM · https://github.com/siboehm/SGEMM_CUDA | 1.3k / 2025-09 | ✅ | ✅✅ | ✅ | ✅ 优化方法论（coalescing/tiling/寄存器）是长期有效的"经典中的经典" |
| NVIDIA CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | 官方持续更新 | ✅ | ✅ | — | ✅ |
| PMPP（Programming Massively Parallel Processors, 4th ed） | 书 | — | ✅ | ✅✅ | 🟡（书非仓库） | ✅ 4 版已覆盖 Hopper |
| GPU MODE（讲座社区，原 CUDA MODE） | https://www.youtube.com/@GPUMODE | — | ✅ | ✅ | ✅ | ✅ 持续更新，SOTA 内核讲座 |

> 审查结论：可靠。**进阶缺口**（TMA/cluster/Tensor Core 编程）GPU MODE 与 PMPP 4ed 可补，
> 不影响本课"教学内核"的时效性。

## Part 10（分布式训练）

| 参考 | 链接 | star / 最后推送 | 可靠 | 详实 | 实践 | 时间性 |
|---|---|---|:--:|:--:|:--:|---|
| pytorch/examples minGPT-ddp（主源） | https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp | 24.0k（整仓）/ 2025-09 | ✅官方 | ✅ | ✅ | ✅ 官方维护版（已随 PyTorch 教程系列更新） |
| PyTorch DDP 官方教程系列 + FSDP 教程 | https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html 等 | 官方 | ✅ | ✅ | ✅ | ✅；⚠️ FSDP1 已弃用 → 教程已同时教 FSDP2 `fully_shard` |
| huggingface/nanotron | https://github.com/huggingface/nanotron | 2.8k / 2026-05 | ✅ | ✅ | ✅ | ✅ |
| Ultra-Scale Playbook（HF） | https://huggingface.co/spaces/nanotron/ultrascale-playbook | — | ✅ | ✅✅ | ✅ | ✅ 2025 年内容，530 卡 Llama 实战，当前最系统的集群训练教程 |
| ZeRO / Megatron-LM / 序列并行论文 | arXiv 1910.02054 / 1909.08053 / 2205.05198 | — | ✅ | ✅ | — | ✅ 奠基论文不过时 |

> 审查结论：全部可靠且新鲜。可补充 torchtitan（PyTorch 官方预训练脚手架，FSDP2 实战）作为延伸。

---

## 拟开课程（主线 Part 11-14 + 应用线）参考清单

### 主线 Part 11：对齐实战（RLHF/GRPO）

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| verl-project/verl（字节，GRPO 事实标准） | https://github.com/verl-project/verl | 23.2k / **2026-08-29** | ✅ 日更级活跃 |
| OpenRLHF | https://github.com/OpenRLHF/OpenRLHF | 10.0k / 2026-08-13 | ✅ |
| huggingface/trl | https://github.com/huggingface/trl | 19.2k / **2026-08-30** | ✅ |
| GRPO/DAPO 等论文 | arXiv 2402.03300 等 | — | ✅ 方法论 |

### 主线 Part 12：微调实战（LoRA/QLoRA）

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| LLaMA-Factory | https://github.com/hiyouga/LlamaFactory | 74.4k / 2026-08-27 | ✅ |
| unsloth（单卡加速微调 + 免费 Colab notebook） | https://github.com/unslothai/unsloth | 75.2k / **2026-08-30** | ✅ |
| huggingface/peft | https://github.com/huggingface/peft | 活跃 | ✅ |

### 主线 Part 13：预训练数据工程

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| Data-Juicer（阿里，全流水线） | https://github.com/datajuicer/data-juicer | 7.0k / **2026-08-28** | ✅ |
| FineWeb 博客 + 技术报告（HF） | https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1 | — | ✅ 工业去重/过滤的事实标准叙述 |
| Gopher/C4 启发式规则论文 | arXiv 2112.11446 / 2104.08758 | — | ✅ 方法论 |

### 主线 Part 14：推理部署实战

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| vLLM | https://github.com/vllm-project/vllm | 90.5k / **2026-08-30** | ✅ 内容 SOTA；⚠️ API 迭代快，教学需锁版本 |
| SGLang | https://github.com/sgl-project/sglang | 32.9k / **2026-08-30** | ✅ |
| llama.cpp（端侧/GGUF） | https://github.com/ggml-org/llama.cpp | 126.3k / **2026-08-30** | ✅ |
| PagedAttention/Orca 论文 | arXiv 2309.06180 / OSDI'22 | — | ✅ 原理不过时 |

### 应用线 A1：最小 RAG

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| ragflow（平台化对照） | https://github.com/infiniflow/ragflow | 89.6k / 2026-08-29 | ✅ |
| llama_index（抽象对照） | https://github.com/run-llama/llama_index | 51.9k / 活跃 | ✅ |
| Anthropic《Contextual Retrieval》（现代检索增强技巧） | https://www.anthropic.com/news/contextual-retrieval | — | ✅ 2024-12，仍为当前最佳实践 |
| ⚠️ LangChain | https://github.com/langchain-ai/langchain | 145k / 活跃 | 🟡 框架本体仅作生态认知（过度封装争议 + 教程偏营销），概念优先 |

### 应用线 A2：Agent

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| pi（极简 harness，OpenClaw 底层） | https://github.com/earendil-works/pi | **99.2k** / **2026-08-30** | ✅ 教学价值极高（读得懂整个 loop） |
| Anthropic《Building Effective Agents》（方法论原文） | https://www.anthropic.com/research/building-effective-agents | — | ✅ 工作流 vs 智能体的界定是当前共识 |
| smolagents（HF，CodeAgent） | https://github.com/huggingface/smolagents | 29.0k / 2026-08-25 | ✅ |
| OpenAI Agents SDK | https://github.com/openai/openai-agents-python | 29.1k / 2026-08-28 | ✅ |
| Google ADK | https://github.com/google/adk-python | 21.3k / **2026-08-30** | ✅ 内容新；⚠️ 2.0 breaking changes、约两周一发，教学需锁版本 |
| LangGraph（只学图/状态机/checkpoint 概念） | https://github.com/langchain-ai/langgraph | 40.7k / **2026-08-30** | 🟡 概念值得学；生态营销内容需甄别 |

### 通用（面试/体系课）

| 参考 | 链接 | star / 最后推送 | 时间性 |
|---|---|---|---|
| wdndev/llm_interview_note | https://github.com/wdndev/llm_interview_note | 15.0k / 活跃 | ✅ 中文八股维护中 |
| mlabonne/llm-course | https://github.com/mlabonne/llm-course | 82.1k / 活跃 | ✅ 英文体系课，持续更新 |
| Karpathy《Deep Dive into LLMs like ChatGPT》（2025 总览视频） | https://www.youtube.com/watch?v=7xTGNNLPyMI | — | ✅ 2025-02，全局图景的最佳单集 |

---

## 版本锁定策略（2026-08-30 按审阅意见修订）

**原则：锁 LTS 或 latest（最新 release tag），不锁任意旧版**——latest 带新特性且是当下最流行，
锁它才有教学时效价值。逐仓落点：vLLM=独立 venv 装 latest（0.28.0/torch 2.13），备选
`vllm==0.6.6`（恰 pin torch==2.5.1+cu121，可复用课程 venv）；verl=latest tag + Docker
（耦合重）；LlamaFactory/Data-Juicer=latest 可直接跟随；课程主 venv（torch 2.5.1）不升级。
详见 [part11_14_tutorial_design.md](part11_14_tutorial_design.md) §0。

## 审查总结（三句话）

1. **全部主源仓库当日/近日仍在推送**（minimind、verl、trl、vLLM、llama.cpp、pi、ADK 均为
   2026-08-29/30），无失活主源；唯一内容级时效风险是 **Part 8 主源 train-llm-from-scratch
   （2024 基线）**——课程已用新增章节覆盖差距，建议按上表加时效声明。
2. **次级时效风险**集中在"API 变动快"的工程仓库（vLLM/ADK/LangGraph）：教学时锁版本 +
   概念优先，指南与课程均已按此策略书写。
3. **经典参考资料**（Karpathy 课程、siboehm 博客、PMPP、奠基论文）属"不过时"类，无需替换；
   Karpathy 部分建议在 README 顶部保留"2022-2023 技术基线，现代组件见 Part 7-10"的定位声明。
