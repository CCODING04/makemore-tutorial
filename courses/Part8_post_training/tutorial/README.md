# Part 8: 从零训练 LLM — 后训练全流程（SFT -> 奖励模型 -> DPO/PPO/GRPO）

> 🚀 从零构建一个 GPT-2，走完 LLM 的完整生命周期：预训练 → SFT → 奖励模型 → 对齐 → 强化学习 → 评估。
> 参考：[train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [GPT-2 与预训练](01_gpt_and_pretrain.md) | 经典 GPT-2 架构（LayerNorm + learned PE + MHA + ReLU）、预训练流水线 | `01` `02` |
| 02 | [SFT 与 Chat Template](02_sft_and_chat.md) | 监督微调、Chat Template、Prompt Masking | `03` |
| 03 | [奖励模型与对齐算法](03_reward_and_dpo.md) | Bradley-Terry 奖励模型、DPO/ORPO/KTO 三种对齐 | `04` `05` |
| 04 | [强化学习：PPO 与 GRPO](04_ppo_and_grpo.md) | PPO（GAE + Clipped Surrogate）、GRPO（Critic-Free RL） | `06` `07` |
| 05 | [评估与推理部署](05_eval_and_deploy.md) | GSM8K 评估、全阶段对比、交互式 Chat | `08` |
| 06 | [推理与服务](06_inference_and_serving.md) | 量化 int8/int4（GPTQ/AWQ）、KV 显存与 KIVI、PagedAttention、连续批处理、投机解码、TTFT/TPOT、vLLM 实操 | `09` |
| 07 | [评估学](07_evaluation.md) | 规则/人工/LLM-judge 三范式、lm-eval-harness（实操 + 自定义 task）、HELM、benchmark 污染（GSM1k）、ppl 陷阱；**幻觉与安全**（语义熵/SelfCheckGPT、温度迷思、ECE 校准、refusal direction、HarmBench/JailbreakBench）；中国合规四件套 | `11` `12` |
| 08 | [LoRA 与分类微调](08_lora_and_classification.md) | 从零写 LoRA（低秩分解注入）、参数量/显存对比、分类微调回顾 | `10` |
| 09 | [推理模型与 test-time compute](09_reasoning_models.md) | R1 四阶段管线、cold start SFT → 推理 RL → self-consistency | `09` |

## 📚 参考来源标注（两个源仓库各管什么）

本部分内容来自两个风格不同的源仓库，按章标注——**学习时按需对照，不要混着读**：

| 章节 | 主源：[train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)（端到端管线） | 延伸：[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（更严谨的分章实现） |
|---|---|---|
| 01 GPT-2 与预训练 | 模型搭建/预训练流水线主线 | ch04-05（GPT 从零重推、**OpenAI GPT-2 权重加载/权重手术**、附录 D LR 调度细节） |
| 02 SFT 与 Chat Template | SFT/Prompt Masking 主线 | ch07（**指令数据的 JSON 格式规约**、Alpaca 数据组织） |
| 03 奖励模型与对齐 | RM/DPO/ORPO/KTO 主线 | ch07 的 `04_preference-tuning-with-dpo`（**DPO 从零 + 偏好数据如何构造**） |
| 04 PPO 与 GRPO | PPO/GRPO 主线 | （主书无 RL）→ 续作 [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) ch06-07（**RLVR-GRPO from scratch、进阶 GRPO 变体**）——与 Part 11 双视角 |
| 05 评估与部署 | GSM8K/Chat 主线 | — |
| 06 推理与服务 | 课程自研（手写模拟） | — |
| 07 评估学 | 课程自研 | ch07 的 `ollama_evaluate.py`（**LLM-as-judge 的最小可运行实现**） |
| 08 LoRA 与分类微调 | — | **本新章主线**：附录 E（LoRA 从零）+ ch06（分类微调） |

> 💡 阅读建议：主源负责"跑通全流程"，rasbt 负责"把某一章做扎实"。修完本部分后，
> 把 rasbt 的 ch06/附录 E/续作 ch06 当作三个巩固模块重走一遍（难度对我们学生约 2.5-4/10）。

## 🧰 前置知识

本部分需要你已经掌握：

- **Part 6 全部内容**：Transformer 架构、self-attention、残差连接、LayerNorm、decoder-only GPT —— Part 8 的模型骨架直接复用 Part 6
- **Part 3 的 BatchNorm**：归一化的思想 —— 讲 LayerNorm 时会和它对照
- **概率论基础**：sigmoid、log-probability、KL 散度 —— DPO/PPO/GRPO 的核心数学工具

> 💡 Part 8 与 Part 7 独立——不依赖 Part 7 代码。重合内容（Transformer 架构、SFT、DPO）视为复习，但从不同角度讲解。

## 🗺️ 学习路线图

```
Part 6 (Transformer 架构、self-attention、decoder-only GPT)
    │
    │  "架构懂了，但模型只会续写，不会对话、不会遵循指令..."
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Part 8: LLM 后训练全流程                                     │
│                                                              │
│  ① GPT-2 架构 + 预训练    — LayerNorm + learned PE + ReLU   │──→ 01_gpt_and_pretrain.md
│  ② SFT + Chat Template    — Prompt Masking                   │──→ 02_sft_and_chat.md
│  ③ 奖励模型 + DPO/ORPO/KTO — Bradley-Terry + 三种对齐       │──→ 03_reward_and_dpo.md
│  ④ PPO + GRPO             — GAE + Clipped Surrogate          │──→ 04_ppo_and_grpo.md
│  ⑤ 评估 + 部署            — GSM8K + 生成策略 + Chat          │──→ 05_eval_and_deploy.md
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 📦 数据与依赖

**本课程完全自包含**：数据、权重都不需要提前下载，全部脚本可直接跑通——

| 需要的东西 | 说明 |
|------|------|
| 数据 | `data/input.txt`（tiny Shakespeare）已在仓库内 |
| Python 依赖 | 脚本 01-10：仅需 `torch`（CPU/GPU）；脚本 11-12 另需 `transformers` + 已缓存 Qwen2.5-0.5B（-Instruct）、脚本 12 需 `lm_eval[hf]`（见根 requirements.txt 可选区） |
| 预训练权重 | 脚本 01-10 不需要（从零训练）；脚本 11-12 首次运行会从 HF 拉取 0.5B 模型（缺失时打印指引优雅退出，rc=0） |

**规模对照表**（想跑原版规模时从这里查）：

| 配置 | n_embed | heads | blocks | 参数量 | 说明 |
|---|:---:|:---:|:---:|---:|---|
| 本课 CPU 模式 | 64 | 4 | 2 | ~2M | 全部脚本默认，<30s |
| 本课 GPU 模式 | 512 | 8 | 12 | ~40M | 单张 4090 余量充足 |
| 原仓库 tutorial base | 512 | 8 | 8 | 77M | train-llm-from-scratch 的基准档 |
| 原仓库 post-training 默认 | 1024 | 16 | 24 | **406M** | 单卡 4090 可跑：模型状态约 6.5GB，用 **batch=4 + 梯度累积**控制激活（详见下方备注）；更稳妥可租 2×24GB 或降为 77M 档 |

> 🖥️ **多卡备注**：本课所有脚本（含 GPU 模式）都是**单卡程序**——一张 4090 可完成
> 课程全部内容与作业；想跑 406M 原版规模的单卡步骤：`batch_size=4` + `gradient_accumulation_steps=8`
> （保有效 batch），激活约 4GB + 模型状态 6.5GB，24GB 余量充足；若再放大 batch 或 seq，
> 租 2×24GB 卡（数据并行 DDP，见 Part 10）或 A100 80GB。

> ⚠️ **参数放大时的超参因果**（面试常问，别死抄数字）：模型放大 10×，
> ① **lr 降**（梯度噪声占比变化，406M 用 ~3e-4 而不是 2M 的 3e-3）；
> ② **effective batch 升**（用梯度累积凑，稳住大 batch 的统计量）；
> ③ **warmup 步数升**（大模型初期更脆）；
> ④ **seq/batch 与激活显存联动**——放大前先按 Part 9 的显存公式估一估，别先改模型后爆显存。

## 📈 演进路线：从"续写器"到"对话助手"

本教程走完 LLM 的全部训练阶段——每一阶段解决上一阶段留下的问题：

| 阶段 | 目标 | 解决什么问题 | 对应脚本 |
|------|:---:|:---:|:---:|
| 预训练 | 预测下一个 token | 学会语言的统计规律 | `02` |
| SFT | 按指令回答 | 从"续写器"变成"对话模型" | `03` |
| 奖励模型 | 给回答打分 | 学会判断"好回答 vs 坏回答" | `04` |
| DPO/ORPO/KTO | 偏好对齐 | 直接用偏好数据优化策略 | `05` |
| PPO | 强化学习 | 用 reward model 在线优化 | `06` |
| GRPO | Critic-Free RL | 不需要 Value Network，更简单 | `07` |
| 评估 | 量化效果 | GSM8K 准确率、生成质量对比 | `08` |
| 幻觉与安全 | 可信与合规 | 语义熵检测幻觉、ECE 校准、refusal direction、lm-eval 实操、合规四件套 | `11` `12` |

> ⚠️ 我们的脚本是 **CPU 缩小版**（更小的 hidden/dim、更少的步数、更短的上下文）。不同超参、不同随机种子，数字都会有差异。**看趋势，别死记数字。** GPU 全量版请参考 train-llm-from-scratch 仓库的超参。

## 📝 课后作业

每一章末尾有 2-3 道思考题（`<details>` 折叠答案）。全部学完后，去这里做动手练习：

👉 [Assignment 8](../../../assignments/assignment_8/)

## 🌟 脚本 11 可选实验：自备数据文件格式

[07 章 §7](07_evaluation.md) 的 refusal direction 演示（Arditi 2406.11717）**不内嵌任何
提示语样本**——想跑完整实验的读者请自备 `scripts/refusal_prompts.jsonl`（每行一个 JSON
对象，normal 组放普通问答语句，refusal 组按论文附录自行构造模型拒绝风格语句，各 ≥ 8 条）：

```jsonl
{"text": "法国的首都是巴黎，这是一句普通的陈述句。", "label": "normal"}
{"text": "以下是一句普通问答：<读者自备的正常问答文本>", "label": "normal"}
{"text": "<读者按论文附录自备的模型拒绝风格语句>", "label": "refusal"}
```

文件缺失时脚本只打印说明并跳过（rc=0）；7B 模型可用时效果最好（0.5B 可演示方法）。

## 🔗 相关资源

- 🐙 [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) — 本部分参考的项目
- 📄 Radford et al. 2019：[Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- 📄 Ouyang et al. 2022：[Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- 📄 Rafailov et al. 2023：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/abs/2305.18290)
- 📄 Hong et al. 2024：[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- 📄 Ethayarajh et al. 2024：[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- 📄 Schulman et al. 2017：[Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347)
- 📄 Shao et al. 2024：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/abs/2402.03300)

---

[← 上一章：Part 7 Minimind](../../Part7_minimind/tutorial/README.md) | [下一章：Part 9 CUDA 内核 →](../../Part9_cuda_kernels/tutorial/README.md)
