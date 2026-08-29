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
| Python 依赖 | 仅需 `torch`（CPU/GPU） |
| 预训练权重 | 不需要——所有权重由脚本从零训练并自动生成 |

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

> ⚠️ 我们的脚本是 **CPU 缩小版**（更小的 hidden/dim、更少的步数、更短的上下文）。不同超参、不同随机种子，数字都会有差异。**看趋势，别死记数字。** GPU 全量版请参考 train-llm-from-scratch 仓库的超参。

## 📝 课后作业

每一章末尾有 2-3 道思考题（`<details>` 折叠答案）。全部学完后，去这里做动手练习：

👉 [Assignment 8](../../../assignments/assignment_8/)

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

[← 上一章：Part 7 Minimind](../../Part7_minimind/tutorial/README.md)
