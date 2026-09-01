# Part 17: Agentic RL — 让模型学会自主使用工具、完成长程任务

> 🧭 Part 11 的 GRPO 解决的是"单轮答题"（一个问题一个回答一个奖励）。
> 2026 年真正的战场是 **Agentic RL**：模型在多轮循环中调用工具、观察结果、规划下一步，
> 用**整条轨迹**的结果作为奖励来学习——字节/B 站/滴滴/NIO 的 2026 JD 都把它写进了原文。
> 本部分手写这条管线的最小闭环（GPU ~15 秒 / 纯 CPU 约半分钟，内置**真实掩码消融对照**），
> 再对照工业框架。
> 锚点仓库：[verl](https://github.com/volcengine/verl)（multi-turn 支持）·
> [RAGEN/StarPO](https://github.com/RAGEN-AI/RAGEN)（多轮 RL 稳定性）· [AgentGym-RL](https://github.com/woooodyy/AgentGym-RL)

## 学习目标

完成本部分后，你将能够：

- ✅ **手写** 多轮工具调用 rollout（观测 mask、上下文重进）并完成 BC 冷启动 → RL 两阶段
- ✅ **实现** 轨迹级 GRPO（组内优势广播 + 观测 loss-mask）并解释其工程权衡
- ✅ **复现** 掩码消融：观测泄漏让策略学会"复读观测"而非泛化（附真实数字）
- ✅ **配置** verl 的 multi-turn 训练与自定义奖励函数
- ✅ **识别** Echo Trap 等 Agentic 陷阱并设计奖励防 hacking 防线

## 理论背景（导览）

核心三问——多轮轨迹怎么采、观测 token 要不要算 loss、稀疏的轨迹级奖励怎么分配到
每个 token——的完整推导与实测放在 01 章展开（问题引入 → 轨迹级 GRPO 数学 →
同 seed 掩码消融），本页只给地图：

| 概念 | 一句话 | 详见 |
|------|--------|------|
| 多轮轨迹 | 问题 → 调工具 → 看结果 → 再调 → 答案，整条轨迹一个奖励 | [01 章](01_from_single_turn_to_agent.md) |
| 观测 mask | 环境给的 token 不进 loss；观测含答案时更要防"复读"泄漏 | [01 章消融实测](01_from_single_turn_to_agent.md) |
| 轨迹级 GRPO | 组内标准化优势广播到全部 assistant token | [01 章](01_from_single_turn_to_agent.md) |
| 奖励与稳定性 | 稀疏 vs 塑形、Echo Trap / StarPO-S、框架与评估选型 | [02 章](02_rewards_and_frameworks.md) |

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [从单轮 RLVR 到 Agentic RL](01_from_single_turn_to_agent.md) | 多轮轨迹/观测 mask/轨迹级奖励/掩码消融/Echo Trap；手写管线 | `01` |
| 02 | [奖励设计与工业框架](02_rewards_and_frameworks.md) | 稀疏 vs 塑形奖励、课程学习、verl/slime/rLLM 选型、评估（τ-bench/GAIA） | —（CLI 实操） |

## 🧰 前置知识

- **必须掌握**：[Part 11 对齐实战](../../Part11_alignment_verl/tutorial/README.md)——
  GRPO 组内优势、KL 惩罚（本部分轨迹级优势 = 它的广播版，"全同组优势全零"性质直接复用）
- **建议掌握**：
  - [Part 8 04 章 PPO/GRPO](../../Part8_post_training/tutorial/04_ppo_and_grpo.md)——
    策略梯度与优势估计的推导（看懂轨迹级 GRPO 的数学）
  - [Part 12 微调实战](../../Part12_finetune_llamafactory/tutorial/README.md)——
    chat template（工具调用协议就是它的扩展）
- **可选**：[Part 14 推理部署](../../Part14_inference_vllm/tutorial/README.md)——
  vLLM/SGLang 引擎背景（verl/slime 的 rollout 层建立在它之上，读框架源码前值得了解）

## 🔗 在 LLM 链路中的位置

```
Part 11（单轮 GRPO）→ 【本部分: 多轮 + 工具 + 环境 = Agentic RL】→ 真实 Agent 产品
```

为什么是主战场：OpenAI Deep Research/Kimi-Researcher 披露"端到端 RL on hard tasks"；
Kimi 平均 23 次工具调用/回答——**会写单轮 GRPO ≠ 会训 Agent**，差异全在多轮机制。

## 📦 环境

脚本 01 **纯 CPU 可跑**（GPU ~15 秒 / 纯 CPU 约半分钟，零新依赖，seed 固定）。
工业框架（verl multi-turn / verl-agent）用 Docker + 0.5B 模型（Part 11 02 章的
环境直接复用）。

## 📈 学习地图

```
多轮轨迹采集（工具调用→观测重进上下文）   ← 点：与单轮的本质差异
   ↓ 轨迹级 GRPO（组内优势广播到全部 assistant token）
BC 冷启动 → RL（R1 同款两阶段）           ← 线：冷启动覆盖度决定 RL 能否启动
   ↓
掩码消融（观测泄漏 → 复读 vs 内化）        ← 线：同 seed 对照实验的真实数字
   ↓
奖励设计与稳定性（Echo Trap）→ 工业框架   ← 面
```

## 📝 课后作业

👉 [Assignment 17](../../../assignments/assignment_17/)

## 🔗 相关资源

- 📄 综述：The Landscape of Agentic Reinforcement Learning for LLMs (arXiv 2509.02547)
- 📄 Search-R1 (2503.09516) · RAGEN/StarPO (2504.20073) · GiGPO (2505.10978) · ToolRL (2504.13958)
- 📝 Silver & Sutton《Welcome to the Era of Experience》（DeepMind，2025）
- 🐙 [verl multi-turn 文档](https://github.com/volcengine/verl) · [verl-agent（GiGPO 官方）](https://github.com/langfengq/verl-agent) · [AgentGym-RL](https://github.com/woooodyy/AgentGym-RL)

---

[← 上一章：Part 16 图像/视频生成](../../Part16_image_video_generation/tutorial/README.md) | 🎓 全课程结业：[返回总览](../../../README.md)

🎓 **到这里，Part 1-17 的全部课程已经走完**——从字符级 bigram 一路到 Agentic RL。
结业导览：回[总览 README](../../../README.md) 检查"如何判断学完一个 Part"清单，
然后带着你的实操记录（本课每章的真实数字）进入
[面试指南 §7b 方向深挖](../../../docs/llm_interview_guide.md)。
