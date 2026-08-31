# Part 17: Agentic RL — 让模型学会自主使用工具、完成长程任务

> 🧭 Part 11 的 GRPO 解决的是"单轮答题"（一个问题一个回答一个奖励）。
> 2026 年真正的战场是 **Agentic RL**：模型在多轮循环中调用工具、观察结果、规划下一步，
> 用**整条轨迹**的结果作为奖励来学习——字节/B 站/滴滴/NIO 的 2026 JD 都把它写进了原文。
> 本章手写这条管线的最小闭环（CPU 可跑），再对照工业框架。
> 锚点仓库：[verl](https://github.com/volcengine/verl)（multi-turn 支持）·
> [RAGEN/StarPO](https://github.com/RAGEN-AI/RAGEN)（多轮 RL 稳定性）· [AgentGym-RL](https://github.com/woooodyy/AgentGym-RL)

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [从单轮 RLVR 到 Agentic RL](01_from_single_turn_to_agent.md) | 多轮轨迹/观测 mask/轨迹级奖励/Echo Trap；手写管线 | `01` |
| 02 | [奖励设计与工业框架](02_rewards_and_frameworks.md) | 稀疏 vs 塑形奖励、课程学习、verl/slime/rLLM 选型、评估（τ-bench/GAIA） | —（CLI 实操） |

## 🧰 前置知识

- **Part 8 04 章 + Part 11**：GRPO 组内优势、KL 惩罚（本章直接复用）
- **Part 12**：chat template（工具调用协议就是它的扩展）

## 🔗 在 LLM 链路中的位置

```
Part 11（单轮 GRPO）→ 【本部分: 多轮 + 工具 + 环境 = Agentic RL】→ 真实 Agent 产品
```

为什么是主战场：OpenAI Deep Research/Kimi-Researcher 披露"端到端 RL on hard tasks"；
Kimi 平均 23 次工具调用/回答——**会写单轮 GRPO ≠ 会训 Agent**，差异全在多轮机制。

## 📦 环境

脚本 01 **纯 CPU 可跑**（~20 秒，零新依赖）。工业框架（verl multi-turn / verl-agent）
用 Docker + 0.5B 模型（Part 11 02 章的环境直接复用）。

## 📈 学习地图

```
多轮轨迹采集（工具调用→观测重进上下文）   ← 点：与单轮的本质差异
   ↓ 轨迹级 GRPO（组内优势广播到全部 assistant token）
BC 冷启动 → RL（R1 同款两阶段）           ← 线：0% → 85% 的可复现实证
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

[← 上一章：Part 16 图像/视频生成](../../Part16_image_video_generation/tutorial/README.md) | [Part 17 README](README.md)
