# 02 — 奖励设计与工业框架（Agentic RL 的工程全景）

> 🧭 01 章跑通了最小闭环。本章补齐工程决策：长程任务的**奖励怎么设计**、
> **训练不稳定怎么治**（Echo Trap）、**框架怎么选**、**怎么评估**。

## 📖 前置知识

- **01 章**：多轮轨迹、观测 mask、轨迹级 GRPO

## 1. 奖励设计谱系（长程任务的核心难题）

| 类型 | 做法 | 优 | 劣 |
|---|---|---|---|
| 稀疏结果奖励 | 只看最终答案对错（Search-R1） | 简单、不可作弊 | 10+ 轮轨迹的 credit assignment 噪声大 |
| 过程/塑形奖励 | ToolRL：格式奖励 + 每步正确性，细粒度分解 | 信号密、收敛快 | 设计不当会引入偏差 |
| ORM（结果监督 RM） | 训练奖励模型打分 | 可泛化到难形式化任务 | RM 可被 hack、有偏差 |
| 课程学习 | WebRL：从失败任务再生任务，逐步加长 | 与模型能力同步演进 | 系统复杂 |

- 🔑 **ToolRL 的实证**（2504.13958）：把奖励分解为"格式正确 + 参数正确 + 结果正确"
  的细粒度塑形，显著优于单一结果奖励。
- 🔑 **Agentic 特有的 reward hacking**（与 Part 8 07 章污染是近亲）：复读机式工具调用
  循环、调用"回显 prompt"的工具、打印预期答案作弊、钻环境 bug。防线：调用去重、
  轮数上限+溢出惩罚、环境沙盒、轨迹抽样人工审查。

## 2. 稳定性：Echo Trap 与 StarPO-S

多轮 RL 特有的失败模式（RAGEN 论文 2504.20073）：策略熵坍缩到**重复模板**
（同样的工具调用循环往复）——奖励曲线看不出来（模板可能还拿低分），但探索已死。

```
发现：监控 rollout 熵 + 轨迹多样性（不同轨迹比例）
缓解：StarPO-S = critic 辅助 + clip-higher（提高上界探索）+ rollout 过滤（丢弃零方差组）
```

- 💡 我们脚本 01 的组内优势在全同组时归零（Part 11 的性质），本质是同一现象的
  单轮版——"无区分度的组没有梯度"。

## 3. 工业框架选型（2026-08）

| 框架 | star | 特点 | 适合 |
|---|---|---|---|
| **verl** | 23.2k | multi-turn/tool-agent 支持最全（docs 好） | 首选入门与生产 |
| **verl-agent** | 2.3k | GiGPO 官方实现，ALFWorld/TextWorld 玩境 | 小模型 agent RL 研究 |
| slime | 8.3k | 智谱系，Megatron+SGLang，custom generate 灵活 | 大规模生产 |
| rLLM | 5.8k | harness/sandbox 无关的干净 env API | 研究原型 |
| AgentGym-RL | 855 | 多环境开箱 | 教学对比 |
| SkyRL / AReaL | 2.2k / 5.7k | 全异步（长尾轨迹场景） | 大规模 |

> 24GB 实操：verl multi-turn + Qwen2.5-0.5B + 计算器/检索工具（Part 11 环境复用）；
> verl-agent 的 TextWorld 玩境（0.5B/1.5B 友好）。SkyRL/AReaL/AgentGym-RL 按
> 文档定位 ≥8 卡，引用不实操。

## 4. 评估（Agentic 版）

| 基准 | 测什么 | 24GB 可评 |
|---|---|---|
| τ-bench | agent+模拟用户+策略合规（零售/航空） | ✅ 轻量（需 LLM 演用户） |
| GAIA L1 子集 | 真实问题（推理+浏览+工具） | ✅ 文本子集 |
| AgentBench | 8 环境（DB/OS/Web…） | ⚠️ 环境重 |
| WebArena / SWE-bench | 自托管网站 / 真实 issue 修复 | ❌ 小模型≈0%，大模型+重环境 |

## 学完本部分你能...

- ✅ 按任务特征选奖励类型并设计防 hacking 防线
- ✅ 识别 Echo Trap 并说出 StarPO-S 的三个缓解件
- ✅ 给出框架选型决策（verl 起步 → verl-agent/slime 进阶）
- ✅ 为 agent 模型选评估基准（τ-bench/GAIA L1 起步）

**课后练习**

<details>
<summary>Q1: GiGPO 相比轨迹级 GRPO 解决什么？</summary>
A: 轨迹级优势对长轨迹粒度太粗（哪个工具调用是关键的？不可知）。GiGPO 在"锚定状态"
（跨 episode 出现的相同状态）上建 step 级分组，给细粒度 credit 且保持 critic-free
（Part 11 04 章 GRPO 思想的 step 级推广）。
</details>

<details>
<summary>Q2: 工具输出动辄几 KB，多轮后撑爆上下文。三种工程处理？</summary>
A: ① 截断/摘要（Search-R1 只保留相关片段）；②观测入上下文但 mask 出 loss 且
history 压缩；③ partial rollout（上下文满时强制截断轨迹并保留已完成部分的优势）。
verl 的 partial rollout 与 slime 的 context engineering 都在此列。
</details>

## 📝 课后作业

👉 [Assignment 17](../../../assignments/assignment_17/)

## 🎓 Part 17 完结：Agentic RL 是 2026 后训练 JD 的第一关键词——你现在拥有
从机制手写（脚本 01）到框架实操（02 章路线）的完整入门。继续：
[面试指南 §7b 方向深挖](../../../docs/llm_interview_guide.md)。

---

[← 上一章](01_from_single_turn_to_agent.md) | [Part 17 README](README.md)
