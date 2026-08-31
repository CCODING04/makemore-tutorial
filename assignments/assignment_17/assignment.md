# Assignment 17：Agentic RL

> 对应 Part 17 教程（[01 从单轮到 Agent](../../courses/Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md) / [02 奖励设计与工业框架](../../courses/Part17_agentic_rl/tutorial/02_rewards_and_frameworks.md)）。
> 四题纯 CPU 可完成。

## 题目（实现 `agentic_exercises.py`）

1. **轨迹 loss mask**（30 分）：给定各 assistant 段切片，构造 0/1 mask；
   `assistant_token_fraction` 算 assistant 占比（Agentic 轨迹典型 10-30%）
2. **轨迹级 GRPO**（30 分）：逐组标准化优势；**全同组 → 全 0**（无区分度组没有梯度）
3. **工具调用解析**（25 分）：从含任意文本的输出中按序解析
   `<tool_call> name arg1 arg2 ... </tool_call>`（正则）
4. **🌟 Echo Trap 检测**（15 分）：多样性得分 = 不同轨迹数 / 总轨迹数；
   低分 = 探索坍缩（RAGEN 论文的 Echo Trap）

## 实验题（观测型）

- 跑脚本 01：把 G 从 8 改成 1（=无组基线），观察学习是否退化（组内标准化消失）
- 把 MAX_TURNS 从 2 改到 4，观察成功率与"观测 token 占比"的变化
  （轨迹越长 assistant 占比越低、credit assignment 越难）

## 🎯 面试直通车

- "Agentic RL 和单轮 RLVR 的实现差异？"——多轮轨迹、观测 mask、轨迹级优势广播、
  异步 rollout、上下文管理（答 3 个即合格）
- "观测 token 为什么 mask 出 loss？"——不是模型的话；算 loss 会教模型幻觉工具结果
- "Echo Trap 是什么？怎么发现/缓解？"——多轮 RL 的熵坍缩到重复模板；监控轨迹多样性；
  StarPO-S（critic + clip-higher + rollout 过滤）
- "奖励怎么设计？"——稀疏结果 / ToolRL 式塑形（格式+参数+结果分解）/ ORM；
  配课程学习（WebRL 式从失败再生任务）
