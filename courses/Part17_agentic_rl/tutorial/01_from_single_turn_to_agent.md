# 01 — 从单轮 RLVR 到 Agentic RL

> 🧭 Part 8/11 的 GRPO：一个问题 → 一次回答 → 一个奖励（单轮）。Agentic RL 把游戏
> 改成了：**问题 → 调工具 → 看结果 → 再调 → …… → 最终答案 → 一个奖励**。
> 三个新问题随之而来：多轮轨迹怎么采、观测 token 要不要算 loss、稀疏的轨迹级奖励
> 怎么分配到每个 token。本章手写最小闭环并逐个回答（跑
> [scripts/01_toy_agent_grpo.py](../scripts/01_toy_agent_grpo.py)，CPU 20 秒）。

## 📖 前置知识

- **Part 11**：GRPO 组内优势（本章轨迹级优势 = 它的广播版）
- **Part 12 01 章**：chat template（工具调用协议 = 它的扩展）

## 1. 多轮轨迹的解剖

```
[user: 3 2 1]                                    ← 任务（mask=0）
  assistant: <tool_call> multiply 3 2 </tool_call>   ← assistant 段（mask=1）
  <tool_result> 6 </tool_result>                      ← 观测（mask=0）⭐
  assistant: <tool_call> add 6 1 </tool_call>          ← assistant 段（mask=1）
  <tool_result> 7 </tool_result>                      ← 观测（mask=0）
  assistant: 答案是 7 <eos>                            ← assistant 段（mask=1）
```

- 🔑 **观测 mask=0 的三重理由**：① 工具输出是环境生成的，不是模型的"话"；② 模型
  无法预测外部结果，算 loss 是浪费容量；③ 更糟——模型可能学会"生成自己期望的
  观测"（幻觉工具结果）。这也是 verl 的 delta-based tokenize + assistant mask
  存在的原因。
- 💡 玩具协议用空格分隔（`<tool_call> multiply 3 2 </tool_call>`）；工业实现
  （verl/Hermes 格式）用 JSON——协议本身可设计，但**BC 示范与解析器必须严格一致**
  （我们实测：示范 JSON 非法 → parse 永远失败 → 奖励恒 0 → RL 死锁）。

## 2. 手写多轮 rollout（脚本核心 ~50 行）

```python
for turn in range(MAX_TURNS + 1):
    # 自回归生成（到 </tool_call> 或 <eos>），token 记入轨迹、mask=1
    gen = generate_until(policy, cur, stop=["</tool_call>", "<eos>"])
    ids += gen; mask += [1] * len(gen)

    if "</tool_call>" in gen:
        call = parse_call(gen)                    # 解析失败 → 本轨迹终止
        obs = run_tool(call)                      # 环境执行
        ids += tokenize(obs); mask += [0] * ...   # 观测 mask=0（不是模型说的）
    else:
        break                                     # 给出最终答案，episode 结束
```

- ⚠️ 观测 token **不经过模型 head**（它们是环境的话）——但必须重进上下文
  （模型下一步要"看见"工具结果）。这个"输入有它、loss 不算它"的区分，
  就是 multi-turn 实现的全部魔法。

## 3. 轨迹级 GRPO：奖励怎么分配

整条轨迹只有末尾一个 0/1 奖励 → 组内标准化后，**优势广播到该轨迹的全部 assistant
token**（工具调用也要学——"学会调对工具"本身就是任务的一部分）：

```
A(每个 assistant token) = (r_trajectory − mean(组内 r)) / std
loss = −Σ (log π(token) × A(token)) / #assistant_tokens
```

## 4. 实测：BC 冷启动 → RL（0% → 85%）

脚本的两阶段（DeepSeek-R1 cold start 的玩具复刻）：

```
（BC 冷启动完成——随机策略采不出合法格式时，RL 会陷入零梯度死锁）
冷启动后成功率: 0.00%   ← BC 只教会"格式"，任务还没学会
round 0: 组平均奖励 = 0.083
round 1: 组平均奖励 = 0.542
round 3: 组平均奖励 = 0.740
训练后成功率: 85.00%（基线 0.00% → +85%）
```

- 🔑 **两个必讲的观察**：
  ① **没有 BC 就没有 RL**：随机初始化的策略采不出合法工具调用 → 奖励恒 0 →
  组内 std=0 → GRPO 无梯度（稀疏奖励死锁）。R1 论文的 cold start SFT 就是解这个。
  ② RL 阶段提升的 85% 来自"在 BC 格式骨架上探索出正确参数"——格式是 BC 给的，
  任务能力是 RL 探索的。两者分工清晰。

## 学完本部分你能...

- ✅ 手写多轮工具调用 rollout（含观测 mask 与上下文重进）
- ✅ 实现轨迹级 GRPO（组内优势广播 + 观测 mask）
- ✅ 复刻 R1 两阶段的玩具版并解释每阶段的分工
- ✅ 说出 Agentic RL 与单轮 RLVR 的至少三个实现差异

**课后练习**

<details>
<summary>Q1: 如果观测 token 误算进 loss，最坏会发生什么？</summary>
A: 模型学会"预测工具输出"——两个失败模式：① 工具输出部分不可预测（外部状态），
预测它是浪费容量；② 模型可能生成"自己期望的观测"而非真实调用工具（幻觉工具结果）。
这正是 Search-R1 做 retrieved-token masking 的原因。
</details>

<details>
<summary>Q2: 为什么长程任务需要异步 rollout？</summary>
A: 轨迹长度重尾分布——有的 2 步结束、有的 100+ 步（Kimi-Researcher 平均 23 次工具
调用）。同步 rollout 里 GPU 要等最长的轨迹，短轨迹早已算完在空转；异步派发
（verl rollout.mode=async / AReaL 全异步）让短轨迹先返回继续采样。
</details>

<details>
<summary>Q3: 什么是 Echo Trap（RAGEN 论文）？怎么发现和缓解？</summary>
A: 多轮 RL 特有的稳定性陷阱：策略熵坍缩到重复模板（同样的工具调用循环往复），
奖励曲线却看不出来（因为模板可能还拿低奖励）。发现：监控 rollout 的熵与
轨迹多样性。缓解：StarPO-S（critic + clip-higher + rollout 过滤）。
</details>

## 📝 课后作业

👉 [Assignment 17](../../../assignments/assignment_17/)

## 下一步

奖励设计（稀疏 vs 塑形 vs 课程）、稳定性（Echo Trap 深挖）、工业框架选型
（verl multi-turn / verl-agent / slime / rLLM）与评估（τ-bench/GAIA）。

👉 [02 — 奖励设计与工业框架](02_rewards_and_frameworks.md)
