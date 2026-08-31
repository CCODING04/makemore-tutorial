# 01 — 从单轮 RLVR 到 Agentic RL

> 🧭 Part 8/11 的 GRPO：一个问题 → 一次回答 → 一个奖励（单轮）。Agentic RL 把游戏
> 改成了：**问题 → 调工具 → 看结果 → 再调 → …… → 最终答案 → 一个奖励**。
> 三个新问题随之而来：多轮轨迹怎么采、观测 token 要不要算 loss、稀疏的轨迹级奖励
> 怎么分配到每个 token。本章手写最小闭环并逐个回答（跑
> [scripts/01_toy_agent_grpo.py](../scripts/01_toy_agent_grpo.py)，CPU 20 秒）。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** 多轮工具调用 rollout（含观测 mask 与上下文重进）
- ✅ **实现** 轨迹级 GRPO（组内优势广播 + 观测 mask）
- ✅ **复刻** R1 两阶段的玩具版并解释每阶段的分工
- ✅ **说出** Agentic RL 与单轮 RLVR 的至少三个实现差异
- ✅ **识别** Echo Trap 等常见陷阱并设计防范策略

## 📖 前置知识

**必须掌握：**
- **Part 11**：GRPO 组内优势（本章轨迹级优势 = 它的广播版）
- **Part 12 01 章**：chat template（工具调用协议 = 它的扩展）

## 理论背景

### 问题引入：为什么需要 Agentic RL？

单轮 RL 虽然强大，但只能处理单轮问答：

1. **轮次限制**：无法处理多轮对话
2. **工具限制**：无法调用外部工具
3. **环境限制**：无法与环境交互

Agentic RL 通过**多轮工具调用**来弥补：

```
单轮 RL:    "一问一答"
Agentic RL: "多轮对话，调用工具，与环境交互"
```

> 💡 **类比**：单轮 RL 像是只会回答问题的人，Agentic RL 像是会使用工具的人。工具的能力让解决问题更高效。

### 数学推导：轨迹级 GRPO

**问题设定：**
- 轨迹：τ = (s_1, a_1, r_1, s_2, a_2, r_2, ..., s_T, a_T, r_T)
- 轨迹级奖励：R(τ)

**推导过程：**

```
Step 1: 采样轨迹
  从当前策略 π 采样 G 条轨迹：τ_1, τ_2, ..., τ_G

Step 2: 计算轨迹级奖励
  R(τ_i) = 最终奖励（如任务完成度）

Step 3: 计算优势
  A_i = (R(τ_i) - mean(R)) / std(R)

Step 4: 策略更新
  L = -Σ log π(a|s) * A
```

**关键洞察：**
- 轨迹级 GRPO 是单轮 GRPO 的自然扩展
- 观测 token 不参与 loss 计算（mask=0）
- 工具调用也需要学习（"学会调对工具"）

## 代码实现

### 1. 多轮轨迹的解剖

运行 [scripts/01_toy_agent_grpo.py](../scripts/01_toy_agent_grpo.py) 验证以下代码。

```
[user: 3 2 1]                                    ← 任务（mask=0）
  assistant:  <tool_call> multiply 3 2  <tool_call>   ← assistant 段（mask=1）
  <tool_result> 6 </tool_result>                      ← 观测（mask=0）⭐
  assistant:  <tool_call> add 6 1  <tool_call>          ← assistant 段（mask=1）
  <tool_result> 7 </tool_result>                      ← 观测（mask=0）
  assistant: 答案是 7 <eos>                            ← assistant 段（mask=1）
```

### 形状追踪：多轮轨迹

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  多轮轨迹数据流                                                              │
│                                                                             │
│  输入: user_message (str)                                                   │
│    ↓ tokenize                                                               │
│  user_ids: (n_user,)                                                        │
│    ↓ generate (到 <tool_call> 或 <eos>)                                     │
│  assistant_ids: (n_assistant,)                                              │
│    ↓ parse_call                                                             │
│  tool_call: (str)                                                           │
│    ↓ run_tool                                                               │
│  tool_result: (str)                                                         │
│    ↓ tokenize                                                               │
│  tool_ids: (n_tool,)                                                        │
│    ↓ 循环直到 <eos> 或 MAX_TURNS                                            │
│                                                                             │
│  最终轨迹:                                                                   │
│  ids = [user_ids, assistant_ids, tool_ids, assistant_ids, ...]              │
│  mask = [0, 1, 0, 1, ...]  # user=0, assistant=1, tool=0                   │
│                                                                             │
│  奖励分配:                                                                   │
│  R(trajectory) = 0 或 1                                                     │
│  A(每个 assistant token) = (R - mean) / std                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

- 🔑 **观测 mask=0 的三重理由**：① 工具输出是环境生成的，不是模型的"话"；② 模型
  无法预测外部结果，算 loss 是浪费容量；③ 更糟——模型可能学会"生成自己期望的
  观测"（幻觉工具结果）。

### 2. 手写多轮 rollout（脚本核心 ~50 行）

```python
for turn in range(MAX_TURNS + 1):
    # 自回归生成（到 <tool_call> 或 <eos>），token 记入轨迹、mask=1
    gen = generate_until(policy, cur, stop=["<tool_call>", "<eos>"])
    ids += gen; mask += [1] * len(gen)

    if "<tool_call>" in gen:
        call = parse_call(gen)                    # 解析失败 → 本轨迹终止
        obs = run_tool(call)                      # 环境执行
        ids += tokenize(obs); mask += [0] * ...   # 观测 mask=0（不是模型说的）
    else:
        break                                     # 给出最终答案，episode 结束
```

### 3. 轨迹级 GRPO：奖励怎么分配

整条轨迹只有末尾一个 0/1 奖励 → 组内标准化后，**优势广播到该轨迹的全部 assistant
token**（工具调用也要学——"学会调对工具"本身就是任务的一部分）：

```
A(每个 assistant token) = (r_trajectory − mean(组内 r)) / std
loss = −Σ (log π(token) × A(token)) / #assistant_tokens
```

### 4. 实测：BC 冷启动 → RL（0% → 85%）

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

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：观测 token 误算进 loss

**症状：**
```
loss 不下降，或模型学会"预测工具输出"
```

**原因：** 观测 token 没有正确 mask

**解法：**
```python
# 确保观测 token 的 mask=0
mask = [0] * n_user + [1] * n_assistant + [0] * n_tool + [1] * n_assistant + ...
```

#### 错误 2：解析失败导致 RL 死锁

**症状：**
```
奖励恒 0，训练不收敛
```

**原因：** 工具调用解析失败，奖励恒为 0

**解法：**
```python
# 检查解析是否成功
if not parse_success:
    reward = 0.0
    # 终止本轨迹，继续下一条
```

#### 错误 3：Echo Trap（策略熵坍缩）

**症状：**
```
奖励曲线看不出来，但 rollout 熵很低
```

**原因：** 策略熵坍缩到重复模板

**解法：**
```python
# 监控 rollout 的熵
entropy = -sum(p * log(p) for p in policy_distribution)

# 如果熵太低，使用 StarPO-S
# - critic + clip-higher + rollout 过滤
```

### 性能数据（实测参考）

| 方法 | 训练轮次 | 成功率 | 说明 |
|------|----------|--------|------|
| 随机策略 | 0 | 0% | 采不出合法格式 |
| BC 冷启动 | 100 | 0% | 只学会格式，没学会任务 |
| BC + RL | 200 | 85% | 格式 + 任务 |

> 📊 数据来源：本课开发机实测（2D 玩具任务）

### 常见陷阱

#### 陷阱 1：没有 BC 就没有 RL

**症状：** 训练不收敛，奖励恒为 0

**原因：** 随机策略采不出合法工具调用

**解法：** 先用 BC 冷启动，学会格式再 RL

#### 陷阱 2：观测 mask 错误

**症状：** 模型学会"预测工具输出"

**原因：** 观测 token 没有正确 mask

**解法：** 确保观测 token 的 mask=0

#### 陷阱 3：Echo Trap

**症状：** 策略熵坍缩到重复模板

**原因：** 多轮 RL 特有的稳定性陷阱

**解法：** 使用 StarPO-S（critic + clip-higher + rollout 过滤）

### 最佳实践

#### 配置推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| MAX_TURNS | 5-10 | 最大工具调用次数 |
| BC epochs | 100-200 | 冷启动训练轮次 |
| RL rounds | 100-200 | RL 训练轮次 |
| 组大小 G | 4-8 | 每个 prompt 的轨迹数 |

#### 调试流程

1. **先 BC 冷启动**：学会格式
2. **检查解析**：确保工具调用能正确解析
3. **监控熵**：确保策略不坍缩
4. **逐步增加难度**：从简单任务到复杂任务

## 学完本部分你能...

- ✅ 手写多轮工具调用 rollout（含观测 mask 与上下文重进）
- ✅ 实现轨迹级 GRPO（组内优势广播 + 观测 mask）
- ✅ 复刻 R1 两阶段的玩具版并解释每阶段的分工
- ✅ 说出 Agentic RL 与单轮 RLVR 的至少三个实现差异
- ✅ 识别 Echo Trap 等常见陷阱并设计防范策略

**概念检验**

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

**动手实践**

<details>
<summary>练习 1: 实现多轮 rollout</summary>

**任务：** 实现一个函数，生成多轮工具调用轨迹。

**验收标准：**
- [ ] 输入：策略、用户消息、最大轮次
- [ ] 输出：轨迹（ids, mask, reward）
- [ ] 正确处理观测 mask=0

**步骤提示：**
```python
def multi_turn_rollout(policy, user_message, max_turns=5):
    """
    Steps:
        1. tokenize 用户消息
        2. 循环 max_turns 次
        3. 生成 assistant 回复（到 <tool_call> 或 <eos>）
        4. 如果是 <tool_call>，解析并执行工具
        5. 如果是 <eos>，结束轨迹
        6. 返回轨迹（ids, mask, reward）
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 2: 实现轨迹级 GRPO</summary>

**任务：** 实现一个函数，计算轨迹级 GRPO 优势。

**验收标准：**
- [ ] 输入：轨迹列表，每条轨迹有 reward
- [ ] 输出：每条轨迹的 advantage
- [ ] 正确处理组内标准化

**步骤提示：**
```python
def trajectory_grpo(trajectories):
    """
    Steps:
        1. 提取每条轨迹的 reward
        2. 计算组内均值和标准差
        3. 计算每条轨迹的 advantage
        4. 返回 advantage
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 3: 实现 BC 冷启动</summary>

**任务：** 实现一个函数，用 BC 冷启动训练策略。

**验收标准：**
- [ ] 输入：策略、示范数据
- [ ] 输出：训练后的策略
- [ ] 学会格式但不一定学会任务

**步骤提示：**
```python
def bc_cold_start(policy, demonstrations, epochs=100):
    """
    Steps:
        1. 遍历示范数据
        2. 计算 BC loss（交叉熵）
        3. 更新策略
        4. 返回训练后的策略
    """
    # TODO: Implement
    pass
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 17 完成练习：

👉 [Assignment 17](../../../assignments/assignment_17/)

## 下一步

奖励设计（稀疏 vs 塑形 vs 课程）、稳定性（Echo Trap 深挖）、工业框架选型
（verl multi-turn / verl-agent / slime / rLLM）与评估（τ-bench/GAIA）。

👉 [02 — 奖励设计与工业框架](02_rewards_and_frameworks.md)
