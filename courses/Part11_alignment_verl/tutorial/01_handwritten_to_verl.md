# 01 — 从手写 GRPO 到 verl：概念桥接

> 🧭 你在 Part 8 已经手写过 GRPO：**采样 G 个回答 → 打分 → 组内标准化优势 → clip 更新**。
> verl 把同一套数学放进工业引擎：rollout 用 vLLM/SGLang、训练用 FSDP2、编排用 Ray。
> 本章先手写这条管线的**最小可验证件**（跑 [scripts/01_reward_and_bridge.py](../scripts/01_reward_and_bridge.py)），
> 再给出逐概念的"手写 ↔ verl"映射表——02 章进 Docker 时你不会迷路。

## 📖 前置知识

- **Part 8 04 章**：GRPO 的组内优势与 KL 惩罚（本章的直接上游）
- **Part 8 07 章**：评估学（RLVR = 用评估学里的"规则评估"当奖励）

## 1. 手写三件套（脚本 01，CPU 可跑）

**① 可验证奖励函数**（verl quickstart 里你唯一必须自己写的代码）：

```python
def gsm8k_reward(response, ground_truth):
    # 抽取顺序（工程惯例）：\boxed{} → '#### 42' → 最后一个数字
    ...  # 对则 1.0 否则 0.0
```

**② 组内优势**：`A_i = (r_i - mean) / std`——实测两件事：
- 组内全对（全 1.0）→ 优势**全 0**：GRPO 天然跳过已掌握的题（"太简单的题没有梯度"）；
- 4 个回答 3 错 1 对 → 对的那个拿 +1.73，其余 -0.58。

**③ k3 KL 估计器**：`exp(d) - d - 1`（恒非负、低方差）——verl 的 KL 惩罚即此形态。

## 2. 手写 ↔ verl 概念映射表（本章核心产出）

| 你手写的（Part 8 04 章） | verl 里的对应 | 说明 |
|---|---|---|
| 玩具模型同时干生成+训练 | **actor_rollout_ref 三个角色** | 训练引擎（FSDP2/Megatron）与推理引擎（vLLM/SGLang）分离 |
| `for step: 采样 G 个回答` | rollout 的 `generate_sequences` | 大 batch 并行生成（这是 RL 训练最贵的阶段） |
| 手动把新权重"告诉"生成器 | **权重回同步（weight sync）** | 3D-HybridEngine 消除 train↔rollout 转换的显存冗余（verl 的招牌） |
| `group_advantages()` | `adv_estimator=grpo` 配置 | 数学一样；配置行替代你的 20 行 |
| k3 KL / ref 模型 | ref policy 角色 + KL 惩罚系数 | ref 是 SFT 模型的冻结副本 |
| `gsm8k_reward()` | **custom reward function** | quickstart 里你唯一必写的代码（RLVR 入口） |
| 单进程 for 循环 | **Ray 单控制器数据流** | 工作器角色化、资源 placement 可编程 |

- 🔑 **一句话理解 verl**：它没有发明新算法——**算法（GRPO/PPO/DAPO…）是配置项，
  基建（rollout 引擎、权重同步、分布式）才是它的本体**。这就是"手写一遍再上工具"的
  价值：你看得懂配置背后的每一段代码在干什么。

## 3. 为什么工业版必须"两个引擎"

手写版玩具模型 1 秒能生成 100 个回答；真实 7B 模型生成一个回答要几百 ms——
rollout 占 RL 训练时间的大头。vLLM/SGLang 的高吞吐生成（Part 14）+ FSDP 训练（Part 10）
各用最强的引擎，代价是**每次更新后要把新权重搬进推理引擎**（大模型上这是 GB 级拷贝）。
verl 的 HybridEngine 用 重分片+原地转换 把这一步的开销压到最低——这就是 02 章跑起来后
日志里 `sync_rollout_weights` 那一行干的事。

## 学完本部分你能...

- ✅ 手写 RLVR 奖励函数（\boxed / #### / 最后数字的抽取链）
- ✅ 解释 GRPO"全对组优势全零"现象及其含义
- ✅ 画出"手写 GRPO 循环 → verl 三角色 + 权重同步"的映射图
- ✅ 说清 RL 训练为什么需要两个引擎

**课后练习**

<details>
<summary>Q1: 为什么规则奖励（RLVR）比训练一个奖励模型更受青睐？什么时候不能用？</summary>
A: 不可作弊（答案可机器验证）、无 RM 偏差、无 RM 被奖励黑客的风险。不能用：答案不可
形式化的任务（创意写作、对话质量）——那才需要 RM/LLM-as-judge（Part 8 07 章）。
</details>

<details>
<summary>Q2: 组内标准化的基线和 PPO 的 Value 网络基线各有什么问题？</summary>
A: GRPO 组内基线：同组样本太少时噪声大；全对/全错组无梯度（浪费算力——可用课程难度
 Curriculum 缓解）。PPO Value 基线：要多训一个网络（显存+不稳定性），但能给单样本基线。
 DAPO/DrGRPO 等 recipe（verl 的生产配方目录，02 章详述） 在修这些边角（02 章读 recipe/dapo）。
</details>

## 📝 课后作业

👉 [Assignment 11](../../../assignments/assignment_11/)

## 下一步

进 Docker，把 0.5B 模型的 GRPO 真正跑起来。

👉 [02 — verl 快速上手：0.5B GRPO 实战](02_verl_quickstart.md)
