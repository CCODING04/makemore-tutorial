# Assignment 11：对齐实战（verl）

> 对应 Part 11 教程（[01 概念桥接](../../courses/Part11_alignment_verl/tutorial/01_handwritten_to_verl.md) / [02 quickstart](../../courses/Part11_alignment_verl/tutorial/02_verl_quickstart.md)）。
> 三题纯 Python；实验题在 Docker 里跑。

## 题目（实现 `alignment_exercises.py`）

1. **稳健奖励函数**（40 分）：boxed → #### → 最后一个数字的三级抽取链 + 千分位/尾点健壮性。
   ⚠️ 注意 "The answer is 100, no wait, 7." 应取最后的 7——自我纠正场景是真实的
2. **组内优势 + 全同组**（30 分）：`A_i=(r_i-mean)/std`，全同组 eps 兜底为全 0；
   并找出"全同奖励组"（无梯度、浪费的 rollout）
3. **KL 惩罚预算**（30 分）：k3 估计器 + "超预算提前停"护栏（策略漂移监控）

## 实验题（Docker，02 章）

- 跑通 quickstart 的 0.5B PPO → 换 `adv_estimator=grpo`，记录：显存峰值变化（省掉 critic）
- 自定义奖励函数（改 1 处规则）并观察 reward hacking：例如只奖励"答案里含数字"会发生什么

## 🎯 面试直通车

- "GRPO 和 PPO 的本质区别？"——基线来源：组内平均 vs 学习出的 Value（省一整个模型的训练状态）
- "RLVR 为什么香？什么时候不行？"——可机器验证且不可作弊；创意/对话类仍需 RM
- "rollout 为什么是 RL 的瓶颈？verl 怎么解？"——生成远贵于训练；双引擎 + HybridEngine 权重回同步
- "你们 RL 的 KL 怎么算？"——k3 估计器（exp(d)-d-1，恒非负低方差）+ 预算护栏
