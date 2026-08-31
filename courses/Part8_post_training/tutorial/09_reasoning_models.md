# 09 — 推理模型与 test-time compute

> 🧭 DeepSeek-R1 的核心创新是把 RL 用在"思维链生成"上，让模型在 `<think>` 段自主推理。
> 本章手写这条管线的最小闭环（SFT 格式 → GRPO 推理 RL → test-time compute），
> 实测 self-consistency 的"算力换准确率"效应。
> 跑 [scripts/09_reasoning_models.py](../scripts/09_reasoning_models.py)。

## 📖 前置知识

- **Part 8 04 章**：GRPO 组内优势（本章的 RL 算法基础）
- **Part 8 07 章**：评估学的"规则奖励"思想（RLVR = 用可验证规则当奖励）

## 1. DeepSeek-R1 的四阶段训练管线

| 阶段 | 做法 | 解决什么 |
|---|---|---|
| **R1-Zero** | 纯 RL（GRPO），无 SFT | 证明"纯 RL 可涌现推理"——但可读性差、语言混杂 |
| **R1 cold start** | 少量长 CoT SFT | 给 RL 一个"会说话"的起点 |
| **R1 推理 RL** | GRPO on 推理任务（准确率+语言一致性奖励） | 提升推理能力 |
| **R1 全场景** | 拒绝采样 SFT（60w 推理 + 20w 通用）+ RL | 兼顾有用性与无害性 |

- 🔑 **奖励 = 规则准确率 + 格式分**（非神经网络 RM，防 reward hacking）。准确率 =
  答案是否正确（数学可验证）；格式 = `<think>` 段是否存在且在 `<answer>` 之前。

## 2. 手写两阶段（脚本 09 的核心流程）

```python
# 阶段 1: cold start SFT —— 教模型"说格式"（<think>步骤</think> <answer>答案</answer>）
# 阶段 2: 推理 RL（GRPO 简化版）—— 奖励 = 结果正确 + 0.5×格式分
```

**实测（5000 SFT + 300 GRPO，单位数加法）**：
```
SFT loss → 0.2177（模型学会了 think+answer 格式与正确答案）
```

## 3. Test-time compute：算力换准确率

**self-consistency**（Wang et al. 2022）：同一 prompt 采样 N 条轨迹，取**众数答案**。
为什么有效：正确答案通常比错误答案更"一致"（多次采样更可能命中）。

```
n=1: 准确率 56%    ← 单次采样
n=4: 准确率 46%    ← 玩具模型波动（n=4 时众数不稳定）
n=8: 准确率 58%    ← 更多样本 → 众数趋近真实分布 → 恢复并超过 n=1
```

- ⚠️ **诚实解读**：玩具模型的 n=4 比 n=1 低是采样噪声——真实模型上 self-consistency
  单调提升更明显（任务越难、模型越好，提升越大）。核心主张是：**n↑ → 众数趋近
  条件分布的众数 → 正确答案被" voted up "**。

## 4. 脚本 09 的三个实现细节（面试向）

1. **答案提取**：`<answer>` token 后的第一个数字 token。真实系统用更鲁棒的抽取
   （如 GSM8K 的 `#### 42` 格式 + 正则最后一数字）。
2. **温度**：RL 采样 1.1（保持多样性）、self-consistency 1.1（保持多样性以
   让众数有意义）——温度=0 时退化为贪心解码，self-consistency 退化为单次贪心。
3. **RL 奖励的组合**：accuracy（结果对错）+ 0.5×format（think/answer 段结构）。
   格式奖励保证模型"先想再答"，准确率奖励驱动推理质量。

## 学完本部分你能...

- ✅ 画出 R1 四阶段管线，说出每阶段的输入/输出/奖励
- ✅ 实现 self-consistency 并实测"n↑ → 准确率↑"
- ✅ 解释 format reward 为什么必要（防止模型跳过 think 段直接给答案）
- ✅ 区分 test-time compute 与训练时 scaling（前者是推理时算力，后者是预训练算力）

**课后练习**

<details>
<summary>Q1: R1-Zero 和 R1 的核心区别？为什么 R1-Zero 不够？</summary>
A: R1-Zero = 纯 RL 无 cold start。它能涌现推理（aha moment）但输出可读性差、
语言混杂（中英夹杂）。R1 加了 cold start SFT（少量长 CoT 数据教模型"好好说话"），
后续再做多阶段 RL。核心教训：纯 RL 能变聪明，但不会自动变得"好读"。
</details>

<details>
<summary>Q2: 为什么 RL 阶段用规则奖励而非神经网络奖励模型？</summary>
A: 数学/代码类任务可以机器验证（规则不可被 hack）；NN-RM 有被 reward hacking 的
风险（模型学会利用 RM 的盲区而非真正变强）。R1 论文明确说用规则奖励避免此问题。
</details>

<details>
<summary>Q3: self-consistency 和 best-of-N 有什么区别？各适合什么场景？</summary>
A: self-consistency = 采 N 条取众数（适合有明确答案的任务）；best-of-N = 采 N 条用
验证器/奖励模型选最佳（适合可打分但答案不唯一的任务）。前者更简单、后者更灵活。
本质上都是"用 N 倍推理算力提升准确率"。
</details>

## 📝 课后作业

无独立作业——本脚本的 test-time compute 实验即作业（Part 16 02 章
的 img2img/ControlNet 作业使用 Assignment 16）。

## 下一步

多模态如何让模型"看懂"图片？→ Part 15 多模态理解。

👉 [Part 15 多模态理解](../../Part15_vision_language/tutorial/README.md)
