# 作业 8：从零训练 LLM —— 后训练全流程

> **对应教程**：Part 8 — 从零训练 LLM（GPT-2 架构 → Pretrain → SFT → Reward → DPO/PPO/GRPO）
>
> **前置**：建议先完成作业 6（Transformer/GPT，理解注意力与残差时需对照）

---

## 📋 概述

本作业带你从零实现 LLM 后训练全流程的八个关键组件。它们覆盖了从架构
基础到强化学习对齐的完整链路：

1. Causal Self-Attention Head（因果自注意力头）
2. Pre-LN Transformer Block（Pre-LN 残差块）
3. Prompt-Masked SFT Loss（提示词遮罩的 SFT 损失）
4. Bradley-Terry Reward Loss（Bradley-Terry 奖励损失）
5. DPO Loss（直接偏好优化损失）
6. 🌟 GAE Advantage Estimation（广义优势估计）
7. 🌟 PPO Clipped Loss（PPO 裁剪策略损失）
8. 🌟 GRPO Group Advantage（GRPO 组相对优势）

完成本作业后，你应该能够：

- 理解因果注意力如何通过 causal mask 保证自回归性质
- 理解 Pre-LN 为什么比 Post-LN 训练更稳定
- 理解 SFT 的 prompt masking 如何避免模型学习"复述提示词"
- 理解 Bradley-Terry 偏好模型如何将人类偏好转化为奖励信号
- 理解 DPO 如何绕过奖励模型直接优化偏好
- 理解 GAE 如何在 bias 和 variance 之间折中优势估计
- 理解 PPO 的裁剪机制如何防止策略更新过大
- 理解 GRPO 如何用"组内标准化"替代价值网络

---

## 🔧 环境准备

### 依赖

```bash
pip install torch
```

### 文件结构

```
assignments/assignment_8/
├── assignment.md                    # 本文件
├── post_training_exercises.py       # 👈 你需要编辑的文件
└── test_post_training_exercises.py  # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_8
python test_post_training_exercises.py
```

未实现的题目会显示为"跳过"（SKIP），不会报错。

---

## 题目

### 题 1：Causal Self-Attention Head（基础）

**文件**：`post_training_exercises.py` → `class Head`

实现单头因果注意力。这是 Transformer 的核心组件——每个 token 只能关注
自己和之前的 token，不能"偷看"未来。

**要求**：
- Q/K/V 三个线性投影（无 bias）
- scaled dot-product：`Q @ K^T / sqrt(d_k)`
- causal mask：上三角填 `-inf`，下三角保留
- softmax 归一化后加权 V

**提示**：
- `torch.tril(torch.ones(T, T))` 生成下三角矩阵
- `masked_fill(mask == 0, float('-inf'))` 遮罩未来位置
- 用 `register_buffer` 注册 mask（不算参数，但随 `.to(device)` 移动）

**验证标准**：
- 输出 shape 为 `(B, T, head_size)`
- 因果性：改变 token `t` 的输入不应影响 token `t-1` 的输出

---

### 题 2：Pre-LN Transformer Block（基础）

**文件**：`post_training_exercises.py` → `class Block`

实现 Pre-LN 残差块。这是 GPT-2 的标准 block 设计：先 LayerNorm 再进子层，
比 Post-LN（原始论文）训练更稳定。

**要求**：
- 两个子层：注意力 + MLP
- 每个子层：`x = x + sublayer(LN(x))`（Pre-LN 模式）
- MLP：`Linear → ReLU → Linear`（4x 扩展）

**提示**：
- 残差连接保证梯度直通
- `head_size = n_embed // n_head`

**验证标准**：
- 输出 shape 与输入一致 `(B, T, n_embed)`

---

### 题 3：Prompt-Masked SFT Loss（基础）

**文件**：`post_training_exercises.py` → `function sft_loss`

实现带 prompt 遮罩的 SFT 损失。普通预训练在所有 token 上算 loss，
但 SFT 只应在 **assistant 回复** 上训练，不应学习"复述提示词"。

**要求**：
- 标准 next-token cross-entropy
- 只在 `loss_mask=1` 的位置计算损失
- 最终 loss 除以 mask 中 1 的数量（取平均）

**提示**：
- 需要 shift：`logits[:, :-1, :]` 预测 `tokens[:, 1:]`
- mask 也要相应 shift：`loss_mask[:, 1:]`
- 公式：`L = sum(CE * mask) / sum(mask)`

**验证标准**：
- 全 mask 为 0 时 loss = 0
- 全 mask 为 1 时等价于普通 CE

---

### 题 4：Bradley-Terry Reward Loss（基础）

**文件**：`post_training_exercises.py` → `function reward_loss`

实现 Bradley-Terry 偏好模型的奖励损失。这是 RLHF 的基础：
假设人类偏好遵循 `P(A > B) = sigmoid(r(A) - r(B))`。

**要求**：
- `L = -log sigmoid(r_chosen - r_rejected)`
- 等价于 `L = -logsigmoid(r_chosen - r_rejected).mean()`

**提示**：
- `F.logsigmoid(x)` 数值稳定
- 当 `r_chosen = r_rejected` 时，`loss = ln(2) ≈ 0.693`

**验证标准**：
- 相等时 loss ≈ 0.693
- `r_chosen >> r_rejected` 时 loss → 0
- `r_chosen << r_rejected` 时 loss → ∞

---

### 题 5：DPO Loss（基础）

**文件**：`post_training_exercises.py` → `function dpo_loss`

实现 DPO（Direct Preference Optimization）损失。DPO 的核心洞察：
把 Bradley-Terry 模型中的 reward 用 policy log-prob 表示后，
reward function 被消掉，得到只依赖 policy 和 reference 的闭式解。

**要求**：
- 输入：policy 和 reference 对 chosen/rejected 的 log-prob（标量，已求和）
- `logits = (pi_ch - pi_rej) - (ref_ch - ref_rej)`
- `loss = -logsigmoid(beta * logits).mean()`

**提示**：
- 当 `pi == ref`（policy 还没训练）时，`logits = 0`，`loss = ln(2)`
- `beta` 控制偏离 reference 的惩罚强度（越大越保守）

**验证标准**：
- `pi == ref` 时 loss ≈ 0.693
- policy 对 chosen 的 log-prob 相对更高时 loss 下降

---

### 题 6：GAE Advantage Estimation（🌟 拓展）

**文件**：`post_training_exercises.py` → `function gae`

实现 GAE（Generalized Advantage Estimation）。PPO 用 GAE 在 TD error 的
低方差和 Monte Carlo 的低 bias 之间折中。

**要求**：
- 从后往前递推：`δ_t = r_t + γ * V(s_{t+1}) - V(s_t)`
- `A_t = δ_t + γ * λ * A_{t+1}`（λ 控制 bias-variance 折中）
- `λ=0` 退化为单步 TD error，`λ=1` 退化为 Monte Carlo

**提示**：
- 只在 response 位置计算（prompt 位置 advantage=0）
- 用 `lastgae` 变量从后往前累积

**验证标准**：
- `λ=0` 时 `A_t = r_t + γ * V(s_{t+1}) - V(s_t)`
- 输出 shape 与输入一致

---

### 题 7：PPO Clipped Loss（🌟 拓展）

**文件**：`post_training_exercises.py` → `function ppo_loss`

实现 PPO 的裁剪策略损失。这是 PPO 的核心创新：用 ratio 裁剪
防止策略更新幅度过大。

**要求**：
- `ratio = exp(logp_new - logp_old)`
- `surr1 = ratio * advantages`
- `surr2 = clamp(ratio, 1-ε, 1+ε) * advantages`
- `loss = -mean(min(surr1, surr2))`

**提示**：
- ratio 在 `[1-ε, 1+ε]` 内时，`surr2 = surr1`，loss 不变
- ratio 超出范围时被裁剪，防止过大更新

**验证标准**：
- `logp_new == logp_old`（ratio=1）时，`loss = -mean(advantages)`
- 优势为正时鼓励增大概率，优势为负时鼓励减小概率

---

### 题 8：GRPO Group Advantage（🌟 拓展）

**文件**：`post_training_exercises.py` → `function group_advantages`

实现 GRPO 的组相对优势。GRPO 的核心创新：用同一 prompt 下多个回答的
组内均值和标准差来标准化奖励，不需要价值网络。

**要求**：
- 输入：rewards 形状 `(num_prompts * group_size,)`，按组连续排列
- reshape 成 `(num_prompts, group_size)` 后组内标准化
- `adv = (r - group_mean) / (group_std + eps)`
- reshape 回一维

**提示**：
- 每个 prompt 的 G 个回答共享同一个 baseline（组均值）
- 组内标准差为 0 时（所有回答奖励相同），advantage 全为 0

**验证标准**：
- 每组内 advantage 均值为 0
- 每组内 advantage 标准差为 1（eps 可忽略时）

---

## 📊 评分标准

| 题目 | 类型 | 分值 |
|------|------|------|
| 题 1：Causal Head | 基础 | 12 分 |
| 题 2：Pre-LN Block | 基础 | 12 分 |
| 题 3：SFT Loss | 基础 | 12 分 |
| 题 4：Reward Loss | 基础 | 12 分 |
| 题 5：DPO Loss | 基础 | 12 分 |
| 题 6：GAE | 拓展 | 15 分 |
| 题 7：PPO Loss | 拓展 | 15 分 |
| 题 8：GRPO Advantage | 拓展 | 10 分 |
| **总计** | | **100 分** |

---

## 💡 学习建议

1. **先读教程**：每道题对应教程中的一个核心概念，建议先理解原理再动手
2. **对照脚本**：每道题的实现都在 `courses/Part8_post_training/scripts/` 中有参考
3. **先基础后拓展**：前 5 题是后训练的必备知识，后 3 题是 PPO/GRPO 的核心
4. **数值稳定**：涉及 log-prob 时用 `F.logsigmoid` 或 `F.log_softmax`，避免手动 `log(sigmoid)`
5. **shape 检查**：每写完一个函数，先打印 shape 确认维度正确

---

## 🧪 实验/观测题（对应教程 06 章「推理与服务」、07 章「评估学」——观测型，不进自动测试）

**实验 A：量化实测（06 章 / 脚本 09）**
跑 `courses/Part8_post_training/scripts/09_quantize_and_serve.py`，记录三行量化数字
（fp32 基线 / int8 逐通道 / int4 g128 的 ppl 与 Δ），并回答：
- 为什么本课 2M 模型的 int8 Δ（约 +0.4）比 7B 论文（<0.05）大？
- 把训练步数 500 改成 1000 再跑，Δ 变大还是变小？为什么？

**实验 B：投机解码扫描（06 章 / 脚本 09）**
把 `speculative_decode` 的 gamma 从 4 改成 2 和 8 各跑一次，记录 α 与
"target 前向次数 / 生成 token 数"。对照公式 `(1-α^(γ+1))/(1-α)` 验证：
γ 增大时单周期产出上限提高，但每个周期的 draft 成本也线性上升——找到你模型上的最优 γ。

**观测题 C：评估污染审查（07 章）**
一份报告声称"我们的 7B 模型 GSM8K 拿到 92 分"。列出你开口追问的 3 个问题：
（提示方向：训练数据与测试集的 n-gram 重叠检查做了吗？GSM1k 式镜像集掉多少分？
92 分的抽取规则是什么——`#### ` 后数字还是"最后一个数字"？）
对照 07 章 §4 的 GSM1k 证据（Mistral −8%、Phi −21%）组织你的答案。
