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

---

## 🎯 面试直通车（话术卡：结论 → 原理 → 边界）

> 每张卡按"总分总"组织：先一句话结论压场，再两三句原理支撑，最后一句边界/代价收尾——面试答题的固定骨架。

**Q1："SFT 为什么要做 prompt masking？mask 与不 mask 两种口径的 loss 能直接比吗？"**

- **结论**：mask 是为了让梯度只流过 response 区域，把模型容量全部花在"学会回答"上；两种口径的 loss 数值不可直接比较。
- **原理**：unmasked（预训练口径）对全序列算 CE，模型会花容量去预测已知的 prompt，甚至学会"复述指令/自问自答"（作业 7 实验 2 的现象级证据）；masked（SFT 口径）只算 $L = \sum \mathrm{CE} \cdot m / \sum m$。教程 02 章实测：masked loss 通常比 unmasked **更大**——因为只看"难的部分"（回答），不看"容易的部分"（复制 prompt）。
- **边界**：mask 必须随 shift 同步对齐（`loss_mask[:, 1:]`，用位置 $t$ 的 logits 预测 $t+1$ 的 token），且分母要 `clamp(min=1)` 防空 response 除零。

**Q2："写出 DPO 的损失函数，β 起什么作用？"**

- **结论**：$L = -\log \sigma\big(\beta[(\log \pi_c - \log \pi_r) - (\log \pi_{ref,c} - \log \pi_{ref,r})]\big)$，β 控制对冻结参考模型的信任度——越大越保守。
- **原理**：从 RLHF 目标 $\max \mathbb{E}[r] - \beta \cdot \mathrm{KL}(\pi \| \pi_{ref})$ 出发，最优策略有闭式解，反解出 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$；代回 Bradley-Terry 偏好损失时 $Z(x)$ 在减法中消掉，奖励模型被"消掉"（教程 03 章完整推导）。Sanity check：$\pi = \pi_{ref}$ 时 loss $= \ln 2 \approx 0.693$（作业 8 题 5 验收标准）。
- **边界**：β 太大策略被锁死在 ref 附近学不到东西，β 太小容易偏离过远而退化（reward hacking）；教程经验区间 0.1~0.5，minimind 官方 β=0.15 且 lr 4e-8（建议 ≤5e-8 防遗忘）；DPO 是离线算法，不能从"尝试"中学习。

**Q3："GAE 里的 λ 和 γ 分别控制什么？"**

- **结论**：$\gamma\lambda$ 是 bias-variance 旋钮：趋 0 退化为单步 TD error（高 bias、低 variance），趋 1 退化为 Monte Carlo（低 bias、高 variance）。
- **原理**：先算 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，再从后往前递推 $A_t = \delta_t + \gamma\lambda A_{t+1}$（作业 8 题 6）；等价形式 $A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$，指数衰减地加权多步 TD error。课程默认 $\gamma = 1.0$、$\lambda = 0.95$——LLM 生成没有"终止状态"，γ 取 1（教程 04 章）。
- **边界**：GAE 需要价值头提供 $V(s)$，PPO 因此要同时维护 policy/ref/value 三个模型——这正是 GRPO 砍掉价值网络的动机。

**Q4："GRPO 怎么做到不需要价值网络？k3 估计器是什么？"**

- **结论**：用同一 prompt 下 G 个回答的组内均值当 baseline、组内标准化当优势，价值网络被"组统计量"替代；KL 用 Schulman 的 k3 估计器。
- **原理**：组内优势 $A = (r - \mathrm{mean}) / (\mathrm{std} + \mathrm{eps})$（按 $(num\_prompts, group\_size)$ reshape 后标准化，作业 8 题 8），同组回答共享 baseline，无需学 $V(s)$——这是 DeepSeek-R1 的选择。k3 估计器 $\mathrm{KL} = e^{d} - d - 1$（$d = \log p_{ref} - \log p_{new}$）三个优点：无偏、恒非负（无需 clamp）、数值稳定（教程 04 章）；课程超参 clip=0.2、kl_coef=0.04。
- **边界**：组内奖励全相同时 std=0、优势全 0，该 prompt 学不到东西；每个 prompt 要采 G 个回答，采样板推理成本乘 G。

**Q5："PPO 的 clip 裁剪在裁什么？"**

- **结论**：裁的是重要性采样比率 $\mathrm{ratio} = \exp(\log p_{new} - \log p_{old})$ 的偏离幅度，防止单次策略更新过大。
- **原理**：$\mathrm{surr2} = \mathrm{clamp}(\mathrm{ratio}, 1-\epsilon, 1+\epsilon) \cdot A$，取 $L = -\mathrm{mean}(\min(\mathrm{surr1}, \mathrm{surr2}))$（作业 8 题 7）；ratio 在 $[1-\epsilon, 1+\epsilon]$ 内时 loss 不变，超出才被截断。Sanity check：ratio=1（还没更新）时 $L = -\mathrm{mean}(A)$；优势为正鼓励增大概率、为负鼓励减小概率。课程与 GRPO 沿用 $\epsilon = 0.2$。
- **边界**：裁剪只限制更新幅度、不改变优化方向；ratio 偏离过远的样本梯度被截断，等效于"一次数据只敢用好几次、每次只挪一小步"。
