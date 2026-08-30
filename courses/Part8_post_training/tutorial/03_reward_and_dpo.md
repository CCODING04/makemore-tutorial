# 03 — 奖励模型与对齐算法：DPO、ORPO、KTO

> 🎯 SFT 之后模型会对话，但质量参差不齐。本章引入 Bradley-Terry 奖励模型来量化"好 vs 坏"，然后用 DPO/ORPO/KTO 三种算法直接优化策略——不需要训练 RL。

## 📖 前置知识

本章需要你已经掌握：

- **02 章全部**：SFT、Chat Template、Prompt Masking、forward_hidden()
- **概率论基础**：sigmoid 函数、log-probability

> 💡 如果你忘了"sigmoid 是什么"，回忆一下：sigmoid(x) = 1/(1+e^{-x})，把任意实数映射到 (0, 1)，可以理解为"概率"。

## 为什么需要对齐？

SFT 之后，模型学会了"按指令回答"，但还有很多问题：

| 问题 | 例子 |
|------|------|
| 幻觉 | "Python 是由 Guido van Rossum 在 1989 年发明的"（年份可能错） |
| 不安全 | 生成有害内容、泄露隐私 |
| 不遵循偏好 | 用户喜欢简洁回答，模型却啰嗦一大堆 |
| 质量不稳定 | 同一个问题，有时回答好，有时回答差 |

🔑 **对齐（Alignment）的目标**：让模型的输出更符合人类的偏好——安全、有用、诚实。

## Bradley-Terry 偏好模型

对齐的第一步是量化"什么是好回答"。Bradley-Terry 模型是最经典的方法。

### 直觉

假设你有两个回答 A 和 B，让你选哪个更好。Bradley-Terry 假设：

```
你选 A 的概率 = sigmoid(r(A) - r(B))
```

其中 `r(A)` 是回答 A 的"潜在奖励"——我们看不到，但可以通过训练学到。

```python
P(A > B) = sigmoid(r(A) - r(B))
```

- `r(A) >> r(B)`：sigmoid → 1，你几乎肯定选 A
- `r(A) << r(B)`：sigmoid → 0，你几乎肯定选 B
- `r(A) = r(B)`：sigmoid → 0.5，你选谁都一样

### 训练损失

```python
def bradley_terry_loss(r_chosen, r_rejected):
    """L = -log sigmoid(r_chosen - r_rejected)"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

🔑 **直觉**：
- `r_chosen > r_rejected`：sigmoid → 1, loss → 0（正确排序，不需要更新）
- `r_chosen < r_rejected`：sigmoid → 0, loss → 很大（错误排序，强烈更新）
- `r_chosen = r_rejected`：sigmoid → 0.5, loss = ln(2)（无法区分）

⚠️ 用 `F.logsigmoid` 而不是 `log(sigmoid())`——前者数值更稳定（内部用 softplus 实现，避免 log(0)）。

## 奖励模型架构

怎么得到 `r(x)`？在 GPT backbone 上加一个"奖励头"：

```python
class RewardModel(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        n_embed = gpt.lm_head.in_features
        self.reward_head = nn.Linear(n_embed, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)  # 零初始化！

    def forward(self, idx):
        hidden = self.gpt.forward_hidden(idx)   # (B, T, n_embed)
        reward = self.reward_head(hidden)         # (B, T, 1)
        return reward[:, -1, 0]                   # (B,) — 取最后 token
```

对应代码在 [04_reward_model.py](../scripts/04_reward_model.py)。

架构图：

```
idx (B, T)
  |
  v
GPT.forward_hidden() → hidden (B, T, n_embed)
  |
  v
reward_head(hidden) → (B, T, 1)
  |
  v
取最后 token → r(x) (B,)
```

🔑 **三个关键设计**：

1. **只取最后 token**：Transformer 是因果模型，最后一个 token 的 hidden state 包含了整个序列的信息（通过注意力机制汇总）。这是 InstructGPT/ChatGPT 的标准做法
2. **零初始化**：`nn.init.zeros_(self.reward_head.weight)` —— 训练初期所有奖励 ≈ 0，意味着 `P(A > B) ≈ 0.5`（无偏好），符合直觉
3. **无 bias**：简化设计，效果差不多

## DPO 推导：从 RLHF 到分类问题

DPO（Direct Preference Optimization）是目前最流行的对齐算法。它的核心贡献是：**把复杂的 RL 问题转化成了简单的分类问题**。

### 起点：RLHF 目标

经典 RLHF 的目标是：

```
max E_{y~π}[r(x,y)] - β * KL(π || π_ref)
```

翻译成人话：让策略 π 生成的回答获得高奖励 r，但不要偏离参考模型 π_ref 太远（KL 散度惩罚）。

- `β` 控制"激进程度"：β 小 → 更激进地追求高奖励；β 大 → 更保守地保持接近参考模型

### 最优策略的闭式解

这个优化问题有闭式解：

```
π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)
```

其中 Z(x) 是归一化常数（确保概率之和为 1）。

### 反解奖励函数

把上式两边取 log，反解出奖励：

```
r(x,y) = β * log(π(y|x) / π_ref(y|x)) + β * log Z(x)
```

🔑 **关键洞察**：奖励可以用策略的 log-prob ratio 来表示！不需要显式的奖励模型。

### 代入 Bradley-Terry

把反解出的奖励代入 Bradley-Terry 的 `P(A > B) = sigmoid(r(A) - r(B))`：

```
r(A) - r(B) = β * [log(π(A)/π_ref(A)) - log(π(B)/π_ref(B))]
```

注意 `Z(x)` 在减法中消掉了！最终的 DPO loss：

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    # 隐式奖励 = beta * (log_pi - log_ref)
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    return loss, chosen_reward, rejected_reward
```

对应代码在 [05_dpo_alignment.py](../scripts/05_dpo_alignment.py)。

🔑 **DPO 的核心公式**：

```
L = -log sigmoid(β * [(log π_chosen - log π_rejected) - (log π_ref_chosen - log π_ref_rejected)])
```

直觉：
- 策略对 chosen 的 log-prob 越高、对 rejected 越低 → loss 越小
- 参考模型的 log-prob 作为 baseline 被减掉 → 只优化"策略比参考模型更偏好 chosen"的部分
- β 控制偏离参考模型的程度

### DPO 训练流程

```
1. 加载 SFT 模型作为 policy
2. 深拷贝 policy 作为 ref（冻结，不更新）
3. 准备 (chosen, rejected) 偏好对
4. 训练循环：
   a. 计算 policy 和 ref 对 chosen/rejected 的 log-prob
   b. 算 DPO loss
   c. 反向传播，只更新 policy
```

⚠️ ref 模型在整个训练过程中**保持不变**——它是"锚点"，防止 policy 跑太远。

## ORPO：无参考模型

ORPO（Odds Ratio Preference Optimization）的核心创新：**不需要参考模型**！

### 核心思想

DPO 需要一个冻结的 ref 模型来计算 baseline。ORPO 用 odds ratio 替代：

```
odds = P / (1-P)
log_odds = log P - log(1-P)
```

ORPO 的 loss：

```python
def orpo_loss(policy_chosen_logps, policy_rejected_logps,
              chosen_n_tokens, rejected_n_tokens, orpo_lambda=1.0):
    # 归一化：per-token 平均
    chosen_mean = policy_chosen_logps / chosen_n_tokens.clamp(min=1)
    rejected_mean = policy_rejected_logps / rejected_n_tokens.clamp(min=1)

    # log odds = log(p/(1-p))
    log_odds = (chosen_mean - _log1mexp(chosen_mean)) - \
               (rejected_mean - _log1mexp(rejected_mean))

    # OR loss = -log sigmoid(log_odds)
    or_loss = -F.logsigmoid(log_odds).mean()

    # NLL loss — 在 chosen 上做 SFT
    nll = -chosen_mean.mean()

    # 总 loss = SFT + 偏好
    loss = nll + orpo_lambda * or_loss
    return loss, chosen_mean, rejected_mean
```

🔑 **ORPO = SFT + 偏好对齐，一步完成**：
- `nll`：在 chosen 上做 SFT（教模型生成好的回答）
- `or_loss`：用 odds ratio 拉大 chosen 和 rejected 的差距

⚠️ `_log1mexp` 是数值稳定的 `log(1-exp(x))` 实现，避免 log(0)。

## KTO：无成对数据

KTO（Kahneman-Tversky Optimization）更进一步：**不需要成对数据**！

### 核心思想

基于 Kahneman & Tversky 的前景理论（Prospect Theory）：
- 人对"损失"比"收益"更敏感（损失厌恶）
- 只需要"好/坏"标签，不需要"哪个更好"的配对

```python
def kto_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps

    # KL baseline：所有 log-ratio 的均值
    kl = torch.cat([chosen_logratio, rejected_logratio]).mean().clamp(min=0).detach()

    # 非对称损失
    chosen_losses = 1.0 - torch.sigmoid(beta * (chosen_logratio - kl))
    rejected_losses = 1.0 - torch.sigmoid(beta * (kl - rejected_logratio))

    loss = (desirable_weight * chosen_losses).mean() + \
           (undesirable_weight * rejected_losses).mean()
    return loss
```

🔑 **前景理论的核心**：
- chosen：让 `log_ratio > kl`（超过 baseline 才有正收益）
- rejected：让 `log_ratio < kl`（低于 baseline 才能惩罚）
- `undesirable_weight` 可以设更大（比如 1.5~2.0），体现"损失厌恶"

## 三种算法对比

| 维度 | DPO | ORPO | KTO |
|------|:---:|:---:|:---:|
| 参考模型 | 需要（冻结） | **不需要** | 需要（冻结） |
| 成对数据 | 需要 | 需要 | **不需要** |
| 训练复杂度 | 中 | 低 | 低 |
| 核心公式 | log-prob ratio 差 | odds ratio | 前景理论 |
| 代表模型 | Zephyr, Tulu-2 | Llama-3, Qwen2 | 稀疏标注场景 |

💡 **怎么选？**
- 有成对偏好数据 + 想要稳定 → DPO
- 有成对偏好数据 + 想省显存（不需要 ref 模型）→ ORPO
- 只有"好/坏"标签、没有配对 → KTO

## 隐式奖励：从 DPO 中提取

DPO 不需要显式的奖励模型，但我们可以从训练好的 DPO 策略中"提取"隐式奖励：

```
r(x,y) = β * (log π(y|x) - log π_ref(y|x))
```

这个隐式奖励可以用来监控训练效果——chosen 的隐式奖励应该逐渐高于 rejected。


> 📚 **延伸对照（LLMs-from-scratch）**：rasbt ch07 的 `04_preference-tuning-with-dpo` 用 Llama 3.1 70B **生成偏好数据**再从零
> 写 DPO——与我们"规则造偏好对"互补，想看真实偏好数据怎么来就读它。

## 课后练习

<details>
<summary>Q1: DPO 的 β 为什么不能太大？</summary>
A: β 控制"偏离参考模型的程度"。β 太大意味着 KL 惩罚很重，策略被"锁死"在参考模型附近，学不到新东西。β 太小则策略可能偏离太远，生成质量反而下降（"reward hacking"）。实际中 β 在 0.1~0.5 之间调优。
</details>

<details>
<summary>Q2: ORPO 为什么不需要参考模型？</summary>
A: DPO 需要 ref 模型来提供 baseline（"参考模型对 chosen/rejected 的偏好是什么"），然后优化"策略比参考模型更偏好 chosen"。ORPO 用 odds ratio 替代了这个 baseline——odds ratio 是 chosen 和 rejected 之间的直接比较，不需要外部参考点。同时 ORPO 的 NLL 项（在 chosen 上做 SFT）提供了"生成好回答"的信号，两者合起来完成了 SFT + 对齐。
</details>

<details>
<summary>Q3: Bradley-Terry 模型假设了什么？有什么局限？</summary>
A: Bradley-Terry 假设"偏好是传递的"——如果 A > B 且 B > C，则 A > C。但人类偏好并不总是传递的（比如你可能觉得 A 的风格比 B 好，B 的内容比 C 好，C 的风格比 A 好）。此外 Bradley-Terry 只建模了"二选一"的场景，无法直接处理"打分"（1~5 分）或多选一。
</details>

## 📝 课后作业

完成本章后，去 Assignment 8 完成题 4（Bradley-Terry）、题 5（DPO）、题 6（ORPO）：

👉 [Assignment 8](../../../assignments/assignment_8/)

## 下一步

DPO 是"离线"算法——只用固定的偏好数据，不能从"尝试"中学习。下一步我们引入 PPO（在线 RL）和 GRPO（不需要 Value Network 的 RL），让模型能从自己的生成中不断改进。

👉 [04 — 强化学习：PPO 与 GRPO](04_ppo_and_grpo.md)
