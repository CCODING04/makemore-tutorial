# 04 — 强化学习：PPO 与 GRPO

> 🎮 DPO 是"离线"的——只用固定数据训练。PPO 和 GRPO 是"在线"的——模型自己生成回答、获得奖励、从"尝试"中学习。PPO 是经典方案，GRPO 是 DeepSeek-R1 的选择。

## 📖 前置知识

本章需要你已经掌握：

- **03 章全部**：Bradley-Terry 奖励模型、DPO 推导、forward_hidden()
- **概率论基础**：期望、方差、指数函数

> 💡 本章数学稍多，但每个公式都有直觉解释。不理解推导不影响读懂代码。

## 为什么 DPO 不够？

DPO 很好，但有一个根本限制：**它是离线的**。

| | DPO（离线） | PPO/GRPO（在线） |
|---|---|---|
| 数据来源 | 固定的偏好对 | 模型自己生成 |
| 学习方式 | 从"别人的回答"学 | 从"自己的回答"学 |
| 能否探索 | 不能 | 能 |
| 训练稳定性 | 高 | 较低（需要调参） |
| 理论上限 | 受限于数据质量 | 可以超越数据 |

💡 **直觉**：DPO 像是"看别人的考试答案学习"，PPO/GRPO 像是"自己做题、看对错、改进"。后者理论上更好，但更难训练。

## PPO 核心思想

PPO（Proximal Policy Optimization）是最流行的 RL 算法之一。ChatGPT 的 RLHF 阶段就用了 PPO。

### 策略梯度的直觉

最简单的策略梯度公式：

```
∇J = E[A * ∇log π(a|s)]
```

- `π(a|s)`：策略在状态 s 下选择动作 a 的概率
- `A`：advantage——这个动作比"平均"好多少
- `A > 0`：增大这个动作的概率
- `A < 0`：减小这个动作的概率

### 问题：更新太大容易崩

如果 advantage 很大，一步更新可能让策略剧变——从"经常选 A"变成"从不选 A"。这会导致训练不稳定。

### 解决：Clipped Surrogate

PPO 的核心创新——用 ratio clip 限制更新幅度：

```python
def ppo_policy_loss(new_logp, old_logp, advantages, mask, clip=0.2):
    ratio = torch.exp(new_logp - old_logp)        # 新旧策略比
    surr1 = ratio * advantages                      # 无截断
    surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages  # 截断后
    loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum())
    return loss
```

对应代码在 [06_ppo_training.py](../scripts/06_ppo_training.py)。

🔑 **ratio clip 的直觉**：

```
ratio = π_new(a|s) / π_old(a|s)
```

- `ratio ≈ 1`：策略变化很小，在"信任域"内，直接用 `ratio * A`
- `ratio >> 1`：策略变化太大，用 `(1+ε) * A` 截断，防止剧变
- `ratio << 1`：策略变化太大（反方向），用 `(1-ε) * A` 截断
- `ε = 0.2`：论文推荐值，实际效果最好

```
        loss
          ^
          |    /
          |   /
          |  /  ← 正常区域：ratio * A
          | /
   ───────┼/────────────→ ratio
          |\
          | \  ← 截断区域：(1+ε) * A
          |  \
          |   \
```

## GAE：估计 Advantage

PPO 需要计算 advantage——"这个动作比平均好多少"。GAE（Generalized Advantage Estimation）是最常用的方法。

### TD Error

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
```

- `r_t`：t 时刻的奖励
- `V(s_t)`：状态 s_t 的价值估计（"从这个状态开始，未来能拿多少奖励"）
- `γ`：折扣因子（控制"看多远"）

💡 **直觉**：`δ_t > 0` 意味着"这个动作比预期好"，`δ_t < 0` 意味着"比预期差"。

### GAE 公式

```
A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
```

从后往前递推：

```python
def compute_gae(rewards, values, values_next, resp_mask, gamma=1.0, lam=0.95):
    B, L = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, device=rewards.device)
    m = resp_mask.float()

    for t in reversed(range(L)):
        nonterminal = m[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * values_next[:, t] * nonterminal - values[:, t]
        lastgae = delta + gamma * lam * nonterminal * lastgae
        adv[:, t] = lastgae

    returns = adv + values
    return adv * m, returns * m
```

🔑 **γλ 权衡**：
- `γλ = 0`：只看单步 TD error δ_t → 高 bias（估计不准）、低 variance（稳定）
- `γλ = 1`：看完整轨迹 → 低 bias（更准）、高 variance（不稳定）
- `γ = 1.0, λ = 0.95`：实际中效果最好（LLM 通常 γ=1.0，因为没有"终止状态"）

## Value Head：估计 V(s)

GAE 需要 V(s)——状态价值函数。怎么得到？在 GPT backbone 上加一个"价值头"：

```python
class TransformerWithValueHead(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        n_embed = transformer.lm_head.in_features
        self.value_head = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)  # 零初始化
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, idx):
        hidden = self.transformer.forward_hidden(idx)
        logits = self.transformer.lm_head(hidden)    # Actor（策略）
        values = self.value_head(hidden).squeeze(-1)  # Critic（价值）
        return logits, values
```

架构图：

```
idx (B, T)
  |
  v
GPT.forward_hidden() → hidden (B, T, n_embed)
  |          |
  v          v
lm_head    value_head
  |          |
  v          v
logits     values
(Actor)    (Critic)
```

💡 **Actor-Critic 架构**：Actor（lm_head）决定"做什么"，Critic（value_head）评估"做得好不好"。两者共享 backbone，但各自有独立的头。

## PPO 训练循环

完整的 PPO 训练流程：

```
每一步:
  1. Rollout: 用当前策略生成 response
  2. Reward: 用奖励函数给 response 打分
  3. KL Penalty: 减去 policy vs ref 的 KL 散度（防止偏离太远）
  4. GAE: 计算 advantage 和 returns
  5. Multi-epoch update: 用 clipped surrogate 更新 N 个 epoch
```

对应代码的核心循环：

```python
for step in range(ppo_steps):
    # 1. Rollout: 生成 response
    with torch.no_grad():
        responses = [sft_model.generate(...) for _ in range(batch_size)]

    # 2. 计算 old log probs
    logits_old, values_old = ppo_model(sequences)
    old_log_probs = per_token_log_probs(logits_old, sequences)

    # 3. 奖励 + KL 惩罚
    rewards = compute_reward(sequences, prompt_len)
    kl = old_log_probs - ref_log_probs
    rewards = rewards - kl_coeff * kl

    # 4. GAE
    advantages, returns = compute_gae(rewards, values_old, ...)

    # 5. Multi-epoch update
    for epoch in range(ppo_epochs):
        logits_new, values_new = ppo_model(sequences)
        p_loss = ppo_policy_loss(new_logp, old_logp, advantages, ...)
        v_loss = ppo_value_loss(values_new, values_old, returns, ...)
        entropy = compute_entropy(logits_new, ...)
        total_loss = p_loss - entropy_coeff * entropy + 0.5 * v_loss
        total_loss.backward()
```

⚠️ PPO 每步都重新生成 response（on-policy），然后在同一组数据上做 N 个 epoch 的更新。这比 DPO 复杂得多——需要维护 policy、ref、value 三个模型。

## GRPO：不需要 Value Network

GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 的选择。核心创新：**不需要 Value Network**！

### 为什么可以去掉 Value Network？

PPO 需要 V(s) 来计算 advantage。GRPO 用一个更简单的方法：

**对同一个 prompt，采样 G 个回答，用组内平均奖励作基线。**

```python
def group_advantages(rewards, group_size, eps=1e-4):
    """A_i = (r_i - group_mean) / (group_std + eps)"""
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1)
```

对应代码在 [07_grpo_training.py](../scripts/07_grpo_training.py)。

🔑 **Group Advantage 的直觉**：

假设 prompt 是 `3+5=`，采样 G=4 个回答：

```
回答 1: "8"     → reward = 1.0（正确）
回答 2: "7"     → reward = 0.0（错误）
回答 3: "9"     → reward = 0.0（错误）
回答 4: "8"     → reward = 1.0（正确）

group_mean = 0.5
group_std  = 0.58
advantages: [+0.87, -0.87, -0.87, +0.87]
```

- 正确答案 advantage > 0 → 增大概率
- 错误答案 advantage < 0 → 减小概率

💡 **不需要 V(s)！** 组内统计量（mean、std）就是天然的 baseline。

### k3 KL 估计器

GRPO 用 Schulman 的 k3 估计器计算 per-token KL 散度：

```python
def k3_kl(new_logp, ref_logp):
    """KL = exp(log_ref - log_new) - (log_ref - log_new) - 1"""
    diff = ref_logp - new_logp
    return torch.exp(diff) - diff - 1.0
```

🔑 **k3 的三个优点**：
- 无偏（unbiased）：期望值等于真实的 KL 散度
- 非负：保证 KL >= 0（不需要 clamp）
- 数值稳定：不需要特殊处理

### GRPO Loss

```python
def grpo_loss(new_logp, old_logp, ref_logp, advantages, resp_mask, clip=0.2, kl_coef=0.04):
    adv = advantages[:, None]  # broadcast over tokens
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    surrogate = torch.min(surr1, surr2)
    kl = k3_kl(new_logp, ref_logp)
    per_token = surrogate - kl_coef * kl
    loss = -(per_token * resp_mask.float()).sum() / resp_mask.float().sum().clamp(min=1)
    return loss
```

和 PPO 的 clipped surrogate 几乎一样，只是 advantage 来自 group-relative 而非 GAE。

## RLVR：可验证奖励

GRPO 最常配合 RLVR（RL with Verifiable Rewards）使用：

```python
def verify_answer(predicted, expected):
    """数学题答案正确 = 1，错误 = 0"""
    numbers = re.findall(r'-?\d+\.?\d*', str(predicted))
    if not numbers:
        return False
    return abs(float(numbers[-1]) - float(expected)) < 0.01
```

🔑 **RLVR 的核心思想**：奖励来自可验证的规则，不需要训练 Reward Model。

| 奖励来源 | 例子 | 优点 | 缺点 |
|---------|------|------|------|
| 人类标注 | "这个回答好不好？" | 灵活 | 贵、慢、有偏差 |
| Reward Model | 训练好的打分模型 | 自动化 | 需要训练、可能 reward hacking |
| **RLVR** | 数学题答案对错 | **免费、可靠** | 只适用于可验证的任务 |

💡 DeepSeek-R1 用 RLVR 训练推理能力——数学题答案对错、代码测试通过/失败，都是可验证的。

## PPO vs GRPO 对比

| 维度 | PPO | GRPO |
|------|:---:|:---:|
| Value Network | 需要（value_head） | **不需要** |
| 额外参数 | ~50%（value_head） | 0 |
| 显存 | 更多（policy + value + ref） | 更少（policy + ref） |
| 优势估计 | GAE（γ, λ） | Group normalization |
| 采样方式 | 每步生成一次 | 每个 prompt 生成 G 次 |
| KL 惩罚 | per-token KL penalty | k3 KL estimator |
| 代表 | InstructGPT, ChatGPT | DeepSeek-R1 |

💡 **怎么选？**
- 有训练好的 Reward Model + 资源充足 → PPO
- 任务可验证（数学/代码）+ 想省资源 → GRPO
- 都不确定 → 先试 DPO（最简单）

## 课后练习

<details>
<summary>Q1: GAE 的 λ 如何控制偏差-方差折中？</summary>
A: λ 控制"看多远的 TD error"。λ=0 只看当前步的 δ_t（高 bias：V(s) 估计不准直接影响 advantage；低 variance：每步独立，不累积误差）。λ=1 看完整轨迹（低 bias：最终结果是最准的信号；高 variance：需要很多步才能得到 advantage，每步的噪声累积）。λ=0.95 是经验最佳值——主要看近期的 TD error，但也"稍微"考虑远期。
</details>

<details>
<summary>Q2: GRPO 为什么不需要 Value Network？</summary>
A: PPO 需要 V(s) 来计算 "这个动作比预期好多少"。GRPO 换了一种思路——不问"比预期好多少"，而问"比同组其他回答好多少"。同一个 prompt 采样 G 个回答，组内平均奖励就是天然的 baseline。这省掉了 Value Network（~50% 额外参数）和 GAE 递推计算，训练更简单、更省显存。
</details>

<details>
<summary>Q3: PPO 的 entropy bonus 有什么用？</summary>
A: entropy = -Σ π(a|s) * log π(a|s)。熵越大 → 策略越"随机" → 探索越多。如果不加 entropy bonus，策略可能很快收敛到"总是选同一个动作"（模式坍缩），错过更好的回答。entropy_coeff=0.01 是个很小的系数——只需要"轻微鼓励"探索，不要太多。
</details>

## 📝 课后作业

完成本章后，去 Assignment 8 完成题 7（GAE）和题 8（PPO Clipped Loss）：

👉 [Assignment 8](../../../assignments/assignment_8/)

## 下一步

训练完成后，如何评估模型质量？下一步我们用 GSM8K 数学题评估各阶段模型，对比生成质量，学习 temperature/top_k 等解码策略。

👉 [05 — 评估与推理部署](05_eval_and_deploy.md)
