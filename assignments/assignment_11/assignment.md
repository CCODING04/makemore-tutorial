# Assignment 11: 对齐实战 — verl 工业级 RL 后训练

> 本作业基于 Part 11 的教程内容，帮助你巩固 RL 后训练的核心概念。
> 三题纯 Python；实验题在 Docker 里跑。

## 学习目标

完成本作业后，你将能够：

- ✅ 手写 RLVR 奖励函数（\boxed / #### / 最后数字的抽取链）
- ✅ 实现 GRPO 的组内优势计算
- ✅ 理解 KL 惩罚的作用
- ✅ 设计防作弊的奖励函数

## 练习

### 练习 1: 稳健奖励函数（40 分）

实现一个 GSM8K 风格的规则奖励函数，从模型回答中抽取数字并判断对错。

**验收标准：**
- [ ] 正确抽取 `\boxed{42}` 格式的答案
- [ ] 正确抽取 `#### 42` 格式的答案
- [ ] 正确抽取最后一个数字
- [ ] 处理千分位逗号（如 "1,234"）
- [ ] 处理浮点数精度问题（用 `abs(pred - gt) < 1e-4`）
- [ ] 处理自我纠正场景（如 "The answer is 100, no wait, 7." 应取最后的 7）

**抽取顺序（工程惯例）：**
```python
# 1. \boxed{} 优先
# 2. '#### 42' 标记
# 3. 最后一个数字
```

**步骤提示：**
```python
def robust_reward(response: str, ground_truth: str) -> float:
    """
    Steps:
        1. 用正则抽取 \boxed{} 中的内容
        2. 如果没有，尝试抽取 '#### 42' 格式
        3. 如果都没有，抽取最后一个数字
        4. 处理千分位逗号和尾点
        5. 比较预测值和真实值
    """
    # TODO: Implement
    return None
```

### 练习 2: 组内优势 + 全同组（30 分）

实现 GRPO 的组内优势计算：`A_i = (r_i - mean) / std`

**验收标准：**
- [ ] 正确计算组内均值和标准差
- [ ] 优势之和为 0（数值精度允许 1e-6 误差）
- [ ] 处理全对/全错组（std = 0 时优势全为 0）
- [ ] 使用 eps 防止除零
- [ ] 找出"全同奖励组"（无梯度、浪费的 rollout）

**数学推导：**
```
mean = (1/G) * Σ r_i
std = sqrt((1/G) * Σ (r_i - mean)^2)
A_i = (r_i - mean) / std

性质：
- Σ A_i = 0（优势之和为零）
- 如果所有 r_i 相同，则 std = 0，所有 A_i = 0
  → "太简单的题没有梯度"
```

**步骤提示：**
```python
def group_advantages(rewards_per_prompt, eps=1e-6):
    """
    Steps:
        1. 遍历每个 prompt 的 G 个回答的奖励
        2. 计算组内均值 mean
        3. 计算组内标准差 std
        4. 如果 std < eps，返回全 0（全同组）
        5. 否则计算 A_i = (r_i - mean) / std
        6. 验证 Σ A_i = 0
    """
    # TODO: Implement
    return None
```

### 练习 3: KL 惩罚预算（30 分）

实现 k3 KL 估计器：`exp(d) - d - 1`，其中 `d = logp_ref - logp_new`

**验收标准：**
- [ ] 正确计算 KL 散度
- [ ] 结果恒非负（因为 `e^x ≥ x + 1`）
- [ ] 当两个分布相同时，KL = 0
- [ ] 实现"超预算提前停"护栏（策略漂移监控）

**数学推导：**
```
KL(q || p) = E_q[log(q/p)] = E_q[log q - log p]
令 d = log p_ref - log p_new
则 KL = E[exp(d) - d - 1]

性质：
- exp(d) - d - 1 ≥ 0 对所有 d 成立（因为 e^x ≥ x + 1）
- 当 d = 0 时取等号（两个分布相同）
```

**步骤提示：**
```python
def k3_kl(logp_ref, logp_new):
    """
    Steps:
        1. 遍历每个 token 的对数概率
        2. 计算 d = logp_ref - logp_new
        3. 累加 exp(d) - d - 1
        4. 返回平均值
    """
    # TODO: Implement
    return None

def kl_budget_guard(kl, budget=0.1):
    """
    Steps:
        1. 如果 kl > budget，返回 True（超预算）
        2. 否则返回 False
    """
    # TODO: Implement
    return None
```

### 🌟 练习 4: 奖励函数设计（Stretch Goal）

设计一个防作弊的奖励函数，用于评估数学推理质量。

**验收标准：**
- [ ] 检查答案正确性
- [ ] 检查推理过程（至少有 "because", "therefore" 等词）
- [ ] 惩罚过短的回答（< 10 字）
- [ ] 返回 0.0-1.0 之间的分数

**思考：**
- 为什么只看答案不够？
- 如何防止模型"钻空子"？

**步骤提示：**
```python
def anti_hacking_reward(response: str, ground_truth: str) -> float:
    """
    Steps:
        1. 检查答案正确性（使用练习 1 的函数）
        2. 检查推理过程（是否有逻辑连接词）
        3. 检查回答长度（惩罚过短）
        4. 综合计算分数
    """
    # TODO: Implement
    return None
```

### 🌟 练习 5: Rollout 成本计算器（Stretch Goal）

实现一个函数，根据模型大小和组大小 n 计算 rollout 成本。

**验收标准：**
- [ ] 输入：模型参数量（B）、组大小 n、prompt 数量、GPU 数量
- [ ] 输出：预估的 rollout 时间（秒）
- [ ] 考虑 GPU 数量和并行度

**步骤提示：**
```python
def estimate_rollout_cost(
    model_params_B: float,  # 模型参数量（单位：B）
    n: int,                 # 组大小
    num_prompts: int,       # prompt 数量
    num_gpus: int = 1,      # GPU 数量
) -> float:
    """
    估算 rollout 时间（秒）

    经验公式：
    - 每个 token 的生成时间 ≈ 0.1ms * model_params_B
    - 每个回答平均 100 tokens
    - 总时间 = prompts * n * tokens * time_per_token / num_gpus

    Steps:
        1. 计算每个 token 的生成时间
        2. 计算总 token 数
        3. 计算总时间
        4. 考虑 GPU 并行度
    """
    # TODO: Implement
    return None
```

### 🌟 练习 6: KL 预算护栏（Stretch Goal）

实现一个函数，监控 KL 散度并在超预算时发出警告。

**验收标准：**
- [ ] 输入：logp_ref、logp_new、budget
- [ ] 输出：(是否在预算内, 当前 KL 值, 警告信息)
- [ ] 如果 KL > budget，返回详细的警告信息

**步骤提示：**
```python
def kl_budget_guard(
    logp_ref: list,
    logp_new: list,
    budget: float = 0.05,
) -> tuple:
    """
    KL 预算护栏

    Returns:
        (is_ok, kl_value, warning_msg)
        - is_ok: bool，是否在预算内
        - kl_value: float，当前 KL 值
        - warning_msg: str，警告信息（如果超预算）

    Steps:
        1. 调用 k3_kl 计算 KL 散度
        2. 比较 KL 与 budget
        3. 如果超预算，生成警告信息
        4. 返回结果
    """
    # TODO: Implement
    return None
```

## 🤔 思考题

**Q1：** 为什么 GRPO 比 PPO 更稳定？

<details>
<summary>💡 提示</summary>

GRPO 的基线来自组内均值，不需要训练额外的 Value 网络。
Value 网络可能不稳定（过拟合、梯度爆炸），而组内比较更稳定。

</details>

**Q2：** 为什么"全对组优势全零"不是 bug 而是 feature？

<details>
<summary>💡 提示</summary>

全对的题已经掌握了，不需要再学习。GRPO 天然跳过这些样本，
把计算资源集中在需要学习的样本上。这是"课程学习"的一种形式。

</details>

**Q3：** RLVR 和 RM（奖励模型）各有什么优缺点？

<details>
<summary>💡 提示</summary>

RLVR：
- 优点：不可作弊、无偏差、简单
- 缺点：只适用于可形式化验证的任务

RM：
- 优点：可以评估主观质量（如对话质量）
- 缺点：可能有偏差、可能被奖励黑客攻击

</details>

**Q4：** 为什么 RL 训练比 SFT 训练更难调参？

<details>
<summary>💡 提示</summary>

- RL 的奖励信号更稀疏（只有最终结果）
- RL 的梯度方差更大（采样随机性）
- RL 需要平衡探索和利用
- RL 的 KL 惩罚需要仔细调整

</details>

**Q5：** verl 的 HybridEngine 解决了什么问题？

<details>
<summary>💡 提示</summary>

rollout 和 training 用不同的引擎（vLLM vs FSDP2），每次更新后要把新权重搬进推理引擎。
HybridEngine 用重分片+原地转换把这一步的开销压到最低。

</details>

## 实验题（Docker，02 章）

- 跑通 quickstart 的 0.5B PPO → 换 `adv_estimator=grpo`，记录：显存峰值变化（省掉 critic）
- 自定义奖励函数（改 1 处规则）并观察 reward hacking：例如只奖励"答案里含数字"会发生什么

## 🎯 面试直通车

- "GRPO 和 PPO 的本质区别？"——基线来源：组内平均 vs 学习出的 Value（省一整个模型的训练状态）
- "RLVR 为什么香？什么时候不行？"——可机器验证且不可作弊；创意/对话类仍需 RM
- "rollout 为什么是 RL 的瓶颈？verl 怎么解？"——生成远贵于训练；双引擎 + HybridEngine 权重回同步
- "你们 RL 的 KL 怎么算？"——k3 估计器（exp(d)-d-1，恒非负低方差）+ 预算护栏

## 参考资源

- 📄 [GRPO 论文](https://arxiv.org/abs/2402.03300)
- 📄 [DAPO 论文](https://arxiv.org/abs/2503.14476)
- 🐙 [verl 官方文档](https://github.com/verl-project/verl)
- 🐙 [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)
