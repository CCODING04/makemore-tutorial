# Assignment 11: 对齐实战 — verl 工业级 RL 后训练

> 本作业基于 Part 11 的教程内容，帮助你巩固 RL 后训练的核心概念。
> 四题核心纯 Python（与 `alignment_exercises.py` / `test_alignment_exercises.py` 签名一致）；
> 一题 stretch 选做；实验题在 Docker 里跑。

## 学习目标

完成本作业后，你将能够：

- ✅ 手写 RLVR 奖励函数（\boxed / #### / 最后数字的抽取链）
- ✅ 实现 GRPO 的组内优势计算（单组语义）
- ✅ 实现 k3 KL 估计器（恒非负、低方差）
- ✅ 实现 KL 预算护栏（策略漂移监控）

## 分值表

| 练习 | 函数 | 分值 | 类型 |
|------|------|------|------|
| 1 | `math_reward` | 25 | 核心 |
| 2 | `group_advantages` | 25 | 核心 |
| 3 | `k3_kl` | 25 | 核心 |
| 4 | `kl_budget_ok` | 25 | 核心 |
| 5 | `zero_gradient_groups` | 🌟 选做 | Stretch（未实现测试优雅跳过，不计失败） |

运行测试：`python test_alignment_exercises.py`（或 `pytest test_alignment_exercises.py -v`）。
stretch 未实现时会打印 `⏭️` 并跳过，不影响核心题通过。

## 练习

### 练习 1: 稳健奖励函数 math_reward（25 分）

实现一个 GSM8K 风格的规则奖励函数，从模型回答中抽取数字并判断对错。
**签名**：`math_reward(response: str, ground_truth: str) -> float`（与骨架 `alignment_exercises.py:20`、测试 `test_alignment_exercises.py` 完全一致）。

**验收标准：**
- [ ] 正确抽取 `\boxed{42}` 格式的答案（单反斜杠的真实 LaTeX 转义）
- [ ] 正确抽取 `#### 42` 格式的答案
- [ ] 前两者都没有时，抽取最后一个数字（自我纠正场景 "The answer is 100, no wait, 7." 应取 7）
- [ ] 处理千分位逗号（如 "1,234"）
- [ ] 处理尾随小数点（如 "value is 42."）
- [ ] 处理浮点数精度问题（用 `abs(pred - gt) < 1e-4`）
- [ ] 抽不到任何数字时返回 0.0（不是 None、不抛异常）
- [ ] 返回值只有 1.0 / 0.0 两种

**抽取顺序（工程惯例）：**
```python
# 1. \boxed{} 优先
# 2. '#### 42' 标记
# 3. 最后一个数字
```

**步骤提示：**
```python
def math_reward(response: str, ground_truth: str) -> float:
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

### 练习 2: 组内优势 group_advantages（25 分）

实现 GRPO 的组内优势计算：`A_i = (r_i - mean) / std`。
**签名**：`group_advantages(rewards, eps=1e-6)`——**单组语义**：`rewards` 是**一个 prompt 的 G 个回答的奖励**（`list[float]`，如 `[1.0, 0.0, 1.0, 0.0]`），返回等长的 `list[float]`。
（多维版本 `(n_prompts, n_responses)` 是脚本 01 的批量形态——外层套一个循环即可，见文末思考题 Q2。）

**验收标准：**
- [ ] 输入 `[1.0, 0.0, 1.0, 0.0]` → 高奖励（1.0）为正优势、低奖励（0.0）为负优势
- [ ] 优势之和为 0（数值精度允许 1e-6 误差）
- [ ] 全同组（如 `[1.0, 1.0, 1.0, 1.0]`，std = 0）→ 优势全 0，而不是 NaN/除零崩溃
- [ ] 使用 eps 防止除零：`max(std, eps)` 而不是 `std + eps`
- [ ] 返回列表长度与输入一致

**数学推导：**
```
mean = (1/G) * Σ r_i
std = sqrt((1/G) * Σ (r_i - mean)^2)
A_i = (r_i - mean) / max(std, eps)

性质：
- Σ A_i = 0（优势之和为零）
- 如果所有 r_i 相同，则 std = 0，所有 A_i = 0
  → "太简单的题没有梯度"
```

**步骤提示：**
```python
def group_advantages(rewards, eps=1e-6):
    """
    Args:
        rewards: list[float]，一个 prompt 的 G 个回答的奖励
        eps: 防止除零的小常数

    Steps:
        1. 计算组内均值 mean
        2. 计算组内标准差 std
        3. 如果 std < eps，返回全 0（全同组）
        4. 否则计算 A_i = (r_i - mean) / std
        5. 验证 Σ A_i = 0
    """
    # TODO: Implement
    return None
```

### 练习 3: KL 估计器 k3_kl（25 分）

实现 k3 KL 估计器：`exp(d) - d - 1`，其中 `d = logp_ref - logp_new`。
**签名**：`k3_kl(logp_ref, logp_new)`，两个参数都是同一批 token 的对数概率列表（`list[float]`），返回 `float`。纯 `math` 实现即可。

**验收标准：**
- [ ] 两个分布相同（`d = 0`）时 KL = 0
- [ ] 结果恒非负（因为 `e^x ≥ x + 1`）
- [ ] 对列表取平均（不是求和）
- [ ] 返回值是 float，不是 None

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
```

### 练习 4: KL 预算护栏 kl_budget_ok（25 分）

实现"超预算提前停"护栏：估计 KL，超 budget 视为策略漂移过大。
**签名**：`kl_budget_ok(logp_ref, logp_new, budget=0.05)`（与骨架一致；注意参数是两个 logp 列表，不是 KL 值——函数内部调用你练习 3 的 `k3_kl`）。

**验收标准：**
- [ ] 同分布（KL = 0 ≤ budget）→ 返回 `True`
- [ ] KL 超 budget → 返回 `False`
- [ ] 返回值是 bool（`is True` / `is False` 可通过断言）
- [ ] 边界：`kl == budget` 算在预算内（`<=`）

**步骤提示：**
```python
def kl_budget_ok(logp_ref, logp_new, budget=0.05):
    """
    Steps:
        1. 调用 k3_kl(logp_ref, logp_new) 计算 KL 散度
        2. 比较 KL 与 budget
        3. 返回 kl <= budget（bool）
    """
    # TODO: Implement
    return None
```

### 🌟 练习 5: 零梯度组检测 zero_gradient_groups（Stretch，选做）

找出"全同奖励"的组下标——这些组本轮优势全零、没有梯度，是浪费的 rollout。
**签名**：`zero_gradient_groups(reward_matrix)`，输入 `(n_prompts, n_responses)` 的二维奖励列表，返回全同组的下标列表（升序）。
未实现（骨架返回 None）时测试会打印 `⏭️` 并优雅跳过，**不影响核心 4 题的通过**。

**验收标准：**
- [ ] `[[1.0, 1.0], [0.0, 1.0], [2.0, 2.0]]` → `[0, 2]`
- [ ] 返回下标升序
- [ ] 浮点比较用 `abs(max - min) < eps`（或 `max == min`）避免浮点噪声误判
- [ ] 没有全同组时返回 `[]`

**步骤提示：**
```python
def zero_gradient_groups(reward_matrix):
    """
    Steps:
        1. 遍历每个 prompt 的奖励列表（一组）
        2. 如果组内 max == min（全同），记下该组下标
        3. 返回全同组的下标列表（升序）
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

**Q2：** 本作业的 `group_advantages` 是单组（一个 prompt）版本，脚本 01 里是批量 `(n_prompts, n_responses)` 版本。两者是什么关系？

<details>
<summary>💡 提示</summary>

数学完全一样：批量版本 = 对每个 prompt 的奖励列表分别调用单组版本（外层套一个循环/列表推导）。
工程上"单组"是纯函数（好测试、好复用），"批量"是训练循环里的向量化调用——
verl 的 `adv_estimator=grpo` 内部就是批量版：按 prompt 分组 → 组内标准化。

</details>

**Q3：** 为什么"全对组优势全零"不是 bug 而是 feature？

<details>
<summary>💡 提示</summary>

全对的题已经掌握了，不需要再学习。GRPO 天然跳过这些样本，
把计算资源集中在需要学习的样本上。这是"课程学习"的一种形式。

</details>

**Q4：** RLVR 和 RM（奖励模型）各有什么优缺点？

<details>
<summary>💡 提示</summary>

RLVR：
- 优点：不可作弊、无偏差、简单
- 缺点：只适用于可形式化验证的任务

RM：
- 优点：可以评估主观质量（如对话质量）
- 缺点：可能有偏差、可能被奖励黑客攻击

</details>

**Q5：** 为什么 RL 训练比 SFT 训练更难调参？

<details>
<summary>💡 提示</summary>

- RL 的奖励信号更稀疏（只有最终结果）
- RL 的梯度方差更大（采样随机性）
- RL 需要平衡探索和利用
- RL 的 KL 惩罚需要仔细调整

</details>

**Q6：** verl 的 HybridEngine 解决了什么问题？

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
