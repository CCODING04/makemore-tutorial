# Part 8 设计方案：从零训练 LLM — 后训练全流程

> 参考：[FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)（MIT）
> 硬件：单张 4090（24GB VRAM），admin02@192.168.0.126（双 4090）验证

## 定位

Part 7 用 minimind 架构（RMSNorm+RoPE+GQA+SwiGLU）复现了现代 LLM。Part 8 **从零开始**，用经典 GPT-2 架构（LayerNorm+learned PE+MHA+ReLU）走完一个 LLM 的**完整生命周期**：构建 → 预训练 → SFT → 奖励模型 → 对齐 → 强化学习 → 评估。

重合部分（Transformer 架构、SFT 基础、DPO 基础）视为**复习**，两部分可独立学习。

```
Part 6 (Transformer/GPT 教学版)
Part 7 (minimind 现代 LLM: RMSNorm+RoPE+GQA+SwiGLU+DPO)
    │
    │  Part 8 独立入口 ↓（不依赖 Part 7 代码）
    │
Part 8: 从零训练 LLM — GPT-2 架构 → Pretrain → SFT → Reward → DPO/PPO/GRPO → Chat
```

---

## 硬件设计

### 三级模型配置

| 模式 | 参数量 | 架构 | 数据 | 设备 | 耗时 |
|------|--------|------|------|------|------|
| **CPU** | ~2M | embed=64, layers=2, heads=4 | 合成数据（脚本内生成） | 任意 CPU | <30s/脚本 |
| **GPU-小** | ~40M | embed=256, layers=6, heads=8 | 下载小数据集 | 单张 4090 | 1-5min/脚本 |
| **GPU-标准** | ~40M | embed=512, layers=12, heads=8 | tiny Shakespeare + 合成 | 单张 4090 | 2-10min/脚本 |

> 注：原计划 embed=1024/24 层（~406M）因 attention 矩阵在 fp32 下占 ~24GB 超出 4090 单卡显存，已调整为 embed=512/12 层（~40M）。406M 需要 gradient checkpointing 或多卡。

### 4090 显存估算（~40M 模型，bf16）

| 项目 | 显存 |
|------|------|
| 模型参数 | 40M × 2B = ~80MB |
| AdamW 优化器状态 | 2 × 80MB = ~160MB |
| 梯度 | ~80MB |
| 激活值（batch=4, seq=512） | ~1-2GB |
| **总计** | **~2-3GB** |
| 4090 剩余 | ~21-22GB（余量充足） |

PPO/GRPO 需要生成多个回答，40M 小模型 + bf16 完全够用。

---

## 文件树

```
courses/Part8_post_training/
├── scripts/
│   ├── 01_gpt_model.py                  # 从零构建 GPT-2（LayerNorm + learned PE + MHA + ReLU）
│   ├── 02_pretrain.py                   # 预训练（数据加载 + AdamW + cosine + bf16 + checkpoint）
│   ├── 03_sft.py                        # SFT + Prompt Masking + Chat Template
│   ├── 04_reward_model.py               # Bradley-Terry 奖励模型
│   ├── 05_dpo_alignment.py              # DPO + ORPO + KTO 三种对齐算法
│   ├── 06_ppo_training.py               # PPO（Value Head + GAE + Clipped Surrogate）
│   ├── 07_grpo_training.py              # GRPO/RLVR（Critic-Free RL, DeepSeek-R1 风格）
│   └── 08_eval_and_chat.py              # GSM8K 评估 + KV Cache 推理 + 交互式 Chat
├── tutorial/
│   ├── README.md                        # 导航 + 前置知识 + 数据说明
│   ├── 01_gpt_and_pretrain.md           # GPT-2 架构 + 预训练流水线
│   ├── 02_sft_and_chat.md               # SFT + Chat Template + Prompt Masking
│   ├── 03_reward_and_dpo.md             # 奖励模型 + DPO/ORPO/KTO
│   ├── 04_ppo_and_grpo.md               # PPO + GRPO 强化学习
│   └── 05_eval_and_deploy.md            # 评估 + 推理 + 全流水线回顾
└── images/

assignments/assignment_8/
├── assignment.md
├── post_training_exercises.py
└── test_post_training_exercises.py
```

---

## 脚本设计

### 脚本 01：从零构建 GPT-2

**目标**：用纯 PyTorch 构建一个完整的 decoder-only Transformer

**核心内容**：
- `Head`：单头因果注意力（Q/K/V 投影 + scaled dot-product + causal mask）
- `MultiHeadAttention`：多头并行 + 输出投影
- `MLP`：4× 扩展 + ReLU + 投影回
- `Block`：Pre-LN 残差块（LayerNorm → Attn → 残差 → LayerNorm → MLP → 残差）
- `Transformer`：token embedding + learned position embedding + N blocks + final LN + lm_head
- `forward_hidden()`：返回 final LN 之后、lm_head 之前的 hidden state（供奖励头/价值头使用）
- 参数量计算与验证

**与 Part 6 的重合**：
- Part 6 已实现类似架构，Part 8 更规范（独立模块、forward_hidden hook）
- 重合内容视为复习，Part 8 的代码**独立可运行**

**输出**：模型实例 + 参数量报告 + 前向 shape 验证

---

### 脚本 02：预训练

**目标**：在文本数据上预训练 GPT-2，学会"续写"

**核心内容**：
- 数据加载：The Pile（GPU 模式）/ 合成文本（CPU 模式）
- tiktoken BPE 分词（r50k_base，50304 词表）
- 训练循环：AdamW + cosine LR schedule + warmup + gradient clipping
- 混合精度：torch.autocast(bf16) + 无 GradScaler（bf16 不需要）
- 梯度累积：小 batch 多次 forward 后再 step
- Checkpoint 保存/恢复
- 训练曲线：loss 下降 + 学习率变化

**4090 适配配置**：
```python
# GPU 标准模式（4090 单卡）
batch_size = 16         # micro-batch
grad_accum = 8          # effective batch = 128
lr = 3e-4
train_steps = 5000      # 演示用，实际 200K
context_length = 1024
n_embed = 1024
n_blocks = 24
n_head = 16
```

**输出**：预训练 checkpoint + 训练曲线 + 生成演示

---

### 脚本 03：SFT + Prompt Masking

**目标**：把"续写器"变成"对话模型"

**核心内容**：
- Chat Template：`<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{response}`
- **Prompt Masking**：只在 response token 上计算 loss
  - 对比 mask vs 不mask 的训练效果
- SFT 数据：Alpaca/Dolly（GPU）/ 合成指令对（CPU）
- 训练超参：lr=1e-5, epochs=3, warmup=100

**关键公式**：
```
标准 CE:      L = -Σ_all log P(token_i | context)
SFT masked:   L = -Σ_response log P(token_i | context)
```

**与 Part 7 的重合**：Part 7 脚本 07 有基础 SFT，Part 8 更完整（prompt masking + chat template）

**输出**：SFT checkpoint + 对话演示

---

### 脚本 04：奖励模型

**目标**：训练一个能判断"哪个回答更好"的评分器

**核心内容**：
- 在 Transformer 骨架上加**标量奖励头**：`nn.Linear(n_embed, 1)`
- 只用**最后一个 token 的 hidden state** 作为序列奖励（InstructGPT 风格）
- Bradley-Terry 偏好模型：`P(A > B) = sigmoid(r(A) - r(B))`
- 训练数据：`(prompt, chosen, rejected)` 三元组
  - GPU：HH-RLHF / UltraFeedback
  - CPU：SFT 模型生成 + 规则打分

**关键公式**：
```
r(x) = reward_head(hidden_last_token)
L = -log sigmoid(r(chosen) - r(rejected))
```

**输出**：奖励模型 checkpoint + 奖励分布可视化

---

### 脚本 05：DPO + ORPO + KTO 对齐

**目标**：实现三种不需要 PPO 的对齐算法

**核心内容**：

**DPO**（Direct Preference Optimization）：
- 完整推导：Bradley-Terry → RLHF 目标 → 消掉奖励函数 → 闭式解
- `logits = (π_chosen - π_rejected) - (ref_chosen - ref_rejected)`
- `L = -logsigmoid(β * logits)`
- 冻结参考策略 ref（SFT 模型）
- β 的作用：越大越保守，越小越激进

**ORPO**（Odds Ratio Preference Optimization）：
- 无参考模型
- `L = NLL(chosen) + λ * -log_sigmoid(log_odds_chosen - log_odds_rejected)`

**KTO**（Kahneman-Tversky Optimization）：
- 无成对数据，只需"好/坏"标签
- 基于前景理论：损失比收益更敏感

**对比表**：

| | 参考模型 | 成对数据 | 复杂度 | 代表 |
|---|---|---|---|---|
| DPO | ✅ | ✅ | 中 | Zephyr |
| ORPO | ❌ | ✅ | 低 | Llama-3 |
| KTO | ✅ | ❌ | 低 | 稀疏标注场景 |

**输出**：三种对齐 checkpoint + 效果对比

---

### 脚本 06：PPO 强化学习

**目标**：实现经典 RLHF 的 PPO 训练

**核心内容**：
- **Value Head**：共享 backbone 的价值函数头 `V(s)`
- **GAE**（Generalized Advantage Estimation）：
  - `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
  - `A_t = Σ (γλ)^l δ_{t+l}`
- **Clipped Surrogate Loss**：
  - `ratio = π/π_old`
  - `L = min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)`
- **Clipped Value Loss** + **Entropy Bonus**
- **Rollout**：用当前策略生成回答 → 奖励模型打分 → 计算优势 → 更新
- 奖励来源：脚本 04 的奖励模型 / 可验证规则（GSM8K）

**PPO 训练循环**：
```
for iteration:
    1. 用 π 生成回答（rollout）
    2. 奖励模型打分
    3. 计算 GAE 优势
    4. 多步更新 π 和 V（clipped）
```

**输出**：PPO checkpoint + 奖励曲线

---

### 脚本 07：GRPO / RLVR（DeepSeek-R1 风格）

**目标**：实现无需 critic 的强化学习

**核心内容**：
- **GRPO vs PPO**：
  - PPO 需要 Value Network 估计基线 V(s)
  - GRPO 用**组内多个采样的平均奖励**作基线
  - 省掉 Value Network → 省显存、省计算
- **Group-Relative Advantage**：
  - 每个 prompt 采样 G 个回答
  - `A_i = (r_i - mean(r)) / std(r)`
- **k3 KL 估计器**：`KL = exp(ref-new) - (ref-new) - 1`
- **Token-Level Clipped Surrogate + KL Penalty**
- **RLVR**：可验证奖励（数学题答案对错）
  - 不需要训练奖励模型

**GRPO 训练循环**：
```
for iteration:
    1. 取 prompt
    2. 采样 G 个回答
    3. verifier 打分
    4. 标准化优势
    5. 更新 π（clipped + KL penalty）
    # 无 Value Network！
```

**输出**：GRPO checkpoint + 数学推理演示

---

### 脚本 08：评估 + 交互式推理

**目标**：量化评估 + 部署推理

**核心内容**：
- **GSM8K 评估**：100 道数学题，提取数值答案，计算准确率
- **全阶段对比**：Base → SFT → DPO → PPO → GRPO
- **KV Cache 推理**：加速自回归生成（复用 Part 7 的 KV Cache 实现思路）
- **交互式 Chat**：加载任意 checkpoint，多轮对话
- **生成参数**：temperature / top_k / top_p / repetition_penalty

**输出**：评估报告 + 交互式 chat 演示

---

## 教程设计（5 章）

### README.md — 导航

```
Part 8: 从零训练 LLM — 后训练全流程

前置知识：
- Part 6 Transformer/GPT（self-attention、残差、LayerNorm、decoder-only）
- Part 3 BatchNorm（归一化概念）
- 概率论基础（sigmoid、log-prob、KL 散度）

章节导航：
01 | GPT-2 架构 + 预训练       | 脚本 01 02
02 | SFT + Chat Template       | 脚本 03
03 | 奖励模型 + DPO/ORPO/KTO  | 脚本 04 05
04 | PPO + GRPO 强化学习       | 脚本 06 07
05 | 评估 + 推理部署            | 脚本 08
```

### 第 1 章：GPT-2 架构 + 预训练

**内容**：
1. GPT-2 架构总览（token embed + pos embed + N blocks + LN + lm_head）
2. 单头注意力 → 多头注意力（Head + MultiHeadAttention）
3. Pre-LN 残差块（LayerNorm → Attn → 残差 → LayerNorm → MLP → 残差）
4. MLP：4× 扩展 + ReLU
5. 参数量计算
6. 预训练：数据加载、tiktoken BPE、AdamW + cosine schedule、bf16、gradient accumulation
7. 与 Part 6/7 的架构对比（复习）

### 第 2 章：SFT + Chat Template

**内容**：
1. 从"续写器"到"对话模型"
2. Chat Template（ChatML 风格）
3. **Prompt Masking** 原理与实现
4. SFT vs 标准 CE 训练的区别
5. SFT 数据格式：(instruction, input, output)
6. 与 Part 7 SFT 的区别（Part 7 全 token 算 loss，Part 8 只在 response 上算）

### 第 3 章：奖励模型 + DPO/ORPO/KTO

**内容**：
1. **为什么需要对齐**：SFT 之后的问题（幻觉、不安全）
2. **Bradley-Terry 偏好模型**：从直觉到公式
3. **奖励模型训练**：last-token reward + pairwise loss
4. **DPO 推导**：Bradley-Terry → RLHF 目标 → 消掉奖励 → 闭式解
5. **ORPO**：无参考模型的对齐
6. **KTO**：无成对数据的对齐
7. 三种算法对比（何时用哪个）

### 第 4 章：PPO + GRPO 强化学习

**内容**：
1. **为什么 DPO 不够**：离线 vs 在线学习
2. **PPO 核心**：策略梯度 → ratio clip → GAE → Value Head
3. **PPO 训练循环**：rollout → reward → GAE → clipped update
4. **GRPO 核心创新**：不需要 Value Network
5. **Group-Relative Advantage**：组内标准化
6. **RLVR**：可验证奖励（数学题答案对错）
7. **PPO vs GRPO 对比表**
8. DeepSeek-R1 为什么选 GRPO

### 第 5 章：评估 + 推理部署

**内容**：
1. GSM8K 评估方法
2. 全阶段对比：Base → SFT → DPO → PPO → GRPO
3. KV Cache 推理加速
4. 交互式 Chat 部署
5. 完整流水线回顾
6. 下一步：scaling up

---

## 作业设计（8 题）

### 基础题（5 题）

**题 1：Causal Self-Attention Head**
- 实现 `Head(nn.Module)`：Q/K/V 投影 + scaled dot-product + causal mask
- 验证：shape 正确、因果遮罩有效（上三角为 -inf）

**题 2：Pre-LN Transformer Block**
- 实现 `Block(nn.Module)`：LayerNorm → MHA → 残差 → LayerNorm → MLP → 残差
- 验证：输出 shape 与输入一致

**题 3：Prompt-Masked SFT Loss**
- 实现 `sft_loss(logits, targets, prompt_mask)`
- 只在 response token 上计算 CE loss
- 验证：全 mask 时 loss=0

**题 4：Bradley-Terry Reward Loss**
- 实现 `reward_loss(r_chosen, r_rejected)`
- `L = -log sigmoid(r_chosen - r_rejected)`
- 验证：相等时 loss = ln2

**题 5：DPO Loss**
- 实现 `dpo_loss(pi_ch, pi_rej, ref_ch, ref_rej, beta=0.1)`
- 验证：pi==ref 时 loss = ln2

### 拓展题（3 题）

**题 6：GAE Advantage Estimation**（🌟）
- 实现 `gae(rewards, values, gamma, lam)`
- 从后往前递推
- 验证：λ=0 退化为 TD error

**题 7：PPO Clipped Loss**（🌟）
- 实现 `ppo_loss(logp_new, logp_old, advantages, eps=0.2)`
- 验证：ratio 在 [1-ε, 1+ε] 内时 loss 不变

**题 8：GRPO Group Advantage**（🌟）
- 实现 `group_advantages(rewards, group_size)`
- 组内标准化
- 验证：组内均值为 0

---

## 数据策略

| 阶段 | CPU 模式（合成） | GPU 模式（真实） |
|------|-----------------|-----------------|
| 预训练 | 脚本内生成随机文本 | The Pile（HuggingFace streaming） |
| SFT | 从文本生成 (指令, 回答) 对 | Alpaca / Dolly |
| 奖励模型 | SFT 模型生成 + 规则打分 | HH-RLHF / UltraFeedback |
| DPO/ORPO/KTO | 奖励模型生成偏好对 | HH-RLHF / UltraFeedback |
| PPO | 规则奖励 | 奖励模型 / GSM8K verifier |
| GRPO | 简单算术题 | GSM8K 训练集 |
| 评估 | 5 道算术题 | GSM8K 测试集 100 题 |

**数据下载**（GPU 模式）：
```python
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train")
ds = load_dataset("Anthropic/hh-rlhf", split="train")
ds = load_dataset("tatsu-lab/alpaca", split="train")
```

---

## 验证计划

### Phase B.1：本地 CPU 验证

```bash
for s in courses/Part8_post_training/scripts/*.py; do
  python "$s" && echo "✅ $s" || echo "❌ $s"
done
python assignments/assignment_8/test_post_training_exercises.py
```

### Phase B.2：admin02 GPU 验证

```bash
# 单卡 4090 验证
for s in courses/Part8_post_training/scripts/*.py; do
  python "$s" && echo "✅ $s" || echo "❌ $s"
done
```

### Phase B.3：内容一致性审查

- 教程代码片段与脚本实现一致
- 教程公式与代码实现一致
- 题号引用与 assignment.md 一致
- 脚本引用路径正确
- 作业验证代码与测试一致

---

## 关键设计决策

1. **从零开始**：不复用 Part 7 代码，Part 8 完全独立可学习
2. **GPT-2 架构**：LayerNorm + learned PE + MHA + ReLU（经典款，不混 minimind 的现代组件）
3. **4090 适配**：~406M 模型 bf16 仅用 ~6-8GB，余量充足
4. **CPU/GPU 双模式**：CPU 合成数据 <30s，GPU 真实数据 5-30min
5. **渐进式脚本**：01→08 每个脚本独立可运行，前后有演进关系
6. **重合即复习**：Transformer 架构、SFT、DPO 与 Part 6/7 重合，视为巩固
7. **GRPO 亮点**：DeepSeek-R1 风格的 critic-free RL，前沿内容单独一章
