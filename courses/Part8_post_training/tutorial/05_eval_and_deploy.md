# 05 — 评估与推理部署：GSM8K、生成策略、完整流水线回顾

> 📊 训练完成后，如何量化模型质量？本章用 GSM8K 数学题评估各阶段模型，对比生成质量，学习 temperature/top_k 等解码策略，然后回顾完整的 LLM 后训练流水线。

## 🎯 学习目标

完成本章后，你将能够：

- ✅ **搭建** 一条 GSM8K mini 评估流水线（出题→生成→抽答案→对答案→算通过率）
- ✅ **解释** temperature/top_k/top_p 各自改变采样分布的哪一部分
- ✅ **对比** pretrain/SFT/DPO 三阶段模型在同一基准上的差异并归因

## 📖 前置知识

本章需要你已经掌握：

- **01~04 章全部**：GPT-2 架构、预训练、SFT、奖励模型、DPO/PPO/GRPO
- **Part 6 的生成**：自回归生成、temperature 采样

> 💡 本章是 Part 8 的收尾——把前面所有阶段串起来，看"从预训练到部署"的全链路。

## GSM8K 评估

GSM8K（Grade School Math 8K）是评估数学推理能力的标准 benchmark。我们用类似的方法评估模型：生成答案 → 提取数字 → 对比金标 → 计算准确率。

对应代码在 [08_eval_and_chat.py](../scripts/08_eval_and_chat.py)。

### 评估流程

```python
def evaluate_model(model, stoi, itos, problems, max_tokens=10, temperature=0.8, top_k=10):
    correct = 0
    for prompt_text, expected in problems:
        prompt_ids = [stoi.get(c, 0) for c in prompt_text]
        prompt_tensor = torch.tensor([prompt_ids], device=device)

        with torch.no_grad():
            gen = model.generate(prompt_tensor, max_new_tokens=max_tokens,
                                 temperature=temperature, top_k=top_k)

        resp_ids = gen[0].tolist()[len(prompt_ids):]
        resp_text = ''.join(itos.get(tid, '?') for tid in resp_ids)
        predicted = extract_number(resp_text)

        is_correct = (predicted is not None) and abs(predicted - expected) < 0.01
        if is_correct:
            correct += 1

    return correct / len(problems)
```

🔑 **三步走**：
1. 用 prompt 编码问题（如 `3+5=`）
2. 让模型生成回答
3. 从回答中提取数字，与期望答案比较

⚠️ 我们的模型很小（CPU 模式 ~0.1M 参数），数学能力有限。评估的目的是**看趋势**（各阶段的相对改进），而不是追求绝对准确率。

### 全阶段对比

理论上，各阶段模型的数学能力应该是：

```
Base（随机）  <  Pretrain（续写）  <  SFT（对话）  <  DPO/PPO/GRPO（对齐）
  ~0%            ~5-10%             ~10-20%           ~15-30%
```

实际效果取决于数据量、模型大小、训练步数。我们的 CPU 缩小版数字会更低，但趋势应该一致。

## 生成参数：控制"创造性 vs 确定性"

生成时有几个关键参数控制输出的"风格"：

### Temperature

```python
logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
```

| temperature | 效果 | 适用场景 |
|:---:|------|------|
| 0.1~0.3 | 非常确定，几乎总是选最高概率的 token | 数学、代码 |
| 0.7~0.9 | 平衡多样性和连贯性 | 日常对话 |
| 1.0 | 原始分布，不做缩放 | 默认值 |
| 1.5+ | 非常随机，可能不通顺 | 创意写作 |

💡 **直觉**：temperature 除以 logits，相当于"拉平"或"拉尖"概率分布。低温度让高概率 token 更突出（确定性），高温度让低概率 token 也有机会（多样性）。

### Top-K

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```

只保留概率最高的 K 个 token，其余设为 `-inf`（softmax 后变成 0）。

| top_k | 效果 |
|:---:|------|
| 1 | 贪心解码：每次选最可能的 token |
| 5~10 | 较保守：候选少，质量稳定 |
| 40~50 | 较多样：候选多，更有趣 |

### Top-P（Nucleus Sampling）

Top-K 的问题是"固定候选数"——不管概率分布是集中还是分散。Top-P 改为"累积概率阈值"：

```
1. 将 token 按概率从高到低排序
2. 从最高概率开始累加，直到累积概率 >= p
3. 只在这些 token 中采样
```

| Top-P | 效果 |
|:---:|------|
| 0.9 | 保留累积概率 90% 的 token |
| 0.95 | 更多样 |
| 1.0 | 退化为原始采样 |

💡 **Top-K vs Top-P**：Top-K 固定候选数，Top-P 动态调整。概率集中时 Top-P 选得少，分散时选得多——更自适应。

⚠️ 本教程的 `generate()` 只实现了 temperature + top_k，未实现 top_p（留作练习）。

### Repetition Penalty

生成时模型容易陷入重复（"the the the..."）。Repetition penalty 通过惩罚已出现过的 token 来缓解：

```python
# 伪代码（本教程未实现）
for token_id in generated_tokens:
    logits[token_id] /= repetition_penalty  # 降低已出现 token 的概率
```

## 推荐的推理超参数

| 场景 | temperature | top_k | 说明 |
|------|:---:|:---:|------|
| 数学/代码 | 0.0~0.3 | 1~5 | 确定性高，减少错误 |
| 日常对话 | 0.7~0.9 | 10~50 | 平衡多样性和连贯性 |
| 创意写作 | 1.0~1.5 | 50+ | 高多样性，更多创意 |
| 翻译/摘要 | 0.3~0.5 | 10~20 | 准确为主，少随机性 |

## KV Cache：推理加速

自回归生成时，每生成一个 token 都要重新计算整个序列的注意力——但前面 token 的 Key/Value 不变。KV Cache 缓存它们，避免重复计算。

```
没有 KV Cache（每步都重算全部）:
  step 1: [A] → K_A, V_A
  step 2: [A, B] → K_A, V_A, K_B, V_B  ← K_A, V_A 重算了
  step 3: [A, B, C] → K_A, V_A, K_B, V_B, K_C, V_C  ← 全部重算

有 KV Cache（只算新 token）:
  step 1: [A] → cache: K_A, V_A
  step 2: [B] → 只算 K_B, V_B，拼接到 cache
  step 3: [C] → 只算 K_C, V_C，拼接到 cache
```

💡 本教程的 `generate()` 没有实现 KV Cache（教学用，代码更清晰）。生产环境的推理引擎（vLLM、TGI）都有 KV Cache。

## 完整流水线回顾

从零到部署的完整 LLM 后训练流程：

```
┌─────────────────────────────────────────────────────────┐
│  预训练（续写能力）                                      │
│  数据：大规模无标注文本                                  │
│  损失：next-token prediction cross-entropy               │
│  脚本：02_pretrain.py                                    │
├─────────────────────────────────────────────────────────┤
│  SFT（对话能力）                                         │
│  数据：(instruction, response) 对                        │
│  损失：response 部分的 cross-entropy（Prompt Masking）   │
│  脚本：03_sft.py                                         │
├─────────────────────────────────────────────────────────┤
│  对齐（人类偏好）                                        │
│  路径 A：DPO/ORPO/KTO（离线，简单）                      │
│    数据：(chosen, rejected) 偏好对                       │
│    脚本：05_dpo_alignment.py                             │
│  路径 B：PPO（在线，需要 Reward Model）                  │
│    数据：prompt + reward model 打分                      │
│    脚本：06_ppo_training.py                              │
│  路径 C：GRPO（在线，RLVR 可验证奖励）                   │
│    数据：prompt + verifier 判定对错                      │
│    脚本：07_grpo_training.py                             │
├─────────────────────────────────────────────────────────┤
│  评估（量化效果）                                        │
│  方法：GSM8K 准确率、生成质量对比、交互式测试           │
│  脚本：08_eval_and_chat.py                               │
└─────────────────────────────────────────────────────────┘
```

### 三条对齐路径

| 路径 | 复杂度 | 数据需求 | 适用场景 | 代表 |
|------|:---:|------|------|------|
| DPO/ORPO/KTO | 低 | 成对偏好 | 通用对齐 | Zephyr, Llama-3 |
| PPO | 高 | Reward Model | 任意奖励 | InstructGPT, ChatGPT |
| GRPO | 中 | 可验证规则 | 数学/代码 | DeepSeek-R1 |

## 脚本运行顺序

推荐的训练流程：

```bash
# 1. 预训练
python 02_pretrain.py   → ckpt_pretrain.pt

# 2. SFT
python 03_sft.py        → ckpt_sft.pt

# 3. 对齐（三选一）
python 05_dpo_alignment.py  → ckpt_dpo.pt    # 最简单
# 或
python 06_ppo_training.py   → ckpt_ppo.pt    # 更强大
# 或
python 07_grpo_training.py  → ckpt_grpo.pt   # DeepSeek 风格

# 4. 评估
python 08_eval_and_chat.py  → 对比所有阶段
```

## 下一步：Scaling Up

本教程用 CPU 缩小版演示了完整的后训练流程。如果你想进一步：

| 方向 | 怎么做 |
|------|------|
| 更大模型 | 增大 n_embed、n_head、n_blocks（需要 GPU） |
| 更多数据 | 用真实对话数据集（ShareGPT、UltraChat） |
| 多 GPU | 用 FSDP 或 DeepSpeed ZeRO 分布式训练 |
| 更长上下文 | 增大 context_length，用 RoPE 替代 learned PE |
| 更好的 tokenizer | 用 BPE（tiktoken 或 HuggingFace tokenizers） |

💡 核心思想不变——只是规模更大、组件更现代。

## 课后练习

<details>
<summary>Q1: Temperature=0 和 top_k=1 的效果一样吗？</summary>
A: 不完全一样。temperature=0 让 logits 除以 0（变成 inf），softmax 后全部概率集中在最大值上——等价于 greedy。top_k=1 也是只保留最高概率的 token。两者效果在大多数情况下相同，但 temperature=0 在多个 token 概率相同时行为取决于实现（可能选第一个），top_k=1 在 tie-breaking 时也有类似问题。实际中通常用 temperature=0.01 而不是严格的 0，避免数值问题。
</details>

<details>
<summary>Q2: 为什么 GSM8K 评估要固定 random seed？</summary>
A: 因为生成涉及随机采样（temperature > 0 时），同样的模型在不同 seed 下可能给出不同答案。固定 seed 保证结果可复现——你跑两次得到的准确率一样。这也是为什么评估时通常用较低的 temperature（减少随机性）或多次采样取平均。
</details>

<details>
<summary>Q3: DPO、PPO、GRPO 三条路径怎么选？</summary>
A: 取决于你的场景。如果你有成对偏好数据且想要简单稳定，选 DPO。如果你有训练好的 Reward Model 且资源充足，选 PPO（理论上上限更高）。如果你的任务是可验证的（数学、代码），选 GRPO（最简单、最省资源）。实际中很多人先试 DPO，不够再上 PPO/GRPO。
</details>

---

恭喜你完成了 Part 8 的全部学习！你已经掌握了 LLM 后训练的完整流程——从预训练到 SFT、从奖励模型到 DPO/PPO/GRPO、从评估到部署。

这套流程是 ChatGPT、Llama、DeepSeek-R1 等模型背后的核心技术栈。虽然我们用的是 CPU 缩小版，但原理和工业级训练完全一样。

[← 上一章：强化学习 PPO 与 GRPO](04_ppo_and_grpo.md) | [下一章：推理与服务 →](06_inference_and_serving.md) | [Part 8 README](README.md)
