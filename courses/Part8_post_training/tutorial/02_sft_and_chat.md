# 02 — SFT 与 Chat Template：从"续写器"到"对话模型"

> 💬 预训练模型会续写但不会遵循指令。SFT（Supervised Fine-Tuning）教它"按指令回答"——Chat Template 结构化输入，Prompt Masking 聚焦 response。

## 📖 前置知识

本章需要你已经掌握：

- **01 章全部**：GPT-2 架构、预训练流水线、forward_hidden()
- **Part 6 的 cross-entropy loss**：`F.cross_entropy` 的输入输出

> 💡 如果你忘了"cross-entropy 怎么算"，先回 Part 6 的 `02_language_model.md`。

## 从"续写器"到"对话模型"

预训练之后，模型学会了一件事：**给定前面的文本，预测下一个 token**。它能续写莎士比亚、补全代码、甚至模仿问答——但这些都是"统计上最可能的续写"，而不是"理解指令后的回答"。

```
预训练模型:
  输入: "What is 1+1?"
  输出: "What is 1+1? What is 2+2? What is 3+3?..."
        ↑ 只是在"续写"模式——重复问题，不是回答

SFT 之后:
  输入: "What is 1+1?"
  输出: "The answer is 2."
        ↑ 学会了"这是问题，我应该回答"
```

🔑 **SFT 的本质**：在"指令-回答"对上微调，让模型学会区分"用户说的话"和"自己应该说的话"。

## Chat Template：结构化输入

对话不是一串连续文本——它有结构：谁在说、说什么。Chat Template 定义了这种结构。

本脚本使用简化版 ChatML 格式（对应 [03_sft.py](../scripts/03_sft.py)）：

```
<|system|>
You are a helpful assistant.
<|user|>
What is 1+1?
<|assistant|>
The answer is 2.
```

各部分的作用：

| 标记 | 作用 | 示例 |
|------|------|------|
| `<\|system\|>` | 系统指令，定义助手行为 | "You are a helpful assistant." |
| `<\|user\|>` | 用户输入 | "What is 1+1?" |
| `<\|assistant\|>` | 模型应该生成的回答 | "The answer is 2." |

```python
SYSTEM_PROMPT = "You are a helpful assistant."

def format_chat(system, user, assistant=""):
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"
```

💡 **为什么需要模板？** 没有模板，模型看到 `What is 1+1?The answer is 2.` 这种纯文本，分不清哪部分是指令、哪部分是回答。模板用特殊 token 划分边界，让模型学会"看到 `<|user|>` 后面是指令，看到 `<|assistant|>` 后面是我该生成的"。

⚠️ CPU 模式下 context_length 只有 64，放不下完整 ChatML，所以用简化格式 `Q: {question}\nA: {response}` 来演示核心思想。原理完全一样。

## Prompt Masking：核心技巧

这是 SFT 中最重要的技巧，也是和"普通微调"的本质区别。

### 问题：为什么不能所有 token 一起算 loss？

假设输入是：

```
<|system|> You are helpful. <|user|> What is 1+1? <|assistant|> The answer is 2.
|_____________ prompt 部分（已知输入）_____________| |______ response ______|
```

如果对所有 token 一起算标准 cross-entropy loss：

```
L = -Σ_all log P(token_i | context)
```

模型会学到什么？它会花大量 capacity 去"预测 prompt"——但 prompt 是已知的输入，预测它毫无意义。更糟的是，模型可能学会"复制指令"而不是"理解并回答"。

### 解决：只在 response token 上算 loss

```python
L = -Σ_response log P(token_i | context)
```

prompt 区域的 token 不贡献 loss，梯度只流过 response 区域。模型被迫把所有学习能力集中在"学会回答"上。

### 实现：loss_mask 张量

```python
def sft_loss(logits, tokens, loss_mask):
    """SFT loss with prompt masking — 只在 response tokens 上计算 loss。"""
    # Step 1: Shift — 用位置 t 的 logits 预测 t+1 的 token
    logits = logits[:, :-1, :]    # (B, T-1, V)
    targets = tokens[:, 1:]       # (B, T-1)
    mask = loss_mask[:, 1:]       # (B, T-1) — mask 也要 shift 对齐

    # Step 2: 逐 token cross entropy（不自动求均值）
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="none"
    )
    ce = ce.view(B, T)

    # Step 3: 乘以 mask（prompt 区域 loss 归零）
    ce = ce * mask

    # Step 4: 求和归一化（只除以 response token 数量）
    loss = ce.sum() / mask.sum().clamp(min=1.0)
    return loss
```

🔑 **四步走**：

1. **Shift**：logits 向前移一位——位置 t 的 logits 预测位置 t+1 的 token（和预训练一样）
2. **逐 token CE**：用 `reduction="none"` 算每个 token 的 loss，不自动求均值
3. **乘以 mask**：prompt 区域的 mask=0，loss 变成 0；response 区域 mask=1，loss 保留
4. **归一化**：只除以 response token 的数量（不是总 token 数）

⚠️ `mask.sum().clamp(min=1.0)` 防止除零——如果某个 batch 里 response 为空（极端情况），不会崩。

### mask 的构造

```python
# 完整序列：system + user + assistant
full_text = format_chat(SYSTEM_PROMPT, "What is 1+1?", "The answer is 2.")
# prompt 部分：system + user（不含 assistant 的回答）
prompt_text = format_chat(SYSTEM_PROMPT, "What is 1+1?", "")

full_tokens = encode(full_text)
prompt_len = len(encode(prompt_text))

# 构造 mask：prompt 部分 = 0，response 部分 = 1
mask = torch.zeros(1, len(full_tokens))
mask[0, prompt_len:] = 1.0
```

```
tokens:  <|system|>  You  are  helpful .  <|user|>  What  is  1+1 ?  <|assistant|>  The  answer  is  2 .
mask:        0        0    0     0     0     0        0     0    0    0       0         1     1      1   1  1
                                              prompt 区域（不计算 loss）              response（计算 loss）
```

## SFT vs 标准 CE 对比

| | 标准 CE（预训练） | SFT Masked CE |
|---|---|---|
| loss 范围 | 所有 token | 只有 response token |
| 公式 | `L = -Σ_all log P(token_i)` | `L = -Σ_response log P(token_i)` |
| 模型学到什么 | 预测下一个 token（包括 prompt） | 只学会"回答" |
| 适用场景 | 预训练 | 指令微调 |

实际效果：masked loss 通常比 unmasked loss **更大**——因为只看"难的部分"（回答），不看"容易的部分"（复制 prompt）。这是正常的。

## SFT 数据格式

SFT 数据通常是三元组 `(instruction, input, output)`：

```python
# 示例（CPU 模式用合成数据）
pairs = [
    ("What is 1+1?", "The answer is 2."),
    ("Say hello.", "Hello! How can I help you?"),
    ("What color is the sky?", "The sky is blue."),
]
```

训练时，每步随机采样一个 pair，构造完整序列 + mask，算 masked loss，更新参数。

## 与 Part 7 SFT 的区别

| | Part 7 SFT | Part 8 SFT（本脚本） |
|---|---|---|
| Loss 计算 | 全 token 算 loss（简化版） | 只在 response 上算（生产版） |
| Chat Template | `<\|im_start\|>` / `<\|im_end\|>` | `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` |
| Prompt Masking | 无 | 有（`loss_mask` 张量） |

Part 7 为了简化教学，跳过了 Prompt Masking。Part 8 补上这个关键技巧——这是生产级 SFT 的标准做法。

## SFT 之后会发生什么？

SFT 之后，模型学会了"按指令回答"：

```
Q: "What is 3+4?"
A: "The answer is 7."  ← 学会了回答（虽然可能不完美）

Q: "Say hello."
A: "Hello! How can I help you?"  ← 学会了礼貌
```

但回答质量参差不齐——有时好、有时差、有时胡说。这是因为 SFT 只教了"什么格式的回答是对的"，没有教"什么样的回答是好的"。下一步我们需要奖励模型来量化回答质量。


> 📚 **延伸对照（LLMs-from-scratch）**：rasbt ch07 开头对**指令数据 JSON 格式规约**的讨论（(instruction, input, output)
> 如何组织成模板、数据去重与改写）——比我们的合成问答对更接近真实数据工程。

## 课后练习

<details>
<summary>Q1: 如果不 mask prompt，模型会怎样？</summary>
A: 模型会把大量 capacity 花在"预测 prompt"上——因为 prompt 占了序列的大部分，而且比 response 更容易预测（就是输入本身）。结果是：prompt 区域的 loss 很低（因为模型学会了复制），但 response 区域的 loss 很高（没学到什么）。更糟的是，模型可能养成"复制指令"的惯性，生成时也倾向于重复输入而不是真正回答。
</details>

<details>
<summary>Q2: loss_mask 的梯度流到哪里去了？</summary>
A: `loss_mask` 本身没有梯度——它是个常量张量（0 和 1）。但它通过乘法操作 `ce * mask` 控制了梯度的流向：mask=0 的位置，loss 变成 0，对应的梯度也是 0，那些 token 的参数不会收到更新信号。mask=1 的位置，梯度正常流过。所以 mask 的作用是"选择性地阻断梯度"，而不是自己参与梯度计算。
</details>

<details>
<summary>Q3: SFT 的 loss 为什么通常比预训练的 loss 大？</summary>
A: 两个原因。第一，SFT 只看 response token，这些 token 通常比 prompt 更难预测（prompt 是已知的、重复的模式，response 是模型需要"创造性"生成的）。第二，SFT 数据量通常远小于预训练数据，模型在更少的数据上做更难的任务，loss 自然更高。看 SFT 的 loss 主要看"下降趋势"，而不是绝对值。
</details>

## 📝 课后作业

完成本章后，去 Assignment 8 完成题 3（Prompt-Masked SFT Loss）：

👉 [Assignment 8](../../../assignments/assignment_8/)

## 下一步

SFT 之后模型会对话，但质量参差不齐。下一步我们引入奖励模型（Bradley-Terry 偏好模型），教模型区分"好回答"和"坏回答"，然后用 DPO/ORPO/KTO 三种算法直接优化策略。

👉 [03 — 奖励模型与对齐算法](03_reward_and_dpo.md)
