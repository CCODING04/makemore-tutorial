# 01 — GPT-2 架构与预训练：从零构建经典 Transformer

> 🏗️ 从零实现 GPT-2 的经典架构（LayerNorm + learned PE + MHA + ReLU），然后用现代训练技巧预训练它——AdamW、cosine LR、bf16 混合精度、gradient accumulation。

## 📖 前置知识

本章需要你已经掌握：

- **Part 6 全部**：Transformer 架构、self-attention、残差连接、LayerNorm、decoder-only GPT
- **Part 3 的 BatchNorm**：归一化的思想、可学习的缩放参数 —— 讲 LayerNorm 时会和它对照

> 💡 如果你忘了"因果注意力怎么 mask"或"残差连接为什么重要"，先回 Part 6 的 `03_multi_head_attention.md`。

## 从 Part 6 结束的地方出发

Part 6 我们从零构建了一个字符级 mini-GPT，学会了"预测下一个字符"。那个模型用了最简单的组件：LayerNorm、learned 位置编码、标准 MHA、ReLU FFN。

Part 7 把每个零件都换成了现代版：RMSNorm、RoPE、GQA、SwiGLU。

**Part 8 又换回经典款**——为什么？因为 train-llm-from-scratch 仓库用的是经典 GPT-2 架构，而且经典款更容易理解后训练的核心思想（奖励头、价值头、DPO loss）。现代组件是"锦上添花"，后训练才是"从续写器到对话助手"的关键。

## GPT-2 架构总览

先看整体结构，然后逐个拆解：

```
输入 token ids (B, T)
     |
     v
token_embed(ids) + pos_embed(positions)    ← 相加（不是拼接）
     |
     v
┌─────────────────────────────────┐
│  Block × N                       │
│  ┌───────────────────────────┐  │
│  │ LN → MHA → + (残差)       │  │  ← Pre-LN: 先归一化再进子层
│  │ LN → MLP → + (残差)       │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
     |
     v
LayerNorm (final)                    ← 最后一层 LN
     |
     v
lm_head (Linear: n_embed → vocab)    ← 预测下一个 token
     |
     v
logits (B, T, vocab_size)
```

关键设计选择：

- **token embedding + learned position embedding（相加）**：位置编码是可学习的参数表，不用 RoPE
- **Pre-LN Block**：先 LayerNorm 再进子层，训练更稳定（后面会详细讲）
- **MHA**：标准多头注意力，不用 GQA
- **MLP**：4x 扩展 + ReLU，不用 SwiGLU
- **forward_hidden()**：返回 final LN 之后、lm_head 之前的 hidden state —— 这是后训练的关键 hook 点

对应代码在 [01_gpt_model.py](../scripts/01_gpt_model.py)。

## 单头注意力 Head

注意力是 Transformer 的核心。先看最简单的单头版本：

```python
class Head(nn.Module):
    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)  # K 投影
        self.query = nn.Linear(n_embed, head_size, bias=False)  # Q 投影
        self.value = nn.Linear(n_embed, head_size, bias=False)  # V 投影
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)       # (B, T, head_size)
        q = self.query(x)     # (B, T, head_size)
        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)     # (B, T, head_size)
        return wei @ v        # (B, T, head_size)
```

🔑 **三个关键步骤**：

1. **Q/K/V 投影**：把输入 `x` 分别投影成 Query、Key、Value 三个向量。`bias=False` 是现代惯例（省参数，效果一样）
2. **Scaled dot-product**：`Q @ K^T / sqrt(d_k)` —— 除以 `sqrt(d_k)` 防止内积值太大导致 softmax 饱和
3. **Causal mask**：用下三角矩阵把"未来位置"填成 `-inf`，softmax 后变成 0 —— 保证每个位置只能看到自己和之前的内容

⚠️ `register_buffer('tril', ...)` 注册的是 buffer 而不是 parameter —— 它不参与梯度更新，但会随 `model.to(device)` 移动到 GPU。

## 多头注意力 MultiHeadAttention

单头注意力只能关注一种模式。多头注意力让模型同时关注多种模式：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([Head(head_size, n_embed, context_length)
                                    for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)  # 输出投影

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接
        return self.proj(x)  # 投影回 n_embed 维
```

💡 **为什么需要多头？** 不同的 head 可以关注不同的模式——比如一个 head 关注相邻词（语法），另一个 head 关注远距离依赖（语义）。`n_head=4` 意味着有 4 个这样的"视角"。

⚠️ **head_size = n_embed // n_head**：每个 head 的维度是总维度除以 head 数。拼接后 `n_head × head_size = n_embed`，正好回到原始维度。

## MLP 前馈网络

每个 Block 里除了注意力，还有一个前馈网络（FFN）：

```python
class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # 4x 扩展
            nn.ReLU(),                         # 激活
            nn.Linear(4 * n_embed, n_embed),   # 投影回
        )
```

经典的"4x 扩展 + ReLU + 投影回"设计。先扩大到 4 倍维度（增加表达力），用 ReLU 激活（引入非线性），再投影回原始维度。

**与 Part 7 SwiGLU 的对比**：

| | 经典 ReLU FFN（本脚本） | SwiGLU FFN（Part 7） |
|---|---|---|
| 激活 | ReLU（硬截断：负值变 0） | SiLU（软门控：负值有小梯度） |
| 结构 | 两层：up → down | 三层：gate + up → down |
| 扩展比 | 4x | ~3.2x（同等参数量） |
| 代表 | GPT-2, BERT | Llama, Qwen, minimind |

💡 ReLU 更简单，SiLU 表达力更强。对于教学目的，经典款更容易理解。

## Pre-LN vs Post-LN

这是 Transformer 架构中一个经常被忽略但非常重要的设计选择：

```
Post-LN（原始论文 "Attention Is All You Need"）:
  x = x + Attn(x)       ← 先算子层
  x = LayerNorm(x)       ← 再归一化

Pre-LN（GPT-2、Llama 等现代模型）:
  x = x + Attn(LayerNorm(x))  ← 先归一化，再算子层
```

对应代码：

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-LN: LN 在子层之前
        x = x + self.mlp(self.ln2(x))
        return x
```

💡 **为什么 Pre-LN 更稳定？**

- Post-LN：LayerNorm 在残差之后，梯度要穿过 LN 才能传回主路径。LN 的梯度在均值/方差计算时有非线性，容易导致梯度爆炸或消失，需要 learning rate warmup
- Pre-LN：LayerNorm 在子层之前，残差连接直接把梯度传回主路径（`x = x + f(LN(x))`，梯度对 `x` 有直通路径）。训练更稳定，不需要 warmup

⚠️ 几乎所有现代 LLM（GPT-2/3/4、Llama、Qwen）都用 Pre-LN。Post-LN 主要出现在早期论文里。

## forward_hidden()：后训练的关键 hook

这是本脚本最重要的设计——一个看似简单的函数：

```python
def forward_hidden(self, idx):
    """backbone 前向：返回 final LN 之后的 hidden state (B, T, n_embed)。"""
    B, T = idx.shape
    tok_emb = self.token_embed(idx)                    # (B, T, n_embed)
    pos_emb = self.position_embed(self.pos_idxs[:T])  # (T, n_embed)
    x = tok_emb + pos_emb
    for block in self.blocks:
        x = block(x)
    return self.ln_f(x)  # ← 在这里停住，不经过 lm_head
```

🔑 **为什么需要这个函数？**

后训练阶段，我们需要在 backbone 上接入不同的"头"：

| 阶段 | 接入什么 | 用途 |
|------|---------|------|
| 预训练 | `lm_head` | 预测下一个 token |
| 奖励模型 | `reward_head` | 给回答打分（标量） |
| PPO | `value_head` | 估计状态价值 V(s) |

`forward_hidden()` 返回的就是"去掉 lm_head 之前的最后一层输出"，奖励头和价值头都从这里接入。后面几章会反复用到它。

## 参数量计算

GPT-2 的参数量有一个简洁的近似公式：

```
参数量 ≈ vocab × embed + n_blocks × (12 × embed²) + embed × vocab
```

其中 `12 × embed²` 来自每个 Block：
- 注意力：Q/K/V 三个投影 = `3 × embed²`，输出投影 = `embed²`，共 `4 × embed²`
- MLP：两个线性层 = `2 × 4 × embed² = 8 × embed²`
- 合计 `12 × embed²`（忽略 LN 的少量参数）

| 配置 | embed | heads | blocks | vocab | 参数量 |
|------|:---:|:---:|:---:|:---:|:---:|
| CPU 缩小版 | 64 | 4 | 2 | 256 | ~0.1M |
| ~1M | 128 | 4 | 4 | 1000 | ~0.8M |
| ~10M | 256 | 8 | 6 | 50304 | ~6M |
| ~100M | 512 | 8 | 12 | 50304 | ~40M |
| 标准 GPT-2 | 1024 | 16 | 24 | 50304 | ~406M |

💡 本教程用 CPU 缩小版（~0.1M 参数）快速演示。GPU 模式用 ~40M 配置（embed=512, 12 层），可在单张 4090 上跑通全流程。原理完全一样，只是规模不同。

## 预训练流水线

模型搭好了，怎么训练？对应代码在 [02_pretrain.py](../scripts/02_pretrain.py)。

### 数据加载

```python
# CPU 模式：字符级编码（vocab=65）
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# GPU 模式：tiktoken BPE（vocab=50304）
# （tiktoken 只能用现成词表、不能训练；与 Part 7 自训 BPE 的对照见
#   courses/Part7_minimind/tutorial/01_bpe_tokenizer.md 的「三种工业实现对照」）
import tiktoken
enc = tiktoken.get_encoding('r50k_base')
data = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
```

### AdamW 优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,             # 学习率
    betas=(0.9, 0.95),   # 动量衰减系数（比默认 (0.9, 0.999) 更保守）
    weight_decay=0.1,    # 权重衰减（L2 正则的"解耦"版本）
)
```

💡 **AdamW vs Adam + L2**：AdamW 把权重衰减从梯度更新中"解耦"出来——先做 Adam 更新，再单独做权重衰减。这比 Adam + L2 更正确，是现代 LLM 训练的标准选择。

### Cosine LR + Linear Warmup

```python
warmup = max(3, max_steps // 10)

def lr_lambda(step):
    if step < warmup:
        return step / warmup                    # 线性 warmup
    p = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p)) # cosine 衰减

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

```
学习率
  ^
  |     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
  |    /                    \
  |   /                      \
  |  /                        \
  | /                          \___
  |/
  +───────────────────────────────→ step
    warmup    cosine decay
```

💡 **为什么要 warmup？** 训练初期参数是随机的，梯度方向不稳定。如果一开始就用大学习率，容易"跑偏"。warmup 让学习率从小到大逐渐增长，等参数稳定后再加速。

### bf16 混合精度

```python
if device == 'cuda':
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, loss = model(xb, yb)
else:
    _, loss = model(xb, yb)  # CPU 不支持 bf16
```

### bf16 vs fp16：为什么推荐 bf16？

| | bf16 | fp16 |
|---|---|---|
| 指数位 | 8 bit（与 fp32 相同） | 5 bit（比 fp32 少） |
| 尾数位 | 7 bit | 10 bit |
| 动态范围 | 大（不容易溢出） | 小（容易溢出/下溢） |
| 精度 | 较低 | 较高 |
| GradScaler | **不需要** | 需要 |
| 推荐硬件 | A100, 4090, 3090 | V100, T4 |

💡 bf16 的动态范围和 fp32 一样大（8 bit 指数），所以不会出现 fp16 那种"梯度太小变 0、太大变 inf"的问题。这就是为什么 bf16 **不需要 GradScaler**——数值范围够大，不会溢出。现代 GPU（A100/4090）推荐 bf16。

### Gradient Accumulation

```python
optimizer.zero_grad(set_to_none=True)
for _ in range(grad_accum):
    xb, yb = get_batch(train_data, batch_size, context_length, device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _, loss = model(xb, yb)
    (loss / grad_accum).backward()  # 梯度累积：除以累积步数
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
optimizer.step()
```

💡 **Gradient Accumulation**：显存不够装大 batch？没关系——做 N 次小 batch 的 forward/backward，把梯度加起来（除以 N），再做一次 step。效果等价于 N 倍大的 batch。

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

把所有参数的梯度范数裁剪到不超过 1.0。防止偶尔出现的"梯度爆炸"把模型炸飞。

### Checkpoint 保存/恢复

```python
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': max_steps,
    'losses': losses,
    'config': {
        'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
        'vocab_size': model_vocab, 'context_length': context_length,
    }
}, ckpt_path)
```

💡 保存 optimizer 状态和 step 数，下次训练就能从断点恢复——不需要从头开始。

## 预训练后会发生什么？

预训练完成后，模型学会了"续写"——给它一段开头，它能接着写下去。但它**不会对话**：

```
输入: "First Citizen:\n"
输出: "We are the people of the world, and we are the ones who..."
      ↑ 像莎士比亚风格的续写，但不是对话
```

这就是预训练模型的局限：它学会了语言的统计规律，但不知道"什么是问题"、"什么是回答"。下一步我们需要 SFT（监督微调）来教它"按指令回答"。


> 📚 **延伸对照（LLMs-from-scratch）**：rasbt/LLMs-from-scratch ch05 的「加载 OpenAI GPT-2 官方权重」与附录 D 的
> LR 调度完整实现——把我们的玩具训练换成真 GPT-2 权重做"权重手术"，是很好的课后实验。

## 课后练习

<details>
<summary>Q1: Pre-LN 为什么比 Post-LN 训练更稳定？</summary>
A: Pre-LN 的残差连接给了梯度一条"直通路径"——`x = x + f(LN(x))` 中，梯度对 `x` 的偏导直接包含一个恒等项 `1`，不需要穿过 LN 的非线性计算。Post-LN 的梯度必须穿过 LN（里面有均值/方差归一化 + 缩放），这些非线性操作容易导致梯度爆炸或消失，所以需要 warmup 来稳定训练。
</details>

<details>
<summary>Q2: bf16 为什么不需要 GradScaler？fp16 为什么需要？</summary>
A: fp16 只有 5 bit 指数，动态范围很小（最大 ~65504，最小 ~6e-8）。训练中梯度经常超出这个范围——太大的变 inf，太小的变 0（下溢）。GradScaler 把 loss 放大 S 倍，让梯度也放大，避免下溢；更新时再缩小回来。bf16 有 8 bit 指数（和 fp32 一样），动态范围够大，梯度几乎不会溢出/下溢，所以不需要 GradScaler。
</details>

<details>
<summary>Q3: 为什么 Gradient Accumulation 要除以 grad_accum？</summary>
A: 因为 `.backward()` 默认是累加梯度（不覆盖）。做 N 次 `.backward()` 后，梯度是 N 个 micro-batch 梯度的**和**，而不是**平均**。除以 N 后，等价于一个大 batch 的平均梯度——这才是我们想要的。如果不除，等效学习率变成了 N 倍，训练会不稳定。
</details>

## 📝 课后作业

完成本章后，去 Assignment 8 完成题 1（单头注意力 Head）和题 2（Pre-LN Block）：

👉 [Assignment 8](../../../assignments/assignment_8/)

## 下一步

预训练完成后，模型会续写但不会对话。下一步我们用 SFT（监督微调）教它"按指令回答"，引入 Chat Template 和 Prompt Masking。

👉 [02 — SFT 与 Chat Template](02_sft_and_chat.md)
