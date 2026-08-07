# 01 — 数据与 Tokenizer：从 ChatGPT 到字符级语言模型

> 🔤 一切的起点：把莎士比亚文本变成神经网络能吃的整数序列，再喂给一个最简单的模型。

## 从 Part 5 结束的地方出发

Part 5 我们用 WaveNet 把上下文**层次化融合**，在 names 数据集上把验证 loss 压到了 **2.0 以下**。但那套方案的骨架是卷积——卷积有固定的感受野、有空间性，你告诉网络"看这几个邻居"。

这一章我们换一个完全不同的思路：**让 token 自己决定去看谁**。这个思路就是 Transformer。

> 💡 这不是推翻 Part 5，而是把它放到更大的图景里：WaveNet/CNN 是我们学习 Transformer 的"脚手架"。后面讲 attention "无空间概念"时，我们会专门拿卷积来做对比。

## 课程动机：ChatGPT 是什么？

你一定听说过 ChatGPT。它底层其实就是一个**语言模型（language model）**：给它一个序列的开头，它逐词（更准确说是逐 **token**）地"续写"下去。

```
你给："帮我写一首关于 AI 的俳句"
ChatGPT: "知识带来繁荣
          拥抱它的力量
          人类永向前"   ← 从左到右，一次吐一个 token
```

同一个提示，两次回答不同——因为它是**概率系统**：每个位置都从"下一个 token 的概率分布"里采样。

- 💡 所以 ChatGPT 做的事，本质上和我们前面 Part 1/2 做的一模一样：**预测序列里的下一个 token**。区别只在于规模、数据量和 token 的粒度。

**GPT** 三个字母拆开：

- **G**enerative —— 生成式，能续写
- **Pretrained** —— 预训练，先在大量文本上训练
- **Transformer** —— 底层的神经网络架构

### Transformer 的起源

Transformer 来自 2017 年那篇里程碑论文 **《Attention is All You Need》**（Vaswani et al.）。有意思的是，这篇论文读起来像一篇**随机的机器翻译论文**——因为作者当时根本没预料到它会统治整个 AI 领域。它是在机器翻译的背景下提出来的，结果这个架构在之后 5 年里被"复制粘贴"进了 AI 的方方面面，包括 ChatGPT 的核心。

> 🔑 **Transformer**：一种神经网络架构，核心组件是 **attention（注意力）**机制，让序列中的每个元素按"重要性"聚合其它元素的信息。

我们现在就动手训练一个**字符级**的 Transformer 语言模型。先定个小目标：让它能写出"看起来像莎士比亚"的文本。

## 数据：tiny Shakespeare

用互联网级别的数据当然不现实，我们用 Karpathy 最喜欢的小数据集——**tiny Shakespeare**：

```
文件：data/input.txt
内容：莎士比亚全部作品的拼接
大小：约 1.1 MB，约 100 万字符
```

[01_explore_data.py](../scripts/01_explore_data.py) 的统计输出（我们实跑的真实结果）：

```
═══ 数据集统计 ═══
  总字符数: 1,115,394
```

前 1000 个字符长这样（开场是《科里奥兰纳斯》的平民戏）：

```
"First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n..."
```

> 🔑 我们做的是**字符级**语言模型：预测"下一个字符是什么"。ChatGPT 用的是**子词级**，后面我们会对比。

## 字符级 Tokenizer

"Tokenizer（分词器）"的意思是：**把原始文本按某种词表转成整数序列**。我们的词表就是"文本中出现过的所有字符"：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

`set(text)` 得到所有唯一字符，`list` 加排序得到一个稳定的顺序。运行结果：

```
═══ 词汇表 ═══
  唯一字符数 (vocab_size): 65
  字符列表:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

注意第一个字符（`''`）其实是换行符 `\n`，列表里第一项是空格、标点、数字 `3`、大写字母、小写字母。**语言模型只能输出它见过的字符**——这正是词汇表的作用。

### encode / decode：字符 ⇄ 整数

构建两张查表字典，然后定义编码器和解码器：

```python
stoi = {ch: i for i, ch in enumerate(chars)}   # char → int
itos = {i: ch for i, ch in enumerate(chars)}   # int → char
encode = lambda s: [stoi[c] for c in s]        # 字符串 → 整数列表
decode = lambda l: ''.join([itos[i] for i in l])  # 整数列表 → 字符串
```

运行结果：

```
═══ Tokenizer 演示 ═══
  encode('hi there') = [46, 47, 1, 58, 46, 43, 56, 43]
  decode([46, 47, 1, 58, 46, 43, 56, 43]) = 'hi there'
  往返一致: True
  索引 0 对应字符: '\n'（换行符）
  索引 1 对应字符: ' '
```

- ⚠️ 索引 0 通常是换行符 `'\n'`（注意它和空格 `' '` 是索引 1，两个不同字符！）。`encode`→`decode` 的"往返一致性"（`decode(encode(s)) == s`）是 tokenizer 正确性的基本自检。
- 💡 用字符级的好处是简单：没有未知字符、不用训练 BPE 词表。代价是序列很长（一句话几十上百个整数）。这是本章的核心权衡，下面展开。

## 其它 Tokenizer 对比：词表大小 vs 序列长度

字符级只是众多 tokenizer 中的一种，而且是最简单的一种。工业界有更常见的方案：

| Tokenizer | 提出方 | 粒度 | 词表大小 | 说明 |
|-----------|--------|------|:---:|------|
| 字符级 | — | 单字符 | 65（我们的） | 最简单，序列最长 |
| **sentencepiece** | Google | 子词 | 几千~几万 | 实践中常见的 subword 方案 |
| **tiktoken / BPE** | OpenAI | 子词 | ~50,000（GPT-2） | GPT 系列用的字节对编码 |

用 OpenAI 的 `tiktoken`（BPE，GPT-2 词表）编码 `"hi there"`，得到的不是 `[46, 47, 1, ...]`，而是**只有 3 个整数**，每个整数的范围在 0 到 50,256 之间。

这就是核心权衡：

```
词表大 + 序列短          vs         词表小 + 序列长
（每个 token 信息量大）        （每个 token 信息量小，要更多步）
        ↕                              ↕
   50K tokens 的 BPE            65 tokens 的字符级
   更贴近"词"的粒度              最简单、无训练开销
```

- 🔑 **subword（子词）tokenizer**：既不是整词，也不是单字符，而是介于两者之间。BPE 从字符开始，逐步把高频相邻片段合并成新的 token，所以它能表达任何词、又能压缩序列长度。
- 💡 我们这一课坚持用字符级，因为它是理解整套流程最干净的脚手架。你把 `encode`/`decode` 换成任何别的 tokenizer，后面的训练代码一行都不用改。

## 训练 / 验证划分

把整个文本编码成一个大整数张量，然后划分：

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

运行结果：

```
═══ Train/Val 划分 (90/10) ═══
  data shape: torch.Size([1115394]), dtype: torch.int64
  train: 1,003,854 字符 (90.0%)
  val:   111,540 字符 (10.0%)
```

- ⚠️ 注意 `dtype=torch.long`——索引/标签必须用整数类型（int64），不能是 float，否则 `nn.Embedding` 和 `F.cross_entropy` 都会报错。
- 💡 为什么要留 10% 的**验证集**？因为我们不想要一个"死记硬背"莎士比亚的模型，而想要一个能**泛化**、能"编造"莎士比亚风格文本的模型。验证集从头到尾不参与训练，用来检测**过拟合**。

## DataLoader：不喂整篇文本，只采样 chunk

一个重要事实：**我们永远不会把整篇 100 万字符喂给 Transformer**，那在计算上不可行。训练时我们随机从数据里采**小块（chunk）**来训练。

两个关键超参数：

```
block_size：一块里放多少个字符（= 上下文长度 / context length，模型预测下一个字符时最多能看多远）
batch_size：每次并行处理多少个独立的块（为了把 GPU 喂满）
```

### 一个 chunk 里藏着多个样本

关键洞察：一块 9 个字符的连续序列，其实包含了 **8 个训练样本**。因为每个位置都对应一个"用前 t 个字符预测第 t+1 个字符"的样本。[02_bigram_baseline.py](../scripts/02_bigram_baseline.py) 里我们用 `train_data[:block_size + 1]` 实际打印出来：

```
═══ 一个 chunk 内含多个样本 ═══
  取前 8+1 = 9 个字符，它们按顺序偏移形成 8 个训练样本：
    x=[18]                         → y=47
    x=[18, 47]                     → y=56
    x=[18, 47, 56]                 → y=57
    x=[18, 47, 56, 57]             → y=58
    x=[18, 47, 56, 57, 58]         → y=1
    x=[18, 47, 56, 57, 58, 1]      → y=15
    x=[18, 47, 56, 57, 58, 1, 15]  → y=47
    x=[18, 47, 56, 57, 58, 1, 15, 47] → y=58
```

- 🔑 这就是"**一个 chunk 内含多个样本**"：`x` 是前 `block_size` 个字符，`y` 是偏移一位的 `block_size` 个字符（`T, T+1` 偏移），两者都是 `(B, T)`。
- 💡 训练时覆盖"上下文长度从 1 到 block_size"的所有情况，不只是为了计算效率——更重要的是让模型**习惯各种长度的上下文**。这样推理时从一个字符开始也能预测。

### get_batch：随机 offset 采样 + torch.stack

```python
def get_batch(split):
    data_local = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_local) - block_size, (batch_size,))
    x = torch.stack([data_local[i:i + block_size] for i in ix])        # (B,T)
    y = torch.stack([data_local[i + 1:i + block_size + 1] for i in ix])  # (B,T)
    x, y = x.to(device), y.to(device)
    return x, y
```

随机采 `batch_size` 个起始位置 `ix`，每个位置切出 `(i, i+block_size)` 作为 `x`，`(i+1, i+block_size+1)` 作为 `y`，然后用 `torch.stack` 叠成一个 `(B, T)` 的 batch。运行结果：

```
═══ 一个 batch ═══
  X shape: torch.Size([32, 8]) (B=32, T=8)，Y shape: torch.Size([32, 8])
  共 256 个独立样本打包在一个 batch 中
```

- ⚠️ 为什么起始位置上限是 `len(data) - block_size`？因为要保证 `i+block_size` 不越界。
- 💡 batch 里的 32 个 chunk 是**互相独立**的——它们之间不通信。这是后面 attention 笔记 ③ 的伏笔（batch 维度上无通信）。

## Bigram 语言模型：最弱的基线

在进入 Transformer 之前，先实现最简单的语言模型——**Bigram**（双字母模型）。Part 1/2 我们已经深入讲过它，这里快速过一遍。

它只有一张表：`nn.Embedding(vocab_size, vocab_size)`。输入一个 token 索引，直接查表得到"下一个 token 的分数（logits）"。

```python
class BigramLanguageModel(nn.Module):
    """最简单基线：每个 token 只看"我是谁"，token 之间不交流"""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
```

- ⚠️ 预测时 token **完全不看上下文**——只看"我是谁"。例如 token 5 只根据自己是 5 来预测下一个，因为某些字符后面常跟另一些字符，所以能学到一点点规律，但上下文信息全被浪费了。

### 交叉熵损失：为什么 reshape 成 (B*T, C)

交叉熵衡量"logits 对 targets 的预测质量"。我们想要：target 对应的那一维 logits 很高，其它维很低。

但 PyTorch 的 `F.cross_entropy` 对**多维输入**有形状要求：它希望 logits 是 `(样本数, 类别数)`。我们的 logits 是 `(B, T, C)`，所以要先合并 B 和 T 两个维度：

```python
logits = logits.view(B * T, C)   # (B, T, C) → (B*T, C)，把"样本"摊平
targets = targets.view(B * T)     # (B, T)   → (B*T,)
loss = F.cross_entropy(logits, targets)
```

- 🔑 这里的 `C`（channel）就是 **vocab_size = 65**——交叉熵的"类别数"必须是词表大小。之所以要 reshape，是因为交叉熵把**每一个 (batch, time) 位置都当成一个独立的分类样本**。
- ⚠️ `view` 只是改变张量的"视图"，不拷贝内存。`-1` 也能让 PyTorch 自动推断，但显式写 `B * T` 更清晰。

初始 loss 以及理论下限：

```
═══ 初始 loss ═══
  初始 loss: 4.7417
  理论下限: -ln(1/65) = ln65 ≈ 4.17（完全均匀分布时的损失）
```

- 💡 为什么理论下限是 4.17？如果模型完全随机、对 65 个字符一视同仁（每个概率 1/65），负对数似然就是 `-ln(1/65) = ln65 ≈ 4.17`。我们初始 4.74 略高于它（原视频里是 ~4.87），说明初始预测还带点"偏差"（在错误方向上自信）。这个差距由随机初始化决定，训练会把它纠正过来。

### generate：采样续写

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)                  # 只看最后一个 token 就够（bigram）
        logits = logits[:, -1, :]                 # (B, C)
        probs = F.softmax(logits, dim=-1)         # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)   # (B, T+1)
    return idx
```

流程：取**最后一个位置**的 logits → `softmax` 变概率 → `torch.multinomial` 按概率采样 1 个 token → 拼回序列 → 重复 `max_new_tokens` 次。

- ⚠️ `generate` 里 `self(idx)` 不传 `targets`，所以 `forward` 里 `targets=None` 分支要返回 `loss = None`（"有 targets 算 loss，没 targets 只给 logits"）。
- 💡 为什么 bigram 也要把整个序列喂进去再只取最后一位？因为我们想让 `generate` 函数**保持不变**，等以后模型真的会看历史了，这个函数就"自动变聪明"。现在看起来很傻，将来会派上用场。

### 训练循环 + AdamW

标准的三大步，以及**优化器换成 AdamW**：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

- 🔑 三大步顺序是固定的：`zero_grad()`（清空上一步梯度）→ `backward()`（算梯度）→ `step()`（更新参数）。
- 💡 **AdamW vs SGD**：makemore 前几课用的都是最朴素的 SGD（随机梯度下降）；AdamW 更先进、更流行，几乎自动调好。**小的网络可以用很大的学习率**（比如 0.1 甚至更高），大网络常用 `3e-4` 这种量级。

训练 1500 步的真实日志：

```
═══ 训练 (AdamW, lr=0.01, 1500 步) ═══
  step    0: train loss 4.7618, val loss 4.7741
  step  500: train loss 2.5877, val loss 2.6073
  step 1000: train loss 2.4985, val loss 2.5239
  step 1499: train loss 2.4972, val loss 2.5017
```

- 🔑 训练后 val loss ≈ **2.50**。这比初始的 ~4.8 好很多，但离"好语言模型"还很远。（原视频跑出的也是 ≈2.5，不同超参/种子会有小差异。）

训练后的生成结果（200 个字符）——已经有零星的英文碎片，但没有真正的词汇/语法结构：

```
CI n:
Wiwist Rorer boomatowig d:
Son cotheraris
STun:
S:
Th y pr;

Mims Fo;

Bony s,
Stece butis y DUSmou s mularet w ke s ur, o aly agre d ndont h seld'lysen t's 'd ffere bl mureligescheple ord otak
```

> 💡 一眼看出问题：bigram 只用了最后一个字符，**上下文完全被浪费**。比如它看到 `Th` 能猜到下一个是 `e`，但要猜出"这句莎士比亚在说什么"，必须看更长的历史。

**这就是 Transformer 要解决的问题：让 token 之间互相交流，根据上下文做更好的预测。**

## 学完本部分你能...

- ✅ 说清楚 ChatGPT = 语言模型 = 逐 token 预测，GPT 三个字母的含义
- ✅ 讲出 Transformer 的起源（2017《Attention is All You Need》，机器翻译背景）
- ✅ 读懂并手写字符级 `encode`/`decode`，理解它与 BPE/sentencepiece 的取舍
- ✅ 划分 train/val（90/10），理解验证集测过拟合的作用
- ✅ 讲清 `block_size`/`batch_size`、一个 chunk 含多样本、`get_batch` 采样
- ✅ 实现 Bigram 基线，理解交叉熵为什么 reshape 成 `(B*T, C)`
- ✅ 看懂 `generate` 采样流程，用 AdamW 训练到 loss ≈ 2.5

## 课后练习

<details>
<summary>Q1: 为什么训练时要让模型覆盖"上下文长度从 1 到 block_size"的所有情况？</summary>
A: 有两个原因。一是效率：一个 chunk 里的每个位置都能贡献一个训练样本，白白浪费很可惜。二是泛化：推理时我们可能从一个字符就开始生成，模型必须习惯各种长度的上下文。如果只训练"固定长度 block_size"的样本，推理开头那些短上下文它就应付不来。
</details>

<details>
<summary>Q2: 为什么交叉熵要把 logits 从 (B, T, C) reshape 成 (B*T, C)？targets 为什么也要 reshape？</summary>
A: PyTorch 的 `F.cross_entropy` 期望的输入是 `(样本数, 类别数)` 和 `(样本数,)`。我们把 (batch, time) 两个维度合并成一个"样本数"维度，每个位置的 65 维 logits 成为一个独立的分类样本，targets 也摊平成 (B*T,)。`view` 只改视图不拷贝内存，很高效。
</details>

<details>
<summary>Q3: 假如把 tokenizer 换成 BPE（词表 50K），其它训练代码要改吗？block_size 应该变吗？</summary>
A: `encode`/`decode` 换掉后，训练/模型代码几乎不用改，因为模型只依赖 `vocab_size`（由新词表决定，变为 ~50K）和 token 序列。但通常应该**增大 block_size**：BPE 的序列更短，相同的上下文长度能覆盖更多"真实内容"；同时因为词表变大、每个位置的信息量变大，也可能需要更大的模型。这就是"词表大小 vs 序列长度"的权衡在实践中的体现。
</details>

## 下一步

数据管线通了、Bigram 基线也建好了（val loss ≈ 2.5）。下一步，我们进入本课的核心——**attention**。先用一个玩具演示"用矩阵乘法做加权聚合"这个数学技巧，再实现真正的 self-attention。

👉 [02 — Attention 从零开始](02_attention_from_scratch.md)
