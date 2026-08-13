# 作业 6：Transformer/GPT — 从零实现一个 decoder-only Transformer

> **对应教程**：Part 6 — Let's build GPT: from scratch（makemore Part 6）
>
> **前置**：建议先完成作业 3（BatchNorm，理解 LayerNorm 时需对照）与作业 5（训练循环）

---

## 📋 概述

本作业带你从零实现一个 **decoder-only Transformer**（也就是 GPT）。你将逐步实现：

1. 字符级 tokenizer 与 train/val 划分
2. 数据加载器 `get_batch`
3. Bigram 基线模型 + 交叉熵 + 文本生成
4. Transformer 的核心——单头 Self-Attention（key/query/value）
5. 🌟 完整 Transformer Block（多头 + 前馈 + 残差 + pre-norm LayerNorm）

完成本作业后，你应该能够：

- 理解字符级 tokenizer 的 encode/decode 往返，以及 train/val 划分的意义
- 理解 `block_size`（上下文长度）与 `batch_size`（并行序列数）
- 理解交叉熵为什么要求把 logits reshape 成 `(B*T, C)`
- 理解 Self-Attention 的 6 条笔记：通信机制、无空间性、batch 隔离、encoder/decoder 遮罩、self/cross 区别、scaled attention 控方差
- 理解残差连接与 pre-norm LayerNorm 在深层网络中的作用

---

## 🔧 环境准备

### 依赖

```bash
pip install torch
```

### 数据

数据文件位于 `../../data/input.txt`（tiny Shakespeare，~1.1M 字符，65 个唯一字符）。
**注意**：这里用的是 `input.txt`（不是之前作业的 `names.txt`）。

### 文件结构

```
assignments/assignment_6/
├── assignment.md               # 本文件
├── transformer_exercises.py    # 👈 你需要编辑的文件
└── test_transformer_exercises.py  # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_6
python test_transformer_exercises.py
# 或（pytest 兼容）
pytest test_transformer_exercises.py
```

> 测试是**属性测试**：只检查 shape / dtype / 数学不变量（三角遮罩、softmax 行和、方差范围），
> 不检查精确数值。未实现的题目（返回 `None`）会被**优雅跳过**，不会报错。

---

## 📝 题目列表

### 题 1：字符级 Tokenizer 与 train/val 划分（基础）

**函数**：`exercise_1_tokenize(text)`

**目标**：把原始文本变成可训练的数据。

**要求**：

- 构建排序后的唯一字符表 `chars`，`vocab_size = len(chars)`（tiny Shakespeare 为 65）
- 构建 `stoi`（字符→整数）与 `itos`（整数→字符）
- 实现 `encode(s)`：字符串 → 整数列表；`decode(l)`：整数列表 → 字符串（往返一致）
- 用 `torch.tensor(encode(text), dtype=torch.long)` 把全文转成 1D 整数张量 `data`
- 前 90% 作 `train_data`，后 10% 作 `val_data`（检测过拟合）

**返回**：一个 dict，包含键 `chars / vocab_size / stoi / itos / encode / decode / data / train_data / val_data`

**验证**：
```python
text = open('../../data/input.txt', encoding='utf-8').read()
r = exercise_1_tokenize(text)
print(r['vocab_size'])                       # 65
print(r['decode'](r['encode']('hi there')))  # 'hi there'（往返一致）
print(r['data'].shape)                       # (1115394,)
print(len(r['train_data']) / len(r['data'])) # ≈ 0.9
```

**思考**：
- 语言模型只能输出它"见过"的字符，为什么 vocab 必须来自数据本身？
- 如果 `data` 的取值不在 `[0, vocab_size)` 内会怎样？

---

### 题 2：get_batch（基础）

**函数**：`exercise_2_get_batch(data, block_size, batch_size, seed=1337)`

**目标**：不把整篇文本喂入 Transformer，而是每次随机采样若干个长度为 `block_size` 的 chunk。

**要求**：

- `torch.manual_seed(seed)` 固定随机性
- 生成 `batch_size` 个随机起始偏移 `ix = torch.randint(len(data) - block_size, (batch_size,))`
- `x = torch.stack([data[i:i+block_size] for i in ix])`：每个偏移取一段连续子串
- `y = torch.stack([data[i+1:i+block_size+1] for i in ix])`：偏移 +1
- 返回 `(x, y)`，两者 shape 都是 `(batch_size, block_size)`，dtype `long`

**关键不变量**：`y[b, t] == x[b, t+1]`（目标 = 输入的后移一位）。
一个 9 字符的 chunk 内含 8 个 `(x, y)` 训练样本。

**验证**：
```python
r = exercise_1_tokenize(text)
x, y = exercise_2_get_batch(r['train_data'], block_size=8, batch_size=4, seed=1337)
print(x.shape, y.shape)          # (4, 8) (4, 8)
print(torch.equal(y[:, :-1], x[:, 1:]))   # True（后移一位）
```

**思考**：
- 为什么训练时用「长度 1 到 block_size 的各种上下文」，而不是只用完整 block_size？
- 为什么 batch 里的不同 chunk 之间不通信？

---

### 题 3：Bigram 基线 + 交叉熵 + generate（基础）

**函数**：`exercise_3_bigram_model(vocab_size)` → 返回 `BigramLanguageModel` 实例

**类**：`BigramLanguageModel(nn.Module)`

**目标**：实现最简单的语言模型基线——每个 token 只看"我是谁"，token 之间不交流。

**要求**：

- `__init__`：`self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)`（查表直接当 logits）
- `forward(idx, targets=None)`：
  - `logits = self.token_embedding_table(idx)` → shape `(B, T, vocab_size)`
  - 有 targets 时，把 logits reshape 成 `(B*T, C)`、targets 展平成 `(B*T)`，再 `F.cross_entropy`
  - targets 为 `None` 时 `loss = None`
  - 返回 `(logits, loss)`
- `generate(idx, max_new_tokens)`：
  - 循环 `max_new_tokens` 次：取 `logits[:, -1, :]` → `F.softmax` → `torch.multinomial(probs, num_samples=1)` → `torch.cat` 拼到时间维
  - 注意：`generate` 调用时 `targets=None`，logits 保持 `(B, T, C)` 三维形状，所以 `[:, -1, :]` 切片是合法的
  - 返回 shape `(B, T + max_new_tokens)` 的 `long` 张量

**验证**：
```python
model = exercise_3_bigram_model(65)
xb = torch.randint(0, 65, (32, 8)); yb = torch.randint(0, 65, (32, 8))
logits, loss = model(xb, yb)
print(logits.shape)          # (256, 65)  ← 有 targets 时 reshape 成 (B*T, vocab)
print(loss.item())           # ≈ 4.6，应接近 ln65 ≈ 4.17
out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
print(out.shape)             # (1, 21)
```

**参考数字**（原视频）：Bigram 初始 loss ≈ 4.87，理论下限 `-ln(1/65) = ln65 ≈ 4.17`；训练后 ≈ 2.5。

**思考**：
- 为什么 PyTorch 的 cross_entropy 要求 logits 是 `(B*T, C)` 而不是 `(B, T, C)`？
- 训练时用 AdamW 而不用 SGD，对学习率有什么影响？

---

### 题 4：单头 Self-Attention（核心，Transformer 的核心）

**函数**：`exercise_4_head(head_size, n_embd, block_size)` → 返回 `SelfAttentionHead` 实例

**类**：`SelfAttentionHead(nn.Module)`

**目标**：实现 Transformer 最核心的组件——单头自注意力。每个 token 发出
**query**（我在找什么）、**key**（我有什么）、**value**（若有趣，我传达什么），
按亲和力聚合过去的信息。

**要求（分两部分）**：

**(a)** 实现模块级函数 `scaled_dot_product_affinity(q, k)`：

- `q, k` shape `(B, T, head_size)`
- 返回 `wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)`，shape `(B, T, T)`
- 这就是 scaled attention：除以 `sqrt(head_size)` 控制方差

**(b)** 实现 `SelfAttentionHead`：

- `__init__`：`key / query / value` 三个 `nn.Linear(n_embd, head_size, bias=False)`；
  用 `register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))` 存三角遮罩
- `forward(x)`，`x` shape `(B, T, n_embd)`：
  1. `k = self.key(x)`、`q = self.query(x)`，shape `(B, T, head_size)`
  2. `wei = scaled_dot_product_affinity(q, k)`（亲和力，数据依赖）
  3. `wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))`（因果遮罩，未来不看过去）
  4. `wei = F.softmax(wei, dim=-1)`（每行归一化，和为 1）
  5. `self.wei = wei`（**务必保存**，测试需要检查遮罩）
  6. `v = self.value(x)`；返回 `wei @ v`，shape `(B, T, head_size)`

**验证**：
```python
head = exercise_4_head(head_size=16, n_embd=32, block_size=8)
x = torch.randn(4, 8, 32)
out = head(x)
print(out.shape)                          # (4, 8, 16)
print(head.wei.sum(-1)[0, 0])             # ≈ 1.0（softmax 行和）
triu = torch.triu(torch.ones(8, 8), diagonal=1).bool()
print((head.wei[0][triu] == 0).all())     # True（严格上三角为 0）
```

**关于 scaled attention（为什么除以 sqrt(head_size)）**：
若 q/k 是 unit gaussian，则 `q @ k^T` 的方差 ≈ `head_size`。不缩放时 softmax 会太尖锐
（趋近 one-hot，初始化时每个 token 只聚合一个 token）；除以 `sqrt(head_size)` 后方差 ≈ 1，
softmax 保持"扩散"。测试会验证：未缩放 `std ≈ sqrt(head_size)`，缩放后 `std ≈ 1`。

**思考**：
- attention 为什么是"通信机制"？wei 矩阵代表了什么？
- 为什么 `bias=False`？

---

### 题 5：完整 Transformer Block（🌟 拓展）

**函数**：`exercise_5_transformer_block(n_embd, n_head, block_size)` → 返回 `Block` 实例
（未实现时返回 `None`，测试会优雅跳过）

**类**：`MultiHeadAttention(num_heads, head_size, n_embd, block_size)`、
`FeedForward(n_embd)`、`Block(n_embd, n_head, block_size)`

**目标**：把单头组装成完整 Transformer block——"通信"（多头注意力）+ "计算"（前馈）+ 残差 + pre-norm LayerNorm。

**要求**：

1. **`MultiHeadAttention`**：多个 `SelfAttentionHead` 并行，沿通道维 `torch.cat`，再用 `self.proj = nn.Linear(head_size * num_heads, n_embd)` 投影回 `n_embd`（类比分组卷积）
2. **`FeedForward`**：`nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))`（内层 4 倍，论文 512→2048 的规律）
3. **`Block`**：`head_size = n_embd // n_head`；pre-norm 结构：
   ```python
   x = x + self.sa(self.ln1(x))    # 先 LayerNorm 再 attention
   x = x + self.ffwd(self.ln2(x))  # 先 LayerNorm 再 ffwd
   ```
   残差连接 = 梯度"超高速公路"：反传时加法把梯度均分给两个分支，从 loss 直达输入。

**验证**：
```python
block = exercise_5_transformer_block(n_embd=32, n_head=4, block_size=8)
block.eval()
x = torch.randn(4, 8, 32)
out = block(x)
print(out.shape)   # (4, 8, 32)（shape 保持）
# 残差恒等：参数清零后 block(x) ≈ x（测试会自动检查）
```

**思考**：
- pre-norm 和原论文的 post-norm 有什么区别？
- 为什么残差让深层网络可优化？

---

## ✅ 提交检查清单

- [ ] 所有 4 道基础题通过测试
- [ ] 拓展题（题 5）已尝试
- [ ] 能回答每道题后面的「思考」问题
- [ ] 代码中添加了必要的注释说明你的理解

---

## 💡 学习建议

1. **按顺序做**：题 1 → 题 2 → 题 3 → 题 4 → 题 5。前 4 题不依赖彼此，但概念递进
2. **先跑测试再写代码**：`python test_transformer_exercises.py` 会告诉你哪些题没实现（跳过）、哪些实现有 bug（失败）
3. **对照参考实现**：`courses/Part6_transformer/gpt.py` 与 `courses/Part6_transformer/scripts/04_self_attention.py` 有完整实现，看不懂时可参考（注意：参考实现的 `forward` 有 targets 时返回 `(B*T, vocab)` 的 reshape 后 logits，无 targets 时返回 `(B, T, vocab)` 原始形状）
4. **loss 演进参考**（原视频数字）：Bigram 2.5 → 单头 self-attn 2.4 → 多头 2.28 → +前馈 2.24 → +残差 2.08 → +LayerNorm 2.06 → Scale up 1.48
5. **遇到 shape 错误先打印 shape**：Transformer 的核心就是 `(B, T, C)` 三种维度的流转

---

## 🤔 思考题

**Q1：** 为什么 scaled attention 要除以 `sqrt(head_size)`？不除会怎样？

<details>
<summary>💡 提示</summary>

如果 query 和 key 是 unit gaussian（零均值、单位方差），那么两个向量的内积
`q·k = Σᵢ qᵢkᵢ` 的方差 ≈ `head_size`（head_size 个独立同分布项之和）。head_size 越大，
`q@k^T` 的值域就越宽（正负越大）。

问题在于这些亲和力要喂进 softmax：softmax 对"很大的值"很敏感，会迅速坍缩成
**one-hot**（趋近于只选最大的那个）。初始化时我们希望注意力是**扩散**的（每个 token
大致均匀地聚合多个 token），而不是一开始就尖峰化。除以 `sqrt(head_size)` 后，
`q@k^T / sqrt(head_size)` 的方差 ≈ 1，softmax 保持扩散、训练更稳定。

如果**不除**，head_size 越大，softmax 越早 one-hot 化——尤其初始化时每个 token 只会
从另一个 token 聚合信息，通信就失效了。

</details>

**Q2：** 为什么 Transformer 需要位置编码？和卷积（Part 5）相比有何不同？

<details>
<summary>💡 提示</summary>

Attention 本身是**对集合（set）操作**的，它只按"亲和力"聚合，对节点的**空间/位置顺序完全没有概念**。
同一个 token 放在位置 0 还是位置 99，对 attention 来说毫无区别（只要内容相同）。

所以我们用 `position_embedding_table` 给每个位置一个可学习的向量，`tok_emb + pos_emb`
广播相加，把"位置信息"注入到每个 token 的表示里，模型才能知道先后顺序。

对比卷积：卷积核在空间上是**有固定感受野和相对位置**的（比如 3×3 卷积天然知道上下左右），
所以 CNN 自带空间性；而 attention 是"全连接集合"式的，必须**显式**加入位置信息。

</details>

**Q3：** encoder block 和 decoder block 有什么区别？我们实现的 GPT 是哪一种？

<details>
<summary>💡 提示</summary>

区别只在**那一行遮罩**：

- **decoder block**：保留 `wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))`，
  形成**三角遮罩**，未来 token 不能看向过去——这是**自回归**（逐 token 生成，
  预测下一个时不能偷看答案）。我们实现的 GPT 就是 **decoder-only**（无 encoder、无 cross-attention）。
- **encoder block**：删除遮罩那一行，所有节点**全连通互相通信**，适合做分类/编码任务
  （比如读取整个法语句子后总结语义）。

原论文"Attention is All You Need"是机器翻译，用的是 **encoder-decoder** 结构：
encoder 读法语（无遮罩、全连通），decoder 用 **cross-attention**（Q 来自 decoder，
K/V 来自 encoder 的输出）条件生成英语。我们没做翻译，只是"无条件的文档补全器"，
所以只需 decoder-only。

</details>

**Q4：** 为什么说 attention 是"通信机制"？`wei` 矩阵到底代表什么？

<details>
<summary>💡 提示</summary>

可以把每个 token 想象成**有向图里的一个节点**，边代表"谁可以向谁聚合信息"。
Attention 做的事情是：每个节点通过加权求和，从指向它的节点那里收集信息——
`out = wei @ v`，`wei` 的第 (t1, t2) 个元素就是"token t1 从 token t2 聚合多少"的**权重**。

关键是这个权重是**数据依赖**的：不是固定的下三角平均，而是 `wei = q @ k^T / sqrt(head_size)`
算出来的。每个 token 发出 query（我在找什么）和 key（我有什么），
两个 token 的 query/key 越"契合"，内积越大、亲和力越高，就聚合更多对方的信息。
所以 attention = 每个节点**主动寻找自己感兴趣的信息**并聚合过来——这就是"通信"。

对比最弱的通信：Script 3 的 bag-of-words 平均（固定权重、无差别平均），
attention 则是**动态、有选择性**的通信。

</details>

**Q5：** BPE 和字符级 tokenizer 有什么权衡？为什么我们选字符级？

<details>
<summary>💡 提示</summary>

这是个**词表大小 vs 序列长度**的 trade-off：

- **字符级（本作业）**：词表很小（65 个），encode/decode 极简单；代价是序列**很长**
  （1 个字符 1 个 token，整个数据集 ~1M token），模型要学习很长的上下文才能看到一个"单词"。
- **subword / BPE（OpenAI tiktoken，GPT-2 用）**：词表 ~50K tokens，把常见词/子词拼成
  单个 token（如 "ing"、"the" 各占一个 token）。序列**显著变短**（信息密度高），
  但需要额外的学习/训练阶段来构建词表，而且稀有词的边界不好处理。
- **sentencepiece（Google）**：另一种常用的 subword tokenizer，原理类似。

工业界几乎都用 subword tokenizer（短序列、高密度）；课程选字符级纯粹是**教学简单**
——`sorted(set(text))` 两行就建好词表，让我们把注意力集中在 Transformer 架构本身。
真实系统的复杂度在 tokenizer，而不只在模型。

</details>

---

*Good luck! 🚀 完成作业 6 后，你就拥有一个能"无限生成莎士比亚"的 mini-GPT 了。*
