# 02 — Attention 从零开始：数学技巧、Self-Attention 与 6 条笔记

> 🎯 本课核心：attention 就是"token 之间按重要程度通信"。我们从最笨的 for 循环平均，一路进化到数据依赖的 self-attention。

## 从 Bigram 的局限出发

上一章 Bigram 训练到 val loss ≈ 2.50，生成结果只有零星碎片。原因很直白：**token 之间完全不交流**，每个位置只用自己的身份预测下一个字符。

要让 token 交流，最简单的方式是——**让每个 token 看看过去的信息，聚合起来帮助预测**。但怎么聚合、聚合多少，很有讲究。

这一章先玩一个数学技巧（[03_attention_trick.py](../scripts/03_attention_trick.py)），再实现真正的 self-attention（[04_self_attention.py](../scripts/04_self_attention.py)）。

## Part A：attention 的数学技巧

### 玩具问题设定

设一个张量 `x`，形状 `(B, T, C)`：`B` 个序列、`T` 个 token、每个 token 有 `C` 维信息。我们想让每个 token **聚合"自己及之前所有 token"的信息**。

```python
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
```

注意：token 只能看**过去和自己**，不能看未来——因为我们要预测未来，不能提前偷看答案。

### v1：for 循环 bag-of-words 平均（最弱，但直观）

对每个序列、每个时间步，把当前及之前的 token 求平均：

```python
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]        # (t+1, C)：当前及之前的 token
        xbow[b, t] = xprev.mean(0)  # 对时间维求平均 → bag of words
```

- 🔑 **bag of words（词袋）**：把一堆向量简单地平均成一条特征向量。这是**最弱的聚合方式**——把所有 token "糊"在一起，丢掉了它们的位置和顺序信息。但它直观，是理解一切的起点。

### v2：用矩阵乘法做加权聚合（数学技巧）

**核心技巧：加权聚合可以用矩阵乘法实现。** 如果有个矩阵 `wei`，其中 `wei[t, i]` 表示"第 t 个 token 聚合第 i 个 token 的权重"，那么 `wei @ x` 就是在做加权和。

用 `torch.tril` 造一个下三角全 1 矩阵：

```python
tril = torch.tril(torch.ones(T, T))
```

运行结果（T=8 的下三角矩阵，`1` 表示聚合该 token，`0` 表示忽略）：

```
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        ...
        [1., 1., 1., 1., 1., 1., 1., 1.]])
```

- 💡 下三角的"几何含义"：第 t 行只在 ≤ t 的位置是 1，其它是 0——**未来（t 之后）的 token 不参与聚合**。这是自回归（autoregressive）的雏形。

把每行归一化成和为 1（变成"平均"），再乘上 x：

```python
wei = tril
wei = wei / wei.sum(1, keepdim=True)  # 每行归一化成和为 1 → 变成"平均"
xbow2 = wei @ x                        # (T,T) @ (B,T,C) → 批矩阵乘法 (B,T,C)
```

归一化后的权重（每行和为 1）：

```
tensor([[1.0000, 0.0000, ...],
        [0.5000, 0.5000, ...],
        [0.3333, 0.3333, 0.3333, ...],
        ...
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```

- ⚠️ `(T,T) @ (B,T,C)`：PyTorch 看到维数不匹配，会把它当**批矩阵乘法**——在 batch 维上对每个序列独立做 `(T,T) @ (T,C)`，得到 `(B,T,C)`。这正是"batch 间不通信"的数学基础。
- 🔑 第 t 行是 `1/t` 均匀分布 → 第 t 个 token 的结果等于前 t 个 token 的**平均**，和 v1 完全一样。用 `torch.allclose` 验证：

```
v1 与 v2 等价 (torch.allclose): True
```

### v3：softmax 版本（亲和力 + 遮罩，重点记忆）

v3 换一个视角：把权重矩阵当成"**亲和力（affinity）**"，初始全 0（无差异），用 `masked_fill` 把未来置为 `-inf`，再让 `softmax` 把它归一化成"概率"：

```python
wei = torch.zeros((T, T))                  # 亲和力矩阵，初始全 0
wei = wei.masked_fill(tril == 0, float('-inf'))  # 未来禁连 → -inf
wei = F.softmax(wei, dim=-1)               # 每行 softmax → 行和为 1
xbow3 = wei @ x
```

- 💡 `softmax(0)=1`，`softmax(-inf)=0`：全 0 行经 softmax 后正好变成均匀分布，和 v2 的归一化结果一样。验证：

```
v2 与 v3 等价 (torch.allclose): True
v1 与 v3 等价 (torch.allclose): True
```

> 🔑 记住 v3：**wei = 亲和力矩阵，softmax 把每一行归一化成概率，`wei @ x` 按亲和力对过去信息加权聚合。**
>
> v3 比 v2 更值得记住，因为它**可扩展**：v2 里权重是写死的 `1` 和 `0`，而 v3 里我们随时可以把 `wei` 换成"数据算出来的"值——这就预告了 self-attention。

### 数学技巧小结

```
v1 for 循环平均        v2 矩阵乘法(tril)        v3 masked_fill + softmax
    │                      │                        │
    └────────── 三者数学等价（allclose 验证） ──────┘
                   │
                   ▼
      加权聚合 = 下三角权重矩阵 @ 数据
      下三角遮罩 = 未来不看向过去
      wei 将来 = 数据依赖的亲和力（Self-Attention）
```

## Part B：self-attention 单头

### 代码清理：引入 n_embd 与 lm_head

进入 `04_self_attention.py` 前，先做两处清理：

1. **去掉 `vocab_size` 参数**：`vocab_size` 已经是全局变量，不用到处传。
2. **引入中间维度 `n_embd`**：不让 embedding 直接输出 logits，而是先输出 32 维的 token embedding，再用一个线性层 `lm_head` 投影到 65 维词表。

```python
n_embd = 32
...
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token → 32 维向量
self.lm_head = nn.Linear(n_embd, vocab_size)                    # 32 维 → 65 维 logits
```

### 位置编码：attention 没有空间概念，我们必须手动加上

attention 是对"一组向量"做操作，**默认不知道每个 token 在序列里的位置**。所以我们用第二张 embedding 表给每个**位置**也学一个向量：

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)

# forward 里：
tok_emb = self.token_embedding_table(idx)          # (B,T,C)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
x = tok_emb + pos_emb                              # 广播相加 (B,T,C)
```

- 🔑 **广播（broadcasting）**：`(B,T,C) + (T,C)`，PyTorch 把 `(T,C)` 右对齐，前面补一个维度变成 `(1,T,C)`，再沿 batch 维广播 → 每个 token 的向量 = "我是谁"（token embedding）+ "我在哪"（position embedding）。
- 💡 为什么 Bigram 阶段这个位置信息没用？因为 bigram 是平移不变的（translation invariant），token 在第 5 位还是第 2 位无所谓。等 attention 真正"看上下文"了，位置信息就开始起作用——因为"第 3 个位置出现的是元音"和"第 7 个位置出现的是元音"，在聚合时意义不同。

### Head：单个 self-attention 头

现在实现本课核心。每个 token 发出**三个向量**：

- **query（q）**：我在找什么？
- **key（k）**：我有什么？
- **value（v）**：如果你觉得我有趣，我把什么传达给你？

亲和力 = query 与所有 key 的内积，然后遮罩、softmax、按权重聚合 value：

```python
class Head(nn.Module):
    """一个自注意力头：每个 token 发出 query（找什么）、key（有什么）、
    value（若有趣就传达什么），按亲和力聚合过去的信息"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril 不是可训练参数，用 register_buffer 注册（会随模型移动设备）
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        # 亲和力 = query 与所有 key 的内积 → 数据依赖
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,T)，scaled
        # 遮罩：未来不能看向过去（decoder 三角遮罩）
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # 每行归一化成概率 (B,T,T)
        v = self.value(x)             # (B,T,head_size)
        out = wei @ v                 # 加权聚合 → (B,T,head_size)
        return out
```

拆解关键几行：

- `q @ k.transpose(-2, -1)`：`(B,T,head_size) @ (B,head_size,T) → (B,T,T)`。每个 token 的 query 和所有 token 的 key 做内积，得到一个 `T×T` 的**亲和力矩阵**。
- `* k.shape[-1] ** -0.5`：即除以 `sqrt(head_size)`，**scaled attention**（笔记 ⑥ 详述）。
- `masked_fill(self.tril[:T, :T] == 0, float('-inf'))`：把未来置 `-inf`，softmax 后它们变成 0。
- `wei @ v`：按亲和力对 value 加权求和。
- `bias=False`：K/Q/V 只是投影，通常不加偏置。
- `register_buffer('tril', ...)`：`tril` 不是可训练参数，但它必须作为模块的一部分（这样 `to(device)` 时它会跟着走）。这是 PyTorch 的 buffer 机制。

> 🔑 你可以把 `x` 想成 token 的"私密信息"；为了**这个头**的通信，token 额外发布"介绍信"：`key`（我有什么）、`query`（我找什么）、`value`（若你对我感兴趣，你会从我这儿拿到的东西）。聚合发生时，聚合的是 `value`，不是原始的 `x`。

#### 数据依赖的亲和力：实跑演示

[04_self_attention.py](../scripts/04_self_attention.py) 里用随机输入实跑了一个 `Head(n_embd)`，打印出亲和力矩阵（每行 = 该 token 对过去各 token 的注意力权重）：

```
亲和力矩阵（每行 = 该 token 对过去各 token 的注意力权重）:
tensor([[1.0000, 0.0000, ...],
        [0.5141, 0.4859, ...],
        [0.3271, 0.3374, 0.3356, ...],
        ...
        [0.1136, 0.0961, 0.1207, 0.0845, 0.1033, 0.1394, 0.1377, 0.2048]])
```

注意这和 Part A 的均匀分布（`1/t`）完全不同：**权重是数据依赖的**，有的 token 被重视（如 0.38、0.31），有的被冷落（如 0.06）。这就是 attention 和"简单平均"的本质区别。

#### scaled attention：为什么除以 sqrt(head_size)

```python
vals = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2])
F.softmax(vals, dim=-1).tolist()
# → [0.2047, 0.1516, 0.2500, 0.1676, 0.2262]  接近 0 的小值 softmax（扩散）
F.softmax(vals * 8, dim=-1).tolist()
# → [0.1180, 0.0107, 0.5847, 0.0238, 0.2627]  放大 8 倍后 softmax（尖锐）
```

- 💡 同样的值放大 8 倍，softmax 就从一个"相对均匀"的分布，变成"几乎只认最大值"的 one-hot 分布。
- 如果输入是 **unit gaussian**（零均值、单位方差），那么 `q` 和 `k` 的内积（`wei`）方差大约是 `head_size`。head_size 越大，`wei` 越尖锐。**除以 `sqrt(head_size)` 让方差回到 ≈1**。
- ⚠️ 初始化时我们**不想**让每个 token 只聚合一个 token（softmax 太尖锐）——我们希望一开始是"广撒网"的扩散分布，让网络自己学会该聚焦谁。所以缩放是必需的。

### 插入单头 self-attention 到网络

把 Head 插进语言模型（[04_self_attention.py](../scripts/04_self_attention.py)，这里类名沿用 `BigramLanguageModel`）：

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)       # 单头，head_size = n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_head(x)                                # 自注意力
        logits = self.lm_head(x)                           # (B,T,vocab_size)
        ...
```

- ⚠️ `generate` 里必须**裁剪 idx**：位置表只有 `block_size` 个位置，生成会不断把新 token 拼进序列，一旦超过 `block_size` 就会越界。

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]    # 只保留最后 block_size 个
        ...
```

训练 3000 步的真实日志（val loss 从 ~2.5 降到 ≈2.39）：

```
═══ 训练 (AdamW, lr=0.003, 3000 步) ═══
  step    0: train loss 4.2351, val loss 4.2337
  step  500: train loss 2.5160, val loss 2.4975
  step 1000: train loss 2.4407, val loss 2.4234
  step 1500: train loss 2.4328, val loss 2.4308
  step 2000: train loss 2.3907, val loss 2.4244
  step 2500: train loss 2.3784, val loss 2.4247
  step 2999: train loss 2.3812, val loss 2.3871
```

生成结果（300 个字符）——文本开始有"词组"的影子了：

```
OMONofr atre kchen, ty yed wine nd hiche arstitha heap's sl lis min ius m:
Wh SOPULorim; fome bet an sur t thire bes, wO!
TORDO CESwi, ous, achirery win-
Torsw ps, anst bitheed I apar:
...
```

- 💡 单头 self-attention 让 token 开始按"数据依赖的亲和力"通信，val loss 从 bigram 的 ≈2.5 降到 ≈2.39。还差得远，但方向对了。

## 6 条 attention 笔记（本课核心，逐一展开）

Karpathy 在视频里用 6 条笔记总结了 attention 的本质。这里全部展开：

### 笔记 1：attention 是通信机制

attention 就是一个**有向图**上的通信：节点（token）通过有向边，按权重聚合指向自己的节点信息。

```
有向图：边从"被看"的节点指向"观看"的节点
（每个节点聚合所有"指向自己"的节点的信息）

  ① → ② → ③ → ④ → ⑤ → ⑥ → ⑦ → ⑧
  ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
 自己  ①②  ①②③ ①②③④ ...  ①..⑦ ①..⑧
 只看  只看   只看           （自回归：第 t 个节点
 自己  前2个  前3个          只聚合自己和之前的节点）
```

在我们语言建模的例子里，图是"自回归"的：第 t 个节点只看自己和之前的节点。但**原则上 attention 可以作用在任意有向图上**——它只是一个通用的通信机制。

### 笔记 2：attention 无空间概念，作用在"集合"上

attention 默认把输入当**一组向量（set）**处理，节点之间**没有位置感**。这正是我们前面加"位置编码"的原因——把"我在哪"的信息显式加到向量里。

- 对比 **卷积**（Part 5）：卷积有非常具体的**空间性**——卷积核按空间位置滑动，它天然知道邻域布局。attention 没有这个先天属性，想要"空间"，就得自己加上。
- 💡 这也说明 attention 不关心 token 的绝对位置，只关心它们之间的信息关系——这是它强大（能适配任意序列）的原因，也是它"需要位置编码"的原因。

### 笔记 3：batch 之间不通信

批矩阵乘法 `wei @ v` 在 batch 维上**各自独立**执行。batch 里的 32 个序列是 32 个互不相关的"图"，每个图内部 8 个节点互相通信，图与图之间**绝不交流**。

- 🔑 我们可以把它看作"4 个独立的池子，每个池子里 8 个节点"（batch_size=4 时）。它们共享同一套权重，但数据完全隔离。

### 笔记 4：decoder block 用三角遮罩；encoder block 全连通

我们实现的是 **decoder block**：用 `masked_fill(self.tril == 0, -inf)` 保证**未来不看向过去**。这叫自回归——预测下一个字符时不能提前看到答案。

但这不是唯一选择。如果做**情感分析**之类任务，让所有 token 互相看完全没问题（甚至更好）：

```
decoder block（我们）：           encoder block：
未来不看向过去（三角遮罩）        所有节点全连通（删除遮罩行）
```

> 🔑 实现 encoder 只需要**删除遮罩那一行**。attention 本身不关心连边方式，它支持任意拓扑。我们讲完 6 条笔记后会回来展开 encoder/decoder 的完整图景（见第 04 章）。

### 笔记 5：attention vs self-attention vs cross-attention

- **attention**：最一般的机制——"按亲和力聚合"。
- **self-attention**：K、Q、V **全部来自同一个 X**（"自己看自己"）。我们实现的就是它。
- **cross-attention**：Q 来自 X，但 **K、V 来自另一个独立的外部源**（比如 encoder 的输出）。用于"从旁边拉信息进来"（如翻译时读法语、写英语）。

```
self-attention:              cross-attention:
   X ──► K                  encoder 输出 ──► K / V
   X ──► Q     → wei@v       X (decoder) ──► Q       → wei@v
   X ──► V
```

- 💡 我们这种"只有自己看自己"的注意力，叫 self-attention；但 attention 本身远比这通用。

### 笔记 6：scaled attention——除以 sqrt(head_size) 控制方差

前面已用实际输出演示过：输入 unit gaussian 时，`q@k` 的方差 ≈ `head_size`，值会随 head_size 变大而尖锐；`softmax` 会把尖锐的值推向 **one-hot**，导致"每个 token 只聚合一个 token"。除以 `sqrt(head_size)` 把方差拉回 ≈1，让初始化时的注意力保持**扩散**（每个 token 雨露均沾）。

```
wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5    # k.shape[-1] = head_size
```

- 🔑 一句话：**缩放 = 在初始化时保护 softmax 的"温和"**。训练中网络自己学会该专注谁，但起点必须温和。

## Multi-Head：多头并行

### 分组卷积的类比

attention 很好，但**一个头只有一个通信通道**。token 想找元音、想找位置、想找标点——这么多"话题"挤在一个 32 维通道里太挤。解决办法：**多个头并行，每个头一个小通道**。

类比：卷积里的 **group convolution（分组卷积）**——不做一个大的卷积，而是分成多组小卷积。多头 self-attention 也是同样的思想。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)  # 投影回残差通路

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接各头输出
        out = self.proj(out)
        return out
```

- 🔑 `head_size = n_embd // n_head`：把 `n_embd` 均分给 `n_head` 个头。例如 `n_embd=32, n_head=4` → 每个头 `head_size=8`。每个头输出 8 维，4 个头拼接回 32 维。
- `proj`：把拼接后的 `(head_size * num_heads)` 维投影回 `n_embd` 维，为后面接残差连接做准备（第 03 章会用到）。

```
单头:  1 个 32 维通信通道
多头:  4 个 8 维并行通道，拼接回 32 维
       ┌─── head1 (8维) ───┐
       ├─── head2 (8维) ───┤  → cat → (32维) → proj → (32维)
       ├─── head3 (8维) ───┤
       └─── head4 (8维) ───┘
```

### 多头的效果（脚本 05，Phase 1）

[05_multihead_feedforward.py](../scripts/05_multihead_feedforward.py) 的 Phase 1 用 `nn.Sequential(MultiHeadAttention(n_head, head_size))` 训练 400 步：

```
═══ 三阶段演进（val loss 逐步下降）═══
  [Phase1 多头     ] train loss 2.3813, val loss 2.4545
```

- ⚠️ 我们的 CPU 小规模（400 步）下多头 val loss ≈ **2.45**，和单头脚本 04 的 ≈2.39 在同一量级——**多头"多个通信通道"的价值，在数据量更大、网络更深时才充分显现**（原视频中 2.4 → 2.28）。所以不要只看这一个数字，看它带来的结构性收益：更多独立的通信通道，能同时捕捉多种"话题"。

## 学完本部分你能...

- ✅ 用三种数学等价的方式实现"过去 token 的加权聚合"（for 循环 / 矩阵乘法 / softmax）
- ✅ 解释为什么下三角遮罩保证"未来不看向过去"
- ✅ 写出单头 self-attention（query/key/value、亲和力、缩放、遮罩、`register_buffer`）
- ✅ 解释为什么 attention 需要位置编码，以及 `(B,T,C)+(T,C)` 的广播机制
- ✅ 把 6 条 attention 笔记逐条讲清楚
- ✅ 实现 Multi-Head（分组卷积类比、`head_size = n_embd // n_head`、`proj`）

## 课后练习

<details>
<summary>Q1: 为什么 v3 的 `wei.masked_fill(tril == 0, float('-inf'))` 之后用 softmax，而不是直接除以行和？</summary>
A: 除以行和（v2）只能处理"权重都是非负数且我们手动归一化"的情况。softmax 版本（v3）更通用：它可以处理任意的实数值亲和力（包括负数），把"任意分数"变成"归一化的非负概率"。更重要的是，它允许权重**不是写死的 1/0**，而是由数据算出来的值——这为 self-attention 的数据依赖亲和力铺平了路。
</details>

<details>
<summary>Q2: 假如把 Head 里的 `* k.shape[-1] ** -0.5` 删掉，训练会出什么问题？为什么？</summary>
A: 如果输入是 unit gaussian，`q @ k` 的方差约为 head_size。head_size 越大，`wei` 里正负值越大，`softmax` 会把每一行推向 one-hot——每个 token 几乎只聚合一个 token。初始化时我们想要的是"扩散"的注意力，这样才能让梯度均匀流动、网络自由学习聚焦谁。缩放让 `wei` 的方差回到 ≈1，保护 softmax 的温和。
</details>

<details>
<summary>Q3: cross-attention 和 self-attention 的区别是什么？为什么机器翻译（法语→英语）需要 cross-attention？</summary>
A: self-attention 的 K/Q/V 全部来自同一个 X（自己看自己）；cross-attention 的 Q 来自当前序列（decoder），而 K/V 来自另一个外部源（encoder 编码后的法语）。翻译时，decoder 要生成英语，但必须"参考"已经读完整句的法语信息——这个"从旁边拉信息"的动作就是 cross-attention。我们第 04 章会详细展开。
</details>

## 下一步

single-head 和 multi-head 都实现了，但网络还很浅（token 看完彼此后立刻就去预测）。下一步我们给 token 一个"思考"的步骤（FeedForward），再用**残差连接**和 **LayerNorm** 让网络能堆到很深。

👉 [03 — Transformer Block](03_transformer_block.md)
