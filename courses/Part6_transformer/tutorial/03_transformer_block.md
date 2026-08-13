# 03 — Transformer Block：FeedForward、残差连接、LayerNorm、Scale Up

> 🧱 把"通信"和"计算"配对成 Block，用残差连接 + LayerNorm 让它能叠得很深，最后 scale up 成一个真正的 mini-GPT。

## 📖 前置知识

本章需要你已经掌握：

- **02 章全部内容**：attention 数学技巧、self-attention 单头、multi-head、6 条 attention 笔记
- **Part 4 加法节点的梯度规则**："加法把梯度均分给两个分支" —— 讲残差连接时直接用到
- **Part 3 BatchNorm**：训练/推理两态、归一化"列" —— 讲 LayerNorm 时会和它对比

## 从"看完就走"到"看完再想想"

上一章，token 通过（multi-head）self-attention 完成**通信**——互相看了看对方。但紧接着就去做预测，token **没时间消化**看到的东西。

论文里的 Transformer Block 其实是"两段式"的：先**通信**，再**计算**，然后整个 Block 重复很多次。

```
                ┌──────────────────────────┐
                │  Transformer Block       │
                │                          │
                │  ① 通信（communication）│  ← Multi-Head Self-Attention
                │  ② 计算（computation）  │  ← FeedForward（MLP）
                └──────────────────────────┘
        （Block 重复 n_layer 次，叠成深层网络）
```

## FeedForward：通信之后，各自思考

attention 让 token 互相交换信息；**FeedForward 则让每个 token 独立地"思考"**——它是对每个 token 分别做的 MLP，token 之间不交流。

```python
class FeedForward(nn.Module):
    """逐 token 的前馈网络：通信之后"各自思考"
    内层 4×n_embd（论文 512→2048 的 4 倍规律）"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
```

- 🔑 结构：`Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd)`。内层扩到 **4 倍**，这是论文里的规律：`512 → 2048`。你可以把它理解为"先是开大的空间里思考，再压缩回原尺寸"。
- ⚠️ 它作用在 `(B, T, C)` 的**最后一个维度**上——每个 token（每个 (b, t) 位置）独立经过同一个 MLP。这就是"per-token"（逐 token）的含义。

我们的脚本 [05_multihead_feedforward.py](../scripts/05_multihead_feedforward.py) 里 Phase 2 在多头后面接上 FeedForward 再训练：

```
  [Phase1 多头     ] train loss 2.3813, val loss 2.4545
  [Phase2 +前馈    ] train loss 2.4372, val loss 2.5006
```

- ⚠️ 注意 Phase 2 的 val loss（≈2.50）比 Phase 1 还略高——**这是少步数 + 前馈层更"深"导致收敛变慢**，多跑步数就会降下来（原视频中 2.28 → 2.24）。不要被单个数字骗了，看趋势、看结构收益。

## 残差连接：梯度超高速公路

现在把 Block 叠起来——**问题来了**：网络变深之后，优化变得困难。论文里有两个"特效药"，残差连接是第一个。

### 核心思想

```python
x = x + self.sa(x)     # 通信后加回残差通路
x = x + self.ffwd(x)   # 思考后加回残差通路
```

可以这样可视化：从输入到输出，数据只通过**加法**走一条"主干道"，每个 Block 是主干道旁边"岔出去"的小作坊——做完事再**加回来**。

```
输入 ─────┬──────┬──────┬── ... ──► 输出
          │      │      │
        +sa(.) +ffwd(.) ...
          ↑      ↑
       （每个 Block 都是加法并入主干）
```

### 为什么加法和梯度有关？

回想到 Part 4（手动反向传播）：**加法节点会把梯度均分给它的两个输入分支**。这意味着：

- 反传时，残差通路（`x + f(x)` 中的 `x`）的梯度是 **1**（恒等映射），所以梯度可以沿残差通路从 loss 直达输入、**不衰减**。加上子模块那一支的梯度，总梯度 = 1 + 子模块梯度。初始化时子模块很小，梯度近似为 1。
- 随着训练，残差块逐步"上线"、逐步贡献，但梯度高速公路始终畅通。

> 🔑 这就是"**梯度超高速公路（gradient super highway）**"。它让深层的 Transformer 可优化。没有它，几层 Block 叠起来就很难训练了。

我们的脚本 [05_multihead_feedforward.py](../scripts/05_multihead_feedforward.py) 的 Phase 3（`BlockNoLN`）实跑结果——残差连接带来最明显的一跃：

```
  [Phase3 +残差    ] train loss 2.1547, val loss 2.2324
```

```
═══ 演进对比 ═══
  Script 4 单头 self-attn: ~2.4
  Phase 1 多头并行:        2.4545
  Phase 2 + 前馈网络:      2.5006
  Phase 3 + 残差连接:      2.2324
```

- 💡 Phase 3 用 `x = x + self.sa(x); x = x + self.ffwd(x)` 把多头 + 前馈 + 残差全部组合，val loss 一下从 ~2.5 降到 **≈2.23**。这是本脚本最明显的一跃，正是"深层网络能优化了"的效果。

### proj：投影回残差通路

`MultiHeadAttention` 末尾的 `proj = nn.Linear(head_size * num_heads, n_embd)` 就是把拼接后的多头输出投影回 `n_embd` 维，好让 `x + sa(x)` 形状匹配。FeedForward 末层 `Linear(4*n_embd → n_embd)` 同理。

## LayerNorm：让更深的网络稳定（与 BatchNorm 对比）

第二个"特效药"是 **LayerNorm（层归一化）**。它和 Part 3 的 **BatchNorm** 关系密切，但有几个关键差异。

### BatchNorm 归一化"列"，LayerNorm 归一化"行"

```
输入 (B, C) = (4, 5)：

  BatchNorm：对每一"列"（每个特征通道，跨 batch 的样本）归一化
              ↓↓↓↓↓      （归一化的是竖着的列）

  LayerNorm：对每一"行"（每个样本的特征向量）归一化
              →→→→→      （归一化的是横着的行）
```

LayerNorm 在我们的 `(B, T, C)` 上，就是对**每个 token** 的 `n_embd` 维特征归一化（把 batch 和 time 都当作"样本"维）。

[06_layernorm_transformer.py](../scripts/06_layernorm_transformer.py) 里用 `x = torch.randn(4, 5)` 实跑演示：

```
═══ LayerNorm vs BatchNorm ═══
  原始 x 每行 mean: tensor([-0.6402,  0.2848, -0.3360,  0.3817])
  原始 x 每行 std:  tensor([0.8810, 0.9499, 1.3110, 1.7137])
  LayerNorm 后每行 mean: tensor([...e-08...]) (≈0)
  LayerNorm 后每行 std:  tensor([1.1180, 1.1180, 1.1180, 1.1180]) (≈1)
```

- 🔑 每行归一化后 mean ≈ 0、std ≈ 1。而且**所有行共享同一个 std（1.1180）**，说明 LayerNorm 是逐行独立归一化的。
- ⚠️ 注意 std 是 1.1180 而不是精确的 1.0——这是因为 PyTorch 的 `nn.LayerNorm` 默认用无偏方差（除以 n-1 而非 n），5 维时 `sqrt(5/4) ≈ 1.118`。这是数值细节，不影响理解。

### 与 BatchNorm 的三个关键差异

| 维度 | BatchNorm（Part 3） | LayerNorm（本课） |
|------|------|------|
| 归一化方向 | 跨 batch 的"列"（特征通道） | per-token 的"行"（n_embd 特征） |
| running buffer | 需要 EMA 的 running_mean/var | **不需要**，无训练/推理区分 |
| 可学习参数 | γ(gamma)/β(beta) | 同样保留 γ(gamma)/β(beta) |

- 💡 LayerNorm **没有 running buffer**，因为它归一化时不依赖其它样本——训练和推理行为完全一致，代码更简单（不需要 `model.train()`/`model.eval()` 切换）。这是它在 Transformer 里比 BatchNorm 更合适的重要原因。
- 归一化后同样保留可学习的 γ 和 β，让网络能"撤掉"归一化效果、学自己想要的分布。

### Pre-norm 结构

论文原版是 **post-norm**（先 attention，后 LayerNorm）。现在更常见的做法是 **pre-norm**：**先 LayerNorm，再进入 attention/ffwd**。本课采用 pre-norm（也几乎成了现代 Transformer 的标准）。

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # pre-norm：先 LN 再 attention
        x = x + self.ffwd(self.ln2(x)) # pre-norm：先 LN 再 ffwd
        return x
```

- ⚠️ 顺序很重要：`x = x + self.sa(self.ln1(x))` 是"**先归一化 x，再做 attention，再加回来**"。别写成 `self.sa(x)` 再归一化——那是 post-norm。

### 完整 decoder-only Transformer

[06_layernorm_transformer.py](../scripts/06_layernorm_transformer.py) 把 Block 叠两层，并加上**最终 LayerNorm `ln_f`**（在 `lm_head` 之前，把最终特征归一化后再投影到词表）：

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)   # 最终 LayerNorm（lm_head 之前）
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        ...
```

完整的 decoder-only Transformer 管线：

```
token 编码 + 位置编码
   → N × Block(残差 + 多头注意力 + 前馈 + LayerNorm(pre-norm))
   → ln_f
   → lm_head → logits
```

2 层 Block + LayerNorm 训练 1200 步的真实日志：

```
═══ 训练 (n_layer=2, AdamW, lr=0.003, 1200 步) ═══
  step    0: train loss 4.3437, val loss 4.2889
  step  400: train loss 2.4303, val loss 2.4364
  step  800: train loss 2.3276, val loss 2.3338
  step 1199: train loss 2.2512, val loss 2.2340
```

- 💡 最终 val loss ≈ **2.23**（和脚本 05 Phase 3 的 2.23 巧合地接近）。这里 `n_layer=2` 已经比脚本 05 的单层更深，说明 LayerNorm 的主要价值**不是"降 loss"，而是让更深的网络也能稳定优化**。
- 🔑 参考原视频的演进（完整超参）：bigram 2.5 → 单头 2.4 → 多头 2.28 → 前馈 2.24 → 残差 2.08 → +LayerNorm 2.06。我们的 CPU 小规模数字（2.50 → 2.39 → 2.45/2.50/2.23 → 2.23）趋势一致、绝对值有差异（超参/种子不同），看趋势别背数字。

生成结果（400 个字符）——已经有点像"对话体"了：

```
Tit ILA:
Uy?

BF

PUY:
Foured
She manst I sipre.

GCAPEErq:
Thondl with youns?

CILA:
Am fore therars.

The dall ill sit hem will elpimpke a'd me chat fraser 'at; he lich he blavk bed;
...
```

## Dropout：随机关掉一些神经元

要 scale up 之前，先加一个正则化技巧——**Dropout**（Srivastava et al., 2014）。它在前向/反传时**随机把一部分神经元（或注意力权重）置零**：

- 每次前向/反传的"置零掩码"都不同 → 等价于**训练了一堆子网络的集成**。
- 测试时所有神经元全开 → 相当于把那一堆子网络合并成一个集成。
- 一句话：**正则化，防止过拟合**。

```python
# Head 里：softmax 后随机屏蔽部分注意力
wei = F.softmax(wei, dim=-1)
wei = self.dropout(wei)      # 随机阻止一些节点通信

# MultiHeadAttention：残差连接前
out = self.dropout(self.proj(out))

# FeedForward：残差连接前
nn.Linear(4 * n_embd, n_embd),
nn.Dropout(dropout),
```

- ⚠️ 放置位置：**残差连接之前**（attention 输出、feedforward 输出），以及 **softmax 之后的注意力权重**。这些是 Dropout 在 Transformer 里的典型位置。

## Scale Up：超参数放大 + 参数统计 + 生成

现在把前面所有组件拼成完整的 `GPTLanguageModel`（[07_scaleup_generate.py](../scripts/07_scaleup_generate.py)，与 `gpt.py` 收敛一致），加 Dropout 和更好的初始化，然后**放大超参**。

### 参数统计

```python
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
```

实跑输出：

```
═══ 模型 ═══
  缩小版超参: batch=16, block=64, n_embd=64, n_head=4, n_layer=2, dropout=0.2
  参数量: 112,193 = 0.112 M
```

- 🔑 `sum(p.numel())` 把所有参数张量的元素数加起来，`/1e6` 就是百万为单位。我们的 CPU 缩小型只有 **≈0.11M 参数**；原视频 GPU 完整版约 **10M 参数**。

### 超参数对比

| 超参 | GPU 完整版（原视频，A100 ~15 分钟） | CPU 缩小型（我们，<30s） |
|------|:---:|:---:|
| batch_size | 64 | 16 |
| block_size | 256 | 64 |
| n_embd | 384 | 64 |
| n_head | 6 | 4 |
| n_layer | 6 | 2 |
| dropout | 0.2 | 0.2 |
| lr | 3e-4 | 3e-4 |
| max_iters | 5000 | 150 |
| val loss | **1.48** | ≈2.80 |

- ⚠️ 我们没有 GPU，所以**主动缩小规模**（更小的 n_layer/n_embd/block_size/步数），让它在 CPU 上 <30s 跑完。原视频里"2.07 → **1.48**"那一步是靠 A100 + 完整超参烧出来的，CPU 上跑不了。

缩小型训练 150 步的真实日志：

```
═══ 训练 (AdamW, lr=0.0003, 150 步) ═══
  step   0: train loss 4.1655, val loss 4.1661
  step 100: train loss 2.9802, val loss 3.0016
  step 149: train loss 2.7744, val loss 2.8044
```

- 💡 150 步太少，val loss 只到 ≈2.80，但趋势对：**如果给它更多步数 + 更大规模，就能逼近视频里的 1.48**。脚本里已经把"完整超参 + 10M 参数 + 1.48"的说明打印出来了。

### 生成：从换行符开始

起始上下文是 `torch.zeros((1, 1))`，也就是一个 token：索引 0 = 换行符。这是个合理的"开场"。

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

- 💡 生成的 token 数可以随意加大：脚本里注释演示了 `max_new_tokens=10000` 并写进文件（`open('more.txt', 'w')...`），就能生成 1 万个字符的"伪莎士比亚"。

缩小型生成的 500 个字符（训 150 步，所以还比较乱——这是**步数不足**的表现，不是架构的错）：

```
'pi!QAz-nh ntt an ,,eerooQLIhAFE,;.rde ce toaise
Whe ninssete d ?JW.RCWeo,jmolshonulodioqoslt mroQ:
TT osaOunoo:TcWVcevisn, e,raOoluncuz hndrAssmo at mi;
...
```

> 💡 对比：原视频用完整超参生成 10K 字符，看起来就"很像莎士比亚了"（虽然读起来仍然无意义）——这是 scale 的力量。

## 学完本部分你能...

- ✅ 理解"通信（attention）vs 计算（FeedForward）"的分工，写出 per-token MLP
- ✅ 解释残差连接 = "梯度超高速公路"：加法均分梯度、梯度直达输入
- ✅ 对照 Part 3 讲清 LayerNorm vs BatchNorm 的差异（行列/无 running buffer/保留 γβ）
- ✅ 写出 pre-norm 的 Block，理解它和论文 post-norm 的区别
- ✅ 组装完整的 decoder-only Transformer，统计参数量
- ✅ 解释 Dropout 为什么能正则化，以及它放在哪些位置
- ✅ 理解 CPU 缩小型与 GPU 完整版（1.48）的区别

## 课后练习

<details>
<summary>Q1: 为什么"残差连接 + 加法把梯度均分给两个分支"能帮助深层网络训练？</summary>
A: 反传时，加法节点的梯度会均分给它的两个输入。残差通路（`x + ...`）从 loss 到输入全程只有加法，梯度可以"跳过"所有残差块、几乎不衰减地直达输入——这就是梯度超高速公路。残差块刚初始化时贡献很小（近似等于没加），所以早期梯度畅通无阻；训练中残差块逐步"上线"。这样网络想学多深都不会"梯度传不到"。
</details>

<details>
<summary>Q2: LayerNorm 和 BatchNorm 的本质区别是什么？为什么 Transformer 里选 LayerNorm？</summary>
A: BatchNorm 对"列"归一化（跨 batch 的特征通道），需要维护 running buffer，训练/推理行为不同；LayerNorm 对"行"归一化（每个 token 的 n_embd 特征），不需要 running buffer，训练/推理无区别。Transformer 的序列长度（T）可变、且依赖样本内归一化，LayerNorm 因为不依赖其它样本、没有训练/推理两态，所以更合适、更简单。两者都保留可学习的 γ/β。
</details>

<details>
<summary>Q3: 我们从 2.5（bigram）一路降到 2.23（2 层 Block），为什么 scale up 到 10M 参数能再降到 1.48？</summary>
A: scale up 同时增加了三样东西：模型容量（n_embd 32→384、6 层 Block、约 10M 参数）、上下文长度（block_size 8→256，模型能看到更长的"莎士比亚台词"）、训练步数与 batch（5000 步 × batch 64）。更大容量能记住更多规律，更长上下文能更好地预测下一个字符。Dropout 0.2 在放大后抑制过拟合。三者叠加，val loss 从 2.07 一路压到 1.48。当然这需要 A100 约 15 分钟——计算是 scale 的燃料。
</details>

## 📝 课后作业

完成本章后，去 Assignment 6 完成题 5（🌟 完整 Transformer Block）：

👉 [Assignment 6](../../../assignments/assignment_6/)

## 下一步

我们的 decoder-only Transformer 已经完整、能训练、能生成。最后一部分，我们把镜头拉远：**我们实现的到底是整个 Transformer 的哪一半？** 另一半（encoder + cross-attention）长什么样？工业界的 nanoGPT 和 ChatGPT/GPT-3 是怎么从我们这 200 行代码走向生产的？

👉 [04 — 超越 Transformer](04_beyond_transformer.md)
