# 02 — 现代组件：RMSNorm 与 RoPE

> 🧭 Transformer 的骨架没变，但两个关键零件换成了"现代款"：归一化从 LayerNorm 换成 RMSNorm，位置编码从可学习参数表换成 RoPE。本课把这两个零件从零实现。

## 📖 前置知识

本章需要你已经掌握：

- **Part 6 03 章**：LayerNorm（pre-norm）、残差连接、Transformer Block
- **Part 3 的 BatchNorm**：归一化的"列"、训练/推理两态、γ/β 可学习参数
- **Part 6 02 章**：位置编码的作用、self-attention 里 q/k/v 怎么用

> 💡 重点回看 Part 6 里 LayerNorm 的实现（`weight`/`bias` 两个可学习参数）——RMSNorm 是它的"瘦身版"。

## 从 Part 6 结束的地方出发

Part 6 的 Transformer Block 长这样：

```
      x
      │
      ├──[LayerNorm]──► [Multi-Head Self-Attention]──► + ──►
      │                                    ↑
      │                            + [position embedding]
      │
      └───────────────────────────────────┘  （残差连接）
```

两个"零件"这一章要被换掉：

1. **LayerNorm** → **RMSNorm**：归一化方式
2. **learned positional embedding**（第二张 embedding 表）→ **RoPE**：位置编码方式

为什么换？这一章我们不只讲"怎么换"，更讲"为什么"。先说归一化。

## 归一化回顾：从 BatchNorm 到 LayerNorm

Part 3 我们实现了 **BatchNorm**：对一个 batch 的所有样本，**按"列"（每个特征）** 减去均值、除以标准差，然后用可学习的 γ/β 做缩放和平移：

```
BatchNorm:  y = γ · (x - mean_batch) / sqrt(var_batch + eps) + β
                ↑ 按 batch 统计        ↑ 可学习缩放      ↑ 可学习平移
```

Part 6 我们换了 **LayerNorm**：归一化的对象从"batch 列"换成"单个样本的行"，并且**不再需要 running buffer**（没有训练/推理两态，直接对所有位置归一化）：

```
LayerNorm:  y = γ · (x - mean_row) / sqrt(var_row + eps) + β
                 ↑ 按单个样本统计      ↑ 可学习缩放      ↑ 可学习平移
```

- 🔑 注意 LayerNorm 有两个可学习参数：**γ（缩放）** 和 **β（平移）**，还有一个可选的 `bias`。它们让归一化之后的分布"不完全固定"，网络能自己学出合适的分布。

## RMSNorm：只算均方根，砍掉均值和平移

### 公式

**RMSNorm（Root Mean Square Normalization，2019）** 的核心洞察：**均值中心化在 Transformer 里信息量很低，可以砍掉。**

先看 RMSNorm 怎么定义。它不做均值中心化，只把每个样本的每一行除以自己的**均方根（RMS）**：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
             ↑          ↑                      ↑
           保持原值    只算平方的均值          可学习缩放（不再有 β/bias）
```

逐项拆解：

- `mean(x²)`：对每个位置的 hidden 向量，把每个元素的平方取平均（**注意：不先减均值**）
- 开根号 + eps：得到"激活的均方根"，eps 防除零
- 除以它：把整行的"尺度"归一化到接近 1
- `weight`：可学习的 γ，逐元素缩放

### 代码：RMSNorm

对照 minimind 的实现，[02_rmsnorm_rope.py](../scripts/02_rmsnorm_rope.py) 里：

```python
class RMSNorm(nn.Module):
    """只做均方根归一化，砍掉均值中心化和 bias"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 只有一个可学习参数 γ

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 均方根
        return self.weight * (x / rms)                                # 除以尺度，再缩放
```

- ⚠️ 注意和 LayerNorm 的三个区别：**① 没有减均值；② 没有 β；③ 没有 bias**。`nn.Parameter` 只有 `weight` 一个。
- 💡 `mean(-1, keepdim=True)` 是沿最后一维（hidden 维）取平均，`keepdim` 保留维度好做广播。rms 的 shape 是 `(B, T, 1)`，和 `x` 广播相除。

### 数值例子：RMSNorm vs LayerNorm（手算）

拿一个 1D 向量手算一遍，对比最直观。设 `x = [1, 2, 3]`，暂时忽略 `weight` 和 `eps`：

```
RMSNorm:
  mean(x²) = (1² + 2² + 3²)/3 = 14/3 ≈ 4.667
  rms = √4.667 ≈ 2.160
  y = x / 2.160 ≈ [0.463, 0.926, 1.389]
  验证：输出自己的均方根 = √((0.463² + 0.926² + 1.389²)/3) = 1   ✅ 归一化到 RMS=1

LayerNorm（同样输入）：
  mean = (1+2+3)/3 = 2
  var  = ((1-2)² + (2-2)² + (3-2)²)/3 = 2/3 ≈ 0.667
  y = (x - 2)/√0.667 ≈ [-1.225, 0, 1.225]
```

- 💡 注意两者的差异：LayerNorm 先减均值，输出**必然以 0 为中心**（有正有负）；RMSNorm **不做减均值**，输出**保持原来的正负形状**，只是把"尺度"压成 1。所以 RMSNorm 对"激活本身的值"更忠实——这正是"均值中心化信息量低，砍掉它"的直观体现。
- ⚠️ 我们的例子为了直观忽略了 `weight`（γ）。真实模型里 γ 初始化为 1，训练中自己学，`y = γ ⊙ (x / rms)` 逐元素缩放——和 LayerNorm 的 γ 作用相同。

### 为什么 RMSNorm 更好？

1. **均值中心化在 Transformer 里信息量低**
   Transformer 的隐藏层经过残差连接，数值分布已经被多次混合，均值这一项几乎不携带有用信息。去掉它，模型几乎不掉点（原文实验里 LayerNorm 换 RMSNorm 性能相当）。

2. **省掉 β/bias，参数更少、计算更省**
   LayerNorm 每个 hidden 维要存 γ 和 β 两个参数；RMSNorm 只要 γ。在 8 层、hidden 512 的模型里，省的参数不多，但**计算上少了一次减均值、一次算均值**，前向/反向都更快。对超大规模模型，这种"每层省一点"会累积成可观的加速。

3. **训练更稳定**
   归一化层的数值只依赖"平方的均值"，函数更平滑、梯度更干净，训练大模型时被证明更稳（这也是 Llama 系列选择它的原因之一）。

> 🔑 一句话：**RMSNorm = LayerNorm 去掉"减均值"和"可学习平移"，只保留"除以均方根 + 缩放"。** 用更少的计算换回相当的性能，是现代 LLM 的默认选择。

### 对比表

| | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | `(x-μ)/√(σ²+eps)·γ + β` | `x/√(mean(x²)+eps)·γ` |
| 减均值 | 有 | **没有** |
| 可学习参数 | γ + β | **只有 γ** |
| bias | 有（可关） | **没有** |
| 计算量 | 算 mean + var | **只算 mean(x²)** |
| 训练稳定性 | 好 | **更好**（小模型也常用） |

### 顺带一提：Query/Key 归一化

minimind 在注意力里还对 **q 和 k** 各自又加了一层 RMSNorm（`q_norm`/`k_norm`，`head_dim` 维）：

```python
xq = self.q_norm(xq)   # 每个 head 的 q 再归一化一次
xk = self.k_norm(xk)
```

- 💡 这层在注意力的"缩放"之外又加了一重归一化，主要作用是把 q/k 的尺度稳定住（内积对尺度很敏感）。它是近几年的流行做法，不是必需——先知道有这回事，重点还是 block 里那两个大的 RMSNorm。
- ⚠️ 我们的教程脚本（[03_gqa_kv_cache.py](../scripts/03_gqa_kv_cache.py)、[05_full_model.py](../scripts/05_full_model.py)）**省略了 q_norm/k_norm**，以简化实现、聚焦核心概念。它是可选增强，不影响对 GQA/RoPE 的理解。minimind 的某些版本包含它。

## RoPE：从"可学习的参数表"到"旋转位置编码"

### Part 6 的做法回顾：learned positional embedding

Part 6 里位置信息靠**第二张 embedding 表**：`nn.Embedding(block_size, n_embd)`，每个位置一个**可学习的向量**，加到 token embedding 上：

```python
tok_emb = token_embedding_table(idx)          # (B,T,C) token 本身
pos_emb = position_embedding_table(arange(T)) # (B,T,C) 位置向量
x = tok_emb + pos_emb                          # 相加注入位置信息
```

- ⚠️ 它的两个短板：**① 要花参数**（block_size × hidden 一张表）；**② 不能外推**——训练时最长只见过 block_size 个位置，推理时序列一超过这个长度，位置编码就"越界"，模型表现崩掉。
- 💡 回想 Part 6 提过的 nanoGPT 用 `nn.Embedding` 位置编码，同样受制于"训练长度 = 最大长度"。

### RoPE 的核心想法：位置 = 旋转

**RoPE（Rotary Position Embedding，旋转位置编码，2021）** 的思路完全不同：不学一张位置表，而是**把 q 和 k 向量按它们的位置"旋转"一个角度**。

为什么"旋转"能编码位置？因为在二维平面上，**旋转一个向量不改变它的长度（范数），但会改变它和另一个向量的夹角**——而注意力算的是内积，内积恰好只取决于"夹角 × 长度"。

> 🔑 两个向量做内积，如果只把其中一个旋转 θ 度、另一个不动，内积会按 `cos(位置差)` 的规律变化。**这个变化只取决于两个位置之差（相对位置），与绝对位置无关**——这正是我们想要的"相对位置感知"。

### 数学：从旋转矩阵到频率

一个二维向量 `(x₀, x₁)` 旋转 θ 角，用旋转矩阵表示：

```
[ x₀' ]   =   [ cos θ   −sin θ ]   [ x₀ ]
[ x₁' ]       [ sin θ    cos θ ]   [ x₁ ]

x₀' = x₀·cos θ − x₁·sin θ
x₁' = x₀·sin θ + x₁·cos θ
```

- 🔑 等价写法是**复数**：把 `(x₀, x₁)` 看成复数 `z = x₀ + i·x₁`，旋转就是乘 `e^{iθ}`——旋转位置编码的官方推导就是这么写的。`θ` 是"这一维的旋转频率"。

但 hidden 维是几百维，不是 2 维。做法是：**把 hidden 维两两分成一组**，每一组用不同的旋转频率 `θ`，沿维度呈指数变化：

```
freq[i] = 1 / theta^(2i / dim)        i = 0, 1, 2, ...
angle = position * freq[i]            第 i 组在这个 position 上旋转 angle 弧度
```

- `theta`（即 RoPE 的 `rope_base`）通常取 `1e4 ~ 1e6`，minimind 取 `1e4`（与 Llama 系列一致）
- 第 0 组频率最高（转得快），后面的组频率指数衰减（转得慢）——**低频慢转、高频快转**，和傅里叶分解同理

### 代码：precompute_freqs_cis + apply_rotary_pos_emb

minimind 的实现分两步，[02_rmsnorm_rope.py](../scripts/02_rmsnorm_rope.py) 里照抄并加了注释：

**第一步：预先算好每个位置的 cos/sin 表（一次性，缓存为 buffer）**

```python
def precompute_freqs_cis(dim, end, rope_base=1e4):
    # 1) 每个维度对的频率：1/theta^(2i/dim)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    # 2) 位置 t 与频率做外积 → (end, dim/2) 的角度矩阵
    t = torch.arange(end)
    angles = torch.outer(t, freqs).float()
    # 3) 转成 cos/sin，并在最后拼接一份（配合 rotate_half 的"对半翻转"技巧）
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return cos, sin     # 各自 shape (max_pos, dim)
```

- 💡 `torch.outer(t, freqs)` 得到 `(end, dim/2)` 的矩阵，`[i, j]` 就是"位置 i 的第 j 组角度"。拼接成 `dim` 长是为了下面的 `rotate_half` 技巧（对半翻转后逐元素相乘），省一次显式矩阵乘法。

**第二步：把 cos/sin 应用（旋转）到 q 和 k 上**

```python
def rotate_half(x):
    """把后一半取负放到前一半 → 等价于复数的 i·z"""
    return torch.cat([-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q 在位置 p 上旋转 p 的角度（cos_p, sin_p 是预计算表里第 p 行）
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

- 🔑 `q * cos + rotate_half(q) * sin` 这行把"复数乘 `e^{iθ}`"翻译成了实数运算：`rotate_half(q)` 恰好实现了 `i·q`（后一半取负放到前一半）。这样在 q/k 上各乘一下，就等价于让它们各自旋转。
- 💡 旋转是**逐元素**的：不需要学参数，只需要查表。位置信息以"旋转角度"的形式被揉进了 q 和 k。

> **📝 脚本实现对照**：上面用实数版（`cos/sin + rotate_half`）讲原理，更直观。实际脚本 [02_rmsnorm_rope.py](../scripts/02_rmsnorm_rope.py) 和 [05_full_model.py](../scripts/05_full_model.py) 用**复数版**实现——把 `(x₀, x₁)` 看成复数 `z = x₀ + i·x₁`，用 `torch.polar(1, angle)` 构造旋转因子 `e^{iθ}`，再用 `torch.view_as_complex` / `torch.view_as_real` 做复数乘法。两者数学上完全等价（`q·cos + rotate_half(q)·sin == view_as_real(view_as_complex(q) * e^{iθ})`），复数版代码更简洁，但需要了解 `torch.view_as_complex` 等 API。作业题 3（RoPE）的测试对两种实现都接受。

**第三步：在 attention 里使用**

```
xq, xk 算出来后：
  1. 取当前位置范围的 cos/sin：cos[pos:pos+seq], sin[pos:pos+seq]
  2. xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)   # 旋转注入位置
  3. 之后照常算内积 (xq @ xk^T)
```

### 为什么 RoPE 更好：相对位置、可外推、零参数

1. **绝对位置不影响内积，相对位置决定内积**
   旋转是**正交变换**：`‖旋转后的向量‖ = ‖原向量‖`。q、k 各自旋转后，内积变成 `q·k·cos(角度差)`——**绝对位置完全不影响**（都旋转不改变夹角差的部分...严格说内积依赖角度差），这正是位置编码想要的"相对位置感知"。相比 learned PE 要硬记位置对，RoPE 直接把相对距离编码进了内积。

2. **可外推（extrapolation）**
   训练时 max position 4096，推理时想要 8192？RoPE 的 cos/sin 表是**公式生成**的，`precompute_freqs_cis(end=8192)` 就能算出来——**不需要重新训练**。learned PE 没有这个能力（表就是参数，没见过就是没见过）。
   ⚠️ 严格说"直接外推"超过训练长度太多，模型精度还是会掉（长上下文的高频维度分布变了）。工业界用 **YaRN / NTK scaling** 这类技巧缓解，minimind 也支持（`inference_rope_scaling`）。我们教程**已经做了 scaling 实测**：[scripts/11_rope_scaling.py](../scripts/11_rope_scaling.py) 用同一模型对比四种方案（naive/PI/NTK/YaRN）"训练 128 → 推理 256"的困惑度——YaRN 的温度因子 **√(1/t) = 0.1·ln(s)+1**（论文/HF 官方做法：√(1/t) 同时乘 q、k，等价 logit ×1/t；本课脚本简化为只乘 q）是面试加分点；[scripts/13_long_context_eval.py](../scripts/13_long_context_eval.py) 再用迷你 RULER 的 KV 检索任务量"外推后还记得住吗"，两份实测数字见第 5 章「进阶实验」。

3. **零参数**
   RoPE 不引入任何可学习参数。对比 learned PE 那张 `block_size × hidden` 的 embedding 表，RoPE 只占一小块 `cos/sin` 缓存（buffer，不算参数）。参数省了，还顺带解决了外推。

### 数值例子：旋转 2D 向量，看"相对位置决定内积"

用最简单的 2D 向量感受一下"旋转为什么能编码相对位置"。设每个位置的旋转频率 `θ = 0.5 rad/位置`，两个单位向量 `q = k = [1, 0]`（长度都是 1，范数不变）。

```
位置 2 的 q：  旋转 2×0.5 = 1.0 rad → q₂ = [cos1.0, sin1.0]
位置 2 的 k：  旋转 2×0.5 = 1.0 rad → k₂ = [cos1.0, sin1.0]
内积 q₂·k₂ = cos(0) = 1.0          ← 相同位置，完全对齐

位置 2 的 q、位置 3 的 k：
  内积 = cos((3-2)×0.5) = cos(0.5) ≈ 0.878   ← 相邻，轻微错开

位置 5 的 q、位置 6 的 k：
  内积 = cos((6-5)×0.5) = cos(0.5) ≈ 0.878   ← 同样是"相差1"，结果一样！

位置 2 的 q、位置 8 的 k：
  内积 = cos((8-2)×0.5) = cos(3.0) ≈ -0.99   ← 隔得远，几乎反向
```

- 🔑 关键观察：**第 2、3 组"位置差都是 1"，内积都是 0.878**——虽然它们的绝对位置不同（2/3 和 5/6），结果一模一样。**内积只取决于位置差，与绝对位置无关**。这就是"旋转正交、范数不变"带来的性质。
- 💡 把这里的 `θ=0.5` 换成真实 RoPE 的多组频率，同一套直觉依然成立：**相邻 token 注意力分数高，相隔越远分数越低**，且不依赖绝对位置。

### 把 RoPE 装进 attention

在完整注意力里，RoPE 只改两个位置：q/k 旋转、然后照常算内积。[02_rmsnorm_rope.py](../scripts/02_rmsnorm_rope.py) 里：

```python
# 1. 预计算好 cos/sin 表（模型初始化时算一次，存成 buffer）
cos, sin = precompute_freqs_cis(dim=head_dim, end=max_position_embeddings)
model.register_buffer('cos', cos, persistent=False)
model.register_buffer('sin', sin, persistent=False)

# 2. forward 里，q/k 算出来后取当前位置的 cos/sin 并旋转
q = q_proj(x).view(B, T, n_heads, head_dim)     # (B,T,8,hd)
k = k_proj(x).view(B, T, n_heads, head_dim)
cos_t = self.cos[:T].unsqueeze(0).unsqueeze(0)   # (1,1,T,hd) 取前 T 个位置
sin_t = self.sin[:T].unsqueeze(0).unsqueeze(0)
q, k = apply_rotary_pos_emb(q, k, cos_t, sin_t)  # 旋转注入位置

# 3. 之后照常：scores = (q @ k^T)/√head_dim → softmax → @v
```

- ⚠️ 两个容易踩的坑：**① `cos/sin` 只取 `[:T]`**——位置从 0 数起，正好和 token 下标对齐；**② 做生成带 KV Cache 时要从 `start_pos` 偏移取**（否则新 token 的位置算错）。第 3 章讲 KV Cache 时会再碰这个坑。

### 对比表

| | learned PE（Part 6） | RoPE（Part 7） |
|---|---|---|
| 形式 | 一张可学习 embedding 表，加到 token 上 | 旋转 q/k，公式生成 |
| 参数 | `block_size × hidden` | **0** |
| 依赖训练长度 | 是，超过就崩 | **否，可外推** |
| 相对位置 | 硬记（要大量数据学） | **结构内建（旋转角差）** |
| 绝对位置 | 显式编码 | **不影响内积** |

## 权重绑定（tie_word_embeddings）

最后一个小零件：**把 token embedding 和最后的输出层（lm_head）共享同一个权重**。

```
embed_tokens: vocab(6400) → hidden(512)    输入侧：token → 向量
lm_head:      hidden(512) → vocab(6400)    输出侧：向量 → token 分数

如果两个权重一样，参数直接从"嵌入"省成"一份"
```

```python
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
if self.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight   # 指向同一份权重！
```

- 🔑 这个技巧叫 **weight tying（权重绑定）**：embedding 层学到的"每个 token 的向量表示"，反过来也能当"预测每个 token 的分数向量"用。省掉 `vocab × hidden` 一整块参数——对我们 26M 的小模型，这一省就是 **6400×512 ≈ 3.3M**，约 **12%**。
- ⚠️ 为什么可以共享？直觉：一个 token 越"容易被预测"，说明它的表示越有区分度，用同一个向量既能"表示它"又能"给它打分"是自洽的。GPT-2 之后的很多模型默认开启。
- 💡 用了绑定之后，`embed_tokens.weight` 和 `lm_head.weight` 是**同一份参数**（内存共享），PyTorch 里梯度会自动累计到同一个 `Parameter` 上，不需要额外处理。

## 学完本部分你能...

- ✅ 写出 RMSNorm，讲清它与 LayerNorm 的三个区别（无减均值、无 β、无 bias）及为什么更优
- ✅ 画出 q_norm/k_norm 在注意力里的位置，知道它是可选增强
- ✅ 用复数/旋转矩阵解释 RoPE："旋转 → 内积依赖角度差 → 相对位置"
- ✅ 手写 `precompute_freqs_cis` + `apply_rotary_pos_emb`，在 attention 里注入位置
- ✅ 对比 learned PE 与 RoPE（参数、外推、相对位置），说清"为什么 RoPE 是默认"
- ✅ 说出 tie_word_embeddings 省了哪块参数（`vocab × hidden`）

## 课后练习

<details>
<summary>Q1: 为什么 LayerNorm 去掉均值中心化后性能几乎不掉？</summary>
A: 均值中心化的作用是让激活"以 0 为中心"，但 Transformer 里每一层都被残差连接叠加，激活的均值分布已经被混合得很有规律，均值项携带的信息量很低。而且 RMSNorm 保留了"除以均方根"和"可学习缩放 γ"这两个真正起作用的归一化因子。实验证明，对激活做不做均值中心化对最终表现影响很小，去掉反而省计算。
</details>

<details>
<summary>Q2: 为什么 RoPE 能让"绝对位置不影响内积"？这对语言建模有什么用？</summary>
A: 旋转是正交变换，范数不变。q、k 各自旋转后，内积 <R(q), R(k)> 只与它们的**角度差**有关——而角度差正是由位置差决定的。所以内积（注意力权重）只反映相对距离：相邻 token 得分高、相隔远的得分低，且与"这发生在序列的第 5 位还是第 500 位"无关。这对语言建模正合适：单词的意义更多取决于它和上下文的相对位置，而不是它在整个语料里的绝对序号。
</details>

<details>
<summary>Q3: 权重绑定为什么能省参数？它有没有副作用？</summary>
A: 输入侧把 token 变成向量、输出侧把向量变成 token 分数，两边的形状都是 vocab×hidden，本质是同一类"token ↔ 向量"映射。让它们共享同一份权重，省掉 vocab×hidden 一整块（我们的小模型约 12%）。副作用通常是极小的（输出和输入侧的任务不完全一样，共享算一点点"参数共享正则化"），实践中往往还能略微提升小模型效果，所以 GPT-2、Llama 这类模型默认开启。
</details>

## 📝 课后作业

完成本章后，去 Assignment 7 完成题 2（RMSNorm）和题 3（RoPE）：

👉 [Assignment 7](../../../assignments/assignment_7/)

## 下一步

归一化和位置编码都换成了"现代款"。但注意力的内部还有一个大问题没解决：**8 个 Q 头各自配了 8 套独立的 K/V，太费内存了。** 下一步我们把注意力升级成 GQA、加上 KV Cache，再把 FFN 从 ReLU 换成 SwiGLU，并看一眼 MoE。

👉 [03 — GQA 与 FFN：SwiGLU、KV Cache、MoE](03_gqa_and_ffn.md)
