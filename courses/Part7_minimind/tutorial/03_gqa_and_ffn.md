# 03 — GQA 与 FFN：KV Cache、SwiGLU、MoE

> ⚙️ 注意力和前馈网络是现代 LLM 里最"吃资源"的两个部件。本课把 MHA 升级成 GQA、加上 KV Cache，把 ReLU FFN 换成 SwiGLU，最后看看"专家"（MoE）是什么。

## 📖 前置知识

本章需要你已经掌握：

- **Part 6 02 章**：Multi-Head Self-Attention（每头独立 Q/K/V、亲和力、缩放、遮罩）
- **Part 6 03 章**：FeedForward（`Linear → ReLU → Linear`）、残差连接
- **Part 2/5**：非线性激活函数、负对数似然

> 💡 重点回看 Part 6 的 4D batched multi-head（把"头"塞进 batch 维）——GQA 就是在它基础上"砍掉一半 K/V"。

## 从 Part 6 结束的地方出发

Part 6 我们实现的 attention 是 **MHA（Multi-Head Attention）**：`n_head` 个头，**每个头都有自己独立的 Q、K、V 线性层**。8 个头就是 8 套 Q/K/V。

这一章要回答一个问题：**K/V 真的需要每个头都来一套吗？**

## GQA：K/V 是显存瓶颈

### 回顾 MHA：每头一套 K/V

```
MHA（8 头）：
  Q 头 0,1,...,7  —— 各要一套（Q 必须每头独立，负责"关注什么"）
  K 头 0,1,...,7  —— 每头一套
  V 头 0,1,...,7  —— 每头一套
```

- 💡 为什么 **Q 必须每头独立**？因为每个 Q 头代表一种"注意力视角"（有的看语法、有的看指代、有的看语义），它们必须不一样才能各司其职。
- ⚠️ 但 **K/V 是"被查询的内容"**，8 个头查的内容其实高度重合。为 8 个头各存一套 K/V，很浪费。

### 关键问题：序列越长，K/V 越占显存

推理时，模型要**缓存所有已生成的 K/V 向量**（这就是下文的 KV Cache），用于计算新 token 的注意力。缓存大小正比于：

```
KV 缓存大小 ≈ n_layers × n_kv_heads × seq_len × head_dim
```

- 序列每长一倍，K/V 缓存大一倍；模型层数越多、越大，K/V 缓存越爆炸。**长上下文的瓶颈不在"计算"，而在"K/V 的显存"**。GQA 正是为压这个指标而生的。

### 两端的尝试：MHA 与 MQA

```
MHA（Part 6）：每头独立 K/V
  8 套 K/V，质量最高，但 K/V 缓存最大

MQA（Multi-Query Attention）：所有 Q 头共享同一组 K/V
  1 套 K/V，K/V 缓存最小，参数最少
  但"一刀切"太狠，不同头被迫用同一份 K/V，质量下降
```

- **MQA**：8 个 Q 头共享 1 套 K/V。缓存压到 1/8，但表达能力受损。

### GQA：分组共享，折中

**从 Part 6 的 MHA 到 GQA，代码只改 3 处**（左侧 [Part 6 脚本 05](../../Part6_transformer/scripts/05_multihead_feedforward.py)，右侧 [Part 7 脚本 05](../scripts/05_full_model.py)）：

```diff
  class Attention:
      def __init__(self, n_embd, n_head, ...):
          self.n_heads = n_head
+         self.n_kv_heads = n_kv_heads              # ① 新增：KV 头数（如 8 头里只留 4 组 K/V）
+         self.n_rep = self.n_heads // self.n_kv_heads
-         self.key   = nn.Linear(n_embd, n_embd)    # ② K/V 投影输出维从 n_embd 缩到
-         self.value = nn.Linear(n_embd, n_embd)    #    n_kv_heads * head_dim
+         self.wk = nn.Linear(n_embd, self.n_kv_heads * self.head_dim, bias=False)
+         self.wv = nn.Linear(n_embd, self.n_kv_heads * self.head_dim, bias=False)

      def forward(self, x):
          ...
-         k = self.key(x).view(B, T, self.n_heads, head_dim)      # ③ 用之前把 K/V
-         v = self.value(x).view(B, T, self.n_heads, head_dim)    #    复制回 Q 的头数
+         k = self.wk(x).view(B, T, self.n_kv_heads, head_dim)
+         v = self.wv(x).view(B, T, self.n_kv_heads, head_dim)
+         k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
```

其余（Q 投影、softmax、加权求和）一行都不用动——这就是"换零件不改骨架"。


**GQA（Grouped-Query Attention，2023）** 把 Q 头**分组**，每组共享一套 K/V：

```
GQA（8 Q 头 / 4 KV 头，2:1 分组）：
  Q 头 0,1  → K/V 组 0        ← 头 0 和 1 共用 K/V 组 0
  Q 头 2,3  → K/V 组 1
  Q 头 4,5  → K/V 组 2
  Q 头 6,7  → K/V 组 3
```

- 🔑 **minimind 用的正是 8 Q 头 / 4 KV 头 = 2:1**。每个 KV 头服务 2 个 Q 头。K/V 缓存直接减半（4 套 vs 8 套），而质量损失远小于 MQA——**用"分组"做平滑的折中**。
- 💡 GQA 还是"参数共享"的另一种形式：KV 的线性层只有 `4 × head_dim × hidden`，而 MHA 要 `8 × head_dim × hidden`，省了一半 K/V 参数。

### repeat_kv：把 K/V 广播回每个 Q 头

训练/推理时，Q 是 8 个头，K/V 只有 4 组。要把 K/V **复制**成 8 份才能和 Q 做矩阵乘法。这就是 `repeat_kv`：

```python
def repeat_kv(x, n_rep):
    """x: (B, T, n_kv_heads, head_dim) → 复制 n_rep 份 → (B, T, n_kv_heads*n_rep, head_dim)"""
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]                  # (B,T,nh,1,hd) 扩一维
            .expand(bs, slen, num_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, num_kv_heads * n_rep, head_dim))
```

- 🔑 `n_rep = n_heads // n_kv_heads = 8 // 4 = 2`。`expand` 是"逻辑复制"（不拷贝内存），`reshape` 把复制的 2 份摊开。**注意 KV 线性层的输出头数是 `n_kv_heads`（4），不是 `n_heads`（8）**——这是 GQA 和 MHA 在代码上最直接的差别。

[03_gqa_kv_cache.py](../scripts/03_gqa_kv_cache.py) 里 GQA 的完整 forward：

```python
q = self.q_proj(x).view(B, T, n_heads, head_dim)      # 8 头
k = self.k_proj(x).view(B, T, n_kv_heads, head_dim)   # 4 组 ← 注意！
v = self.v_proj(x).view(B, T, n_kv_heads, head_dim)   # 4 组
# 应用 RoPE（上章）
q, k = apply_rotary_pos_emb(q, k, cos, sin)
# GQA 关键：把 K/V 广播回 8 份，然后转成 (B, n_heads, T, head_dim)
k = repeat_kv(k, n_rep).transpose(1, 2)               # (B,8,T,hd)
v = repeat_kv(v, n_rep).transpose(1, 2)
scores = (q.transpose(1, 2) @ k.transpose(-2, -1)) / math.sqrt(head_dim)
```

- ⚠️ 对比 Part 6 的 MHA：那里 `k_proj` 输出 `n_heads` 份、不用 `repeat_kv`。GQA 只改了两处——**KV 的投影头数**和**多一次 `repeat_kv`**。数学上 GQA 训练结果与"等参数量 MHA"接近，但推理 KV 缓存小一半。

### GQA 到底省了多少：参数与 KV 缓存（以 hidden=512 / 8 头 / 8 层为例）

| | KV 头数 | KV 线性层参数（每层） | KV 缓存/token（每层） | 4096 token 全 8 层缓存（fp16） |
|---|:---:|:---:|:---:|:---:|
| MHA | 8 | 2×(512×512) ≈ **524K** | 2×8×64 = 1024 | ≈ **67 MB** |
| **GQA（minimind）** | **4** | 2×(512×256) ≈ **262K** | 2×4×64 = 512 | ≈ **33 MB** |
| MQA | 1 | 2×(512×64) ≈ **65K** | 2×1×64 = 128 | ≈ **8 MB** |

- 🔑 读这张表：**GQA 相对 MHA 把 KV 参数和 KV 缓存都减半**（262K vs 524K、33MB vs 67MB），而质量损失远小于 MQA。KV 缓存随 `seq_len` 线性增长，序列越长、层数越多，省得越多——这就是大模型长上下文推理几乎都用 GQA 的原因。
- 💡 注意：**Q 的投影完全不受影响**（还是 8 头、`8×64×512`）。GQA 只"砍 K/V"，不碰 Q——因为"多视角查询"的能力全靠 Q 头。

## KV Cache：生成时只算最后一个 token

推理生成（自回归）时，每次只产生**一个新 token**，然后拿它拼到序列尾部再跑一遍整个 Transformer——这样做非常浪费：前面所有 token 的 K/V 每次都要重算。

**KV Cache 的洞察**：第 `t` 步算出的 K/V，在第 `t+1`、`t+2`... 步里**完全不变**（它们只依赖前面的 token，而前面的 token 不变）。所以：

```
朴素生成（每个新 token 重跑整个序列）：
  step 1: 重算 token 0~0 的 K/V → 输出 token 1
  step 2: 重算 token 0~1 的 K/V → 输出 token 2   ← 0~1 的 K/V 重复算了！
  step 3: 重算 token 0~2 的 K/V → 输出 token 3   ← 又重复！

带 KV Cache：
  step 1: 算 token 0 的 K/V，缓存起来 → 输出 token 1
  step 2: 只算 token 1 的 K/V，和缓存拼接 → 输出 token 2
  step 3: 只算 token 2 的 K/V，和缓存拼接 → 输出 token 3
```

- 🔑 关键点：**注意力里对 token `t` 而言，K/V 只来自它之前的 token**（因果遮罩）。新 token 的 K/V 只由"输入序列"决定，与"之后生成了什么"无关，所以可以缓存、拼接。
- 💡 复杂度对比：朴素生成每步是 `O(T²)`（重算全序列），带 KV Cache 每步是 `O(T)`（只算最后一个 token 的注意力）。生成 N 个 token，从 `O(N²·L)` 降到 `O(N·L)`——**长文本生成的加速是数量级的**。

代码里 KV Cache 就是"拼接 + 存储"：

```python
# 推理时把历史 K/V 拼接起来（generation 时传入 past_key_value）
if past_key_value is not None:
    k = torch.cat([past_key_value[0], k], dim=1)   # 历史 K + 新 K
    v = torch.cat([past_key_value[1], v], dim=1)   # 历史 V + 新 V
# 只对最后一个位置算注意力（生成时只需最后一行的分数）
scores = (q[:, -1:, :, :] @ k.transpose(-2, -1)) / math.sqrt(head_dim)
```

- ⚠️ 有了 KV Cache 后，RoPE 的角度要**接着之前的位置算**（`start_pos` 偏移），不能从头数——这是"带缓存 + RoPE"最容易踩的坑。

### 把 GQA + RoPE + KV Cache 组装进一个 Attention

前面是"零件"，这里把它们装成一个完整的 `Attention` 模块（对照 minimind 的 `Attention` 类）：

```python
class Attention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads):
        super().__init__()
        self.n_rep = n_heads // n_kv_heads              # 8//4 = 2
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=False)      # 8 头
        self.k_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)   # 4 组
        self.v_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)   # 4 组
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)

    def forward(self, x, cos, sin, past_key_value=None):
        bsz, seq_len, _ = x.shape
        start_pos = past_key_value[0].shape[1] if past_key_value else 0

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # ① RoPE：只旋转本段（从 start_pos 接着算）
        q, k = apply_rotary_pos_emb(q, k, cos[start_pos:start_pos+seq_len],
                                        sin[start_pos:start_pos+seq_len])

        # ② KV Cache：拼接历史 K/V（只在推理生成时走）
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)   # 历史 K + 新 K
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v)

        # ③ GQA：把 K/V 广播回 8 份，转成 (B, n_heads, T, head_dim)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)
        q = q.transpose(1, 2)

        # ④ 注意力：生成时 q 只取最后一行，且无需再遮罩（缓存里全是历史）
        q = q[:, -1:] if past_key_value is not None else q
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if past_key_value is None:                    # 只有训练/预填充才需要因果遮罩
            scores = scores.masked_fill(torch.triu(torch.ones_like(scores), diagonal=1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out), past_kv
```

- 🔑 四个步骤的职责非常清晰：**① RoPE 给位置，② KV Cache 给历史，③ GQA 广播 KV，④ 算注意力**。`start_pos` 是"带缓存 + RoPE"的关键——新 token 的角度要从 `start_pos` 接着转，不能从 0 重新数。
- ⚠️ 训练和推理走同一份代码：训练时 `past_key_value=None`（②跳过、④算全序列、带因果遮罩），推理时传入缓存（②拼接、④只算最后一行、且**无需再遮罩**——缓存里全是当前 token 之前的历史）。这正是现代 LLM 推理加速的全部秘密。

## Flash Attention：让 GPU 更快

> 🔧 这些优化本质都是 GPU 内核层面的活（shared memory tiling、fused kernel）——
> 想亲手写一遍的话，见 [Part 9 CUDA 内核](../../Part9_cuda_kernels/tutorial/02_matmul_optimization.md)：
> 其中 02 章手写的 SMEM tiling 与 Flash Attention 共享同一套核心思想。

Part 6 提过 PyTorch 2.0 的 `F.scaled_dot_product_attention`（SDPA），minimind 默认也用它：

```python
if self.flash and (seq_len > 1):
    output = F.scaled_dot_product_attention(
        xq, xk, xv, is_causal=True)   # 自带因果遮罩，又快又省显存
else:
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(head_dim)
    # ... 手动加因果遮罩 ...
```

- 💡 Flash Attention 的核心里面没新数学：只是**按块（tile）计算、不落整张 attention 矩阵**，把对显存的读写从 `O(T²)` 降到 `O(T)`。**结果和普通 attention 数值上几乎一致，只是更快、更省内存**。我们用 `F.scaled_dot_product_attention` 一行拿到底。

> 🔧 这些优化本质都是 GPU 内核层面的活（shared memory tiling、fused kernel）——
> 想亲手写一遍的话，见 [Part 9 CUDA 内核](../../Part9_cuda_kernels/tutorial/02_matmul_optimization.md)：
> 其中 02 章手写的 SMEM tiling 与 Flash Attention 共享同一套核心思想。


## SwiGLU：把 FFN 的非线性换掉

### 回顾 Part 6 的 ReLU FFN

Part 6 的 FeedForward 是：

```python
Linear(hidden → 4·hidden) → ReLU → Linear(4·hidden → hidden)
```

"开大的空间思考，再压回去"。ReLU 的问题：**在 0 处有个尖角（不可导），负半轴直接归零**——信息一旦为负就"死了"，且梯度容易在负区间为 0。

### SwiGLU：三投影 + 自适应门控

**SwiGLU（2020）** 是 GLU（门控线性单元）家族的一员。GLU 的想法：**用另一个投影当作"门"，决定主投影保留多少**：

```
SwiGLU:  FFN(x) = down_proj( silu( gate_proj(x) ) * up_proj(x) )
                          ↑ 门（开关）           ↑ 内容
```

拆开看，FFN 变成**三个投影**（gate/up/down）：

```python
class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)  # 门
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)  # 内容
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)  # 压回

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- `gate_proj(x)` 过 `silu`（即 Swish，PyTorch 里叫 `F.silu`）后变成 0~1 之间的"软开关"：`silu(z) = z · σ(z)`，在 0 处**平滑**（不是尖角），负区间不会完全死掉（保留很小但不为 0 的梯度）
- `up_proj(x)` 是"内容"
- 两者**逐元素相乘** = "门控内容"：门想放多少就放多少
- `down_proj` 把结果压回 hidden

### 为什么 SwiGLU 更好？

1. **平滑、梯度干净**：silu 处处可导、负区间梯度不为 0，比 ReLU 的"尖角 + 归零"更好优化，小模型上往往更稳。
2. **自适应门控**：ReLU 是"硬开关"（<0 一律关），SwiGLU 是"软开关"（每个维度有独立的 0~1 门，由数据学出来）——表达力更强。
3. **效果更好**：论文和 Llama 系列证明，同参数量下 SwiGLU 优于 ReLU FFN（Llama 2 的 FFN 就是这个结构）。

> ⚠️ 代价：从 2 个投影变成 **3 个**，中间维度却从经典的 `4×` 缩到 minimind 的 `~3.2×`（`ceil(hidden·π/64)·64`），参数总量和 ReLU FFN 差不多——**用"更宽但更高效的结构"换性能**。
>
> 💡 `ceil(hidden·π/64)·64` 这个公式：经典 ReLU FFN 中间维度是 `4×hidden`；SwiGLU 多一个投影，为了控制总参数量，minimind 把中间维度缩到约 `3.14×hidden`（π ≈ 3.14），再向上对齐到 64 的倍数（GPU tensor core 对齐友好，64 是常见的 tile 大小）。脚本 [04_swiglu_ffn_moe.py](../scripts/04_swiglu_ffn_moe.py) 用 `int((math.pi * hidden / 64) + 0.5) * 64` 实现（四舍五入版，效果等价）。

### 数值例子：silu vs ReLU（手算）

`silu(z) = z · σ(z)`，σ 是 sigmoid。拿几个值对比 ReLU：

| z | ReLU(z) | silu(z) | 说明 |
|:---:|:---:|:---:|---|
| -3 | 0 | -0.14 | ReLU 直接关死；silu 保留一点"负的微量" |
| -1 | 0 | -0.27 | 同上，梯度仍不为 0 |
| 0 | 0 | 0 | 平滑经过（ReLU 是尖角） |
| 1 | 1 | 0.73 | silu 略"收一点" |
| 3 | 3 | 2.86 | 大正值接近线性 |

- 💡 关键差别在负半轴：ReLU 对任何负数都输出 0（信息"死"了、梯度为 0）；silu 对负数输出**很小的负值**，梯度仍然存在——优化器能"告诉"这个维度该往哪调，而不是断了信号。门控场景里，这个"软开关"远比"硬截断"平滑。

### 对比表

| | Part 6 ReLU FFN | Part 7 SwiGLU |
|---|---|---|
| 结构 | 2 投影 | **3 投影（gate/up/down）** |
| 激活 | ReLU | **silu（平滑）** |
| 门控 | 无（硬截断） | **软门控（逐维 0~1）** |
| 中间维度 | 4× | ~3.2× |
| 表达力/效果 | 基线 | **更好** |

## MoE：把 FFN 换成一群"专家"

### 概念：多个 FFN + 路由器

**MoE（Mixture of Experts，混合专家）** 的思路：不只有一个 FFN，而是**训练很多个 FFN（专家）+ 一个路由器**。每个 token 输入时，路由器先给它打个分，**只把它路由到最合适的 top-k 个专家**去处理：

```
                token x
                   │
              ┌────▼────┐
              │ router  │  算每个专家的分数，选 top-k
              └────┬────┘
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   专家 0       专家 1      专家 2   （每个都是一个小 FFN）
     FFN₀        FFN₁       FFN₂
        └──────────┼──────────┘
           加权求和（只对 top-k 加权）
                   │
                  y
```

- 🔑 **稀疏性**：每个 token 只激活 top-k 个专家（如 4 选 1），其它专家"睡觉"。模型参数量很大（一堆专家），但**每个 token 的实际计算量很小**——"参数多、算力省"。
- 💡 一个直觉：**MoE = 按 token 内容"分工"**。代码片段走"写代码专家"，散文走"写作专家"。路由器学会这个分工。

### 代码：top-k 路由

minimind 的 MoE FFN（[04_swiglu_ffn_moe.py](../scripts/04_swiglu_ffn_moe.py)，minimind 用 4 专家 / top-1）：

```python
class MoE(nn.Module):
    def __init__(self, hidden, num_experts=4, top_k=1):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)      # 路由器
        self.experts = nn.ModuleList([FFN(hidden) for _ in range(num_experts)])

    def forward(self, x):
        scores = F.softmax(self.gate(x), dim=-1)      # 每个专家一个分数
        topk_w, topk_idx = torch.topk(scores, self.top_k, dim=-1)   # 取 top-k
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)              # top-k 内归一化
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)                    # 哪些 token 选了专家 i
            if mask.any():
                out[mask] += topk_w[mask] * expert(x[mask])        # 加权
        return out
```

- ⚠️ **负载均衡（load balance）**：如果路由器"偏爱"某个专家，其它专家就废了。所以要加 **auxiliary loss（辅助损失）**：惩罚"某个专家被选得过多"。minimind 用 `aux_loss = (load * scores.mean()).sum() * num_experts * coef`（load 是每个专家的平均被选次数，scores.mean 是平均得分，两者乘积大 = 负载不均），把它加到总损失上。这是 MoE 工程里**必做**的一步。

### minimind 的 MoE 是可选项

- 🔑 minimind 的默认 `MiniMindConfig(use_moe=False)` 是 **Dense（稠密）模型**；把 `use_moe=True` 就切换成 MoE 版（4 专家 / top-1）。
- 💡 对我们 26M 的小模型，MoE 属于"锦上添花"：理解概念为主，训练脚本默认不开 MoE。等你有 GPU 想复现 minimind-MoE（145M）再开。

## 学完本部分你能...

- ✅ 讲清"为什么 K/V 是显存瓶颈"（KV 缓存 ∝ n_kv_heads × seq_len）
- ✅ 对比 MHA / MQA / GQA，说出 minimind 用 8 Q 头 / 4 KV 头（2:1 分组）
- ✅ 手写 `repeat_kv`，指出 GQA 与 MHA 在代码上的差别（KV 投影头数）
- ✅ 实现 KV Cache：生成时只算最后 token、复用历史 K/V，理解 `O(T²)→O(T)` 的加速
- ✅ 手写 SwiGLU（gate/up/down 三投影），对比 ReLU FFN
- ✅ 讲清 MoE 的路由、top-k、负载均衡损失，知道它是 minimind 的可选项

## 课后练习

<details>
<summary>Q1: 为什么 GQA 能省显存却不怎么掉质量？"分组"到底在省什么？</summary>
A: 省的是 K/V 的"套数"（KV 头数）：MHA 每头一套，GQA 把 Q 头分组、每组共享一套。KV 缓存正比于 KV 头数，所以 8→4 就减半。质量损失小的原因是：K/V 表达的是"被查询的内容"，不同 Q 头对内容的需求高度重合，分组合并只是去掉冗余；而 Q 头仍然各自独立，保留了"多视角查询"的能力。从 MHA→GQA 是平滑折中，MQA（1 套）就砍得太狠了。
</details>

<details>
<summary>Q2: KV Cache 为什么对推理（生成）有效，对训练没用？</summary>
A: 训练时每个 batch 里所有位置的 Q/K/V 都要一起算、一起反向传播，缓存 K/V 反而打乱计算图，毫无意义。推理生成是"逐步、不反向"的，且前面的 token 固定不变，它们的 K/V 也就不变——缓存后每步只算新 token 的注意力，把每步成本从 O(T²) 降到 O(T)。本质是"自回归生成中，前缀计算天然可复用"。
</details>

<details>
<summary>Q3: MoE 的参数量和计算量为什么不相等？负载均衡损失解决了什么问题？</summary>
A: MoE 把所有专家都"装"进模型，所以参数量很大（一堆 FFN）；但每个 token 只走 top-k 个专家，实际计算量只和 top-k 成正比，两者脱钩——"参数多、算力省"。负载均衡损失解决"路由器把 token 全堆给某几个专家"的问题：它会惩罚"被选过多次的专家"（load 大）和"得分高的专家"（scores.mean 大）的重叠，逼路由器把 token 摊开，避免专家"饿死/撑死"。
</details>

## 📝 课后作业

完成本章后，去 Assignment 7 完成题 4（repeat_kv）和题 5（SwiGLU）：

👉 [Assignment 7](../../../assignments/assignment_7/)

## 下一步

现代 LLM 的"零件"全部到齐了：BPE tokenizer、RMSNorm、RoPE、GQA + KV Cache、SwiGLU（MoE 可选）。但**零件好不等于模型好用**——接下来是真正的重头戏：把模型按现代方式**训练**成助手。预训练 → SFT → DPO，每一步解决什么问题、代码怎么写，下一章见分晓。

👉 [04 — 训练流水线：Pretrain → SFT → DPO](04_training_pipeline.md)