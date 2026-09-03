# RoPE 旋转位置编码 · 学习笔记

> 来源：Part7_minimind/tutorial/02_modern_components.md
> 配套交互式教程：rope-rotary-position-embedding.html

---

## 一、概念清单（一句话定义）

| 概念 | 一句话定义 |
|------|-----------|
| **位置编码** | 让 Transformer 模型知道 token 在序列中位置的机制。没有它，模型无法区分词序。 |
| **可学习位置编码（Learned PE）** | 用一张可学习的 embedding 表存储每个位置的向量，直接加到 token embedding 上。绝对位置编码，不能外推。 |
| **RoPE（Rotary Position Embedding）** | 旋转位置编码：把 q/k 向量按位置旋转一个角度，使内积只依赖相对位置。零参数、可外推。 |
| **旋转矩阵** | 正交变换的一种，将向量绕原点旋转 θ 角。保范数、保内积的角度差性质。 |
| **频率分解** | RoPE 把高维向量两两分组，每组用不同的旋转频率（指数衰减分布），类似傅里叶分解。 |
| **rotate_half 技巧** | 把向量后半取负放前半，配合 cos/sin 逐元素相乘实现旋转，避免显式矩阵乘法。 |
| **RoPE Scaling** | 增强 RoPE 长上下文外推能力的技术，包括 PI（位置插值）、NTK-aware、YaRN 等方案。 |

---

## 二、核心要点（What / Why / How / How much）

### RoPE 的 What
一种**相对位置编码方法**，通过旋转 q/k 向量注入位置信息，使 attention 内积只依赖位置差。

### RoPE 的 Why
解决可学习 PE 的两大痛点：
1. **需要参数**（block_size × hidden 一张表）
2. **不能外推**（超过训练长度直接崩）

同时内建相对位置感知——语言中词语的意义更多依赖相对位置，而非绝对位置。

### RoPE 的 How
分两步：

**第一步：precompute_freqs_cis（预计算 cos/sin 表）**
```python
freqs = 1.0 / (rope_base ** (arange(0, dim, 2).float() / dim))  # 每组频率
angles = outer(arange(end), freqs)  # 位置 × 频率 = 角度表
cos = cat([cos(angles), cos(angles)], dim=-1)  # 拼接一份配合 rotate_half
sin = cat([sin(angles), sin(angles)], dim=-1)
```

**第二步：apply_rotary_pos_emb（应用旋转）**
```python
def rotate_half(x):
    return cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)

q_rot = q * cos + rotate_half(q) * sin  # 等价于复数乘法 x * e^(iθ)
k_rot = k * cos + rotate_half(k) * sin
```

**第三步：放进 Attention**
- q、k 投影后，取对应位置的 cos/sin
- apply_rotary_pos_emb 旋转
- 照常算 attention scores

### RoPE 的 How much
- **参数**：0 个（cos/sin 是 buffer 不是 parameter）
- **计算开销**：多一步逐元素乘加（相比可学习 PE 的加法）
- **外推能力**：可直接外推（但超长会衰减，需 Scaling 辅助）

---

## 三、卡点记录

### 卡点 1：旋转的是 q/k，不是 token embedding
**正确理解**：RoPE 作用在 attention 的 q 和 k 上，通过影响内积来影响注意力。v 不旋转。位置信息是在 attention 计算阶段注入的，不是在输入阶段。

### 卡点 2：为什么"零参数"还能工作？
**正确理解**：频率是人工设计的固定分布（θ_base = 10000 的指数分布），但模型通过学习 q/k 的内容向量来适应这个"尺子"，让旋转后的向量恰好能匹配正确的相对位置。

### 卡点 3：rotate_half 为什么等价于旋转？
**正确理解**：旋转公式的复数形式是 x' = x·cosθ + i·x·sinθ。
- rotate_half(x) 实现了 i·x（后一半取负放前半 = 乘以虚数单位）
- 配合 cos 和 sin 逐元素相乘，就等价于完整的旋转运算
- 这样做避免了显式的矩阵乘法，更高效

### 卡点 4：KV Cache 场景下的位置偏移
**正确理解**：生成时已有 start_pos 个 token，新 token 的位置从 start_pos 开始计数。cos/sin 表要从 start_pos 偏移取（cos[start_pos:start_pos+T]），否则位置算错。

### 卡点 5：base 越大越好？
**正确理解**：base 控制频率的分布范围。base 越大，低频组越慢、覆盖的位置范围越广，但每组之间的频率分辨率越低。base = 10000 是经验值。

---

## 四、检测题答案

### 第 1 题：B
RoPE 的核心是旋转 q/k 向量，用角度差编码相对位置。A 是可学习 PE，C 是正弦位置编码，D 是相对位置偏置（如 T5 的方法）。

### 第 2 题：C
旋转是正交变换，旋转后的内积 = 原内积的角度差形式。q 旋转 mθ、k 旋转 nθ，内积只取决于 (n-m)θ，即位置差 × 频率。

### 第 3 题：B
freq[i] = 1/θ_base^(2i/dim)，随 i 指数衰减。i=0 是高频（快转、细粒度），i 增大频率降低（慢转、粗粒度）。

### 第 4 题：C
rotate_half 把向量后一半取负放到前一半，等价于乘以虚数 i。配合 cos/sin 逐元素相乘，实现 x' = x·cos + i·x·sin。

### 第 5 题：D
RoPE 的计算量比可学习 PE 更大（需要逐元素乘加，可学习 PE 只是加法）。A、B、C 都是 RoPE 的优势。

### 第 6 题（变体题）：B
base 增大 → 整体频率降低 → 覆盖更大位置范围，但高频维度之间的区分度下降。

### 第 7 题（串联题）：B
KV Cache 中的旧 k 已经在正确位置旋转过了，不需要重转。新 token 的 q/k 要从当前已生成长度（start_pos）开始取 cos/sin 表。

---

## 五、知识网络（概念关系图）

```
                    ┌─────────────┐
                    │  旋转矩阵   │
                    │ 正交·保范数 │
                    └──────┬──────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  频率分解   │────▶│    RoPE     │◀────│ 相对位置编码 │
│ 指数衰减分布 │     │ 旋转位置编码 │     │ 内积依赖位置差│
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌────────────┐ ┌──────────┐ ┌─────────────┐
      │ 可学习 PE  │ │Self-Attn │ │ RoPE Scaling│
      │ (对比基准) │ │  q·k内积 │ │ PI/NTK/YaRN │
      └────────────┘ └──────────┘ └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             ┌──────────┐  ┌──────────┐
             │KV Cache  │  │  MLA     │
             │ / GQA    │  │(DeepSeek)│
             │位置偏移  │  │解耦 RoPE  │
             └──────────┘  └──────────┘
```

**核心关系**：
- 旋转矩阵 + 频率分解 → 构成 RoPE 的数学基础
- RoPE → 实现相对位置编码（内积只依赖位置差）
- RoPE → 替代可学习 PE（零参数、可外推）
- RoPE → 作用于 Self-Attention 的 q/k 内积
- RoPE + Scaling → 长上下文外推
- RoPE + KV Cache → 注意位置偏移问题
- RoPE + MLA → 解耦 RoPE（低秩压缩时的特殊处理）

---

## 六、一句话总结

> **RoPE = 给 q/k 按位置转一下，用角度差编码相对位置。零参数、可外推，现代 LLM 标配。**
