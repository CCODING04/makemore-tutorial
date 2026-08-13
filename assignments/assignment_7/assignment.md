# 作业 7：从零复现 minimind —— 现代 LLM 的六大核心组件

> **对应教程**：Part 7 — 从零复现 minimind（tiny LLM 实战）
>
> **前置**：建议先完成作业 6（Transformer/GPT，理解注意力与残差时需对照）

---

## 📋 概述

本作业带你从零实现现代 LLM（以 minimind 为蓝本）的六个关键组件。它们把 Part 6
的"教学版 Transformer"升级成生产级架构：

1. BPE 分词器的编码（subword tokenizer，替代字符级）
2. RMSNorm（简化归一化，替代 LayerNorm）
3. RoPE 旋转位置编码（替代可学习位置编码）
4. GQA 分组的 K/V 头复制 repeat_kv（替代标准 MHA）
5. SwiGLU 前馈网络（替代 ReLU FFN）
6. DPO 直接偏好优化损失（对齐，替代 RLHF 的奖励模型 + PPO）
7. 🌟 KV Cache 推理缓存（自回归生成加速）

完成本作业后，你应该能够：

- 理解 BPE 为什么平衡"词表大小"与"序列长度"，能按 rank 顺序合并相邻对
- 理解 RMSNorm 为什么比 LayerNorm 更省（去掉均值中心化与 bias）
- 理解 RoPE 用"旋转"编码位置：正交变换保范数、只依赖相对位置
- 理解 GQA 用"分组共享 K/V"在参数量与表达力之间折中
- 理解 SwiGLU 用"软门控"替代 ReLU 的硬截断
- 理解 DPO 为什么能"不需要奖励模型"直接优化偏好
- 理解 KV Cache 为什么能把自回归生成从 O(T²) 降到 O(T)

---

## 🔧 环境准备

### 依赖

```bash
pip install torch
```

### 数据

本作业的数据路径指向 `../../data/names.txt`（32K 个人名），但**题目本身不依赖
具体数据**——所有组件都在随机张量上验证。数据路径保留是为了与其它作业的
路径约定保持一致。

### 文件结构

```
assignments/assignment_7/
├── assignment.md               # 本文件
├── minimind_exercises.py       # 👈 你需要编辑的文件
└── test_minimind_exercises.py  # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_7
python test_minimind_exercises.py
# 或（pytest 兼容）
pytest test_minimind_exercises.py
```

> 测试是**属性测试**：只检查 shape / dtype / 数学不变量（范数、顺序、方向、值域），
> 不检查精确数值。未实现的题目（返回 `None`）会被**优雅跳过**，不会报错。

---

## 📝 题目列表

### 题 1：BPE 编码（基础）

**函数**：`exercise_1_bpe_encode(text, merges, vocab)`

**目标**：实现 BPE 的**编码**阶段——按合并规则把文本切分成子词 token。

**要求**：

- 把 `text` 拆成单字符列表
- 反复找 **rank 最小**（`merges` 列表里最靠前）且当前出现的相邻对 `(a, b)`
- 若多个候选 rank 相同，取**最靠左**的那对
- 合并后重新扫描（合并可能产生新的可合并对）
- 用 `vocab` 把最终 token 列表映射成 id 列表

**验证**：
```python
merges = [('a', 'b'), ('ab', 'c')]
tokens = ['a', 'b', 'c', 'ab', 'abc']
vocab = {t: i for i, t in enumerate(tokens)}
print(exercise_1_bpe_encode('abcabc', merges, vocab))  # [4, 4] 即 ['abc','abc']
```

**思考**：
- 为什么 BPE 按"出现频率"合并，而不是按"字符位置"？
- `'abcabc'` 里 `(a,b)` 出现两次，为什么只合并最靠左的那对？

---

### 题 2：RMSNorm（基础）

**类**：`RMSNorm(nn.Module)`

**目标**：实现 RMSNorm 归一化层——现代 LLM 替代 LayerNorm 的标准选择。

**要求**：

- `__init__(self, dim, eps=1e-6)`：`self.weight = nn.Parameter(torch.ones(dim))`
- `forward(x)`：`rms = sqrt(mean(x²) + eps)`，返回 `x / rms * weight`
- 输入 `(B, T, dim)` 或任意形状，输出同形状

**验证**：
```python
norm = RMSNorm(32)
out = norm(torch.randn(4, 8, 32))
print(out.shape)                  # (4, 8, 32)
print(out.pow(2).mean(-1).sqrt()) # ≈ 1（weight 全 1）
```

**与 LayerNorm 的区别**（Part 6 用的是 LayerNorm）：
- LayerNorm 先减均值再除以标准差；RMSNorm **不减均值**，只按均方根缩放
- LayerNorm 有 bias（beta）；RMSNorm **没有 bias**
- 参数量减半、计算更省，效果相近且更稳

**思考**：
- 为什么 Transformer 里"均值中心化"可以省掉？（提示：residual + 相邻 token 均值的
  信息在 attention 里可被吸收）
- 零输入时 `mean(x²)=0`，`eps` 如何避免除零？

---

### 题 3：RoPE 旋转位置编码（基础）

**函数**：`exercise_3_apply_rope(q, freqs_cis)`

**目标**：把旋转位置编码应用到 query/key 上——用复数乘法实现"按位置旋转"。

**要求**：

- `q` 形状 `(B, T, num_heads, head_dim)`，`head_dim` 为偶数
- `freqs_cis` 形状 `(T, head_dim//2)`，是**单位模长**的复数（`exp(iθ)`）
- 把 `head_dim` 维每对元素 `(x0, x1)` 看作复数 `x0 + x1·i`
- 与 `freqs_cis` 逐元素复数相乘（旋转），再拼回实数
- 返回形状与 `q` 相同

**验证**：
```python
q = torch.randn(2, 6, 4, 8)
theta = 10000.0 ** (-torch.arange(4).float() / 4)
t = torch.arange(6).float()
freqs_cis = torch.complex(torch.cos(t[:,None]*theta[None,:]),
                          torch.sin(t[:,None]*theta[None,:]))
out = exercise_3_apply_rope(q, freqs_cis)
print(out.shape)                       # (2, 6, 4, 8)
print((out.norm(-1) - q.norm(-1)).abs().max())  # ≈ 0（范数不变）
```

**关键性质**：
- 旋转是**正交变换**：向量范数不变，信息不丢失
- 两个 token 的 q/k 内积**只依赖相对位置**（差 m−n），不依赖绝对位置
- 相比 Part 6 的 learned position embedding：**无参数**、可外推到更长序列

**思考**：
- 为什么旋转能编码位置？（提示：频率 θᵢ 不同的维度旋转速度不同）
- `view_as_complex` 要求最后一维大小为 2，怎么 reshape？

---

### 题 4：GQA 分组的 K/V 头复制（基础）

**函数**：`exercise_4_repeat_kv(x, n_rep)`

**目标**：实现 GQA（Grouped-Query Attention）的核心——把 K/V 头复制到与 Q 头一致。

**要求**：

- `x` 形状 `(B, num_kv_heads, T, head_dim)`
- `n_rep = num_heads // num_kv_heads`
- 返回形状 `(B, num_kv_heads * n_rep, T, head_dim)`
- **第 `i` 个输出头 == 第 `i // n_rep` 个原始头**（按 `[h0,h0,h1,h1,...]` 排列）

**验证**：
```python
x = torch.randn(3, 4, 7, 16)
out = exercise_4_repeat_kv(x, 2)
print(out.shape)              # (3, 8, 7, 16)
print(torch.allclose(out[:, 3], x[:, 1]))  # True（第 3 头 == 第 1 个原始头）
```

**背景**：
- **MHA**：每头独立 K/V（8 个 Q 头配 8 个 K/V 头）
- **MQA**：所有 Q 头共享一组 K/V（省但表达力下降）
- **GQA**：分组共享（8 个 Q 头配 4 个 K/V 头，`n_rep=2`）——折中，minimind 采用

**思考**：
- 为什么 K/V 头是显存瓶颈？（提示：KV Cache 里每个 token 都要存 K/V）
- `repeat_interleave` 与 `expand` 的实现有何区别？

---

### 题 5：SwiGLU 前馈网络（基础）

**类**：`SwiGLU(nn.Module)`

**目标**：实现现代 LLM 的标准 FFN——用"软门控"替代 ReLU 的硬截断。

**要求**：

- `__init__(self, dim, hidden_dim=None)`：`hidden_dim` 默认 `4*dim`
- 定义三个无 bias 的线性层：`gate_proj` / `up_proj` / `down_proj`
- `forward(x)`：`down_proj(silu(gate_proj(x)) * up_proj(x))`
- 输入 `(B, T, dim)`，输出同形状

**验证**：
```python
ffn = SwiGLU(32)
out = ffn(torch.randn(4, 8, 32))
print(out.shape)   # (4, 8, 32)
```

**与 ReLU FFN 的区别**（Part 6 用的是 ReLU）：
- ReLU FFN：`down(relu(x@W1))`，负值硬截断为 0
- SwiGLU：`down(silu(gate(x)) * up(x))`，`gate` 学习"放行多少"，`up` 提供内容
- SiLU（swish）平滑可微，负区仍有小梯度，训练更稳

**思考**：
- `gate` 和 `up` 分别起什么作用？
- 为什么不用 bias？（提示：现代 LLM 的初始化与参数省）

---

### 题 6：DPO 直接偏好优化损失（基础）

**函数**：`exercise_6_dpo_loss(pi_logps_chosen, pi_logps_rejected,
                        ref_logps_chosen, ref_logps_rejected, beta=0.1)`

**目标**：实现 DPO 损失——不需要奖励模型和 PPO，直接用偏好对优化策略。

**要求**：

```python
log_pi_chosen   = pi_logps_chosen   - ref_logps_chosen
log_pi_rejected = pi_logps_rejected - ref_logps_rejected
logits          = log_pi_chosen     - log_pi_rejected
loss = -F.logsigmoid(beta * logits).mean()
```

**验证**：
```python
# chosen 明显更优时 loss 更小
loss_good = exercise_6_dpo_loss(torch.randn(8)*0.5+1.0, torch.randn(8)*0.5-1.0,
                                torch.randn(8)*0.3, torch.randn(8)*0.3)
loss_bad  = exercise_6_dpo_loss(torch.randn(8)*0.5-1.0, torch.randn(8)*0.5+1.0,
                                torch.randn(8)*0.3, torch.randn(8)*0.3)
print(loss_good.item() < loss_bad.item())  # True
```

**背景**：
- 从 RLHF 出发：奖励模型 + PPO 复杂、不稳定
- DPO 的关键洞察：Bradley-Terry 模型可以**消掉奖励函数**，直接最大化
  "chosen 相对 rejected 的策略优势"（以冻结的参考策略为锚点）
- `beta` 控制与参考策略的偏离程度，防止策略跑偏太远

**思考**：
- 为什么需要参考策略 `ref` 且它**冻结**？
- `chosen == rejected` 时 loss 应该是多少？（提示：`-log sigmoid(0) = ln2`）

---

### 题 7：KV Cache 推理缓存（🌟 拓展）

**函数**：`exercise_7_kv_cache(k, v, past_k, past_v)`

**目标**：实现自回归生成的关键加速——把历史 K/V 缓存起来复用。

**要求**：

- `k, v` 形状 `(B, num_heads, T_new, head_dim)`
- `past_k`/`past_v` 为 None（首次）时，直接返回 `(k, v)`
- 否则在**时间维（dim=2）**拼接：`torch.cat([past_k, k], dim=2)`
- 返回 `(new_past_k, new_past_v)`，形状 `(B, num_heads, T_old+T_new, head_dim)`

**验证**：
```python
k0 = torch.randn(2, 4, 3, 16); v0 = torch.randn(2, 4, 3, 16)
pk, pv = exercise_7_kv_cache(k0, v0, None, None)     # 首次
print(torch.equal(pk, k0))                            # True
k1 = torch.randn(2, 4, 1, 16); v1 = torch.randn(2, 4, 1, 16)
pk2, pv2 = exercise_7_kv_cache(k1, v1, pk, pv)
print(pk2.shape)                                      # (2, 4, 4, 16)
```

**为什么快**：没有 cache 时，生成第 T 个 token 要重新计算前 T−1 个 token 的
K/V（总 O(T²)）；有 cache 时每步只算新 token 的 K/V（总 O(T)）。

**思考**：
- KV Cache 与因果遮罩的关系：缓存里都是历史，还需要遮罩吗？
- 拼接为什么是 dim=2（时间维）而不是其它维？

---

## ✅ 提交检查清单

- [ ] 所有 6 道基础题通过测试
- [ ] 拓展题（题 7）已尝试
- [ ] 能回答每道题后面的「思考」问题
- [ ] 代码中添加了必要的注释说明你的理解

---

## 💡 学习建议

1. **按顺序做**：题 1 → 题 2 → 题 3 → 题 4 → 题 5 → 题 6 → 题 7。
   题 3/4 是注意力的一部分（RoPE + GQA），题 5 是 FFN，题 6 是训练目标，彼此独立。
2. **先跑测试再写代码**：`python test_minimind_exercises.py` 会告诉你哪些题没实现
   （跳过）、哪些实现有 bug（失败）。
3. **对照脚本**：`courses/Part7_minimind/scripts/` 下有对应组件的完整实现，
   `02_rmsnorm_rope.py`、`03_gqa_kv_cache.py`、`04_swiglu_ffn_moe.py`、
   `08_dpo_alignment.py`。看不懂时可参考。
4. **组件思维**：这些组件就像积木——理解每个单独组件，再在完整模型里把它们
   组合起来，就是 minimind 的架构。
5. **验证用脚本**：`python test_minimind_exercises.py` 的每个测试都只检查数学
   不变量（范数、顺序、方向），只要实现公式正确就能过。

---

## 🤔 思考题

**Q1：** 为什么现代 LLM 用 RMSNorm 而不是 LayerNorm？省掉"减均值"为什么没问题？

<details>
<summary>💡 提示</summary>

LayerNorm 做了两件事：减均值（中心化）+ 除标准差（缩放）。RMSNorm 只做缩放。

"减均值"在 Transformer 里信息量很低，原因有两层：
1. **残差连接**：每个 Block 的输出 `x = x + sublayer(x)`，均值信息已经被加性结构
   传递出去了，重复中心化是冗余的。
2. **相邻 token 均值可被吸收**：attention 是按权重聚合的线性运算，某个 token 的
   均值偏移可以被 attention 权重或后续投影吸收。

因此去掉均值中心化（省一次 mean 与减法）不损失表达能力，还能少一层可学习参数
（没有 bias），计算更省、训练更稳。实验上两者效果相当，RMSNorm 更快更省。

</details>

**Q2：** RoPE 为什么能编码相对位置？和 Part 6 的 learned position embedding 有何不同？

<details>
<summary>💡 提示</summary>

RoPE 把每个维度 i 的频率设为 θᵢ = 1/τ^(2i/dim)，位置 m 的旋转角 = m·θᵢ。
两个 token 在位置 m 和 n 的 q/k 内积，等于先各自旋转再内积；而两个旋转矩阵的
**相对**旋转角恰好是 (m−n)·θᵢ——所以内积只依赖位置差，不依赖绝对位置。

对比 Part 6 的 learned PE：
- **learned PE**：一张可学习 embedding 表，位置 m 查一个向量加进去。它隐含位置
  信息，但绝对位置是显式的，且超出训练长度无法外推。
- **RoPE**：没有参数，旋转是固定的数学变换。相对位置内积天然成立，且可以通过
  调整 θᵢ（YaRN 外推）处理更长序列。

RoPE 的代价是：需要在 q/k 上加一次旋转运算（复数乘法），略复杂于查表。

</details>

**Q3：** GQA 为什么是"分组共享 K/V"？它相比 MHA 和 MQA 各有什么取舍？

<details>
<summary>💡 提示</summary>

- **MHA**：8 个 Q 头，配 8 个独立的 K/V 头。表达力最强，但 K/V 投影的参数量和
  KV Cache 的显存占用最大（每个 token 要存 8 份 K/V）。
- **MQA**：8 个 Q 头，共享 **1 组** K/V。K/V 参数和显存降到 1/8，但共享到极限后
  表达力下降明显，模型质量受损。
- **GQA**：8 个 Q 头，配 **4 组** K/V（`n_rep=2`）。折中：K/V 参数和显存减半，
  质量损失很小。minimind 采用 8Q/4KV，这就是 `exercise_4_repeat_kv` 做的事——
  把 4 组 KV 头复制 2 次，让形状与 8 个 Q 头匹配做注意力。

一句话：**GQA = MHA 的质量，接近 MQA 的效率**。

</details>

**Q4：** DPO 为什么能"不需要奖励模型"？它和 RLHF/PPO 的关系是什么？

<details>
<summary>💡 提示</summary>

RLHF 的三步：
1. SFT：先让模型会回答问题
2. 奖励模型：让人类对回答排序，训练一个预测"哪个回答更好"的模型
3. PPO：用奖励模型当评分器，强化学习优化生成策略

DPO 的关键洞察是数学上的：在 Bradley-Terry 偏好模型 + RLHF 的目标函数下，
最优策略有闭式解，可以反解出奖励函数 = 当前策略与参考策略的 log-prob 之差
（乘上 β）。把这个奖励函数代回偏好损失，奖励模型就被"消掉"了——只剩
`chosen` 和 `rejected` 在当前策略与参考策略下的 log-prob。

所以 DPO 只需要偏好数据对 `(prompt, chosen, rejected)`，一步直接优化策略，
不需要训练奖励模型、不需要 PPO 采样。参考策略 `ref` 是**训练前冻结**的模型，
作为"锚点"防止策略跑偏太远、失去语言能力。

</details>

**Q5：** 为什么"先学组件、再组装模型"是复现 minimind 的正确路径？

<details>
<summary>💡 提示</summary>

minimind 并不是一个全新的架构，它只是把 Part 6 的"教学版 Transformer"换上了
六个更现代的零件：

| Part 6（教学版） | Part 7 / minimind（现代版） |
|---|---|
| 字符级 tokenizer（65 vocab） | BPE tokenizer（6400 vocab） |
| LayerNorm | RMSNorm |
| learned position embedding | RoPE |
| 标准 MHA | GQA（8Q/4KV） |
| ReLU FFN | SwiGLU |
| 简单训练循环 | 预训练→SFT→DPO 流水线 |

每个组件都是独立、可单独验证的数学小块（本作业就是逐个实现 + 验证）。
理解每一块之后，把它们组装成 `MiniMindBlock`（pre-norm 残差 + GQA + SwiGLU），
再叠 8 层，就是一个能真正训练、生成文本的 mini-LLM。
从"理解组件"到"组装系统"，正是从"看懂教程"到"掌握工程"的跨越。

</details>

---

*Good luck! 🚀 完成作业 7 后，你就拥有了从零构建现代 LLM 的全部零件。*
