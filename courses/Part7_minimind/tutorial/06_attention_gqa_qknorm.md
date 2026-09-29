# 06 — 注意力的两次手术：MHA → GQA → QK-Norm（v4 续）

> 🧭 上一章解决了"位置从哪来"（RoPE），注意力还剩两笔账：**推理时 KV Cache 的显存账**
> （GQA 来还）和**训练中 q/k 范数漂移的稳定账**（QK-Norm 来还）。本章配套脚本
> [06_gqa_qknorm.py](../scripts/06_gqa_qknorm.py)——它是 [05_rope.py](../scripts/05_rope.py) 的叠加生长版：全部 05 章
> 代码 + 本章新零件（RMSNorm 类 + GQA/QK-Norm 开关。RMSNorm：只除均方根、不减均值的自归一化——本章只当 q/k 的"尺度钉"用，数学与手算在 08 章）。两笔账的动机完全不同：一笔为推理
> 省钱，一笔为训练保命。都掰开到公式和代码级。

## 🎒 前置回忆包（不翻旧章也能读）

- **自回归生成**：喂序列 → 取最后位置的打分 → 采样新 token → 拼回 → 重复（02 章 `generate`）。
- **MHA**：8 个 Q 头，每个头有**自己独立**的 K/V 头（02-05 章脚本的 Attention 一直如此）。
- **RoPE**（05 章）：q/k 在打分前被旋转——本章的 QK-Norm 要插在旋转**之前**，顺序有讲究。
- **实验纪律**：同 seed、同数据、只动一个开关（05 章 §7）。

## 📍 你现在的位置：脚本 diff 视角

```text
05_rope.py
  └+ 06_gqa_qknorm.py（本章）
       ├── 【新增】class RMSNorm                    ← §3.2（q/k 的"尺度钉"）
       ├── 【修改】Attention.__init__：+attn/qk_norm 两个开关，K/V 投影 ÷4   ← §3.1
       ├── 【修改】Attention.forward：+qk_norm 分支（先 norm 再 RoPE）        ← §3.3
       ├── 【修改】Attention.forward：+repeat_kv（2 组 K/V 复制成 8 组）      ← §3.4
       └── 【新增】gqa_vs_mha / qk_norm_experiment（实验②③）                ← §4/§5

v4a RoPE 就位 ──本章──▶ v4：现代注意力（GQA + QK-Norm 补完）
```

## §1 先算账：KV Cache 是推理系统的第一性成本

### 1.1 为什么生成需要 Cache

生成第 t 个 token 时，注意力要拿新 q 去和**前 t−1 个 token 的 k** 做点积。若每步整段重算
前向：第 t 步的计算量 ∝ t，全程 Σt ∝ **T²**。观察：前 t−1 个 token 的 k、v 在上一步就算过
且不会变（因果注意力，历史不动）——把它们**存下来复用**，每步只算新 token 的 q/k/v：全程
∝ **T**。这就是 KV Cache，O(T²) 到 O(T) 的免费午餐——**直到你看显存账单**。

### 1.2 显存账：从零推导

Cache 里存什么：**每一层、每一个历史位置、每一个 K/V 头、一个 head_dim 维向量**（K 和 V 各
一份）。fp16 每数 2 字节，公式和代数：

```text
KV Cache 显存 = 2(K,V) × L层 × T位置 × n_kv头 × d_head × 2字节

代 26M 官方配置（L=8, n_kv=8, d_head=64, T=2048, fp16）：
  每层  = 2 × 2048 × 8 × 64 × 2 = 4.19 MB
  全模型 = 4.19 MB × 8 层       = 33.6 MB
```

对照模型本体：25.83M 参数 × 2 字节 = **51.7 MB 权重**。也就是说：

- seq=2048 时，**KV Cache = 权重的 65%**；
- batch=8（推理系统常态）：268 MB = **权重的 5.2 倍**——账单上最大的一项不是模型，是缓存；
- T=32768（05 章 θ 深挖里那个长度）：再 ×16 → 每条请求的缓存 ≈537MB ≈ 权重的 **10 倍**
  （65%×16=1040%）；batch=8 时 ≈**83 倍**——长上下文推理的账单从此起飞。

🔑 **"KV Cache 比模型还大"是推理系统的第一性事实**。模型练得再小，缓存不瘦下来，吞吐和
并发就上不去。减肥的三个方向——**减头数**（GQA，本章）、**减精度**（KV 量化）、**减窗口**
（sliding window / 稀疏注意力）——本章做第一个，后两个在
[Part 14 vLLM](../../Part14_inference_vllm/tutorial/README.md)。

## §2 MHA → MQA → GQA：头数的光谱

公式里唯一好动的是 `n_kv`。三个刻度：

```text
MHA   n_kv = 8   （= n_q）   cache 33.6 MB   质量基线
GQA   n_kv = 2               cache ÷4        质量损失≈0（本节实验②）
MQA   n_kv = 1               cache ÷8        质量损失明显（大模型实测）
        ↑ 8 个 Q 头共享 K/V 的分组数
```

- **MQA**（multi-query，2020）：8 个 Q 头全部共享 1 组 K/V。缓存最省，但所有头被迫"看同一份
  索引"，多样性损失大——GQA 论文在大模型上实测质量下降明显。
- **GQA**（grouped-query，2023）：折中——8 个 Q 头分 **2 组**，组内共享。它还有一个优雅的
  解释：**MHA 和 MQA 是 GQA 的两个端点**（n_kv=8 即 MHA，n_kv=1 即 MQA），n_kv 是一条可以
  连续调节的"质量-成本"旋钮。
- 为什么 K/V 头之间有冗余可压？不同 Q 头的"查询视角"固然不同，但"被检索内容的索引方式"
  高度相关——8 份独立 K/V 里大量信息重复。GQA 论文的消融显示这个冗余在小、中、大模型上
  普遍存在。

## §3 打开脚本：Attention 在 [06_gqa_qknorm.py](../scripts/06_gqa_qknorm.py) 里长出了什么

### 3.0 模型类生长日志（叠加阅读的账本）

| 章 | Attention 的形态 |
|---|---|
| 01-03 | MHA（8Q/8KV），无位置编码分支 |
| 05 | + pos 开关：'learned' 查表 / 'rope' 旋转 |
| **06（本章）** | **+ attn 开关（8Q/2KV）+ qk_norm 开关 + repeat_kv** |
| 08 | 被装进 25.83M 完全体（对账） |

### 3.1 变化①②：构造期——n_kv 开关与 K/V 投影 ÷4

```python
def __init__(self, hidden, n_heads, pos='learned', attn='mha', qk_norm=False,
             max_pos=MAX_POS, theta=THETA):                 # 05 章的 pos 参数原样保留
    ...
    self.n_heads = n_heads
    self.n_kv = n_heads if attn == 'mha' else max(1, n_heads // 4)
    #            ↑ MHA: n_kv=8（每 Q 头独享）  GQA: n_kv=2（4 个 Q 头共享 1 组）
    self.q_proj = nn.Linear(hidden, self.n_heads * self.hd, bias=False)   # Q 不变
    self.k_proj = nn.Linear(hidden, self.n_kv * self.hd, bias=False)      # K/V 投影输出 ÷4
    self.v_proj = nn.Linear(hidden, self.n_kv * self.hd, bias=False)      # ——参数省在这里
    self.o_proj = nn.Linear(hidden, self.n_heads * self.hd, bias=False)
```

K/V 投影的输出维度从 8×64 降到 2×64：单层 K+V 参数 0.52M → 0.13M（实验②实测）。

**📐 diff 对照：与 [Part 6 脚本 05](../../Part6_transformer/scripts/05_multihead_feedforward.py) 逐行比对，MHA → GQA 只改 3 处。**
Part 6 的 MHA 是"每个头一套独立投影"（`Head` 模块 × 8 再拼接，`self.key/query/value`）；
本课 02 章起换成融合投影写法（一根 `Linear` 一次性产出所有头）。抛开写法差异，GQA 本身的
改动浓缩在 3 处（左：Part 6 脚本 05；右：[06_gqa_qknorm.py](../scripts/06_gqa_qknorm.py)）：

```diff
  class Attention:                                    # 左 Part6·05  → 右 06_gqa_qknorm
      def __init__(self, ...):
          self.n_heads = n_heads
+         self.n_kv = n_heads if attn == 'mha' else max(1, n_heads // 4)   # ① KV 头数：8 → 2
-         self.key   = nn.Linear(n_embd, head_size, bias=False)           # ② K/V 投影输出
-         self.value = nn.Linear(n_embd, head_size, bias=False)           #    从 n_heads×hd
+         self.k_proj = nn.Linear(hidden, self.n_kv * self.hd, bias=False)  #   缩到 n_kv×hd
+         self.v_proj = nn.Linear(hidden, self.n_kv * self.hd, bias=False)

      def forward(self, x):
          ...
-         k = self.key(x).view(B, T, self.n_heads, head_dim)              # ③ 用之前把 2 组
-         v = self.value(x).view(B, T, self.n_heads, head_dim)            #    K/V 广播回 8 组
+         k = self.k_proj(x).view(B, T, self.n_kv, self.hd)
+         v = self.v_proj(x).view(B, T, self.n_kv, self.hd)
+         rep = self.n_heads // self.n_kv
+         k = k[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_heads, self.hd)
+         v = v[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_heads, self.hd)
```

其余一行不动——**Q 投影、softmax、加权求和全部照旧**。①定义 KV 头数、②把 K/V 投影的输出
宽度从 `n_heads × hd` 缩到 `n_kv × hd`（参数省在这里）、③用前把 2 组 K/V 广播回 8 组
（§3.4 的 expand/reshape）。这就是"换零件不改骨架"的又一次演示：从 Part 6 的 MHA 到现代
GQA，注意力数学一分没变，变的只是 K/V 的"份数"。

### 3.2 变化③：新类 RMSNorm——q/k 的"尺度钉"

```python
class RMSNorm(nn.Module):
    """本章先当 q/k 的稳定器用；08 章把全模型的 LayerNorm 都换成它。"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))          # 只有缩放增益，没有 bias

    def forward(self, x):
        return (self.weight * x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        #            ↑ fp32 里算再转回原 dtype——bf16 下直接算会丢精度，
        #              一行代码里的混合精度意识（08 章展开）
```

为什么它治"范数漂移"，看 §5 的五步因果链。

### 3.3 变化④：forward 开头的 qk_norm 分支——顺序有讲究

```python
q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
k = self.k_proj(x).view(B, T, self.n_kv, self.hd)
v = self.v_proj(x).view(B, T, self.n_kv, self.hd)
if self.qk_norm_on:
    q, k = self.q_norm(q), self.k_norm(k)          # ← 插在 RoPE 之前，原因见下
if self.pos == 'rope':
    ...                                            # 05 章的旋转分支原样在后
```

**为什么只 norm q/k、不 norm v？** 打分的尺度只由 q·k 决定；v 是**被加权的内容**，它的幅度
本身就是信息（"这个位置的内容有多强"）。norm v 等于强行抹平内容能量——治打分的药，不该
灌进内容里。而且 v 后面还有 o_proj 投影与每块 pre-norm 的归一化兜底尺度。

**为什么先 norm、再 RoPE？（把机理拆开，不背口诀）**

- 数学上，RMSNorm 的**纯缩放部分** `x/rms(x)` 是标量乘法，与旋转**可交换**（标量缩放与正交
  变换互换次序结果不变）；真正不可交换的是 RMSNorm 的**逐维可学习增益 g**：
  `g ⊙ R(x) ≠ R(g ⊙ x)`（⊙ 是逐维的，R 会混合维度对内的两个坐标）。所以顺序确实影响行为，
  但影响的精确位置在"g 与旋转坐标系耦合"上。
- **工程约定 norm→RoPE 的三条理由**：① 语义清晰——先把 q/k 尺度钉回 1，再注入位置；位置是
  最后写入的信息，不被后续算子再扰动；② 若 RoPE 在前、norm 在后，增益 g 的每一维对着
  "转过角度后的坐标"学习，可学习参数与位置函数耦合，训练动态变差；③ 数值稳定：旋转的
  角度表按单位尺度向量设计，先归一化让假设成立。
- 📝 一次"教科书口诀 → 机理"的示范：常见的说法是"归一化会破坏旋转的模长不变性"，不精确
  ——纯缩放与旋转其实可交换。**能推导就不要背口诀**。

### 3.4 变化⑤：repeat_kv——把 2 组复制成 8 组

GQA 的 K/V 头数和 Q 头数不同，注意力矩阵要 (B, n_q, T, T)——广播成对的关键一步：

```python
rep = self.n_heads // self.n_kv                       # 8 // 2 = 4：每组被 4 个 Q 头共享
if rep > 1:
    k = k[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_heads, self.hd)
    #    ↑ 先在中插一维（视图，零拷贝）→ expand 沿新维"广播"（仍是视图，不复制内存）
    #      → reshape 摊平成 (B,T,8,64)：此时才物化成真拷贝
    v = v[:, :, :, None, :].expand(...).reshape(...)  # v 同样处理
```

expand 后的形状是 (B, T, 2, 4, 64)——"2 组、每组 4 份"；reshape 摊平成 (B, T, 8, 64)——
"8 个头"，其中头 0-3 共享组 0、头 4-7 共享组 1。**expand 是免费的（引用），reshape 才真正
复制**——如果直接 `k.repeat()`，从头就是 8 份内存拷贝，白花推理显存。

📝 工程里这一步常用 `F.scaled_dot_product_attention` 的 `enable_gqa=True` 或 flash-attn 内置
分组，连显式复制都省了——本课保留显式版本，是为了让你看见"共享"发生的确切位置。

### 3.5 SDPA 与 FlashAttention：三行注意力换一行

本课脚本从 02 章到现在，注意力一直是手写三步：`scores = q@kᵀ/√d → 因果遮罩 → softmax → @v`。
PyTorch 2.0 起这三步可以换成**一行 SDPA**（Scaled Dot-Product attention，minimind 官方默认
也用它）：

```python
if self.flash and seq_len > 1:
    output = F.scaled_dot_product_attention(
        xq, xk, xv, is_causal=True)      # 缩放/遮罩/softmax/加权全在内，又快又省显存
else:
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(head_dim)   # 教学版：手写三步
    scores = scores.masked_fill(causal_mask, float('-inf'))      # ——每一步肉眼可查
    output = F.softmax(scores, dim=-1) @ xv
```

- 🔑 **FlashAttention 里面没有新数学**：它只是**按块（tile）计算、不落整张 T×T attention
  矩阵**，把对显存的读写从 O(T²) 降到 O(T)——softmax 用在线归一化（running max/sum）流式
  算出。结果与手写版数值上几乎一致，只是更快、更省显存。
- 📝 **课程脚本为什么不用**：手写版让"缩放、遮罩、softmax"每一步都可指认——QK-Norm 治的
  打分尺度（§5）、因果遮罩的位置、GQA 的广播，全部发生在肉眼可见的三行里。**教学透明与
  工业效率在这里分道**：读懂手写版，再去看工业代码的 SDPA/flash-attn 调用，你知道那一行
  展开后是什么。
- 💡 GQA 场景下新版 PyTorch 还能给 SDPA 传 `enable_gqa=True`，连 §3.4 的显式复制都省掉
  （kernel 内部直接按组读取）。融合内核（fused kernel：把多个算子拼进一个 GPU 内核）的内部世界（SRAM（片上高速缓存）tiling、online softmax）是
  [Part 9 CUDA](../../Part9_cuda_kernels/tutorial/README.md) 的主题。

## §4 实验②：质量几乎无损，缓存省 4 倍

**设置**（`gqa_vs_mha` 函数）：MHA 和 GQA 两个模型（都带 RoPE），同 seed 各短训 300 步 +
纯算术账本。**能用算术讲清的不用训练——先账本后实验，是成本最低的论证顺序**：

```bash
python 06_gqa_qknorm.py    # 实验② + 实验③
```

```text
单层 K+V 投影参数:  MHA 0.52M → GQA 0.13M（÷4）
KV Cache(seq=2048, fp16): MHA 4.2MB/层 → GQA 1.0MB/层（8 层合计 8.4MB）
短训后验证集 ppl: MHA 363.53 vs GQA 352.44 —— GQA 不降反低 11.1 ppl（3.1%），收益全在推理
```

三个读数：① **参数 ÷4**（08 章参数账本的 attention 行从 8.39M → 5.24M 主要就是它）；
② **缓存 ÷4**——§1.2 里 33.6 MB → 8.4 MB，从"权重的 65%"降到"权重的 16%"；③ **质量差
11.1 个 ppl 点（3.1%，GQA 反而更低）**——在 tiny 规模单种子的波动量级内，GQA 质量不劣于
MHA（同步数的 GQA 少了冗余参数，小数据上有时更易优化）。

**两代官方口径**（对照时的常见困惑，值得背下来）：

| 版本 | 口径 | 取舍 |
|---|---|---|
| 26M（minimind2-small） | 8Q/**2KV** | 压得更狠：缓存 ÷4，tiny 模型质量余量足 |
| minimind-3（master） | 8Q/**4KV** | 留质量余量：缓存只 ÷2，参数更大后冗余变小 |

📜 **论文对照**：GQA（Ainslie et al. 2023）在 70B~540B 规模报告——MQA 质量下降明显，GQA-8
与 MHA 几乎持平；这也是 Llama-2 70B 之后几乎所有开源模型（Llama/Qwen/minimind 全线）标配
GQA 的直接原因。我们 26M 上 GQA 反而低 3.1%（单种子波动量级）与论文"几乎无损"方向一致，
但注意规模差 2500 倍，**不能拿小模型的数字外推大模型的收益**。

## §5 实验③：QK-Norm——训练稳定器的"官方补丁"

GQA 是推理经济学；QK-Norm 是训练保险。它治的病从一条**五步因果链**开始，每步都值得看懂：

```text
① 训练中，q、k 向量的范数会漂移（所有权重都在更新，没有任何约束钉住头内尺度）
② 打分 = q·k/√d：范数涨 → 打分尺度涨。实验③模拟"漂移 3 倍"：logits std = 8.85
③ softmax 对尺度极端敏感：两个候选 logit 差 8.85 → 概率比 e^8.85 ≈ 7000 : 1
   → 分布几乎 one-hot（注意力"熵塌缩"：只剩一个位置被看）
④ one-hot 的 softmax 梯度处处趋近 0（饱和区）
⑤ 梯度消失 → 这一层"装死" → 训练不稳 / loss 尖峰
```

🔑 注意力打分的**尺度**本来有设计过的稳定器：除以 √d。但 √d 只对"单位方差输入"成立——它防
的是初始化时的方差失衡，防不住**训练动态中的范数漂移**。QK-Norm 补的正是这一刀。

**实测**（`qk_norm_experiment`，造病实验的运行结果）：

```text
未归一化 logits: std=8.85（logit 差 8.85 → 概率比 e^8.85≈7000:1，≈one-hot）
q/k RMSNorm 后:  std=1.00（尺度钉死——Qwen3/Gemma 系同款）
```

std 从 8.85 到 1.00：概率比从 7000:1 拉回 ~3:1 的健康区间，注意力恢复"该看的都看一点"的
软分布。🏭 **为什么 tiny 模型没这个药也能训**：26M + 短训练 + 裁剪（04 章）下漂移幅度小。
但这正是"工业代码每一行补丁背后都有一次真实事故"的活例子——QK-Norm 是 Qwen3/Gemma2/
OLMo2 的标配，它们加它是因为**在大规模训练中真实地翻过车**。读工业代码的正确姿势：看到
一个"多余"的归一化，先假设它堵过一个你还没遇到的坑。

## §6 🔬 实验是怎么搭的：账本实验与"造病"实验

本章两组实验的设计思路不同，恰好对应两类标准实验形态：

**实验② 是"账本 + 短训"实验**。它的说服力一半来自纯算术（§1.2 的显存公式不依赖任何训练
——参数 ÷4、缓存 ÷4 是恒等式），另一半才是短训对照：MHA/GQA 都带 RoPE、同 seed、同数据、
只差 `attn` 一个开关。**能用算术讲清的不用训练**——先账本后实验，是成本最低的论证顺序。

**实验③ 是"造病"实验**，这是它和实验①②的本质区别：

```python
q = torch.randn(B, T, hd) * 3.0        # ×3：人为注入"范数漂移 3 倍"的病
k = torch.randn(B, T, hd) * 3.0
raw = (q @ k.transpose(-1, -2)) / math.sqrt(hd)    # 病理读数：std=8.85
qn = q / (q.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()   # 给药：只 norm q/k
kn = k / (k.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
normed = (qn @ kn.transpose(-1, -2)) / math.sqrt(hd)       # 药效读数：std=1.00
```

训练中范数漂移是缓慢、随机、难复现的——**等它自然发生再测量，成本高到不可行**。造病实验
的做法：人为注入**已知、可控、可复现**的扰动（×3 恰好把 std 推到 8.85 这个"one-hot 边缘"），
然后验证药物对该病有效。注意它证明的是"药对这种病有效"（QK-Norm 能钉死尺度），**不证明
"训练中真的会出现这种病"**——后者是 Qwen3/Gemma 们用大规模训练事故证明的。两类证据分开
陈述，才是不糊弄的实验叙事。

## 🏭 工业界

① §1 的账单是推理系统（[Part 14 vLLM](../../Part14_inference_vllm/tutorial/README.md)）的
入口题：PagedAttention 解决"缓存怎么管理"，GQA 解决"缓存有多少"——两层一起才构成现代推理
栈。② 注意力"算得快"（FlashAttention 融 softmax 进 SRAM）是
[Part 9 CUDA](../../Part9_cuda_kernels/tutorial/README.md) 的主题。③ 减缓存的另两路——
KV 量化（fp8/int4 cache）与 sliding-window——在 Qwen/Gemma 生产配置里与 GQA **叠加使用**：
GQA 减 4 倍、量化再减 2-4 倍、窗口封顶，三刀下去 seq=32k 的缓存才进得了单卡。

## ▶️ 运行本章成果

```bash
python 06_gqa_qknorm.py    # 实验②= MHA vs GQA（约 2 分钟），实验③= QK-Norm 造病实验
```

## 🪝 引子

1. 注意力的两次手术完成——08 章参数账本里 attention 行从 8.39M 瘦到 5.24M。但注意力本就
   只占参数 20%，**参数大头（57.9%）在 FFN**——下一章在同参数预算下换 FFN 的形状
   （ReLU→SwiGLU），赢的比 GQA 省的多得多。
2. QK-Norm 的 RMSNorm 本体是"LayerNorm 的减法"——08 章把归一化全局换掉并算清省了多少。

## 🎯 面试直通车

<details>
<summary>Q1: GQA 为什么几乎无损？n_kv 怎么选？</summary>
A: 不同 Q 头的"查询视角"不同，但 K/V 的"索引方式"高度冗余——组内共享损失很小。GQA 论文在
70B~540B 上实测 GQA-8≈MHA、MQA 明显掉点；n_kv 是质量-成本连续旋钮（8=MHA、1=MQA）。选法
看规模与预算：tiny（26M）取 8Q/2KV 激进压缩，更大模型取 8Q/4KV 留余量（minimind 两代口径
正是这么分的）。收益全在推理：缓存与 K/V 投影参数同步 ÷(n_q/n_kv)。
</details>

<details>
<summary>Q2: QK-Norm 为什么只 norm q/k？为什么放在 RoPE 之前？</summary>
A: 只 q/k：打分尺度由 q·k 决定，v 是被加权的内容、幅度即信息，norm v 会抹掉内容能量且后续
本有归一化兜底。放 RoPE 前：RMSNorm 的纯缩放部分与旋转可交换，但逐维增益 g ⊙ 不可交换
——先 norm 后 RoPE 让"尺度钉死"与"位置注入"两个动作解耦：位置是最后写入的信息不被增益
扰动，g 的学习也不与旋转坐标耦合。若反序，g 每维对着旋转后的坐标学习，训练动态变差。
</details>

## ✅ 本章验收

- [ ] 能从零推导 KV Cache 显存公式并代出"seq=2048 时缓存=权重的 65%"这笔账
- [ ] `python 06_gqa_qknorm.py` 实验②③：能报出 ppl 363.53/352.44、缓存 4.2→1.0MB、std 8.85→1.00
- [ ] 能对照 §3.0 生长日志说出 06 相对 05 的五处变化，并解释 expand 与 reshape 的分工（视图 vs 物化）
- [ ] 能讲清 QK-Norm 五步因果链，并说出"只 norm q/k"和"norm 在 RoPE 前"的精确理由
- [ ] 能说出 GQA 两代官方口径（26M 8Q/2KV vs minimind-3 8Q/4KV）各自的取舍
- [ ] 能说出"造病实验"证明什么、不证明什么（药对病有效 ≠ 训练中必得这种病）

---

[← 上一章：位置编码进化](05_rope.md) | [课程 README](README.md) | [下一章：FFN 进化 →](07_ffn_moe.md)
