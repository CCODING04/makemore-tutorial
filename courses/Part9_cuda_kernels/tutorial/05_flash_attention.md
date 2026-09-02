# 05 — Flash Attention：亲手写出毕业内核

> 🧭 Part 7 复现 minimind 时我们调用过 `F.scaled_dot_product_attention`，02 章讲 matmul
> 阶梯时也预告过它。这一章把 Part 9 的全部家当——02 章的 tiling、07 章的 Triton
> softmax——组装成 Flash Attention 前向内核，并对照 PyTorch SDPA 的四个后端验收
> 正确性与性能。这是本课程的"毕业内核"。

## 🎯 学习目标

完成本章后，你将能够：

- ✅ **推导** online softmax 的滚动更新公式（m / l / acc 三个状态如何跨块传递）
- ✅ **手写** 带 causal mask 与边界处理的 Flash Attention 前向 Triton 内核（bf16 输入、fp32 累加）
- ✅ **解释** FA1→FA2→FA3→FlexAttention→SageAttention 的演进逻辑，以及为什么 4090 用不了 FA3
- ✅ **验收** 自写内核：与 naive 对照数值、与 SDPA 四后端对照吞吐（≥50% 合格线）

## 📖 前置知识

**必须掌握：**

- **[02 章 matmul 优化阶梯](02_matmul_optimization.md)**：SMEM tiling、算力墙/内存墙——
  Flash Attention 就是"tiling 思想搬到 attention"，L3 一节已给出它的预告
- **[04 章 Triton / 脚本 07](04_triton_and_extensions.md)**：`tl.load` 的 mask 用法、
  "一个 program 管一块数据"的编程模型、softmax 内核里"减最大值防上溢"的技巧
  （[脚本 07](../scripts/07_triton_kernels.py) 的 `softmax_kernel` 是本章的直接前作：
  它对**整行**做 softmax，本章要解决"整行装不下怎么办"）

**建议掌握：**

- **Part 6**：attention 公式 `softmax(QK^T/√d)V`（知道形状怎么变换即可）
- **Part 7**：`F.scaled_dot_product_attention` 的调用位置（[Part7 脚本 05](../../Part7_minimind/scripts/05_full_model.py)
  的 `MiniMindAttention`）、KV Cache 与长上下文的痛点

**可选：**

- **Part 8**：bf16 autocast 的位置（本章内核就是 bf16 输入 + fp32 累加的活例子）

## 一、为什么需要 Flash Attention

### naive 的显存账本

脚本 09 段 1 实测的教科书实现（[scripts/09_flash_attention_triton.py](../scripts/09_flash_attention_triton.py)）：

```python
att = (q @ k.transpose(-2, -1)) * scale   # (B, H, T, T)  <- 第一个巨型中间矩阵
att = att.masked_fill(~keep, -inf)        # causal：又物化一份拷贝
p = att.softmax(dim=-1)                   # (B, H, T, T)  <- 第二个
out = p @ v
```

**实测（RTX 4090，bf16，B=2, H=8, T=4096, D=64；脚本 09 段 4 的计时协议）**：

| 实现 | causal | 耗时 | 附注 |
|---|---|---|---|
| naive | 否 | 3.579 ms | 物化 2 个 (B,H,T,T) bf16 矩阵，各 0.5 GB |
| naive | 是 | 6.198 ms | `masked_fill` 再物化第 3 份，反而更慢 |
| 手写 Triton FA | 是 | **0.279 ms** | 零 (T,T) 中间矩阵 |
| 手写 Triton FA | 否 | **0.436 ms** | |

> 📝 以上为共享 GPU 环境实测（2026-09-02）；独占整卡时手写内核对 SDPA 最优后端的比例
> 会更高（曾实测 105%~163%）——计时基准的"独占 vs 共享"本身就是一个公平性变量（见
> 4.4 的陷阱 4）。

naive 的三个痛点：

1. **显存 O(T²)**：每个 (B,H,T,T) 矩阵在 T=4096 时 0.5 GB；T 翻倍翻 4 倍。
   T=8K 时 naive 光中间矩阵就要 ~2 GB × 3，T=32K 直接爆卡。
2. **HBM 往返**：这些矩阵写回显存再读回来——02 章的语言：**内存墙**。attention
   的计算本身该是 compute-bound，naive 实现却被 O(T²) 的访存拖住。
3. **训练更惨**：backward 需要 softmax 的结果 P，还得把 (T,T) 存着（或重算）。

> 💡 **类比**：算 100 万个数的平均值，你不会先把 100 万个数全抄到一张超大的纸上
> 再求和——你维护一个"到目前为止的和"滚动更新。Flash Attention 对 softmax 做的
> 就是这件事，只是多了一个"到目前为止的最大值"来保证数值稳定（online softmax）。

> 🔑 **Flash Attention 一句话**：把 Q/K/V 切块（tiling）搬进片上 SRAM，用 online
> softmax 让"整行信息"也能分块流式计算——**从不物化 (T,T) 矩阵**，显存 O(T²)→O(T)，
> HBM 读写减少一个数量级。注意它是**精确**算法，不是近似（与稀疏/线性 attention 相对）。

### 演进史：我们要抄的是 FA2 的作业

| 年份 | 工作 | 论文 | 关键改进 | 4090 (SM89) 能用吗 |
|---|---|---|---|---|
| 2022 | FlashAttention (FA1) | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | tiling + online softmax，首次系统做 **IO-aware** 的精确 attention | ✅ |
| 2023 | FlashAttention-2 (FA2) | [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) | 减少非 matmul FLOPs、更优的并行与 warp 分工（约 2× FA1） | ✅（torch SDPA 的 flash 后端即此类实现） |
| 2024 | FlashAttention-3 (FA3) | [arXiv 2407.08608](https://arxiv.org/abs/2407.08608) | **WGMMA** + 异步流水 + FP8，Hopper 专属 | ❌ WGMMA 是 **SM90** 指令，4090 是 SM89 |
| 2024-25 | FlexAttention | [PyTorch 官方博客](https://pytorch.org/blog/flexattention/) | mask_mod/score_mod → torch.compile 生成 Triton 内核 | ✅（本章段 5 实测） |
| 2024-25 | SageAttention(2) | [arXiv 2411.10958](https://arxiv.org/abs/2411.10958) | QK^T 用 INT8、PV 用 **INT4** 量化，4090 上 ~3× FA2 | ✅（近似，推理向） |

> ⚠️ **FA3 与 4090**：FA3 的速度来自 Hopper 的 warp 级矩阵指令 **WGMMA**（SM90a 专属）
> 与 TMA/异步执行。4090（Ada，SM89）只有 `mma` 路径，编译都过不了。所以在我们的卡上，
> "工业天花板"就是 FA2 类实现（SDPA flash/cudnn 后端）+ 量化路线（SageAttention）。
> 另外本章内核沿用的在线 softmax 收敛形式最早由 Rabe & Staats（[arXiv 2112.05682](https://arxiv.org/abs/2112.05682)）给出。

## 二、online softmax：逐步推导

### 2.1 从"必须看整行"说起

数值稳定的 softmax（Part 1 / 07 章的老朋友）：

```
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)，  m = max_j x_j
```

07 章的 softmax 内核能把**整行**一次性 `tl.load` 进片上——attention 的行却是
T 个 key 的打分，T=4096 时一行 fp32 有 16 KB，16 个 (b,h) × 多个行块根本铺不开，
更别说 (T,T) 全矩阵。**分块计算softmax 的障碍**：归一化分母需要**整行**的 exp 和。

### 2.2 两块合并：alpha 从哪来

把 key 序列切成两块，先算第 1 块：

```
m₁ = max(x[块1])， l₁ = Σ_{j∈块1} exp(x_j - m₁)，  o₁ = Σ_{j∈块1} exp(x_j - m₁)·v_j
```

第 2 块到来时，新的全局最大值 `m_new = max(m₁, m₂)`。关键观察——分母可以**重缩放**
而不是重算：

```
l_new = Σ_{j∈块1} exp(x_j - m_new) + Σ_{j∈块2} exp(x_j - m_new)
      = Σ_{j∈块1} exp(x_j - m₁)·exp(m₁ - m_new) + l₂
      = l₁·α + l₂                    其中 α = exp(m₁ - m_new)
```

输出的分子部分同理乘 α：`o_new = o₁·α + o₂`。最终 `out = o_new / l_new`——因为
softmax 的分子分母同乘 `exp(-m_new)`，结果不变（这就是"减最大值"技巧的推广：
最大值可以**事后修正**）。

> 🔑 **online softmax 三状态**：行最大值 `m`、分母滚动和 `l`、输出累加 `acc`。
> 每来一个新块：更新 m → 算 α 把历史 l/acc 折算到新基准 → 累加新块。逐块进行，
> 整行从不完整出现。数学细节见 FA1 论文 §3.1（算法 1）。

### 2.3 exp2 技巧：还差一个工程优化

GPU 上 `exp2`（2^x）比 `exp`（e^x）快一个量级（exp 通常是 exp2 加乘法的宏展开）。
利用 `exp(x) = 2^(x·log2e)`，把换底系数**提前折叠进 scale**：

```python
qk_scale = sm_scale * 1.44269504            # scale × log2(e)，只乘一次
qk_scaled = qk * qk_scale                   # 此后所有分数都在"log2 域"
p     = tl.math.exp2(qk_scaled - m_new)     # m_new 也是 log2 域的最大值，直接相减
alpha = tl.math.exp2(m_i - m_new)           # 同域相减，无需再乘系数
```

这就是内核里那 5 行核心的来历（下一节逐行讲）。官方 Triton tutorial 06 与
t-vi 的 GPU MODE 讲座都用这个技巧。

## 三、内核实现（[scripts/09_flash_attention_triton.py](../scripts/09_flash_attention_triton.py)）

### 3.1 数据流与形状

一个 program 负责 O 的一块（BLOCK_M 行），网格 `(cdiv(T, BLOCK_M), B*H)`：

```
q 块 (BLOCK_M, D)  ←—— 一次载入，全程驻留寄存器/SMEM
        │
        ▼  对每个 K/V 块 (BLOCK_N 列)：
kT (D, BLOCK_N) ──tl.dot──► qk (BLOCK_M, BLOCK_N) fp32
                               │  ×qk_scale(已含log2e) + mask(-1.0e6)
                               ▼
                    m_new = max(m, 行最大)   ──► α = exp2(m - m_new)
                    p = exp2(qk_scaled - m_new)          (BLOCK_M, BLOCK_N)
                               │ .to(bf16)
v (BLOCK_N, D) ──tl.dot──► acc = acc·α + p @ v           (BLOCK_M, D) fp32
                    l = l·α + Σp                          (BLOCK_M,)
        │  所有块扫完
        ▼
out = acc / l  ──►  写回 O 的这一块 (BLOCK_M, D) bf16
```

状态 `m / l / acc` 全程 **fp32**；进 `tl.dot` 的 `qk`、`p`、`v` 是 **bf16**（Tensor
Core 要求低精度输入 + fp32 累加，这正是 04 章"bf16 autocast 几乎不掉点"的硬件根基）。

### 3.2 内核核心逐行解释

```python
# ---- online softmax 的 5 行核心（_fa_fwd_inner 内）----
m_new = tl.maximum(m_i, tl.max(qk_scaled, 1))   # ① 行最大值（fp32），跨块修正基准
p     = tl.math.exp2(qk_scaled - m_new[:, None])# ② 稳定 softmax 分子（未归一，log2 域）
alpha = tl.math.exp2(m_i - m_new)               # ③ 旧累加和的缩放因子（见 2.2 推导）
l_i   = l_i * alpha + tl.sum(p, 1)              # ④ 分母滚动和（fp32）
acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # ⑤ 输出滚动累加
m_i   = m_new
```

- ① `tl.max(qk_scaled, 1)`：沿 key 维（axis=1）归约出每行最大值；与历史 `m_i`
  取 max 得新基准。首次迭代 `m_i = -inf`，`α = exp2(-inf - m_new) = 0`，
  恰好把空历史清零——初始化不需要特判。
- ② 分子只算**这一块**的，用 exp2（换底系数已折叠进 `qk_scale`）。减 `m_new`
  保证最大值为 0，`exp2(0)=1`，不溢出。
- ③ 历史 `l/acc` 是按旧基准 `m_i` 算的，换新基准要乘 `α = exp2(m_i - m_new)`。
  注意 `α ≤ 1`（m 只增不减），且多数块 `m_new == m_i` 时 `α = 1` 零损耗。
- ⑤ `p.to(v.dtype)`：`p` 是 fp32，必须降回 bf16 才能和 `v` 一起进 Tensor Core；
  `acc` 的 fp32 精度由 `tl.dot` 的累加器保证。官方版把 `acc·α` 合并进
  `tl.dot(p, v, acc)` 的三参数形式，省一条指令，教学版拆开写更清楚。

### 3.3 causal 三阶段分解（对齐官方 tutorial 06）

对第 `start_m` 个 query 行块，K/V 轴分成三段：

```
key 下标 ──►
0        diag_lo      行块末尾      T
├──────────┤────────────┤──────────┤
│ 阶段1    │ 阶段2       │ 阶段3
│ 带外块   │ 对角块      │ 全 mask 块
│ 无 mask  │ 逐元素 mask │ 直接跳过
└──────────┴────────────┴──────────┘
             ▲ query 行块的因果边界 (start_m+1)*BLOCK_M
```

- **阶段 1（带外块）**：key 列号全部 `< ` 行块起点，因果条件必然满足——连
  mask 判断都省掉，走最快路径（`MASK_MODE=0`，连边界 mask 都不需要：列必在界内）。
- **阶段 2（对角块）**：块内部分元素越界（列 > 行），需要 `offs_m >= offs_kv`
  的逐元素 mask。
- **阶段 3**：整块全被 mask 掉，**根本不进循环**——causal 省 ~一半计算量的来源，
  也是 4K 序列上 causal 比 full 快 ~1.6×（0.279 vs 0.436 ms）的原因。

脚本 09 段 2 打印的实际分解（本次运行 autotune 选中 BLOCK_M=128, BLOCK_N=64；示例块号
经钳制取"倒数第二块"，保证任何 BLOCK 组合下都指向真实存在的行）：

```
[autotune] T=1024 选中 BLOCK_M=128 BLOCK_N=64 num_warps=8 num_stages=3（16 个组合实测选出）
[causal 三阶段] 以第 6 个 query 块（行 768..896）为例：
               带外块 [0, 768) 无 mask | 对角块 [768, 896) 逐元素 mask | [896, 1024) 直接跳过
```

> ⚠️ **对角带起点必须对齐到 BLOCK_N**：脚本里 `diag_lo = (start_m*BLOCK_M) // BLOCK_N * BLOCK_N`
> （向下取整）。若直接用 `start_m*BLOCK_M`，当 BLOCK_N > BLOCK_M（如 128 > 64）时，
> 阶段 1 的最后一个块会**越界扫进对角带**，把对角元素算两遍——这正是"误差集中在
> 特定行"的一类 bug（陷阱 3 详述定位法）。官方 tutorial 的 autotune 网格里
> BLOCK_N ≤ BLOCK_M（截至本文核对的版本），隐式避开了这个问题；我们的网格允许 (64, 128)
> 组合，必须显式对齐。

### 3.4 边界处理：`other=0` 与 `-1.0e6`

T 不是 BLOCK 的整数倍时（脚本段 3 用 T=1000 专测）：

```python
kv_ok = offs_kv < N_CTX                                        # 列越界判断
kT = tl.load(..., mask=kv_ok[None, :], other=0.0)              # 越界填 0：进 dot 无害
qk_scaled = qk * qk_scale + tl.where(valid, 0.0, -1.0e6)       # 但分数必须挡成"负无穷"
```

两步缺一不可：`other=0` 只保证**加载**不越界、dot 不出垃圾；但 0 分数经过
`exp2(0 - m)` 会变成正权重混进 `l` 和 `acc`，所以还要用 `tl.where` 把非法位置的
**分数**压到 `-1.0e6`，`exp2(-1e6 - m)` 下溢为精确的 0。

> ⚠️ **为什么不用 `-inf`**：全非法的行（如 padding 行）会得到 `max = -inf`，
> 接着 `-inf - (-inf) = NaN`，一次污染整块。`-1.0e6` 足够小（exp2 后精确下溢到 0）
> 又不会触发 `inf - inf`。这是社区踩了多年的坑（陷阱 1 详述）。

### 3.5 autotune：02 章那行"Autotuning"落到实处

```python
fa_configs = [triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_warps=w, num_stages=s)
              for BM in [64, 128] for BN in [64, 128] for w in [4, 8] for s in [2, 3]]
@triton.autotune(configs=fa_configs, key=['N_CTX', 'HEAD_DIM'])
```

16 个组合，首次遇到新 `(N_CTX, HEAD_DIM)` 时逐一实测选优。本次运行 T=1024 选中
`BLOCK_M=128 BLOCK_N=64 num_warps=8 num_stages=3`（autotune 的选择随 GPU 状态/时序
波动，另一次运行选中过 64/128/4/2——这是正常的）。`num_stages` 是 K/V 加载的
软件流水深度（02 章"double buffering"一行的自动版）。

### 3.6 包装函数

```python
def fa_forward(q, k, v, causal=True):
    """q/k/v: (B, H, T, D) bf16 连续 → 输出同形状 bf16（内部 m/l/acc 全 fp32）。"""
    ...
    stage = 3 if causal else 1     # 与官方 tutorial 06 相同的阶段编码
    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_M']), B * H)
    _fa_fwd[grid](q, k, v, o, sm_scale, T, HEAD_DIM=D, STAGE=stage)
```

教学版做了两个简化（工业版都支持）：要求连续布局（工业版传 16 个 stride 任意支持）；
只做前向且不写 logsumexp（backward 需要它，见官方 tutorial 06 的 `_attn_bwd`）。

## 四、实测（RTX 4090 / torch 2.6.0+cu124 / triton 3.2.0；2026-09-02，GPU 与其他任务共享）

### 4.1 SDPA 四后端：锁定并打印实际命中者

不假设、用 profiler 抓 kernel 名作证据（段 1 输出）：

```
锁定后端        实际命中（profiler 证据 kernel）
------------------------------------------------------------------
OK flash     -> flash     | void pytorch_flash::flash_fwd_kernel<...
OK efficient -> efficient | fmha_cutlassF_bf16_aligned_64x64_rf_sm80(...
OK cudnn     -> cudnn     | cudnn_generated_fort_native_sdpa_sm80_knob_6_...
OK math      -> math      | void at::native::elementwise_kernel<128, 2, ...

[默认调度（不锁定）] 命中 flash | void pytorch_flash::flash_fwd_kernel<...
```

- bf16 + D=64 + causal 下，4090 上**四个后端全部可用**，默认调度命中 flash。
- 判别特征：kernel 名里的 `flash` / `fmha` / `cudnn` / (`gemm`+`softmax` = math 拆成
  的多个 eager kernel)。`No available kernel` 的 RuntimeError 则表示锁不住（比如
  fp32 输入锁 flash 就会失败——可以自己试试）。

### 4.2 数值验收（段 3 输出）

与 naive fp32 参考对照，`rtol=atol=1e-2`（与官方 tutorial 一致）：

```
[T=1024 causal=True ] assert_close PASS | max|Δ| 7.70e-03 | 相对误差(|ref|>=0.1 处) 1.6e-02 (SDPA flash 同口径 1.6e-02)
[T=1024 causal=False] assert_close PASS | max|Δ| 9.30e-04 | 相对误差(|ref|>=0.1 处) 6.6e-03 (SDPA flash 同口径 6.2e-03)
[T=1000  causal=True ] assert_close PASS | max|Δ| 8.18e-03 | 相对误差(|ref|>=0.1 处) 1.4e-02 (SDPA flash 同口径 1.4e-02)
[T=1000  causal=False] assert_close PASS | max|Δ| 1.06e-03 | 相对误差(|ref|>=0.1 处) 6.2e-03 (SDPA flash 同口径 5.4e-03)
[泄漏检查] 改动 v[j>i]: 行 <=i 输出最大变化 = 0.00e+00 (应=0)，行 >i 最大变化 = 0.57 (应>0) -> 无泄漏 OK
```

> 📝 **误差从哪来**：T=1000（非 64 的倍数）专测边界 mask，与 T=1024 同样通过。
> 相对误差与 **SDPA flash 完全同一量级**——它由 bf16 输入量化（尾数仅 8 位，~0.4%
> 起步）+ exp2/scale 折叠顺序主导，属于预期；"逐元素相对误差"要在 |ref|≥0.1 的
> 区间算才有意义（|ref|→0 处分母失真，SDPA 也一样大，那部分靠 atol 兜住）。
> 泄漏检查：改动未来位置的 v，历史行输出逐位不变——causal 语义正确。

### 4.3 性能（段 4 输出，预热 10 + 测 50，`torch.cuda.Event`）

```
T= 1024 causal | naive   0.185 ms | triton  0.031 ms ( 69.3 TF) | best flash  0.031 ms ( 69.0 TF) -> 100.5% 优秀
T= 1024 full   | naive   0.158 ms | triton  0.033 ms (131.9 TF) | best cudnn  0.031 ms (139.4 TF) ->  94.6% 优秀
T= 2048 causal | naive   1.807 ms | triton  0.086 ms (100.4 TF) | best flash  0.109 ms ( 79.0 TF) -> 127.1% 优秀
T= 2048 full   | naive   1.041 ms | triton  0.113 ms (151.7 TF) | best cudnn  0.108 ms (159.5 TF) ->  95.1% 优秀
T= 4096 causal | naive   6.198 ms | triton  0.279 ms (123.3 TF) | best cudnn  0.308 ms (111.6 TF) -> 110.5% 优秀
T= 4096 full   | naive   3.579 ms | triton  0.436 ms (157.6 TF) | best cudnn  0.414 ms (165.9 TF) ->  95.0% 优秀

[总评] 最慢场景 94.6% of SDPA 最优后端 -> 优秀 (>85%)
```

验收承诺（写进脚本输出）：**教学版前向吞吐 ≥ SDPA 最优后端的 50% 合格、> 85% 优秀**。
依据：PyTorch 官方 FlexAttention 博客实测 Triton 路径达 FA2 前向的 90%（A100）——
这是"通用 Triton 内核"离手工调优 CUTLASS 内核的距离上限，教学版再让一档到 85%。

> 📝 **教学版为什么能反超（诚实解读）**：① 我们只做前向，不物化 backward 需要的
> logsumexp（SDPA 每次前向都要写它）；② autotune 恰好在被测的 (N_CTX, HEAD_DIM)
> 上选优；③ B=2/H=8/D=64 的"小"形状下，SDPA 通用内核的固定开销占比大。换 D=128、
> 大 batch、或加上 backward，FA2 类实现会重新拉开——**看量级，别抠个位数**。
> 测量条件：本节数字为共享 GPU 环境实测（另一次独占运行曾测得 105-163%——独占时手写内核
> 比例更高）。比较内核快慢时，同卡同负载才有可比性；这正是陷阱 4 的核心。

📊 一张表看懂趋势：序列越长，naive 的 O(T²) 越痛（6.4 ms vs 0.31 ms，20.6×），
而 Triton 版 4K 时到 110-138 TFLOPS——attention 从"被内存墙拖死的 matmul 链"
变回了接近 compute-bound 的内核。这就是 02 章"迁移"学习目标的完整闭环。

## 五、常见陷阱（症状 → 原因 → 解法）

### 陷阱 1：mask 用 `-inf`，输出 NaN

**症状**：输出含 NaN，或 `max|Δ|` 爆炸；往往集中在 padding 行 / 首块。

**原因**：`qk_scaled = tl.where(valid, qk, -inf)` 后，若某行**所有**位置都被 mask
（padding 行、或 bug 导致 mask 写反），`m_new = max(-inf, -inf) = -inf`，随后
`p = exp2(-inf - (-inf)) = exp2(NaN) = NaN`，`α` 同理，一次污染整块。

**解法**：用足够大的负数 `-1.0e6`（fp32 下 `exp2(-1e6)` 精确下溢到 0），见 3.4 节。
真正合法的行永远至少有一个有效位置（自身），不会触发全 mask；padding 行的垃圾
结果靠 store 的行 mask 丢弃。

### 陷阱 2：用 bf16 存 softmax 状态 / 累加器

**症状**：长序列下误差显著大于 SDPA（>1e-1 量级），短序列却正常。

**原因**：`l` 是上千个 ≤1 的数相加，`acc` 是 T 次 dot 累加——bf16 只有 8 位尾数，
每次累加丢 ~0.4%，误差随块数线性增长。Tensor Core 的设计用法就是**低精度输入 +
fp32 累加**（`tl.dot` 输入 bf16、累加器 fp32 是免费的，反而更快）。

**解法**：`m_i / l_i / acc` 一律 `tl.zeros(..., dtype=tl.float32)`；只在进 `tl.dot`
前把 `p` 降回 bf16（`p.to(v.dtype)`），写出结果前再转输出 dtype。

### 陷阱 3：误差集中在特定行 —— causal 块边界 bug 的定位法

**症状**：整体 `assert_close` 勉强过，但逐行看误差**扎堆在某些固定行号**。

**定位法**（先量、再猜，呼应 03 章的 profiling 思维）：

```python
err = (tri.float() - ref).abs().amax(dim=-1)   # (B, H, T) 每行最大误差
worst = err.flatten(-2).argmax(-1)             # 最差行号
print(worst)                                    # 看它是否落在 BLOCK_M 的倍数附近
```

- 误差集中在 `BLOCK_M` 的倍数附近 → 块边界 bug：查 `diag_lo` 是否按 BLOCK_N
  对齐（3.3 节的 `// BLOCK_N * BLOCK_N`）、对角 mask 的比较方向（`>=` 写成 `>`）。
- 误差均匀分布 → 数值路径问题（dtype、累加精度、scale 折叠），走陷阱 2 排查。
- 只在 T 非 BLOCK 倍数时出现 → 边界 mask：查 `kv_ok` 是否同时用于 load 和 where。

### 陷阱 4：benchmark 里的隐形不公平

**症状**：SDPA 被"锁后端"计时后莫名变慢 10-20%。

**原因**：把 `with sdpa_kernel([be])` 写进了被计时的 lambda——每次迭代都付一遍
上下文进出（backend 开关）的 Python 开销。

**解法**：上下文管理器包在 `bench()` 外面，只进一次；内核外的开销在微秒级内核
对比里是决定性的。同理预热不可省（首次调用含编译/autotune）。

## 六、生态：FlexAttention 与 SageAttention（段 5）

### 6.1 FlexAttention：几行 PyTorch 复现同样的 mask

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mod(b, h, q_idx, kv_idx):          # 返回 True = 参与注意力
    return q_idx >= kv_idx

def sliding_mod(b, h, q_idx, kv_idx, W=256):  # causal + 最近 W 个 token
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= W)

causal_bm  = create_block_mask(causal_mod, None, None, T, T)   # 编译成 128x128 块级 Bitmap
sliding_bm = create_block_mask(sliding_mod, None, None, T, T)
out = flex_attention(q, k, v, block_mask=sliding_bm)
```

实测（段 5a 输出）：causal 与 sliding(W=256) 都与 naive 参考 `assert_close` 通过
（相对误差 3.7e-02，bf16 同口径）。`create_block_mask` 把 Python 函数变成**块级**
Bitmap，sliding 这类带状 mask 的全 0 块直接不进内核——正是我们三阶段分解里
"阶段 3 跳过"的通用化。FlexAttention 经 torch.compile 生成 Triton 内核，官方实测
为 FA2 前向的 90%（A100）——**它就是"本章内核的自动化版本"**。

### 6.2 SageAttention：4090 的甜点（只介绍，不实现）

论文 [arXiv 2411.10958](https://arxiv.org/abs/2411.10958)（SageAttention2）：QK^T 用
INT8、PV 用 **INT4** 量化并平滑 outlier。4090 恰是甜点卡——RTX 40 系的 INT4 Tensor
Core 吞吐极高（论文实测 INT8 只有 INT4 一半速度），整体约 **3× FA2**。代价是
近似（量化误差），推理推荐、训练慎用。与 FA3 的"Hopper 专属 WGMMA + FP8"是两条
不同的路线：一条榨**指令集**，一条榨**数值格式**；我们 SM89 的卡只能走第二条。

## 学完本章你能...

- ✅ 推导 online softmax 的 α-重缩放更新（m / l / acc 三状态）
- ✅ 手写带 causal 三阶段、边界 mask、exp2 技巧的 Triton Flash Attention 前向
- ✅ 用 profiler 证据回答"SDPA 到底命中了哪个后端"
- ✅ 说出 FA1→FA3→Flex→Sage 各自的改进点和 4090 的边界（SM89 无 WGMMA）
- ✅ 按验收承诺（≥50% 合格 / >85% 优秀）给自己的内核打分

## 练习与思考

### 概念检验

<details>
<summary>Q1: online softmax 为什么必须同时保留 m（行最大值）？只留 l 不行吗？</summary>
A: 不行。l 是"以当前基准 m 为底"的指数和，新块到来若最大值变大（m_new > m），
历史 l 必须乘 α = exp(m - m_new) 折算到新基准，否则历史项被系统性高估。
只留 l 就丢失了"历史项是按哪个基准算的"这一信息。反过来说，m 不变时 α=1 零损耗，
所以 m 的作用是**让分块流式计算保持数值稳定**（防 exp 上溢），这正是它从 Part 1
的"减最大值"一脉相承的地方。
</details>

<details>
<summary>Q2: Flash Attention 省了显存，省了 FLOPs 吗？它到底省了什么？</summary>
A: 不省 FLOPs——QK^T 与 PV 的乘加数不变（精确算法）。它省的是**HBM 读写**：
naive 要把 (T,T) 的 S 写回显存、读回来做 softmax、再写回读回来做 PV，O(T²) 流量；
FA 把这一切留在片上（SRAM/寄存器），HBM 流量降到 O(T·D)。所以 FA 论文标题里有
"IO-Awareness"：在内存墙语境（02 章）下，减少访存比减少计算更值钱。附带收益是
训练显存 O(T²)→O(T)，长序列从"放不下"变"放得下"。
</details>

<details>
<summary>Q3: 实测 causal 只有 full 的 1.6×（0.310 vs 0.499 ms），不是理论上的 2×。差在哪？</summary>
A: 三块不减的开销：① 每个 program 的固定成本——Q 块加载、epilogue 除法与写回
都是 O(BLOCK_M·D)，与扫多少 key 块无关；② 对角块（阶段 2）仍要全量算再 mask，
FLOPs 没省一半；③ 网格/启动开销。带外块（阶段 1）确实省成了"无 mask 快速路径"，
加上阶段 3 整段跳过，总账就是 ~1.6×。序列越长、BLOCK_M 相对越小，越接近 2×。
</details>

### 动手实践

#### 练习 1：把内核接回 minimind，替换 SDPA 测 ppl

把 `fa_forward` 接进 [Part7 脚本 05](../../Part7_minimind/scripts/05_full_model.py) 的
`MiniMindAttention`，用训练好的 checkpoint 对比验证集 ppl。

**步骤提示**：
```python
# Part7 里 q/k/v 是 (B, T, H, D)，且 K/V 是 GQA 的 4 个头——先转成内核要的布局：
q_h = q.transpose(1, 2)                       # (B, T, H, D) -> (B, H, T, D)
# GQA：把 4 个 kv 头 repeat 成 8 个（torch.repeat_interleave）再喂 fa_forward
# 或者：8 个 q 头按 kv 分组循环调用（更省显存）
```

**验收标准**：
- [ ] 验证 ppl 与 SDPA 版相差 < 0.02（bf16 噪声级）
- [ ] 前向耗时与 SDPA flash 同量级（不要求更快）
- [ ] 写一句结论：GQA 下你的调用方式浪费在哪（提示：repeat 了 K/V）

#### 练习 2：关掉 autotune，画 BLOCK 尺寸的性能曲线

固定 `BLOCK_M/BLOCK_N` 手动传参（把 `@triton.autotune` 去掉或直接调
`_fa_fwd.run`），扫 (BM, BN) ∈ {64,128}² × T ∈ {1K, 2K, 4K}，matplotlib 画柱状图
（图表标题用英文，如 "FA throughput vs block size"）。

**验收标准**：
- [ ] 复现 autotune 的选择在你的机器上确实（接近）最优
- [ ] 找出最差组合并解释：SMEM 用量（BM·BN 越大流水线越深）vs occupancy 的权衡
- [ ] 图上注明环境（GPU / dtype / 独占与否）

### 扩展思考

- **backward 怎么写？** 前向存下 m + log(l)（官方 tutorial 06 的 M 矩阵），反向
  沿同样的分块重算 p 并累计 dq/dk/dv。试试读懂官方 `_attn_bwd`，画出它与前向
  对称的"行块/列块"分工图。
- **GQA/KV Cache 与本章内核怎么组合？** Part 7 的 KV Cache 让 K/V 长度 ≠ Q 长度，
  本章内核的循环边界要怎么改？（提示：Q 块的因果边界从 `offs_m` 变成
  `offs_m + kv_offset`。）

## 参考资源

- 📖 [Triton 官方 tutorial 06: fused-attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)——
  本章内核的出处（含 backward 与 fp8 变体）
- 📺 [GPU MODE Lecture 12: Flash Attention](https://www.youtube.com/watch?v=zEuwuCTEf_0)
  （主讲 **Thomas Viehmann**，2024-03-30；[讲义代码](https://github.com/gpu-mode/lectures)在
  `lecture_012` 目录）——从 online softmax 直觉讲到 Triton 实现，本章的"视频版"
- 🐙 [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)——
  极简单文件实现，适合逐行对照
- 📝 [FlashAttention-2 in CuTe, from Scratch（Edwin Chen, echen.io）](https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/)——
  用 CUTLASS CuTe 从零写 FA2 的系列，读懂它就摸到 FA3 的大门
- 📚 论文：FA1 [2205.14135](https://arxiv.org/abs/2205.14135) / FA2 [2307.08691](https://arxiv.org/abs/2307.08691) /
  FA3 [2407.08608](https://arxiv.org/abs/2407.08608) / online softmax 收敛形式
  Rabe & Staats [2112.05682](https://arxiv.org/abs/2112.05682) / SageAttention2
  [2411.10958](https://arxiv.org/abs/2411.10958)
- 📖 [FlexAttention 官方博客](https://pytorch.org/blog/flexattention/)（90% FA2 的出处）与
  [torch.nn.attention.flex_attention 文档](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)

## 下一步

到这里，Part 9 的闭环完成：**从 vector add 到 matmul 阶梯，从 Triton softmax 到
Flash Attention**——Part 7 里那个"黑盒" `scaled_dot_product_attention`，你已经把
它的内部亲手写了一遍。回到课程主线（04 章"路线 C"）：带着内核视角去看 Part 10 的
分布式训练与 Part 14 的 vLLM 推理引擎，你会看到同一个内核出现在不同的系统位置上。

---

[← 上一章：04 Triton 与 PyTorch 扩展](04_triton_and_extensions.md) | [返回 Part 9 目录](README.md)
