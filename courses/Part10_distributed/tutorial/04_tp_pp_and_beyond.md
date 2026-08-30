# 04 — 张量并行、流水线并行与工业栈（进阶可选）

> 🧭 最后一章是"进阶可选"：前两节的并行把模型状态切开了，但如果**单层权重/激活**就装不下，
> 要把一层之内的计算切开（张量并行 TP），或者让不同的卡负责不同的层（流水线并行 PP）。
> 这两种并行的代码复杂度显著更高——理解机制 + 亲手验证过一次最小实现，就是本章的合格线；
> 不要求在生产里自己写 TP/PP（那是 Megatron/DeepSpeed/nanotron 的工作）。
>
> 本章数字与公式全部在脚本 05/06 中实测可复现。

## 📖 前置知识

- **03 章**：ZeRO/FSDP 的通信原语（all-gather / reduce-scatter）
- **脚本 05/06**：本章的实证载体（torchrun 双卡验证，单进程也兼容）

## 1. 张量并行（Megatron 式）：把一层切成两半

以 MLP `Y = gelu(X·W1ᵀ)·W2ᵀ` 为例，Megatron 的切法：

```
W1（第一层）列并行：按输出维切 → 各 rank 算 gelu(X·W1_rᵀ)，前向无通信
W2（第二层）行并行：按输入维切 → 各 rank 算 H_r·W2_rᵀ → all-reduce 求和 = 完整 Y
```

- 🔑 为什么能"中间无通信"？因为 gelu 作用在**分片内部**：H 的列被切开，每列的计算只依赖
  W1 的对应行块。两个共轭算子：**f**（forward 恒等 / backward all-reduce）与
  **g**（forward all-reduce / backward 恒等）→ 每层 forward 恰 1 次、backward 恰 1 次
  all-reduce。
- [脚本 05](../scripts/05_tensor_parallel.py) 实测（2×4090）：

```
前向 max |Y_tp - Y_dense| = 5.96e-07  → ✅（<1e-5 验收线）
梯度 max |dX_tp - dX_dense| = 5.82e-11 → ✅
```

- Attention 同构：QKV 投影按**头**切（天然列并行，呼应 Part 7 GQA 的多头结构），
  输出投影行并行。
- ⚠️ TP 的通信在**每层内部**，频率极高 → 只适合 NVLink 互联的同一台机器内（机内 8 卡），
  跨机用 TP 会把带宽吃光。进阶：all-reduce 还能拆成 reduce-scatter + all-gather，
  顺带把 LayerNorm/Dropout 的激活也分片（sequence parallelism，arXiv 2205.05198）。

## 2. 流水线并行（GPipe / 1F1B）：按层接力

```
4 层模型、2 个 stage：
  stage0（rank0）：embedding + blocks[0:2]   stage1（rank1）：blocks[2:] + head
数据切成 m=4 个 micro-batch 填流水线：
  stage0: F1 F2 F3 F4 ──────────────── B4 B3 B2 B1
  stage1:    F1 F2 F3 F4 ──── B4 B3 B2 B1      （F=forward, B=backward）
            ↑______气泡______↑
```

- [脚本 06](../scripts/06_pipeline_parallel.py) 用两个自定义 autograd.Function
  （Send/RecvActivation，forward 传激活、backward 传梯度）实现了最小 GPipe，
  实测**流水线 loss 与单进程整模型完全一致（4.380254 == 4.380254）**——
  "按层切开不改变数学"的最硬证据。
- 🔑 **bubble 公式**：气泡占比 = (p−1)/(m+p−1)。p=2, m=4 → 20%；m 增大气泡被摊薄，
  但 m 个 micro-batch 的激活也要驻留（GPipe）；**1F1B** 调度交错执行 forward/backward，
  把激活驻留从 m 个降到 p 个——大模型流水线的标配。
- ⚠️ 工程实测坑（本课开发机踩到）：4090+4090D 混合机型上 NCCL 的 send/recv 点对点会
  互相卡死（集合通信正常）。脚本 06 的解法：点对点单独建 **gloo 组、CPU 中转**。
  教训：分布式问题不总是逻辑 bug，通信后端与硬件拓扑的组合也要怀疑。

## 3. 拼起来：3D 并行与工业栈

```
总卡数 = TP × PP × DP          （例：LLaMA 2 70B = 8×TP（机内） × 16×PP? 实际配置见下）
LLaMA 2 70B 公开配置：TP=8 × PP=? — 官方报告用 2000+ 卡，MFU ≈ 46%
（读配置的习惯：TP 尽量小且机内 → PP 其次 → 剩下全部给 DP；再叠 ZeRO-1/激活重计算）
```

工业参考栈（按"想继续深入"排序）：

| 资源 | 适合谁 |
|---|---|
| [nanotron](https://github.com/huggingface/nanotron) | 想读"能跑通的 3D 并行极简实现"（PP 有 AFAB/1F1B，ZeRO-1，MoE 专家并行） |
| [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | 想系统学"集群训 LLM"（530 卡 Llama-1B 全过程讲解，本课的下一站） |
| Megatron-LM / DeepSpeed | 生产级 TP/PP/ZeRO 的原始出处（读文档理解语义即可，不必上手） |
| torchtitan | PyTorch 官方的 LLM 预训练脚手架（FSDP2 + 并行组合的现成配置） |

## 学完本章你能...

- ✅ 画出 MLP 的列/行并行切法，解释 f/g 算子与"每层恰 2 次 all-reduce"
- ✅ 画出 GPipe 时间线，算 bubble=(p−1)/(m+p−1)，说出 1F1B 省了什么
- ✅ 复述 TP 适合机内（NVLink）、PP 消息少但气泡大、DP 最便宜的分工逻辑
- ✅ 说出"读配置"的顺序：先 TP 后 PP 剩 DP，再叠 ZeRO 与激活重计算

**课后练习**

<details>
<summary>Q1: 为什么 TP 的 all-reduce 不能像 DDP 那样与计算重叠，导致它对带宽最敏感？</summary>
A: DDP 的梯度 all-reduce 在 backward 尾声、可按桶异步化，与"其余层反传"重叠；TP 的
all-reduce 在【每一层的正中间】，前后都是依赖它的计算，遮不住。所以 TP 只放机内
（NVLink ~900GB/s 级），跨机（~25-100GB/s）会被通信吃掉大半 MFU。
</details>

<details>
<summary>Q2: p=8 个 stage，想让 bubble < 10%，micro-batch m 至少多大？代价是什么？</summary>
A: (p-1)/(m+p-1) < 0.1 → m+p-1 > 70 → m ≥ 63。代价：GPipe 下 63 个 micro-batch 的激活
都要驻留 → 激活显存爆炸，所以要换 1F1B（激活驻留降到 p 个）+ 梯度检查点。
</details>

<details>
<summary>Q3: 4×8 卡（机内 4 台、每台 8 卡）训一个 70B，你会怎么分配 TP/PP/DP？</summary>
A: TP=8（占满机内 NVLink），剩 32 个"流水线单元"→ PP=4 × DP=8 或 PP=8 × DP=4，按
激活显存与气泡权衡（模型越深 PP 越大 bubble 越贵，配 1F1B）；DP 部分叠 ZeRO-1。
这不是唯一解——面试重点是给出"机内 TP 优先、气泡与显存权衡、剩余给 DP"的推理链。
</details>

## 📝 课后作业

👉 [Assignment 10](../../../assignments/assignment_10/) 题 4（TP 分块数学）+ 🌟 题 5（bubble 计算器）

## 全课程毕业

```
Part 1-5   会训练                     Part 6-8   会造模型、会后训练
Part 9     懂硬件                     Part 10    懂集群
下一站：nanotron / Ultra-Scale Playbook / llm.c —— 带着这套地基去读它们 ✈️
```

---

[← 上一章：显存账本与 ZeRO/FSDP](03_memory_zero_fsdp.md) | [Part 10 README](README.md)
