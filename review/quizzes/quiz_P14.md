# 测验 · Part 14（推理部署实战：vLLM）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** TTFT、TPOT、吞吐三个 serving 指标的定义与公式是什么？各由推理的哪个阶段主导？单请求端到端延迟怎么估？

- **答案**：TTFT（首 token 延迟）= 请求提交到第一个 token 到达，由 prefill 阶段主导，对应"用户反应快不快"；TPOT = (最后 token 时刻 − 首 token 时刻)/(n−1)，由 decode 阶段主导，对应"打字机流畅度"；吞吐 = total_tokens/total_time（服务方视角的全体合计）。端到端延迟约 $\text{E2E} \approx \text{TTFT} + \text{TPOT} \times (n_{\text{out}} - 1)$（作业 14 题 1：e2e_latency_ms(200, 50, 5) = 200 + 50×4 = 400.0 ms）。分布式 serving 还要看 p50/p90/p99——真实体验由尾部决定。
- **锚点**：教程 01_naive_baseline.md §代码实现（指标的可操作定义）+ 作业 14 题 1

**Q2（数字）** 课程 naive 基线（4090、Qwen2.5-0.5B、64 请求 × 32 token）实测的三个指标是多少？静态批处理（batch=8）为什么能立刻赢近 7 倍？

- **答案**：逐请求循环（serving 反模式）：吞吐 158 tok/s、TTFT p50/p90 = 7.5/7.6 ms、TPOT p50/p90 = 6.2/6.3 ms，显存约 2GB。静态批 batch=8 达 1071 tok/s（6.8 倍）——推理是 memory-bound：decode 每步都要把全部权重从显存搬一遍，批 8 个请求等于权重只读一次喂 8 个人，搬运成本被摊薄。p90≈p50 是因为串行逐请求无排队，TTFT 就是恒定 prefill 时间（"空载 TTFT"）；真 serving 高负载下尾部才会被排队+批内干扰拉开。
- **锚点**：教程 01_naive_baseline.md §代码实现（基线结果）+ 概念检验 Q1/Q2

**Q3（对比）** 静态批处理与连续批处理的本质区别是什么？用作业里的浪费率数字说明；PagedAttention 又把 KV 显存浪费从多少降到多少？

- **答案**：静态批把整批 pad 到最长请求并陪跑到最慢的完成——作业 14 题 3：jobs=[100,10,10,10] 时分配 4×100 而实际只需 130，浪费率 0.675（67.5%）；连续批处理"早走早换人"，理想浪费 0（Orca 论文杀死的正是 static batching 浪费）。PagedAttention 把 KV cache 按固定 block（常用 block_size=16）按需分配，代替"每请求整块预留 max_seq_len"：手写模拟从整块预留的 41% 浪费降到 ~5%，论文口径 <4%。vLLM 的增益 = 连续批处理 + PagedAttention（batch 开得更大）+ prefix caching。
- **锚点**：教程 02_vllm_serving.md §理论背景（PagedAttention 的显存优化）+ 作业 14 题 3 + 01 章 §代码实现

**Q4（数字）** LLaMA-7B（fp16、MHA 32 个 KV 头、head_dim=128、seq=2048）单请求的 KV cache 有多大？GQA 把 KV 头降到 8 后呢？vLLM 里 `--max-model-len` 和 `--gpu-memory-utilization` 分别控制什么？

- **答案**：KV 公式 $2(\text{K+V}) \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{seq} \times \text{batch} \times \text{bytes}$，代入得 32 层 × 32 头 × 128 × 2048 × 2B × 2(K+V) ≈ 1.07GB/序列；GQA 把 kv_heads 32→8，KV 随之线性缩为 1/4（0.27GB）。反推最大并发：24GB 显存 − 4GB 权重 − 2GB 余量 = 18GB 可用，18/0.27 ≈ 66 个并发。vLLM 中 `--max-model-len` 是单请求序列长度上限（KV 池上限），`--gpu-memory-utilization` 控制分给模型+KV 池的显存占比（默认约 0.9），`--enable-prefix-caching` 让共享前缀免重算。
- **锚点**：作业 14 题 2（KV 容量账）+ 教程 02_vllm_serving.md §代码实现（离线推理/服务参数映射）

**Q5（诊断）** 你测 naive 基线：`time.time()` 包住 `model.generate` 得到 0.001s；批处理时不同长度请求混批生成结果错乱；vLLM 上线后发现 TPOT 反而比 naive 略高。各是什么问题？

- **答案**：一是异步陷阱——GPU 异步执行，`generate` 返回的是"提交 kernel"的时刻，掐表前后必须 `torch.cuda.synchronize()` 才测到"算完"（脚本 7 处同步点）。二是 padding 陷阱——decoder-only 因果注意力必须左 padding，右 padding 会让模型"看到"后面的 padding token、位置算歪。三是这不是故障：大 batch 下 decode 每步算更多请求，单步变慢（TPOT 略升），但吞吐大涨（158 → 数千 tok/s）——serving 的本质是吞吐换延迟，优化的是每卡 token 成本；延迟敏感场景用小 batch/SLO 路由。
- **锚点**：教程 01_naive_baseline.md §代码实现（测量陷阱）+ §调试展示（错误 1/2）+ 02_vllm_serving.md 概念检验 Q1

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 理解推理部署在 LLM 链路中的位置和价值 | Q2 | "模型变产品"：成本/速度/并发三证据并入指标动机 |
| 手写朴素推理基线并测量 TTFT/TPOT/吞吐，完成公平对比 | Q1、Q2 | 指标定义 + 实测数字 + 公平性三原则 |
| 解释 vLLM 的核心优化（连续批处理、PagedAttention、投机解码） | Q3 | 批处理浪费 0.675 vs 0、KV 浪费 41%→<4%；投机解码数学并入闪卡与 Q4 数字面 |
| 配置 vLLM 的推理服务并理解每个参数的含义 | Q4 | max-model-len/gpu-memory-utilization/prefix-caching |
| 识别推理部署中的常见陷阱并设计防范策略 | Q5 | 异步/padding/吞吐换延迟三类 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| E2E 延迟估算公式？ | $\text{E2E} \approx \text{TTFT} + \text{TPOT} \times (n_{\text{out}} - 1)$ |
| TTFT / TPOT 各由哪个阶段主导？ | TTFT ← prefill；TPOT ← decode |
| naive 基线实测三指标？ | 吞吐 158 tok/s、TTFT p50 7.5ms、TPOT p50 6.2ms（4090 / Qwen2.5-0.5B / 64×32） |
| 静态批 batch=8 为什么赢 6.8 倍？ | memory-bound：权重搬运被 8 个请求摊薄（1071 tok/s） |
| 静态批 vs 连续批的浪费率？ | [100,10,10,10] 静态 0.675；连续批理想 0（早走早换人，Orca） |
| PagedAttention 的核心思想与效果？ | KV 按固定 block 按需分配（block_size=16）；整块预留 41% 浪费 → <4%（论文口径） |
| KV cache 大小公式？ | $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{seq} \times \text{batch} \times \text{bytes}$；LLaMA-7B fp16 seq2048 ≈ 1.07GB/序列 |
| GQA 省 KV 的比例？ | kv_heads 32→8，KV 线性缩 4 倍（1.07GB→0.27GB），并发上限 18/0.27 ≈ 66 |
| 投机解码每周期期望 token 数？ | $E = (1 - \alpha^{\gamma+1})/(1 - \alpha)$；α≈0.60、γ=4 时 ≈2.31，全拒仍白赚 1 个 |
| 为什么 vLLM 的 TPOT 可能略高于 naive 还要用它？ | 大 batch 单步变慢但吞吐大涨——serving 是吞吐换延迟 |
| decoder-only 批处理必须哪种 padding？ | 左 padding；右 padding 会被因果注意力"看到"、位置算歪 |
| GPU 计时的铁律？ | 掐表前后 `torch.cuda.synchronize()`，否则测的是"提交 kernel"时间 |
| prefix caching 何时收益最大？ | 共享前缀长且重复率高（系统提示词/few-shot/多轮历史）；随机 prompt 纯开销 |
