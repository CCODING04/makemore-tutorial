# 01 — 朴素基线与对比设计：TTFT/TPOT/吞吐怎么测

> 🧭 "vLLM 快 20 倍"这种话在面试里没有价值；**"同模型、同 prompt、同指标下快 X 倍，
> 数据在这"** 才有价值。本章先建立测量方法（跑
> [scripts/01_naive_generate_baseline.py](../scripts/01_naive_generate_baseline.py)），
> 产出一张留空的对比表——02 章 vLLM 来填空。

## 📖 前置知识

- **Part 8 06 章**：TTFT/TPOT/goodput 定义、memory-bound 直觉
- **Part 9 02 章**：内存墙（为什么批处理能赢）

## 1. 指标的可操作定义（测量代码里怎么算）

```
TTFT（首 token 延迟）：提交请求 → 第一个 token 到达。prefill 主导，用户"反应快不快"
TPOT（每 token 间隔）：(总时间 - TTFT) / (n_tokens - 1)。decode 主导，"打字机流畅度"
吞吐 throughput     ：全体请求 tok/s 合计（服务方视角）
p50/p90             ：分布式 serving 的真实体验由尾部决定，别只报平均
E2E 延迟            ≈ TTFT + TPOT × (输出 token 数 − 1)
```

⚠️ 测量陷阱（本脚本全部真实处理过）：
- **异步陷阱**：GPU 是异步的——不 `synchronize`/同步读结果，测出的时间是假的（Part 9 01 章）；
- **padding 陷阱**：decoder-only 批处理必须**左 padding**（右 padding 会把位置算歪）；
- **首 token 单独测**：HF `generate` 是一次性调用，TTFT 需要用 `max_new_tokens=1` 的
  独立探测来近似——工程上 serving 引擎会流式返回，天然可测。

## 2. 基线结果（Qwen2.5-0.5B，64 请求 × 32 token，4090）

```
[1] 逐请求循环（serving 反模式）:
    TTFT  p50/p90 : 6.3 / 6.5 ms
    TPOT  p50/p90 : 5.2 / 5.3 ms
    吞吐          : 181 tok/s
[2] 静态批处理（batch=8）: 1158 tok/s（吞吐×6.4！）
    —— 但早完成的请求陪跑到最慢的：这就是 Orca 论文要杀死的"static batching 浪费"
```

- 🔑 静态批处理已经赢 6 倍：**权重只读一次喂 8 个请求**（memory-bound 的直接推论）。
  vLLM 的增量 = 连续批处理（早走早换人）+ PagedAttention（batch 开得更大）+ prefix caching。

## 3. 对比实验设计（02 章的填空表）

| 指标 | naive 循环（本脚本） | vLLM（02 章） | 差异来自 |
|---|---|---|---|
| 吞吐 tok/s | 181（实测） | ? | 连续批处理 |
| TTFT p50 | 6.3 ms（实测） | ? | prefill 调度/chunked prefill |
| TPOT p50 | 5.2 ms（实测） | ? | decode batch 更大 + CUDA graphs |
| KV 显存 | 每请求整块预留 | ? | PagedAttention |

> 公平性三原则：同模型同 dtype、同 prompt 集（sonnet.txt 或固定 64 条）、同 max_new_tokens。
> 换任何一个，数字就不可比。

## 学完本部分你能...

- ✅ 用代码正确测出 TTFT/TPOT/吞吐的 p50/p90，避开三个测量陷阱
- ✅ 解释静态批处理为什么已经赢 6 倍、vLLM 又赢在哪
- ✅ 设计一个公平的 serving 对比实验

**课后练习**

<details>
<summary>Q1: 为什么 TTFT 的 p90 和 p50 几乎一样（6.5 vs 6.3）？什么场景下会拉开？</summary>
A: 本基线是串行逐请求——每个请求独占 GPU、无排队，TTFT≈恒定的 prefill 时间。
真 serving 在高负载下 TTFT 尾部会被排队+批内干扰拉长（这正是 P99 SLO 和 goodput 存在的
原因）。所以本脚本的 TTFT 是"空载 TTFT"，跟 vLLM 对比时注意负载条件要一致。
</details>

<details>
<summary>Q2: 静态批处理 8 路吞吐 1158 tok/s，是不是继续加 batch 就线性涨？</summary>
A: 在 memory-bound 区间近似线性（权重搬运被摊薄），直到 compute-bound 或显存（KV）耗尽
——4090 上 0.5B 模型 KV 很小，瓶颈先出现在调度/内存拷贝。真实大模型上 KV 显存先爆，
这正是 PagedAttention 的用武之地（02 章日志里 GPU KV cache usage 会印证）。
</details>

## 📝 课后作业

👉 [Assignment 14](../../../assignments/assignment_14/)

## 下一步

装 vLLM（两案选一），把填空表填上。

👉 [02 — vLLM 实战与对比](02_vllm_serving.md)
