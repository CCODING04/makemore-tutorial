# Assignment 14：推理部署（vLLM）

> 对应 Part 14 教程（[01 朴素基线](../../courses/Part14_inference_vllm/tutorial/01_naive_baseline.md) / [02 vLLM 实战](../../courses/Part14_inference_vllm/tutorial/02_vllm_serving.md)）。
> 题 1-4 纯纸笔/纯 Python 可完成（4 × 25 = 100 分）；题 5 🌟 为 stretch 选做；
> 实验题跑脚本 01 + 完成 02 章 CLI 实操（观测型，不占分）。

## 题目（实现 `serving_exercises.py`）

| 题 | 分值 | 主题 | 考点 |
|---|---|---|---|
| 1 | 25 | serving 指标 | E2E = TTFT + TPOT×(n−1)；吞吐 = 总 token / 墙钟 |
| 2 | 25 | KV 容量账 | KV 公式（LLaMA-7B fp16 seq2048 = 1.07GB；GQA 1/4）+ 反推最大并发 |
| 3 | 25 | 批处理浪费 | 静态批 pad 到 max 的浪费率（100/10/10/10 → 67.5%！）vs 连续批理想 0 |
| 4 | 25 | 投机解码数学 | 接受率 α → 每周期期望 token 数 (1−α^{γ+1})/(1−α) 与加速比上界 |
| 5 🌟 | 选做 | 连续批处理调度模拟器 | 静态 vs 连续的 makespan/浪费率——Orca 动机亲手算出来 |

函数签名、步骤提示、验收标准都写在 `serving_exercises.py` 的 docstring 里；
实现后跑 `python test_serving_exercises.py`（或 `pytest`）逐题验证。

### 题 1：serving 指标（25 分）

实现 `e2e_latency_ms(ttft_ms, tpot_ms, n_out)` 与
`throughput_tokens_per_s(n_requests, n_out_tokens, wall_seconds)`。
公式见 [01 章](../../courses/Part14_inference_vllm/tutorial/01_naive_baseline.md)：
首 token 只等 TTFT，之后每个 token 等 TPOT；吞吐是服务方视角（全体请求合计）。

**验收标准：**
- [ ] `e2e_latency_ms(200, 50, 5) == 400.0`（200 + 50×4）
- [ ] `e2e_latency_ms(200, 50, 1) == 200.0`（单 token 输出 = 纯 TTFT）
- [ ] `throughput_tokens_per_s(8, 32, 2.0) == 128.0`（8×32 token / 2s）

### 题 2：KV 容量账（25 分）

实现 `kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2)`
与 `max_batch_for_vram(kv_gb_per_seq, vram_gb=24.0, model_gb=4.0, headroom_gb=2.0)`。
公式来源：[Part 8 06 章 §2](../../courses/Part8_post_training/tutorial/06_inference_and_serving.md)
（2(K+V) × layers × kv_heads × head_dim × seq × batch × bytes）。

**验收标准：**
- [ ] `kv_cache_gb(32, 32, 128, 2048, 1) ≈ 1.07`（LLaMA-7B fp16，seq 2048）
- [ ] GQA 把 kv_heads 32→8，KV 恰好缩小 4 倍（线性缩放不变量）
- [ ] KV 随 batch、seq_len 均线性缩放
- [ ] `max_batch_for_vram(0.27, 24, 4, 2) == 66`（可用 18GB / 每序列 0.27GB）
- [ ] 预算不足时至少返回 1（`max(1, ...)` 保底）

### 题 3：批处理浪费（25 分）

实现 `static_batch_waste(jobs, pad_to_max=True)`：静态批处理把整个 batch
pad 到最长 job，浪费 = pad 掉的 token 份额；`pad_to_max=False` 表示连续批处理的
理想情况（逐请求，无浪费）。这是 [Orca 论文](https://www.usenix.org/conference/osdi22/presentation/yu)
动机的量化版，也是 [02 章](../../courses/Part14_inference_vllm/tutorial/02_vllm_serving.md)
"连续批处理" 行的数学内核。

**验收标准：**
- [ ] 等长 `[10,10,10,10]` → 0（无浪费）
- [ ] `[100,10,10,10]` → 0.675（分配 4×100、实际 130）
- [ ] `pad_to_max=False` → 恒 0.0（连续批理想）
- [ ] 返回值落在 `[0, 1)` 区间

### 题 4：投机解码——接受率与有效加速（25 分）

实现 `spec_tokens_per_cycle(alpha, gamma)` 与
`spec_decode_speedup(alpha, gamma, draft_overhead=0.0)`。
机制见 [Part 8 06 章 §4](../../courses/Part8_post_training/tutorial/06_inference_and_serving.md)：
draft 先写 γ 个 token，target 一次前向并行验证；接受率 α 下每个周期的期望产出
E = (1−α^{γ+1})/(1−α)（全拒时仍白赚 1 个 token；α=1 时取极限 γ+1）。
加速比上界 = E / (1 + draft_overhead)——draft 开销折算成 target 前向的倍数。

**验收标准：**
- [ ] `spec_tokens_per_cycle(0.0, 4) == 1.0`（全拒：每周期仍出 1 个 token）
- [ ] `spec_tokens_per_cycle(1.0, 4) == 5.0`（全收：γ+1，公式 0/0 的极限）
- [ ] `spec_tokens_per_cycle(0.6, 4) ≈ 2.3056`（Part 8 实测 α≈0.60 的理论口径）
- [ ] 对 α 单调递增，且上界 γ+1
- [ ] `spec_decode_speedup(0.6, 4, 0.0) == spec_tokens_per_cycle(0.6, 4)`（零草稿开销 → 上界）
- [ ] `spec_decode_speedup(0.0, 4, 0.5) ≈ 0.667 < 1`（低接受率 + 贵草稿 = 负收益）

### 题 5 🌟 stretch：连续批处理调度模拟器（选做，不计入 100 分）

实现 `simulate_batching(arrivals, gen_lens, max_batch=8, mode="continuous")`：
离散时间步模拟，每步每个在跑请求出 1 个 token。`static` 模式逐批组队、整批陪跑
到最慢的；`continuous` 模式完成即腾位、等待者立刻补位。未实现返回 `None`
（测试会优雅 SKIP ⏭️，不影响核心 4 题得分）。

**验收标准：**
- [ ] `[0,0]` 到达、`[10,2]` 生成、max_batch=2：static 浪费率 0.4，continuous 0.0
- [ ] `[0,0,2]` 到达、`[10,2,5]` 生成、max_batch=2：static makespan=15，
      continuous makespan=10（批跑期间到达的新请求立刻补位——Orca 的胜利）
- [ ] 两种模式 `tokens == sum(gen_lens)`；浪费率 ∈ [0,1)
- [ ] max_batch=1 且全部单 token：两种模式 makespan 相同（退化为串行）

## 实验题（4090，观测型）

- 跑脚本 01 出 naive 基线（吞吐/TTFT/TPOT），装 vLLM（两案选一）完成 02 章 CLI 实操，
  填完三行对比表——面试即用的实证
- 打开 `--enable-prefix-caching`，构造"共享系统提示词的 64 请求"vs"随机 64 请求"，
  对比 TTFT 差异，写出 prefix caching 的适用条件

## 🤔 思考题

**Q1：** 题 3 里静态批处理的浪费率是"token 份额"，它和 GPU 时间的浪费是什么关系？
什么情况下 token 浪费率高但 GPU 时间浪费率低？

<details>
<summary>💡 参考答案</summary>

静态批处理中，每个槽位都跑满 max(gen_lens) 步，所以 **slot×step 与 GPU 时间一一对应**：
token 浪费率 ≈ GPU 时间浪费率（把 pad 的槽位也算成在"陪跑"）。
但如果所有请求长度几乎一样（等长负载，如批处理翻译），max ≈ 实际长度，
token 浪费率趋近 0——此时静态批几乎不浪费。反过来，长尾分布（100/10/10/10）
浪费最狠。这正是 Orca/vLLM 连续批处理对**异构负载**收益最大的原因；
负载本身同质时，连续批的收益主要来自"新请求立刻补位"而不是"提前释放槽位"。

</details>

**Q2：** 题 4 中 α=1 时公式 (1−α^{γ+1})/(1−α) 是 0/0。为什么极限恰好是 γ+1？
（提示：把它看成等比级数求和）

<details>
<summary>💡 参考答案</summary>

(1−α^{γ+1})/(1−α) = 1 + α + α² + … + α^γ（等比级数恒等式）。
α=1 时每一项都是 1，共 γ+1 项，和为 γ+1。
直觉：接受率 100% 意味着 draft 的 γ 个 token 全部被 target 认可，
再加 target 自己续写的 1 个"白赚" token，每周期产出 γ+1 个。
实现上必须显式处理这个分支（`if alpha >= 1: return gamma + 1`），
否则浮点除零——这也是测试里专门设 `tpc(1.0, 4) == 5.0` 的原因。

</details>

**Q3：** 投机解码的加速比上界公式里，为什么"验证 γ+1 个 token 的一次前向"
可以近似当成"生成 1 个 token 的一次前向"？什么时候这个近似失效？

<details>
<summary>💡 参考答案</summary>

decode 阶段是 memory-bound（Part 8 06 章的核心结论）：一步 decode 的成本
几乎全在把权重从显存搬到 SM，算力大量闲置。验证 γ+1 个位置是把它们拼成
batch=γ+1 一起前向——权重**只搬一遍**，多算的 γ 份矩阵乘几乎免费。
所以验证成本 ≈ 单 token decode 成本，上界公式成立。
失效场景：① batch 已经很大（decode 进入 compute-bound 区间，多算不免费）；
② γ 非常大导致激活/中间张量挤占带宽；③ draft 模型不够小，
`draft_overhead` 项吃掉收益（题 4 的 α=0、overhead=0.5 → 加速 2/3 就是反例）。

</details>

**Q4：** 题 2 算出 LLaMA-7B fp16 在 seq=2048 下单请求 KV 就要 ~1.07GB。
PagedAttention 改变的是这个公式的哪一项？它为什么能提升"可服务并发数"？

<details>
<summary>💡 参考答案</summary>

公式 `2 × layers × kv_heads × head_dim × seq × batch × bytes` 里，
PagedAttention 不改变**任何一个系数**——它改变的是 seq 这一项的**计量方式**：
朴素方案按 max_seq_len（如 2048）整块预留，实际平均用到几百；
分页方案按需以 16 token/块分配，浪费只剩最后一个块的内碎片（<4%）。
效果上等于把"每请求预留 seq"从 2048 压到实际长度，
`max_batch_for_vram` 的分子（可用显存）没变、分母（每序列 KV）缩小数倍，
可服务并发数同比例上升。再叠加共享前缀（prefix caching）让多个请求
**共享**同一批物理块，等效并发进一步提升。

</details>

**Q5：** 题 5 的模拟器假设"连续批处理换入换出零开销、零重组成本"。
真实的 vLLM 里这个假设哪些地方不成立？方向上会低估还是高估连续批的收益？

<details>
<summary>💡 参考答案</summary>

不成立之处：① 请求进/出 batch 会改变 batch 形状，CUDA kernel 要重排
（vLLM 用 CUDA graphs 缓解，但换档仍有开销）；② PagedAttention 块表维护、
采样与调度器（Python 侧）的每步开销随 batch 波动；③ prefill 与 decode
混跑时的干扰（chunked prefill 就是为它设计的）。
方向：这些都是**连续批的额外成本**，模拟器没算，所以**高估**连续批相对静态批的优势。
但静态批在真实系统里还有另一个模拟器没算的坏处——早完成的请求占着显存不放，
KV 无法复用，所以真实差距通常仍站在连续批这边。做实验报告时要说明
"理想模型 + 真实系统开销"这一层系统性偏差。

</details>

## 🎯 面试直通车

- "你怎么证明 vLLM 快？"——同模型/prompt/指标的三行对比表 + 归因（连续批处理/分页/缓存）
- "TPOT 变差了为什么还用它？"——serving 优化吞吐成本，单请求延迟用 SLO 路由（goodput）
- "GQA 省的是什么？"——KV cache 随 kv_heads 线性缩（题 2 第一手数字：1.07GB→0.27GB）
- "静态批处理的浪费怎么算？"——题 3：pad 到 max 的份额；连续批处理消掉它
- "投机解码什么时候负收益？"——题 4：α 低 + draft 贵，加速比 = E/(1+overhead) < 1
