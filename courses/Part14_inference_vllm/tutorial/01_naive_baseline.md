# 01 — 朴素基线与对比设计：TTFT/TPOT/吞吐怎么测

> 🧭 "vLLM 快 20 倍"这种话在面试里没有价值；**"同模型、同 prompt、同指标下快 X 倍，
> 数据在这"** 才有价值。本章先建立测量方法（跑
> [scripts/01_naive_generate_baseline.py](../scripts/01_naive_generate_baseline.py)），
> 产出一张留空的对比表——02 章 vLLM 来填空。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** TTFT/TPOT/吞吐的测量代码
- ✅ **解释** 每个指标的含义和测量陷阱
- ✅ **设计** 一个公平的 serving 对比实验
- ✅ **识别** 异步陷阱、padding 陷阱等常见错误

## 📖 前置知识

**必须掌握：**
- **Part 8 06 章**：TTFT/TPOT/goodput 定义、memory-bound 直觉
- **Part 9 02 章**：内存墙（为什么批处理能赢）

## 理论背景

### 问题引入：为什么需要测量指标？

推理部署虽然能跑通，但需要量化指标来评估性能：

1. **TTFT（首 token 延迟）**：用户"反应快不快"
2. **TPOT（每 token 生成时间）**：用户"打字机流畅度"
3. **吞吐（Throughput）**：服务方"能同时处理多少请求"

> 💡 **类比**：TTFT 像是餐厅上第一道菜的速度，TPOT 像是后续上菜的速度，
> 吞吐像是餐厅同时能服务多少桌客人。

### 数学推导：指标测量方法

**问题设定：**
- TTFT：首 token 延迟
- TPOT：每 token 生成时间
- 吞吐：每秒生成的 token 数

**推导过程：**

```
Step 1: TTFT 测量
  TTFT = t_first_token - t_request_start

  测量方法：
  - 记录请求开始时间 t_request_start
  - 记录第一个 token 生成时间 t_first_token
  - TTFT = t_first_token - t_request_start

Step 2: TPOT 测量
  TPOT = (t_last_token - t_first_token) / (n_tokens - 1)

  测量方法：
  - 记录第一个 token 生成时间 t_first_token
  - 记录最后一个 token 生成时间 t_last_token
  - TPOT = (t_last_token - t_first_token) / (n_tokens - 1)

Step 3: 吞吐测量
  Throughput = total_tokens / total_time

  测量方法：
  - 统计所有请求生成的 token 总数 total_tokens
  - 统计总耗时 total_time
  - Throughput = total_tokens / total_time
```

**关键洞察：**
- TTFT 主要由 prefill 阶段决定（处理输入 token）
- TPOT 主要由 decode 阶段决定（生成输出 token）
- 吞吐取决于批处理大小和 GPU 利用率

## 代码实现

### 1. 指标的可操作定义（测量代码里怎么算）

运行 [scripts/01_naive_generate_baseline.py](../scripts/01_naive_generate_baseline.py) 验证以下代码。

```
TTFT（首 token 延迟）：提交请求 → 第一个 token 到达。prefill 主导，用户"反应快不快"
TPOT（每 token 间隔）：(总时间 - TTFT) / (n_tokens - 1)。decode 主导，"打字机流畅度"
吞吐 throughput     ：全体请求 tok/s 合计（服务方视角）
p50/p90             ：分布式 serving 的真实体验由尾部决定，别只报平均
E2E 延迟            ≈ TTFT + TPOT × (输出 token 数 − 1)
```

### 形状追踪：测量过程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TTFT/TPOT 测量过程                                                         │
│                                                                             │
│  输入: prompt (str)                                                         │
│    ↓ tokenize                                                               │
│  input_ids: (1, seq_len)                                                    │
│    ↓ prefill (处理输入)                                                      │
│  第一个 token 生成 ← 记录 t_first_token                                      │
│    ↓ decode (逐个生成)                                                       │
│  token 2, 3, ... ← 逐个记录时间                                              │
│    ↓ 最后一个 token                                                          │
│  t_last_token ← 记录                                                        │
│                                                                             │
│  计算:                                                                       │
│  TTFT = t_first_token - t_request_start                                     │
│  TPOT = (t_last_token - t_first_token) / (n_tokens - 1)                     │
│  吞吐 = total_tokens / total_time                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

⚠️ 测量陷阱（本脚本全部真实处理过——计时点前后共 7 处 `torch.cuda.synchronize()`）：
- **异步陷阱**：GPU 是异步的——不 `synchronize`/同步读结果，测到的是"提交 kernel"
  而非"算完"的时间（Part 9 01 章）。本脚本的 TTFT 单步探测、逐请求循环、静态批对照
  三处计时都在掐表前后显式同步，**保证测的是完成时刻**（新版 transformers 的
  `generate` 内部已带若干同步点，但依赖内部实现是脆弱的——自己 sync 才是契约）；
- **padding 陷阱**：decoder-only 批处理必须**左 padding**（右 padding 会把位置算歪）；
- **首 token 单独测**：HF `generate` 是一次性调用，TTFT 需要用 `max_new_tokens=1` 的
  独立探测来近似——工程上 serving 引擎会流式返回，天然可测。

### 2. 基线结果（实测）

> 环境：RTX 4090，torch 2.6.0+cu124，transformers 4.57.6，
> Qwen2.5-0.5B-Instruct，64 请求 × 32 token，greedy，全部计时点显式同步。
> （脚本头两行会打印你自己的环境；数字随机器状态略有波动属正常——方向不变。
> 本页早期版本引用过 181 tok/s 等未受控数字，已全部替换为复跑实测值；
> 吞吐口径修正：分母只含 64 次正式 generate 的计时段合计，不含 TTFT 单步探测
> ——早期版本把探测时间计入总时长却不计其 token，吞吐被系统性低估约 5-10%。）

```
[1] 逐请求循环（serving 反模式）:
    TTFT  p50/p90 : 7.5 / 7.6 ms
    TPOT  p50/p90 : 6.2 / 6.3 ms
    吞吐          : 158 tok/s（计时段 12.95s；wall 13.44s 含 TTFT 探测，不作分母）
[2] 静态批处理（batch=8）: 1071 tok/s（吞吐×6.8！）
    —— 但早完成的请求陪跑到最慢的：这就是 Orca 论文要杀死的"static batching 浪费"
```

- 🔑 静态批处理已经赢近 7 倍（6.8×）：**权重只读一次喂 8 个请求**（memory-bound 的直接推论）。
  vLLM 的增量 = 连续批处理（早走早换人）+ PagedAttention（batch 开得更大）+ prefix caching。

### 3. 对比实验设计（02 章的填空表）

| 指标 | naive 循环（本脚本） | vLLM（02 章） | 差异来自 |
|---|---|---|---|
| 吞吐 tok/s | 158（实测） | ? | 连续批处理 |
| TTFT p50 | 7.5 ms（实测） | ? | prefill 调度/chunked prefill |
| TPOT p50 | 6.2 ms（实测） | ? | decode batch 更大 + CUDA graphs |
| KV 显存 | 每请求整块预留 | ? | PagedAttention |

> 公平性三原则：同模型同 dtype、同 prompt 集（脚本内置的固定 64 条）、同 max_new_tokens。
> 换任何一个，数字就不可比。

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：异步陷阱

**症状：**
```python
start = time.time()
output = model.generate(input_ids, max_new_tokens=32)
end = time.time()
print(f"耗时: {end - start:.3f}s")  # 输出: 耗时: 0.001s（错误！）
```

**原因：** GPU 是异步的，`generate` 立即返回，不等待计算完成

**解法：**
```python
start = time.time()
output = model.generate(input_ids, max_new_tokens=32)
torch.cuda.synchronize()  # 等待 GPU 计算完成
end = time.time()
print(f"耗时: {end - start:.3f}s")  # 输出: 耗时: 0.123s（正确）
```

#### 错误 2：padding 陷阱

**症状：**
```python
# 右 padding（错误）
input_ids = [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]

# 左 padding（正确）
input_ids = [[0, 0, 1, 2, 3], [0, 0, 0, 4, 5]]
```

**原因：** decoder-only 模型用因果注意力，右 padding 会把位置算歪

**解法：** 始终使用左 padding

#### 错误 3：TTFT 测量不准

**症状：** TTFT 测量结果与预期不符

**原因：** 用 `max_new_tokens=32` 测量，包含了后续 token 的生成时间

**解法：** 用 `max_new_tokens=1` 单独测量 TTFT

### 性能数据（实测参考）

| 方法 | 吞吐 tok/s | TTFT p50 | TPOT p50 | 说明 |
|------|------------|----------|----------|------|
| 逐请求循环 | 158 | 7.5ms | 6.2ms | serving 反模式（实测） |
| 静态批处理 (batch=8) | 1071 | —（未单独测） | —（未单独测） | 权重只读一次（实测） |
| vLLM (batch=64) | ~3000+ | ~3ms | ~2ms | 连续批处理 + PagedAttention（预期，未本机实测） |

> 📊 数据来源：本课开发机实测（RTX 4090，torch 2.6.0+cu124，transformers 4.57.6，
> Qwen2.5-0.5B，64 请求 × 32 token）；吞吐分母 = 64 次正式 generate 的计时段合计
> （不含 TTFT 探测/tokenize 开销）；vLLM 行为量级预期，02 章自己跑出来回填。

### 常见陷阱

#### 陷阱 1：测量环境不一致

**症状：** 不同次测量结果差异大

**原因：** GPU 频率、后台进程等干扰

**解法：** 测量前清理后台进程，多次测量取平均

#### 陷阱 2：prompt 集不一致

**症状：** 不同 prompt 集的结果不可比

**原因：** prompt 长度、复杂度不同

**解法：** 使用固定的 prompt 集

#### 陷阱 3：模型配置不一致

**症状：** 不同配置的结果不可比

**原因：** dtype、量化方式等不同

**解法：** 使用相同的模型配置

### 最佳实践

#### 测量流程

1. **清理环境**：关闭后台进程，确保 GPU 空闲
2. **预热**：先跑几个请求预热 GPU
3. **多次测量**：至少测量 3 次，取平均
4. **记录配置**：记录模型、prompt、batch size 等配置

#### 公平对比原则

1. **同模型**：相同的模型权重
2. **同 dtype**：相同的精度（fp16/bf16）
3. **同 prompt**：相同的输入数据
4. **同 max_new_tokens**：相同的输出长度

## 学完本部分你能...

- ✅ 用代码正确测出 TTFT/TPOT/吞吐的 p50/p90，避开三个测量陷阱
- ✅ 解释静态批处理为什么已经赢近 7 倍（6.8×）、vLLM 又赢在哪
- ✅ 设计一个公平的 serving 对比实验
- ✅ 识别异步陷阱、padding 陷阱等常见错误

**概念检验**

<details>
<summary>Q1: 为什么 TTFT 的 p90 和 p50 几乎一样（7.6 vs 7.5）？什么场景下会拉开？</summary>

A: 本基线是串行逐请求——每个请求独占 GPU、无排队，TTFT≈恒定的 prefill 时间。
真 serving 在高负载下 TTFT 尾部会被排队+批内干扰拉长（这正是 P99 SLO 和 goodput 存在的
原因）。所以本脚本的 TTFT 是"空载 TTFT"，跟 vLLM 对比时注意负载条件要一致。

</details>

<details>
<summary>Q2: 静态批处理 8 路吞吐 1071 tok/s，是不是继续加 batch 就线性涨？</summary>

A: 在 memory-bound 区间近似线性（权重搬运被摊薄），直到 compute-bound 或显存（KV）耗尽
——4090 上 0.5B 模型 KV 很小，瓶颈先出现在调度/内存拷贝。真实大模型上 KV 显存先爆，
这正是 PagedAttention 的用武之地（02 章日志里 GPU KV cache usage 会印证）。

</details>

<details>
<summary>Q3: 为什么 decoder-only 批处理必须左 padding？</summary>

A: decoder-only 模型用因果注意力，每个 token 只能看到前面的 token。
右 padding 会把 padding token 放在后面，导致模型"看到" padding token，
从而影响生成结果。左 padding 把 padding token 放在前面，模型不会"看到"它们。

</details>

**动手实践**

<details>
<summary>练习 1: 实现 TTFT 测量函数</summary>

**任务：** 实现一个函数，测量首 token 延迟。

**验收标准：**
- [ ] 输入：模型、input_ids、max_new_tokens
- [ ] 输出：TTFT（秒）
- [ ] 正确处理异步陷阱

**步骤提示：**
```python
def measure_ttft(model, input_ids, max_new_tokens=1):
    """
    Steps:
        1. 记录开始时间
        2. 调用 model.generate(max_new_tokens=1)
        3. 同步 GPU
        4. 记录结束时间
        5. 返回 TTFT
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 2: 实现吞吐测量函数</summary>

**任务：** 实现一个函数，测量批量请求的吞吐。

**验收标准：**
- [ ] 输入：模型、prompts、max_new_tokens
- [ ] 输出：吞吐（tok/s）
- [ ] 正确统计总 token 数

**步骤提示：**
```python
def measure_throughput(model, prompts, max_new_tokens=32):
    """
    Steps:
        1. 记录开始时间
        2. 批量调用 model.generate
        3. 同步 GPU
        4. 记录结束时间
        5. 统计总 token 数
        6. 计算吞吐
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 3: 设计公平对比实验</summary>

**任务：** 设计一个公平的 naive vs vLLM 对比实验。

**验收标准：**
- [ ] 列出需要控制的变量
- [ ] 设计测量流程
- [ ] 设计结果展示方式

**步骤提示：**
```python
def design_experiment():
    """
    Steps:
        1. 列出需要控制的变量（模型、dtype、prompt、max_new_tokens）
        2. 设计测量流程（预热、多次测量、取平均）
        3. 设计结果展示方式（表格、图表）
    """
    # TODO: Implement
    pass
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 14 完成练习：

👉 [Assignment 14](../../../assignments/assignment_14/)

## 下一步

装 vLLM（两案选一），把填空表填上。

👉 [02 — vLLM 实战与对比](02_vllm_serving.md)
