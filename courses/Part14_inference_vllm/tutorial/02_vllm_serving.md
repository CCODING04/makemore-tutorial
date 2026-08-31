# 02 — vLLM 实战：离线推理 → 服务 → benchmark → 量化/投机解码

> 🧭 填空时间。装 vLLM（README 两案），同一模型（Qwen2.5-0.5B-Instruct）、同一批 prompt，
> 把 01 章的对比表填完。每一步都标注"对应 Part 8 06 章手写的哪一块"。

## 学习目标

完成本章后，你将能够：

- ✅ **完成** vLLM 安装（两案）、离线推理、OpenAI 服务部署
- ✅ **用** 官方 benchmark 出 TTFT/TPOT 分位数并填完对比表
- ✅ **部署** 量化模型与 n-gram 投机解码，解释各自的适用条件
- ✅ **把** 课程的手写模块逐一对应到工业实现，形成"懂原理 + 会工具"的完整叙事

## 📖 前置知识

**必须掌握：**
- **01 章**：指标定义与 naive 基线（181 tok/s / 6.3ms / 5.2ms，待超越）
- **Part 8 06 章**：PagedAttention/连续批处理/投机解码的手写模拟

## 理论背景

### 问题引入：为什么需要 vLLM？

朴素推理虽然能跑通，但有三个根本限制：

1. **显存浪费**：每个请求独立分配显存，导致碎片化
2. **计算浪费**：早完成的请求等待最慢的请求，导致 GPU 空转
3. **延迟过高**：没有批处理，每个请求单独计算

vLLM 通过**连续批处理 + PagedAttention**来弥补：

```
朴素推理:  "每个请求独立处理，显存碎片化，计算浪费"
vLLM:      "连续批处理，显存分页，高吞吐低延迟"
```

> 💡 **类比**：朴素推理像是每个顾客单独结账，vLLM 像是超市收银台。
> 收银台可以同时处理多个顾客，效率更高。

### 数学推导：PagedAttention 的显存优化

**问题设定：**
- 朴素推理：每个请求独立分配 KV cache 显存
- PagedAttention：KV cache 分页管理

**推导过程：**

```
Step 1: 朴素推理的显存浪费
  每个请求分配 max_seq_len 的 KV cache
  实际使用 only_used_seq_len
  浪费 = max_seq_len - used_seq_len

  示例：max_seq_len=2048, used_seq_len=512
  浪费 = 2048 - 512 = 1536 tokens
  浪费率 = 1536 / 2048 = 75%

Step 2: PagedAttention 的分页管理
  把 KV cache 分成固定大小的 block
  按需分配 block，而非预分配 max_seq_len
  浪费 = 最后一个 block 的内部碎片

  示例：block_size=16, used_seq_len=512
  需要 block 数 = ceil(512 / 16) = 32
  浪费 = 16 - (512 % 16) = 16 - 0 = 0 tokens
  浪费率 ≈ 0%
```

**关键洞察：**
- PagedAttention 把显存碎片化问题转化为分页问题
- block_size 越小，碎片越少，但管理开销越大
- 实践中 block_size=16 是常见配置

## 代码实现

### 1. 离线推理（5 行）

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")     # 启动日志找 'GPU KV cache size'
outs = llm.generate(["The quick brown fox", "Deep learning models are"],
                    SamplingParams(temperature=0.7, max_tokens=64))
print(outs[0].outputs[0].text)
```

启动日志三行必看：`# GPU blocks`（KV 池的块数——**PagedAttention 的页表容量**）、
`max seq len`、`GPU KV cache usage`（运行时占用率）。对照手写：Part 8 06 章模拟里
"整块预留浪费 41%"的问题，这里因为分页只剩 ~5%（论文口径 <4%）。

### 2. OpenAI 兼容服务

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

概念映射：`--max-model-len` = KV 池上限；连续批处理自动开启（无需配置）；
`--gpu-memory-utilization` = KV 池占比调参；`--enable-prefix-caching`（新版默认开）=
共享前缀免重算（Part 8 06 章"prefix sharing"）。

### 3. Benchmark（官方脚本，01 章基线的同款指标）

```bash
# 服务端起好后：
python benchmarks/benchmark_serving.py --backend vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random --num-prompts 64 \
  --request-rate inf      # 吞吐、TTFT/TPOT 的 p50/p99 一次出齐
```

**填空表参考形态**（4090 实测量级，你的数字会不同——这正是要自己跑的原因）：
吞吐从 naive 的 ~181 tok/s 到 **数千 tok/s**（0.5B 小模型上 10×+；7B 上同样量级收益），
TPOT 反而可能略升（batch 大了 decode 稍慢）但吞吐大涨——**serving 的本质是吞吐换延迟**。

### 4. 量化服务（Part 8 06 章"手写量化"的工业对应）

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4   # GPTQ/AWQ/FP8 权重直接加载
# 对比：模型文件体积（fp16 ~1GB → int4 ~0.4GB）、KV 不变、吞吐与精度的实测变化
```

### 5. n-gram 投机解码（无需 draft 模型——24GB 卡友好）

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct",
          speculative_config={"method": "ngram", "prompt_lookup_num_tokens": 4})
# ngram 投机 = prompt lookup：从 prompt 已有文本里检索匹配片段当草稿 token，
# 所以不需要 draft 模型；prompt_lookup_num_tokens = 每步草稿长度
# 对比开/关投机解码的吞吐；Part 8 09 的"接受率 α"概念 = vLLM 日志里的 acceptance rate
```

### 6. 总账：本课程"手写 → 工具"的完整对照

| 手写（Part 8/9） | vLLM/工业 | 你能做的验证 |
|---|---|---|
| KV cache 字典（P7） | PagedAttention 块表 | 日志 KV usage + 显存对比 |
| 分页模拟（P8 09：41%→5%） | 真实 <4% | 同 batch 下 KV 显存 |
| 手写量化（P8 09） | GPTQ/AWQ 权重 | 文件体积 + ppl/acc |
| 手写投机解码（P8 09，α≈0.60） | n-gram/EAGLE | acceptance rate + 吞吐 |
| naive/静态批基线（P14 01） | 连续批处理 | 三行对比表 |

> 🔑 面试结论模板："我在 4090 上用 Qwen2.5-0.5B 做过 naive→vLLM 的对比，吞吐 181→N tok/s，
> 差异归因于连续批处理和 PagedAttention——我的手写模拟复现了同样的方向性。"——
> 这段话的每个数字你都能现场推导。

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：vLLM 安装失败

**症状：**
```
ERROR: Could not find a version that satisfies the requirement vllm
```

**原因：** Python 版本不对，或 CUDA 版本不兼容

**解法：**
```bash
# 检查 Python 版本
python --version  # 需要 3.8+

# 检查 CUDA 版本
nvcc --version  # 需要 11.8+

# 使用正确的版本
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

#### 错误 2：显存不足

**症状：**
```
CUDA out of memory. Tried to allocate 2.00 MiB
```

**原因：** 模型太大，或 KV cache 太大

**解法：**
```bash
# 减小 max-model-len
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 1024

# 或减小 gpu-memory-utilization
vllm serve Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.8
```

#### 错误 3：服务启动失败

**症状：**
```
Error: Address already in use
```

**原因：** 端口被占用

**解法：**
```bash
# 查找占用端口的进程
lsof -i :8000

# 杀掉进程
kill -9 <PID>

# 或使用其他端口
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001
```

### 性能数据（实测参考）

| 方法 | 吞吐 tok/s | TTFT p50 | TPOT p50 | 显存占用 |
|------|------------|----------|----------|----------|
| 逐请求循环 | 181 | 6.3ms | 5.2ms | ~2GB |
| 静态批处理 (batch=8) | 1158 | 6.3ms | 5.2ms | ~4GB |
| vLLM (batch=64) | ~3000+ | ~3ms | ~2ms | ~3GB |
| vLLM + 量化 | ~4000+ | ~2ms | ~1.5ms | ~1.5GB |

> 📊 数据来源：本课开发机实测（RTX 4090，Qwen2.5-0.5B，64 请求 × 32 token）

### 常见陷阱

#### 陷阱 1：版本不兼容

**症状：** 安装失败，或运行时报错

**原因：** Python/CUDA/PyTorch 版本不兼容

**解法：** 使用官方推荐的版本组合

#### 陷阱 2：显存估算不准

**症状：** 服务启动后 OOM

**原因：** 没有考虑 KV cache 的显存开销

**解法：** 使用 vLLM 的显存估算功能，或手动计算

#### 陷阱 3：量化模型精度下降

**症状：** 量化后模型效果变差

**原因：** 量化方式不合适，或量化参数不对

**解法：** 尝试不同的量化方式（GPTQ/AWQ/FP8），调整量化参数

### 最佳实践

#### 配置推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max-model-len` | 2048-4096 | 根据任务调整 |
| `gpu-memory-utilization` | 0.9 | 显存利用率 |
| `enable-prefix-caching` | true | 共享前缀免重算 |
| `quantization` | awq/gptq | 量化方式 |
| `speculative_config` | ngram | 投机解码 |

#### 调试流程

1. **先用小模型**：0.5B 模型快速验证
2. **检查日志**：查看 KV cache 使用率、batch size 等
3. **逐步增大**：从 0.5B 到 7B，从 batch 1 到 batch 64
4. **监控显存**：使用 nvidia-smi 监控显存使用

## 学完本部分你能...

- ✅ 独立完成 vLLM 安装（两案）、离线推理、OpenAI 服务部署
- ✅ 用官方 benchmark 出 TTFT/TPOT 分位数并填完对比表
- ✅ 部署量化模型与 n-gram 投机解码，解释各自的适用条件
- ✅ 把课程的手写模块逐一对应到工业实现，形成"懂原理 + 会工具"的完整叙事

**概念检验**

<details>
<summary>Q1: vLLM 的 TPOT 有时比 naive 略高，为什么还用它？</summary>

A: 大 batch 下 decode 步变慢（每步算更多请求），但吞吐（tok/s 合计）大增——
serving 优化的是"每瓦特/每卡的 token 成本"。单请求延迟敏感场景用小 batch/SLO 路由，
吞吐场景用大 batch——goodput 的含义（Part 8 06 章）。

</details>

<details>
<summary>Q2: prefix caching 什么时候收益最大？什么时候没收益？</summary>

A: 共享前缀长且重复率高（系统提示词、few-shot 模板、多轮对话历史）时收益巨大
（TTFT 降几倍）；prompt 完全随机时纯开销（查表成本）。看业务 prompt 分布决定开关。

</details>

<details>
<summary>Q3: 量化模型和原模型的精度差距有多大？</summary>

A: 取决于量化方式和模型大小：
- 4bit GPTQ/AWQ：精度下降 1-3%，显存节省 75%
- 8bit FP8：精度下降 <1%，显存节省 50%
- 小模型（<1B）量化后精度下降更明显

</details>

**动手实践**

<details>
<summary>练习 1: 部署 vLLM 服务</summary>

**任务：** 部署一个 vLLM 服务并测试。

**验收标准：**
- [ ] 成功安装 vLLM
- [ ] 成功启动服务
- [ ] 成功发送请求并获取响应

**步骤提示：**
```bash
# 1. 安装 vLLM
pip install vllm

# 2. 启动服务
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048

# 3. 发送请求
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

</details>

<details>
<summary>练习 2: 运行 benchmark</summary>

**任务：** 运行 vLLM 官方 benchmark 并记录结果。

**验收标准：**
- [ ] 成功运行 benchmark
- [ ] 记录 TTFT/TPOT/吞吐
- [ ] 与 01 章基线对比

**步骤提示：**
```bash
# 运行 benchmark
python benchmarks/benchmark_serving.py --backend vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random --num-prompts 64 \
  --request-rate inf

# 记录结果
# 吞吐: ??? tok/s
# TTFT p50: ??? ms
# TPOT p50: ??? ms
```

</details>

<details>
<summary>练习 3: 测试量化模型</summary>

**任务：** 部署量化模型并对比精度和性能。

**验收标准：**
- [ ] 成功部署量化模型
- [ ] 记录显存占用
- [ ] 记录吞吐变化
- [ ] 测试精度变化

**步骤提示：**
```bash
# 部署量化模型
vllm serve Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4

# 记录显存占用
nvidia-smi

# 记录吞吐变化
python benchmarks/benchmark_serving.py --backend vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 --dataset-name random --num-prompts 64
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 14 完成练习：

👉 [Assignment 14](../../../assignments/assignment_14/)

## 🎓 课程毕业

Part 1-14 完整覆盖：从手写 bigram 到工业 RL 与 serving。
回 [课程总览](../../../README.md) 规划你的 Part 15（多模态）与面试（docs/llm_interview_guide.md）。
