# Part 14: 推理部署实战 — vLLM

> 🧭 Part 8 06 章手写了 PagedAttention 模拟、量化与投机解码的原理；本部分把它们放上
> 工业标准引擎 **vLLM**，做一次真正的 serving 实验：同一模型、同一批 prompt，
> naive 循环 vs vLLM 的 TTFT/TPOT/吞吐对比——"手写 vs 工具"的收官之战。
> 主源：[vllm-project/vllm](https://github.com/vllm-project/vllm)（90.5k，Apache-2.0）

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 推理部署在 LLM 链路中的位置和价值
- ✅ **手写** 朴素推理基线并测量 TTFT/TPOT/吞吐
- ✅ **解释** vLLM 的核心优化（连续批处理、PagedAttention）
- ✅ **配置** vLLM 的推理服务并理解每个参数的含义
- ✅ **完成** 朴素基线 vs vLLM 的性能对比
- ✅ **识别** 推理部署中的常见陷阱并设计防范策略

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [朴素基线与对比设计](01_naive_baseline.md) | 手写侧基线：TTFT/TPOT/吞吐的测量方法与指标定义 | `01` |
| 02 | [vLLM 实战与对比](02_vllm_serving.md) | 两案安装 → 离线推理 → OpenAI 服务 → benchmark → 量化/投机解码 | —（CLI 实操） |

## 🧰 前置知识

**必须掌握：**
- **Part 8 06 章**：memory-bound、PagedAttention、投机解码（手写模拟）——本章的"手写侧"
- **Part 9**：GPU 执行模型（为什么连续批处理能赢）

**建议掌握：**
- **Part 7**：Transformer 架构（理解推理过程）

**可选：**
- **Part 10**：分布式推理（多卡推理用到）

## 🔗 在 LLM 链路中的位置

```
预训练 → 微调/对齐(Part 8/11/12) → 【本部分: 推理与服务（模型变产品）】
                                      ↑
                                      你在这里
```

**为什么推理部署是"模型变产品"的关键：**

| 证据 | 说明 |
|------|------|
| 成本 | 推理成本占 LLM 总成本的 70%+ |
| 速度 | 用户体验取决于首 token 延迟（TTFT）和生成速度（TPOT） |
| 规模 | 需要处理并发请求，而非单个请求 |

## 理论背景

### 问题引入：为什么需要推理优化？

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

### 数学推导：TTFT/TPOT/吞吐

**问题设定：**
- TTFT（Time to First Token）：首 token 延迟
- TPOT（Time Per Output Token）：每 token 生成时间
- 吞吐（Throughput）：每秒生成的 token 数

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

### 历史脉络：推理优化演进

```
2018: 朴素推理（逐请求处理）
  ↓ 显存浪费，计算浪费
2020: 静态批处理（Static Batching）
  ↓ 早完成的请求等待最慢的请求
2022: 连续批处理（Continuous Batching，Orca）
  ↓ 动态添加/移除请求
2023: PagedAttention（vLLM）
  ↓ 显存分页，减少碎片
2024: 投机解码（Speculative Decoding）
  ↓ 用小模型加速大模型推理
```

**关键论文：**
- PagedAttention: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Orca: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- vLLM: [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## 📦 环境与版本策略（⚠️ 全课程最重要的安装决策）

| 方案 | 版本 | 适合 | 代价 |
|---|---|---|---|
| **A（推荐）** | 独立 venv + **vLLM latest**（0.28.0，pin torch 2.13） | 教学时效最好 | ~5GB 下载，与课程 venv 隔离 |
| B（复用课程 venv） | `vllm==0.6.6`（恰好 pin torch==2.5.1+cu121） | 不想再建 venv | 已知 flash-attn 解析冲突（issue #11283），概念演示够用 |

```bash
# 方案 A：
uv venv .venv-vllm && source .venv-vllm/bin/activate
pip install vllm                 # CUDA 12.x wheel；4090(sm89) 完整支持
python -c "import vllm; print(vllm.__version__)"
```

| 你有什么 | 能做什么 |
|---|---|
| CPU only | 脚本 01 的基线思想可读；vLLM 部分用 Colab（免费 T4） |
| 1×4090 | 全部内容（0.5B 模型显存无压力；GPTQ/AWQ 4bit 与 n-gram 投机解码都可玩） |

## 📈 学习地图

```
指标定义 + naive 基线（01：手写侧）     ← 点
   ↓ 同模型同 prompt 同指标
vLLM 离线 → 服务 → benchmark（02）      ← 线 → 面
   ↓ 量化服务 / n-gram 投机解码
三行对比表填空完成                       →  面试即用的实证
```

## 📝 课后作业

每章末尾有思考题（`<details>` 折叠答案）。全部学完后：

👉 [Assignment 14](../../../assignments/assignment_14/)

## 🔗 相关资源

- 🐙 [vLLM](https://github.com/vllm-project/vllm)（docs.vllm.ai + examples/ 树是最好的教程）
- 📄 PagedAttention（arXiv 2309.06180）· Orca（OSDI'22）
- 🐙 [SGLang](https://github.com/sgl-project/sglang)（32.9k，对照引擎）· [llama.cpp](https://github.com/ggml-org/llama.cpp)（端侧/GGUF）

---

[← 上一章：Part 13 数据工程](../../Part13_data_engineering/tutorial/README.md) | [下一章：Part 15 多模态理解 →](../../Part15_vision_language/tutorial/README.md)
