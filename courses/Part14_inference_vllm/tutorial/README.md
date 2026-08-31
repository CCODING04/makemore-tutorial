# Part 14: 推理部署实战 — vLLM

> 🧭 Part 8 06 章手写了 PagedAttention 模拟、量化与投机解码的原理；本部分把它们放上
> 工业标准引擎 **vLLM**，做一次真正的 serving 实验：同一模型、同一批 prompt，
> naive 循环 vs vLLM 的 TTFT/TPOT/吞吐对比——"手写 vs 工具"的收官之战。
> 主源：[vllm-project/vllm](https://github.com/vllm-project/vllm)（90.5k，Apache-2.0）

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [朴素基线与对比设计](01_naive_baseline.md) | 手写侧基线：TTFT/TPOT/吞吐的测量方法与指标定义 | `01` |
| 02 | [vLLM 实战与对比](02_vllm_serving.md) | 两案安装 → 离线推理 → OpenAI 服务 → benchmark → 量化/投机解码 | —（CLI 实操） |

## 🧰 前置知识

- **Part 8 06 章**：memory-bound、PagedAttention、投机解码（手写模拟）——本章的"手写侧"
- **Part 9**：GPU 执行模型（为什么连续批处理能赢）

## 🔗 在 LLM 链路中的位置

```
预训练 → 微调/对齐(Part 8/11/12) → 【本部分: 推理与服务（模型变产品）】
```

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

👉 [Assignment 14](../../../assignments/assignment_14/)

## 🔗 相关资源

- 🐙 [vLLM](https://github.com/vllm-project/vllm)（docs.vllm.ai + examples/ 树是最好的教程）
- 📄 PagedAttention（arXiv 2309.06180）· Orca（OSDI'22）
- 🐙 [SGLang](https://github.com/sgl-project/sglang)（32.9k，对照引擎）· [llama.cpp](https://github.com/ggml-org/llama.cpp)（端侧/GGUF）

---

[← 上一章：Part 13 数据工程](../../Part13_data_engineering/tutorial/README.md) | [下一章：Part 15 多模态理解 →](../../Part15_vision_language/tutorial/README.md)
