# 🌱 Makemore 中文教程

> 基于 Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列
> 从零构建字符级语言模型，逐步深入神经网络核心概念

---

## 📖 学习路线

```
Part 1: Bigrams          ─── 最简单的语言模型（频率计数 → 概率 → 神经网络）
  ↓
Part 2: MLP              ─── 多层感知机（Embedding + 隐藏层 + 反向传播）
  ↓
Part 3: BatchNorm        ─── 训练诊断与优化（初始化 + BN + 深层网络）
  ↓
Part 4: Backpropagation  ─── 手动反向传播（逐层推导梯度，理解 autograd 原理）
  ↓
Part 5: WaveNet          ─── 层次化架构（PyTorch 化代码 + WaveNet + 卷积预览）
  ↓
Part 6: Transformer/GPT  ─── 从零构建 decoder-only Transformer（Attention + 迷你 ChatGPT）
  ↓
Part 7: Minimind 复现    ─── 从零复现现代 LLM（BPE + RMSNorm + RoPE + GQA + SwiGLU + MoE + Pretrain→SFT→DPO）
  ↓
Part 8: 后训练全流程     ─── 从零训练 LLM（GPT-2 架构 → Pretrain → SFT → Reward → DPO/PPO/GRPO）
  ↓
Part 9: CUDA 内核编程    ─── 打开深度学习的引擎盖（GPU 架构 + 手写 matmul 优化阶梯 + Triton + PyTorch 扩展）
  ↓
Part 10: 分布式训练      ─── 从单卡到集群（DDP + ZeRO/FSDP + 张量并行 + 流水线并行）
  ↓
Part 11: 对齐实战        ─── verl 工业级 GRPO（手写原理 → 工业框架，Docker 起步）
  ↓
Part 12: 微调实战        ─── LLaMA-Factory（手写 LoRA SFT → 工具：QLoRA 7B → DPO-LoRA）
  ↓
Part 13: 数据工程        ─── 手写 MinHash/LSH 去重 → Data-Juicer 工业管线
  ↓
Part 14: 推理部署        ─── vLLM（naive 基线 vs 工业引擎的 TTFT/TPOT/吞吐对比）
```

| Part | 主题 | 核心概念 | 教程入口 | 原始视频 |
|------|------|----------|----------|----------|
| 1 | Bigrams | 频率矩阵、Softmax、NLL 损失、梯度下降 | [📖 开始学习](courses/Part1_bigrams/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| 2 | MLP | Embedding、多层感知机、Minibatch SGD、Train/Dev/Test | [📖 开始学习](courses/Part2_mlp/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I) |
| 3 | BatchNorm | 激活诊断、Kaiming 初始化、BatchNorm、诊断工具 | [📖 开始学习](courses/Part3_batchnorm/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=P6sfmUTpUmc) |
| 4 | Backpropagation | 链式法则、手动梯度、CrossEntropy 反传、BN 反传 | [📖 开始学习](courses/Part4_backprop/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=q8SA3rM6ckI) |
| 5 | WaveNet | Sequential 容器、层次融合、FlattenConsecutive、卷积 | [📖 开始学习](courses/Part5_wavenet/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) |
| 6 | Transformer/GPT | 字符级 Tokenizer、Self-Attention、Multi-Head、残差、LayerNorm、decoder-only GPT | [📖 开始学习](courses/Part6_transformer/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 7 | Minimind 复现 | BPE 分词器、RMSNorm、RoPE、GQA、KV Cache、SwiGLU、MoE、Pretrain→SFT→DPO 流水线 | [📖 开始学习](courses/Part7_minimind/tutorial/README.md) | [minimind](https://github.com/jingyaogong/minimind) |
| 8 | 后训练全流程 | GPT-2 架构、AdamW、bf16 混合精度、SFT Prompt Masking、Bradley-Terry、DPO/ORPO/KTO、PPO/GAE、GRPO/RLVR | [📖 开始学习](courses/Part8_post_training/tutorial/README.md) | [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) |
| 9 | CUDA 内核编程 | GPU 架构、线程层级、手写 matmul 优化阶梯（coalesced/SMEM/block tiling vs cuBLAS）、atomics/streams、Triton、PyTorch 自定义扩展 | [📖 开始学习](courses/Part9_cuda_kernels/tutorial/README.md) | [cuda-course](https://github.com/infatoshi/cuda-course) |
| 10 | 分布式训练 | 集合通信原语、DDP（桶化 all-reduce/no_sync）、显存账本与 ZeRO 三阶段、FSDP、Megatron 式张量并行、GPipe/1F1B 流水线、3D 并行 | [📖 开始学习](courses/Part10_distributed/tutorial/README.md) | [minGPT-ddp](https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp) · [nanotron](https://github.com/huggingface/nanotron) |
| 11 | 对齐实战 | RLVR 奖励函数、组内优势/KL 手写 → verl 概念映射（HybridEngine/三角色/Ray）→ 0.5B GRPO Docker 实战 → 双卡 | [📖 开始学习](courses/Part11_alignment_verl/tutorial/README.md) | [verl](https://github.com/verl-project/verl) · [slime](https://github.com/THUDM/slime) |
| 12 | 微调实战 | 手写 LoRA SFT 管线（yaml 字段逐行对照）→ LLaMA-Factory：identity → WebUI → QLoRA 7B → export → DPO-LoRA | [📖 开始学习](courses/Part12_finetune_llamafactory/tutorial/README.md) | [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) · [unsloth](https://github.com/unslothai/unsloth) |
| 13 | 数据工程 | 手写 MinHash+LSH 去重（LSH 概率性质）→ Data-Juicer YAML 管线、算子全家桶与审计 → FineWeb 对照 | [📖 开始学习](courses/Part13_data_engineering/tutorial/README.md) | [Data-Juicer](https://github.com/datajuicer/data-juicer) · [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) |
| 14 | 推理部署 | TTFT/TPOT/吞吐测量（naive 基线）→ vLLM 两案安装 → 服务/benchmark/量化/n-gram 投机解码 → 三行对比表 | [📖 开始学习](courses/Part14_inference_vllm/tutorial/README.md) | [vLLM](https://github.com/vllm-project/vllm) |

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 安装

```bash
# 使用 uv（推荐）
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

### 数据

数据集已包含在 `data/` 目录：
- `names.txt`（32,032 个美国人名）— Part 1-5 使用
- `input.txt`（tiny Shakespeare 全文，~1.1M 字符）— Part 6、Part 7 和 Part 8 使用
- Part 9 不需要数据文件（脚本内合成数据），但需要 **NVIDIA GPU + CUDA Toolkit**，环境自检见 [Part 9 README](courses/Part9_cuda_kernels/tutorial/README.md)

### 学习方式

每课包含三个部分：

1. **📖 Tutorial** — `courses/PartX/tutorial/` — 中文讲解，概念 + 代码 + 配图
2. **💻 Scripts** — `courses/PartX/scripts/` — 渐进式可运行代码，跟着教程一步步跑
3. **📝 Assignment** — `assignments/assignment_X/` — 练习题，带自动测试

```bash
# 示例：运行 Part 1 的第一个脚本
cd courses/Part1_bigrams/scripts
python 01_explore_data.py
```

### 推荐学习顺序

```bash
# 1. 阅读教程（按顺序）
courses/Part1_bigrams/tutorial/README.md
courses/Part2_mlp/tutorial/README.md
courses/Part3_batchnorm/tutorial/README.md
courses/Part4_backprop/tutorial/README.md
courses/Part5_wavenet/tutorial/README.md
courses/Part6_transformer/tutorial/README.md
courses/Part7_minimind/tutorial/README.md
courses/Part8_post_training/tutorial/README.md
courses/Part9_cuda_kernels/tutorial/README.md
courses/Part10_distributed/tutorial/README.md
courses/Part11_alignment_verl/tutorial/README.md
courses/Part12_finetune_llamafactory/tutorial/README.md
courses/Part13_data_engineering/tutorial/README.md
courses/Part14_inference_vllm/tutorial/README.md

# 2. 运行脚本（每个 Part 的 scripts/ 目录）
#    Part 9 的 CUDA 脚本需要先编译：cd courses/Part9_cuda_kernels/scripts && make
#    Part 10 的多卡体验：torchrun --standalone --nproc_per_node=2 <脚本>（单进程也能跑）
# 3. 完成作业（每个 Part 的 assignments/ 目录）
# 4. 运行测试验证
pytest assignments/  # 运行所有测试
```

### 如何判断学完一个 Part

✅ **完成标准**：
- [ ] 能解释本 Part 的核心概念（不看笔记）
- [ ] 能独立运行所有 scripts
- [ ] 能完成 assignment 并通过测试
- [ ] 能回答思考题
- [ ] 能向别人解释（费曼检验）

### 推荐前置课程

本教程假设你了解反向传播的基本概念。如果你对反向传播不熟悉，推荐先学习：

- **Micrograd** — Andrej Karpathy 的反向传播入门课
  - 📺 [YouTube 视频](https://www.youtube.com/watch?v=VMj-3S1tku0)
  - 📁 [GitHub 仓库](https://github.com/karpathy/micrograd)
  - 时长约 2.5 小时，从零实现一个自动微分引擎

> 💡 Micrograd 不是必须的，但学完后你会对 Part 4（手动反向传播）有更好的理解。

### 课程范围说明

本仓库覆盖 Karpathy 的 **makemore 系列**（Part 1-6），并额外补充三个现代实战：
- **Part 1-5** — Bigrams / MLP / BatchNorm / Backpropagation / WaveNet
- **Part 6** — **GPT from Scratch**（Transformer 架构）：从零实现一个 decoder-only Transformer，对应原视频全部知识要点
- **Part 7** — **Minimind 复现**（非 Karpathy 原课）：从零复现 minimind 的六大核心组件（BPE / RMSNorm / RoPE / GQA / KV Cache / SwiGLU / MoE），并跑通 Pre-train → SFT → DPO 完整训练流水线
- **Part 8** — **后训练全流程**（非 Karpathy 原课）：从零训练 LLM 的完整生命周期——构建 GPT-2 → 预训练 → SFT → 奖励模型 → DPO/ORPO/KTO → PPO/GRPO，参考 [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)
- **Part 9** — **CUDA 内核编程**（非 Karpathy 原课）：GPU 架构与线程层级、手写 matmul 优化阶梯（对照 cuBLAS）、atomics/streams/profiling、Triton 与 PyTorch 自定义扩展，参考 [cuda-course](https://github.com/infatoshi/cuda-course)。需要 NVIDIA GPU（无 GPU 也可学概念 + 完成 CPU 作业题）
- **Part 10** — **分布式训练**（非 Karpathy 原课）：集合通信与 DDP、显存账本与 ZeRO/FSDP、Megatron 式张量并行、GPipe/1F1B 流水线并行，参考 [minGPT-ddp](https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp) 与 [nanotron](https://github.com/huggingface/nanotron)。全部脚本单进程可跑，多卡体验需 ≥2 GPU 或 CPU 多进程
- **Part 11-14** — **工业实战四部曲**（非 Karpathy 原课，"手写 → 工具"双轨教学）：对齐实战（verl）、微调实战（LLaMA-Factory）、数据工程（Data-Juicer）、推理部署（vLLM）——每章先手写核心原理再用工具放大，作业与 README 均含"面试直通车"。安装摩擦与版本策略见各章 README

不包含（makemore 系列之外）：
- **Micrograd** — 反向传播基础（独立课程，推荐前置）

如果你想继续深入，这些课程都在 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 中。

---

## 📝 如何做作业

每个 Part 学完后都有配套作业来巩固理解，流程如下：

1. **学完教程** — 完成某个 Part 的 tutorial 阅读和 scripts 运行
2. **进入作业目录** — 对应 `assignments/assignment_X/`（X = Part 编号）
3. **阅读 README.md** — 了解作业要求和背景说明
4. **完成练习** — 在 `xxx_exercises.py` 中找到 `TODO` 标记，填写你的实现
5. **验证答案** — 运行 `test_xxx.py`，确保所有测试通过

```bash
# 示例：完成 Part 1 作业
cd assignments/assignment_1
cat README.md                  # 阅读作业要求
vim bigram_exercises.py        # 完成 TODO 部分
python test_bigram_exercises.py  # 运行测试验证
```

---

## 📁 项目结构

```
makemore-tutorial/
├── README.md                 # 本文件
├── data/
│   └── names.txt             # 训练数据（32K 人名）
├── tools/
│   └── extract_images.py     # 从 notebook 提取图片的工具
├── skills/
│   └── tutorial-creator/     # 用于创建此类教程的 Agent Skill（见下方说明）
├── courses/
│   ├── Part1_bigrams/
│   │   ├── makemore_part1_bigrams.ipynb  # 原始 notebook
│   │   ├── images/                       # 教程配图
│   │   ├── scripts/                      # 可运行脚本
│   │   └── tutorial/                     # 中文教程（从 README.md 开始）
│   ├── Part2_mlp/
│   ├── Part3_batchnorm/
│   ├── Part4_backprop/
│   ├── Part5_wavenet/
│   ├── Part6_transformer/
│   ├── Part7_minimind/
│   ├── Part8_post_training/
│   ├── Part9_cuda_kernels/
│   ├── Part10_distributed/
│   ├── Part11_alignment_verl/
│   ├── Part12_finetune_llamafactory/
│   ├── Part13_data_engineering/
│   └── Part14_inference_vllm/
└── assignments/
    ├── assignment_1/         # Part 1 作业
    │   ├── README.md         # 作业说明
    │   ├── bigram_exercises.py   # TODO 骨架
    │   └── test_bigram_exercises.py  # 自动测试
    ├── assignment_2/
    ├── assignment_3/
    ├── assignment_4/
    ├── assignment_5/
    ├── assignment_6/
    ├── assignment_7/
    ├── assignment_8/
    ├── assignment_9/
    ├── assignment_10/
    ├── assignment_11/
    ├── assignment_12/
    ├── assignment_13/
    └── assignment_14/
```

---

## 🤖 skills/tutorial-creator

`skills/` 目录包含用于**自动生成此类教程**的 Agent Skill：

- **tutorial-creator** — 一个 OpenClaw skill，能从 Jupyter Notebook 自动生成结构化的中文教程（tutorial Markdown）、渐进式脚本（scripts）和带自动测试的作业（assignments）
- 适用于想要复刻此教程模式来覆盖其他课程或项目的场景
- 详见 `skills/tutorial-creator/SKILL.md`

---

## 📝 作业参考答案

- [assignment_reference/](assignment_reference/README.md) — **Assignment 1-14 的参考答案**，每份都在本课程环境实测通过（含验证状态表）。⚠️ 先自己做再看答案。
- [docs/datasets.md](docs/datasets.md) — 大尺寸数据集下载指南（minimind 语料 / The Pile / Alpaca / HH-RLHF / GSM8K / FineWeb / 模型权重）：页面、命令、体积、格式介绍。

## 💼 面试备战

- [docs/llm_interview_guide.md](docs/llm_interview_guide.md) — LLM 算法岗面试指南：Boss 直聘 JD 需求 × 课程章节映射（✅/🟡/❌）、讲故事链复习法、本课实测硬数字清单、手写代码 TOP8、以及按岗位方向的补课路线（含优质开源仓库推荐）。

---

## 🙏 致谢

- [Andrej Karpathy](https://karpathy.ai/) — 原始课程和代码
- [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — 原始仓库
- [makemore](https://github.com/karpathy/makemore) — makemore 项目仓库

---

## 📄 License

本教程的中文讲解内容为原创，代码部分遵循原始仓库的 MIT License。
