# Part 10: 分布式训练 — 从单卡到多卡

> 🧭 Part 8 训练的后训练模型、Part 9 打开的 GPU 引擎盖——都建立在一个前提上：**单卡装得下**。
> 真实的 LLM 训练动辄几百上千张卡。这一部分回答：多卡为什么不是"买个交换机插上就行"，
> DDP/ZeRO/FSDP/张量并行/流水线并行各自解决什么问题、代价是什么。
> 参考来源：[pytorch/examples minGPT-ddp](https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp)、
> PyTorch DDP/FSDP 官方系列、[huggingface/nanotron](https://github.com/huggingface/nanotron) 与
> [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [为什么并行 + 分布式 Hello World](01_why_and_collectives.md) | 并行分类、rank/world_size 心智模型、集合通信原语、torchrun 报错 FAQ | `01` |
| 02 | [DDP：数据并行深入](02_ddp.md) | 梯度平均、桶化 all-reduce、DistributedSampler、no_sync、吞吐实测 | `02` |
| 03 | [显存账本与 ZeRO/FSDP](03_memory_zero_fsdp.md) | 16Ψ 公式、ZeRO 1/2/3、FSDP 实战与显存对比 | `03` `04` |
| 04 | [张量并行、流水线并行与工业栈](04_tp_pp_and_beyond.md)（进阶可选） | Megatron f/g 算子、GPipe/1F1B、bubble 公式、3D 并行、LLaMA 训练配置 | `05` `06` |

## 🧰 前置知识

- **Part 8**：训练循环、AdamW、梯度累积（DDP 章直接建立在其上）
- **Part 9 01 章**：GPU 执行模型（知道"内核异步"即可）
- **不需要**：任何分布式经验——01 章从零建立心智模型

## 🗺️ 学习路线图

```
Part 8/9（单卡：模型装得下、算得动）
    │
    │  "模型 70B 装不下 / 数据大到单卡喂不饱 / 想要线性加速"
    ▼
┌────────────────────────────────────────────────────────────────┐
│  Part 10: 分布式训练                                            │
│                                                                │
│  ① 心智模型 + 集合通信   — rank/collective/torchrun           │──→ 01_why_and_collectives.md
│  ② DDP                  — 梯度平均 + 通信重叠（数据并行）     │──→ 02_ddp.md
│  ③ 显存账本 + ZeRO/FSDP — 把"模型状态"切开来存               │──→ 03_memory_zero_fsdp.md
│  ④ TP / PP（进阶）       — 把"一层计算"切开算 / 按层流水      │──→ 04_tp_pp_and_beyond.md
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 📦 环境要求与"无多卡学习路径"

全部脚本**单进程可直接跑**（自动退化为 world_size=1 的 gloo，CPU 也行）；
多卡验证用 `torchrun --standalone --nproc_per_node=2 脚本名`。

| 你有什么 | 能做什么 |
|---|---|
| 只有 CPU | 脚本 01/02/03/05/06 全部可跑（gloo，多进程也支持：`torchrun --nproc_per_node=2` 不要求 GPU！） |
| 1 张 GPU | 以上全部 + 04 的 FSDP（单卡分片，显存对比意义有限但能跑通） |
| ≥2 张 GPU | 完整体验：02 的吞吐对比、04 的跨卡分片、05/06 的切分验证 |
| 0 GPU 且想体验多卡 | **CPU 多进程同样演示分布式语义**：`torchrun --standalone --nproc_per_node=2 01_xxx.py`（gloo 后端）——并行逻辑与 GPU 完全一致，只是算得慢 |

本课开发机验证环境：2× RTX 4090 + torch 2.5.1 + NCCL。所有"验收数字"（吞吐对比、显存对比、
loss 一致性）都来自该环境实测。

## 📈 一张表看懂四种并行

| 并行 | 切什么 | 通信 | 解决 | 代价 |
|---|---|---|---|---|
| **数据并行 DDP** | 数据 | 每步 all-reduce 梯度 | 数据太大 / 加速 | 每 rank 存完整模型 |
| **ZeRO/FSDP** | 模型状态（参数/梯度/优化器） | all-gather + reduce-scatter | 模型状态装不下 | 通信 ≈1.5× |
| **张量并行 TP** | 单层权重矩阵 | 每层 2 次 all-reduce | 单层放不下 / 大激活 | 通信最频繁，只适合机内 |
| **流水线并行 PP** | 按层分组 | 点对点传激活 | 层数太多装不下 | bubble 空转 (p-1)/(m+p-1) |

## 📝 课后作业

每章末尾有思考题。全部学完后：

👉 [Assignment 10](../../../assignments/assignment_10/)（题 1-4 纯 CPU 可完成：
all-reduce 语义 / 显存账本计算器 / DistributedSampler 证明 / TP 分块数学；🌟 流水线 bubble）

## 🔗 相关资源

- 🐙 [pytorch/examples minGPT-ddp](https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp) — Karpathy 原 minGPT-ddp 的官方维护版
- 📖 [PyTorch DDP 设计笔记](https://docs.pytorch.org/docs/stable/notes/ddp.html) · [FSDP 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- 🐙 [huggingface/nanotron](https://github.com/huggingface/nanotron) — 3D 并行极简实现
- 📺 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — HF 的集群训练系统教程（本课的"下一站"）
- 📄 [ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054) · [Megatron-LM (1909.08053)](https://arxiv.org/abs/1909.08053) · [Reduced Activation Re-computation (2205.05198)](https://arxiv.org/abs/2205.05198)

---

[← 上一章：Part 9 CUDA 内核](../../Part9_cuda_kernels/tutorial/README.md) | [下一章：Part 11 verl 对齐实战 →](../../Part11_alignment_verl/tutorial/README.md)
