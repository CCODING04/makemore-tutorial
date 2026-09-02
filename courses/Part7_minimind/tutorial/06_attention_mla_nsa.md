# 04 — 注意力演进：MLA 与原生稀疏注意力（NSA）

> 🧭 RoPE/GQA 之后，注意力机制还在演进。两条主线都来自 DeepSeek：
> **MLA**（纵向压缩 KV 缓存）与 **NSA**（横向稀疏化注意力计算）。
> 本章配 [scripts/12_mla_nsa_accounting.py](../scripts/12_mla_nsa_accounting.py)
> （CPU 5 秒，账本+三分支机制全手写）。**论文阅读实战**：
> [DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434) §MLA 节与
> [NSA 论文](https://arxiv.org/abs/2502.11089) §2-3——用 docs/paper_reading_guide.md 的
> 五步法读（符号表→直觉→边界→单调性→数值验证）。

## 📖 前置知识

- **必须掌握**：**03 章**（GQA 与 KV Cache——MLA 的对照面、NSA 的稀疏对象）
- **建议掌握**：**02 章**（RoPE——MLA 的"解耦 RoPE"是它的直接后续）
- **可选**：DeepSeek-V2/V3 技术报告的注意力章节（先学课程版再看原文更顺）

## 1. MLA：把 KV 压成一个 latent 向量

GQA 靠"少几个 KV 头"省缓存；MLA 换一条路——**低秩压缩**：

```
MHA/GQA：每个 token 存 n_kv_heads × head_dim 的 K 和 V
MLA     ：每个 token 存 1 个 c_KV 向量（kv_lora_rank 维），
          注意力时用上投影矩阵还原各头的 K/V
```

DeepSeek-V2 的真实数字（32 层/128 head_dim/seq 2048/fp16，脚本 Part A 实测）：

```
MHA       : 1.07 GB
GQA(kv8)  : 0.27 GB
MLA       : 0.08 GB   (KV 缓存降至 MHA 的 7.0%；逐 token 复算与公式一致 ✅)
```

- 🔑 **解耦 RoPE 的坑**：RoPE 是位置相关的旋转矩阵，会破坏低秩压缩的"矩阵吸收"
  （W_UK 无法吸收进 W_UQ）。MLA 的方案：给每个 token 额外携带一个小的**位置专用
  key**（k^R），与内容 latent 分离，注意力得分两部分相加。
- 💡 **矩阵吸收**：推理时将 W_UK 吸收进 W_UQ，decode 直接从压缩缓存计算注意力——
  无需还原完整 K。这是 MLA 工程实现的关键一步。

## 2. NSA：三分支原生稀疏注意力

NSA（2502.11089，ACL'25 最佳论文）的核心主张：**从预训练起就原生训练稀疏模式**
（而非推理期后置近似），三分支并行：

| 分支 | 做什么 | 成本 |
|---|---|---|
| **压缩分支** | 每块平均成摘要 token，提供全局粗读 | n_blocks |
| **选择分支** | 由压缩分数选 top_k 关键块，块内精细注意力 | top_k 块 |
| **滑动窗口** | 只看最近 window 个 token | window |

三分支输出经**可学习门控**线性组合。脚本 Part B 用固定等权门控的最小实现展示
三分支机制（等权门控下与全注意力的 max diff ≈ 2.3，门控可学习后收敛）。

## 3. MLA vs GQA vs NSA：三条压缩路线的对照

| 维度 | GQA | MLA | NSA |
|---|---|---|---|
| 压缩对象 | KV 头数 | KV 缓存维度（低秩） | 注意力计算的模式（稀疏化） |
| 缓存降幅 | 线性（按头数） | **按 latent 维大幅降** | 不减缓存，减 FLOPs |
| 训练方式 | 端到端 | 端到端 | **原生**稀疏训练 |
| 硬件亲和 | — | 矩阵吸收优化 | GPU kernel 对齐 |

## 学完本部分你能...

- ✅ 算出 MHA/GQA/MLA 的 KV 缓存字节数（逐 token 复算与公式一致）
- ✅ 解释 MLA 的解耦 RoPE 与矩阵吸收
- ✅ 画出 NSA 三分支的分工与门控融合
- ✅ 说出"原生可训练稀疏"与"推理期后置近似"的区别

**课后练习**

<details>
<summary>Q1: MLA 为什么不能把 RoPE 也压进 latent？</summary>
A: RoPE 是位置相关的旋转矩阵——对每个 token 施加不同的旋转。低秩压缩要求
K/V 能被同一个上投影矩阵还原，但旋转后的 K/V 位置相关、投影矩阵无法统一吸收
（W_UK 被 R_t 打断）。MLA 的解法：位置部分单独走一个小的 RoPE 通路（k^R），
与内容 latent 解耦。
</details>

<details>
<summary>Q2: NSA 的"原生可训练"为什么比"推理期后置稀疏化"效果好？</summary>
A: 后置稀疏化（如 H2O/StreamingLLM）是在训练好的稠密模型上近似——模型从未见过
稀疏模式，误差随稀疏度放大。原生训练让模型学会"在稀疏模式下表现最好"，且配合
硬件对齐 kernel（NSA）实际速度反超稠密。类比：LoRA vs 后置剪枝的关系。
</details>

### 动手实践：改 latent 维度，画 KV 显存-精度权衡曲线

**任务**：修改 [scripts/12_mla_nsa_accounting.py](../scripts/12_mla_nsa_accounting.py)
中 MLA 的 `d_c`（latent 维度），在 {32, 64, 96, 128} 四档下重跑，记录每档的
KV cache 字节数与（若脚本含小模型评测段的）loss 变化，画成一张折线图。

**验收标准**：
- [ ] KV 字节数随 d_c 线性增长（每档打印值可对上 d_c × dtype 字节数的账本）
- [ ] d_c=128 与 GQA 的 KV 字节数对比方向正确（谁省谁费说得清）
- [ ] 能用一句话解释"为什么 d_c 越小越省、但小到某个点精度会崩"（信息瓶颈）

## 📝 课后作业

完成 [scripts/12_mla_nsa_accounting.py](../scripts/12_mla_nsa_accounting.py) 的两个
Part 后，回答：
1. 把 kv_lora_rank 从 512 改成 256，MLA 的缓存变多少？质量预期如何变化？
2. 把 NSA 的 window 从 4 改成 8，三分支的相对贡献怎么变？

## 下一步

Part 11 的 GRPO 是单轮 RLVR——Agentic RL 把它扩展到多轮工具调用与长程任务
（→ Part 17 Agentic RL，与本章同属 DeepSeek 架构创新的延伸）。

👉 [Part 17 Agentic RL](../../Part17_agentic_rl/tutorial/README.md)

---

[← 上一章：Part 16 图像/视频生成](../../Part16_image_video_generation/tutorial/README.md) | [下一章：Part 17 Agentic RL →](../../Part17_agentic_rl/tutorial/README.md)
