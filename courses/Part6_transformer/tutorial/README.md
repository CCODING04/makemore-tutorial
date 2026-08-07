# Part 6: Transformer / GPT — 从零构建一个迷你 ChatGPT

> 🤖 200 行代码，训练一个 decoder-only Transformer，让神经网络写出"莎士比亚"。

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [数据与 Tokenizer](01_data_and_tokenizer.md) | ChatGPT 动机、tiny Shakespeare、字符级 tokenizer、train/val 划分、Dataloader、Bigram 基线、交叉熵、AdamW | `01` `02` |
| 02 | [Attention 从零开始](02_attention_from_scratch.md) | 聚合的数学技巧三版本、位置编码、self-attention 单头（K/Q/V）、**6 条 attention 笔记**、Multi-Head | `03` `04` `05` |
| 03 | [Transformer Block](03_transformer_block.md) | FeedForward、残差连接、LayerNorm（pre-norm）、完整 decoder-only Transformer、Dropout、Scale Up 与生成 | `05` `06` `07` |
| 04 | [超越 Transformer](04_beyond_transformer.md) | Encoder vs Decoder、Cross-Attention、nanoGPT 走读、回到 ChatGPT/GPT-3、预训练 vs 微调、RLHF | 走读 |

## 🧰 前置知识

本部分需要你已经掌握：

- **Python + PyTorch 基础**：`nn.Module`、`nn.Embedding`、`F.softmax`、交叉熵损失
- **语言建模框架**：Part 1/2 里"预测下一个 token、负对数似然损失、训练循环"的整套思路
- **BatchNorm 的训练/推理两态**：Part 3 的 `02_batchnorm.md` —— 讲 LayerNorm 时会和它对照
- **手动反向传播的直觉**：Part 4 的"加法节点把梯度均分给两个分支" —— 讲残差连接时会用到
- **卷积的空间性**：Part 5 的 WaveNet —— 讲 attention"无空间概念"时会和卷积对比

> 💡 如果你卡住了，随时回看前几章的 `tutorial/` 目录。

## 🗺️ 学习路线图

```
Part 5 (WaveNet / 卷积层次化融合)
    │
    │  "卷积有空间性、有固定感受野，能不能让 token 自己决定看谁？"
    ▼
┌─────────────────────────────────────────────┐
│  Part 6: Transformer / GPT                  │
│                                             │
│  ① 数据与 Tokenizer — 字符级编码           │──→ 01_data_and_tokenizer.md
│  ② Attention 从零开始 — 通信机制           │──→ 02_attention_from_scratch.md
│  ③ Transformer Block — 通信+计算+残差      │──→ 03_transformer_block.md
│  ④ 超越 Transformer — GPT-3 / RLHF         │──→ 04_beyond_transformer.md
│                                             │
└──────────────┬──────────────────────────────┘
               │
               │  "理解了 attention，下一步是让模型学会推理..."
               ▼
          Transformer 之后（阅读 Karpathy 的 micrograd / minGPT 等）
```

## 🎯 学完这一部分你能...

- ✅ 讲清楚 **ChatGPT 底层就是语言模型**，以及 GPT = Generative Pretrained Transformer
- ✅ 手写**字符级 tokenizer**（`encode`/`decode`），理解字符级 vs subword（BPE/sentencepiece）的取舍
- ✅ 实现 **Dataloader**：一个 chunk 里装多个样本、随机 offset 采样、batch 独立并行
- ✅ 理解 attention 的**数学技巧**：用 `torch.tril` + softmax 做加权聚合
- ✅ 从零实现 **self-attention 单头**（query/key/value、亲和力、缩放、遮罩），并展开 **6 条 attention 笔记**
- ✅ 组装 **Multi-Head + FeedForward + 残差连接 + LayerNorm（pre-norm）** 的完整 Transformer Block
- ✅ 用 **Dropout** 正则化、理解 scale up 后 loss 如何从 2.5 一路降到 1.48
- ✅ 区分 **encoder / decoder / 完整架构**，看懂 **nanoGPT** 代码
- ✅ 讲清楚 ChatGPT 的 **预训练（文档补全器）→ 微调（SFT → 奖励模型 → RLHF）** 全流程

## 📈 演进路线：loss 一路怎么降的

本教程会反复看到这张表。**两套数字**：

| 阶段 | Karpathy 视频 | 本仓库脚本（CPU 小规模） |
|------|:---:|:---:|
| Bigram 初始 | ~4.87 | ~4.74 |
| Bigram 训练后 | ≈2.5 | ≈2.50 |
| + 单头 self-attention | 2.4 | ≈2.39 |
| + multi-head | 2.28 | ≈2.45 |
| + feedforward | 2.24 | ≈2.50 |
| + 残差连接 | 2.08 | ≈2.23 |
| + LayerNorm | 2.06 | ≈2.23 |
| Scale up（GPU） | **1.48** | CPU 缩小型 ≈2.80 |

> ⚠️ 视频里的数字是 **A100 GPU + 完整超参** 跑出来的；我们的脚本是 **CPU 缩小版**（更小的 batch/block/层数/步数）。不同超参、不同随机种子，数字都会有差异，所以都带 ≈。**看趋势，别死记数字。**

## 📝 课后作业

每一章末尾有 2-3 道思考题（`<details>` 折叠答案）。全部学完后，去这里做动手练习：

👉 [Assignment 6](../../../assignments/assignment_6/)

## 🔗 相关资源

- 📺 Andrej Karpathy 原视频：[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)（makemore Part 6）
- 📄 Vaswani et al. 2017 论文：[Attention is All You Need](https://arxiv.org/abs/1706.03762)
- 📄 He et al. 2015：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 📄 Srivastava et al. 2014：[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://arxiv.org/abs/1207.0580)
- 📄 Ba et al. 2016：[Layer Normalization](https://arxiv.org/abs/1607.06450)
- 🐙 [nanoGPT](https://github.com/karpathy/nanoGPT) — 训练 Transformer 的最简参考实现
- 📄 [GPT-3 论文](https://arxiv.org/abs/2005.14165)：175B 参数、300B tokens 预训练
- 📄 OpenAI 博客：[ChatGPT 对齐阶段（SFT → 奖励模型 → RLHF）](https://openai.com/blog/chatgpt)

---

[← 上一章：Part 5 WaveNet](../../Part5_wavenet/tutorial/README.md)
