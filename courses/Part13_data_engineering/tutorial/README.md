# Part 13: 数据工程 — 从手写 MinHash 到 Data-Juicer

> 🧭 "模型质量的上限是数据质量"——但数据工程是从零课程最常缺失的一环（也是面试簇 G 的
> 空白）。本部分先**手写**工业去重的核心算法（MinHash + 分带 LSH，~60 行），
> 再用工业工具 **Data-Juicer**（阿里，200+ 算子）复跑同一条管线，对照"手写 60 行 vs
> 工业工程差 4 个数量级"的每一步。
> 主源：[datajuicer/data-juicer](https://github.com/datajuicer/data-juicer)（7.0k，Apache-2.0，阿里通义）

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 数据工程在 LLM 链路中的位置和价值
- ✅ **手写** MinHash + LSH 去重算法（~60 行），**解释**其概率性质
- ✅ **配置** Data-Juicer 的 YAML 管线并理解每个算子的作用
- ✅ **设计** 一条完整的数据清洗管线
- ✅ **识别** 常见的数据质量问题并设计解决方案

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 MinHash + LSH 去重](01_dedup_from_scratch.md) | shingling → 签名 → 分带 LSH → Jaccard 验证，LSH 概率性质 | `01` |
| 02 | [Data-Juicer 管线](02_data_juicer_pipeline.md) | YAML 配置驱动、算子全家桶、追踪审计、FineWeb 对照 | —（YAML/CLI 实操） |

## 🧰 前置知识

**必须掌握：**
- **[Part 8 · 07 评估学](../../Part8_post_training/tutorial/07_evaluation.md)**：评估学里的
  "污染去重"（n-gram 重叠检查——本章是它的算法底座）。为什么需要：先见过"为什么要去重"，
  本章才回答"怎么去重"。

**建议掌握：**
- **概率直觉**：P[两集合最小哈希相等] = Jaccard（[01 章](01_dedup_from_scratch.md)推导）。
  为什么需要：MinHash 的全部正确性都建立在这一个等式上。

**可选：**
- **[Part 7 · Minimind 预训练](../../Part7_minimind/tutorial/README.md)**：预训练流程
  （了解数据在预训练中的作用）。为什么需要：去重/过滤的收益最终要在预训练 loss 上兑现。

## 🔗 在 LLM 链路中的位置

```
【本部分: 数据工程】→ 预训练(Part 7) → SFT/对齐(Part 8/11/12) → 部署(Part 14)
    ↑
    你在这里
```

**为什么数据工程是"模型质量的上限"：**

| 证据 | 说明 |
|------|------|
| FineWeb | 用 5-gram MinHash 全局去重 + 质量过滤，拿下当时最佳开源预训练集 |
| Gopher | 去重提升基准最高 +1.5%，去污染 +2.6% |
| Llama 3 | 数据质量是模型性能的关键因素之一 |

## 理论背景

### 问题引入：为什么需要数据工程？

预训练数据虽然"多"，但质量参差不齐：

1. **重复数据**：同一文档出现多次，导致模型"记住"而非"理解"
2. **低质量数据**：垃圾邮件、广告、机器生成的内容
3. **污染数据**：测试集泄露到训练集，导致评估失真

数据工程通过**清洗、去重、过滤**来提升数据质量：

```
原始数据:  "大量但质量参差不齐"
  ↓ 去重
去重后:    "去除重复，减少记忆"
  ↓ 过滤
过滤后:    "去除低质量，保留高质量"
  ↓ 混合
混合后:    "平衡不同领域，提升泛化能力"
```

> 💡 **类比**：原始数据像是未经筛选的食材，数据工程像是洗菜、切菜、调味。
> 食材质量决定了菜品的上限。

### 数学推导：MinHash 的概率性质

MinHash 的核心思想是：**用随机哈希函数近似 Jaccard 相似度**。

**问题设定：**
- 两个集合 A 和 B
- Jaccard 相似度：J(A,B) = |A∩B| / |A∪B|

**推导过程：**

```
Step 1: 定义 MinHash
  h(S) = min{x ∈ S | h(x)}  # 集合 S 中哈希值最小的元素

Step 2: 概率性质
  P[h(A) = h(B)] = J(A,B)

  证明：
  - 设 U = A∪B
  - 对于 U 中的任意元素 x，h(x) 是随机的
  - h(A) = h(B) 当且仅当 A∪B 中哈希值最小的元素在 A∩B 中
  - P[min ∈ A∩B] = |A∩B| / |A∪B| = J(A,B)

Step 3: 多次哈希估计
  使用 k 个独立的哈希函数 h_1, h_2, ..., h_k
  签名向量：sig(A) = [h_1(A), h_2(A), ..., h_k(A)]
  估计 Jaccard：J_est = (1/k) * Σ I[h_i(A) = h_i(B)]

  性质：
  - E[J_est] = J(A,B)（无偏估计）
  - Var[J_est] = J(A,B) * (1 - J(A,B)) / k
  - k 越大，估计越准
```

**关键洞察：**
- MinHash 把集合相似度问题转化为向量比较问题
- 签名向量的维度 k 控制估计精度
- 工业上 k=128-256 是常见配置

### 历史脉络：数据工程演进

```
2018: 精确去重（hash-based）
  ↓ O(n²) 无法扩展
2020: MinHash + LSH（近似去重）
  ↓ 近线性复杂度
2022: 质量过滤（perplexity, classifier）
  ↓ 自动化过滤
2024: FineWeb（5-gram MinHash + 质量过滤）
  ↓ 工业级数据工程
```

**关键论文：**
- MinHash: [Similarity Estimation Techniques from Rounding Algorithms](https://cs.brown.edu/research/pubs/theses/ugrad/2005/broder.pdf)
- FineWeb: [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)
- Gopher: [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)

## 📦 环境与版本策略

```bash
# 01 章手写：无需任何安装（纯 Python 标准库）
# 02 章 Data-Juicer：跟随 latest，文本管线 CPU 即可
pip install py-data-juicer        # 重依赖（Ray/多模态）为可选 extras，按需装
```

| 你有什么 | 能做什么 |
|---|---|
| 任何机器（含 CPU 笔记本） | 全部内容——01 手写 + 02 Data-Juicer 小管线 |

## 📈 学习地图

```
手写 MinHash/LSH（01：数学+实现）   ← 点
   ↓ "这 60 行被工业版怎么放大？"
Data-Juicer YAML 管线（02）         ← 面（200+ 算子、审计、分布式）
   ↓ 读 FineWeb/Gopher 的真实做法
自己设计一条清洗管线                →  面试/工作就绪
```

## 📝 课后作业

每章末尾有思考题（`<details>` 折叠答案）。全部学完后：

👉 [Assignment 13](../../../assignments/assignment_13/)

## 🔗 相关资源

- 🐙 [Data-Juicer](https://github.com/datajuicer/data-juicer) · [data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)（50+ 配方）
- 📝 [FineWeb 博客](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)（工业去重的最佳叙述：5-gram、14 band × 8 row、阈值≈0.7）
- 📄 [Deduplicating Training Data Makes LMs Better](https://arxiv.org/abs/2107.06499) · Gopher（arXiv 2112.11446）

---

[← 上一章：Part 12 LLaMA-Factory](../../Part12_finetune_llamafactory/tutorial/README.md) | [下一章：Part 14 vLLM →](../../Part14_inference_vllm/tutorial/README.md)
