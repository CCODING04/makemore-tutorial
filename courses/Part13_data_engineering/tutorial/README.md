# Part 13: 数据工程 — 从手写 MinHash 到 Data-Juicer

> 🧭 "模型质量的上限是数据质量"——但数据工程是从零课程最常缺失的一环（也是面试簇 G 的
> 空白）。本部分先**手写**工业去重的核心算法（MinHash + 分带 LSH，~60 行），
> 再用工业工具 **Data-Juicer**（阿里，200+ 算子）复跑同一条管线，对照"手写 60 行 vs
> 工业工程差 4 个数量级"的每一步。
> 主源：[datajuicer/data-juicer](https://github.com/datajuicer/data-juicer)（7.0k，Apache-2.0，阿里通义）

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 MinHash + LSH 去重](01_dedup_from_scratch.md) | shingling → 签名 → 分带 LSH → Jaccard 验证，LSH 概率性质 | `01` |
| 02 | [Data-Juicer 管线](02_data_juicer_pipeline.md) | YAML 配置驱动、算子全家桶、追踪审计、FineWeb 对照 | `02` |

## 🧰 前置知识

- **Part 8 07 章**：评估学里的"污染去重"（n-gram 重叠检查——本章是它的算法底座）
- 概率直觉：P[两集合最小哈希相等] = Jaccard（01 章推导）

## 🔗 在 LLM 链路中的位置

```
【本部分: 数据工程】→ 预训练(Part 7) → SFT/对齐(Part 8/11/12) → 部署(Part 14)
```

数据工程的产出决定预训练语料质量：**FineWeb 用 5-gram MinHash 全局去重（阈值≈0.7）+ 质量
过滤拿下当时最佳开源预训练集**；Gopher 报告去重提升基准最高 +1.5%、去污染 +2.6%。

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
自己设计一条清洗管线                → 面试/工作就绪
```

## 📝 课后作业

👉 [Assignment 13](../../../assignments/assignment_13/)

## 🔗 相关资源

- 🐙 [Data-Juicer](https://github.com/datajuicer/data-juicer) · [data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)（50+ 配方）
- 📝 [FineWeb 博客](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)（工业去重的最佳叙述：5-gram、14 band × 8 row、阈值≈0.7）
- 📄 [Deduplicating Training Data Makes LMs Better](https://arxiv.org/abs/2107.06499) · Gopher（arXiv 2112.11446）

---

[← 上一章：Part 12 LLaMA-Factory](../../Part12_finetune_llamafactory/tutorial/README.md) | [下一章：Part 14 vLLM →](../../Part14_inference_vllm/tutorial/README.md)
