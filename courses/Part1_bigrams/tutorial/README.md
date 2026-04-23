# Part 1：从零构建 Bigram 字符级语言模型

> 🎯 本节对应 Andrej Karpathy 的 [makemore 系列第 1 讲](https://www.youtube.com/watch?v=PaCmpygFfXo)，用中文重新讲解，配合可运行的代码脚本。

## 📖 章节导航

| 序号 | 主题 | 内容概要 |
|:---:|------|---------|
| 01 | [前置知识与课程预告](01_introduction.md) | 你需要知道什么、学完能做什么、语言模型是什么 |
| 02 | [Bigram 模型：从统计到采样](02_bigram_model.md) | Bigram 概念、频率统计、概率矩阵、采样生成、NLL Loss |
| 03 | [用神经网络重新实现](03_neural_network.md) | one-hot 编码、Softmax、梯度下降、与计数法的等价性 |

## 📂 配套资源

- **代码脚本**：[`../scripts/`](../scripts/) — 每个关键步骤都有独立 Python 脚本
- **输出图片**：[`../images/`](../images/) — Jupyter Notebook 的输出截图
- **课后作业**：[`../../../assignments/assignment_1/`](../../../assignments/assignment_1/) — 动手实践

## 🗺️ 学习路线

```
01 引言 ──→ 02 Bigram 统计模型 ──→ 03 神经网络版
  (理解背景)    (核心：计数与采样)      (进阶：可扩展框架)
                                         │
                                         ▼
                                    课后作业
                                  (动手巩固)
```

建议按顺序阅读，每节大约 15-30 分钟。遇到代码片段时，打开对应的脚本文件一起看效果更好 🚀
