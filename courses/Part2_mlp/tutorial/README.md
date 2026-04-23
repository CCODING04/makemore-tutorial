# Part 2: 从 Bigram 到 MLP — 多层感知机名字生成

> 🚀 从只看 1 个字符，升级到看 3 个字符的上下文窗口！

## 📚 章节导航

| 序号 | 章节 | 内容 |
|------|------|------|
| 01 | [从 Bigram 到 MLP](01_introduction.md) | 数据集准备、block_size、Train/Dev/Test 划分 |
| 02 | [MLP 架构](02_mlp_architecture.md) | Embedding 层、前向传播、CrossEntropy Loss |
| 03 | [训练与评估](03_training_and_eval.md) | Minibatch SGD、学习率、过拟合诊断、采样生成 |

## 🗺️ 学习路线图

```
Part 1 (Bigram)
    │
    │  "Bigram 只看 1 个字符，太少了！"
    ▼
┌─────────────────────────────┐
│  Part 2: MLP                │
│                             │
│  ① block_size=3 的数据集    │──→ 01_introduction.md
│  ② Embedding + MLP 架构     │──→ 02_mlp_architecture.md
│  ③ 训练、评估、采样         │──→ 03_training_and_eval.md
│                             │
└─────────────┬───────────────┘
              │
              │  "训练起来了，但不稳定..."
              ▼
         Part 3 (优化器与初始化)
```

## 🎯 学完这一部分你能...

- ✅ 理解 **Embedding**：把离散的字符变成连续向量
- ✅ 搭建一个 **两层 MLP**（Embedding → 隐藏层 → 输出层）
- ✅ 掌握 **Train/Dev/Test** 数据划分的正确姿势
- ✅ 用 **Minibatch SGD** 高效训练
- ✅ 诊断 **过拟合**，看懂 train loss vs dev loss
- ✅ 从训练好的模型 **采样生成** 新名字

## 📝 课后作业

完成教程后，去这里做练习：

👉 [Assignment 2](../assignment_2/)

## 🔗 相关资源

- 📺 Andrej Karpathy 原视频：[Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- 📄 Bengio et al. 2003 论文：[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
