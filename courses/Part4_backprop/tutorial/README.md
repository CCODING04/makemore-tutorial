# Part 4：手动反向传播 🔧

> 把 PyTorch 的 autograd 扒开，看看里面到底在干什么。

## 📖 章节导航

| 序号 | 教程 | 配套脚本 | 难度 |
|------|------|----------|------|
| 01 | [为什么要手写反向传播](01_why_backprop.md) | [`01_forward_pass_steps.py`](../scripts/01_forward_pass_steps.py) | ⭐⭐ |
| 02 | [前向+反向逐步推导](02_forward_and_backward.md) | [`02_backprop_step_by_step.py`](../scripts/02_backprop_step_by_step.py) [`03_verify_gradients.py`](../scripts/03_verify_gradients.py) | ⭐⭐⭐⭐ |
| 03 | [简化公式与手动训练](03_simplified_and_training.md) | [`04_cross_entropy_backward.py`](../scripts/04_cross_entropy_backward.py) [`05_batchnorm_backward.py`](../scripts/05_batchnorm_backward.py) [`06_manual_training.py`](../scripts/06_manual_training.py) | ⭐⭐⭐ |

## 🗺️ 学习路线

```
Part 3 MLP+BN
      │
      ▼
┌──────────────────────┐
│ 01 为什么要手写反传   │  ← 动机 + 链式法则回顾
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 02 前向+反向逐步推导  │  ← 核心！12步推导 + 梯度验证
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 03 简化公式与手动训练  │  ← 一行CE反传 + 一行BN反传 + 完整训练
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Assignment 4 📝    │  ← 动手练一练
└──────────┬───────────┘
           │
           ▼
      Part 5 WaveNet
```

## 📝 课后作业

学完本 Part 后，完成 [Assignment 4](../../../assignments/assignment_4/)：

1. 实现逐步前向传播函数
2. 实现单步反向传播
3. 实现简化 CrossEntropy 反传
4. 实现简化 BatchNorm 反传
5. （拓展）手动梯度训练完整网络

## 🔗 相关资源

- [Andrej Karpathy - Building makemore Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI) — 本 Part 的原始视频
- [Part 3 笔记](../Part3_mlp/) — MLP + BatchNorm 架构回顾

## 🎯 学习目标

学完本 Part 你应该能：

- [ ] 说出 PyTorch autograd 的基本原理（计算图 + 链式法则）
- [ ] 手动推导 CrossEntropy Loss 的梯度（3 行简化版）
- [ ] 手动推导 BatchNorm 的梯度（1 行简化版）
- [ ] 用手动梯度训练完整网络（不用 `loss.backward()`）
- [ ] 用 `cmp()` 函数验证手写梯度的正确性

---

[← 上一章：Part 3 BatchNorm](../Part3_batchnorm/tutorial/README.md) | [下一章：Part 5 WaveNet →](../Part5_wavenet/tutorial/README.md)
