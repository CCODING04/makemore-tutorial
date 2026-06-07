# Part 5: WaveNet — 从展平到层次化融合

> 🌊 从"一口气看完所有上下文"到"从局部到全局的层次化融合"！

## 📚 章节导航

| 序号 | 章节 | 内容 |
|------|------|------|
| 01 | [PyTorch 化：让代码更优雅](01_pytorchify.md) | Embedding/Flatten/Sequential 模块化、BatchNorm 的训练/推理坑 |
| 02 | [WaveNet 架构：层次化融合](02_wavenet_architecture.md) | 为什么要层次化、FlattenConsecutive、Linear 支持多维输入 |
| 03 | [训练与 Bug 修复](03_training_and_bugs.md) | BatchNorm 3D bug、放大训练、卷积预览 |

## 🗺️ 学习路线图

```
Part 3 (BatchNorm)
    │
    │  "网络稳定了，但性能到瓶颈了..."
    ▼
┌──────────────────────────────────────┐
│  Part 5: WaveNet                     │
│                                      │
│  ① PyTorch 化代码 — 模块化重构     │──→ 01_pytorchify.md
│  ② WaveNet 架构 — 层次化融合       │──→ 02_wavenet_architecture.md
│  ③ 训练修复 — 3D BatchNorm bug     │──→ 03_training_and_bugs.md
│                                      │
└──────────────┬───────────────────────┘
               │
               │  "理解了 WaveNet，接下来是卷积神经网络..."
               ▼
          Part 6 (后续)
```

## 🎯 学完这一部分你能...

- ✅ 把散乱的参数管理重构成 **PyTorch 风格的模块化代码**
- ✅ 理解 **Sequential 容器** 和 `parameters()` 统一接口
- ✅ 掌握 **FlattenConsecutive** 层：逐步融合上下文
- ✅ 理解 **Linear 层天然支持多维输入**的洞察
- ✅ 修复 **BatchNorm1D 的 3D 输入 bug**
- ✅ 构建 **WaveNet 层次化架构**，验证 loss 降到 < 2.0

## 📝 课后作业

完成教程后，去这里做练习：

👉 [Assignment 5](../../assignments/assignment_5/)

## 🔗 相关资源

- 📺 Andrej Karpathy 原视频：[Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- 📄 Van den Oord et al. 2016 论文：[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- 📄 Dilated Causal Convolutions：WaveNet 的卷积等价形式

---

[← 上一章：Part 4 Backpropagation](../../Part4_backprop/tutorial/README.md)
