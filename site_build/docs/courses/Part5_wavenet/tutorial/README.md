# Part 5: WaveNet — 从展平到层次化融合

> 🌊 从"一口气看完所有上下文"到"从局部到全局的层次化融合"！

## 📚 章节导航

| 序号 | 章节 | 内容 |
|------|------|------|
| 01 | [PyTorch 化：让代码更优雅](01_pytorchify.md) | Embedding/Flatten/Sequential 模块化、BatchNorm 的训练/推理坑、loss 曲线平滑 |
| 02 | [WaveNet 架构：层次化融合](02_wavenet_architecture.md) | 为什么要层次化、FlattenConsecutive、Linear 支持多维输入、view vs cat、上下文 3→8 |
| 03 | [训练与 Bug 修复](03_training_and_bugs.md) | BatchNorm 3D bug、参数量手算、放大训练、卷积预览 |

## 🖥️ 如何运行脚本

```bash
cd courses/Part5_wavenet/scripts
python 01_pytorchify_layers.py            # 默认档（与教程数字对应）
STEPS=2000 python 03_increase_context.py  # 短程档：只跑 2000 步，快速验证流程
python 07_scaled_wavenet.py --quick       # 07 专用快速档（1000 步）
```

- 默认档行为与教程/脚本注释中的实测数字对应；`STEPS` 短程档只改训练步数，用于快速验证代码能跑通（默认档完全不受影响）。
- CPU 参考时长：01 约 1 分钟；02/03/05 默认档约 5-8 分钟；07 完整档约 15-25 分钟（`--quick` 约 1 分钟）。
- 若多开脚本互相拖慢，用 `OMP_NUM_THREADS=4` 限制每个进程的线程数。

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
               │  "理解了 WaveNet，接下来是让 token 自己决定看谁..."
               ▼
          Part 6: Transformer/GPT（[开始学习](../../Part6_transformer/tutorial/README.md)）
```

## 🎯 学完这一部分你能...

- ✅ 把散乱的参数管理重构成 **PyTorch 风格的模块化代码**
- ✅ 理解 **Sequential 容器** 和 `parameters()` 统一接口
- ✅ 掌握 **FlattenConsecutive** 层：逐步融合上下文
- ✅ 理解 **Linear 层天然支持多维输入**的洞察
- ✅ 修复 **BatchNorm1D 的 3D 输入 bug**
- ✅ 构建 **WaveNet 层次化架构**，把验证 loss 从 ≈2.17 压到 ≈2.0

## 📝 课后作业

完成教程后，去这里做练习：

👉 [Assignment 5](../../../assignments/assignment_5/)

## 🔗 相关资源

- 📺 Andrej Karpathy 原视频：[Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- 📄 Van den Oord et al. 2016 论文：[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- 📄 Dilated Causal Convolutions：WaveNet 的卷积等价形式

---

[← 上一章：Part 4 Backpropagation](../../Part4_backprop/tutorial/README.md)
