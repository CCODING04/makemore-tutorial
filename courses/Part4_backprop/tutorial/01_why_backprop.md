# 01 为什么要手写反向传播？🤔

> "如果你不能手算梯度，你就不算真正理解深度学习。" —— Andrej Karpathy

## 前置知识：Part 3 MLP+BN 回顾

在 Part 3 里，我们搭了这样一个网络：

```
输入(context 3个字符)
    │
    ▼
  Embedding (每个字符 → 10维向量)
    │
    ▼
  拼接 (3×10 = 30维)
    │
    ▼
  Linear (30 → 200)
    │
    ▼
  BatchNorm (标准化 + γ缩放 + β平移)
    │
    ▼
  Tanh (激活函数)
    │
    ▼
  Linear (200 → 27)
    │
    ▼
  Softmax → CrossEntropy Loss
```

训练的时候，我们只需要：

```python
loss = F.cross_entropy(logits, Yb)
loss.backward()  # 🪄 魔法！自动算出所有梯度
```

一行 `loss.backward()` 就搞定了所有参数的梯度计算。但...这到底是怎么做到的？

## 为什么要手动推导梯度？

### 理由 1：理解 autograd 的原理

PyTorch 的 autograd 本质上就是**计算图 + 链式法则**。当你理解了手写梯度的过程，autograd 就不再是黑魔法，而是理所当然的工具。

### 理由 2：调试能力

当你的模型不收敛、梯度爆炸/消失、loss 是 NaN... 如果你能手算梯度，就能快速定位问题出在哪一层。

### 理由 3：实现自定义操作

有时候标准库没有你需要的操作（比如特殊的 loss function、特殊的归一化），你需要自己写前向和反向传播。

### 理由 4：面试 🎯

"请推导 BatchNorm 的反向传播" —— 这是常见面试题。理解了本 Part，你就能自信回答。

## 链式法则回顾

链式法则是反向传播的数学基础。简单来说：

> 如果 `z = f(y)` 且 `y = g(x)`，那么 `dz/dx = dz/dy × dy/dx`

用计算图来看：

```
x ──→ g ──→ y ──→ f ──→ z

反向传播（从 z 往回推）：
dz/dx = dz/dy × dy/dx
       = (z 对 y 的梯度) × (y 对 x 的梯度)
```

### 多维情况的链式法则

对于矩阵运算 `C = A @ B`：

```
dL/dA = dL/dC @ B^T     (梯度传播到 A)
dL/dB = A^T @ dL/dC     (梯度传播到 B)
```

这就是为什么我们在反向传播里到处看到 `@` 和 `.T`（转置）。

## 本 Part 的学习路线

1. **先看前向传播是怎么走的** → 运行 [`01_forward_pass_steps.py`](../scripts/01_forward_pass_steps.py)
   - 每一步都保存中间变量
   - 打印每步 tensor 的形状
   - 画流程图理解数据流动

2. **然后从 loss 往回推 12 步** → 下节课 [02_forward_and_backward.md](02_forward_and_backward.md)

## 小结 📌

| 概念 | 一句话总结 |
|------|-----------|
| 前向传播 | 数据从输入流向输出，每步保存中间变量 |
| 反向传播 | 梯度从 loss 流回参数，每步用链式法则 |
| autograd | 自动帮你算链式法则，但本质和你手算一样 |
| 为什么要手算 | 理解原理 + 调试 + 自定义操作 + 面试 |

---

**下一步** → [02 前向+反向逐步推导](02_forward_and_backward.md)
