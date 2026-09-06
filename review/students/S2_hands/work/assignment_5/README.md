# 作业 5：WaveNet — 层次化融合

> **对应教程**：Part 5 — Building makemore Part 5: Building a WaveNet
>
> **前置**：完成作业 3（BatchNorm）

---

## 📋 概述

本作业聚焦于 WaveNet 架构的实现。你将亲手实现层次化融合的关键组件、修复 BatchNorm 的 3D 输入 bug，并训练一个性能超过 2.0 的模型。

完成本作业后，你应该能够：

- 实现 FlattenConsecutive 层，理解 shape 变换
- 构建 WaveNet 模型，理解层次化融合的设计思想
- 修复 BatchNorm1d 对 3D 输入的支持
- 验证 tensor 形状在网络中的正确流转

---

## 🔧 环境准备

### 依赖

```bash
pip install torch
```

### 数据

数据文件位于 `../../data/names.txt`，每行一个英文名字。

### 文件结构

```
assignments/assignment_5/
├── assignment.md               # 本文件
├── wavenet_exercises.py        # 👈 你需要编辑的文件
└── test_wavenet_exercises.py   # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_5
python test_wavenet_exercises.py
```

---

## 📝 题目列表

### 题 1：FlattenConsecutive 层（基础）

**类**：`FlattenConsecutive`

**要求**：
- `__init__(self, n)`：n 是要拼接的连续位置数
- `__call__(self, x)`：把 `(B, T, C)` reshape 成 `(B, T//n, C*n)`
- `parameters()`：返回空列表（无可训练参数）
- 当 T 不能被 n 整除时，抛出 `AssertionError`

**示例**：
```python
fc = FlattenConsecutive(2)
x = torch.randn(4, 8, 10)   # (B=4, T=8, C=10)
y = fc(x)
assert y.shape == (4, 4, 20)  # (B=4, T=4, C=20)
```

**思考**：
- 为什么用 `view` 而不是 `torch.cat`？
- 如果 n=4，`(4, 8, 10)` 应该变成什么 shape？
- FlattenConsecutive 和 `view(B, -1, C*n)` 有什么关系？

---

### 题 2：WaveNet 模型构建（基础）

**函数**：`build_wavenet(vocab_size, n_embd=10, n_hidden=68, block_size=8)`

**要求**：
- 构建一个 3 层 WaveNet 模型，返回 `Sequential` 对象
- 结构：`Embedding → [FC(2) → Linear → BN → Tanh] × 3 → Linear`
- 每层 Linear 的输入维度是 `前一层输出 × 2`（因为 FC(2) 拼接了 2 个）
- 最后一层 Linear 的输出维度是 `vocab_size`
- 所有 Linear 层用 Kaiming 初始化
- 最后一层 Linear 的 weight 乘以 0.1（降低初始 loss）
- 所有参数设置 `requires_grad=True`

**返回**：
- `model`：Sequential 对象

**验证**：
```python
model = build_wavenet(27)
assert len(model.parameters()) > 0
# Forward 测试
x = torch.randint(0, 27, (4, 8))
logits = model(x)
assert logits.shape == (4, 27)
```

**思考**：
- 为什么每一层的 Linear 输入维度是 `n_hidden * 2`？（提示：FC(2) 把两个 n_hidden 向量拼成一个）
- 如果 block_size 不是 2 的幂（比如 6），这个架构还能用吗？
- 参数量大约是多少？和同样 n_hidden 的 MLP 比，参数量差距大吗？

---

### 题 3：BatchNorm1d 的 3D 支持（基础）

**类**：`BatchNorm1d3D`

**要求**：
- 继承或重新实现 BatchNorm1d，支持 2D 和 3D 输入
- 2D 输入 `(B, C)`：在 dim=0 上 reduce
- 3D 输入 `(B, T, C)`：在 dim=(0,1) 上同时 reduce
- `running_mean` 和 `running_var` 始终是 1D tensor `(C,)`
- eval 模式下正确广播：3D 时 unsqueeze 两次

**关键代码**：
```python
if x.ndim == 2:
    dim_reduce = 0
else:
    dim_reduce = (0, 1)  # 同时在 batch 和 time 上归一化
```

**思考**：
- 为什么 3D 输入要在 dim=(0,1) 上 reduce，而不是只在 dim=0？
- 如果只在 dim=0 上 reduce，每个时间步会独立归一化——这有什么问题？
- PyTorch 的 `nn.BatchNorm1d` 怎么处理 3D 输入的？

---

### 题 4：Tensor 形状流转验证（基础）

**函数**：`verify_shapes(model, vocab_size=27, block_size=8, batch_size=4)`

**要求**：
- 创建一个 `(batch_size, block_size)` 的随机输入
- 逐层 forward，记录每层输出的 shape
- 返回一个列表 `shapes`，每个元素是 `(layer_name, output_shape)` 的元组
- 验证最终输出 shape 是 `(batch_size, vocab_size)`

**示例输出**：
```python
shapes = verify_shapes(model)
for name, shape in shapes:
    print(f"  {name}: {shape}")
# Embedding: (4, 8, 10)
# FlattenConsecutive: (4, 4, 20)
# Linear: (4, 4, 68)
# ...
```

**思考**：
- 在 WaveNet 中，tensor 的第 2 维（T）是如何逐步缩小的？（8→4→2→1）
- 最终 `(B, 1, C)` 的 squeeze/flatten 是怎么变成 `(B, C)` 的？
- 如果中间某一层的 shape 不对，问题最可能出在哪里？

---

### 题 5：调参将验证 loss 降到 < 2.0（🌟 拓展）

**函数**：`train_wavenet(words, n_embd=24, n_hidden=128, block_size=8, steps=50000, seed=42)`

**要求**：
- 使用题目 1-3 的组件构建 WaveNet 并训练
- 数据划分 80/10/10
- 应用最佳实践：
  - Kaiming 初始化
  - BatchNorm（3D 修复版）
  - 学习率衰减（如 0.1 → 0.05 → 0.01）
  - Mini-batch 训练
- 返回 `(dev_loss, model)`

**目标**：dev_loss < 2.0

**调参建议**：
- `n_embd=24, n_hidden=128` 是一个好的起点
- `batch_size=128` 加速训练
- 学习率在 30000 步时从 0.1 降到 0.05，40000 步时降到 0.01
- 训练 50000 步应该足够

---

## ✅ 提交检查清单

- [ ] 所有 4 道基础题通过测试
- [ ] 拓展题（题 5）已尝试
- [ ] 能回答每道题后面的「思考」问题
- [ ] 代码中添加了必要的注释说明你的理解

---

## 💡 学习建议

1. **先理解 shape 再写代码**：WaveNet 的核心就是 shape 变换。在写代码前，先在纸上画出每层的输入输出 shape
2. **用小例子调试**：`(B=4, T=8, C=10)` 足够小，打印每层 shape 来验证
3. **3D BatchNorm 是隐藏的坑**：很多人在这里卡住，因为 loss 看起来在下降但性能不好
4. **对比实验**：WaveNet vs 直接展平 MLP，同样的参数量，比较验证 loss

---

## 🤔 深度思考题

**Q1：** 同样 `block_size=8`，直接 flatten 的 MLP 和 WaveNet 哪个参数更多？为什么？
<details>
<summary>💡 提示</summary>

直接 flatten 的 MLP 参数更多！因为 `Linear(8*10, 200)` 需要 `80*200=16,000` 个参数，而 WaveNet 的第一层 `Linear(2*10, 200)` 只需要 `20*200=4,000` 个参数。WaveNet 通过层次化融合，让每层的输入维度更小，从而减少参数量。

</details>

**Q2：** WaveNet 的树状结构和 Transformer 的注意力机制有什么本质区别？
<details>
<summary>💡 提示</summary>

- **WaveNet**：固定模式，只融合相邻的 2 个向量，感受野随层数指数增长（1→2→4→8）
- **Attention**：动态模式，每个 token 可以关注任意其他 token，感受野一层就能覆盖全部
- WaveNet 更高效（O(n log n)），但灵活性不如 Attention（O(n²)）

</details>

**Q3：** 为什么 FlattenConsecutive 用 `view` 而不是 `torch.cat`？
<details>
<summary>💡 提示</summary>

`view` 只改变 tensor 的形状（shape 和 stride），不移动数据，是 O(1) 操作。`torch.cat` 需要分配新内存并复制数据，是 O(n) 操作。在训练循环中，这个差异会累积成显著的性能差距。

</details>

---

*Good luck! 🚀*
