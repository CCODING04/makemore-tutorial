# 03 — 训练与 Bug 修复

> 🐛 波折之后才能到达终点。BatchNorm 的 3D bug 是 WaveNet 训练的关键障碍。

## BatchNorm1D 的 3D Bug

### 问题

WaveNet 中间层的输出是 **3D tensor** `(B, T, C)`，但我们之前写的 BatchNorm1d 只考虑了 2D：

```python
# Bug 版本
xmean = x.mean(dim=0, keepdim=True)  # 对 3D 输入: shape (1, T, C)
```

这会导致每个时间步**独立**归一化，而不是在整个 batch × time 上归一化。

### 修复

```python
def __call__(self, x):
    if self.training:
        if x.ndim == 2:
            dim_reduce = 0          # (B, C) → 在 B 维度上 reduce
        else:
            dim_reduce = (0, 1)     # (B, T, C) → 在 B 和 T 维度上 reduce

        xmean = x.mean(dim=dim_reduce, keepdim=True)  # shape (1, 1, C) 或 (1, C)
        xvar = x.var(dim=dim_reduce, keepdim=True)
```

**关键**：
- `running_mean` 和 `running_var` 始终是 1D tensor `(C,)`
- 2D eval：`unsqueeze(0)` → `(1, C)` 广播到 `(B, C)`
- 3D eval：`unsqueeze(0).unsqueeze(0)` → `(1, 1, C)` 广播到 `(B, T, C)`

### 验证

```python
# 3D 输入 (B=32, T=4, C=10)
x3d = torch.randn(32, 4, 10)
bn = BatchNorm1d(10)
y = bn(x3d)

print(f"running_mean shape: {bn.running_mean.shape}")  # 应该是 (10,)
print(f"输出均值: {y.mean():.6f}")  # 应该接近 0
print(f"输出标准差: {y.std():.6f}")  # 应该接近 1
```

## 放大训练

把网络容量增大：

| 参数 | 小模型 | 放大模型 |
|------|--------|---------|
| n_embd | 10 | 24 |
| n_hidden | 68 | 128 |
| 参数量 | ~22K | ~170K |
| batch_size | 32 | 128 |
| 训练步数 | 20K | 50K |

```python
model = Sequential([
    Embedding(vocab_size, 24),  # 更大的 embedding
    FlattenConsecutive(2), Linear(48, 128, bias=False), BatchNorm1d(128), Tanh(),
    FlattenConsecutive(2), Linear(256, 128, bias=False), BatchNorm1d(128), Tanh(),
    FlattenConsecutive(2), Linear(256, 128, bias=False), BatchNorm1d(128), Tanh(),
    Linear(128, vocab_size),
])
```

### 性能对比

| 模型 | block_size | 验证 loss |
|------|-----------|-----------|
| MLP (Part 2) | 3 | ~2.10 |
| 深层 BN (Part 3) | 3 | ~2.07 |
| WaveNet 小模型 | 8 | ~2.07 |
| **WaveNet 放大** | **8** | **~1.99** |

验证 loss 首次降到 **2.0 以下**！

## 卷积预览：WaveNet 的另一种视角

我们用 FlattenConsecutive + Linear 实现的层次融合，本质上等价于 **Dilated Causal Convolution**（膨胀因果卷积）。

```
传统卷积（kernel=2）:
  c1 c2 c3 c4 c5 c6 c7 c8
  ├─┤ ├─┤ ├─┤ ├─┤ ├─┤ ├─┤ ├─┤
  (每次看 2 个相邻字符)

膨胀卷积（dilation=2）:
  c1 c2 c3 c4 c5 c6 c7 c8
  ├─────┤ ├─────┤ ├─────┤
  (每次看间隔 2 的字符)

膨胀卷积（dilation=4）:
  c1 c2 c3 c4 c5 c6 c7 c8
  ├───────────┤
  (每次看间隔 4 的字符)
```

这种卷积的感受野指数增长，和我们的层次融合完全等价。区别只是实现方式：
- **我们的方式**：FlattenConsecutive + Linear（更直观）
- **卷积方式**：Dilated Causal Conv（更高效，特别是对于长序列）

> 💡 这就是为什么论文叫 "WaveNet"——它最初是为音频波形设计的，用膨胀因果卷积处理超长序列。

## 代码参考

- 👉 [06_batchnorm_3d_fix.py](../scripts/06_batchnorm_3d_fix.py) — BatchNorm 3D bug 修复
- 👉 [07_scaled_wavenet.py](../scripts/07_scaled_wavenet.py) — 放大训练到 loss < 2.0

## 课后作业

动手实践！去完成练习：

👉 [Assignment 5](../../assignments/assignment_5/)
