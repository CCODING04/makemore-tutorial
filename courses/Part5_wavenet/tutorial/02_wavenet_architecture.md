# 02 — WaveNet 架构：层次化融合

> 🌊 从"一口吞下 8 个字符"到"先消化 2 个，再消化 4 个，最后消化 8 个"。

## 动机：为什么不能直接展平？

之前的方法：8 个字符的 embedding 展平成 80 维向量，直接送进 Linear 层。

```
8 chars → 展平成 80 维 → Linear(80, 200) → ...
```

**问题**：80 维向量一次性混合所有信息，网络很难学到"相邻字符之间的关系"。

**WaveNet 的思路**：逐步融合，从局部到全局。

```
8 chars → 4 bigrams → 2 fourgrams → 1 eightgram
```

每一步只融合相邻的两个向量，形成层次化的特征提取。

## 树状融合结构

```
                    [8-gram 表征]
                   /              \
            [4-gram]            [4-gram]
           /        \          /        \
      [bigram]  [bigram]  [bigram]  [bigram]
       /  \      /  \      /  \      /  \
      c1  c2   c3  c4    c5  c6    c7  c8
```

从底向上，每层把两个相邻向量融合成一个。

## 关键洞察：Linear 支持多维输入

```python
x = torch.randn(4, 8, 10)   # (B, T, C)
W = torch.randn(10, 20)      # (C, H)
y = x @ W                     # (4, 8, 20)  ← 自动广播！
```

Linear 层**只在最后一个维度做矩阵乘法**。前面的维度（B, T）自动保留。

这意味着我们可以：
1. 不展平 T 维度
2. 直接在 3D tensor 上做 Linear 变换
3. 在每一步 FlattenConsecutive 后接 Linear

## FlattenConsecutive 层

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        # (B, T, C) → (B, T//n, C*n)
        x = x.view(B, T // self.n, C * self.n)
        self.out = x
        return self.out
```

**示例**：

```
输入:  (B, 8, 10)   — 8 个字符，每个 10 维
FC(2): (B, 4, 20)   — 4 个 bigram，每个 20 维
FC(2): (B, 2, 40)   — 2 个 fourgram，每个 40 维
FC(2): (B, 1, 80)   — 1 个 eightgram，80 维
```

## WaveNet 完整架构

```python
model = Sequential([
    Embedding(vocab_size, n_embd),
    # 层 1: 8 → 4
    FlattenConsecutive(2),
    Linear(n_embd * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    # 层 2: 4 → 2
    FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    # 层 3: 2 → 1
    FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    # 输出层
    Linear(n_hidden, vocab_size),
])
```

## 扩大上下文窗口

block_size 从 3 增大到 8：

| block_size | 上下文示例 | 验证 loss |
|------------|-----------|-----------|
| 3 | `...e`mma → `m` | ~2.10 |
| 8 | `....emma` → `n` | ~2.02 |

仅靠更多上下文就能提升性能。但直接展平 8 个字符不如层次化融合效果好。

## view vs cat

```python
# 方法 1: view（高效，无内存拷贝）
x = x.view(B, T // 2, C * 2)

# 方法 2: cat（低效，有内存拷贝）
left = x[:, ::2, :]
right = x[:, 1::2, :]
x = torch.cat([left, right], dim=2)
```

view 只是改变了 tensor 的视图（stride 和 shape），**零拷贝**。cat 会分配新内存。在训练循环中，这个差异会累积。

## 代码参考

- 👉 [03_increase_context.py](../scripts/03_increase_context.py) — block_size 扩大到 8
- 👉 [04_flatten_consecutive.py](../scripts/04_flatten_consecutive.py) — FlattenConsecutive 演示
- 👉 [05_wavenet_architecture.py](../scripts/05_wavenet_architecture.py) — 完整 WaveNet 架构

## 下一步

架构搭好了，但训练时发现 BatchNorm 在 3D 输入上有 bug。

👉 [03 — 训练与 Bug 修复](03_training_and_bugs.md)
