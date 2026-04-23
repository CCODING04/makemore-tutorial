#!/usr/bin/env python3
"""
04_flatten_consecutive.py - FlattenConsecutive 层：层次化融合的核心

WaveNet 的关键操作：把连续的 n 个 embedding 拼在一起，逐步融合。

FlattenConsecutive(n)：
  输入 shape: (B, T, C)
  输出 shape: (B, T//n, C*n)
  即把每 n 个连续位置的 embedding 拼接成一个更长的向量。

关键洞察：
  1. 用 view + transpose 比 torch.cat 高效得多
  2. Linear 层天然支持多维输入——只在最后一个维度做矩阵乘法
  3. 这就是 WaveNet 的"逐步融合"：2 chars → bigram → 4-gram → 8-gram
"""

import torch


# ═══════════════════════════════════════════════════════════════════
#  FlattenConsecutive 层
# ═══════════════════════════════════════════════════════════════════

class FlattenConsecutive:
    """
    把连续 n 个位置的 embedding 拼接成一个。

    输入:  (B, T, C)
    输出:  (B, T//n, C*n)

    示例：
      FlattenConsecutive(2)
      (4, 8, 10) → (4, 4, 20)
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        assert T % self.n == 0, f"T={T} 不能被 n={self.n} 整除"
        # 方法：view 成 (B, T//n, n, C) 然后 reshape 成 (B, T//n, n*C)
        x = x.view(B, T // self.n, C * self.n)
        self.out = x
        return self.out

    def parameters(self):
        return []


# ═══════════════════════════════════════════════════════════════════
#  演示
# ═══════════════════════════════════════════════════════════════════

print("═══ FlattenConsecutive 演示 ═══\n")

# === 演示 1：基本 reshape ===
torch.manual_seed(42)
x = torch.randn(4, 8, 10)
print(f"输入 shape: {x.shape}")

fc = FlattenConsecutive(2)
y = fc(x)
print(f"FlattenConsecutive(2) 输出: {y.shape}")
print(f"  期望: (4, 4, 20)")
assert y.shape == (4, 4, 20), f"shape 错误: {y.shape}"

# === 演示 2：逐步 flatten（模拟 WaveNet） ===
print(f"\n--- 模拟 WaveNet 的层次融合 ---")
x = torch.randn(4, 8, 10)  # 8 个字符，每个 10 维 embedding
print(f"初始:    {x.shape}  (8 chars × 10 dim)")

fc2 = FlattenConsecutive(2)
x = fc2(x)
print(f"FC(2):   {x.shape}  → 4 个 bigram, 每个 20 维")

x = fc2(x)
print(f"FC(2):   {x.shape}  → 2 个 4-gram, 每个 40 维")

x = fc2(x)
print(f"FC(2):   {x.shape}  → 1 个 8-gram, 80 维")


# === 演示 3：Linear 支持多维输入 ===
print(f"\n--- Linear 层天然支持多维输入 ---")

# Linear 层只在最后一个维度做矩阵乘法
# 所以 (B, T, C) @ (C, H) = (B, T, H) —— 不需要先展平！
x = torch.randn(4, 8, 10)
W = torch.randn(10, 20)
y = x @ W
print(f"x shape: {x.shape}")
print(f"W shape: {W.shape}")
print(f"x @ W shape: {y.shape}")
print(f"  → Linear 自动在最后两个维度广播矩阵乘法")


# === 演示 4：view vs cat 对比 ===
print(f"\n--- view vs torch.cat 对比 ---")

x = torch.randn(4, 8, 10)

# 方法 1: view（高效，无内存拷贝）
y1 = x.view(4, 4, 20)
print(f"view 方法:  {y1.shape}")

# 方法 2: cat（低效，有内存拷贝）
left = x[:, ::2, :]    # (4, 4, 10) - 偶数位置
right = x[:, 1::2, :]  # (4, 4, 10) - 奇数位置
y2 = torch.cat([left, right], dim=2)
print(f"cat 方法:   {y2.shape}")

print(f"结果相等: {torch.allclose(y1, y2)}")
# 注意：view 和 cat 的元素顺序不一定相同
# 但在 WaveNet 中，我们要的是连续位置的拼接，view 更直观


# === 演示 5：完整层次融合流程 ===
print(f"\n--- 完整 WaveNet 层次融合流程 ---")
torch.manual_seed(42)

B, T, C = 4, 8, 10
x = torch.randn(B, T, C)
print(f"输入: {x.shape}  (B={B}, T={T} chars, C={C} dim)")

# 第 1 层：2 chars → bigram
fc1 = FlattenConsecutive(2)
x = fc1(x)
print(f"  FC(2) → {x.shape}  (4 bigrams × 20 dim)")

# Linear 作用于最后一维：20 → 10
W1 = torch.randn(20, 10) / 20 ** 0.5
x = x @ W1
print(f"  Linear(20→10) → {x.shape}")

# 第 2 层：2 bigrams → 4-gram
x = fc1(x)
print(f"  FC(2) → {x.shape}  (2 × 20 dim)")

W2 = torch.randn(20, 10) / 20 ** 0.5
x = x @ W2
print(f"  Linear(20→10) → {x.shape}")

# 第 3 层：2 4-grams → 8-gram
x = fc1(x)
print(f"  FC(2) → {x.shape}  (1 × 20 dim)")

W3 = torch.randn(20, 10) / 20 ** 0.5
x = x @ W3
print(f"  Linear(20→10) → {x.shape}")

print(f"\n最终 shape: {x.shape}  → 可以直接送进输出层")

print(f"""
═══ 总结 ═══

FlattenConsecutive 是 WaveNet 的核心操作：
  1. (B, T, C) → (B, T//n, C*n)：连续 n 个位置拼接
  2. 配合 Linear 层：逐步从字符 → bigram → 4-gram → 8-gram
  3. Linear 天然支持多维输入，只在最后一维做矩阵乘法
  4. 用 view 比 cat 高效（无内存拷贝）
  5. 层次融合让网络学到"从局部到全局"的特征层级
""")
