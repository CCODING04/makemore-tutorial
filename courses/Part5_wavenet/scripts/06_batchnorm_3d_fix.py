#!/usr/bin/env python3
"""
06_batchnorm_3d_fix.py - 修复 BatchNorm1D 的 3D 输入支持

WaveNet 中间层的输出是 3D tensor (B, T, C)，但之前 BatchNorm1d
只在 dim=0 上 reduce，对 3D 输入会得到错误的结果。

修复：当 x.ndim == 3 时，在 dim=(0, 1) 上同时 reduce。

关键点：
  1. BatchNorm 的统计量应该在所有 batch 和 time 步上计算
  2. running_mean/running_var 始终是 1D (dim,) tensor
  3. 3D 输入时 unsqueeze 两次用于广播
"""

import torch


# ═══════════════════════════════════════════════════════════════════
#  修复前的 BatchNorm（有 bug）
# ═══════════════════════════════════════════════════════════════════

class BatchNorm1dBuggy:
    """只支持 2D 输入的 BatchNorm"""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # BUG: 不管 2D 还是 3D，都只在 dim=0 上 reduce
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean.unsqueeze(0)
            xvar = self.running_var.unsqueeze(0)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ═══════════════════════════════════════════════════════════════════
#  修复后的 BatchNorm
# ═══════════════════════════════════════════════════════════════════

class BatchNorm1d:
    """支持 2D 和 3D 输入的 BatchNorm"""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                # 2D 输入: (B, C) → 在 dim=0 上 reduce
                dim_reduce = 0
            else:
                # 3D 输入: (B, T, C) → 在 dim=(0,1) 上同时 reduce
                dim_reduce = (0, 1)

            xmean = x.mean(dim=dim_reduce, keepdim=True)
            xvar = x.var(dim=dim_reduce, keepdim=True, unbiased=False)

            with torch.no_grad():
                if x.ndim == 2:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze(0)
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze(0)
                else:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze((0, 1))
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze((0, 1))
        else:
            if x.ndim == 2:
                xmean = self.running_mean.unsqueeze(0)
                xvar = self.running_var.unsqueeze(0)
            else:
                xmean = self.running_mean.unsqueeze(0).unsqueeze(0)
                xvar = self.running_var.unsqueeze(0).unsqueeze(0)

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ═══════════════════════════════════════════════════════════════════
#  演示 bug 和修复
# ═══════════════════════════════════════════════════════════════════

print("═══ BatchNorm 3D Bug 演示 ═══\n")

torch.manual_seed(42)
dim = 10

# === 2D 输入：两个版本结果一致 ===
print("--- 2D 输入 (B=32, C=10) ---")
x2d = torch.randn(32, dim)

bn_buggy = BatchNorm1dBuggy(dim)
bn_fixed = BatchNorm1d(dim)

y_buggy_2d = bn_buggy(x2d)
y_fixed_2d = bn_fixed(x2d)

print(f"  Buggy 输出 shape: {y_buggy_2d.shape}")
print(f"  Fixed 输出 shape: {y_fixed_2d.shape}")
print(f"  Buggy running_mean shape: {bn_buggy.running_mean.shape}")
print(f"  Fixed running_mean shape: {bn_fixed.running_mean.shape}")
print(f"  结果一致: {torch.allclose(y_buggy_2d, y_fixed_2d, atol=1e-5)}")

# === 3D 输入：bug 版本结果错误 ===
print(f"\n--- 3D 输入 (B=32, T=4, C=10) ---")
x3d = torch.randn(32, 4, dim)

bn_buggy3 = BatchNorm1dBuggy(dim)
bn_fixed3 = BatchNorm1d(dim)

y_buggy_3d = bn_buggy3(x3d)
y_fixed_3d = bn_fixed3(x3d)

print(f"  Buggy 输出 shape: {y_buggy_3d.shape}")
print(f"  Fixed 输出 shape: {y_fixed_3d.shape}")

# Buggy 版本在 dim=0 reduce：mean shape 是 (1, 4, 10) 而不是 (1, 1, 10)
buggy_mean = x3d.mean(dim=0, keepdim=True)
fixed_mean = x3d.mean(dim=(0, 1), keepdim=True)
print(f"\n  Buggy mean shape: {buggy_mean.shape}  ← 错误！每个 T 位置独立归一化")
print(f"  Fixed mean shape: {fixed_mean.shape}  ← 正确！所有 (B,T) 一起归一化")

# 验证 running_mean 形状
print(f"\n  Buggy running_mean shape: {bn_buggy3.running_mean.shape}")
print(f"  Fixed running_mean shape: {bn_fixed3.running_mean.shape}")
# Buggy 版本的 running_mean 可能被 3D squeeze 破坏

# === 修复后的统计特性 ===
print(f"\n--- 修复后的统计特性验证 ---")
y = bn_fixed3(x3d)
print(f"  输出均值（应接近 0）: {y.mean():.6f}")
print(f"  输出标准差（应接近 1）: {y.std():.6f}")

# === 修复后 eval 模式 ===
print(f"\n--- Eval 模式验证 ---")
bn_fixed3.training = False
x3d_new = torch.randn(16, 4, dim)
y_eval = bn_fixed3(x3d_new)
print(f"  Eval 输出 shape: {y_eval.shape}")
print(f"  running_mean 保持 1D: {bn_fixed3.running_mean.shape == (dim,)}")

print(f"""
═══ 总结 ═══

BatchNorm1D 的 3D 修复：
  1. Bug：x.ndim==3 时在 dim=0 reduce → mean shape (1, T, C)
     - 每个时间步独立归一化，不是真正的 batch normalization
  2. Fix：x.ndim==3 时在 dim=(0,1) reduce → mean shape (1, 1, C)
     - 在所有 batch × time 步上归一化
  3. running_mean/running_var 始终是 1D tensor (dim,)
  4. Eval 时 3D 输入要 unsqueeze 两次来广播
""")
