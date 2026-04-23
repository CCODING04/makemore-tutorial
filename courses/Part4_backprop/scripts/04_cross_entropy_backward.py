"""
04_cross_entropy_backward.py - 简化版 CrossEntropy 反向传播

前面我们用 8 步才搞定 loss → dlogits 的反传。
但其实整个 CrossEntropy 的反向传播可以压缩成一行代码！🎯

魔法公式：
  dlogits = F.softmax(logits, 1)
  dlogits[range(n), Yb] -= 1
  dlogits /= n

等价于 loss = F.cross_entropy(logits, Yb) 的反向传播。
这节课就来验证这个魔法 ✨
"""

import os
import math
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 固定随机种子 ───────────────────────────────────────────────
torch.manual_seed(42)

# ─── 数据加载 ───────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

with open(data_path, 'r') as f:
    words = f.read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
Xtr, Ytr = build_dataset(words[:n1])

# ─── 网络参数 ──────────────────────────────────────────────────
n_embd = 10
n_hidden = 64

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / math.sqrt(n_embd * block_size)
b1 = torch.zeros(n_hidden)
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, bngain, bnbias, W2, b2]
for p in parameters:
    p.requires_grad = True

# ─── Mini-batch ────────────────────────────────────────────────
batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# ═══════════════════════════════════════════════════════════════
# 前向传播
# ═══════════════════════════════════════════════════════════════
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1

bnmeani = hprebn.mean(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = bndiff2.mean(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)
logits = h @ W2 + b2

# ═══════════════════════════════════════════════════════════════
# 方法 1：逐步展开的 CrossEntropy 反向传播（8 步）
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 1：逐步展开的 CrossEntropy 反向传播（8 步）")
print("=" * 60)

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[torch.arange(batch_size), Yb].mean()

print(f"Loss = {loss.item():.4f}")
print()

# 逐步反传
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[torch.arange(batch_size), Yb] = -1.0 / batch_size
dprobs = dlogprobs * (1.0 / probs)
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts = dprobs * counts_sum_inv
dcounts_sum = dcounts_sum_inv * (-counts_sum ** -2)
dcounts += torch.ones_like(counts) * dcounts_sum
dnorm_logits = dcounts * counts
dlogits_step = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
max_indices = logits.argmax(dim=1)
dlogit_maxes_grad = torch.zeros_like(logits)
dlogit_maxes_grad[torch.arange(batch_size), max_indices] = dlogit_maxes.squeeze()
dlogits_step += dlogit_maxes_grad

print(f"dlogits (逐步): shape = {tuple(dlogits_step.shape)}")
print(f"  前 5 行前 5 列:\n{dlogits_step[:5, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 2：简化版 —— 一行代码的魔法 🪄
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 2：简化版 CrossEntropy 反向传播（一行代码）")
print("=" * 60)
print()

n = batch_size

# 🪄 魔法三行：
dlogits_simple = F.softmax(logits, 1)       # 先算 softmax
dlogits_simple[range(n), Yb] -= 1           # 正确类别位置减 1
dlogits_simple /= n                          # 除以 batch size

print(f"dlogits (简化): shape = {tuple(dlogits_simple.shape)}")
print(f"  前 5 行前 5 列:\n{dlogits_simple[:5, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 3：autograd 参考值
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 3：PyTorch Autograd 参考值")
print("=" * 60)

logits_ref = logits.clone().detach().requires_grad_(True)
loss_ref = F.cross_entropy(logits_ref, Yb)
loss_ref.backward()
dlogits_autograd = logits_ref.grad

print(f"dlogits (autograd): shape = {tuple(dlogits_autograd.shape)}")
print(f"  前 5 行前 5 列:\n{dlogits_autograd[:5, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 验证三种方法的一致性
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("🔍 三种方法对比")
print("=" * 60)

diff_step_vs_auto = (dlogits_step - dlogits_autograd).abs().max().item()
diff_simple_vs_auto = (dlogits_simple - dlogits_autograd).abs().max().item()
diff_step_vs_simple = (dlogits_step - dlogits_simple).abs().max().item()

print(f"  逐步 vs Autograd:  max diff = {diff_step_vs_auto:.2e}  {'✅' if diff_step_vs_auto < 1e-5 else '❌'}")
print(f"  简化 vs Autograd:  max diff = {diff_simple_vs_auto:.2e}  {'✅' if diff_simple_vs_auto < 1e-5 else '❌'}")
print(f"  逐步 vs 简化:      max diff = {diff_step_vs_simple:.2e}  {'✅' if diff_step_vs_simple < 1e-5 else '❌'}")
print()

# ═══════════════════════════════════════════════════════════════
# 原理解释
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("💡 原理解释")
print("=" * 60)
print("""
CrossEntropy Loss = -log(softmax(logits)[correct_class])

反向传播：
  ∂L/∂logits_i = softmax(logits)_i - 𝟙(i == correct_class)
                ——————————————————————————————————
                再除以 N (因为 loss 取了 mean)

所以：
  dlogits = softmax(logits)          ← 每个位置都是概率值
  dlogits[正确位置] -= 1              ← 正确类别位置减 1
  dlogits /= N                        ← 除以 batch size

直觉理解：
  - 模型预测的概率已经在正确类别上比较高了 → softmax 值大
  - 减 1 相当于说 "还需要再高一点"
  - 其他类别位置梯度 = 概率值本身，表示 "需要把这些压低"
  - 整体就是：让正确类别的概率 ↑，其他类别的概率 ↓
""")
print()

# ═══════════════════════════════════════════════════════════════
# 可视化 dlogits 热力图
# ═══════════════════════════════════════════════════════════════
print("📊 生成热力图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 热力图 1：简化版 dlogits
im1 = axes[0].imshow(dlogits_simple.detach(), cmap='RdBu_r', aspect='auto')
axes[0].set_title('简化版 dlogits\n(softmax - one_hot) / N', fontsize=12)
axes[0].set_xlabel('字符类别 (vocab)')
axes[0].set_ylabel('样本 (batch)')
plt.colorbar(im1, ax=axes[0])

# 标注正确类别位置
for i in range(min(batch_size, 16)):
    axes[0].plot(Yb[i].item(), i, 'kx', markersize=6)

# 热力图 2：每行梯度绝对值
row_grad_norms = dlogits_simple.detach().abs().sum(dim=1)
im2 = axes[1].bar(range(batch_size), row_grad_norms.numpy(), color='steelblue', alpha=0.7)
axes[1].set_title('每个样本的梯度 L1 范数', fontsize=12)
axes[1].set_xlabel('样本索引')
axes[1].set_ylabel('|dlogits| L1 norm')
axes[1].axhline(row_grad_norms.mean().item(), color='red', linestyle='--', label=f'mean={row_grad_norms.mean():.4f}')
axes[1].legend()

plt.tight_layout()

# 保存到当前脚本所在目录
save_path = os.path.join(script_dir, 'dlogits_heatmap.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 热力图已保存到: {save_path}")
print()
print("=" * 60)
print("🎉 简化版 CrossEntropy 反向传播验证完成！")
print("   记住这三行：softmax → 减1 → 除以N")
print("=" * 60)
