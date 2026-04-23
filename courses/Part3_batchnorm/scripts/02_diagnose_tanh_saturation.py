"""
02_diagnose_tanh_saturation.py - 诊断 Tanh 饱和问题

本脚本在训练几步后检查隐藏层 h = tanh(hpreact) 的分布。
如果 tanh 输入（hpreact）绝对值太大，输出会"饱和"在 ±1 附近，
导致梯度接近 0（梯度消失），这些神经元"死掉"了。

诊断方法：
  - 统计 |h| > 0.99 的比例
  - 可视化 h 的分布直方图
"""

import os
import torch
import torch.nn.functional as F
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

# ─── 构建数据集 ─────────────────────────────────────────────────
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
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

# ─── 模型参数（基线，未修正 W1）────────────────────────────────
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, n_embd), generator=g)
# 注意：W1 没有缩放，fan_in = 30，randn 的标准差 = 1
# hpreact = emb @ W1 + b1 的方差会很大 → tanh 饱和
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# ─── 训练几步 ───────────────────────────────────────────────────
print("训练 1000 步以观察 tanh 饱和情况...")
for i in range(1000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    hpreact = emb.view(-1, n_embd * block_size) @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1 * p.grad

    if (i + 1) % 200 == 0:
        print(f"  step {i+1:4d} | loss = {loss.item():.4f}")

# ─── 在完整训练集上诊断 tanh ────────────────────────────────────
print("\n═══ Tanh 饱和诊断 ═══")
with torch.no_grad():
    emb = C[Xtr]
    hpreact = emb.view(-1, n_embd * block_size) @ W1 + b1
    h = torch.tanh(hpreact)

    # 统计饱和度
    saturated = (h.abs() > 0.99).float().mean().item()
    very_saturated = (h.abs() > 0.999).float().mean().item()

    print(f"tanh 输出 |h| > 0.99  的比例: {saturated * 100:.2f}%")
    print(f"tanh 输出 |h| > 0.999 的比例: {very_saturated * 100:.2f}%")
    print(f"tanh 输出均值: {h.mean().item():.4f}")
    print(f"tanh 输出标准差: {h.std().item():.4f}")
    print(f"hpreact 均值: {hpreact.mean().item():.4f}")
    print(f"hpreact 标准差: {hpreact.std().item():.4f}")

    # 逐神经元统计：有多少神经元在几乎所有样本上都饱和了
    per_neuron_saturation = (h.abs() > 0.99).float().mean(dim=0)
    dead_neurons = (per_neuron_saturation > 0.9).sum().item()
    print(f"\n超过 90% 样本上饱和的神经元: {dead_neurons}/{n_hidden}")

# ─── 可视化 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：tanh 输出 h 的分布
axes[0].hist(h.detach().numpy().flatten(), bins=50, density=True, alpha=0.7,
             color='steelblue', edgecolor='white')
axes[0].axvline(x=-0.99, color='red', linestyle='--', linewidth=1.5, label='±0.99')
axes[0].axvline(x=0.99, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title(f'tanh 输出分布 (饱和率: {saturated*100:.1f}%)', fontsize=13)
axes[0].set_xlabel('h 值')
axes[0].set_ylabel('密度')
axes[0].legend()

# 右图：hpreact（tanh 之前）的分布
axes[1].hist(hpreact.detach().numpy().flatten(), bins=50, density=True, alpha=0.7,
             color='coral', edgecolor='white')
axes[1].set_title(f'hpreact 分布 (标准差: {hpreact.std().item():.2f})', fontsize=13)
axes[1].set_xlabel('hpreact 值')
axes[1].set_ylabel('密度')

plt.suptitle('Tanh 饱和诊断 — hpreact 方差过大导致 tanh 饱和', fontsize=14, y=1.02)
plt.tight_layout()

save_path = os.path.join(script_dir, 'tanh_saturation.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存到: {save_path}")

# ─── 解释 ───────────────────────────────────────────────────────
print(f"""
═══ 诊断结论 ═══

问题：W1 没有缩放 → hpreact 方差过大 → tanh 饱和

为什么 tanh 饱和是问题？
  - tanh 在 ±1 附近梯度 ≈ 0
  - 梯度为 0 → 反向传播时该神经元不更新 → "死掉"
  - 即使输入变化，输出几乎不变，学习停滞

解决方案（下个脚本）：
  - Kaiming 初始化：用合适的 gain / √fan_in 缩放 W1
  - 使 hpreact 保持合理范围，tanh 工作在线性区附近
""")
