"""
03_kaiming_init.py - Kaiming/He 初始化

核心思想：每层的激活值方差应保持稳定（不会指数增长或消失）。
对于 tanh 激活函数，Kaiming 初始化使用 gain = 5/3：
    W = randn(...) * (5/3) / sqrt(fan_in)

本脚本：
  1. 用 Kaiming 初始化修正 W1
  2. 重新训练，观察 tanh 饱和改善
  3. 打印不同激活函数对应的 gain 值
"""

import os
import math
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

# ─── 模型参数 ───────────────────────────────────────────────────
n_embd = 10
n_hidden = 200
fan_in = n_embd * block_size  # = 30

# ─── Kaiming 初始化的 gain 值表 ─────────────────────────────────
print("═══ Kaiming 初始化 Gain 值表 ═══")
print(f"{'激活函数':<15} {'gain':<10} {'公式':<30}")
print("-" * 55)
gain_table = [
    ("Linear", 1.0, "1"),
    ("Tanh", 5 / 3, "5/3"),
    ("ReLU", math.sqrt(2), "√2"),
    ("Leaky ReLU", math.sqrt(2 / (1 + 0.01**2)), "√(2/(1+α²))"),
    ("Sigmoid", 1.0, "1 (不推荐)"),
]
for name, gain, formula in gain_table:
    print(f"{name:<15} {gain:<10.4f} {formula:<30}")

tanh_gain = 5 / 3
print(f"\n本脚本使用 tanh gain = {tanh_gain:.4f}")
print(f"fan_in = {fan_in}")
print(f"缩放因子 = {tanh_gain:.4f} / √{fan_in} = {tanh_gain / math.sqrt(fan_in):.4f}")


# ─── 训练函数 ───────────────────────────────────────────────────
def train_model(use_kaiming, n_steps=10000, lr=0.1):
    """训练模型，返回 loss 历史和中间激活值"""
    g = torch.Generator().manual_seed(42)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((fan_in, n_hidden), generator=g)
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.zeros(vocab_size)

    if use_kaiming:
        W1 = W1 * (tanh_gain / math.sqrt(fan_in))

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    losses = []
    for i in range(n_steps):
        ix = torch.randint(0, Xtr.shape[0], (32,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        emb = C[Xb]
        hpreact = emb.view(-1, fan_in) @ W1 + b1
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -lr * p.grad

        losses.append(loss.item())

    return losses, C, W1, b1, W2, b2


# ─── 对比训练 ───────────────────────────────────────────────────
print("\n═══ 训练对比: 基线 vs Kaiming 初始化 ═══")

print("\n--- 基线（无 Kaiming）---")
losses_no_kaiming, *_ = train_model(use_kaiming=False)
print(f"  初始 loss: {losses_no_kaiming[0]:.4f}")
print(f"  最终 loss: {losses_no_kaiming[-1]:.4f}")

print("\n--- Kaiming 初始化 ---")
losses_kaiming, C_k, W1_k, b1_k, W2_k, b2_k = train_model(use_kaiming=True)
print(f"  初始 loss: {losses_kaiming[0]:.4f}")
print(f"  最终 loss: {losses_kaiming[-1]:.4f}")

# ─── 评估 dev loss ─────────────────────────────────────────────


def eval_loss(C, W1, b1, W2, b2, X, Y):
    with torch.no_grad():
        emb = C[X]
        h = torch.tanh(emb.view(-1, fan_in) @ W1 + b1)
        logits = h @ W2 + b2
        return F.cross_entropy(logits, Y).item()


dev_loss_no_kaiming = eval_loss(*train_model(False)[1:], Xdev, Ydev)
dev_loss_kaiming = eval_loss(C_k, W1_k, b1_k, W2_k, b2_k, Xdev, Ydev)

print(f"\n═══ 验证集 Loss ═══")
print(f"基线（无 Kaiming）: {dev_loss_no_kaiming:.4f}")
print(f"Kaiming 初始化:     {dev_loss_kaiming:.4f}")

# ─── 检查 Kaiming 模型的 tanh 饱和度 ────────────────────────────
with torch.no_grad():
    emb = C_k[Xtr]
    hpreact = emb.view(-1, fan_in) @ W1_k + b1_k
    h = torch.tanh(hpreact)
    sat = (h.abs() > 0.99).float().mean().item()
    print(f"\nKaiming 模型 tanh 饱和率 (|h|>0.99): {sat*100:.2f}%")

# ─── 可视化 loss 曲线对比 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 平滑 loss
def smooth(losses, window=100):
    return [sum(losses[max(0,i-window):i+1]) / min(i+1, window) for i in range(len(losses))]

axes[0].plot(smooth(losses_no_kaiming), label='基线（无 Kaiming）', alpha=0.8)
axes[0].plot(smooth(losses_kaiming), label='Kaiming 初始化', alpha=0.8)
axes[0].set_title('训练 Loss 对比', fontsize=13)
axes[0].set_xlabel('训练步数')
axes[0].set_ylabel('Loss')
axes[0].legend()

# hpreact 方差对比
g = torch.Generator().manual_seed(42)
C_tmp = torch.randn((vocab_size, n_embd), generator=g)
W1_base = torch.randn((fan_in, n_hidden), generator=g)
W1_kaim = torch.randn((fan_in, n_hidden), generator=g) * (tanh_gain / math.sqrt(fan_in))

with torch.no_grad():
    emb = C_tmp[Xtr[:1000]]
    x = emb.view(-1, fan_in)
    preact_base = x @ W1_base
    preact_kaim = x @ W1_kaim

axes[1].hist(preact_base.detach().numpy().flatten(), bins=50, density=True,
             alpha=0.5, label=f'基线 (std={preact_base.std():.2f})', color='coral')
axes[1].hist(preact_kaim.detach().numpy().flatten(), bins=50, density=True,
             alpha=0.5, label=f'Kaiming (std={preact_kaim.std():.2f})', color='steelblue')
axes[1].set_title('hpreact 初始分布对比', fontsize=13)
axes[1].set_xlabel('hpreact 值')
axes[1].legend()

plt.suptitle('Kaiming 初始化效果对比', fontsize=14, y=1.02)
plt.tight_layout()

save_path = os.path.join(script_dir, 'kaiming_init.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"图表已保存到: {save_path}")

# ─── 总结 ───────────────────────────────────────────────────────
print(f"""
═══ 总结 ═══

Kaiming 初始化公式: W ~ N(0, gain²/fan_in)
  - gain 取决于激活函数（tanh → 5/3, relu → √2）
  - fan_in 是输入维度

为什么有效？
  - 前向传播时：激活值方差保持稳定
  - 反向传播时：梯度方差保持稳定
  - tanh 不会过早饱和，梯度不会消失

注意：在现代深度学习中，残差连接、LayerNorm/BatchNorm 等技术
使得网络对初始化不那么敏感，但理解原理仍然很重要。
""")
