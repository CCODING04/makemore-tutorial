"""
05_batchnorm_backward.py - 简化版 BatchNorm 反向传播

BatchNorm 的逐步反传要 5-6 步，但其实也有一个简化公式！🎯

一行公式（与 1/n 有偏方差前向配套）：
  dhprebn = bngain * bnvar_inv / n * (
      n * dhpreact
    - dhpreact.sum(0)
    - bnraw * (dhpreact * bnraw).sum(0)
  )

⚠️ 口径必须与前向方差匹配：
  - 本仓库前向 bnvar = bndiff2.mean(0)，是 1/n 的有偏方差
    → 第三项系数是 1（本脚本采用）
  - 若前向改成无偏方差 bndiff2.sum(0)/(n-1)
    → 第三项系数才是 n/(n-1)
  - 两者混用（如 1/n 前向配 n/(n-1) 系数）会产生 ~4.6e-05 的系统性偏差
    （n=32 时恰好略超 1e-5 阈值，容易被误当浮点噪声）

一行公式做了什么？
  1. 处理均值减法的梯度传播
  2. 处理方差归一化的梯度传播
  3. 处理 BatchNorm 缩放 (bngain) 的梯度传播

来验证它和逐步版本 / autograd 是不是一致！
"""

import os
import math
import torch
import torch.nn.functional as F

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
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8")

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

# BatchNorm
bnmeani = hprebn.mean(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = bndiff2.mean(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)

print(f"Loss = {loss.item():.4f}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 1：逐步 BatchNorm 反向传播（5-6 步）
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 1：逐步 BatchNorm 反向传播")
print("=" * 60)

# 先拿到 dhpreact（从 loss 到 hpreact 的梯度）
for p in parameters:
    p.grad = None
hpreact.retain_grad()   # ⚠️ hpreact 是非叶子节点，.grad 默认不保存，必须 retain_grad()
loss.backward()
dhpreact_auto = hpreact.grad.clone()

# 手动逐步反传 BN
# hpreact = bngain * bnraw + bnbias → dbngain, dbnbias, dbnraw
dbngain = (dhpreact_auto * bnraw).sum(0, keepdim=True)
dbnbias = dhpreact_auto.sum(0, keepdim=True)
dbnraw = dhpreact_auto * bngain

# bnraw = bndiff * bnvar_inv
dbndiff = dbnraw * bnvar_inv
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)

# bnvar_inv = (bnvar + eps)^{-0.5}
dbnvar = dbnvar_inv * (-0.5) * (bnvar + 1e-5) ** -1.5

# bnvar = bndiff2.mean(0)
dbndiff2 = torch.ones_like(bndiff2) * (dbnvar / batch_size)

# bndiff2 = bndiff^2
dbndiff += dbndiff2 * 2.0 * bndiff

# bndiff = hprebn - bnmeani → dbnmeani
dbnmeani = -dbndiff.sum(0, keepdim=True)

# bnmeani = hprebn.mean(0)
dhprebn_step = dbndiff + torch.ones_like(hprebn) * (dbnmeani / batch_size)

print(f"  逐步 dhprebn: shape = {tuple(dhprebn_step.shape)}")
print(f"  前 3 行前 5 列:\n{dhprebn_step[:3, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 2：简化版 —— 一行公式 🪄
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 2：简化版 BatchNorm 反向传播（一行公式）")
print("=" * 60)
print()

n = batch_size

# 🪄 一行魔法公式（第三项系数为 1，与 1/n 有偏方差前向配套）：
dhprebn_simple = (bngain * bnvar_inv / n) * (
    n * dhpreact_auto
    - dhpreact_auto.sum(0)
    - bnraw * (dhpreact_auto * bnraw).sum(0)
)

print(f"  简化 dhprebn: shape = {tuple(dhprebn_simple.shape)}")
print(f"  前 3 行前 5 列:\n{dhprebn_simple[:3, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 方法 3：Autograd 直接对 hprebn 求梯度
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("方法 3：PyTorch Autograd 参考值")
print("=" * 60)

# 重新前向，这次 hprebn 保留梯度
for p in parameters:
    p.grad = None

emb2 = C[Xb]
embcat2 = emb2.view(emb2.shape[0], -1)
hprebn2 = embcat2 @ W1 + b1
hprebn2.retain_grad()

bnmeani2 = hprebn2.mean(0, keepdim=True)
bndiff2_2 = hprebn2 - bnmeani2
bndiff2_sq = bndiff2_2 ** 2
bnvar2 = bndiff2_sq.mean(0, keepdim=True)
bnvar_inv2 = (bnvar2 + 1e-5) ** -0.5
bnraw2 = bndiff2_2 * bnvar_inv2
hpreact2 = bngain * bnraw2 + bnbias

h2 = torch.tanh(hpreact2)
logits2 = h2 @ W2 + b2
loss2 = F.cross_entropy(logits2, Yb)
loss2.backward()

dhprebn_auto = hprebn2.grad.clone()
print(f"  Autograd dhprebn: shape = {tuple(dhprebn_auto.shape)}")
print(f"  前 3 行前 5 列:\n{dhprebn_auto[:3, :5].detach()}")
print()

# ═══════════════════════════════════════════════════════════════
# 三种方法对比
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("🔍 三种方法对比")
print("=" * 60)

diff_step_vs_auto = (dhprebn_step - dhprebn_auto).abs().max().item()
diff_simple_vs_auto = (dhprebn_simple - dhprebn_auto).abs().max().item()
diff_step_vs_simple = (dhprebn_step - dhprebn_simple).abs().max().item()

THRESH = 1e-5
ok_step = diff_step_vs_auto < THRESH
ok_simple = diff_simple_vs_auto < THRESH
ok_pair = diff_step_vs_simple < THRESH
all_ok = ok_step and ok_simple and ok_pair

print(f"  逐步 vs Autograd:  max diff = {diff_step_vs_auto:.2e}  {'✅' if ok_step else '❌'}")
print(f"  简化 vs Autograd:  max diff = {diff_simple_vs_auto:.2e}  {'✅' if ok_simple else '❌'}")
print(f"  逐步 vs 简化:      max diff = {diff_step_vs_simple:.2e}  {'✅' if ok_pair else '❌'}")
print(f"  (判定阈值 {THRESH:.0e}，任一超阈值脚本将以非零码退出)")
print()

# ─── 对拍误差条形图（存 images/，G4）──────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ['step vs autograd', 'simple vs autograd', 'step vs simple']
    diffs = [diff_step_vs_auto, diff_simple_vs_auto, diff_step_vs_simple]
    colors = ['#2a9d8f' if d < THRESH else '#e76f51' for d in diffs]
    bars = ax.bar(names, diffs, color=colors)
    ax.axhline(THRESH, color='#e76f51', linestyle='--', linewidth=1, label=f'threshold 1e-5')
    ax.set_yscale('log')
    ax.set_ylabel('max abs diff (float32)')
    ax.set_title('BatchNorm backward: manual vs autograd (n=32)')
    for b, d in zip(bars, diffs):
        ax.text(b.get_x() + b.get_width() / 2, d * 1.3, f'{d:.1e}', ha='center', fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    images_dir = os.path.join(script_dir, '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    fig.savefig(os.path.join(images_dir, 'bn_backward_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  🖼️ 对拍误差图已保存: images/bn_backward_comparison.png")
    print()
except Exception as e:
    print(f"  (跳过绘图: {e})")
    print()

# ═══════════════════════════════════════════════════════════════
# 原理解释
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("💡 公式拆解")
print("=" * 60)
print("""
BatchNorm 前向：
  μ = mean(x, dim=0)           ← batch 均值
  σ² = var(x, dim=0)           ← batch 方差（本仓库 = bndiff².mean(0)，1/n 有偏口径）
  x̂ = (x - μ) / √(σ²+ε)    ← 标准化
  y = γ · x̂ + β               ← 缩放平移

反向传播（简化公式，与 1/n 有偏前向配套）：
  dhprebn = (γ / √(σ²+ε)) / n * (
      n · dhpreact              ← 直接传播
    - Σ(dhpreact)               ← 均值减法的修正
    - x̂ · Σ(dhpreact · x̂)      ← 方差归一化的修正（系数 1）
  )

三个项的含义：
  📌 n · dhpreact: 直接把梯度传回来（因为 x̂ 包含了 x）
  📌 -Σ(dhpreact): 减去均值 μ 导致的修正（μ 依赖于所有 x）
  📌 -x̂ · Σ(dhpreact · x̂): 除以标准差 σ 导致的修正

⚠️ 第三项的系数由前向方差口径决定：
  - 前向用 1/n 有偏方差（本仓库，与 PyTorch BN training 一致）→ 系数 1
  - 前向用 1/(n-1) 无偏方差 → 系数 n/(n-1)
  - 口径混用会引入 ~4.6e-05（n=32）的系统性偏差，不是浮点噪声

关键洞察：
  - BatchNorm 的梯度依赖于整个 batch 的统计量
  - 这就是为什么 BatchNorm 的行为和 batch size 有关
""")
print()

# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
if all_ok:
    print("🎉 简化版 BatchNorm 反向传播验证通过（全部 max diff < 1e-5）！")
    print()
    print("   记住这个一行公式（面试利器，配套 1/n 有偏方差前向）：")
    print("   dhprebn = bngain * bnvar_inv / n * (")
    print("       n * dhpreact - dhpreact.sum(0)")
    print("       - bnraw * (dhpreact * bnraw).sum(0))")
    print("   )")
else:
    print("⚠️ 验证失败！存在 max diff >= 1e-5 的对拍项。")
    print("   请检查简化公式的方差口径是否与前向一致：")
    print("   1/n 有偏前向 → 第三项系数 1；1/(n-1) 无偏前向 → n/(n-1)")
    print("=" * 60)
    sys.exit(1)
print("=" * 60)
