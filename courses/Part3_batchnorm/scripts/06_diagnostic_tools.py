"""
06_diagnostic_tools.py - 训练诊断工具

最重要的 4 种诊断工具：
  1. 激活值分布 — 检查 tanh 饱和度
  2. 梯度分布 — 检查梯度消失/爆炸
  3. 权重梯度/数据比率 — 检查各层梯度是否匹配
  4. 更新/数据比率（最重要！）— 检查学习率是否合适

其中第 4 个是 Karpathy 最强调的指标：
  ratio = (lr × grad_std) / data_std
  理想值约 1e-3，过高说明学习率太大，过低说明学习率太小
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


# ─── BatchNorm1d ────────────────────────────────────────────────
class BatchNorm1d:
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
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        else:
            xmean = self.running_mean.unsqueeze(0)
            xvar = self.running_var.unsqueeze(0)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ─── 构建深层网络 ───────────────────────────────────────────────
n_embd = 10
n_hidden = 100
layers_config = [n_embd * block_size, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, vocab_size]

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, n_embd), generator=g)

layers = []
for i in range(len(layers_config) - 1):
    fan_in_i = layers_config[i]
    fan_out_i = layers_config[i + 1]
    W = torch.randn((fan_in_i, fan_out_i), generator=g) * (5 / 3) / math.sqrt(fan_in_i)
    b = torch.zeros(fan_out_i)
    layers.append({'type': 'linear', 'W': W, 'b': b, 'name': f'linear_{i}'})

    if i < len(layers_config) - 2:
        bn = BatchNorm1d(fan_out_i)
        layers.append({'type': 'batchnorm', 'bn': bn, 'name': f'bn_{i}'})
        layers.append({'type': 'activation', 'act': 'tanh', 'name': f'tanh_{i}'})

# 收集参数
parameters = [C]
param_names = ['C']
for layer in layers:
    if layer['type'] == 'linear':
        parameters.append(layer['W'])
        parameters.append(layer['b'])
        param_names.append(f"{layer['name']}.W")
        param_names.append(f"{layer['name']}.b")
    elif layer['type'] == 'batchnorm':
        parameters.extend(layer['bn'].parameters())
        param_names.append(f"{layer['name']}.gamma")
        param_names.append(f"{layer['name']}.beta")

for p in parameters:
    p.requires_grad = True

total_params = sum(p.nelement() for p in parameters)
print(f"模型参数总量: {total_params:,}")


# ─── 前向传播 ───────────────────────────────────────────────────
def forward(X, training=True):
    emb = C[X]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        if layer['type'] == 'linear':
            x = x @ layer['W'] + layer['b']
        elif layer['type'] == 'batchnorm':
            layer['bn'].training = training
            x = layer['bn'](x)
        elif layer['type'] == 'activation':
            if layer['act'] == 'tanh':
                x = torch.tanh(x)
    return x


# ─── 训练并收集诊断数据 ─────────────────────────────────────────
max_steps = 1000
batch_size = 32
lr = 0.1

# 诊断数据收集
ud_ratio_history = {name: [] for name in param_names}  # update/data ratio
activation_stats = []   # 激活值统计
gradient_stats = []     # 梯度统计

print(f"\n═══ 训练 {max_steps} 步并收集诊断数据 ═══")

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = forward(Xb, training=True)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    # 更新参数
    for p in parameters:
        p.data += -lr * p.grad

    # ─── 收集诊断数据（每 10 步）───────────────────────────────
    if i % 10 == 0:
        # 诊断 4: 更新/数据比率
        for j, (p, name) in enumerate(zip(parameters, param_names)):
            if p.ndim >= 2:  # 只看权重矩阵，不看偏置
                update_std = (lr * p.grad).std().item()
                data_std = p.data.std().item()
                ratio = update_std / data_std if data_std > 0 else 0
                ud_ratio_history[name].append(ratio)

    if (i + 1) % 200 == 0:
        print(f"  step {i+1:5d} | loss = {loss.item():.4f}")

# ─── 在一批数据上做完整诊断 ──────────────────────────────────────
print(f"\n═══ 诊断工具 1: 激活值分布（tanh 饱和度）═══")
ix = torch.randint(0, Xtr.shape[0], (256,))
Xb, Yb = Xtr[ix], Ytr[ix]

with torch.no_grad():
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)

    activation_data = {}
    for layer in layers:
        if layer['type'] == 'linear':
            x = x @ layer['W'] + layer['b']
            activation_data[layer['name'] + '_pre'] = x.clone()
        elif layer['type'] == 'batchnorm':
            layer['bn'].training = False
            x = layer['bn'](x)
        elif layer['type'] == 'activation':
            if layer['act'] == 'tanh':
                x = torch.tanh(x)
                activation_data[layer['name']] = x.clone()

    for name, act in activation_data.items():
        if 'tanh' in name:
            sat = (act.abs() > 0.99).float().mean().item()
            print(f"  {name:15s} | mean={act.mean():.4f} | std={act.std():.4f} | 饱和率={sat*100:.1f}%")
        else:
            print(f"  {name:15s} | mean={act.mean():.4f} | std={act.std():.4f}")

# ─── 诊断 2: 梯度分布 ──────────────────────────────────────────
print(f"\n═══ 诊断工具 2: 梯度分布 ═══")

# 重新做一次 forward + backward 来获取梯度
logits = forward(Xb, training=False)
loss = F.cross_entropy(logits, Yb)
for p in parameters:
    p.grad = None
loss.backward()

for p, name in zip(parameters, param_names):
    if p.grad is not None and p.ndim >= 2:
        grad = p.grad
        print(f"  {name:20s} | grad_mean={grad.mean():.6f} | grad_std={grad.std():.6f} | "
              f"data_std={p.data.std():.6f}")

# ─── 诊断 3: 权重梯度/数据比率 ─────────────────────────────────
print(f"\n═══ 诊断工具 3: 权重梯度/数据比率 ═══")
print(f"  （梯度标准差 / 数据标准差，反映梯度相对于参数的尺度）")

for p, name in zip(parameters, param_names):
    if p.grad is not None and p.ndim >= 2:
        ratio = p.grad.std() / p.data.std()
        print(f"  {name:20s} | grad/data = {ratio.item():.6f}")

# ─── 诊断 4: 更新/数据比率（最重要的指标！）─────────────────────
print(f"\n═══ 诊断工具 4: 更新/数据比率（最重要！）═══")
print(f"  公式: update_std / data_std = lr × grad_std / data_std")
print(f"  理想值: ≈ 1e-3 (对数尺度约 -3)")
print(f"  过高(>-1): 学习率太大，训练不稳定")
print(f"  过低(<-4): 学习率太小，训练太慢")
print()

for p, name in zip(parameters, param_names):
    if p.ndim >= 2:
        ratios = ud_ratio_history.get(name, [])
        if ratios:
            final_ratio = ratios[-1]
            log_ratio = math.log10(final_ratio) if final_ratio > 0 else float('-inf')
            status = "✅" if -4 < log_ratio < -1 else "⚠️"
            print(f"  {status} {name:20s} | log10(ratio) = {log_ratio:.2f} | ratio = {final_ratio:.6f}")

# ─── 可视化诊断图表 ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图 1: tanh 激活值分布
ax = axes[0, 0]
tanh_names = [name for name in activation_data if 'tanh' in name]
for name in tanh_names:
    act = activation_data[name].flatten()
    ax.hist(act.numpy(), bins=50, density=True, alpha=0.5, label=name)
ax.set_title('诊断1: tanh 激活值分布', fontsize=12)
ax.set_xlabel('激活值')
ax.legend(fontsize=8)

# 图 2: 梯度分布
ax = axes[0, 1]
for p, name in zip(parameters, param_names):
    if p.grad is not None and p.ndim >= 2 and 'linear' in name and 'W' in name:
        grad = p.grad.flatten()
        ax.hist(grad.numpy(), bins=50, density=True, alpha=0.5, label=name)
ax.set_title('诊断2: 权重梯度分布', fontsize=12)
ax.set_xlabel('梯度值')
ax.legend(fontsize=8)

# 图 3: 权重梯度/数据比率（柱状图）
ax = axes[1, 0]
names_for_bar = []
ratios_for_bar = []
for p, name in zip(parameters, param_names):
    if p.grad is not None and p.ndim >= 2 and 'W' in name:
        names_for_bar.append(name.replace('linear_', 'L'))
        ratios_for_bar.append((p.grad.std() / p.data.std()).log10().item())
ax.bar(names_for_bar, ratios_for_bar, color='steelblue', alpha=0.7)
ax.axhline(y=-3, color='green', linestyle='--', label='理想值 -3')
ax.set_title('诊断3: log10(梯度/数据) 比率', fontsize=12)
ax.set_ylabel('log10(ratio)')
ax.legend()

# 图 4: 更新/数据比率随训练变化（最重要的图！）
ax = axes[1, 1]
for name in sorted(ud_ratio_history.keys()):
    if 'W' in name and ud_ratio_history[name]:
        ratios = ud_ratio_history[name]
        log_ratios = [math.log10(r) if r > 0 else -5 for r in ratios]
        ax.plot(log_ratios, label=name.replace('linear_', 'L'), alpha=0.7)
ax.axhline(y=-3, color='green', linestyle='--', linewidth=2, label='理想值 -3')
ax.set_title('诊断4: 更新/数据比率（最重要！）', fontsize=12)
ax.set_xlabel('训练步数 (×10)')
ax.set_ylabel('log10(update/data)')
ax.legend(fontsize=8)

plt.suptitle('训练诊断工具 — 4 种关键指标', fontsize=14, y=1.02)
plt.tight_layout()

save_path = os.path.join(script_dir, 'diagnostic_tools.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存到: {save_path}")

# ─── 总结 ───────────────────────────────────────────────────────
print(f"""
═══ 总结 ═══

4 种诊断工具的用途：

  1. 激活值分布 → 发现 tanh/relu 饱和
     - tanh: |h| > 0.99 的比例应很低
     - relu: 负值比例不应过高

  2. 梯度分布 → 发现梯度消失/爆炸
     - 各层梯度标准差应相似
     - 不应有数量级差异

  3. 权重梯度/数据比率 → 检查各层训练是否均衡
     - ratio = grad_std / data_std
     - 各层应大致相同

  4. 更新/数据比率 → 检查学习率是否合适（最重要！）
     - ratio = (lr × grad_std) / data_std
     - log10(ratio) ≈ -3 是理想值
     - 这是 Karpathy 最推荐的诊断指标

     过高 → 降低学习率
     过低 → 提高学习率或检查是否有瓶颈层
""")
