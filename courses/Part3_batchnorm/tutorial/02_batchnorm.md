# 02 — BatchNorm：深度学习的维生素

## 🤔 为什么需要 BatchNorm？

上一课我们学了一个"朴素"的方案：**用 Kaiming 初始化让每层的输出保持合理范围**。

但这个方案有个问题 —— 它只在**初始化时**管用。训练开始后，权重不断更新，各层的激活分布还是会逐渐跑偏。

💡 BatchNorm 的核心想法：**别只在初始化时修一次，每一步都自动矫正！**

就像维生素一样，每天吃一粒，保持身体的"均值和方差"在正常范围内 🍊

---

## 🧬 BN 原理

### 第一步：标准化（Normalize）

对每个 mini-batch 的激活值，减均值、除标准差：

```python
# hpreact 形状: (batch_size, n_hidden)
# 沿 batch 维度（dim=0）计算均值和方差

bnmean = hpreact.mean(0, keepdim=True)   # (1, n_hidden)
bnstd  = hpreact.std(0, keepdim=True)     # (1, n_hidden)

# 标准化
hpreact_norm = (hpreact - bnmean) / bnstd  # 均值 0，方差 1
```

标准化后，每个神经元的激活值大致在 `[-2, 2]` 范围内 —— tanh 不会饱和了。

### 第二步：可学习参数 γ 和 β

光标准化还不够 —— 如果强行把所有值压成均值 0、方差 1，网络的表达能力就受限了。

所以 BN 引入了**两个可学习参数**：

```python
# γ (gamma/bngain): 缩放参数
# β (beta/bnbias):  偏移参数

hpreact = bngain * hpreact_norm + bnbias
```

```
标准化后的分布：

    -2  -1   0   1   2     ← 均值 0，方差 1
     ║   ║   ║   ║   ║

经过 γ 缩放和 β 偏移后：

         -1   0   1   2   3   4   5   ← γ=2, β=3 时
          ║   ║   ║   ║   ║   ║   ║
```

🔑 网络可以**自己学**需要什么样的分布。初始时 γ=1、β=0（即不改变标准化结果），训练过程中慢慢调整。

### 第三步：Running Mean/Var（推理时用）

训练时我们用 mini-batch 的均值和方差。但**推理时可能只有一个样本**（batch_size=1），怎么算均值方差？

解决方案：训练时维护一个**指数移动平均**（EMA）的 running mean 和 running var：

```python
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running  = torch.ones((1, n_hidden))

# 训练循环中，每次更新 running stats
# 这里用 0.999/0.001 的系数（等价于 momentum=0.001），更新较慢但更稳定
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstd
```

> 💡 **momentum 参数说明**：PyTorch 的 `nn.BatchNorm1d` 默认 `momentum=0.1`（即 `0.9 * running + 0.1 * batch`），更新较快。上面的内联代码用 `0.001`（即 `0.999 * running + 0.001 * batch`），更新较慢但更平滑。两种都可以，区别在于 running stats 的收敛速度。下面的 `BatchNorm1d` 类使用 `momentum=0.1`（PyTorch 默认值）。

```
训练模式：用当前 batch 的 mean/std（精确）
评估模式：用 running mean/std（全局近似）

两者切换：
    model.train()   → 用 batch stats
    model.eval()    → 用 running stats
```

---

## 🔧 从零实现

> 📜 完整代码见 [`../scripts/04_batchnorm_implementation.py`](../scripts/04_batchnorm_implementation.py)

### BN 在网络中的位置

```
Embedding → [Linear → BatchNorm → Tanh] × N → Linear → 输出
                           ↑
                      每个隐藏层后面
                      都跟一个 BN
```

### 完整实现

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps              # 防止除零的小数
        self.momentum = momentum    # EMA 的更新速率
        self.training = True        # 训练 or 评估模式

        # 可学习参数（通过反向传播更新）
        self.gamma = torch.ones(dim)   # 缩放
        self.beta = torch.zeros(dim)   # 偏移

        # 缓冲区（通过 EMA 更新，不参与反向传播）
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        # 标准化
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # 训练时更新 running stats
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

### 在 MLP 中使用

```python
C  = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

# BN 参数
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

# 前向传播
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1

# ---- BatchNorm 层 ----
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi = hpreact.std(0, keepdim=True)
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
# ----------------------

h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)
```

### 推理时

```python
@torch.no_grad()
def split_loss(split):
    x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev)}[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1
    # 用 running stats 代替 batch stats
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
```

---

## ⚠️ BN 的坑

### 坑 1：忘记切 training/eval 模式

```python
# ❌ 训练完直接评估，忘了切模式
model.eval()  # ← 必须调这个！

# 或者手动设置
for layer in layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False
```

如果你不切换，推理时 BN 还在用 batch 的均值方差 —— batch_size=1 时结果会很离谱。

### 坑 2：batch_size=1 时方差为 NaN

```python
# batch_size=1 时，方差 = 0（只有一个样本！）
# std = sqrt(0) = 0 → 除以 0 → NaN 💀

# 解决方案：
# 1. 加 eps（我们设了 eps=1e-5）
# 2. 推理时用 running stats（而不是 batch stats）
```

### 坑 3：Running Mean 没收敛

```python
# momentum=0.1 时，大约需要 30-50 步让 running mean 稳定
# 如果训练步数太少（比如只跑了 100 步），running mean 可能不准

# 解决方案：
# 1. 跑够步数（我们跑了 200000 步，没问题）
# 2. 训练结束后，用整个训练集重新校准一次
with torch.no_grad():
    emb = C[Xtr]
    hpreact = emb.view(emb.shape[0], -1) @ W1
    bnmean = hpreact.mean(0, keepdim=True)  # 精确的全局均值
    bnstd = hpreact.std(0, keepdim=True)    # 精确的全局标准差
```

---

## 🧪 课后练习

### 练习 1：去掉 BN 会怎样？

> 把网络中的 BatchNorm 层去掉，只保留 Linear + Tanh。观察训练 loss 曲线有什么变化。然后加回 BN，对比 loss 收敛速度。

<details>
<summary>💡 提示</summary>

去掉 BN 后，深层网络的 loss 曲线通常会：
- 初始 loss 更高
- 训练过程更不稳定（抖动大）
- 最终 loss 也可能更高

</details>

### 练习 2：自己实现 EMA

> 不用 PyTorch 的 `running_mean`，自己维护一个 Python 列表记录每一步的 batch mean。训练结束后，画出 batch mean 随训练步数的变化曲线。观察大约多少步后 running mean 趋于稳定。

<details>
<summary>💡 提示</summary>

```python
mean_history = []
for i in range(max_steps):
    # ... forward pass ...
    mean_history.append(bnmeani.mean().item())

# 画出 mean_history
plt.plot(mean_history)
plt.xlabel('step')
plt.ylabel('batch mean')
```

</details>

### 练习 3：Gamma 和 Beta 的作用

> 初始化时让 gamma=0.5（而不是 1.0），观察 tanh 饱和度有什么变化。再让 beta=1.0（而不是 0），观察 tanh 输出的分布有什么偏移。

---

## 🧭 下一步

BN 学完了，现在我们有了一个稳定的"积木"。接下来用这些积木搭建更深的网络，并学习一套完整的诊断工具箱！

👉 [03 — 深层网络与诊断工具](03_deep_network.md)
