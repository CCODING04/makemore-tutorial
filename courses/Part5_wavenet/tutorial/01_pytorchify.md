# 01 — PyTorch 化：让代码更优雅

> 📦 从字典管理层到模块化层对象，让代码的形状匹配思想的形状。

## 从 Part 3 的问题出发

Part 3 教程里的层已经写成类；但它配套的脚本（`Part3_batchnorm/scripts/05_deep_network.py`）用的是**字典**手动管理层，另一种常见的散装写法长这样：

```python
layers = []
layers.append({'type': 'linear', 'W': W, 'b': b})
layers.append({'type': 'batchnorm', 'bn': bn})

# forward 时要判断类型
for layer in layers:
    if layer['type'] == 'linear':
        x = x @ layer['W'] + layer['b']
    elif layer['type'] == 'batchnorm':
        x = layer['bn'](x)
```

这行得通，但很丑。forward 逻辑被 if/elif 污染，每加一种层就要改 forward 函数。

**更好的方式**：每个层是一个对象，有统一的 `__call__` 接口。本章就是把上面这种写法重构为统一的层对象。

## PyTorch 化的层

### Embedding 层

```python
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]
```

封装了 `C[IX]` 操作。之前写 `emb = C[X]`，现在写 `emb = embedding(X)`。

### Flatten 层

```python
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []
```

封装了 `view(x.shape[0], -1)` 操作：保留 batch 维，把其余维度压平。

### Linear 层

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

自带 **Kaiming 初始化**：`randn / fan_in**0.5` 是 gain=1 的 He normal（标准差 1/√fan_in）。Part 3 的 Tanh 层前用 gain=5/3，是为了抵消 tanh 压缩方差；本 Part 的 Linear 后面紧跟 BatchNorm，它会把方差重新拉回 1，所以 gain=1 就够。（更准确地说：Kaiming 初始化是一族公式，gain 按后面的激活函数选。）

### Sequential 容器

```python
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

一行 `model.parameters()` 收集所有参数！

### BatchNorm1d 和 Tanh 从哪来？

上面的模型用了 `BatchNorm1d` 和 `Tanh`，它们在 Part 3 已实现。为了本章代码能独立拼出来，这里给出完整定义（2D 版，来自脚本 01；下一章会修复它对 3D 输入的 bug）：

```python
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


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

归一化做的就是 $\\hat{x} = (x - \\mu) / \\sqrt{\\sigma^2 + \\epsilon}$，再乘 $\gamma$ 加 $\beta$。`unbiased=False` 表示用有偏方差（除以 N 而不是 N-1），与训练时的归一化保持一致。

## 用 Sequential 重构网络

```python
model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
])

# 所有参数一行搞定
for p in model.parameters():
    p.requires_grad = True
```

脚本里还有一步小技巧（脚本 01 L187-188）：把最后一层 Linear 的权重整体缩小 10 倍，让初始 loss 接近均匀分布的 $\ln(27) \approx 3.30$，而不是随机大权重带来的高初始 loss：

```python
with torch.no_grad():
    model.layers[-1].weight *= 0.1
```

对比 Part 3 的写法，清爽多了。

## BatchNorm 的训练/推理坑

BatchNorm 有两种模式，通过 `self.training` 标志切换：

| 模式 | 统计量来源 | 更新 running stats？ |
|------|-----------|---------------------|
| Training (`training=True`) | 当前 mini-batch | ✅ 是 |
| Eval (`training=False`) | running statistics | ❌ 否 |

**忘记切换的后果**：
- 训练时用 eval 模式 → running stats 不更新 → 推理时统计量不准
- 推理时用 training 模式 → 输出取决于当前 batch → 不确定性

```python
# 训练
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = True

# 评估
for layer in model.layers:
    if hasattr(layer, 'training'):
        layer.training = False
```

> 💡 PyTorch 的 `nn.Module` 用 `model.train()` / `model.eval()` 做这件事。我们手动管理是为了理解底层机制。（等价的简写：`for layer in model.layers: layer.training = ...` 只需对有 `training` 属性的层生效，`hasattr` 判断就是干这个的。）

## 延伸：把噪声 loss 曲线变平滑

batch_size=32 时，每个 step 的 loss 噪声很大，肉眼难判断趋势。配套脚本 [02_fix_lr_plot.py](../scripts/02_fix_lr_plot.py) 训练 20000 步并记录每一步 loss，然后用 reshape 分窗取平均做平滑：

```python
losses = torch.tensor(losses)
smoothed = losses.view(-1, 1000).mean(1)   # 每 1000 步取平均 → 20 个点
```

`view(-1, 1000)` 把 20000 个点折成 (20, 1000)，`mean(1)` 沿第二维平均。平滑后能看清三段：快速下降 → 缓慢下降 → lr 衰减后平台。这也是检查"学习率该不该调"最便宜的诊断工具。

## 代码参考

- 👉 [01_pytorchify_layers.py](../scripts/01_pytorchify_layers.py) — 完整的 PyTorch 化重构（2000 步训练；`STEPS=500` 可短程验证）
- 👉 [02_fix_lr_plot.py](../scripts/02_fix_lr_plot.py) — loss 曲线平滑（20K 步实测 dev ≈2.17；`STEPS=2000` 可短程验证）

## 下一步

代码模块化了，接下来要做的是改变网络结构——从"展平所有上下文"到"层次化融合"。

👉 [02 — WaveNet 架构](02_wavenet_architecture.md)
