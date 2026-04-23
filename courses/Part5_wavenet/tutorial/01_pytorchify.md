# 01 — PyTorch 化：让代码更优雅

> 📦 从字典管理层到模块化层对象，让代码的形状匹配思想的形状。

## 从 Part 3 的问题出发

Part 3 的深层网络用**字典**管理层：

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

**更好的方式**：每个层是一个对象，有统一的 `__call__` 接口。

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

封装了 `view(0, -1)` 操作。

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

自带 **Kaiming 初始化**。

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

> 💡 PyTorch 的 `nn.Module` 用 `model.train()` / `model.eval()` 做这件事。我们手动管理是为了理解底层机制。

## 代码参考

👉 [01_pytorchify_layers.py](../scripts/01_pytorchify_layers.py) — 完整的 PyTorch 化重构

## 下一步

代码模块化了，接下来要做的是改变网络结构——从"展平所有上下文"到"层次化融合"。

👉 [02 — WaveNet 架构](02_wavenet_architecture.md)
