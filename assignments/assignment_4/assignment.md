# Assignment 4：手动反向传播 ✏️

> 把 autograd 扔掉，自己算梯度！

## 📋 题目列表

### Q1：forward_pass ⭐⭐

实现一个函数，完成逐步前向传播，保存所有中间变量。

```python
def forward_pass(C, W1, b1, bngain, bnbias, W2, b2, Xb):
    """
    返回一个字典，包含所有中间变量和 loss。
    """
    # TODO: 实现逐步前向传播
    pass
```

**要求：**
- 返回字典包含：`emb, embcat, hprebn, bnmeani, bndiff, bndiff2, bnvar, bnvar_inv, bnraw, hpreact, h, logits, loss`
- `loss` 必须和 `F.cross_entropy(logits, Yb)` 一致（误差 < 1e-5）

### Q2：backward_step ⭐⭐⭐

实现单步反向传播，给定某一步的输出和上游梯度，计算该步的梯度。

```python
def backward_step(step_name, upstream_grad, cache):
    """
    step_name: 'tanh', 'linear', 'bn_scale', 'elementwise_mul' 等
    upstream_grad: 上游传来的梯度
    cache: 前向传播时保存的中间变量（dict）
    """
    # TODO: 根据不同的 step_name 计算对应梯度
    pass
```

**要求：**
- 至少支持：`tanh`, `linear`, `bn_scale`, `softmax_ce` 四种操作
- 梯度和 autograd 对比误差 < 1e-5

### Q3：cross_entropy_backward ⭐⭐

实现简化版 CrossEntropy 反向传播。

```python
def cross_entropy_backward(logits, Yb):
    """
    用简化公式计算 dlogits。
    返回和 F.cross_entropy 反向传播一致的梯度。
    """
    # TODO: 3 行代码
    pass
```

**要求：**
- 不用展开 softmax（直接用 `F.softmax`）
- 梯度误差 < 1e-5

### Q4：batchnorm_backward ⭐⭐⭐

实现简化版 BatchNorm 反向传播。

```python
def batchnorm_backward(dhpreact, bnraw, bngain, bnvar_inv, eps=1e-5):
    """
    用简化公式计算 dhprebn。
    """
    # TODO: 一行公式
    pass
```

**要求：**
- 使用简化公式（不是逐步展开）
- 梯度误差 < 1e-5
- 理解公式中 `n/(n-1)` 的含义

### Q5：manual_train（拓展）⭐⭐⭐⭐

用手动梯度训练完整网络，不用 `loss.backward()`。

```python
def manual_train(Xtr, Ytr, Xdev, Ydev, n_embd=10, n_hidden=200,
                 max_steps=10000, batch_size=32, lr=0.1):
    """
    手动梯度训练循环。
    返回训练好的参数字典和 loss 历史。
    """
    # TODO: 实现完整训练循环
    pass
```

**要求：**
- 不使用 `loss.backward()`
- 打印训练过程中 loss 的变化
- 最终 train loss < 2.5, dev loss < 2.8

## 🚀 开始

1. 打开 [`backprop_exercises.py`](backprop_exercises.py) — 所有 TODO 都在这里
2. 运行 [`test_backprop_exercises.py`](test_backprop_exercises.py) — 检查答案
3. 每过一题就测一下，不要攒到最后

## 💡 提示

- 先跑通 Q1，再开始 Q2
- Q3 和 Q4 是独立的，可以先做你觉得简单的
- Q5 需要把 Q1-Q4 的知识串起来
- 不确定对不对就用 `cmp()` 函数对比 autograd

## 📚 参考资料

- [Part 4 教程](../../courses/Part4_backprop/tutorial/)
- [02_backprop_step_by_step.py](../../courses/Part4_backprop/scripts/02_backprop_step_by_step.py) — 12 步推导参考
- [04_cross_entropy_backward.py](../../courses/Part4_backprop/scripts/04_cross_entropy_backward.py) — CE 简化参考
- [05_batchnorm_backward.py](../../courses/Part4_backprop/scripts/05_batchnorm_backward.py) — BN 简化参考
