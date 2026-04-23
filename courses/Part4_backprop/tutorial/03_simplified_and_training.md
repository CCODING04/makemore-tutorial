# 03 简化公式与手动训练 🚀

> 12 步太多？一行代码就能搞定 CrossEntropy 和 BatchNorm 的反传！

## CrossEntropy 简化反传

前面我们用 8 步（Step 1-8）才从 loss 算到 dlogits。但其实整个 CrossEntropy 的反向传播可以简化成 **3 行代码** 🪄

```python
dlogits = F.softmax(logits, 1)       # 先算 softmax
dlogits[range(n), Yb] -= 1           # 正确类别位置减 1
dlogits /= n                          # 除以 batch size
```

### 为什么？

CrossEntropy Loss 对 logits 的梯度有一个优雅的解析解：

```
∂L/∂logits_i = softmax(logits)_i - 𝟙(i == correct_class)
```

再除以 N（因为 loss 取了 mean）。

### 直觉理解

- softmax 输出的是概率分布 → 模型给每个类别的"信任度"
- 正确类别位置减 1 → "你对正确答案的信心还不够，要再加把劲"
- 其他位置就是概率值本身 → "你对错误答案太有信心了，要压下去"
- 整体效果：**正确类别概率 ↑，其他类别概率 ↓**

> 运行 [`04_cross_entropy_backward.py`](../scripts/04_cross_entropy_backward.py) 查看验证 + 热力图可视化

## BatchNorm 简化反传

BatchNorm 的逐步反传（Step 12）有 5 个子步骤，但也可以压缩成 **一行公式** 🎯

```python
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact
    - dhpreact.sum(0)
    - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
)
```

### 公式拆解

三个项分别代表：

| 项 | 含义 |
|----|------|
| `n * dhpreact` | 直接传播（因为 x̂ 包含了 x） |
| `- Σ(dhpreact)` | 减均值 μ 的修正（μ 依赖于所有样本） |
| `- n/(n-1) · x̂ · Σ(dhpreact·x̂)` | 除标准差 σ 的修正 |

其中 `n/(n-1)` 是 **Bessel 校正**（用 batch 方差的无偏估计）。

> 运行 [`05_batchnorm_backward.py`](../scripts/05_batchnorm_backward.py) 查看验证

## 完整手动训练

把前面的所有简化公式串起来，就能用 **手动梯度** 训练整个网络！

### 手动反向传播清单

```
1️⃣  CrossEntropy 反传（3 行）
    dlogits = softmax(logits)
    dlogits[正确位置] -= 1
    dlogits /= n

2️⃣  Linear 2 反传
    dh   = dlogits @ W2.T
    dW2  = h.T @ dlogits
    db2  = dlogits.sum(0)

3️⃣  Tanh 反传
    dhpreact = dh * (1 - h²)

4️⃣  BatchNorm 反传（1 行）
    dhprebn = ... (公式见上)

5️⃣  Linear 1 反传
    dembcat = dhprebn @ W1.T
    dW1     = embcat.T @ dhprebn

6️⃣  Embedding 反传
    dC[Xb] += demb  (scatter 操作)
```

### 训练循环

```python
for step in range(max_steps):
    # 前向传播（展开所有中间变量）
    emb = C[Xb]
    hprebn = embcat @ W1
    # ... BatchNorm ...
    h = tanh(hpreact)
    logits = h @ W2 + b2
    loss = cross_entropy(logits, Yb)

    # 手动反向传播（不用 loss.backward()！）
    dlogits = softmax(logits); dlogits[range(n), Yb] -= 1; dlogits /= n
    # ... 其余梯度 ...

    # 参数更新
    C.data -= lr * dC
    W1.data -= lr * dW1
    # ...
```

> 运行 [`06_manual_training.py`](../scripts/06_manual_training.py) 查看完整训练过程 + 采样生成

## 📝 课后作业

学完本 Part 后，完成 **[Assignment 4](../../../assignments/assignment_4/)**：

| 题目 | 内容 | 难度 |
|------|------|------|
| 1 | forward_pass | 实现逐步前向传播函数 | ⭐⭐ |
| 2 | backward_step | 实现单步反向传播 | ⭐⭐⭐ |
| 3 | cross_entropy_backward | 实现简化 CE 反传 | ⭐⭐ |
| 4 | batchnorm_backward | 实现简化 BN 反传 | ⭐⭐⭐ |
| 5 | manual_train（拓展） | 手动梯度训练 | ⭐⭐⭐⭐ |

## 🎯 本 Part 总结

| 你学到了什么 | 关键公式 |
|-------------|---------|
| 前向传播展开 | 每步保存中间变量 |
| 12 步反传 | 链式法则 × 12 |
| CE 简化反传 | softmax → 减1 → 除N |
| BN 简化反传 | 一行公式（含 Bessel 校正）|
| 手动训练 | 不用 loss.backward() 也能训练 |

## 🔮 下一课预告：Part 5 WaveNet

Part 4 的网络用固定的 3 个上下文字符预测下一个字符。但如果上下文更长呢？

Part 5 将引入 **WaveNet** 架构：
- 用层次化的方式处理越来越大的上下文
- 从 "3个字符 → 1个预测" 升级到 "8+个字符 → 1个预测"
- 引入 **dilated causal convolution** 的思想

```
Part 4: [a][b][c] → Linear → ... → 下一个字符
Part 5: [a][b][c][d][e][f][g][h] → WaveNet → ... → 下一个字符
                                  ┌─┐
                              ┌───┤ ├─┐
                          ┌───┤   └─┤ ├───┐
                      ┌───┤   │    │ │   ├───┐
                    [a] [b] [c] [d] [e] [f] [g] [h]
```

---

**做完作业了吗？** → [Assignment 4](../../../assignments/assignment_4/)
