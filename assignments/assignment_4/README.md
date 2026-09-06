# Assignment 4：手动反向传播 ✏️

> 把 autograd 扔掉，自己算梯度！

## 📋 题目列表

### Q1：forward_pass ⭐⭐

实现一个函数，完成逐步前向传播，保存所有中间变量。

```python
def forward_pass(params, Xb, Yb=None):
    """
    params: 参数字典 {'C', 'W1', 'b1', 'bngain', 'bnbias', 'W2', 'b2'}
    Xb: 输入 batch (B, 3)
    Yb: 标签 batch (B,)（计算 cache['loss'] 必需）

    返回一个字典（cache），包含所有中间变量和 loss。
    """
    # TODO: 实现逐步前向传播
    pass
```

**要求：**
- 参数从 `params` 字典中按 key 取用；`Yb` 只在计算 `cache['loss']` 时需要
- 返回字典包含：`emb, embcat, hprebn, bnmeani, bndiff, bndiff2, bnvar, bnvar_inv, bnraw, hpreact, h, logits, loss`
- `loss` 必须和 `F.cross_entropy(logits, Yb)` 一致（误差 < 1e-4）

### Q2：backward_step ⭐⭐⭐

实现单步反向传播：按算子拆成 4 个独立函数，给定该算子的输出和上游梯度，计算传向输入的梯度。

```python
def backward_tanh(dh, h):
    """dh: 上游梯度 (B, H)；h: tanh 的输出 (B, H)。返回 dhpreact。"""
    # TODO: dhpreact = ...
    pass

def backward_linear(dout, input_tensor, weight):
    """out = input @ weight + bias 的反传。
    返回 (dinput, dweight)。"""
    # TODO: dinput = ...; dweight = ...
    pass

def backward_bn_scale(dhpreact, bnraw, bngain):
    """hpreact = bngain * bnraw + bnbias 的反传。
    返回 (dbngain, dbnbias, dbnraw)。"""
    # TODO: dbngain = ...; dbnbias = ...; dbnraw = ...
    pass

def backward_softmax_ce(logits, Yb):
    """softmax + NLL 合并的反传。返回 dlogits (B, V)。
    （骨架中已给出参考实现，可作为其余三个函数的写法参照）"""
```

**要求：**
- 共 4 个函数：`backward_tanh`, `backward_linear`, `backward_bn_scale`, `backward_softmax_ce`
- 每个梯度和 autograd 对比误差 < 1e-5

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
def batchnorm_backward(dhpreact, bnraw, bngain, bnvar_inv):
    """
    用简化公式计算 dhprebn。
    """
    # TODO: 一行公式
    pass
```

**要求：**
- 不接收 `eps` 参数：`eps` 已在前向计算 `bnvar_inv = (bnvar + eps) ** -0.5` 时用掉，反传只需 `bnvar_inv`
- 使用简化公式（不是逐步展开）
- 梯度误差 < 1e-5
- **理解公式系数与前向方差口径的配套关系**：
  - 本课前向用 **1/n 有偏方差**：`bnvar = bndiff2.mean(0)` → 简化公式为
    `dhprebn = bngain * bnvar_inv / n * (n * dhpreact - dhpreact.sum(0) - bnraw * (dhpreact * bnraw).sum(0))`，
    第三项系数是 **1，公式中没有 `n/(n-1)`**
  - 只有前向改用 **1/(n-1) 无偏方差**（`bndiff2.sum(0) / (n-1)`）时，公式才会多出 `n/(n-1)` 因子
  - 两种口径各自自洽、**混用即错**：1/n 前向配 `n/(n-1)` 系数会引入 ~4.6e-05（n=32）的系统性偏差，正好卡在 1e-5 阈值外被测试判红

### Q5：manual_train（拓展）⭐⭐⭐⭐

用手动梯度训练完整网络，不用 `loss.backward()`。

```python
def manual_train(n_embd=10, n_hidden=200, max_steps=10000,
                 batch_size=32, lr=0.1, seed=42):
    """
    手动梯度训练循环（训练数据 Xtr/Ytr、验证数据 Xdev/Ydev
    已在文件顶部按 80/10/10 划分好，无需传参）。

    返回:
        result: 字典，包含:
            params: 训练好的参数字典
            lossi: loss 历史
            bnmean_running: running mean
            bnvar_running: running var
    """
    # TODO: 实现完整训练循环
    pass
```

**要求：**
- 不使用 `loss.backward()`，全部用手动梯度更新参数
- 打印训练过程中 loss 的变化
- **测试口径**：测试用 `n_hidden=64, max_steps=1000` 的小预算，只断言 loss 在下降（前 100 步均值 → 后 100 步均值）
- **完整训练目标**：`n_hidden=200, max_steps=10000` 时最终 train loss < 2.5、dev loss < 2.8（自我检验目标）

## 🚀 开始

1. 打开 [`backprop_exercises.py`](backprop_exercises.py) — 所有 TODO 都在这里
2. 运行 [`test_backprop_exercises.py`](test_backprop_exercises.py) — 检查答案：
   ```bash
   # 方式一：直接运行测试脚本（末尾给得分）
   python test_backprop_exercises.py

   # 方式二：用 pytest 逐题运行（逻辑失败会变红）
   pip install pytest
   pytest test_backprop_exercises.py -v
   ```
3. 每过一题就测一下，不要攒到最后

## 💡 提示

- 先跑通 Q1，再开始 Q2
- Q3 和 Q4 是独立的，可以先做你觉得简单的
- Q5 需要把 Q1-Q4 的知识串起来

## 🤔 思考题

完成作业后，思考以下问题加深理解：

**Q1：** 如果省略 `bngain` 的梯度（设为 0），训练会怎样？
<details>
<summary>💡 提示</summary>

BatchNorm 的 `gamma`（bngain）控制激活值的缩放。如果梯度为 0，gamma 不会更新，激活值的缩放比例固定。网络仍然能训练，但表达能力受限——相当于 BatchNorm 退化为只做标准化，不做缩放。

</details>

**Q2：** 手动梯度和 autograd 梯度的误差大约在什么量级？
<details>
<summary>💡 提示</summary>

误差在 **1e-9 ~ 5e-5** 之间，随算子和维度而变：简单的逐元素算子（如 tanh 反传）对拍可达 1e-9 量级，而涉及求和、除法、大矩阵乘的算子（如 BN 反传、CE 反传）误差会大一些。`float32` 只有约 7 位有效数字，运算链越长累积的舍入误差越大。这是正常的，只要误差 < 1e-5 就算正确；如果误差 ≈ 5e-5（如 BN 反传误用 `n/(n-1)` 系数时的 ≈4.8e-05），说明公式口径有系统性错误，不是浮点噪声。

</details>

**Q3：** 为什么 Karpathy 要我们手写反向传播，而不是直接用 `loss.backward()`？
<details>
<summary>💡 提示</summary>

1. **理解原理**：知道 autograd 在做什么，不再是黑魔法
2. **调试能力**：当梯度出问题时，能定位到具体哪一步
3. **自定义操作**：有些操作 PyTorch 没有内置反向传播，需要自己写
4. **面试**：手推 BatchNorm 梯度是常见面试题

</details>

**Q4：** BatchNorm 反向传播的简化公式里，`n/(n-1)` 因子什么时候出现，什么时候没有？
<details>
<summary>💡 提示</summary>

**系数必须与前向方差口径配套**，这是同一个方差两种写法的求导结果：

- 本课前向用的是 **1/n 有偏方差**：`bnvar = bndiff2.mean(0)`（PyTorch `var(unbiased=False)` 口径）。对它求导得到的简化式第三项系数是 **1**，公式里**没有** `n/(n-1)`：
  `dhprebn = bngain * bnvar_inv / n * (n*dhpreact - dhpreact.sum(0) - bnraw*(dhpreact*bnraw).sum(0))`，与 autograd 对拍 ≈ 1e-9。
- 只有前向改用 **1/(n-1) 无偏方差**（`bndiff2.sum(0) / (n-1)`，即 Bessel 校正口径）时，公式才多出 `n/(n-1)` 因子。

两种口径各自自洽、**混用即错**：1/n 前向配 `n/(n-1)` 系数会在 n=32 时引入 ~4.6e-05 的系统性偏差（实测 ≈4.8e-05），恰好超过 1e-5 阈值被测试抓住。所以答案不是"该用 n 还是 n-1"，而是"反传系数要和你前向的方差定义一致"。

</details>

- 不确定对不对就用 `cmp()` 函数对比 autograd

## 📚 参考资料

- [Part 4 教程](../../courses/Part4_backprop/tutorial/)
- [02_backprop_step_by_step.py](../../courses/Part4_backprop/scripts/02_backprop_step_by_step.py) — 12 步推导参考
- [04_cross_entropy_backward.py](../../courses/Part4_backprop/scripts/04_cross_entropy_backward.py) — CE 简化参考
- [05_batchnorm_backward.py](../../courses/Part4_backprop/scripts/05_batchnorm_backward.py) — BN 简化参考

---

## 🎯 面试直通车（话术卡：结论 → 原理 → 边界）

> 每张卡按"总分总"组织：先一句话结论压场，再两三句原理支撑，最后一句边界/代价收尾——面试答题的固定骨架。

**Q1："说说反向传播的本质，autograd 到底帮你做了什么？"**

- **结论**：autograd 就是前向按拓扑序缓存中间结果、反向按逆拓扑序用链式法则累乘局部梯度。
- **原理**：每一步满足"梯度 = 上游梯度 × 局部梯度"；矩阵乘法反传是 $dL/dA = dL/dC \cdot B^T$、$dL/dB = A^T \cdot dL/dC$。课程把 MLP+BN 的前向拆成 12 步逐一手推，并用 `cmp()` 与 autograd 对齐，所有梯度误差小于 $10^{-5}$。
- **边界**：链式法则只覆盖可微算子；一个变量被多处使用时梯度要累加，漏加是手推最典型的错误。

**Q2："写一下 cross_entropy 的反向传播。"**

- **结论**：存在一行魔法公式 $\mathrm{dlogits} = (\mathrm{softmax}(logits) - onehot(Y))/n$，把 8 步展开压缩成 3 行代码。
- **原理**：softmax 与交叉熵合并求导后，正确类位置的梯度是 $p - 1$、其余位置是 $p$，再除以 batch 大小 $n$ 对应 mean；课程热力图显示每个样本只有正确类别位置被减 1，其余位置梯度就是 softmax 概率本身。
- **边界**：推导默认 logits 已做减 max 的数值稳定处理，自己从零实现时这一步要一起写。

**Q3："BatchNorm 的反向传播怎么算？公式里的 n/(n-1) 是怎么来的？"**

- **结论**：BN 反传有一行简化公式，但系数必须与前向方差口径配套——本课前向用有偏方差 $1/n$，公式中**没有** $n/(n-1)$ 项。
- **原理**：本课前向 `bnvar = bndiff2.mean(0)`（$1/n$ 有偏），对它求导的简化式为 `dhprebn = bngain * bnvar_inv / n * (n*dbnraw - dbnraw.sum(0) - (bnraw*dbnraw).sum(0))`（修复版 `05_batchnorm_backward.py`，对拍 autograd ≈ 9.3e-10）；若前向改用无偏 $1/(n-1)$，才需要多出 $n/(n-1)$ 因子——两种口径各自自洽、混用即错（本课台账 P0 实证）。
- **边界**：该简化式只在 training 模式（batch 统计）下成立；eval 模式下 BN 是用固定统计量的纯仿射变换，梯度形式完全不同。

**Q4："实际工作都直接用 autograd，为什么还要手推梯度？"**

- **结论**：因为手推把 autograd 从黑魔法变成可调试的工具，而且 BN/CE 反传本身就是高频面试题。
- **原理**：课程给出四大理由——理解原理、定位梯度异常（NaN/爆炸/不收敛时能定位到具体一步）、支撑 PyTorch 未内置的自定义算子、应对面试；验收方式是把每个手写梯度与 autograd 逐一对比，误差小于 $10^{-5}$ 即正确。
- **边界**：工程日常仍应使用 autograd，手推是理解与排障手段，不是替代品。

**Q5："softmax 前为什么要减 max？这对反向传播有影响吗？"**

- **结论**：减 max 利用 softmax 的平移不变性防止 exp 上溢，但它在计算图里是真实节点，反传时会分走梯度。
- **原理**：$e^{z-c}/\sum_j e^{z_j-c}$ 与原式结果完全相同；课程把每行最大值单独存成 `logit_maxes`，autograd 显示它的梯度只落在每行最大值所在的位置。
- **边界**：这是数值技巧而非数学必需；手推时若忽略 logits → logit_maxes 这条分支，得到的梯度会与 autograd 对不上。
