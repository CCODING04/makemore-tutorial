# 测验 · Part 4（手动反向传播）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** PyTorch autograd 的本质是什么？反向传播每一步的通用公式是什么？矩阵乘法 `C = A @ B` 的梯度怎么传？

- **答案**：autograd = 计算图 + 链式法则：前向按拓扑序保存中间变量，反向按逆拓扑序逐步回传。每步满足"梯度 = 上游梯度 × 局部梯度"。矩阵乘法反传为 `dL/dA = dL/dC @ B.T`、`dL/dB = A.T @ dL/dC`。注意一个变量被多处使用时梯度要**累加**（如 `dcounts` 在 Step 3 与 Step 5 两处都要加），漏加是手推最典型的错误。
- **锚点**：教程 01_why_backprop.md §链式法则回顾；02_forward_and_backward.md Step 3 / Step 5（累加警告）

**Q2（对比）** CrossEntropy 的反向传播：12 步展开里的 Step 1-8，如何被压缩成 3 行代码？为什么成立？

- **答案**：`dlogits = F.softmax(logits, 1); dlogits[range(n), Yb] -= 1; dlogits /= n`。softmax 与 NLL 合并求导后有解析解：正确类位置的梯度是 $p - 1$、其余位置是 $p$，除以 n 对应 loss 的 mean。直觉：正确类"信心还不够，加把劲"，错误类"太有信心，压下去"。课程热力图显示每个样本只有正确类别位置被减 1。
- **锚点**：教程 03_simplified_and_training.md §CrossEntropy 简化反传；02_forward_and_backward.md Step 1-8

**Q3（诊断）** BatchNorm 简化反传公式里的 `n/(n-1)` 因子什么时候该出现？混用口径会发生什么？

- **答案**：系数必须与前向方差口径**配套**。本课前向用 1/n 有偏方差（`bnvar = bndiff2.mean(0)`），对它求导的简化式第三项系数是 1，公式中**没有** `n/(n-1)`；只有前向改用 1/(n-1) 无偏方差（Bessel 校正）时才多出该因子。两种口径各自自洽、混用即错：1/n 前向配 `n/(n-1)` 在 n=32 时引入约 4.6e-05 的系统性偏差，恰好卡在 1e-5 验收阈值外被判红。
- **锚点**：assignment_4/README.md 题 4 与思考题 Q4；教程 03_simplified_and_training.md §BatchNorm 简化反传（注意：教程正文公式带 `n/(n-1)`，与作业的修正口径冲突，以作业为准）

**Q4（概念）** 不用 `loss.backward()` 训练完整网络，手动梯度清单有哪些步？Embedding 的梯度为什么用 `+=`？

- **答案**：六步：CE 反传（3 行）→ Linear2（`dh = dlogits @ W2.T`、`dW2 = h.T @ dlogits`、`db2 = dlogits.sum(0)`）→ Tanh（`dhpreact = dh * (1 - h**2)`）→ BatchNorm 一行公式 → Linear1（`dembcat = dhprebn @ W1.T`、`dW1 = embcat.T @ dhprebn`）→ Embedding scatter：`dC[Xb[i,j]] += demb[i,j]`。用 `+=` 是因为 batch 内多个样本可能查到同一行 embedding，它们的梯度必须累加。
- **锚点**：教程 03_simplified_and_training.md §完整手动训练；02_forward_and_backward.md Extra: dhprebn → dC

**Q5（数字）** 手写梯度用 `cmp()` 怎么验收？误差在什么量级算正常，什么量级说明公式错了？

- **答案**：`cmp` 用 autograd 当标准答案：`torch.allclose(dt, t, atol=1e-5)` 加最大绝对差。正常对拍误差在 1e-9 ~ 5e-5 之间，随算子而变：逐元素算子（tanh 反传）可到 1e-9 量级，涉及求和、除法、大矩阵乘的（BN/CE 反传）会大一些——float32 只有约 7 位有效数字，运算链越长舍入误差累积越多。若误差停在约 5e-5（如 BN 误用 `n/(n-1)` 时实测约 4.8e-05），是系统性公式错误而非浮点噪声。
- **锚点**：教程 02_forward_and_backward.md §梯度验证；assignment_4/README.md 思考题 Q2

## 覆盖映射（学习目标 → 题号）

学习目标来源：教程 README.md「学习目标」。

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 说出 PyTorch autograd 的基本原理（计算图 + 链式法则） | Q1 | 上游梯度 × 局部梯度、矩阵乘法反传 |
| 手动推导 CrossEntropy Loss 的梯度（3 行简化版） | Q2 | 8 步展开 vs 3 行解析解 |
| 手动推导 BatchNorm 的梯度（1 行简化版） | Q3 | ⚠️ 正文与作业口径冲突，见下注 |
| 用手动梯度训练完整网络（不用 `loss.backward()`） | Q4 | 六步清单与 scatter 累加 |
| 用 `cmp()` 函数验证手写梯度的正确性 | Q5 | 验收阈值与误差量级 |

> ⚠️ 正文覆盖不足（口径冲突）：教程 03_simplified_and_training.md「BatchNorm 简化反传」的公式包含 `n/(n-1)` 并解释为 Bessel 校正，但本课前向方差是 1/n 有偏口径，按作业 4 题 4 的推导与实测（混用引入约 4.6e-05 系统偏差、对拍失败），该因子此时**不应出现**。教程正文需按作业口径修正后方可作为唯一命题依据；本卷 Q3 以作业口径为准。

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| autograd 的本质 | 计算图 + 链式法则：前向缓存中间量，反向逆拓扑序累乘局部梯度 |
| 矩阵乘法的反向传播 | `dL/dA = dL/dC @ B.T`，`dL/dB = A.T @ dL/dC` |
| CE 简化反传 3 行 | `softmax(logits, 1)` → 正确类位置减 1 → 除以 n |
| softmax 前减 max 的作用 | 利用平移不变性防 exp 上溢；它是计算图真实节点，反传会分走梯度（`logit_maxes` 只在最大值位置有梯度） |
| tanh 的反传 | `dhpreact = dh * (1 - h**2)` |
| BN 简化反传（1/n 有偏前向） | `dhprebn = bngain * bnvar_inv / n * (n*dhpreact - dhpreact.sum(0) - bnraw*(dhpreact*bnraw).sum(0))`；此口径下没有 `n/(n-1)` |
| Embedding 的反传 | `dC[Xb] += demb`（scatter），同一索引被查多次必须累加 |
| cmp() 的验收标准 | `allclose(atol=1e-5)`；正常误差 1e-9 ~ 5e-5；约 5e-5 说明公式系统性错误 |
| 手推梯度的四大理由 | 理解 autograd 原理、定位梯度异常（NaN/爆炸/不收敛）、支持自定义算子、应对面试 |
| 为什么要手写反传而不是只用 autograd | 手推把黑魔法变成可调试的工具；工程日常仍用 autograd，手推是理解与排障手段 |
