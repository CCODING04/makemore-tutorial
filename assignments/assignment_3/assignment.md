# 作业 3：训练诊断与 BatchNorm

> **对应教程**：Part 3 — Building makemore Part 3: Activations & Gradients, BatchNorm
>
> **前置**：完成作业 2（MLP）

---

## 📋 概述

本作业聚焦于训练过程的诊断与优化。你将亲手体验「初始 loss 为什么不对」「tanh 饱和意味着什么」「BatchNorm 是如何解决这些问题的」，从诊断问题到解决问题，完整走一遍 Andrej 在视频中演示的核心实验。

完成本作业后，你应该能够：

- 用初始 loss 诊断网络初始化是否合理
- 理解 tanh 饱和现象及其对梯度的影响
- 从零实现 BatchNorm1d，理解 training/eval 两种模式
- 掌握 Kaiming 初始化的动机和使用方式

---

## 🔧 环境准备

### 依赖

```bash
pip install torch
```

### 数据

数据文件位于 `../../data/names.txt`，每行一个英文名字。如果数据文件不存在，请先从项目根目录下载：

```bash
# 在项目根目录下
wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt -O data/names.txt
```

### 文件结构

```
assignments/assignment_3/
├── assignment.md               # 本文件
├── batchnorm_exercises.py      # 👈 你需要编辑的文件
└── test_batchnorm_exercises.py # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_3
python test_batchnorm_exercises.py
```

---

## 📝 题目列表

### 题 1：初始 Loss 诊断（基础）

**函数**：`diagnose_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647)`

**要求**：
- 构建一个标准 MLP（Embedding → 隐藏层 tanh → 输出层），用默认随机初始化
- 对全部训练数据做一次 forward，返回交叉熵 loss
- 固定随机种子以保证可重复

**关键点**：
- 初始化时所有权重用 `torch.randn`（标准正态），不做任何缩放
- 输出层的 `b2` 初始化为 `torch.randn(27)`（不为零）
- forward 后用 `F.cross_entropy` 计算 loss

**预期结果**：
- loss ≈ 3.7 ~ 4.0，远大于理论初始值 ln(27) ≈ 3.298
- 这说明网络「过于自信」，初始参数分布不合理

**思考**：
- 为什么随机初始化的网络 loss 会比 ln(27) 大？「过于自信」是什么意思？
- 如果一个分类器的输出完全均匀（每个类概率 1/27），loss 应该是多少？
- 初始 loss 偏高会对训练造成什么实际影响？（提示：早期梯度的浪费）

---

### 题 2：修正初始 Loss（基础）

**函数**：`fix_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647)`

**要求**：
- 在题 1 的基础上，对输出层做初始化修正：
  - `W2 *= 0.01`（缩小输出层权重，使 logits 趋近于零）
  - `b2 = torch.zeros(27)`（偏置归零）
- 同样固定随机种子，返回修正后的初始 loss

**预期结果**：
- loss ≈ 3.29 ~ 3.30，非常接近 ln(27) ≈ 3.298
- 说明修正后网络不再「过于自信」，输出接近均匀分布

**思考**：
- 为什么缩放 `W2` 和归零 `b2` 能让 loss 降到 ln(27)？
- 这个技巧的本质是什么？（提示：让 softmax 输入趋近于零 → 输出趋近均匀）
- 类似的思路还能用在哪里？（提示：隐藏层的初始化 → 题 4 的 tanh 饱和问题）

---

### 题 3：BatchNorm1d 实现（基础）

**类**：`BatchNorm1d`

**要求**：
- 从零实现一维 Batch Normalization 层
- 包含以下方法和属性：
  - `__init__(dim, eps=1e-5, momentum=0.1)`：初始化可训练参数 `gamma`（全 1）、`beta`（全 0），以及 running statistics
  - `__call__(self, x)`：根据 `self.training` 标志决定是 training 模式还是 eval 模式
  - `parameters()`：返回 `[self.gamma, self.beta]`
- `gamma` 和 `beta` 需要设置 `requires_grad=True`
- running_mean 和 running_var 用 `torch.no_grad()` 更新

**training 模式**（`self.training = True`）：
1. 计算当前 mini-batch 的均值和方差
2. 标准化：`x_hat = (x - mean) / sqrt(var + eps)`
3. 缩放和平移：`y = gamma * x_hat + beta`
4. 更新 running statistics（指数移动平均）

**eval 模式**（`self.training = False`）：
1. 直接使用 running_mean 和 running_var 进行标准化
2. 不更新 running statistics

**思考**：
- 为什么 training 时用 batch 统计量，eval 时用 running 统计量？
- `momentum` 参数控制什么？值太大会怎样？太小会怎样？
- `gamma` 和 `beta` 的作用是什么？如果没有它们，BN 会损失什么表达能力？
- 为什么 BN 能让深层网络更容易训练？（提示：减少 internal covariate shift）

---

### 题 4：Tanh 饱和诊断（基础）

**函数**：`diagnose_tanh_saturation(hpreact)`

**要求**：
- 输入：隐藏层线性输出 `hpreact`（经过 `tanh` 之前的值），形状 `(N, n_hidden)`
- 输出：饱和比例 `saturation_ratio`（Python float），即 `|h| > 0.99` 的元素占总元素的比例
- 其中 `h = torch.tanh(hpreact)`

**背景**：
- 当 `tanh` 的输出接近 ±1 时，梯度接近 0（因为 `tanh'(x) = 1 - tanh(x)²`）
- 大量神经元饱和 → 梯度消失 → 训练停滞
- 这是诊断网络初始化质量的重要工具

**预期结果**：
- 未修正的初始化：饱和比例可能 > 40%
- 使用 Kaiming 初始化后：饱和比例显著下降

**思考**：
- `|h| > 0.99` 这个阈值是怎么来的？用 0.95 或 0.999 有什么区别？
- 饱和的神经元「死了」是什么意思？它的梯度是多少？
- 为什么 Kaiming 初始化能缓解 tanh 饱和问题？（提示：保持激活值的方差在前向传播中稳定）
- 如果把 tanh 换成 ReLU，还会有饱和问题吗？会有什么不同的问题？

---

### 题 5：含 BatchNorm 的深层网络（🌟 拓展）

**函数**：`train_deep_bn(words, block_size=3, n_embd=10, n_hidden=200, steps=200000, seed=2147483647)`

**要求**：
- 构建含 BatchNorm 的 MLP 训练到 dev loss < 2.1
- 网络结构：Embedding → [Linear → BatchNorm → Tanh] × 1 → Linear → 输出
- 应用以下最佳实践：
  - Kaiming 初始化：`W1 *= (5/3) / (n_embd * block_size) ** 0.5`
  - 输出层修正：`W2 *= 0.01, b2 *= 0`
  - BatchNorm 偏置替代：`b1 = torch.zeros(n_hidden)`（BN 有 beta）
  - Mini-batch 训练（batch_size=32 或 64）
  - 学习率衰减

**步骤提示**：
```python
# 1. 数据划分（80/10/10）
# 2. 初始化参数 + BatchNorm 层
# 3. 训练循环：forward → BN → loss → backward → update
# 4. 每 10000 步打印 train loss
# 5. eval 模式下评估 dev loss
# 6. 返回 dev loss 和参数
```

**思考**：
- BN 层放在激活函数前面还是后面？Andrej 的选择是什么？为什么？
- 有了 BN 后，W1 的初始化还那么重要吗？尝试不同的初始化，观察 BN 的「鲁棒性」
- BN 的 running_mean 在训练初期波动大，这会影响推理吗？怎么做能减轻影响？
- 为什么 dev loss < 2.1 在这个任务上是一个合理的目标？

---

## ✅ 提交检查清单

- [ ] 所有 4 道基础题通过测试
- [ ] 拓展题（题 5）已尝试
- [ ] 能回答每道题后面的「思考」问题
- [ ] 代码中添加了必要的注释说明你的理解

---

## 💡 学习建议

1. **先观察再修正**：题 1 和题 2 的核心是「先诊断出问题，再对症下药」。这比直接给出正确初始化更有教育意义
2. **理解数字背后的含义**：ln(27) ≈ 3.298 不是一个魔法数字，它是「27 个类完全均匀预测时的交叉熵」
3. **画图帮助理解**：用 `plt.hist(h.view(-1).tolist(), bins=50)` 画出 tanh 输出的分布，直观看到饱和
4. **对比实验**：有 BN vs 无 BN、有 Kaiming vs 无 Kaiming，四组实验做下来感受最深

---

*Good luck! 🚀*
