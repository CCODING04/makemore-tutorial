# 作业 1：Bigram 字符级语言模型

> **对应教程**：Part 1 — Bigrams
>
> **截止日期**：完成 Part 1 学习后

---

## 📋 概述

本作业检验你对 Part 1 Bigram 语言模型的理解。你将亲手实现一个完整的字符级语言模型流水线：从构建计数矩阵、计算概率、采样生成名字，到计算损失函数，最后用梯度下降训练一个等价的神经网络版本。

完成本作业后，你应该能够：

- 理解 Bigram 模型如何捕捉字符间的共现关系
- 掌握 Laplace 平滑的作用和实现
- 理解负对数似然（NLL）作为模型质量度量的意义
- 建立「计数方法 ≈ 梯度方法」的直觉

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
assignments/assignment_1/
├── README.md              # 本文件
├── bigram_exercises.py    # 👈 你需要编辑的文件
└── test_bigram_exercises.py  # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_1

# 方式一：直接运行测试脚本
python test_bigram_exercises.py

# 方式二：用 pytest 逐题运行
pip install pytest
pytest test_bigram_exercises.py -v
```

---

## 📝 题目列表

### 题 1：Bigram 计数矩阵（基础）

**函数**：`build_bigram_matrix(words)`

**要求**：
- 输入：名字列表 `words`（list of str）
- 输出：`(27, 27)` 的计数 tensor（dtype=int32）
- 字符映射：`'.' = 0, 'a' = 1, 'b' = 2, ..., 'z' = 26`
- 每个名字的首尾添加 `.` 作为起止符，统计相邻字符对的出现次数

**提示**：
```python
# stoi 映射
stoi = {s: i+1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
stoi['.'] = 0
# itos 反向映射
itos = {i: s for s, i in stoi.items()}
```

**思考**：
- 为什么要在名字首尾加 `.`？如果只加开头不加结尾会怎样？
- 矩阵中 `N[0, :]` 表示什么物理含义？

---

### 题 2：概率矩阵（基础）

**函数**：`compute_probabilities(N, smoothing=1)`

**要求**：
- 输入：计数矩阵 `N`（27×27），平滑系数 `smoothing`（默认 1）
- 输出：概率矩阵 `P`（27×27），每行之和为 1
- 实现模型平滑（Model Smoothing / Laplace Smoothing）：`P[i, j] = (N[i, j] + smoothing) / (N[i, :].sum() + 27 * smoothing)`

**思考**：
- `smoothing=0` 时会出现什么问题？对生成和损失计算各有什么影响？
- `smoothing` 值越大，生成的名字会有什么变化趋势？为什么？
- 尝试 `smoothing=0, 1, 10, 100`，观察生成名字的变化。

---

### 题 3：采样生成（基础）

**函数**：`generate_names(P, n=5, seed=2147483647)`

**要求**：
- 输入：概率矩阵 `P`，生成数量 `n`，随机种子 `seed`
- 输出：生成的名字列表（list of str）
- 从起始符 `.` 开始，根据 `P` 逐字符采样，遇到 `.` 结束

**提示**：
```python
g = torch.Generator().manual_seed(seed)
ix = 0  # 从 '.' 开始
while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    if ix == 0:
        break
    # 收集字符
```

**思考**：
- 固定随机种子的作用是什么？为什么机器学习实验中要重视可重复性？
- 你生成的名字质量如何？和真实英文名比呢？这说明 Bigram 模型有什么局限性？

---

### 题 4：NLL 损失（基础）

**函数**：`compute_nll_loss(P, words)`

**要求**：
- 输入：概率矩阵 `P`，名字列表 `words`
- 输出：平均负对数似然（float）
- 计算公式：`loss = -Σ log(P[ch1, ch2]) / count`，对所有 bigram 取平均

**提示**：
```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        log_likelihood += torch.log(P[ix1, ix2])
        n += 1
nll = -log_likelihood / n
```

**思考**：
- 均匀分布下（每个字符等概率出现），NLL 是多少？（提示：`log(27)` ≈ 3.296）
- 你的模型 NLL 是多少？比均匀分布好还是差？好多少？
- NLL 和「每个字符平均需要多少 bit 来编码」有什么关系？

---

### 题 5：神经网络训练（🌟 拓展）

**函数**：`train_bigram_nn(words, epochs=100, lr=50, seed=2147483647)`

**要求**：
- 输入：名字列表 `words`，训练轮数 `epochs`，学习率 `lr`
- 输出：训练好的权重 `W`（27×27），最终 loss（float）
- 用 one-hot 编码 + 单层线性网络 + softmax 实现等价于计数方法的模型
- 用梯度下降训练，使 NLL 最小化

> ⚠️ **关于跳过逻辑**：测试对未实现的拓展题做了"返回 `None` 即跳过"的处理。直接运行 `python test_bigram_exercises.py` 时会明确打印"跳过"；但在 pytest 下，跳过的用例会显示为 **PASSED**（实为跳过），请以脚本输出或自己实现的情况为准。

**提示**：
```python
# 1. 构建 training set (xs, ys)
# 2. 初始化 W
g = torch.Generator().manual_seed(seed)
W = torch.randn((27, 27), generator=g, requires_grad=True)
# 3. 训练循环
for i in range(epochs):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(xs)), ys].log().mean()
    # backward pass
    W.grad = None
    loss.backward()
    W.data += -lr * W.grad
```

**思考**：
- 训练足够久后，神经网络的 loss 应该趋近于什么值？为什么？
- 为什么说「神经网络方法」和「计数方法」在数学上是等价的？
- 正则化（对 W 加权衰减）对应计数方法中的什么操作？（提示：和 smoothing 有关）

---

## ✅ 提交检查清单

- [ ] 所有 4 道基础题通过测试
- [ ] 拓展题（题 5）已尝试
- [ ] 能回答每道题后面的「思考」问题
- [ ] 代码中添加了必要的注释说明你的理解

---

## 💡 学习建议

1. **先看视频再动手**：确保你理解了 Part 1 的核心概念
2. **先手动推演**：用 `words = ['emma']` 手动算一遍 bigram 计数，验证你的函数
3. **多实验**：改变 smoothing、learning rate、epochs，观察结果变化
4. **理解直觉**：最重要不是代码本身，而是理解「为什么这样做」

---

*Good luck! 🚀*

---

## 🎯 面试直通车（话术卡：结论 → 原理 → 边界）

> 每张卡按"总分总"组织：先一句话结论压场，再两三句原理支撑，最后一句边界/代价收尾——面试答题的固定骨架。

**Q1："bigram 的计数方法和神经网络方法为什么说是等价的？"**

- **结论**：one-hot 接单层线性再过 softmax，就是把计数法改写成可微分的条件概率表，两者学到同一个分布。
- **原理**：one-hot 向量乘 $W$ 等价于按索引取行，softmax 把每一行变成该字符下一位的条件分布。课程实测：计数法平均 NLL 约 2.45，神经网络训练后收敛到约 2.49（实测 2.4901），差距只来自随机初始化和有限步优化；参数 $27 \times 27 = 729$ 个，恰好一一对应 729 个 bigram 计数。
- **边界**：等价性只在"单层线性 + softmax"这一结构下成立，一旦加隐藏层（Part 2 的 MLP）就进入了不同的函数族。

**Q2："为什么损失函数要取负对数，而不是直接最大化概率？"**

- **结论**：log 把概率连乘变成求和、防止数值下溢，负号把"最大化似然"翻转成习惯上的"最小化损失"。
- **原理**：整个数据集的似然是成千上万个小于 1 的概率相乘，会下溢成 0；取 log 后变成稳定可加的 $-\frac{1}{N}\sum \log P$。均匀猜测的基线是 $\log 27 \approx 3.296$，bigram 模型实测约 2.45，越小越好；换底成 $\log_2$ 后直接对应"每个字符平均需要多少 bit 编码"。
- **边界**：$\log 0 = -\infty$，遇到训练集未出现的 bigram 会直接爆炸，必须配合平滑。

**Q3："概率归一化时最容易踩的 broadcasting 陷阱是什么？"**

- **结论**：`(27, 27)` 的矩阵除以 `counts.sum(1)` 时若不加 `keepdims=True`，会沿错误的方向广播。
- **原理**：PyTorch 从末维对齐形状，`(27, 27)` 遇到 `(27,)` 会被当作 `(1, 27)`，结果变成每列和为 1，而不是想要的行为 1；正确写法是 `counts / counts.sum(1, keepdims=True)`。
- **边界**：这个 bug 不报错、loss 照样能下降，只有采样质量悄悄变差，必须用逐行 `P.sum(1)` 验证。

**Q4："loss 降到 2.45，生成的名字就一定像样了吗？"**

- **结论**：不一定，2.45 只是 bigram 这类模型的极限，生成质量本质上受"只能看 1 个字符上下文"的限制。
- **原理**：课程实测 bigram 平均 NLL 约 2.45，但采样仍产出大量伪词；把上下文扩到 3 个字符后（Part 2 MLP），dev loss 降到约 2.2，名字明显更像样。采样本身是从 $P$ 的行分布做多项式抽样，固定 `seed=2147483647` 才能复现。
- **边界**：loss 衡量分布匹配程度，与人类观感不完全一致，评估要两者结合。

**Q5："对 W 加 L2 正则化，对应计数方法里的什么操作？"**

- **结论**：对应 Laplace 平滑（模型平滑），两者都是把分布"往均匀拉"的软约束。
- **原理**：课程在 loss 中加 $0.01 \times (W^2).mean()$，压着 $W$ 趋向 0、logits 趋向 0、softmax 输出趋向均匀，与 $P = (N+1)/(N+27)$ 的平滑方向一致；教程特意用 `mean()` 而非 `sum()`，让正则项与 NLL 量纲匹配。
- **边界**：平滑强度是超参数，太弱挡不住 $\log 0$，太强会把模型压向均匀分布的 $\log 27 \approx 3.296$。
