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
├── assignment.md          # 本文件
├── bigram_exercises.py    # 👈 你需要编辑的文件
└── test_bigram_exercises.py  # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_1
python test_bigram_exercises.py
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
