# 作业 2：MLP 字符级语言模型

> **对应教程**：Part 2 — MLP
>
> **前置**：完成作业 1（Bigram）

---

## 📋 概述

本作业检验你对 Part 2 MLP 字符级语言模型的理解。你将从 Bigram 的单字符上下文，扩展到多字符上下文（block_size=3），亲手实现 Embedding 查表、隐藏层前向传播、完整训练循环和模型评估。

完成本作业后，你应该能够：

- 理解 Embedding 如何将离散字符映射为连续向量
- 掌握 block_size（上下文窗口）对模型能力的影响
- 手动实现 MLP 的 forward + backward + 参数更新
- 理解训练集 / 验证集 / 测试集的划分意义

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
assignments/assignment_2/
├── assignment.md          # 本文件
├── mlp_exercises.py       # 👈 你需要编辑的文件
└── test_mlp_exercises.py  # 测试脚本
```

### 运行测试

```bash
cd assignments/assignment_2
python test_mlp_exercises.py
```

---

## 📝 题目列表

### 题 1：构建数据集（基础）

**函数**：`build_dataset(words, block_size=3)`

**要求**：
- 输入：名字列表 `words`（list of str），上下文长度 `block_size`（默认 3）
- 输出：`(X, Y)`，其中 `X` 形状 `(N, block_size)`，`Y` 形状 `(N,)`，dtype 均为 int64
- 字符映射：`'.' = 0, 'a' = 1, ..., 'z' = 26`
- 每个名字首尾添加 `.`，用滑动窗口构造样本

**示例**（`block_size=3`，`words=['emma']`）：
```
输入上下文 → 预测目标
... → e
..e → m
.em → m
emm → a
mma → .
```
对应 `X = [[0,0,0],[0,0,5],[0,5,13],[5,13,13],[13,13,1]]`，`Y = [5,13,13,1,0]`

**提示**：
```python
for w in words:
    context = [0] * block_size  # 初始上下文全是 '.'
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]  # 滑动窗口
```

**思考**：
- `block_size=1` 时和 Bigram 模型是什么关系？
- 为什么 `X` 中第一个样本全是 0？它的物理含义是什么？
- 如果把 `block_size` 增大到 10，`X` 的形状会变成什么样？有什么好处和坏处？

---

### 题 2：MLP 前向传播（基础）

**函数**：`mlp_forward(X, C, W1, b1, W2, b2)`

**要求**：
- 输入：
  - `X`：`(N, block_size)` 整数 tensor，每行是一个上下文
  - `C`：`(27, n_embd)` Embedding 矩阵
  - `W1`：`(block_size * n_embd, n_hidden)` 第一层权重
  - `b1`：`(n_hidden,)` 第一层偏置
  - `W2`：`(n_hidden, 27)` 第二层权重
  - `b2`：`(27,)` 第二层偏置
- 输出：`logits`，形状 `(N, 27)`，不需要 softmax
- 激活函数：隐藏层用 `tanh`

**步骤**：
1. Embedding 查表：`emb = C[X]` → `(N, block_size, n_embd)`
2. 拼接：`emb_cat = emb.view(emb.shape[0], -1)` → `(N, block_size * n_embd)`
3. 隐藏层：`h = torch.tanh(emb_cat @ W1 + b1)` → `(N, n_hidden)`
4. 输出层：`logits = h @ W2 + b2` → `(N, 27)`

**思考**：
- 为什么用 `tanh` 而不是 `ReLU`？在这个小模型里有区别吗？
- `C[X]` 是怎么工作的？和 one-hot + 矩阵乘法有什么等价关系？
- 如果 `n_embd=2`，画出 `C` 的 2D 散点图，能看到什么有趣的模式吗？

---

### 题 3：训练单步（基础）

**函数**：`train_step(X, Y, C, W1, b1, W2, b2, lr=0.1)`

**要求**：
- 执行一次完整的 forward → loss → backward → 参数更新
- 使用交叉熵损失：`F.cross_entropy(logits, Y)`
- 更新所有 5 组参数：`C, W1, b1, W2, b2`
- 返回当前 loss（Python float）

**步骤**：
1. Forward：调用 `mlp_forward` 得到 logits
2. Loss：`F.cross_entropy(logits, Y)`
3. Backward：清零梯度 → `loss.backward()`
4. Update：`param.data -= lr * param.grad`
5. 返回 `loss.item()`

**思考**：
- 为什么要先清零梯度（`param.grad = None`）再 backward？
- 如果忘了清零梯度，会发生什么？loss 会正常下降吗？
- 为什么这里用 `param.data -= lr * param.grad` 而不是 `param -= lr * param.grad`？

---

### 题 4：模型评估（基础）

**函数**：`evaluate(X, Y, C, W1, b1, W2, b2)`

**要求**：
- 在给定数据上计算 loss，**不计算梯度**，不修改参数
- 返回 loss（Python float）

**提示**：
```python
with torch.no_grad():
    logits = mlp_forward(X, C, W1, b1, W2, b2)
    loss = F.cross_entropy(logits, Y)
```

**思考**：
- `torch.no_grad()` 的作用是什么？为什么要用它？
- 评估时不用 `no_grad()` 会怎样？结果会错吗？
- 训练 loss 和验证 loss 的差距说明了什么？

---

### 题 5：调参实验（🌟 拓展）

**函数**：`tuning_experiment(words, block_size=3, n_embd=10, n_hidden=200, steps=200000, lr=0.1)`

**要求**：
- 实现 `tuning_experiment` 函数，尝试不同的超参数组合训练模型
- 目标：在验证集上达到 loss < 2.2
- 至少尝试以下组合中的 3 种，记录结果：

| 配置 | n_embd | n_hidden | block_size | 预期验证 loss |
|------|--------|----------|------------|---------------|
| A | 10 | 200 | 3 | ~2.3 |
| B | 20 | 300 | 3 | ~2.2 |
| C | 10 | 200 | 5 | ~2.1 |
| D | 20 | 300 | 5 | ~2.0 |

**提示**：
```python
# 数据划分
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
train_words = words[:n1]
val_words = words[n1:n2]

# 学习率调度
for i in range(steps):
    lr = 0.1 if i < 100000 else 0.01  # 简单的 lr decay
    loss = train_step(Xb, Yb, C, W1, b1, W2, b2, lr=lr)
```

**思考**：
- 增加 `n_embd` 和增加 `n_hidden` 各自的好处是什么？
- 为什么 `block_size=5` 比 `block_size=3` 效果好？什么时候会增加不大？
- 如果把模型再加大，验证 loss 会一直下降吗？为什么？

---

## ✅ 提交检查清单

- [ ] 所有 4 道基础题通过测试
- [ ] 拓展题（题 5）已尝试
- [ ] 能回答每道题后面的「思考」问题
- [ ] 代码中添加了必要的注释说明你的理解

---

## 💡 学习建议

1. **回顾 Part 1**：确保你理解了 Bigram 模型的局限——只能看前一个字符
2. **手动推演**：用 `words=['emma']`, `block_size=3` 手动写出 `X` 和 `Y`，验证你的 `build_dataset`
3. **理解维度**：这是 MLP 最容易出错的地方。每一步都检查 tensor 的形状是否符合预期
4. **可视化 Embedding**：如果 `n_embd=2`，用 `plt.scatter` 画出 `C` 的二维分布，你能看到元音字母聚在一起吗？

---

*Good luck! 🚀*
