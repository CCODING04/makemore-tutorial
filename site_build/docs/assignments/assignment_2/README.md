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
├── README.md              # 本文件
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

**函数**：`tuning_experiment(words, block_size=3, n_embd=10, n_hidden=200, steps=200000, lr=0.1, seed=2147483647)`

**要求**：
- 实现 `tuning_experiment` 函数，尝试不同的超参数组合训练模型
- `seed` 参数用于初始化参数的随机种子（`torch.Generator().manual_seed(seed)`），保证实验可复现
- 目标阈值（两种口径）：
  - **完整挑战**：`steps=200000`（200k 步）完整训练后，验证 loss **< 2.3**
  - **测试口径**：测试只跑 `steps=1000` 的小预算，只要求验证 loss **< 2.5**（用于验证流程正确，不要求达到完整训练的精度）
- 至少尝试以下组合中的 3 种，记录结果：

| 配置 | n_embd | n_hidden | block_size | 预期验证 loss |
|------|--------|----------|------------|---------------|
| A | 10 | 200 | 3 | ~2.3 |
| B | 20 | 300 | 3 | ~2.2 |
| C | 10 | 200 | 5 | ~2.1 |
| D | 20 | 300 | 5 | ~2.0 |

> 注：表中为 **200k 步完整训练**下的参考值，反映配置间的相对趋势；达标线以默认配置 A 的 **< 2.3** 为准。

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

# 用题 1 的 build_dataset 构建训练/验证集
Xtr, Ytr = build_dataset(train_words, block_size=block_size)
Xval, Yval = build_dataset(val_words, block_size=block_size)

# 初始化参数（⚠️ 必须小尺度初始化！）
# std=1 时初始 logits 过大（初始 CE 实测 ≈18），1000 步内降不到 <2.5（实测 3.64）；
# 缩放后初始 CE≈3.37≈ln(27)=3.296，训练才能正常收敛。
# ⚠️ 正确写法是「先乘缩放系数，再 .requires_grad_(True)」：
# 若写成 torch.randn(..., requires_grad=True) * 0.1，乘法产生的是非叶子张量，
# 梯度永远存不进去（p.grad 恒为 None）。
g = torch.Generator().manual_seed(seed)
C  = torch.randn(27, n_embd, generator=g, requires_grad=True)
W1 = (torch.randn(block_size * n_embd, n_hidden, generator=g) * 0.1).requires_grad_(True)
b1 = (torch.randn(n_hidden, generator=g) * 0.01).requires_grad_(True)
W2 = (torch.randn(n_hidden, 27, generator=g) * 0.1).requires_grad_(True)
b2 = (torch.randn(27, generator=g) * 0.01).requires_grad_(True)

# 训练循环：mini-batch 采样 + 学习率调度
for i in range(steps):
    # mini-batch：随机采 32 个样本索引，只在小批量上计算梯度
    ix = torch.randint(0, Xtr.shape[0], (32,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    current_lr = 0.1 if i < 100000 else 0.01  # 简单的 lr decay
    loss = train_step(Xb, Yb, C, W1, b1, W2, b2, lr=current_lr)
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

---

## 🎯 面试直通车（话术卡：结论 → 原理 → 边界）

> 每张卡按"总分总"组织：先一句话结论压场，再两三句原理支撑，最后一句边界/代价收尾——面试答题的固定骨架。

**Q1："`C[X]` 这种 embedding 写法到底在做什么？"**

- **结论**：在做查表——`C` 是 `(27, n_embd)` 的可学习矩阵，`C[X]` 按整数索引取出对应行。
- **原理**：它和 one-hot 乘 `C` 数学等价，但不用构造 `(N, 27)` 的稠密 one-hot 张量，省内存省计算；输出 `(N, block_size, n_embd)` 再 `view` 拼平送进 MLP。课程建议把 `n_embd` 设为 2 画散点图，可以直接观察元音等字符在 embedding 空间聚簇。
- **边界**：`n_embd` 是容量旋钮，课程调参表里 `n_embd` 10→20、`n_hidden` 200→300 才把 dev loss 从约 2.3 压到约 2.2，加大 embedding 不是免费的。

**Q2："为什么数据要切 train / dev / test 三份？"**

- **结论**：train 负责拟合、dev 负责选模型、test 只在最终碰一次，防止拿"考卷"调参。
- **原理**：课程按 80/10/10 划分，先 `random.seed(42)` 打乱再切，保证可复现；健康状态是 train loss 约 2.1、dev loss 约 2.2 的小差距，说明模型学到的是可迁移规律。所有超参（`n_embd`、`block_size`、学习率）都只依据 dev 选。
- **边界**：反复对着 dev 调参，dev 也会被间接"过拟合"，test 集只能用于一次性的最终报告。

**Q3："训练前为什么要做初始 loss 的 sanity check？"**

- **结论**：理想初始 loss 应约等于 $\ln 27 \approx 3.296$，即对 27 个类均匀瞎猜的水平，明显偏离说明初始化有问题。
- **原理**：输出层用 randn 初始化时 logits 又大又不均衡，softmax 一上来就"过度自信"，本课实测：Part 2 的 randn 初始化初始 loss ≈19.5，Part 3 教程记录 3.7~4.0（浅层结构）、极端结构超过 20；把 `W2` 缩到 0.01 倍、`b2` 置零即可压回约 3.29。
- **边界**：它只校准输出层置信度，隐藏层的方差问题要另做激活/饱和度诊断，且初始 loss 正常不代表训练必然收敛。

**Q4："怎么从 train / dev loss 判断欠拟合还是过拟合？"**

- **结论**：先看 train loss 高不高（欠拟合），再看 dev 与 train 的差距大不大（过拟合）。
- **原理**：课程给出三档对照：train 2.5 / dev 2.6 双高是欠拟合，应加大模型；train 2.1 / dev 2.2 是健康；train 1.5 / dev 2.5 差距拉大是过拟合，应加数据或正则。课程调参表把 `block_size` 3→5 后 dev loss 从约 2.3 降到约 2.1，就是修欠拟合的典型操作。
- **边界**：差距的解读依赖数据规模，小数据集上轻微差距属正常，不能一刀切。

**Q5："block_size（上下文长度）应该如何权衡？"**

- **结论**：它是"看得越远、学得越好，但输入维度和计算线性变贵"的旋钮。
- **原理**：`block_size=1` 时模型退化为 Part 1 的 bigram 网络；课程调参表中 `block_size` 3→5（`n_embd=10, n_hidden=200` 不变）使 dev loss 约 2.3→约 2.1。代价是拼接后的输入维度按 $block\_size \times n\_embd$ 线性增长。
- **边界**：收益递减，扩到 10 未必继续降 loss，还可能引入更多无关上下文，一切以 dev 曲线为准。
