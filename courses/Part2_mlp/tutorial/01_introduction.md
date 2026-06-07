# 01 — 从 Bigram 到 MLP

## 📖 前置知识：Bigram 的局限

在 [Part 1](../../Part1_bigrams/) 里，我们做了一个 Bigram 语言模型：**只看前 1 个字符**，预测下一个字符。

比如名字 `emma`，Bigram 看到的是：

```
. → e
e → m
m → m
m → a
a → .
```

⚠️ 问题来了：当你看到 `m` 的时候，模型完全不知道前面是 `e` 还是别的什么字母。它只知道 "m 后面可能跟什么"，但不知道 "前面是 e 的 m 后面应该跟什么"。

这就像你只看了一个字的上下文就要猜下一个字 —— 信息太少了！

> 💡 语言模型的核心思想：**上下文越长，预测越准**。

---

## 🎯 这一课能学到什么

学完 Part 2，你将：

1. 把上下文从 **1 个字符扩展到 3 个字符**（block_size=3）
2. 理解 **Embedding** —— 把字符变成向量
3. 搭建一个 **多层感知机（MLP）**，比 Bigram 强得多
4. 学会 **Train/Dev/Test** 数据划分
5. 用 **Minibatch SGD** 高效训练

---

## 📦 数据集准备：block_size=3

### 为什么看 3 个字符？

Bengio 等人 2003 年的论文中提出：用前 3 个字符预测下一个字符，效果比只看 1 个好很多。

> 🔑 **block_size**（也叫 context length）：用多少个历史字符来预测下一个。

### 展开数据集

当 `block_size=3` 时，名字 `emma` 展开成：

```
block_size=3 时，名字 "emma" 展开为：

输入       →  输出(预测)
...        →  e
..e        →  m
.em        →  m
emm        →  a
mma        →  .(结束)
```

你看，每一行的输入都是 **3 个字符**（不够的用 `.` 填充），输出是下一个字符。

对应到代码：

```python
block_size = 3

for word in words[:5]:  # 先看前 5 个名字
    context = [0] * block_size  # [0, 0, 0] → 0 代表 '.'
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        y.append(ix)
        context = context[1:] + [ix]  # 滑动窗口！
```

🔑 注意 `context = context[1:] + [ix]` —— 这就是**滑动窗口**，每次把最老的字符丢掉，加入新字符。

> 📜 完整代码见 [`../scripts/01_explore_data.py`](../scripts/01_explore_data.py) 和 [`../scripts/02_dataset_with_context.py`](../scripts/02_dataset_with_context.py)

### 数据长什么样？

```python
# X.shape = (N, 3)  → N 是样本数，3 是 block_size
# y.shape = (N,)    → 每个样本的目标字符索引

print(X[:5])
# tensor([[0, 0, 0],    # ... → e
#         [0, 0, 5],    # ..e → m
#         [0, 5, 13],   # .em → m
#         [5, 13, 13],  # emm → a
#         [13, 13, 1]]) # mma → .

print(y[:5])
# tensor([ 5, 13, 13,  1,  0])
```

每个数字是字符的索引（0='.', 1='a', 2='b', ...）。

---

## ✂️ Train / Dev / Test 划分

💡 我们把所有名字分成**三份**：

```
┌──────────────────────────────────────────────────────┐
│              全部名字数据 (32033 个名字)              │
├────────────────────┬──────────┬───────────────────────┤
│   Train (80%)      │ Dev(10%) │   Test (10%)          │
│   ~25626 个        │ ~3203 个 │   ~3204 个            │
│                    │          │                       │
│   用来训练模型     │ 调超参数 │   最终评估（只用一次）│
│   (反复使用)       │ (适量用) │   (神圣不可侵犯)      │
└────────────────────┴──────────┴───────────────────────┘
```

### 为什么需要三份？

| 数据集 | 用途 | 使用频率 |
|--------|------|----------|
| **Train** | 训练模型参数（权重） | 每个迭代都用 |
| **Dev**（验证集） | 调超参数（学习率、embedding 维度等） | 定期评估 |
| **Test** | 最终报告模型效果 | **只用一次** |

⚠️ 如果你用 Test 集来调参数，那就等于 "考试前看到了答案" —— 你的模型看起来很好，但遇到新数据就不行了。

```python
import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])        # Train
Xdev, Ydev = build_dataset(words[n1:n2])     # Dev
Xte, Yte = build_dataset(words[n2:])         # Test
```

---

## 🧭 下一步

数据准备好了，接下来该搭建模型了！

👉 [02 — MLP 架构](02_mlp_architecture.md)
