# 02 — MLP 架构

## 🧱 Embedding 层：把字符变成向量

### 什么是 Embedding？

在 Bigram 里，我们用 one-hot 向量表示字符 —— 一个长度 27 的向量，只有一个位置是 1。这样太浪费了，而且字符之间没有任何关系。

💡 **Embedding** 的想法：给每个字符分配一个**低维向量**（比如 2 维或 10 维），让模型自己学习这些向量应该长什么样。

```
字符 'a' (索引1)  →  C[1]  →  [0.3, -0.1]  (2维向量)
字符 'm' (索引13) →  C[13] →  [-0.5, 0.8]
字符 '.' (索引0)  →  C[0]  →  [0.1, 0.2]
```

`C` 是一个形状为 `(27, 2)` 的矩阵，27 个字符，每个字符 2 维。这 2 维就是字符在"空间"中的位置。

### 代码实现

```python
import torch

# Embedding 矩阵：27 个字符，每个用 2 维向量表示
# requires_grad=True：C 是要训练的参数，PyTorch 需要为它记录梯度
C = torch.randn((27, 2), requires_grad=True)

# 查表：输入索引 5，得到对应的 2 维向量
print(C[5])        # tensor([0.xx, -0.xx])
print(C[[5, 13]])  # 也可以一次查多个 → (2, 2) 矩阵
```

🔑 关键点：`C` 就是我们要训练的参数！一开始是随机的，训练过程中会慢慢变成有意义的向量。所以创建时必须带 `requires_grad=True`（后面 03 章的 `loss.backward()` 要靠它算梯度，漏掉会直接报错）。

> 📜 完整代码见 [`../scripts/03_embedding.py`](../scripts/03_embedding.py)

---

## 🏗️ MLP 前向传播

### 网络结构

我们参考 Bengio 2003 论文的架构，搭一个两层 MLP：

```
                ┌──────────────────────────────────────────┐
                │            MLP 网络结构                  │
                │                                          │
输入:           │                                          │
  3 个字符索引  │    ┌─────┐                                │
  [5, 13, 13] ─┼───→│  C  │ Embedding (27×2)               │
                │    └──┬──┘                               │
                │       │ 查表得到 3 个 2 维向量            │
                │       ▼                                  │
                │    [emb] 拼接成 6 维向量 (3×2)           │
                │       │                                  │
                │    ┌──┴──┐                               │
                │    │ W1  │ Linear (6 → 100)              │
                │    │+b1  │                               │
                │    └──┬──┘                               │
                │       │                                  │
                │    ┌──┴──┐                               │
                │    │tanh │ 激活函数                      │
                │    └──┬──┘                               │
                │       │ 100 维隐藏层                     │
                │    ┌──┴──┐                               │
                │    │ W2  │ Linear (100 → 27)             │
                │    │+b2  │                               │
                │    └──┬──┘                               │
                │       │                                  │
                │       ▼                                  │
                │   logits (27维) → softmax → 概率          │
                └──────────────────────────────────────────┘
```

### 一步步写代码

**Step 1: Embedding + 拼接**

```python
# X 是 (N, 3) 的索引矩阵，C 是 (27, 2) 的 Embedding
emb = C[X]           # (N, 3, 2) → 每个样本 3 个字符，每个字符 2 维
emb_cat = emb.view(emb.shape[0], -1)  # (N, 6) → 拼成一维向量
```

### `C[X]` 高级索引详解

`C[X]` 是 PyTorch 的**高级索引**操作，也是 Part 2 最核心的操作之一。让我们拆开看：

```python
C = torch.randn(27, 2)    # 27 个字符，每个字符用 2 维向量表示
X = torch.tensor([[0, 0, 0],   # 第 1 个样本：上下文是 "...", 即 ".",".","."
                  [0, 0, 5]])  # 第 2 个样本：上下文是 "..e", 即 ".",".","e"

emb = C[X]  # 形状：(2, 3, 2)
```

**发生了什么？**

`X` 中的每个整数都被当作 `C` 的**行索引**，替换成 `C` 中对应的行：

```text
X[0] = [0, 0, 0]  →  C[[0, 0, 0]] = [C[0], C[0], C[0]]  →  3 个 2 维向量
X[1] = [0, 0, 5]  →  C[[0, 0, 5]] = [C[0], C[0], C[5]]  →  3 个 2 维向量
```

**形状推导规则：**

```text
C 的形状：(27, 2)        ← 词汇表大小 × 嵌入维度
X 的形状：(N, 3)         ← N 个样本 × 上下文长度
C[X] 的形状：(N, 3, 2)  ← N 个样本 × 3 个字符 × 2 维向量
```

> 💡 **和 one-hot 的关系**：`C[X]` 在数学上等价于 `F.one_hot(X, 27) @ C`。形状链：`X` (N,3) → one-hot 后 (N,3,27) → 乘 `C` (27,2) → (N,3,2)（可自行验证 `torch.allclose(C[X], F.one_hot(X, 27).float() @ C)` 为 `True`）。
>
> ⚠️ 但两者开销不同：one-hot 要先创建 (N,3,27) 的大矩阵再做矩阵乘法，而 `C[X]` 直接按行索引查表，省掉这份内存和计算。注意"省"的不是拷贝——`C[X]` 返回的是一个**新张量**（查表结果被拷贝出来），并不是 `C` 的视图，改 `emb` 不会影响 `C`。

#### 🔬 延伸（面试点）：嵌入的梯度是稀疏的

`C[X]` 的反向传播有个常被追问的性质：**每一步只有 batch 中出现过的字符行会收到非零梯度**。原因：对等价式 `C[X] = one_hot(X) @ C` 求导，one-hot 里的 0 会把所有未出现的行乘成 0，梯度像"按行投递"（scatter-add）一样只落进 batch 中出现过的行。

```python
C = torch.randn(27, 2, requires_grad=True)
Xb = torch.tensor([[0, 0, 5], [0, 5, 13]])  # batch 里只出现 {0, 5, 13}
C[Xb].sum().backward()
print((C.grad.abs().sum(1) > 0).nonzero().flatten().tolist())
# 输出: [0, 5, 13]  ← 非零梯度行恰好就是 batch 中出现过的字符
```

27 个字符的小词表看不出好处；但大词表（比如 5 万 token）时，每步只有极小比例的行被更新——这是"稀疏 embedding 更新"以及大模型训练里优化器状态按行懒初始化的动机。

### `view(-1)` 详解

```python
emb = C[X]                        # (N, 3, 2)
emb_cat = emb.view(emb.shape[0], -1)  # (N, 6)
```

**`view` 是什么？**

`view` 改变 tensor 的**形状**，但不改变数据在内存中的排列。它只是改变了 tensor 的"视图"（shape 和 stride）。

**`-1` 是什么意思？**

`-1` 告诉 PyTorch："这个维度你帮我自动算"。计算规则是：

```text
总元素数 = N × 3 × 2 = 6N
view(N, -1) → 第 2 维 = 6N / N = 6
所以结果是 (N, 6)
```

**具体例子：**

```python
emb.shape = (32, 3, 2)  # 32 个样本，每个有 3 个 2 维向量

emb.view(32, -1) → (32, 6)   # 3 个 2 维向量拍扁成 1 个 6 维向量
emb.view(32, 6)  → (32, 6)   # 等价写法，但 -1 更灵活（不用手动算 6）

emb.view(-1)     → (192,)    # -1 用在第 1 维，完全展平：32×3×2 = 192
emb.view(32, 3, 2) → (32, 3, 2)  # 形状不变
```

🔑 **拍扁的顺序**：`view` 按内存顺序拼接——对每个样本，是 3 个 2 维向量**按时间步依序**首尾相接成 6 个数（第 1 个字符的 2 维、第 2 个字符的 2 维、第 3 个字符的 2 维），不是把不同样本混在一起。

> ⚠️ **`view` vs `reshape`**：`view` 要求内存连续（contiguous），如果 tensor 不连续会报错。`reshape` 更灵活，可能返回拷贝。初学阶段两者可以互换，但 `view` 更高效。

🔑 `emb.view(emb.shape[0], -1)` 就是把 3 个 2 维向量拍扁成 1 个 6 维向量。`-1` 让 PyTorch 自动算第二维。

**Step 2: 隐藏层**

```python
W1 = torch.randn((6, 100), requires_grad=True)   # 6 → 100
b1 = torch.randn(100, requires_grad=True)

h = torch.tanh(emb_cat @ W1 + b1)  # (N, 100)
```

💡 `tanh` 把输出压到 [-1, 1] 之间，给网络非线性能力。`emb_cat @ W1` 是 (N,6)@(6,100)→(N,100)；再加 `b1` (100,) 时会自动**广播**到每一行。

📊 **交互动图（页内）**：tanh 与 sigmoid 的对比——**悬停**查看任意点的取值，注意 tanh 关于原点对称（零中心），而 sigmoid 输出全为正：

```plot
{
  "title": "tanh vs sigmoid：为什么隐藏层选 tanh",
  "x": [-6, 6], "y": [-1.3, 1.3],
  "curves": [
    {"expr": "(Math.exp(x)-Math.exp(-x))/(Math.exp(x)+Math.exp(-x))", "label": "tanh(z)"},
    {"expr": "1/(1+Math.exp(-x))", "label": "sigmoid(z)"}
  ],
  "hlines": [
    {"y": 1, "label": "y = 1"}, {"y": -1, "label": "y = -1"}, {"y": 0, "label": "y = 0"}
  ],
  "points": [[0, 0, "(0,0)：tanh 过原点"]]
}
```

**Step 3: 输出层**

```python
W2 = torch.randn((100, 27), requires_grad=True)  # 100 → 27（27 个字符）
b2 = torch.randn(27, requires_grad=True)

logits = h @ W2 + b2          # (N, 27)
```

到这里，5 组参数都创建好了。训练时把它们放进一个列表，方便统一管理（03 章的训练循环会用到）：

```python
parameters = [C, W1, b1, W2, b2]
print(sum(p.numel() for p in parameters))  # 3481 ≈ 3,500
```

**Step 4: Softmax → 概率**

```python
counts = logits.exp()                # 等价于 Bigram 里的 N 矩阵
prob = counts / counts.sum(1, keepdim=True)  # 归一化
# keepdim=True：每行求和后保持 (N, 1) 形状，按行广播除进 (N, 27)；
# 若不加 keepdim，和的形状是 (N,)，无法按行对齐
```

> 📜 完整代码见 [`../scripts/04_mlp_forward.py`](../scripts/04_mlp_forward.py)

---

## 📉 CrossEntropy Loss

### 为什么不用手动算？

你可以手动算 loss：

```python
# 手动版本（不推荐）
logits = h @ W2 + b2
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
loss = -prob[torch.arange(N), Y].log().mean()
```

⚠️ 但这样有三个问题：

1. **数值不稳定**：`exp()` 对大数会溢出 → `inf`
2. **效率低**：PyTorch 没法对这三步做融合优化
3. **反向传播复杂**：自己写 grad 容易出错

### 数学形式

CrossEntropy Loss = softmax + 负对数似然（NLL）。设第 $i$ 个样本的 logits 为 $z_i \in \mathbb{R}^{27}$，正确类别为 $y_i$：

$$p_{i,k} = \frac{e^{z_{i,k}}}{\sum_{j=1}^{27} e^{z_{i,j}}} \qquad \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{i,y_i}$$

一句话：softmax 把 logits 变成概率 → 取出正确类的那个概率 → 取负对数 → 对 batch 求平均。概率越接近 1，$-\log p$ 越接近 0；概率越小，损失越大。

### 一个 3 类 toy 数值例

先不管 27 个字符，用 3 个类别手算一遍（batch=1，正确类别 $y=0$）：

| 步骤 | 计算 | 结果 |
|------|------|------|
| logits | 给定 | $[2.0,\ 0.0,\ -1.0]$ |
| exp | $e^{2.0},\ e^{0},\ e^{-1}$ | $[7.389,\ 1.000,\ 0.368]$ |
| 归一化 | 各自除以和 $8.757$ | $[0.844,\ 0.114,\ 0.042]$ |
| 取正确类 | $y=0$ 对应的概率 $0.844$ | |
| loss | $-\log(0.844)$ | $\approx 0.170$ |

用 PyTorch 验证：`F.cross_entropy(torch.tensor([[2.0, 0.0, -1.0]]), torch.tensor([0]))` → `0.1698`，与手算一致。`F.cross_entropy` 内部做了：

1. 先减去最大值（防止溢出；对本例即算 $e^{z-2}$，概率结果不变）
2. 用 log-sum-exp 技巧算 log-softmax
3. 取负对数似然的均值

数学等价，但**更快、更稳定**。

### 🔎 初始 loss sanity check（训练前必做）

训练前先问一句：**一个"一无所知"的模型，loss 应该是多少？**

- **理论基线**：瞎猜意味着 27 个字符概率均匀，各为 $1/27$，此时：

$$\mathcal{L} = -\log \frac{1}{27} = \ln 27 \approx 3.2958$$

- **全零初始化恰好命中基线**：把 `W2`、`b2` 设成全 0，logits 全为 0，softmax 后恰好是均匀分布，初始 loss **恰好等于** $\ln 27 \approx 3.30$。这就是"一开始模型应该几乎是瞎猜的"sanity check：训练前 loss 若明显偏离 3.3，说明初始化有问题。
- **本教程当前的 `randn` 初始化实测 ≈19.5，远大于 3.30，为什么？** `W1/W2` 用 std=1 的随机数，算出的 logits 方差很大（std ≈ 6.5）。logits 一大，softmax 就会**过度自信**地押注某个（往往是错的）类别，$-\log p$ 被推得很高。跑 `04_mlp_forward.py` 会看到初始 loss = **19.51**——这不是 bug，是初始化尺度问题。
- **现在该怎么办？** 不影响先跑通训练（几十步内 loss 会快速降到 3 以下）。作业题 5 会先用 `(randn * 0.1)` 缩小尺度，让初始 loss 回到 ≈3.3（实测 3.37）；**Part 3** 会系统讲初始化策略（BatchNorm / Kaiming）。

---

## 🧪 课后练习

### Q1: Embedding 维度

> 如果把 Embedding 从 2 维改成 10 维，`W1` 的形状应该是什么？（block_size=3）

<details>
<summary>点击查看答案</summary>

`W1` 的输入维度 = block_size × emb_dim = 3 × 10 = 30。所以 `W1` 的形状是 `(30, 100)`。
</details>

### Q2: view vs reshape

> `emb.view(N, -1)` 和 `emb.reshape(N, -1)` 有什么区别？什么情况下用哪个？

<details>
<summary>点击查看答案</summary>

- `view` 要求张量在内存中是连续的（contiguous），更快但不总是能用
- `reshape` 任何情况都能用，如果内存不连续会自动复制一份
- 实践中：先试 `view`，报错了再用 `reshape`
</details>

### Q3: 为什么用 tanh？

> 隐藏层为什么用 `tanh` 而不是 `sigmoid` 或 `ReLU`？你觉得各有什么优缺点？

<details>
<summary>点击查看答案</summary>

- `tanh`：输出 [-1, 1]，零中心，梯度比 sigmoid 大 → 训练更快
- `sigmoid`：输出 [0, 1]，两端梯度极小 → 容易梯度消失
- `ReLU`：简单高效，但 Bengio 2003 论文用的是 tanh，这里是跟原论文一致

实际上在后续 Part 中会看到，tanh 在这里效果不错，但隐藏层太大会导致 "dead neurons" 问题。
</details>

---

## 🧭 下一步

架构搭好了，接下来训练它！

👉 [03 — 训练与评估](03_training_and_eval.md)
