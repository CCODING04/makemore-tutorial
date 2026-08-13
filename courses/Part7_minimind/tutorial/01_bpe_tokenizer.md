# 01 — BPE Tokenizer：从 65 个字符到 6400 个子词

> 🔤 现代 LLM 的第一块拼图：用字节对编码（BPE）把莎士比亚从"一字符一整数"升级成"一子词一整数"，顺便预告 chat 格式。

## 📖 前置知识

本章需要你已经掌握：

- **Part 6 01 章**：字符级 tokenizer 的 `encode`/`decode`、`vocab_size`、`block_size`、train/val 划分
- **Part 6 的核心权衡**：词表大小 vs 序列长度

> 💡 如果你忘了"为什么字符级序列很长、词级会 OOV"，先回 Part 6 的 `01_data_and_tokenizer.md`。

## 从 Part 6 结束的地方出发

Part 6 我们用的 tokenizer 是最简单的字符级：把文本里的 **65 个唯一字符**（换行、空格、标点、大小写字母）做成一张词表，一个字符一个整数。

```
"To be, or not to be"  →  encode  →  [1, 58, 33, 46, 43, 56, 43, 58, 1, 45, 43, 58, 58, 33, 46, 43, 56, 43]
                          65 个词表里的整数，一个字符一个
```

这个方案**简单**，但有一个明显代价：**序列很长**。一句话四五十个字符，就是四五十个整数；整个莎士比亚 111 万字符，就是 111 万个 token。模型要一步步"吞"这么多 token，训练慢、上下文也覆盖不了多少真实内容。

这一章我们把 tokenizer 升级成 **BPE（Byte Pair Encoding，字节对编码）**——现代 LLM（GPT、Llama、Qwen）几乎都在用的方案。目标是：**词表变大到 6400，但序列大幅变短**。

## 为什么需要 subword tokenizer？

先看三个候选的"粒度"，它们正好构成一个三难问题：

```
词级（word）             字符级（character）          subword（子词）★
─────────────           ─────────────────          ─────────────────
词表几万~几十万          词表很小（65）               词表几千~几万
序列很短                 序列超长                    序列中等
❌ OOV：没见过的词        ✅ 任何文本都能编            ✅ 任何文本都能编
   直接崩掉               ❌ 一个词要拆成一串字符       ✅ 高频片段被合并，低频拆开
```

- **词级**：词表太大，而且**有未知词（OOV）问题**——"ChatGPT"、"quoth"这类词训练时没见过，就编不了。
- **字符级**：没有 OOV，但"一个词 = 一串字符"，序列太长，模型要花很多步才能"读懂"一个词。
- **subword（子词）**：介于两者之间——**高频的常见片段**（如 `ing`、`the`、`tion`）被合并成独立的 token，**低频的罕见词**则被拆成更小的子词。既没有 OOV，序列又比字符级短得多。

> 🔑 **subword 的核心思想**：常见的组合合并成整体，罕见的词退化成字符组合。**任何词都能被编码**（没有 OOV），**常见的词只占 1~2 个 token**（序列短）。BPE 就是自动做这件事的算法。

## BPE 算法原理：一句话版本

> 从字符集出发，**反复合并出现频率最高的相邻 token 对**，直到词表达到目标大小。

听起来抽象，我们用一个玩具例子走一遍。假设文本只有一句：`low low low low low low low lower lower`（7 个 `low` + 2 个 `lower`），目标词表 10。

```
第 0 步：初始词表 = 全部字符
        ['l', 'o', 'w', 'e', 'r']  （5 个，去掉重复）
        文本：l o w _ l o w _ l o w _ ...（每个字符一个 token）

第 1 步：统计相邻 token 对出现的次数
        ('l','o') 出现 9 次 ← 最多！
        ('o','w') 出现 9 次
        合并 ('l','o') → 新 token 'lo'
        词表：['l','o','w','e','r','lo']（6 个）

第 2 步：重新统计相邻对
        ('lo','w') 出现 9 次 ← 最多！
        合并 ('lo','w') → 新 token 'low'
        词表：['l','o','w','e','r','lo','low']（7 个）

第 3 步：('low','er')? 统计相邻对
        ('low','e') ... ('e','r') ...
        继续合并出现最多的，直到词表达到目标 10 个
```

- 🔑 每一步都问同一个问题：**当前文本里，哪两个相邻 token 一起出现的次数最多？** 把它们合并成一个新 token。重复，直到词表够大。
- 💡 注意 `l`、`o`、`w` 这些单字符**永远留在词表里**（作为"最底层"），所以任何词哪怕从没合并过，也能用字符拼出来——这就是"没有 OOV"的保证。
- ⚠️ 合并是**贪心**的：每一步只合并当下最频繁的对，不管未来。这不能保证"全局最优压缩"，但足够好用，而且实现简单。

### 完整手推一遍：一个更真实的合并过程

上面那个例子只展示到第 3 步，容易让人误以为"合并就几下"。真实的合并往往要**连续几十上百步**。我们用语料 `aaabdaaabac` 推一遍（目标是直观感受"合并表"怎么一步步长出来，不追求跑完整个词表）：

```
语料：  a a a b d a a a b a c

初始词表：['a','b','c','d']
第 1 步：统计相邻对 → ('a','a') 出现 4 次最多 → 合并成 'aa'
         文本：aa a b d aa a b a c        （注意 aaa 合并成 "aa"+"a"）
第 2 步：重新统计 → ('a','b') 出现 2 次最多 → 合并成 'ab'
         文本：aa ab d aa ab a c
第 3 步：重新统计 → ('aa','ab') 出现 2 次最多 → 合并成 'aaab'
         文本：aaab d aaab a c
第 4 步：剩下所有相邻对都只出现 1 次（并列）→ 取最靠前的 ('aaab','d') → 'aaabd'
         文本：aaabd aaab a c
...
```

- 🔑 观察三点：**① 合并顺序完全由统计驱动**（`aaab` 因为反复出现被合并出来）；**② 每一步都在更新相邻对统计**（合并会创造新的相邻对，比如第 2 步之后才出现 `aaab` 这个候选）；**③ 词表从 4 个字符慢慢长到目标大小**。真实 BPE 就是在百万字符上把这个过程重复几千次。
- ⚠️ 两个容易忽略的细节：**① 重叠的处理**——`aaa` 合并时只取前两个 `a` 成 `aa`，剩下一个 `a` 单独留下（合并是"从左到右、不重用"的）；**② 平局的处理**——并列时取最靠前/字典序最小的对，不同实现可能有细微差异，但结果都差不多。
- 💡 这个例子还能看出：BPE **完全不管语义**——`aaab` 在人类眼里是乱码，但在数据里高频出现，就会被合并。合并的唯一标准是**统计频率**，不是词义。

## 用 HuggingFace tokenizers 训练 BPE

自己手写 BPE 完全可以（上面的玩具例子就是原理），但工程上我们直接用 HuggingFace 的 `tokenizers` 库来训练。**注意：我们只"借"训练器，模型还是我们自己的。**

[01_bpe_tokenizer.py](../scripts/01_bpe_tokenizer.py) 的核心代码：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# 1. 指定模型：ByteLevel BPE（GPT-2 同款，从 UTF-8 字节出发，天然无 OOV）
tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 2. 指定训练目标：词表 6400，预留 3 个特殊 token 的"坑"
trainer = BpeTrainer(
    vocab_size=6400,
    special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
)

# 3. 在 data/input.txt 上训练
tokenizer.train(files=[data_path], trainer=trainer)

# 4. 编码 / 解码 / 保存
tokenizer.save(model_path)   # 存成 tokenizer.json
```

- 🔑 三个关键参数：`vocab_size=6400`（目标词表大小）、`special_tokens=[...]`（特殊 token 提前占坑）、`ByteLevel`（从 UTF-8 字节出发，保证任何输入都能编）。
- ⚠️ 特殊 token **必须在训练时就用 `special_tokens` 预留**，否则训练器会把它们当普通文本吃掉，词表里就腾不出它们的固定位置了。后面讲 chat 格式时会看到它们多重要。

### 字节级 BPE 与预分词：为什么"任何词都能编"

`ByteLevel` 两个词拆开理解：

- **Byte（字节）级**：编码时先把每个字符映射成 UTF-8 字节（0~255），BPE 在**字节序列**上做合并。因为字节只有 256 个，**再偏门的字符（emoji、中文、任何语言）都能用字节拼出来**——这是"没有 OOV"的最终保证。GPT-2 系列正是这么做的。
- **预分词（pre-tokenizer）**：在跑 BPE 之前，先按空格把文本粗切成"词块"。这样 `the`、`and` 这类词天然被完整保留，BPE 只需要在词块内部和少量跨词块边界做合并。

```
原文本:   "To be or not to be"
预分词:   ["To", " be", " or", " not", " to", " be"]   ← 按空格切，保留空格前缀
字节化:   每个词块 → UTF-8 字节序列
BPE 合并: 在字节/字符片段上反复合并高频对 → 最终 token 序列
```

- 💡 预分词对英文意义重大：没有它，空格会被当成普通字符混进合并，`the` 可能被拆成 `t`+`he` 甚至更碎。有了它，**整词完整保留**，BPE 只负责把词内和常见词组（如 `ing`、`ion`）也压缩掉。
- ⚠️ 注意 ByteLevel 的 `add_prefix_space`：训练时如果是 `False`，那么"句首的 To"和"句中 the 前面的空格"处理会有细微差别——这类细节会影响 token 分布，但对我们的教程结果影响很小。

### 在莎士比亚上：BPE 合并出了哪些子词？（预期输出）

在 110 万字符的莎士比亚上跑 BPE，词表 6400，最靠前（合并最成功、最频繁）的子词大致是（具体 id 因种子而异）：

```
═══ 高频子词示例（前 12 个，≈） ═══
  Ġthe      Ġand      Ġof      Ġto       Ġa       Ġin
  Ġthat     Ġis       Ġfor     Ġmy       ing      Ġwith
            （Ġ 表示"前面带空格"）
```

- 🔑 看到规律了吗：**最高频的 token 都是带空格的整词**（`Ġthe`、`Ġand`），其次是常见后缀（`ing`、`ed`、`ion`），再往后是更小的片段。这就是"**高频合并成整体、低频退化成碎片**"的直接体现。
- 💡 这也解释了压缩率从哪来：莎士比亚里 `the`、`and` 出现几千次，每次都只占 **1 个 token**（字符级要 3 个）。常见词越多，压缩越狠——英文文本平均能压到 1/3 左右，正是我们看到的 ≈3.5×。

### 运行结果（预期输出）

实跑脚本（训练在 CPU 上大约十几秒），输出大致如下：

```
═══ BPE Tokenizer 训练 ═══
  词表大小: 6400
  特殊 token: <|endoftext|>(0), <|im_start|>(1), <|im_end|>(2)
  训练数据: data/input.txt (1,115,394 字符)

═══ 编码演示 ═══
  encode('To be or not to be') =
    [3876, 509, 573, 4824, 3876, 509]        ← 6 个 token！
  decode(...) = 'To be or not to be'
  往返一致: True

═══ 压缩率 ═══
  字符数: 1,115,394
  token 数: ≈ 320,000          ← 约 3.5 字符/token
  压缩率: ≈ 3.5×
```

- 💡 同一个句子，字符级要 19 个整数，BPE 只要 **6 个**——`To be`、`or`、`not` 这些常见片段都成了独立 token。模型每"看"一个 token 的信息量变大，上下文覆盖的真实内容就多了。
- ⚠️ 不同种子/训练轮次，具体 token id 会不同（比如 `3876` 可能变别的数），但**数量级不变**：6400 词表、约 3.5 倍压缩。

### 怎么检验一个 tokenizer 好不好

训练完别急着用，先跑三道"体检"：

1. **往返一致性**：`decode(encode(s)) == s` 对任意输入都要成立。这是最底线的正确性检查（Part 6 我们讲过，BPE 同样适用）。
2. **覆盖性**：把整个训练集重新编码一遍，确认**没有产生 `unk` token**（ByteLevel 下理论上不可能，但值得确认）。
3. **压缩率**：`字符数 / token 数`。英文通常 3~4；如果只有 1~2，说明词表太小或预分词配置不对；如果高于 4~5，可能词表过大（对 6400 词表而言压缩过头反而说明词表浪费）。

```python
# 脚本里的体检部分
total_chars = len(text)
total_tokens = len(tokenizer.encode(text).ids)
print(f"压缩率: {total_chars / total_tokens:.2f}x")   # 预期 ≈ 3.5
```

- 🔑 这三项对应三个不同层面的问题：**正确性（往返）、健壮性（无 unk）、效率（压缩率）**。以后你用任何 tokenizer，都值得先做这三道体检。

## 与字符级的对比：65 vs 6400

| 维度 | Part 6 字符级 | Part 7 BPE |
|------|:---:|:---:|
| 词表大小 | 65 | **6400** |
| 编码 "To be or not to be" | 19 个整数 | **6 个整数** |
| 压缩率 | 1× | ≈ 3.5× |
| OOV | 无 | 无（ByteLevel 从字节出发） |
| 训练开销 | 0（直接 set） | 需要跑一次 BPE 训练 |
| 模型输入维度 | 65 | 6400 |

- 💡 词表变大，意味着 `nn.Embedding(vocab_size, hidden)` 和最后的 `lm_head`（`hidden → vocab_size`）都会变大——这是模型参数增加的一个来源。但换来的是序列变短、上下文变长，整体收益远大于开销。
- 🔑 **模型代码几乎不用改**：Part 6 的 Transformer 只依赖 `vocab_size` 这个数。把 65 换成 6400，训练代码一行不用动。这正是"tokenizer 与模型解耦"的好处——你可以在不改模型的情况下，随便换 tokenizer。

## 特殊 token 与 chat 格式：预告

词表里除了普通子词，我们还预埋了 3 个特殊 token。它们的 id 是词表**最前面**的几个：

```
<|endoftext|>(id 0)   文本结束 / 填充
<|im_start|>(id 1)    message 开始（im = message）
<|im_end|>(id 2)      message 结束
```

预告一下 Part 7 第 4 章：我们要把模型从"文档补全器"变成"问答助手"，靠的就是**chat 格式**——用特殊 token 把"谁在说话"标记出来：

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

- 🔑 模型看到 `<|im_start|>assistant\n` 就会"知道"：轮到我说话了。这就是 SFT 阶段教给它的格式。现在只需要记住：**特殊 token 是模型"语言的标点符号"，和文本本身一起编码。**

## 对比 minimind 的 6400 词表：小而精

minimind 用的正是 **6400 词表**。对比一下主流模型：

| 模型 | 词表大小 |
|------|:---:|
| GPT-2 / GPT-3 | ~50,000 |
| Llama 2 | 32,000 |
| **minimind** | **6,400** |

- 💡 为什么 minimind 选 6400？因为它是 **~26M 参数的小模型**。embedding 层占用的参数 = `vocab_size × hidden_size`，词表每大一倍，embedding 就翻一倍。对大模型 5 万词表无所谓，对 26M 的小模型，6400 是"小而精"的平衡点——**英文压缩效果足够，参数占用可控**。
- ⚠️ 注意：我们的数据是英文莎士比亚。6400 词表对英文很够用；但如果做中文，字符数量更大，通常需要更大的词表（几万）——这是语言特性决定的，不是算法问题。

## 学完本部分你能...

- ✅ 讲清"字符级 / 词级 / subword"三难问题，说透为什么 subword 是平衡点
- ✅ 用手推一遍 BPE 算法（初始字符集 → 反复合并最高频相邻对 → 达到目标词表）
- ✅ 用 HuggingFace `tokenizers` 训练一个 6400 词表的 BPE，理解 `vocab_size`/`special_tokens`/`ByteLevel`
- ✅ 对比字符级（65）与 BPE（6400）的压缩率（≈3.5×），明白模型代码为什么不用改
- ✅ 说出 `<|im_start|>` / `<|im_end|>` 的作用，读懂 chat 格式

## 课后练习

<details>
<summary>Q1: 为什么特殊 token（如 <|im_start|>）必须在训练 BPE 时就用 special_tokens 预留？</summary>
A: 因为 BPE 训练器是"从统计里长出来"的——它只产生文本里出现过的片段。如果不用 special_tokens 预留，<|im_start|> 这些词要么被当普通文本合并掉、要么根本不在词表里，模型就没有固定的 id 来表示"user 说话开始"。预留之后，训练器会给它们固定的词表位置（通常是词表前几个），之后 encode/decode 才能稳定使用。
</details>

<details>
<summary>Q2: 把 tokenizer 从字符级换成 BPE 后，Part 6 的 Transformer 模型代码需要改哪些地方？block_size 呢？</summary>
A: 模型代码几乎不用改——它只依赖 vocab_size（65 → 6400）这个数字，nn.Embedding 和 lm_head 的维度会自动跟着变。但通常应该**增大 block_size**：BPE 序列更短，同样的上下文长度能覆盖更多真实内容；同时因为词表变大、每个 token 信息量变大，通常也需要把模型做大一点（hidden 增大）来承载。
</details>

<details>
<summary>Q3: 为什么 BPE 没有 OOV 问题？"ChatGPT"这种训练时没见过的词怎么编？</summary>
A: 因为 ByteLevel BPE 从 UTF-8 字节出发，单字节（甚至空字节）永远在词表里。没见过的高频组合就拆成更小的子词，再不行就退化成单个字符/字节——总能编码。代价是"越陌生的词，token 越长"，但永远不会编不出来。这正是 subword 相对词级最大的优势。
</details>

## 📝 课后作业

完成本章后，去 Assignment 7 完成题 1（BPE Tokenizer）和题 2（编码/解码往返一致性）：

👉 [Assignment 7](../../../assignments/assignment_7/)

## 下一步

tokenizer 搞定了：文本变成了更短、信息更密的 6400 子词序列。但模型的骨架还是 Part 6 那套——**LayerNorm + learned 位置编码**。下一步我们开始"换零件"：先换归一化（RMSNorm），再换位置编码（RoPE）。

👉 [02 — 现代组件：RMSNorm 与 RoPE](02_modern_components.md)
