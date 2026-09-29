# 01 — BPE Tokenizer：从字符级到 6400 子词，官方词典与你的第一版对照

> 🧭 Part 7 的第一课只干一件事：**给模型配一本词典**。Part 6 用 65 个字符"看"莎士比亚；
> 本章换官方 minimind 路线——**6400 词表的 BPE 子词 + 官方 mini 数据**（BPE 是什么、为什么，正是本章第一节）。我们先体检官方
> tokenizer，再在 mini 语料上**亲手从零训一版同规格的**，两相对照，你会明白"词典是公共
> 资产"：换词典 = 换词表 = 权重全部重训。
>
> 📦 **前置**：`python scripts/00_download_data.py --sample-only --no-full`（约 48MB，一次性）。
> 本章脚本纯 CPU，10-30 秒跑完。

## 🎒 前置回忆包（不翻旧章也能读）

- **token**：模型眼中的"字符"。Part 6 的 token = 1 个英文字符（词表 65）；本章的 token = 一个子词。
- **词表（vocab）**：所有合法 token 的编号表。模型输入输出都是 id，`embedding` 行数 = 词表大小。
- **压缩率**：平均每个 token 代表几个字符。越高 → 同样算力看到越长上下文。

## 为什么需要 subword tokenizer？

先看三个候选的"粒度"，它们正好构成一个三难问题：

```
词级（word）             字符级（character）          subword（子词）★
词表几万~几十万          词表很小（65）               词表几千~几万
序列很短                 序列超长                    序列中等
❌ OOV：没见过的词        ✅ 任何文本都能编            ✅ 任何文本都能编
   直接崩掉               ❌ 一个词要拆成一串字符       ✅ 高频片段被合并，低频拆开
```

- **词级**：词表太大，而且**有未知词（OOV）问题**——"ChatGPT"、生僻词训练时没见过，就编不了。
- **字符级**：没有 OOV，但"一个词 = 一串字符"，序列太长，模型要花很多步才能"读懂"一个词。
- **subword（子词）**：介于两者之间——**高频的常见片段**（如 `的`、`ing`、`，`）被合并成独立的 token，**低频的罕见词**则被拆成更小的子词。既没有 OOV，序列又比字符级短得多。

> 🔑 **subword 的核心思想**：常见的组合合并成整体，罕见的词退化成字符组合。**任何词都能被编码**（没有 OOV），**常见的词只占 1~2 个 token**（序列短）。BPE 就是自动做这件事的算法。

## BPE 算法原理：一句话版本

> 从字符集出发，**反复合并出现频率最高的相邻 token 对**，直到词表达到目标大小。

用一个玩具例子走一遍。假设文本只有一句：`low low low low low low low lower lower`（7 个 `low` + 2 个 `lower`），目标词表 10。

```
第 0 步：初始词表 = 全部字符
        ['l', 'o', 'w', 'e', 'r']  （5 个，去掉重复）
        文本：l o w _ l o w _ l o w _ ...（每个字符一个 token，_ 代表空格）

第 1 步：统计相邻 token 对出现的次数
        ('l','o') 出现 9 次 ← 并列最多（与 ('o','w') 平局，裁决取 ('l','o')）
        ('o','w') 出现 9 次
        合并 ('l','o') → 新 token 'lo'
        词表：['l','o','w','e','r','lo']（6 个）

第 2 步：重新统计相邻对
        ('lo','w') 出现 9 次 ← 最多！
        合并 ('lo','w') → 新 token 'low'
        词表：['l','o','w','e','r','lo','low']（7 个）

第 3 步：继续合并出现最多的，直到词表达到目标 10 个
```

- 🔑 每一步都问同一个问题：**当前文本里，哪两个相邻 token 一起出现的次数最多？** 合并它。重复。
- 💡 单字符**永远留在词表里**（最底层兜底），所以任何词哪怕从没合并过，也能用字符拼出来——这就是"没有 OOV"的保证。
- ⚠️ 合并是**贪心**的：每步只合并当下最频繁的对，不保证全局最优压缩，但足够好用、实现简单。

### 完整手推一遍：一个更真实的合并过程

上面的例子只展示到第 3 步。真实的合并往往要**连续几十上百步**。用语料 `aaabdaaabac` 推一遍：

```
语料：  a a a b d a a a b a c

初始词表：['a','b','c','d']
第 1 步：统计相邻对 → ('a','a') 出现 4 次最多 → 合并成 'aa'
         文本：aa a b d aa a b a c        （注意 aaa 合并成 "aa"+"a"）
第 2 步：重新统计 → ('aa','a') 与 ('a','b') 各出现 2 次（并列最多）→ 平局裁决取 ('a','b') → 合并成 'ab'
         文本：aa ab d aa ab a c
第 3 步：重新统计 → ('aa','ab') 出现 2 次最多 → 合并成 'aaab'
         文本：aaab d aaab a c
第 4 步：剩下所有相邻对都只出现 1 次（并列）→ 取 id 序最小的 ('a','c') → 'ac'
         文本：aaab d aaab ac
...
```

- 🔑 三点观察：**① 合并顺序完全由统计驱动**（`aaab` 因为反复出现被合并出来）；**② 每一步都在更新相邻对统计**（合并会创造新的相邻对，第 2 步之后才出现 `aaab` 这个候选）；**③ 词表从 4 个字符慢慢长到目标大小**。真实 BPE 就是在百万字符上把这个过程重复几千次。
- ⚠️ 两个细节：**重叠的处理**——`aaa` 合并时只取前两个 `a` 成 `aa`，剩下单独留（合并"从左到右、不重用"）；**平局的处理**——训练侧并列时取 **token id 序最小**的对（HuggingFace `BpeTrainer` 的行为，tokenizers 0.22.2 / 0.23.1 双版本实测一致：初始字符按字节值排序进场，开局等价于"字典序最小"；合并出的新 token id 更大，天然排后）。"语料中最靠前出现"是 subword-nmt / Karpathy minbpe 那类 Python 参考实现的扫描序裁决——两套规则并不等价，上面的第 2 步按后者就轮不到 `ab`。编码侧（作业题 1）的规则是"**rank 最小（合并表中最早出现）优先，平局取最左**"——训练与编码是两个方向。
- 📝 这串手推不是示意：用 `train_from_iterator` 在 `aaabdaaabac` 上真跑一遍（tokenizers 0.22.2 / 0.23.1 结果一致），merges 顺序正是 `aa → ab → aaab → ac`——第 2、4 步的平局裁决与上面逐字一致。
- 💡 BPE **完全不管语义**——`aaab` 在人类眼里是乱码，但在数据里高频出现就会被合并。唯一标准是**统计频率**。

## 官方词典体检：6400 词表里都住了谁

官方 minimind 用 HF `tokenizers` 训了一个 **ByteLevel BPE**（先转字节再合并，字节只有 256 种，天然无 OOV——emoji、生僻字都能编）：

| 设计选择 | 为什么 |
|---|---|
| **ByteLevel**（先转 UTF-8 字节再合并） | 字节只有 256 种，天然无 OOV |
| **vocab_size=6400** | 词表是参数大户：6400×512≈3.3M，占 26M 的 ~13%。tiny 模型必须小词表 |
| **在 SFT 对话语料上训** | 压缩率要面向部署分布优化——聊天高频说法编码得更短。实锤：官方 [`trainer/train_tokenizer.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_tokenizer.py) 的 `DATA_PATH` 就是 `sft_t2t_mini.jsonl`（约 90 万行 SFT 语料） |
| **官方不建议重训** | 词表一换，所有 token id 与已训权重全部作废。"词典"是公共资产 |

36 个保留槽位里你只需记三个角色：

```text
pad = <|endoftext|> (id 0)   填充用，不进 loss
bos = <|im_start|>  (id 1)   序列开始 / 对话角色标记
eos = <|im_end|>    (id 2)   序列结束——"说完闭嘴"的开关
```

🔑 小巧思：bos/eos 直接复用 im_start/im_end——预训练样本被包成 `<|im_start|> 文本 <|im_end|>`，模型在阶段一（09 章）就见过"文本在 im_end 终止"，SFT 的"说完闭嘴"不是从零学的。

> ⚠️ **口径提醒（本章实测，与网传老资料对齐一下）**：官方 tokenizer 是上面这个**三 token 布局**
> （`endoftext=0 / im_start=1 / im_end=2`）。你网上可能见过"只有 im_start/im_end 两个"的
> 写法——那是自己从零训、只挂两个特殊 token 的版本（我们下面自己训的就是）。**特殊 token
> 的 id 布局取决于"挂了哪些、挂在哪"，以 `tokenizer.get_vocab()` 实查为准**。这个 id 是
> 10 章 loss masking 的承重墙。

**压缩率实测**（官方版，`00_download_data.py --check` 可复现）：

| 样例 | 字符数 | token 数 | 压缩率 |
|---|---|---|---|
| `人工智能是计算机科学的一个分支，它企图了解智能的实质。` | 27 | 17 | **1.59 字/token** |
| `Large language models are trained on vast amounts of text data.` | 63 | 16 | **3.94 字/token** |

中文一个字约占 1.5~1.7 字节，高频字词合并后能整个进一个 token；英文单词信息密度高、空格结构规整，压缩率更高。对比 Part 6 字符级的 1.0：**同样算力，看到的上下文长 30%~290%**。

## 自己训一版：HuggingFace tokenizers

手写 BPE 完全可以（上面的玩具例子就是原理），工程上直接用 `tokenizers` 库——**只"借"训练器，词典规格我们自己定**。[01_bpe_tokenizer.py](../scripts/01_bpe_tokenizer.py) 的核心：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token=None))            # byte-level 无需 unk
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()   # 必须配套，decode 才能无损还原

trainer = BpeTrainer(
    vocab_size=6400,                                 # 与官方同规格
    special_tokens=["<|im_start|>", "<|im_end|>"],   # 教学版只挂 2 个（官方另挂 think/vision 等 36 个）
    initial_alphabet=ByteLevel.alphabet(),           # 256 个字节全部进初始词表
)
tokenizer.train([corpus_path], trainer=trainer)      # 语料 = mini 预训练 sample（19500 条）
```

- 🔑 三个关键参数：`vocab_size`（目标词表）、`special_tokens`（**训练时就预留坑位**，否则会被当普通文本吃掉）、`initial_alphabet`（256 字节打底，保证无 OOV）。
- ⚠️ **评测要公平**：脚本把语料前 200 条**留出**（heldout）不参与训练——在训练数据上测压缩率是"开卷考试"，测出的是记性不是能力。留出取前 200 条、训练从第 501 条起，中间 300 条弃作**缓冲带**（文件序相邻的样本常是同类主题，隔开防泄漏）——所以训练语料是 19,500 条。

## 运行结果（实测，RTX 4090 机器 CPU 档，2026-09）

```text
═══ 2/3 自训教学版 BPE（vocab=6400，语料 19,500 条，留出 200 条不参与训练）═══
  训练完成: 词表 6,400，耗时 7.1s（语料 14.3 MB → ../temp/p7_bpe_corpus.txt）

  ── 体检 1/4：encode/decode 往返无损 ──
    原文 91 字 → 46 token → 还原 91 字
    ✅ 往返无损（byte-level BPE 天然覆盖所有文本，不会出现 <unk>）

  ── 体检 2/4：特殊 token 是不是一个整体 ──
    chat 文本编码为 17 个 token，其中 <|im_start|> → id 0、<|im_end|> → id 1（各占 1 个，不会被拆散）

  ── 体检 3/4：压缩率 ──
    评测集 56,168 字 → 平均 1.52 字/token（字符级=1.0；越高=同算力看到越长上下文）

  ── 体检 4/4：BPE 学到的常见子词（中英混合语料）──
     1. '，'             次数  1,962
     2. '。'             次数  1,367
     3. '的'             次数    524
     4. '和'             次数    469
     5. '\n'            次数    409
     6. '、'             次数    406
     7. ' '             次数    286
     8. '：'             次数    246
     9. '.'             次数    233
    10. '在'             次数    205
    11. '是'             次数    180
    12. '中'             次数    133
    '的'      ✅ 在词表中
    '我们'     ✅ 在词表中
    '人工'     ✅ 在词表中
    'the'    ❌ 不在（被拆成更小子词——常见词未必整词收录）
    'ing'    ✅ 在词表中
    'tion'   ❌ 不在（被拆成更小子词——常见词未必整词收录）

═══ 3/3 对照：教学版（pretrain sample 训练）vs 官方（SFT 语料训练）═══
                                 教学版        官方
  词表大小                         6,400     6,400
  留出集压缩率(字/token)             1.52      1.33
  特殊 token 数                       2        36
  词表重合度(Jaccard)              39.2%
  ↳ 留出集上教学版更省：拿 pretrain sample 训、在 pretrain 留出集上测——同分布占优；
    官方 tokenizer 在 SFT 对话语料上训（trainer/train_tokenizer.py 的 DATA_PATH 即
    sft_t2t_mini.jsonl），测 pretrain 留出集属跨分布。1.52 vs 1.33 主因是语料分布
    匹配，数据量（2 万 vs 90 万行）是纠缠的次变量。换 tokenizer = 换词表 = 权重全部
    重训，故官方不建议重训 tokenizer，下游（02 章起）一律用官方版。
```

> 📝 口径：体检 3 的"评测集 56,168 字"= 留出 200 条（纯 55,877 字）+ 2 条评测句（27+63 字）+ 201 个换行，折合 **≈280 字/条**——09 章"seq=340 容量 ≈500~580 字 vs 本语料中位数 ≈230 字"与这里是同一分布，全册字数口径以此对账。

三个读数各有讲头：

- **教学版在留出集上反而更省（1.52 vs 1.33）**：教学版拿 pretrain sample 训、在 pretrain 留出集上测——**同分布**占优；官方 tokenizer 是在 **SFT 对话语料**上训的（`sft_t2t_mini` 约 90 万行，见上表实锤），在 pretrain 留出集上测属**跨分布**。这 0.19 的差距主因是**语料分布匹配**，数据量（2 万 vs 90 万行）是被纠缠在一起的次变量——想单独归因，得再补一组"同分布、只变量"的对照。**分布匹配的 tokenizer 确实能更省 token——但跨分布的通用性换不来，这是取舍**。
- **词表重合度只有 39.2%**：同规格、同算法、不同语料，合并出的子词一多半不同。**换 tokenizer = 换词表 = 权重全部重训**——这就是官方"不建议重训 tokenizer"的完整理由：差异可感知、收益为零。
- **压缩率 1.3~1.5 字/token**（中文为主语料）vs Part 6 字符级 1.0：BPE 的收益直接兑现在"同算力更长上下文"。

## 三种工业实现对照：HF tokenizers / tiktoken / sentencepiece

BPE 是算法，下面是不同定位的实现：

| | HF `tokenizers`（本章用） | `tiktoken`（Part 8 用） | `sentencepiece`（Llama 系用） |
|---|---|---|---|
| 能力 | **能训练**新词表 + 能推理 | 只能**使用**已发布的词表（GPT-2/3.5/4） | 能训练（BPE/Unigram）+ 推理 |
| 本课用途 | 自训 6400 教学版 + 加载官方版 | 直接加载 GPT-2 的 50304 词表 | （未使用，认识即可） |
| 词表 | 自己定（6400） | 固定（50257/50304） | 自己定 |
| 特色 | ByteLevel 预分词，任何字符串可编 | Rust 实现、极快的编码 | 把空格变成 `▁`，"词首"信息内建 |

- 🔑 **为什么 Part 7 训、Part 8 用现成的**：本章的重点是"词表从 0 训出来"（亲眼看合并过程与压缩率）；Part 8 走 GPT-2 生命周期，直接沿用它的词表才能对上 50304 的 logits（模型对每个词的打分）形状。
- 💡 同一段文本在不同词表下压缩率不同。**比较两个模型的 ppl 前先比较 tokenizer**（不同词表的 ppl 不可比，详见 Part 8 07 章）。

## 特殊 token 与 chat 格式：预告

官方 tokenizer 的 36 个保留槽位里，除了三角色，还有 `<think>`（推理链标记，09 章数据工序里 80% 概率删掉它）、`<vision>`/`<audio>`（多模态占位，本课不用）等。它们在 **chat 模板**里组成 minimind 的对话格式——10 章 SFT 会逐 token 走查这个模板怎么渲染、哪段进 loss。

## 学完本部分你能...

- ✅ 说清词级/字符级/subword 的三难取舍，解释"为什么选 BPE"
- ✅ 手推 BPE 合并过程（`low/lower` 与 `aaabdaaabac`），说清贪心、重叠、平局三个细节
- ✅ 用 HF `tokenizers` 从零训一个 byte-level BPE，并做四道体检（往返/特殊 token/压缩率/子词）
- ✅ 讲清官方 6400 词表的设计选择与三角色 token（`endoftext=0 / im_start=1 / im_end=2`）
- ✅ 解释"为什么官方不建议重训 tokenizer"（重合度 39.2% 的实测版本）

## 课后练习

<details>
<summary>练习 1（观察）：换语料，词表怎么变</summary>

把 [01_bpe_tokenizer.py](../scripts/01_bpe_tokenizer.py) 里的语料换成 `sft_t2t_mini_sample.jsonl`（需改 `extract_texts` 读 `conversations` 拼文本），重训一次。对比：Top 子词变了吗？压缩率变了吗？为什么 SFT 语料训出的词典更"口语"？
</details>

<details>
<summary>练习 2（思考）：压缩率的极限</summary>

如果 `vocab_size` 从 6400 涨到 128000（GPT-4 级），压缩率会一直涨吗？代价是什么？（提示：embedding 参数 6400×512≈3.3M 占 26M 的 13%，128000×512 呢？tiny 模型还剩多少参数给"脑子"？）
</details>

## 📝 课后作业

本章对应 [Assignment 7](../../../assignments/assignment_7/assignment.md) **题 1**：手写一个玩具 BPE 编码器（给定合并表，把新文本编码成 token 序列）——训练侧手推你已经会了，编码侧的"rank 优先、平局取最左"规则在作业里实现。

## 下一步

词典就位。下一章[02 基线诞生](02_minimal_model.md)：用这本官方词典，搭一个**故意全古董**（MHA + 可学习位置表 + ReLU + LayerNorm）的 28.98M minimind，当场训练 60 步——先跑通，再变强。
