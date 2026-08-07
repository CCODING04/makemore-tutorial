# 04 — 超越 Transformer：Encoder/Decoder、nanoGPT、回到 ChatGPT

> 🌍 我们把镜头拉远：我们的 mini-GPT 是完整 Transformer 的哪一半？工业界的 nanoGPT 怎么写？ChatGPT/GPT-3 是怎么从这 200 行代码走向生产的？

## 📖 前置知识

本章需要你已经掌握：

- **01-03 章全部内容**：Tokenizer、Dataloader、Bigram、Self-Attention、Multi-Head、FeedForward、残差、LayerNorm、完整 decoder-only Transformer

> 💡 本章是"全景回顾 + 展望"，不再有新代码实现，重在建立全局理解。

## 从"我们做了什么"出发

前 3 章我们用约 200 行代码训练了一个 **decoder-only Transformer**，能在 tiny Shakespeare 上生成"伪莎士比亚"。现在的问题：**这和 2017 年那篇论文《Attention is All You Need》里的"Transformer"是同一个东西吗？**

答案是：**是，但不完整。** 论文画的是一个 **encoder-decoder（编码器-解码器）**架构，我们只实现了它的**解码器（decoder）**那一半。下面把这幅"官方全家福"补齐。

## Encoder vs Decoder vs 完整架构

### 我们实现的：decoder-only

- 只有 **decoder**：带三角遮罩的自回归 Transformer。
- 没有 encoder、没有 cross-attention。
- 为什么？因为我们的任务是"无条件生成文本"——没有额外的"输入"需要去编码，只需要照着数据集"喋喋不休"。

### 论文里的：encoder-decoder（机器翻译）

原论文做的是**机器翻译**（法语 → 英语），所以需要两套 Transformer：

```
              法语句子 "Je suis très heureux"
                     │
                     ▼
            ┌───────────────────┐
            │      Encoder      │
            │ （无三角遮罩！）   │   所有 token 全连通，互相"读懂"整句话
            └───────────────────┘
                     │
               (编码后的法语表示)
                     │ K / V
                     ▼
   <START> ──► ┌───────────────────┐
   "I am"      │      Decoder      │──► "very happy" <END>
               │ （三角遮罩）       │   Q 来自 decoder，K/V 来自 encoder
               │  + Cross-Attention │
               └───────────────────┘
```

几个关键点：

- **decoder 的特征** = 三角遮罩（自回归）：预测下一个词时不能看未来的答案。这就是"decoder"的定义，我们实现的就是它。
- **encoder 的特征** = 删除遮罩行，所有节点互相通信：它要"通读"整句法语，不用自回归，所以可以让所有 token 互相看个够。
- **cross-attention**：decoder 生成时，Q 仍然来自 decoder 自己（"我想生成什么"），但 **K/V 来自 encoder 的输出**（"我参考的是已编码的法语"）。这就是把"读法语"和"写英语"接起来的桥梁。
- **特殊 token**：<START> 放在生成序列开头（告诉模型"开始翻译了"），<END> 表示生成结束。这些是为任务**新增的专用 token**，不在自然文本词表里。

```
decoder block（我们）：           encoder block：
  未来不看向过去（三角遮罩）        所有节点全连通（删除遮罩）
  自回归、可采样                    通读整句、提取表示

完整架构 = encoder + decoder + cross-attention（翻译等"条件生成"任务）
```

- 💡 为什么 GPT 只用了 decoder？因为 GPT 是"文档补全器"——没有额外条件输入，只需要自回归生成。**decoder-only 恰好就是生成语言模型最需要的形态。**

## nanoGPT 走读

Karpathy 把"生产级但极简"的代码放在了 [nanoGPT](https://github.com/karpathy/nanoGPT)：**两个文件，各约 300 行**。

### model.py vs train.py 的分工

- **train.py**：训练样板（boilerplate）。我们有过的训练循环、AdamW、`get_batch`，它都有，只是**复杂得多**：保存/加载 checkpoint、学习率衰减、`torch.compile` 编译、分布式训练（多节点/多 GPU）。这些和模型本身无关，是"工程"。
- **model.py**：模型定义。和我们写的**几乎一模一样**——position/token embedding、Block、ln_f、lm_head、generate。

### 三个与我们不同的细节

1. **Batched multi-head attention（4D 张量）**

   我们的写法是：多个 `Head` 各自算完再 `cat`。nanoGPT 用**一个** `c_attn = nn.Linear(n_embd, 3 * n_embd)` 一次性算出所有头的 q/k/v，然后 `view + transpose` 成 4D 张量 `(B, nh, T, hs)`——**把"头"也当成一个 batch 维**：

   ```python
   q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
   k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
   # ... 然后在 (B, nh, T, hs) 上做矩阵乘法 ...
   y = y.transpose(1, 2).contiguous().view(B, T, C)
   ```

   - 🔑 数学上和我们**完全等价**，只是把"头"塞进 batch 维、批量并行计算——效率更高，代码更紧凑。nanoGPT 还用了 **Flash Attention**（PyTorch 2.0 的 `scaled_dot_product_attention`，自带因果遮罩），让 GPU "brrrr"。

2. **GeLU 非线性**（替代 ReLU）

   ```python
   self.gelu = nn.GELU()
   ```

   - 💡 为什么不用 ReLU？**为了能加载 OpenAI 官方的 GPT-2 预训练权重**。OpenAI 用了 GeLU，nanoGPT 要能 `from_pretrained('gpt2')` 加载 checkpoint，就必须对齐非线性。这是"工程妥协"的典型例子。

3. **参数分组：weight decay 与不 decay**

   ```python
   # 2D 及以上（矩阵乘法的权重 + embedding）→ 施加 weight decay
   decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   # 1D（bias、LayerNorm）→ 不施加
   nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   ```

   - 🔑 经验法则：**权重矩阵 decay，偏置和归一化不 decay**。这能提升泛化，也是训练大模型的标准做法。
   - 另外还有：`c_proj.weight` 用 `0.02/sqrt(2*n_layer)` 缩放初始化（残差投影特殊初始化）、token embedding 与 lm_head **权重绑定**（weight tying）等。细节更多，但骨架和你写的一模一样。

## 回到 ChatGPT / GPT-3：预训练 vs 微调

nanoGPT 专注的是**预训练（pre-training）**。想得到 ChatGPT，需要**两个阶段**。

### 阶段一：预训练（Pre-training）

这就是我们做的事——只是"婴儿版"：

- 在**一大块互联网文本**上训练一个 decoder-only Transformer，让它学会"补全文档"。
- 我们的对比：模型 ~10M 参数，数据 ~100 万字符（按 OpenAI 的 50K 子词词表折算大约 **30 万 tokens**）。
- GPT-3（2020 论文）：最大模型 **175B 参数**（是我们的约 1 万倍），训练于 **300B tokens**（比我们大约 100 万倍）。

| | 我们的 mini-GPT | GPT-3 最大 |
|---|---|---|
| 参数量 | ~10M | **175B** |
| 训练 tokens | ~300K | **300B** |
| 架构 | decoder-only Transformer | decoder-only Transformer（几乎相同） |

- ⚠️ 规模差了 6~7 个数量级，但**架构几乎一样**——这就是为什么"理解这 200 行代码"是有价值的。
- 💡 预训练产出的**不是助手**，而是**文档补全器**：你问它问题，它可能回你更多问题、或者续写一篇新闻稿。行为完全不可控（"unhinged"）。它还**不可用**。

### 阶段二：微调 / 对齐（Fine-tuning / Alignment）

把"文档补全器"变成"问答助手"，大致三步（OpenAI 官方博客）：

```
文档补全器（预训练结果）
   │
   │ ① Supervised Fine-Tuning (SFT)
   ▼
  用"问题在上、答案在下"的问答格式数据微调
  → 学会"等待问题、给出答案"的格式
   │
   │ ② 奖励模型（Reward Model）
   ▼
  让模型对同一问题生成多个回答，人类标注排序
  → 训练一个独立的"奖励模型"预测哪个回答更受欢迎
   │
   │ ③ RLHF / PPO
   ▼
  用策略梯度（PPO）强化学习优化采样策略
  → 让生成的回答在奖励模型下期望得分最高
   │
   ▼
  问答助手（ChatGPT）
```

- **SFT（监督微调）**：用几千个"问答格式"的标注样例微调，让模型从"补全文档"转向"补全答案"。大规模模型的微调非常**样本高效**，几千条数据就能起作用。
- **奖励模型**：人类对多个回答排序，训练一个模型预测"哪个回答更好"。
- **RLHF/PPO**：奖励模型当"评分器"，用强化学习（策略梯度）调整生成策略，让采样出的回答期望奖励最高。

> 🔑 一句总结：**预训练 = 让模型学会"像文本一样说话"；微调 = 让模型学会"像助手一样回答"**。前者的数据是海量互联网，后者的数据是人工标注的问答偏好，量级差了很远，且大多不公开。

## 总结与展望

这一路我们干了什么：

- 用约 **200 行代码**训练了一个 **decoder-only Transformer（= GPT）**
- 在 tiny Shakespeare 上从 bigram 的 ~2.5 一路降到 2.23（CPU 缩小型），GPU 完整版可达 **1.48**
- 生成的文本看起来像莎士比亚——虽然读起来无意义

**这就是 ChatGPT 的骨架**：预训练阶段与它同构；微调（SFT/奖励模型/RLHF）是加在它上面的"对齐"层。

> 💡 关于"Transformer 之后的路径"：我们的下一步可以是——
> - **规模**：用 GPU 跑完整超参（1.48），或读 nanoGPT 学分布式训练
> - **微调**：学 SFT / LoRA / 奖励模型 / RLHF，把"补全器"变"助手"
> - **新架构**：关注注意力之外的演进（线性注意力、MoE、Mamba 等）
> - 推荐继续读 Karpathy 的 micrograd/minGPT/nanoGPT，它们是同一套思路的不同复杂度

如果还想深入，原视频在结尾建议："go forth and transform"。

## 学完本部分你能...

- ✅ 区分 **decoder-only / encoder / encoder-decoder** 三种 Transformer，画出翻译场景的 cross-attention 数据流
- ✅ 解释特殊 token（<START>/<END>）在条件生成里的作用
- ✅ 读懂 **nanoGPT** 的 model.py：batched multi-head（4D）、GeLU、参数分组、Flash Attention
- ✅ 讲清 **预训练 vs 微调**：文档补全器 →（SFT → 奖励模型 → RLHF/PPO）→ 问答助手
- ✅ 用参数量/tokens 把"我们的 10M / 30 万 tokens"和"GPT-3 的 175B / 300B tokens"对比
- ✅ 说出为什么"理解这 200 行代码"能迁移到理解 ChatGPT

## 课后练习

<details>
<summary>Q1: 为什么 GPT 用 decoder-only，而《Attention is All You Need》原文是 encoder-decoder？</summary>
A: 原文做机器翻译，需要 encoder 通读源语言（法语）、再让 decoder 基于它生成目标语言（英语），所以有 encoder + cross-attention。GPT 是"文档补全器"：没有额外的条件输入，只需要自回归地生成文本——decoder-only（带三角遮罩）恰好就是这种"无条件续写"所需的最小完整形态。任务决定了架构选择。
</details>

<details>
<summary>Q2: nanoGPT 的 batched multi-head 和我们的 `ModuleList([Head(...)]) + cat` 数学上等价吗？为什么 4D 实现更高效？</summary>
A: 完全等价。我们的写法是每个 Head 独立做 `(B, T, hs)` 的注意力的 `wei @ v`，再把结果在通道维 cat 起来；nanoGPT 用一个 Linear 一次算全部 q/k/v，view+transpose 成 `(B, nh, T, hs)` 把"头"当成 batch 维，一次批矩阵乘法并行算所有头。4D 版本把头的并行也纳入矩阵乘法（GPU 擅长的大块运算），且避免了逐个 Head 循环的调度开销，所以更高效。数学结果一模一样。
</details>

<details>
<summary>Q3: 为什么"预训练产出文档补全器，而不是助手"？微调阶段的三步各解决了什么问题？</summary>
A: 预训练的目标函数就是"预测下一个 token"（补全文档），所以模型只会补全——给它问题，它可能续写更多问题或新闻稿，行为不可控。微调三步逐步对齐：①SFT 用"问答格式"数据教它"等一个问题、答一个答案"的格式；②奖励模型把人类的"哪个回答更好"的偏好编码成一个可微的评分器；③RLHF/PPO 用强化学习让生成策略在奖励模型下期望得分最高。三步把"会说话"变成"会好好回答问题"。
</details>

## 完结

🎉 恭喜你完成 Part 6（Transformer / GPT）全部四章！

学完这套教程，去 Assignment 6 动手实践吧：

👉 [Assignment 6](../../../assignments/assignment_6/)

回顾完整路线：Part 1 Bigram → Part 2 MLP → Part 3 BatchNorm → Part 4 反向传播 → Part 5 WaveNet → **Part 6 Transformer/GPT**。现在你已经能从零构建一个语言模型家族了。

> 💡 别忘了回到 README 的"演进路线"表格，对照一下每一步 loss 是怎么降下来的。

---

[← 上一章：Part 5 WaveNet](../../Part5_wavenet/tutorial/README.md)
