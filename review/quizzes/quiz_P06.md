# 测验 · Part 06（Transformer / GPT：从零构建迷你 ChatGPT）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** GPT 三个字母拆开是什么？为什么说"ChatGPT 底层就是一个语言模型"？

- **答案**：G=Generative（生成式，能续写）、P=Pretrained（先在大量文本上训练）、T=Transformer（底层架构）。ChatGPT 做的事与 Part 1/2 的"预测下一个 token"同构：逐 token 从"下一个 token 的概率分布"里采样续写——所以同一提示两次回答不同（概率系统）。区别只在规模、数据量和 token 粒度。
- **锚点**：教程 01_data_and_tokenizer.md §"课程动机：ChatGPT 是什么？"

**Q2（对比）** 字符级 tokenizer 与 subword（BPE/sentencepiece）的核心权衡是什么？为什么 BPE 没有 OOV？

- **答案**：核心权衡是"词表大小 vs 序列长度"：字符级词表只有 65、任何文本都能编，但一个词拆成一串字符、序列超长（111 万字符 ≈ 111 万 token）；subword 词表几千~几万（GPT-2 的 BPE 约 5 万），高频片段合并成整体、序列大幅变短，但要训练词表。BPE 没有 OOV 是因为单字符/字节永远留在词表底层作"最底层"——再偏门的词也能退化成字符（字节）拼出来，代价是"越陌生的词 token 越长"。
- **锚点**：教程 01_data_and_tokenizer.md §"其它 Tokenizer 对比：词表大小 vs 序列长度"

**Q3（数字）** 报出本课 loss 演进的实测数字：bigram 初始 loss、理论下限、训练后；加残差连接后；GPU 完整超参 scale-up 后。另答：一个 9 字符的 chunk 里藏有几个训练样本？

- **答案**：初始 loss ≈ 4.74；完全均匀分布的理论下限 $-\ln(1/65)=\ln 65 \approx 4.17$（略低于初始值，说明随机初始化带点"错误方向的自信"）；AdamW 训练后 val loss ≈ 2.50；加残差连接是最大一跃，从 ≈2.5 降到 ≈2.23；GPU 完整超参（A100、约 10M 参数）压到 **1.48**（CPU 缩小型 ≈2.80，看趋势别背数字）。9 字符 chunk（block_size=8，取前 8+1 个）按位置偏移含 **8 个**训练样本——这也是训练要覆盖"上下文长度从 1 到 block_size"的原因。
- **锚点**：教程 01_data_and_tokenizer.md §"交叉熵损失：为什么 reshape 成 (B*T, C)"、§"一个 chunk 里藏着多个样本"；03_transformer_block.md §"Scale Up：超参数放大 + 参数统计 + 生成"

**Q4（诊断）** 把单头 self-attention 里的 `* k.shape[-1] ** -0.5` 删掉，训练初期最可能出什么问题？为什么？

- **答案**：q、k 近似 unit gaussian 时，内积 `q @ k^T` 的方差 ≈ head_size；不缩放时 head_size 越大 wei 越尖锐，softmax 会把每行推向 one-hot——初始化时每个 token 几乎只聚合一个 token，注意力失去"广撒网"的扩散起点，梯度流动与学习都受损。除以 $\sqrt{\mathrm{head\_size}}$ 把方差拉回 ≈1，一句话：**缩放是在初始化时保护 softmax 的温和**；训练中网络自己学会聚焦谁。
- **锚点**：教程 02_attention_from_scratch.md §"scaled attention：为什么除以 sqrt(head_size)"、§"笔记 6：scaled attention"

**Q5（对比）** decoder block 和 encoder block 的区别只在哪一行代码？cross-attention 的 Q/K/V 各来自哪里？预训练产出的"文档补全器"要变成问答助手，对齐三步是什么？

- **答案**：区别只在三角遮罩那一行：decoder 保留 `masked_fill(tril == 0, -inf)`（未来不看过去，自回归）；encoder 删除遮罩行、所有 token 全连通（通读整句）。cross-attention 的 Q 来自 decoder 当前序列，K/V 来自 encoder 的输出——"读法语、写英语"的桥梁。对齐三步：①SFT（问答格式数据微调，学会"等一个问题、给一个答案"）→ ②奖励模型（人类对多回答排序，训练打分器）→ ③RLHF/PPO（以奖励模型为评分器做策略梯度强化）。
- **锚点**：教程 04_beyond_transformer.md §"Encoder vs Decoder vs 完整架构"、§"回到 ChatGPT / GPT-3：预训练 vs 微调"

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 讲清 ChatGPT 底层是语言模型、GPT 三字母含义 | Q1 | 含 Transformer 起源（2017 机器翻译论文） |
| 手写字符级 tokenizer，字符级 vs subword（BPE）取舍 | Q2 | 词表大小 vs 序列长度、无 OOV 的原因 |
| 实现 Dataloader：chunk 多样本、随机 offset、batch 并行 | Q3 | 9 字符 chunk 含 8 样本已考；随机 offset 细节见闪卡 |
| attention 数学技巧：tril + softmax 加权聚合 | Q4 | v1/v2/v3 三版等价为 Q4 缩放问题的铺垫，见闪卡 |
| 从零实现单头 self-attention 并展开 6 条笔记 | Q4 | 考笔记 6（缩放）；其余 5 条见闪卡 |
| 组装 Multi-Head + FeedForward + 残差 + LayerNorm（pre-norm） | Q3 | 残差带来 2.50→2.23 最大一跃；LN 对比见闪卡 |
| Dropout 正则化、scale up 后 loss 2.5 → 1.48 | Q3 | Dropout 具体放置位置未单独出题（残差连接前的输出与 softmax 后权重，见闪卡） |
| 区分 encoder / decoder / 完整架构，读懂 nanoGPT | Q5 | nanoGPT 三细节（4D batched MHA/GeLU/参数分组）见闪卡 |
| 讲清预训练（文档补全器）→ 微调（SFT → 奖励模型 → RLHF） | Q5 | 含 mini-GPT（10M/30 万 tokens）vs GPT-3（175B/300B tokens）对比 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| tiny Shakespeare 的规模与词表？ | 1,115,394 个字符、65 个唯一字符（索引 0 是换行符、1 是空格）；train/val 按 90/10 划分 |
| attention 数学技巧 v1/v2/v3 指什么？ | v1 for 循环逐步平均 → v2 下三角全 1 矩阵归一化后矩阵乘 → v3 亲和力矩阵 masked_fill(-inf) + softmax；三者 allclose 等价 |
| K/Q/V 各自的含义？聚合的是哪个？ | key=我有什么、query=我在找什么、value=若你觉得有趣我传达什么；`wei @ v` 聚合的是 value，不是原始 x |
| 6 条 attention 笔记是哪 6 条？ | ①通信机制（有向图）②无空间概念（需位置编码）③batch 间不通信 ④decoder 三角遮罩/encoder 全连通 ⑤self vs cross-attention ⑥scaled 除以 $\sqrt{\mathrm{head\_size}}$ 控方差 |
| 为什么 attention 需要位置编码？ | attention 作用在"集合"上、无空间概念；用第二张 embedding 表给每个位置学一个向量，`tok_emb + pos_emb` 广播相加 |
| 残差连接为什么让深层网络可训练？ | 加法节点把梯度均分给两个分支，残差通路是恒等映射、梯度从 loss 直达输入不衰减——"梯度超高速公路" |
| LayerNorm 与 BatchNorm 的三个关键差异？ | ①按"行"（per-token 特征）而非"列"归一化 ②无 running buffer、无训练/推理两态 ③同样保留 γ/β（PyTorch LayerNorm 默认无偏方差，5 维时 std≈1.1180） |
| pre-norm 的 Block 写法？ | `x = x + self.sa(self.ln1(x))`、`x = x + self.ffwd(self.ln2(x))`——先归一化再进子层（论文原版 post-norm 相反） |
| FeedForward 的结构与"思考"的含义？ | `Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd)`，per-token 独立 MLP——通信（attention）之后的"各自思考" |
| Dropout 放在 Transformer 的哪些位置？ | softmax 之后的注意力权重、残差连接之前的 attention/ffwd 输出；训练随机置零 = 子网络集成，测试全开 |
| `get_batch` 为什么上限是 len(data) - block_size？ | 保证 `i+block_size` 不越界；x 取 `[i, i+block_size)`、y 取 `[i+1, i+block_size+1)`，均为 (B, T) |
| 交叉熵为什么 reshape 成 (B*T, C)？ | `F.cross_entropy` 要求 (样本数, 类别数)；每个 (batch, time) 位置都是一个独立分类样本，类别数 = vocab_size = 65 |
| mini-GPT vs GPT-3 的规模对比？ | 我们 ~10M 参数 / ~30 万 tokens；GPT-3 175B 参数 / 300B tokens——参数差 4 个数量级、数据差 6 个数量级，架构几乎相同 |
| nanoGPT 与我们实现的三处不同？ | ①一个 `c_attn` 线性层一次算出 q/k/v 再 view 成 (B, nh, T, hs) 的 4D batched MHA（数学等价、更快）②GeLU 替代 ReLU（为加载 GPT-2 权重对齐）③2D 权重 decay、1D bias/LN 不 decay |
