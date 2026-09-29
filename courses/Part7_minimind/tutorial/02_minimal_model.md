# 02 — 基线诞生：用最"古董"的配置，训起你的 minimind（v1）

> 🧭 这是"动手篇"的第一章。我们不直接写 RoPE/GQA/SwiGLU——**先故意不用它们**。本章用最接近
> Part 6 的"古董配置"（MHA + 可学习位置表 + ReLU FFN + LayerNorm）搭出最小可用的 minimind，
> 并当场训练 60 步，看 loss 从「均匀猜测」往下走。**先跑通，再变强**：后面每一章都是针对这个
> 基线的一次"发现问题 → 升级零件 → 实验证明"。

## 🎒 前置回忆包（不翻旧章也能读）

- **自注意力**：每个 token 发射 q/k/v 三个向量，`输出 = softmax(q·kᵀ/√d)·v`——按相关度加权汇聚。
- **多头（MHA）**：把 d_model 切成 8 份各做一遍注意力再拼回——8 个"视角"。
- **可学习位置编码**：额外学一张 `max_len × d` 的位置表，加到 token embedding 上。
- **FFN**：两层线性夹 ReLU，中间维度 `4×hidden`，对每个 token 独立变换。
- **pre-norm 残差块**：`x = x + 子层(归一化(x))`——梯度有"高速公路"。

## 📍 你现在的位置

```text
v0（空文件） ──本章──▶ v1：能建、能训、loss 会降的最小 minimind（28.98M）
                        ❌ 没有实验仪器（03 章）、fp32 慢（04 章）、零件全古董（05-07 章）
```

## 第 1 件：字典——BPE 与特殊 token（官方同款，直接用）

字符级的致命账：**序列长度 = 字符数**。BPE 反复合并语料里最高频的字节对，高频组合变短、
罕见词拆字节兜底——**任何文本都能编（无 OOV），序列大幅变短**。官方用 HF tokenizers 训了
一个 6400 词表的 ByteLevel BPE（`trainer/train_tokenizer.py`，值得抄的事实）：

| 设计选择 | 为什么 |
|---|---|
| **ByteLevel**（先转字节再合并） | 字节只有 256 种，天然无 OOV——emoji、生僻字都能编 |
| **vocab_size=6400** | 词表是参数大户：6400×512≈3.3M，占 26M 的 ~13%。tiny 模型必须小词表 |
| **在 SFT 对话语料上训** | 压缩率要面向部署分布优化——聊天高频说法应该编码得更短 |
| **官方不建议重训** | 词表一换，所有 token id 与已训权重全部作废。"词典"是公共资产 |

36 个保留槽位里你只需记三个：**pad=`<|endoftext|>`(0)、bos=`<|im_start|>`(1)、eos=`<|im_end|>`(2)**。
🔑 小巧思：bos/eos 复用 im_start/im_end——预训练样本被包成 `<|im_start|> 文本 <|im_end|>`，
模型在阶段一就见过"文本在 im_end 终止"，09 章 SFT 的"说完闭嘴"不是从零学的。

## 第 2 件：embedding 与权重绑定（tie）

```python
self.embed_tokens = nn.Embedding(vocab, hidden)      # 输入查表: (V, D)
self.lm_head = nn.Linear(hidden, vocab, bias=False)  # 输出打分: (D, V)
self.lm_head.weight = self.embed_tokens.weight       # 两张互为转置的表共用同一份参数
# （26M 全线 tie；Llama 8B 起大模型 untie——参数预算越紧越划算，见下方"为什么绑"）
```

- **为什么绑**：26M 模型省下 3.3M 参数（**12.6%**），输入输出共享几何还有微弱正则效果。
- **为什么大模型反而不绑**（Llama 8B+ untied）：hidden 大、embedding 只占百分之几，省不了
  多少；共享的约束反而碍事。**tie 是"参数预算越紧越划算"的选择**。
- 🏭 官方用 `_tied_weights_keys` 把绑定关系登记进 HF 生态，`from_pretrained` 自动维护。

## 第 3 件：基线模型——完整代码（此后每章只改开关）

**叠加生长**：从本章起每章一个脚本，**第 N 章的脚本包含第 1..N 章的全部代码**——你读
`02_baseline.py` 时看不到任何 RoPE/GQA/SwiGLU 的影子，它们还没被发明。本章的基线就是全部
现状（模型类与 `scripts/02_baseline.py` 逐行一致）：

```python
class MiniMindForCausalLM(nn.Module):
    def __init__(self, vocab, hidden, n_layers, n_heads, max_pos):
        super().__init__()
        self.max_pos = max_pos
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(max_pos, hidden)          # learned PE：一张可训练位置表
        self.layers = nn.ModuleList(Block(hidden, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight        # tie：输入输出共用一张表
        self.apply(self._init)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        T = input_ids.size(1)
        x = x + self.pos_emb(torch.arange(T, device=input_ids.device))   # 位置 i 查第 i 行
        for layer in self.layers:
            x = layer(x)                                      # 8 个 pre-norm 块串行
        logits = self.lm_head(self.norm(x))                   # (B, T, D) → (B, T, V)
        loss = None
        if labels is not None:                                # labels[j]=input_ids[j]：位置 j 的目标
            x_, y_ = logits[:, :-1, :], labels[:, 1:]         # shift 只在这里做一次！
            loss = F.cross_entropy(x_.reshape(-1, x_.size(-1)),
                                   y_.reshape(-1), ignore_index=-100)
        return logits, loss
```

基线零件（`scripts/02_baseline.py` 里的 `Attention`/`FFN`/`Block`）：

```python
class Attention(nn.Module):                     # MHA：8 个 Q 头，各有独立 K/V
    def __init__(self, hidden, n_heads):
        self.n_heads, self.hd = n_heads, hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)   # 06 章起这里只输出 2 组（GQA）
        ...
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)  # (B,T,8,64) 先切头
        k = self.k_proj(x).view(B, T, self.n_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_heads, self.hd)
        # （05 章在这里插入 RoPE 旋转；06 章在更前面插入 q/k RMSNorm）
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))      # (B,8,T,64)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)   # (B,8,T,T)
        scores = scores + torch.triu(torch.full((T, T), float('-inf'),
                                            device=x.device), 1)  # 因果 mask：上三角 -inf
        out = F.softmax(scores.float(), dim=-1).type_as(q) @ v    # float() 保 softmax 数值稳定
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

class Block(nn.Module):                         # pre-norm 残差块——Part 6 的骨架原样保留
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
```

两个现在就要注意的细节：① **shift 只在 forward 里做一次**——数据侧再偏移就是"预测上上个
token"（09 章的头号排雷点）；② causal mask 用 `triu(-inf)`——位置 j 看不到 j 之后，pad 全在
尾部所以真实 token 永远看不到 pad（08 章展开）。

## 第 4 件：参数账本——28.98M，先记账再减肥

`python 02_baseline.py` 实测（RTX 4090）：

```text
基线模型（MHA+learned PE+ReLU+LayerNorm）参数账本:
    embedding    3.277 M  ( 11.3%)
    pos_emb      0.524 M  (  1.8%)     ← RoPE 上岗后这 0.52M 白拿
    attention    8.389 M  ( 28.9%)     ← GQA 上岗后砍到 5.24M
    ffn         16.777 M  ( 57.9%)
    norm         0.017 M  (  0.1%)
总参数: 28.98 M（对照官方现代版 25.83M——升级之路先记账）
```

🔑 记住这四行账。07 章组装完现代版（25.83M）你会回来对账：**每一处瘦身都能指出是哪个开关省的**。

## ▶️ 运行本章成果：第一次训练

```bash
python 00_download_data.py        # 一次性：官方数据 + 官方 tokenizer
python 02_baseline.py
```

实测输出（RTX 4090，fp32，60 步微训，Part 6 三行循环）：

```text
数据: 8,000 条 × seq=340（截断 1,131 条 = 14%）
step   0/60: loss 8.8748 lr 5.00e-04
step  59/60: loss 7.2265 lr 5.00e-04
📉 初始 loss 应≈ln(6400)≈8.76；用时 3s ｜ 峰值显存 2.5 GB
```

- 🔑 **loss ≈ ln(6400) ≈ 8.76**：随机初始化时模型对每个词的打分接近均匀，交叉熵就是 -ln(1/6400)。
  以后任何模型训起来第一眼看这个数——**离 8.76 很远 = 数据/损失接错了**。
- 60 步就到 7.2：管道是通的。但它现在只是"背高频词"，离会说话还远。

## 🏭 工业界怎么看这个基线

你刚搭的就是 **GPT-2 时代的标准架构**（2019）。现代 LLM 的进化清单——位置 learned→RoPE、
注意力 MHA→GQA、FFN ReLU→SwiGLU→MoE、归一化 LayerNorm→RMSNorm——正是 05-08 章的每一章。
**读懂一只模型的正确姿势是列零件清单对比，而不是重学架构**；而"为什么换零件"，要靠 03 章
先装的"实验仪器"来回答。

## 🪝 引子

1. 词表 6400 → 初始 loss ≈ 8.76 这个锚点，02/07 章的训练曲线都要回来对它。
2. 语料是官方下载的 jsonl——数据怎么清洗/去重出来的？外扩一步就是
   [Part 13 数据工程](../../Part13_data_engineering/tutorial/README.md)。
3. 现在训练"快不快、稳不稳"全凭体感——下一章装仪表：验证集 ppl + 曲线记录。

## 🎯 面试直通车

<details>
<summary>Q: 为什么 tiny 模型词表只有 6400？tie 绑定什么时候划算？</summary>
A: 词表大小决定 embedding/lm_head 参数量（V×D）与 softmax 开销。26M 模型配 6400，embedding
占比 ~13%；抄大模型的 15 万词表，查表部件要吃掉一半参数。tie 在"参数预算紧"时划算（省一份
V×D）；大模型 hidden 大、embedding 占比低，绑定约束碍事，Llama 8B 起 untie。
</details>

## ✅ 本章验收

- [ ] ``python 02_baseline.py` 跑通，loss 从 ≈8.76 往下走；能口述 pad/bos/eos 三个 id 与分工
- [ ] 能背基线四行账（embedding 11.3% / pos 1.8% / attention 28.9% / ffn 57.9%）
- [ ] 能指出 forward 里 shift 发生在哪一行、为什么数据侧不能再偏移

---

[← 课程 README](README.md) | [下一章：实验仪器 →](03_experiment_lab.md)
