# 04 — 训练流水线：Pretrain → SFT → DPO

> 🏭 零件齐了，该训练了。现代 LLM 不是"一次训练到位"，而是走一条流水线：先预训练成"文档补全器"，再 SFT 成"助手"，最后 DPO 让回答更讨喜。本课把三步从头实现。

## 📖 前置知识

- **必须掌握**：**[Part 6 04 章](../../Part6_transformer/tutorial/04_beyond_transformer.md)**（预训练 vs 微调、
  SFT → 奖励模型 → RLHF/PPO 完整概念）
- **建议掌握**：**[Part 6 01 章](../../Part6_transformer/tutorial/01_data_and_tokenizer.md)**（训练循环、AdamW、
  交叉熵）；**[本部分 01 章](01_bpe_tokenizer.md)**（BPE tokenizer、chat 格式
  `<|im_start|>`/`<|im_end|>`）
- **可选**：[本部分 02 章](02_modern_components.md)/[03 章](03_gqa_and_ffn.md)
  （组件细节——本章按黑盒引用，卡住时回查）

> 💡 这一章是"理论 + 流程"章，代码重点是 SFT 的 **loss masking** 和 DPO 的**参考模型冻结**，这是两个最容易出错的环节。

## 从 Part 6 结束的地方出发

Part 6 最后一章（04）我们画过 ChatGPT 的完整对齐流程：

```
预训练（文档补全器）→ SFT → 奖励模型 → RLHF/PPO → 问答助手
```

当时只是"看个图景"。这一章我们把最核心的三步真正写出来：**Pretrain → SFT → DPO**。DPO 是 2023 年提出的方案，它**用简单的分类损失替代了复杂的 RLHF/PPO**——这正是我们现在要学的。

## 完整流水线总览

```
        data/input.txt (莎士比亚，110 万字符)
                    │
                    ▼
             ① train_tokenizer.py
                    训练 BPE，6400 词表（第 1 章）
                    │
                    ▼
          ┌─────────────────────────────────┐
          │ ② 预训练 (pretrain)             │
          │  目标：预测下一个 token          │
          │  数据：纯文本（未分角色）        │
          │  产出：能"续写"的模型           │
          └────────────────┬────────────────┘
                           │
                           ▼
          ┌─────────────────────────────────┐
          │ ③ SFT（有监督微调）             │
          │  目标：只对 assistant 回答算 loss│
          │  数据：问答对（chat 格式）      │
          │  产出：会"回答问题"的助手       │
          └────────────────┬────────────────┘
                           │
                           ▼
          ┌─────────────────────────────────┐
          │ ④ DPO（直接偏好优化）           │
          │  目标：让回答更"讨喜"           │
          │  数据：(好回答, 坏回答) 偏好对  │
          │  产出：对齐人类偏好的模型       │
          └────────────────┬────────────────┘
                           │
                           ▼
                    部署 / 生成（KV Cache）
```

- 🔑 每一步的**数据不同、损失不同、目标不同**，但**模型架构从头到尾是同一个**（上三章的零件）。这正是现代 LLM 的范式：**一个骨架，多阶段训练。**

## ② 预训练（Pretrain）：文档补全器

### 目标：预测下一个 token

预训练的目标函数和 Part 6 **一模一样**：给定前 `t` 个 token，预测第 `t+1` 个，用交叉熵衡量。唯一区别是 tokenizer 从字符级换成了 BPE（第 1 章）。

```
输入  [<|im_start|> ... 一段莎士比亚 ...]
               │
           预测下一个 token
```

- 💡 预训练数据是**裸文本**：我们直接把 110 万字符的莎士比亚编码成 BPE token 序列去训练，不分 user/assistant。模型在这里学的是"语言的统计规律"——它会续写，但**不会回答问题**（你问它问题，它可能回你更多问题）。
- ⚠️ 预训练产出的模型叫 **base model（基座模型）**，行为不可控。minimind 里这一步叫 `train_pretrain.py`，产出 `pretrain_hidden.pth`。

### 训练技巧：从 Part 6 的"三行循环"到现代套路

Part 6 的训练循环是"零钱"：

```python
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

现代 LLM 的训练循环加了四个"工程件"，[06_pretrain_pipeline.py](../scripts/06_pretrain_pipeline.py)：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

for step in range(max_steps):
    loss = model.compute_loss()          # ① 前向 + loss
    loss = loss / accum_steps            # ② gradient accumulation
    loss.backward()
    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ③ clipping
        optimizer.step(); optimizer.zero_grad()
        # ④ cosine 学习率调度（每个 step 后更新 lr）
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * step / max_steps))
        for g in optimizer.param_groups: g['lr'] = lr
```

1. **混合精度（mixed precision）**：`torch.autocast`（即 `torch.cuda.amp`）下，矩阵乘法等计算自动降到 bf16/fp16，梯度用 fp32 累加——**显存减半、速度翻倍**。bf16 比 fp16 动态范围更大（不容易溢出），是现代 GPU 的默认选择。CPU 上 bf16 指令支持有限，不一定加速。教程脚本以 fp32 为主（保证 CPU 兼容），脚本 [06_pretrain_pipeline.py](../scripts/06_pretrain_pipeline.py) 的 GPU 路径会自动开启 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`。
2. **梯度累积（gradient accumulation）**：显存放不下大 batch？拆成小 batch 跑 `accum_steps` 次，梯度累加后再 `step` 一次——**等效于大 batch**。
3. **梯度裁剪（gradient clipping）**：`clip_grad_norm_(1.0)` 把梯度范数限制在 1 内，防"梯度爆炸"（语言模型的常见病）。
4. **Cosine 学习率调度 + AdamW**：学习率从 `max_lr` 按余弦曲线衰减到 `min_lr`（通常 `min_lr ≈ 0.1·max_lr`），前期大步快走、后期小步精调。

- 🔑 对比 Part 6：**损失函数、模型结构没变，变的是"怎么更新参数"更稳更快**。预训练的核心思想还是那句——**预测下一个 token，压缩语言的结构。**
- 💡 预期输出（CPU 缩小版，BPE、~26M、跑 5000 步左右）：val loss 从初始 ≈ **8.8**（均匀分布的熵 `ln(6400) ≈ 8.76`，随机初始化略高于它）一路降到 **≈ 2.0**，对应 ppl ≈ **7~12**（`e^2.0 ≈ 7.4`）。⚠️ BPE 的 loss 数字和 Part 6 字符级的 2.23 **不可直接比**——词表大了 100 倍、每个 token 携带的信息更多，初始 loss 自然高得多；真正可比的是**下降趋势**和生成质量。训练完生成出来是"伪莎士比亚"。

## ③ SFT（Supervised Fine-Tuning）：把补全器变成助手

### 为什么需要 SFT

预训练模型是"文档补全器"。要它当"助手"，必须用**问答格式**的数据教它"等一个问题、答一个答案"。这一步就是 **SFT（有监督微调）**——minimind 里叫 `train_full_sft.py`。

- 🔑 微调之所以**样本高效**（几千~几万条数据就能起效），是因为模型已经通过预训练学会了语言，SFT 只是"重新训练格式与行为"：**在预训练的底座上，用很少的高质量问答数据做微调**。

### Chat Template：数据长什么样

SFT 数据长这样（第 1 章预告的 chat 格式）：

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

把这条文本 encode 成 token 序列后，**整个序列一起喂给模型预测下一个 token**。对莎士比亚文本做 SFT，就是把一段段"提问+回答"组装成这种格式（比如"问：讲讲 V 夫人的性格。答：<引用原文>…"）。

**数据怎么构造**（[07_sft_training.py](../scripts/07_sft_training.py)）——其实就是"字符串拼接 + 编码"，外加**记录 assistant 区间**（为后面 masking 做准备）：

```python
def build_chat_sample(question, answer, tokenizer):
    # 1. 拼出 chat 格式的完整文本
    text = (f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>\n")
    # 2. 整体编码成 token 序列
    input_ids = tokenizer.encode(text).ids
    # 3. 记下 assistant 内容在 token 序列里的 [start, end)
    a_start = len(tokenizer.encode(f"<|im_start|>user\n{question}<|im_end|>\n"
                                   f"<|im_start|>assistant\n").ids)
    a_end = len(input_ids) - len(tokenizer.encode("<|im_end|>\n").ids)
    return input_ids, a_start, a_end

# 用莎士比亚原文当"答案"：问一句，答一句（我们人为构造的玩具 SFT 集）
samples = [build_chat_sample(q, a, tokenizer) for q, a in shakespeare_qa_pairs]
```

- 🔑 划重点：**`a_start`/`a_end` 是 token 级的下标**，必须用 tokenizer 编码来算（不能直接数字符）。`a_start` 停在 `<|im_start|>assistant\n` 编码完之后的位置——从这里开始才是模型要学的回答。
- 💡 对莎士比亚做 SFT 的一个简单玩法：问题用"关于某角色的提问"，回答直接引用原文段落。数据量不用大，几百条就能让模型"学会问答的格式"。

### 关键：Loss Masking（只对 assistant 算 loss）

⚠️ 这是 SFT 最容易出错、也最关键的地方。

如果像预训练那样对**整条序列**算 loss，模型就会学会预测"user 的问题"——但我们**根本不在乎模型能不能预测问题**（问题是我们给的），我们只在乎它**能不能把 assistant 的答案续写对**。

做法：把"非 assistant 部分"的标签设成 `-100`，`F.cross_entropy` 的 `ignore_index=-100` 会**自动跳过**它们：

```python
# labels 与 input 对齐，先全设成 -100（不计算 loss），再只对 assistant 区间保留
# 用上面 build_chat_sample 记下的 (a_start, a_end) 填 mask
labels = torch.full_like(input_ids, -100)
for s, (a_start, a_end) in enumerate(assistant_spans):
    # 关键对齐：位置 t 的标签 = 下一个 token input_ids[t+1]
    labels[s, a_start - 1 : a_end - 1] = input_ids[s, a_start : a_end]  # 监督整段回答（含开头第一个词）
    labels[s, a_end - 1] = eos_token_id            # 回答结束处学 <|im_end|>

loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (B*T, vocab)
    labels.view(-1),               # (B*T,)
    ignore_index=-100)             # ← 跳过所有 -100 的位置
```

- ⚠️ 注意这里的**偏移细节**：`labels[t]` 存的是"位置 t 应该预测出的下一个 token"，所以是 `input_ids[t+1]`。`a_start-1`（`<|im_start|>assistant\n` 的最后一个 token）的标签是**回答的第一个词** `input_ids[a_start]`——模型正是从这一刻开始"发言"；接着一路监督到最后一个回答词；`a_end-1` 处再监督 `<|im_end|>`（让模型学会"说完就闭嘴"）。user 问题、`<|im_start|>`、格式 token 这些位置全是 -100，被 `ignore_index` 跳过。
- 🔑 顺带一提：预训练也可以不做 masking（整个序列都监督），但 SFT **必须 masking**——这正是预训练与 SFT 在损失上的本质区别。

- 🔑 **Loss Masking 的核心**：`ignore_index=-100` 让交叉熵**无视** user/格式部分，**只训练 assistant 的生成**。模型学会的是"看到 user 问题 → 生成 assistant 答案"，而不是"背下所有问题"。
- 💡 这呼应了 Part 6 04 章 SFT 的概念：SFT = 用"问题在上、答案在下"的格式微调。现在你知道了**实现它的关键就是 loss masking**。
- ⚠️ 别犯的错：直接对整个序列算 loss、忘记 masking——那样模型会去"预测问题"，浪费训练信号且行为学歪。

## ④ DPO（Direct Preference Optimization）：用偏好直接对齐

### 从 RLHF 出发

Part 6 讲的对齐第三步是 RLHF：**先训练一个奖励模型**（给回答打分），再用 **PPO** 强化学习最大化奖励。这条路效果虽好，但**工程极重**——要训奖励模型、要维护策略/参考/奖励/价值四个网络、要调 PPO 的稳定系数……对小项目不现实。

### DPO 的关键洞察：Bradley-Terry 消掉奖励函数

**DPO（Direct Preference Optimization，2023）** 有一个漂亮的数学洞察：

> 我们真正想要的，是"**让好回答概率高、坏回答概率低**"。如果能写出这个偏好目标的解析解，就能**绕开奖励模型和 PPO**，直接用偏好数据算损失。

它用 **Bradley-Terry 模型**把"哪个回答更受欢迎"建模成排序概率：给定提示 `x`，回答 `y_w`（chosen，更被喜欢）优于 `y_l`（rejected，更不被喜欢）的概率是

```
P(y_w > y_l | x) = σ( r(x, y_w) − r(x, y_l) )
```

其中 `r(x, y)` 是潜在奖励函数，σ 是 sigmoid。DPO 证明：**最优策略的解可以把奖励函数"替换掉"**——用当前模型和参考模型的对数概率比来表示。最终 DPO loss：

```
L_DPO(πθ) = −E[ log σ( β · log( πθ(y_w|x) / πref(y_w|x) )   −   β · log( πθ(y_l|x) / πref(y_l|x) ) ) ]
                        ↑ chosen 相对参考提升的幅度            ↑ rejected 相对参考提升的幅度
```

- 🔑 拆开看：`log(πθ/πref)` 叫**隐式奖励**——"当前模型比参考模型更看好这个回答多少"。DPO 就是**让 chosen 的隐式奖励高、让 rejected 的隐式奖励低**，用一个 `logsigmoid` 把它们塞进同一个分类目标。**不需要奖励模型，不需要 PPO。**
- ⚠️ 其中 `β`（温度/系数）控制"离参考模型多远"，`πref` 是**冻结的参考模型**（通常是 SFT 完的模型）。参考模型**不更新**，只是给 chosen/rejected 各自一个"基准概率"，防止模型在优化偏好时把语言能力"忘了"。

### 代码：DPO 训练

[08_dpo_alignment.py](../scripts/08_dpo_alignment.py) 的核心：

```python
# 参考模型 = SFT 权重复制一份，冻结
ref_model = build_model(cfg); ref_model.load_state_dict(sft_weights)
for p in ref_model.parameters():
    p.requires_grad = False

def compute_dpo_loss(policy_logprobs_w, policy_logprobs_l,
                     ref_logprobs_w,   ref_logprobs_l,   beta=0.1):
    # 隐式奖励：当前模型 - 参考模型
    reward_w = policy_logprobs_w - ref_logprobs_w      # chosen 的 log(πθ/πref)
    reward_l = policy_logprobs_l - ref_logprobs_l      # rejected 的 log(πθ/πref)
    loss = -F.logsigmoid(beta * (reward_w - reward_l)).mean()
    return loss

# 每个 (prompt, chosen, rejected) 样本：
#  policy 模型算 chosen/rejected 的对数概率（可微）
#  ref 模型用 no_grad 算同样的对数概率（冻结）
#  loss = compute_dpo_loss(...); loss.backward(); optimizer.step()
```

- 💡 数据长这样：`{"prompt": "...", "chosen": "好的回答", "rejected": "差的回答"}`。我们可以在莎士比亚 SFT 模型上**人为构造**偏好对（比如"回答引用正确的原文" vs "回答胡编"）来演示 DPO。
- 🔑 训练时 **policy 模型（要训的）** 要算梯度，**ref 模型**在 `torch.no_grad()` 下跑、只提供概率基线。这是 DPO 参数更新的标准形态。
- 预期输出：DPO loss 下降、chosen 的回答在隐式奖励上逐渐高于 rejected（可以打印一个 `acc = (reward_w > reward_l).float().mean()` 当"偏好准确率"，从 ~50% 升到 80%+）。

### 细节：怎么算"一个回答的对数概率"

DPO 的核心原料是 `log πθ(y|x)`——"模型给回答 y 的平均每个 token 的对数概率"。它不是一次前向就能直接拿到的，要把回答的每个 token 的对数概率**取平均**（用均值而非累加，避免长回答被不公平地惩罚）：

```python
def compute_response_logprobs(model, prompt_ids, response_ids):
    """返回模型对 response 的平均对数概率（DPO 里当'隐式奖励'用）"""
    seq = torch.cat([prompt_ids, response_ids])          # prompt + 回答 拼起来
    logits = model(seq.unsqueeze(0)).logits              # (1, T, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    shift = len(prompt_ids)
    # 位置 t 预测 seq[t+1]：预测 response 每个 token 的分布是 log_probs[shift-1 : -1]
    token_logp = log_probs[0, shift-1:-1].gather(
        -1, seq[shift:].unsqueeze(-1)).squeeze(-1)       # (len(response),)
    return token_logp.mean()                             # 平均 = log π(y|x) / len
```

- ⚠️ 两个坑：**① 用 `log_softmax` 而不是在 `softmax` 后取 log**（数值更稳）；**② 只对 response 部分取平均**——`shift = len(prompt_ids)` 保证我们只把"回答"的 token 概率加起来，prompt 部分不算（prompt 是给定条件，不参与奖励）。用**均值**而非累加，是因为不同回答长度不同，累加会让长回答天然有更大的绝对值，均值让长短回答可比。
- 🔑 **SFT 正则防崩坏**：纯 DPO 训练有时会让模型语言能力退化（只顾拉开偏好差距，忘了怎么正常说话）。实际脚本 [08_dpo_alignment.py](../scripts/08_dpo_alignment.py) 会在 DPO loss 上加一项 SFT 正则：`loss = dpo_loss + 0.1 * sft_loss`，让模型在优化偏好的同时保持对 chosen 回答的基本语言建模能力。这是 DPO 训练的常见技巧。
- 💡 chosen/rejected 各自算一份，再让 policy 和 ref 各算一次，就有了 `compute_dpo_loss` 需要的四个数。policy 那份**可微**（走反向传播），ref 那份在 `no_grad` 下**只当基准**。

### 呼应 Part 6 的 RLHF

- 回顾：RLHF = 奖励模型 + PPO（策略梯度），工程复杂。
- 现在：**DPO 用 Bradley-Terry + 参考模型，把"排序偏好"直接变成损失**。省掉了奖励模型和 PPO 的稳定性调参。
- 代价：DPO 是"离线"的（只用固定偏好数据集），探索性不如 PPO；但对我们的小模型，**DPO 是性价比最高的对齐方式**。这也是为什么 minimind 流水线用 DPO 而非 PPO。

## 完整流水线 + 与 minimind 的对应

| 阶段 | 我们（教程脚本） | minimind | 数据 | 产出 |
|------|------|------|------|------|
| Tokenizer | `01_bpe_tokenizer.py` | `train_tokenizer.py` | 莎士比亚 | 6400 词表 |
| 预训练 | `06_pretrain_pipeline.py` | `train_pretrain.py` | 裸文本 | `pretrain.pth`（补全器） |
| SFT | `07_sft_training.py` | `train_full_sft.py` | 问答对 | `full_sft.pth`（助手） |
| DPO | `08_dpo_alignment.py` | `train_dpo.py` | 偏好对 | `dpo.pth`（对齐） |

- 🔑 minimind 官方流水线是 **`train_tokenizer → train_pretrain → train_full_sft → train_dpo`**，和我们教程的路径一一对应。唯一区别：minimind 用大规模开源数据集（中文+英文），我们用莎士比亚做玩具版——**流程同构，规模缩小。**

## 部署/生成：把模型用起来

训练完，部署就是加载权重 + 用 KV Cache 逐步生成（第 3 章的 KV Cache 在这里派上用场）。现代生成还会加几个采样技巧，[05_full_model.py](../scripts/05_full_model.py)：

```python
logits = logits[:, -1, :] / temperature          # 温度：压低/抬高分布
logits = top_k_filter(logits, top_k=50)          # top-k：只留前 50 个候选
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, 1)         # 采样（而不是贪心 argmax）
```

- `temperature`：越低越保守（>1 更随机）
- `top_k` / `top_p`：截掉低概率尾巴，让采样集中在靠谱候选里
- 配合 chat 格式：拼上 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` 作 prompt，让模型续写 assistant 的回答，遇到 `<|im_end|>` 停止。

## 学完本部分你能...

- ✅ 画出并讲清 **Pretrain → SFT → DPO** 完整流水线及每步的目标/数据/损失
- ✅ 说出预训练四个技巧（混合精度、梯度累积、梯度裁剪、cosine LR）各自解决什么
- ✅ 写出 SFT 的 **loss masking**（`ignore_index=-100`），解释"为什么只对 assistant 算 loss"
- ✅ 用 Bradley-Terry 讲清 DPO："偏好排序可以直接变成损失，绕开奖励模型和 PPO"
- ✅ 写出 DPO loss，说出参考模型**冻结**的作用（防"忘掉语言"）
- ✅ 对照 minimind 的 `train_tokenizer → train_pretrain → train_full_sft → train_dpo`

## 课后练习

<details>
<summary>Q1: 为什么 SFT 的 loss 要 masking，只对 assistant 部分算？如果对整个序列算会怎样？</summary>
A: 目标不同——我们只关心模型能不能"根据 user 问题生成 assistant 答案"。user 问题是输入、不是输出，让模型预测它是浪费训练信号，还会让模型去"背诵问题"。masking 用 ignore_index=-100 跳过非 assistant 位置，让交叉熵只监督答案的生成。如果对整个序列算 loss，模型会同时学习预测问题，行为学歪、收敛也变差。
</details>

<details>
<summary>Q2: DPO 为什么能"不需要奖励模型和 PPO"？"参考模型冻结"是干什么的？</summary>
A: DPO 的数学核心是：在"与参考模型保持 KL 距离"的约束下，最优偏好目标可以解析求解，且解里把奖励函数用「当前模型和参考模型的对数概率比」（隐式奖励）替换掉。所以直接用 (chosen, rejected) 偏好数据 + logsigmoid 就能训练，省掉了奖励模型和 PPO。参考模型（通常取 SFT 权重）冻结、不更新，是给隐式奖励提供"基准"——防止模型为了讨好偏好数据而把语言能力（流畅度、事实）丢掉。
</details>

<details>
<summary>Q3: 如果只做预训练不做 SFT/DPO，模型生成会是什么样？三阶段分别补了什么能力？</summary>
A: 只做预训练：模型是"文档补全器"——给任何前缀都能继续写（风格像训练数据），但你问它问题它可能反问你、或续写新闻稿，行为不可控。SFT 补"对话格式"：学会等一个 user 问题、生成一个 assistant 答案，从"续写"变成"问答"。DPO 补"质量对齐"：让回答更讨喜（更贴偏好），从"会答"变成"答得好"。三者分别对应"会语言 → 会对话 → 会好好对话"。
</details>

## 📝 课后作业

完成本章后，去 Assignment 7 完成题 6（DPO loss）和题 7（KV Cache）：

👉 [Assignment 7](../../../assignments/assignment_7/)

## 完结

🎉 恭喜你完成 **Part 7（现代 LLM / Minimind）** 全部四章！

回顾整条路线：Part 1 Bigram → Part 2 MLP → Part 3 BatchNorm → Part 4 反向传播 → Part 5 WaveNet → Part 6 Transformer/GPT → **Part 7 现代 LLM**。现在你已经从一个"预测名字的 2×2 查表"一路走到**从零复现了一个现代 LLM 的全部核心**：

```
6400 BPE 词表  →  RMSNorm  →  RoPE  →  GQA + KV Cache  →  SwiGLU  →  Pretrain → SFT → DPO
```

> 💡 这和 minimind 的关系：教程用莎士比亚 + CPU 缩小版演示**每一步的原理与代码**；minimind 用中文/英文大语料 + GPU 把同一条流水线放大到 ~26M 参数真正能聊天的模型。**你写的代码和 minimind 的 `model_minimind.py` 结构是同一个。**

- 别忘了回到 README 的"演进路线"表格，把每个零件再对照一遍。
- 动手做 [Assignment 7](../../../assignments/assignment_7/)，然后可以去读 minimind 源码，你会发现全都能看懂了。
- **毕业下一步**：跑 [scripts/09_eval_demo.py](../scripts/09_eval_demo.py) 做三阶段验收（Base/SFT/DPO 生成对比 + ppl），
  然后按 [05 — 复现 minimind 毕业指南](05_reproduce_minimind.md) 在真实中文数据上跑官方仓库。

---

[← 上一章：Part 6 Transformer](../../Part6_transformer/tutorial/README.md)
