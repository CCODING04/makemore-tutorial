# 10 — 阶段二 SFT：教会"等问完再答"（v8）

> 🧭 v7 的模型会续写：你问"如何摆脱拖延症"，它给你续写一段更像问题的话。阶段二用官方
> `sft_t2t_mini.jsonl`（90 万行多轮对话）把它变成"等 user 问完、以 assistant 身份回答、
> 说完就停"的助手。本章的主角不是训练循环（和 09 章**一字不差**），而是**数据机器**：一条
> 对话怎么渲染、怎么精确标出"哪些 token 该学"。这章的 mask 是全课程最精妙的代码，值得
> 逐 token 走查一遍。

## 🎒 前置回忆包（不翻旧章也能读）

- **Chat 格式**（02 章特殊 token 的回收）：一条对话渲染成
  `<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>\n`。
  特殊 token 是词表里的原子 id，"找回答在哪"可以变成"找 token 子序列在哪"。
- **-100 的约定**（09 章第 1 件）：交叉熵的 `ignore_index`，标"这个位置不用学"。
- **v7 模型**：会续写任何文本，包括你的问题本身——这正是要修的毛病。

## 📍 你现在的位置

```text
v7 会续写（base）──本章──▶ v8：会按角色对话、会在 <|im_end|> 停（assistant）
```

## 第 1 步：渲染——一条对话的两道"随机"工序

官方数据管线在渲染前后各撒一次骰子：

```python
# 工序 A（渲染前）：20% 概率给对话插一条随机 system prompt
pre_processing_chat(conversations, rng, add_system_ratio=0.2)
# 工序 B（渲染后）：80% 概率删掉空的 <think>\n\n</think>\n\n 标签
post_processing_chat(prompt, rng)
```

| 随机工序 | 为什么不是 0% / 100% |
|---|---|
| 20% 注入 system（10 条中英 prompt 轮换） | 100%：模型把"有 system"当对话前提，部署时不给就失常；0%：给了 system 反而不会用。**训练分布要覆盖推理分布** |
| 80% 删空 `<think>` | 官方模板给每个回答固定加空思考标签（为推理模型留口子）；删删留留让模型两种格式都见过 |

- 💡 随机发生在 `__getitem__` 里——同一行数据第二个 epoch 掷出不同的骰子，**免费的数据增广**。
- ⚠️ 复现要点：`rng = random.Random(42)` 固定种子——同一份数据每次构建出**完全相同**的样本
  序列，这是 03 章"实验可对比"原则在数据侧的落实。

## 第 2 步：打标签——token 子序列扫描（SFT 最精妙的一段）

渲染完是一整条 token 序列，"哪些 token 是 assistant 说的"只能事后找。官方不数轮次，而是在
token 流里**滑窗搜索两个锚点子序列**：

```python
bos_id = tok('<|im_start|>assistant\n', add_special_tokens=False).input_ids  # 角色头的 id 串
eos_id = tok('<|im_end|>\n',            add_special_tokens=False).input_ids  # 回答尾的 id 串

def generate_labels(input_ids, bos_id, eos_id, max_length):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i+len(bos_id)] == bos_id:       # 撞到角色头锚点
            start = i + len(bos_id)                    # 回答从锚点之后开始
            end = start
            while end < len(input_ids):                # 向后找回答尾锚点
                if input_ids[end:end+len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), max_length)):
                labels[j] = input_ids[j]               # [start, eos尾] 全保留
            i = end + len(eos_id)                      # 多轮：继续扫，不 break——
        else:                                          # 每一轮 assistant 都监督
            i += 1
    return labels
```

**逐 token 对齐走查**（SFT 头号 bug 温床是偏移，把下标写死看一遍）。一条极短样本（每格一个 token）：

```text
x:        [im_start][user][\n][在吗][im_end][\n][im_start][assistant][\n][ 好 ][ 的 ][im_end][\n]
下标:         0      1    2    3     4    5      6         7       8   9    10    11    12
labels:       ·      ·    ·    ·     ·    ·      ·         ·       ·   ▓    ▓     ▓    ▓
```

- 扫描：`bos_id` 恰是下标 `[6,7,8]` 三个 id，在 6 处命中 → `start=9`；找到 `[11,12]` 的
  `eos_id` → `labels[9..12]` 保留，其余 -100。模型 forward 内部 shift 后，实际监督对是
  **logits[8]→"好"、logits[9]→"的"、logits[10]→`<|im_end|>`、logits[11]→`\n`**。
- 两个细节：角色头自己的 `\n`（下标 8）不被监督，但它"发出"第一条监督对——这正是"看到
  `<|im_start|>assistant\n` 就开口"的训练信号来源；结尾 `\n` 也在监督里（模板格式的一部分）。
- 🔑 **为什么扫 token 而不是拼段时记录**：官方序列是 Jinja 模板渲染的完整字符串，角色信息已
  "融进" token 流。子序列扫描对模板改版稳健——只要锚点在，模板加 think、加 tool_call 都不
  破坏定位。
- ⚠️ shift 只做一处：`labels[j]=input_ids[j]`（"该位置的目标"），shift 在 02 章模型的 forward
  里。数据侧再偏移一次=双重 shift=模型学"预测上上个 token"——复现头号对齐 bug。

### 教学版对照：make_chat_tokens 与 −100 的"错一位"细节

官方管线是"渲染整串、事后扫描锚点"；**教学版（拼段构造）**反着来——拼接的时候顺手给每段
打"是否属于 assistant"的标记。两种写法都值得会（读开源代码两种都会撞见）：

```python
def make_chat_tokens(enc, user_text, assistant_text):
    """拼接一条 chat 序列，返回 (tokens, is_assistant_mask)。"""
    segs = [
        (f"{IM_START}user\n", False),          # IM_START/IM_END 即 <|im_start|>/<|im_end|>
        (user_text + "\n", False),
        (f"{IM_END}\n", False),
        (f"{IM_START}assistant\n", False),
        (assistant_text + "\n", True),         # 只有 assistant 回答计入 loss
        (IM_END, False),
    ]
    tokens, mask = [], []
    for s, is_asst in segs:
        ids = enc(s)                           # 逐段编码——mask 与 token 严格对齐
        tokens.extend(ids)
        mask.extend([is_asst] * len(ids))
    return tokens, mask
```

- 🔑 mask 是 **token 级**的布尔序列，必须逐段编码来对齐（不能直接数字符数）——
  `<|im_start|>assistant\n` 编码完之后 mask 才变 True，从那里开始才是模型要学的回答。
  上文官方管线的 `(a_start, a_end)` 区间记号与这个 mask 是同一件事。

然后是 **labels 与 logits 错一位**——SFT 头号 bug 温床的另一副面孔。教学版在数据侧手动
构造监督对：

```python
# labels[t] = "位置 t 应预测的下一个 token" = input_ids[t+1]，所以区间整体左移一位
labels = torch.full_like(input_ids, -100)
labels[s, a_start - 1 : a_end - 1] = input_ids[s, a_start : a_end]   # 监督整段回答（含第一个词）
labels[s, a_end - 1] = eos_token_id            # 回答结束处学 <|im_end|>（"说完就闭嘴"）

loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index=-100)
```

- ⚠️ 三个下标别看花：**`a_start−1`**（`<|im_start|>assistant\n` 的最后一个 token）的标签是
  **回答的第一个词** `input_ids[a_start]`——模型正是从这一刻"开口"；监督一路推进到最后
  一个回答词；**`a_end−1`** 处再监督 `<|im_end|>`。user 问题、格式 token 全是 −100，被
  `ignore_index` 自动跳过。
- 📝 **两代写法的 shift 口径（只能选一种）**：教学版把 shift 做在**数据侧**——labels 相对
  input_ids 手动左移一位，模型 forward 直接拿 logits 对 labels 算、**不再**内部 shift；
  官方/本课管线（第 2 步）把 labels 与 input_ids **同位**存放、shift 交给模型 forward。
  两种写法数学等价；**混用即双重 shift**——上面刚警告过的复现头号 bug，就是在这里等着
  把两代写法缝在一起的人。

## 第 3 步：训练——循环没变，两个数变了

| 项 | 09 章 Pretrain | 10 章 SFT | 为什么 |
|---|---|---|---|
| Dataset | 四步流水线 | 渲染+扫描 | 学"回答"不学"提问" |
| lr | 5e-4 | **官方 1e-5**（quick 档 5e-5） | 预训练底子不能搅乱：lr 大了 loss 照降，但语言能力被"格式学习"冲垮（灾难性遗忘：loss 好看、生成变糊）。越靠后的阶段改动越"表面"，lr 递减是通用规律 |
| 其余（循环/裁剪/bf16/日志） | 同 | **同一份代码** | 差异全在数据与超参——这就是现代 LLM"一个骨架、多阶段训练"的含义 |

## 第 4 步：行为验收——mask 生效了吗？问 ppl，也问行为

quick 档实测（6,000 条 × seq=768，300 步，bf16）：

```text
数据: 6,000 条 × seq=768，有效监督占比 ≈71%（回答比问题长）
    ↳ 验证集: loss 6.0931  ppl 442.78
    ↳ 验证集: loss 4.6799  ppl 107.76
    ↳ 验证集: loss 4.2986  ppl 73.60
    ↳ 验证集: loss 4.1961  ppl 66.43
📉 loss 6.2714 → 4.1123 ｜ 峰值显存 8.6 GB
Q: 你是谁？
A: '他能处理例子，需要提供信息的信息。他们可能想知道你的你，或者想看看看。…'
Q: 怎么才能坚持跑步？
A: '我有哪的电影，可能用户可能理解我涉及具体。同时，要确保回答符合系统规则，
    不透露身份信息。我确保回答简洁，不透露身份信息，不使用不过等。…'
```

两个诚实的读数：

- 🔑 **mask 在起作用**：监督占比 ≈71%（回答比问题长），SFT 验证 ppl 443→66——模型在"assistant
  接话"这件事上进步显著。
- 🔑 **内容还不像话，但"腔调"已经学会了**：第二个回答反复出现"系统规则 / 身份信息 / 简洁"——
  这些词全部来自训练集的 system prompt 和回答风格。**SFT 先学会的是格式与姿态，内容质量要靠
  数据量与数据质量**（quick 档只有 6 千条 × 300 步）。官方 full 档（epochs=2、batch 16、
  lr 1e-5、seq 768，2026-09 核对官方 argparse）才是真复现。同一代码上次运行甚至背出了数据
  首行的自我介绍——**小步数 + 小数据下运行间方差很大，别拿单次输出下结论**（03 章的实验素养）。
- 生成停在 `<|im_end|>`：09 章 bos/eos 埋线 + 本章把 `<|im_end|>` 放进监督区间，两章合力。

## 🏭 工业界

SFT 模板多、改版勤，模板管理与 mask 自动校验是一套工程（渲染后可视化检查 ▓ 区间，就是
`--stage 8` 打印的"监督占比"）；参数高效替代方案 LoRA 只训 0.x% 参数即可完成同类任务
（[Part 12](../../Part12_finetune_llamafactory/tutorial/README.md)）；SFT 只是后训练第一站，
完整版图在 [Part 8](../../Part8_post_training/tutorial/README.md)。"LIMA：千条精品胜过百万
杂毛"是工业界 SFT 数据观——你的 quick 档 6 千条 already 显示了格式学习的效率。

## ▶️ 运行本章成果

```bash
python my_minimind.py --stage 8      # 自动加载 stage 7 的权重
```

## 🪝 引子

- 模型现在"会模仿好回答"，但分不清"更好和更坏"——下一章给它看**成对的**好坏回答。

## 🎯 面试直通车

<details>
<summary>Q: SFT 的 loss mask 为什么用"token 子序列扫描"？有什么坑？</summary>
A: chat template 渲染后角色边界只存在于 token 流里的两个 id 子序列（&lt;|im_start|&gt;assistant\n
与 &lt;|im_end|&gt;\n）上，滑窗匹配锚点即可圈出每轮回答（含结尾 im_end），区间外全 -100。坑：
① 锚点是多 token 串，换 tokenizer/改模板会让它永远匹配不上——labels 全 -100、loss"完美收敛"
到 0，最阴险；② 循环要继续扫完全序列，多轮每轮都监督；③ shift 只能在模型 forward 做一次，
数据侧不能再偏移。
</details>

## ✅ 本章验收

- [ ] 能画出一条多轮样本的 ▓/· 标注图，指出"模型学的第一个 token"和"闭嘴 token"
- [ ] 能解释 20% system / 80% 空 think 共同遵守的"训练分布覆盖推理分布"
- [ ] `--stage 8` 跑通：报出监督占比 ≈71% 和 val ppl 终值，说出 lr 降 50 倍的原因

---

[← 上一章：阶段一 Pretrain](09_pretrain.md) | [课程 README](README.md) | [下一章：阶段三 DPO →](11_dpo.md)
