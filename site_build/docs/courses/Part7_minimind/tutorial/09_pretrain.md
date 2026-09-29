# 09 — 阶段一 Pretrain：把训练机器全件上阵（v7）

> 🧭 模型在 v6 已经完整（25.83M 官方口径），但它一无所知。本章开始三阶段训练的第一阶段：
> 用官方 `pretrain_t2t_mini.jsonl`（127 万行中文问答）把随机权重训成"会续写的 base model"。
> 03 章的仪器、04 章的稳定提速件，这一章**全件上阵**——你会第一次看到完整的工业形训练循环：
> 数据四步 → bf16 → 累积 → 裁剪 → 余弦调度 → 每 100 步验证 ppl → 全程曲线入 CSV/tensorboard。

## 🎒 前置回忆包（不翻旧章也能读）

- **训练目标**：交叉熵让"模型对下一个 token 的打分"向正确答案靠拢——预训练的目标函数和
  02 章一字不差（shift 在 forward 里）。
- **工程件**（04 章逐个装过）：bf16 autocast、梯度累积、`clip_grad_norm_(1.0)`、
  无 warmup 余弦调度、CSV/tensorboard 记录、验证 ppl。
- **初始 loss 锚点**：≈ ln(6400) ≈ 8.76——随机初始化的均匀猜测。

## 📍 你现在的位置

```text
v6 完整模型（随机权重）──本章──▶ v7：在 127 万行语料上训过 → 会续写（base model）
```

## 第 1 件：数据机器——PretrainDataset 的四步流水线

Part 6 把整本书拼成一条长流随机切窗口；官方是**一行 jsonl 一个独立样本**，四步加工：

```python
# ① 编码+截断：max_length-2 是给 bos/eos 留的坑
ids = tok(text, add_special_tokens=False, max_length=seq-2, truncation=True).input_ids
# ② 手工包边界：bos/eos 复用 <|im_start|>/<|im_end|>（02 章的巧思）
input_ids = [bos] + ids + [eos]
# ③ 右补 pad 到定长 340
input_ids += [pad] * (seq - len(input_ids))
# ④ labels = 克隆输入，只把 pad 置 -100（bos/eos/正文全部自监督）
labels = [t if t != pad else -100 for t in input_ids]
```

三个"为什么"：

1. **为什么每行独立、不拼长流**？拼流会让样本之间互相"续写"（莎士比亚式伪上下文）；独立
   样本用 bos/eos 隔离，文档边界清晰——工业语料必分文档。
2. **为什么只 mask pad、别的全学**？预训练是自监督：每个真实 token 的"下一个"都是学习目标，
   bos/eos 也要学（eos 教会"文本会结束"）。pad 是填充物，预测它零信息量 → -100 跳过。
3. **为什么不传 attention_mask，pad 不会污染注意力吗**？不会——pad 全在序列**尾部**，因果
   注意力下真实 token 看不到后面的位置；pad 自己的目标又是 -100。两头堵死，官方因此省掉了
   mask 张量。

🔑 **seq=340 的来历**（官方 argparse 原话就一句"训练的最大截断长度（中文1token≈1.5~1.7字符）"）：
注意这是**容量**口径——340 个 token × 1.5~1.7 字 ≈ 500~580 字，说的是"seq=340 最多装下多长的样本"，
**不是**样本的典型长度。本语料的实际分布（2026-09 实测 2 万条）：中位数 ≈230 字（按官方 tokenizer
≈180 token）、p90 ≈480 字——绝大多数样本远低于容量线，长尾才被截。**截断率现在是被测量的**
（03 章的仪器精神）：

```text
数据: 20,000 条 × seq=340（截断 2,849 条 = 14%；bos=1 eos=2 pad=0）
```

14% 截断率是个健康的取舍——压到 0% 要么加长 seq（计算变贵），要么丢长样本（信息丢失）。
顺带对个账：01 章留出集 56,168 字 ÷ 200 条 ≈ 280 字/条，与本处中位数 230、均值 ≈270 是同一份
`pretrain_t2t_mini` 的同一分布——全册单条样本字数以这组实测为准。

## 第 2 件：训练循环——全件上阵的完整形态

```python
for step in range(steps):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accum):                                   # ③ 梯度累积（04 章）
        xb, yb = next_batch()
        with torch.autocast('cuda', dtype=torch.bfloat16):   # ① bf16（04 章）
            _, loss = model(xb, yb)
            loss = loss / accum
        scaler.scale(loss).backward()                        # bf16 下 scaler 直通
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ④ 裁剪
    for g in optimizer.param_groups:
        g['lr'] = get_lr(step, steps, lr)                    # ⑤ 无 warmup 余弦 1.0×→0.1×
    scaler.step(optimizer); scaler.update()
    logger.log(step, train_loss=…, lr=…, tok_s=…)            # ⑥ 曲线入 CSV/tensorboard（03 章）
    if step % 100 == 0:
        vl, vp = evaluate(model, val_data)                   # ⑦ 验证 ppl（03 章）
        logger.log(step, val_loss=vl, val_ppl=vp)
```

调度器三个读数（官方 `get_lr`，无 warmup）：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
# t/T=0 → 1.0×lr；t/T=0.5 → 0.55×；t/T=1 → 0.1×（收在 5e-5，不是 0）
```

- **没有 warmup**：小模型 + 大有效 batch 下这是成立的工程选择；大模型/大 lr 场景工业
  普遍加 0.5~2% 步数的 warmup，甚至改用 WSD（warmup-stable-decay 三段调度）方便中途退火。
- **跨 epoch 不重置**：current_step = epoch×iters + step，余弦横跨全部 epoch。
- 📝 官方 AdamW 用默认 betas (0.9, 0.999)、weight_decay 未暴露（即 0.01）；课程教学脚本用
  (0.9, 0.95) 是 GPT-3 系惯例。几十 M 的模型两套都稳——**复现时口径差不用慌**。

### 📐 四件套的"为什么"层：AMP 安全在哪、累积凭什么等效、裁剪剪什么、余弦为何收 0.1×

循环代码只有十几行，但每一件都值得问一句"凭什么是安全的"：

**① 混合精度（AMP）为什么安全——"计算降精度、更新保精度"。** `torch.autocast` 下矩阵乘
等算子降到 bf16 跑（显存减半、速度翻倍），但**权重主副本和 `.grad` 累加仍是 fp32**——
bf16 只出现在算子的输入输出里，不出现在参数和优化器状态里。bf16 与 fp32 共享 8 位指数
（动态范围一样大）、只牺牲尾数精度——**不会溢出，只会丢一点低位有效数字**，而 SGD 类更新
对这点丢失不敏感。fp16 指数只有 5 位，小梯度会下溢，所以才需要 GradScaler 先放大再缩回
（04 章实验对比过两档）。一行总结：**降精度的是"乘加"，不降的是"累加与更新"**。

**② 梯度累积的算术——凭什么 8×32 等效 256。** 每个 micro-batch 的 loss 先除以 accum 再
backward，`.grad` 里累的是 8 段"平均梯度"的平均；batch 256 一次算的也是 256 个样本 loss
的平均梯度——期望相同、方差同量级，**等效**；显存却只付 batch 32 激活的钱。两个纪律：
裁剪和 `optimizer.step()` 只在累积周期末做一次（对"等效大梯度"裁剪，不是对碎片）；打印
loss 时乘回 accum 还原量级。

**③ 梯度裁剪的直觉——方向保留、长度封顶。** `clip_grad_norm_(1.0)` 做的是
`g ← g / max(1, ‖g‖)`：范数 ≤1 时**一行不动**（不是无脑缩），超了才等比例缩回单位长度——
方向完全保留。它防的是**个别 batch 的尖峰**（一条病句/一个难样本让某层梯度突然 ×100）把
参数一步带飞，是语言模型的常见病保险丝，不是常规操作。🔎 与 06 章"造病"实验的关系：裁剪
治的是**梯度幅度**的尖峰，06 章 QK-Norm 治的是**激活/打分尺度**的漂移——两层保险各管一段。

**④ 余弦为什么收在 0.1×，不是 0。** 前期大步快走（高 lr 跨过平坦区）、后期小步精调
（低 lr 沉进锐谷）。`get_lr` 里那个 `0.1` 常数项让终点是 `0.1×lr` 而非 0：**收在 0 会让
最后一段"几乎不学习"**（lr→0 时参数近乎冻结，白跑）；保留 10% 让退火期仍在有效更新——
官方 quick/full 档收在 5e-5（= 0.1×5e-4）。同一个道理的变体：工业界大模型常改 WSD 调度
（warmup-stable-decay），把"何时退火"变成可以中途插进去的决定。

## ▶️ 运行本章成果

```bash
python my_minimind.py --stage 7        # quick 档：2 万行采样 300 步，4090 约 1 分钟
python my_minimind.py --stage 7 --profile full   # 官方默认超参 + 全量 127 万行
```

quick 档实测（RTX 4090，bf16）：

```text
数据: 20,000 条 × seq=340（截断 2,849 条 = 14%）
训练: steps=300 lr=0.0005 batch=16×accum2 amp=bf16 clip=1.0 (无 warmup 余弦→0.1×)
    ↳ 验证集: loss 8.6031  ppl 5448.66
    ↳ 验证集: loss 6.4838  ppl 654.43
    ↳ 验证集: loss 5.8772  ppl 356.82
    ↳ 验证集: loss 5.6291  ppl 278.41
📉 loss 8.8623 → 5.5683 ｜ 峰值显存 3.2 GB
prompt: '如何才能摆脱拖延症？'
续写  : '如何才能摆脱拖延症？\n你把你一些需要做家的本食谱和面粉，你为您否帮您计算一个分类…'
↳ 它在『续写文本』而不是『回答问题』——base model 的宿命，10 章 SFT 来修
```

- 🔑 step 0 的 loss ≈ 8.86 ≈ ln(6400)：**初始锚点每次都要对**（02 章的锚点）。
- 🔑 **对账（08 章埋的钩子）**：mini 语料 ≈2.6 亿 token（实测 204.8 token/条 × 127 万行，
  官方 tokenizer 口径）× 2 epoch ≈ 5.2 亿，与 Chinchilla 法则"26M × 20 = 5.2 亿"严丝合缝。
  full 档（官方默认 epochs=2、
  batch 32×accum 8、lr 5e-4、seq 340，2026-09 逐项核对官方 argparse）就是把这个量级真跑完。

## 🏭 工业界放大

单卡循环 → FSDP/张量并行（把权重按注意力头切到多卡） + DistributedSampler（[Part 10](../../Part10_distributed/tutorial/README.md)，
04 章你已见过它的双卡首秀）；现算 tokenize → 离线编码成二进制
（[Part 13](../../Part13_data_engineering/tutorial/README.md)）；loss 曲线要配"体检表"和训练中
评测（[Part 8](../../Part8_post_training/tutorial/README.md)）——03 章装的 CSV/tensorboard
就是它的最小版。

## 🪝 引子

- 会续写 ≠ 会对话：它对"如何摆脱拖延症"的回应是**续写更像问题的话**。下一章用 90 万行
  多轮对话教它"等人问完再答"。

## 🎯 面试直通车

<details>
<summary>Q: 梯度累积为什么等效大 batch？loss 为什么要先除 accum？</summary>
A: 累积是把 8 个 micro-batch 的梯度在 .grad 里相加后再 step——平均梯度的期望与 batch 256
相同，方差同量级，等效；显存却只付 batch 32 激活的钱。先除 accum 是因为 backward 累加的是
"各 micro-batch loss 的和"，除以 8 才等价于 256 样本的均值；打印时要乘回来还原量级。裁剪和
step 都只在累积周期末做一次。
</details>

## ✅ 本章验收

- [ ] 四步数据流水线能默写，且能答"为什么不传 attention_mask 也不怕 pad"
- [ ] `--stage 7` 跑通：能报出 val ppl 5448→278，并解释截断率 14% 的取舍
- [ ] `get_lr` 三个读数（1.0/0.55/0.1×）能当场算出
- [ ] 能对账"2 epoch ≈ 5 亿 token ≈ Chinchilla 20×"

---

[← 上一章：组装对账](08_assemble_model.md) | [课程 README](README.md) | [下一章：阶段二 SFT →](10_sft.md)
