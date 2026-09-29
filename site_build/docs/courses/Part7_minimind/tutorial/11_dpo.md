# 11 — 阶段三 DPO：学会偏爱，也学会发现过拟合（v9·毕业）

> 🧭 v8 会"照着学"（给标准答案就模仿），但分不清"更好的回答"和"更差的回答"。阶段三用官方
> `dpo.jsonl`（1.7 万对 chosen/rejected）教会它**偏好**。这是三章里数学最漂亮、实现最讲究的
> 一章——而且由于 03 章装的仪器，我们这次能亲眼看到一个教科书级的现象：**训练池偏好 acc
> 大涨、heldout 反而回落——DPO 小池过拟合的现场**。跑完 stage 9，毕业。

## 🎒 前置回忆包（不翻旧章也能读）

- **SFT 的局限**：loss=交叉熵"照着标准答案学"。但同一个问题没有唯一标准答案——工业做法是
  **成对地**比较："回答 A 比回答 B 好"。数据长这样：
  `{"chosen": [完整对话], "rejected": [完整对话]}`（user 轮也在里面，稍后靠 mask 抠掉）。
- **sigmoid**：把任意实数压到 (0,1)——本章用它把"偏好概率"变成损失。
- **10 章的扫描**：本章的 mask 是同一段扫描代码的 0/1 版，只统计"assistant 说了什么"。

## 📍 你现在的位置

```text
v8 会模仿好回答 ──本章──▶ v9：给一对好坏回答，能拉开偏好（毕业：你自己的 minimind）
```

## 第 1 步：一个公式，消掉奖励模型

传统 RLHF 要先训一个"打分器"（奖励模型 r(x,y)）再用 PPO 强化学习，工程极重。DPO（2023）的
洞察是：**在"别离参考模型太远"的约束下，最优偏好的解可以解析写出，奖励函数恰好被消掉**——
剩下一行损失：

```text
隐式奖励差 = [log πθ(yw)/πref(yw) − log πθ(yl)/πref(yl)]     （yw=好回答, yl=坏回答）
L = −log sigmoid( β × 隐式奖励差 )
```

读法：**"当前策略比参考模型更看好 yw 多少、更不看好 yl 多少"**，这个差值过 sigmoid 就该是
"偏好 yw 的概率"→ 拉大它。完整推导（Bradley-Terry → 消 r）见下面「📐 数学推导」小节——
那里还有把整条推理链做成 5 步动画的交互演示。这里记住三个角色：**策略 πθ（在训）、参考
πref（冻结的 10 章自己）、β（信任域宽度）**。

- 🔑 为什么必须有个冻结的 ref：没有锚点，模型可以同时抬高所有 token 的概率"作弊"拉开偏好
  差，语言能力随之崩。ref 就是"别飘太远"的那根锚——**ref 不是另一个模型，是 10 章你自己的
  复印件**（`ref.load_state_dict(policy.state_dict())` 后冻结，`requires_grad_(False)`）。

### 📐 数学推导：Bradley-Terry → DPO loss

**第一步：把"偏好"写成概率（Bradley-Terry 模型）。** 给定提示 $x$，回答 $y_w$（chosen，
更被喜欢）优于 $y_l$（rejected）的概率，建模成 sigmoid 上的排序：

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

其中 $r(x,y)$ 是**潜在奖励函数**——正是 RLHF 里要先花一轮训练的那个"打分器"。

**第二步：消掉 r（DPO 的关键一步）。** DPO 证明：在"与参考模型保持 KL 距离"（KL 散度：衡量两个分布差多大，用来约束"别离 ref 太远"）的约束下最大化
偏好目标，**最优解可以解析写出**，且奖励函数恰好被对数概率比替换：

$$r(x, y) \;=\; \beta \log \frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)} \;+\; \beta \log Z(x)$$

（$Z(x)$ 是只依赖提示的配分项（归一化常数，保证概率和为 1），做差时消掉。）代回 Bradley-Terry，两项相减：

$$L_{\mathrm{DPO}} = -\,\mathbb{E}\Bigl[\log \sigma\Bigl(\beta \log \tfrac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \tfrac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Bigr)\Bigr]$$

——就是上面那行损失。**奖励模型和 PPO 同时消失**：偏好排序目标 + 参考模型锚点，直接塌缩
成一个 logsigmoid 分类损失。

- 🔑 拆开看：$\log(\pi_\theta/\pi_{\mathrm{ref}})$ 叫**隐式奖励**——"当前模型比参考模型
  更看好这个回答多少"。DPO 让 chosen 的隐式奖励高、rejected 的低。每个样本只需要 4 个数：
  policy/ref 各给 $y_w$/$y_l$ 算一次序列对数概率（第 3 步的 sum 口径）。
- 📝 **β 的口径**：教学示例/作业用 β=0.1；本课 quick 档与官方都用 β=0.15（quick 放大的
  是 lr，不是 β）。β 是隐式奖励进 sigmoid 前的缩放——"离参考模型多远"的信任域
  宽度，且与 sum/mean 口径联动（第 3 步）。

> 🎛️ **交互演示**：下面把 DPO 的整条推理链做成了 5 步动画——①Bradley-Terry 排序模型
> （sigmoid 曲线上的 chosen/rejected 示例点）→ ②隐式奖励替换 r（两根柱子的对比）→
> ③DPO 损失函数（loss 随差值 Δ 变化的曲线）→ ④β 的作用（三条 β 曲线，正是上面三处
> 口径）→ ⑤全貌总结。支持自动播放/键盘 ←→ 翻页。

```widget
dpo
760
```

## 第 2 步：数据机器——在数据侧就切好 x/y

```python
for conv in (sample['chosen'], sample['rejected']):
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    ids = tok(text, truncation=True, max_length=seq, padding='max_length').input_ids   # 定长 1024
    mask = generate_labels(ids, bos_id, eos_id, seq)     # 同一段扫描的 0/1 版（只圈 assistant）
    pair.append((ids[:-1], ids[1:], mask[1:]))           # 预切分：trainer 不走模型内置 loss
```

- 为什么预切分：DPO 的 trainer 要的是"每个 token 的 log 概率"自己去拼损失，直接拿对齐好的
  (x, y, mask) 三根张量喂模型再 gather（下面的 `logits_to_log_probs`）。

## 第 3 步：三处实现差异——官方与"教科书 DPO"不一样的地方

官方 `train_dpo.py` 与多数教科书实现差在三处，**每一处都影响调参**：

1. **序列 logp 用 `sum` 不是 `mean`**：`(logp·mask).sum(dim=1)`——隐式奖励正比于回答长度
   （几百 token 的回答，logp 量级到千）。mean 则长度无关。**口径不同，β 和 lr 不能混抄**：
   把 mean 口径教科书实现常用的 1e-4 量级 lr 直接抄给官方 sum 口径会瞬间崩（sigmoid 全
   饱和），反之官方的 4e-8 在 mean 口径上什么也不会发生。**调 DPO 先问"sum 还是 mean"**。
2. **chosen/rejected 拼进同一个 batch**：`x = cat([x_chosen, x_rejected])`——bs=4 对前向 8 条
   序列（policy + ref 各一轮）。好处：四个量一次拿齐、浮点环境完全对称；代价：batch 维度
   翻倍，显存大头（logits 是 (2B, 1023, 6400)）。
3. **没有 SFT 正则**：不少实现加 `+0.1×SFT loss` 防崩坏；官方靠**极小 lr + 1 epoch + β 信任域**
   三重托底，一行正则都不加。

还有超参的配对逻辑：官方 lr=4e-8（argparse 原话"建议≤5e-8 避免遗忘"）配 β=0.15、只训 1
epoch——偏好是"挪一挪分布"，多训必退化。quick 档把 lr 放大到 1e-6（官方 4e-8 的 **25×**）让几百步内可见效果；真复现 `--profile full`（epochs=1、batch 4、lr 4e-8、β 0.15、seq 1024，2026-09 核对）。

## 第 4 步：监控——仪器最好的舞台：一升一降，看见过拟合

只看 loss 你会以为一切正常。03 章的仪器精神在这里开花：quick 档**故意只在训练池 200 对上
训练**（小池才容易几百步内看到过拟合现场；full 档用全量 1.7 万对），另划 200 对 heldout
（从没参与训练）当泛化考官，两组各在训练前后测一次。⚠️ 训练前 policy=ref，隐式奖励差
恒为 0 无从测起，所以基线报的是**绝对序 acc**（模型自己按 logp 排序的命中率）——与训练后
的**隐式偏好 acc**不是同一指标，下文箭头读作"从 SFT 起点到 DPO 终点"：

```text
训练前（policy=ref）:
    训练池 200 对: policy=ref，绝对序 acc 56.5%
    heldout 200 对: policy=ref，绝对序 acc 62.0%
训练后（quick 档 200 步，lr 1e-6）:
    训练池 200 对（见过的偏好）: 隐式偏好 acc 56.5%→80.5%（隐式奖励 margin +9529.8）
    heldout 200 对（没见过的偏好）: 隐式偏好 acc 62.0%→45.0%（隐式奖励 margin -834.7）
```

🔑 **一升一降是全课程最贵的一张图**：DPO 对见过的偏好拟合极快（80.5%），但泛化反向恶化
（62.0%→45.0%，比随机猜还差）——偏好被"挤"坏了，这就是**小池过拟合的现场**，不是假设。
官方"全量 1.7 万对 + lr 4e-8 + 1 epoch"每一条都在防它：数据够多、步长极小、只过一遍。
这正是 09-11 章 lr 递减（5e-4 → 1e-5 → 4e-8）的终点：**越靠后的阶段，改动越"表面"，越要
保住底座**。

- 🔑 margin 的定义：200 对隐式奖励差的**平均值**（sum-logp 口径、未乘 β），量级上千正是
  长回答 sum 累加的结果。
- 🔑 为什么用"隐式"acc 而不是直接比 logp：直接比的是绝对序（模型自己的偏好），其中混着
  SFT 时就会的偏好；减去 ref 的同题打分（`Δpolicy − Δref`）才是 **DPO 这 200 步教出来的**
  偏好——控制变量的思想又一次登场（03 章）。

## ▶️ 运行本章成果 + 毕业

```bash
python my_minimind.py --stage 9      # 自动加载 stage 8 的权重
```

```text
训练池 200 对（见过的偏好）: 隐式偏好 acc 56.5%→80.5%
heldout 200 对（没见过的偏好）: 隐式偏好 acc 62.0%→45.0%
✅ 权重 → dpo_512.pth
═══ 毕业验收 ═══
Q: 你是谁？   A(DPO 后): '\n\n我作为MiniM创建的背景，'
你自己的 minimind 现在会什么：
  v1 基线能训 → v2 有仪器 → v3 稳又快 → v4 注意力现代 → v5 FFN 现代
  → v6 完整架构 25.83M → v7 会续写 → v8 会对话 → v9 偏好更讨喜 🎉
```

- ⚠️ 诚实读数：quick 档的 DPO 为了"看得见效果"把 lr 放大 25 倍（4e-8→1e-6），代价是语言质量回退（margin
  +9529 意味着 sigmoid 已经饱和）——**毕业 demo 的答案比 v8 更糊是快速档的已知取舍**，官方
  full 档（4e-8）下不会这样。管线是真的，放大即复现。

## 🪝 毕业引子：你的 minimind 从这里出发（路线图）

| 想做的事 | 去哪 | 用到本章的什么 |
|---|---|---|
| 系统学完整后训练（奖励模型 RM、PPO、GRPO（组相对策略优化）与评估） | [Part 8](../../Part8_post_training/tutorial/README.md) | DPO 是其中的第三站 |
| 在线对齐/训练框架 | [Part 11 verl](../../Part11_alignment_verl/tutorial/README.md) / [Part 17](../../Part17_agentic_rl/tutorial/README.md) | 本章的 ref/β/口径问题在 verl 里全是配置键 |
| 参数高效微调 | [Part 12 LoRA](../../Part12_finetune_llamafactory/tutorial/README.md) | 只训增量，ref 省一份显存 |
| 把它部署成服务 | [Part 14 vLLM](../../Part14_inference_vllm/tutorial/README.md) | 06 章的 KV Cache、08 章的 config |
| 注意力算得更快 | [Part 9 CUDA](../../Part9_cuda_kernels/tutorial/README.md) | 05 章的 FlashAttention 伏笔 |
| 训练上多卡/数据怎么来 | [Part 10](../../Part10_distributed/tutorial/README.md) / [Part 13](../../Part13_data_engineering/tutorial/README.md) | 04 章的 DDP/08 章的数据源头 |

## 🎯 面试直通车

<details>
<summary>Q: DPO 的序列 logp 用 sum 还是 mean？为什么官方选 sum？</summary>
A: sum 得到的隐式奖励正比于回答长度（100 token 的回答尺度约是 10 token 的 10 倍），mean 长度
无关。官方用 sum（(logp·mask).sum(dim=1)），β=0.15、lr=4e-8 是配套实测值；换 mean 口径这两个
数要放大。语义上 sum 比较"整段回答被采出的总概率"，更贴近真实偏好；代价是 β/lr 的可选区间
变窄（数值大 → sigmoid 易饱和），且要防长度膨胀。跨实现抄超参前必须先对齐口径。
</details>

## ✅ 本章验收（= v0→v9 全程验收）

- [ ] 不看书写出 DPO loss（隐式奖励差 → ×β → −logsigmoid），并说出 ref 为什么必须冻结
- [ ] 三处实现差异（sum/拼批/无正则）各配一句"后果"
- [ ] `--stage 9` 跑通：报出"训练池 56.5%→80.5%、heldout 62.0%→45.0%"并解释这一升一降
- [ ] 完成毕业路线图任选一站的"外扩阅读"，说清它接住了本章哪个钩子

---

[← 上一章：阶段二 SFT](10_sft.md) | [课程 README](README.md) | [🎓 回 README 看全课程路线](README.md)
