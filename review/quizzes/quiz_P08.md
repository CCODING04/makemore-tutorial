# 测验 · Part 08（后训练全流程：SFT → 奖励模型 → DPO/PPO/GRPO → 推理服务 → 评估）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（诊断）** SFT 训练后，prompt 区域 loss 很低、response 区域 loss 很高，生成时模型倾向重复输入而不回答。最可能的原因是什么？用 `sft_loss` 的四步说明修法，并解释为什么修好后 loss 通常反而更大。

- **答案**：没做 **Prompt Masking**——对全部 token 算 CE，模型把容量花在"预测/复制 prompt"上，养成复述指令的惯性。修法四步：①shift（位置 t 的 logits 预测 t+1，mask 同步移位）②`reduction="none"` 逐 token CE ③乘 loss_mask（prompt 位 0、response 位 1）④只除以 response token 数（`mask.sum().clamp(min=1.0)` 防除零）。修好后 loss 更大是正常的：只看"难的部分"（回答），不再被"容易的部分"（复制 prompt）拉低均值。
- **锚点**：教程 02_sft_and_chat.md §"Prompt Masking：核心技巧"、§"SFT vs 标准 CE 对比"

**Q2（对比）** DPO、ORPO、KTO 三种对齐算法在"参考模型"和"成对数据"上的需求各是什么？DPO 为什么能绕开显式奖励模型？

- **答案**：DPO 需要冻结的参考模型 + 成对 (chosen, rejected)；ORPO **不需要参考模型**（用 odds ratio $\log\frac{p}{1-p}$ 做 chosen/rejected 直接比较，并加 chosen 上的 NLL 项 = SFT + 对齐一步完成）；KTO **不需要成对数据**（前景理论：只要好/坏标签，以所有 log-ratio 均值为 KL 基线做非对称损失，可调大 undesirable_weight 体现损失厌恶）。DPO 的推导：RLHF 目标 $\max E[r] - \beta \cdot KL(\pi \| \pi_{ref})$ 有闭式解，反解出 $r = \beta\log(\pi/\pi_{ref}) + \beta\log Z(x)$，代入 Bradley-Terry 时 $Z(x)$ 在相减中消掉——奖励用策略的 log-prob ratio 表示，不再需要奖励模型。
- **锚点**：教程 03_reward_and_dpo.md §"DPO 推导：从 RLHF 到分类问题"、§"三种算法对比"

**Q3（数字）** 推理服务四组实测/论文数字：LLaMA-7B 级 seq 2048 的 KV 显存在 MHA / GQA / GQA+int8 / GQA+KIVI 各多少？PagedAttention 把 KV 浪费从多少降到多少？连续批处理吞吐提升？投机解码实测 vs 理论产出？这些优化背后的同一个原理是什么？

- **答案**：KV 显存 MHA（kv_heads=32）1.07 GB → GQA（kv_heads=8）0.27 GB → +KV int8 0.13 GB → +KIVI 2bit ≈0.03 GB。PagedAttention（16 token/块，块表映射）把 vLLM 论文实测 **60-80%** 的 KV 浪费降到 **<4%**（课程模拟 41% → 5%）。连续批处理（Orca）同延迟下吞吐 **36.9×**。投机解码 draft γ=4、α≈0.60：实测 ≈2.47 tokens/cycle，理论 $E = (1-\alpha^{\gamma+1})/(1-\alpha) \approx 2.31$。共同原理：decode 是 **memory-bound**——每步把全部权重读一遍，算术强度 ≈1 FLOP/字节，远低于 4090 约 1:150 的平衡点，所以量化/批处理/投机解码/KV 管理都在"让每次权重搬运多干活"。
- **锚点**：教程 06_inference_and_serving.md §"0. 一个原理打全部：decode 是 memory-bound"、§"2. KV Cache 显存"、§"3. PagedAttention 与连续批处理"、§"4. 投机解码"

**Q4（概念）** GRPO 为什么不需要 Value Network？用 prompt `3+5=`、G=4 的例子算出组内优势；PPO 与 GRPO 在优势估计、KL 惩罚、代表模型上各是什么？

- **答案**：PPO 用 Critic（value_head）估 V(s) 再走 GAE；GRPO 对同一 prompt 采 G 个回答，用组内均值/标准差作天然基线：$A_i = (r_i - \mathrm{mean}) / (\mathrm{std} + \epsilon)$。例：奖励 [1, 0, 0, 1] → mean 0.5、std 0.58 → 优势 [+0.87, −0.87, −0.87, +0.87]，正确答案概率增大。对比：优势估计 GAE（γ=1.0、λ=0.95）vs 组归一化；KL 惩罚 per-token KL penalty vs k3 估计器 $\exp(d) - d - 1$（无偏、非负、数值稳定，$d = \log\pi_{ref} - \log\pi$）；两者都用 clipped surrogate（ε=0.2）；代表 InstructGPT/ChatGPT vs DeepSeek-R1；GRPO 常配 RLVR（数学/代码可验证奖励，免训 RM）。
- **锚点**：教程 04_ppo_and_grpo.md §"GRPO：不需要 Value Network"、§"k3 KL 估计器"、§"PPO vs GRPO 对比"

**Q5（诊断）** 从零写 LoRA 时，一位同学把 A 和 B 都初始化为 0，"这样起点更干净"。会发生什么？LoRA 省的是哪部分显存、不省哪部分？课程实测的参数比例与精度是多少？

- **答案**：B=0 已保证初始 $\Delta W = BA = 0$（起点等价原模型）；若 A 也为 0，则 $\partial L/\partial A \propto B = 0$、$\partial L/\partial B \propto A = 0$，两个矩阵的梯度恒为 0，永远学不动——必须"一零一非零"打破对称（A 高斯初始化并按 $1/\sqrt{r}$ 缩放，B 置零）。LoRA 省的是**可训练参数对应的梯度 + AdamW 两个状态**（全参 7B 按 12 字节/可训练参数 ≈84 GB 起步），权重本体（bf16 推理副本）仍全量在显存——"单卡 LoRA 70B"靠 4-bit 量化底座（QLoRA）。实测：可训练参数 7,872 / 230,226（3.4%），验证 acc 0.924 vs 全参 0.955。
- **锚点**：教程 08_lora_and_classification.md §"1. 问题：全参微调贵在哪"、§"2. 从零实现"

## 覆盖映射（学习目标 → 题号）

| 学习目标（按章节学完/学习目标节归纳） | 题号 | 说明 |
|---|---|---|
| 01 章：GPT-2 经典架构、Pre-LN vs Post-LN、bf16 vs fp16（GradScaler）、梯度累积除以 N、参数量 ≈ 12·embed² | ⚠️ 正文覆盖不足 | 题量所限未出题；README 明言与 Part 6/7 重合内容"视为复习"。关键点收录为闪卡（bf16 8 位指数、累积除以 N） |
| 02 章：SFT 本质、Chat Template、Prompt Masking 四步、masked loss 更大的原因 | Q1 | — |
| 03 章：Bradley-Terry 奖励模型（最后 token、零初始化）、DPO 推导、ORPO/KTO | Q2 | 奖励头三个设计见闪卡 |
| 04 章：PPO（clip ε=0.2、GAE γλ 折中、Actor-Critic）、GRPO（组内优势、k3 KL）、RLVR | Q4 | GAE 偏差-方差折中见闪卡 |
| 05 章：GSM8K mini 评估流水线、temperature/top_k/top_p 各改分布哪部分、三阶段对比归因 | ⚠️ 正文覆盖不足 | 题量所限未出题；解码参数已在 Part 7 04 章部署节初次覆盖。Top-K vs Top-P 收录为闪卡 |
| 06 章：KV 显存公式、量化/投机解码/连续批处理解决什么、TTFT/TPOT/goodput | Q3 | TTFT/TPOT/goodput 定义见闪卡 |
| 07 章：三种评估范式与坑、lm-eval 四元组种子、幻觉检测（语义熵/ECE）、benchmark 污染（GSM1k）、ppl 陷阱 | ⚠️ 正文覆盖不足 | 题量所限未单独出题（正文 07 章本身很充分）；核心数字（GPT-4 judge 85% vs 人类互评 81%、arc_easy 0/5-shot 58%/59%、ECE 0.171→0.198）收录为闪卡 |
| 08 章：手写 LoRALinear、A/B 初始化与 α/r、省哪部分显存、分类微调 4 步 | Q5 | 分类微调 vs SFT 的本质区别见闪卡 |
| 09 章：R1 四阶段管线、规则奖励、self-consistency、test-time compute | ⚠️ 正文覆盖不足 | 题量所限未出题；选修短章（无独立作业）。实测 n=1/4/8 准确率 56%/46%/58% 收录为闪卡 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| bf16 为什么不需要 GradScaler？ | 8 位指数与 fp32 相同、动态范围大，梯度几乎不溢出/下溢；fp16 只有 5 位指数（最大 ~65504）需要 loss 放大 |
| 梯度累积为什么要 `(loss / grad_accum).backward()`？ | `.backward()` 累加梯度，N 次后是和而非均值；不除等效学习率放大 N 倍 |
| 奖励模型的三个关键设计？ | 只取最后 token 的 hidden（因果模型汇总全序列）、reward_head 零初始化（初始 P(A>B)≈0.5 无偏好）、无 bias |
| Bradley-Terry 损失在三种情形下的取值？ | $r_c \gg r_r$：loss→0；$r_c \ll r_r$：loss 很大；$r_c = r_r$：loss=ln 2 |
| DPO 的 β 太大/太小各会怎样？ | 太大：KL 惩罚重、策略锁死在参考模型附近学不到新东西；太小：偏离太远、reward hacking；实践 0.1~0.5 |
| GAE 的 γλ 折中？ | γλ=0 只看单步 TD error（高 bias 低 variance）；γλ=1 看完整轨迹（低 bias 高 variance）；LLM 常用 γ=1.0、λ=0.95 |
| PPO 的 clipped surrogate 在做什么？ | ratio=π_new/π_old，取 min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)，ε=0.2——限制单步策略剧变（信任域） |
| Top-K vs Top-P？ | Top-K 固定候选数（logits 外置 −inf）；Top-P 累积概率阈值、候选数随分布集中度自适应；temperature 除 logits 拉尖/拉平分布 |
| TTFT / TPOT / goodput 的定义？ | TTFT 首 token 延迟（prefill 主导）；TPOT 每 token 间隔（decode 主导）；goodput 满足 SLO 的吞吐（批越大吞吐越高但 TTFT/TPOT 变差） |
| KIVI 为什么 K 按通道、V 按 token 量化？ | K 存在跨 token 一致的固定离群通道，必须按通道隔离；V 无此现象；另保留开头 + 最近 token 的全精度窗口 |
| GPTQ 与 AWQ 一句话对比？ | GPTQ 事后补偿误差（Hessian 逐列摊误差）；AWQ 事前保护重要通道（按激活幅值等价缩放）；int8 逐通道几乎无损，int4 需分组 g128 |
| 为什么 ppl 不能跨模型比较？ | ppl 依赖 tokenizer，不同词表不可比（6400 词表 ppl 11 vs 15 万词表 ppl 3 无可比性）；跨模型看同一语料的 bits-per-byte |
| LLM-as-judge 的一致率与两个必做控制？ | GPT-4 judge 与人类 ~85%（高于人类互评 81%），但必须做位置交换与长度控制（AlpacaEval 2.0 length-controlled） |
| GSM1k 证明了什么？ | 造 GSM8K 镜像新题（同难度同格式）重测——掉分幅度暴露 benchmark 污染/过拟合；常规防御是 13/20-gram 重叠去污染 |
| lm-eval 的随机性"四元组"？ | random_seed / numpy_random_seed / torch_random_seed / fewshot_random_seed；换 fewshot 种子 = 换示例题 = 分数会动；0-shot 58% 与 5-shot 59% 是两条基线 |
| 语义熵检测幻觉的原理？ | 采样 n 次按语义聚簇，SE=簇分布的熵——答案高度一致则熵低；不需要标准答案，代价是每题 n 次采样 |
| 分类微调与 SFT 的本质区别？ | 换头 lm_head→cls_head 取最后位置隐向量；数据是 (sequence, label) 对，不需要 prompt/masking；评测用 accuracy 而非 ppl |
| self-consistency 实测与解读？ | 单位数加法玩具模型：n=1 56%、n=4 46%（众数不稳的采样噪声）、n=8 58%——n↑ 众数趋近条件分布众数；温度 1.1 保持多样性，温度 0 退化为单次贪心 |
| R1 为什么用规则奖励而非神经网络 RM？ | 数学/代码可机器验证、规则不可被 hack；NN-RM 有 reward hacking 风险。奖励 = 准确率 + 格式分（think 段在 answer 之前） |
