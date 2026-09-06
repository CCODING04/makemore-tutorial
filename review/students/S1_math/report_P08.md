# S1 学习报告·P08（后训练全流程）

- 审计角色：学生 agent S1（数学畏难型），全程中文
- 审计范围：`courses/Part8_post_training/tutorial/` 全部 10 个 .md + `../scripts/` 全部 13 个 .py + `assignments/assignment_8/assignment.md` 题面
- 审计方式：数学处逐抠（每个公式自己复算一遍）+ 脚本实跑（scratch：`makemore-tutorial-review/scratch/S1_P8`）+ 概念先自答再核对

## 总分：9.0 / 10

一句话：这是全课程数学密度最高的 Part，但几乎每个公式都配了直觉翻译，畏难学生能跟上；扣分点集中在"给结论不给理由"的 k3 估计器、个别无数字例的目标函数、以及少量数字前后不一致。

## 卡点清单（按严重度排序）

| # | 位置 | 卡点 | 严重度 |
|---|------|------|:---:|
| 1 | tutorial/04_ppo_and_grpo.md「k3 KL 估计器」 | 只列了"无偏、非负、数值稳定"三个**结论**，没有讲**为什么无偏**（缺关键一步：E_{y~π}[exp(log π_ref − log π)] = 1），"低方差"（相比 k1=log-ratio 可正可负）更是全文未提。我自答时只能硬背结论，用数值实验才自己验证出来（E_π[k3]=0.3681=真实 KL）。这恰是任务点名要求讲清的点 | 高 |
| 2 | tutorial/04_ppo_and_grpo.md「ratio clip」下的 ASCII 图 | 图画错了/误导：只画一条对角线加一句"截断区域：(1+ε)*A"，没有体现 `min(surr1, surr2)` 的分段平坦形状，也没区分 A>0 / A<0 两种情形。畏难学生盯着这张图会比只看代码更糊涂 | 高 |
| 3 | README.md vs 01_gpt_and_pretrain.md | 参数量**自相矛盾**：01 章表格说 CPU 缩小版（embed=64, blocks=2）≈ 0.1M（我复算：12·64²·2 + 256·64×2 + 64·64 ≈ 0.135M，0.1M 正确），README 规模对照表却写"本课 CPU 模式 ~2M"，差 15 倍 | 中 |
| 4 | tutorial/04_ppo_and_grpo.md GRPO 数字例 | group_std=0.58 没说明是**样本标准差（除以 N−1）**。我第一次复算按总体标准差得 0.5，对不上 0.58，卡了 2 分钟才反应过来 torch.std 默认 unbiased。加半句"torch.std 默认除以 N−1，故 0.58 而非 0.5"就能救所有较真的学生 | 中 |
| 5 | tutorial/02_sft_and_chat.md + 03_sft.py 自身输出 | 教程与脚本结尾都断言"masked loss 通常比 unmasked **大**"，但实跑输出 Unmasked=2.7361 > Masked=2.7182，**与声明相反**。合成数据+欠训练时该断言不成立，教程应把"通常"的适用条件说清，或把 demo 输出的解释改成"本例中接近/略小也正常" | 中 |
| 6 | scripts/08_eval_and_chat.py 实跑 | (a) 阶段表只列出 pretrain/sft/dpo 三行，GRPO ckpt 加载直接 `[FAIL]`（报 size mismatch：ckpt 里 n_embed=512 vs 当前 64——checkpoint 配置不一致时脚本 FAIL 而不是跳过并提示重训）；(b) 10 题准确率**全 0%**，教程 05 章给的"SFT ~10-20%"预期区间在 CPU 全流程达不到，"看各阶段趋势"实际上没有趋势可看（虽然 05 章有"CPU 数字会更低"的免责声明） | 中 |
| 7 | tutorial/03_reward_and_dpo.md / 04 章 PPO/GAE | DPO、PPO clip、GAE 三处**有公式无数字例**（部分被作业题面的验证标准兜住：BT/DPO 的 ln2、GAE λ=0 退化、ratio=1 → loss=−mean(A)）。对畏难学生，正文缺一个"代入 4 个数算一遍"的例子 | 低 |
| 8 | tutorial/03_reward_and_dpo.md 课后 Q1 | "β 太大意味着 KL 惩罚很重"——严格说 DPO 里没有显式 KL 惩罚项，β 是隐式奖励 β·log(π/π_ref) 的缩放/温度。这是常见简化说法，但和正文推导出的"无显式 KL 项"放在一起读会显得前后口径不一 | 低 |
| 9 | scripts/09_reasoning_models.py 输出 | 笔误："奖励 = 规则准确率 + 格式分（非 **NM**，防 reward hacking）"应为 NN（神经网络）；教程 09 章写的是"非神经网络 RM"，正确 | 低 |
| 10 | scripts/09_reasoning_models.py 实跑 vs 教程数字 | 我的 CPU 跑出 SFT loss=0.68（教程引 0.2177）、self-consistency n=4=70%>n=1=58%（教程是 n=4 反而更低）。两组数字环境不同不具可比性，教程"看趋势别死记数字"的声明覆盖了这点，故仅记录 | 信息 |

未发现乱码、残句或死链：10 个 md 逐一扫描无 `�` 类乱码；抽查章节互链（01→02、03→04、04→05、05→06→07、README 导航、跨 Part 链接 Part7/9/12/15/18）全部有效。

## 分章评分表（数学准确性 × 对畏难学生的友好度，各 5 分）

| 章 | 内容 | 数学准确 | 友好度 | 亮点/问题 |
|---|---|:---:|:---:|---|
| 01 GPT 与预训练 | 注意力缩放、参数量公式 | 5 | 5 | 参数量公式 12·embed² 拆解清楚；数值全对 |
| 02 SFT 与 Chat Template | prompt masking | 5 | 5 | mask 口径与 loss 计算严格一致（见附录 A1）；mask 对齐图是好的数字例；仅"masked>unmasked"断言与 demo 输出相悖 |
| 03 奖励模型与 DPO/ORPO/KTO | BT→DPO 推导 | 5 | 4 | 推导四步完整、β 讲了三处；缺数字例；Q1 的 KL 说法略含糊 |
| 04 PPO 与 GRPO | ratio/clip、GAE、组内优势、k3 | 4.5 | 3.5 | GRPO 数字例满分；k3 只给结论；clip ASCII 图误导；符号 γ/λ 定义齐全 |
| 05 评估与推理部署 | 解码策略 | 4.5 | 4 | temperature/top-k/top-p 表格清楚；准确率预期区间对 CPU 版偏乐观 |
| 06 推理与服务 | 量化/KV/投机解码 | 5 | 5 | 数学最扎实：KV 公式我逐项复算全对（1.073GB）、E 公式复算 2.31 与文一致；"CPU Δ 变负"的诚实预判被实跑验证 |
| 07 评估学 | ECE/SE/拒绝方向 | 5 | 4.5 | ECE、语义熵、diff-in-means 公式都有且标注了"看方向别看效应量" |
| 08 LoRA 与分类微调 | 低秩分解 | 5 | 5 | ΔW=BA、α/r 缩放、B=0 起点无损，Q1"两个都为 0 会梯度恒 0"是漂亮的数学点 |
| 09 推理模型 | R1 管线/self-consistency | 4.5 | 4 | 公式少而清楚；实测数字环境敏感已声明 |
| README | 导航/规模表 | 4 | 4 | 结构清晰；CPU 参数量 ~2M 与 01 章 ~0.1M 矛盾 |

## 只改 3 件事

1. **补 k3 的两行推导 + 一个数字例**（04 章）。加上"E_{y~π}[exp(log π_ref − log π)] = Σ_π π_ref = 1，故 E[k3] = 1 + KL − 1 = KL"即无偏；再对比 k1 = log(π/π_ref)（可正可负、单样本方差大）说明低方差。数字例可用我验证的：π=(0.9,0.1)、ref=(0.5,0.5)，KL=0.3681，k3 逐点值 (0.143, 2.391)，期望恰为 0.3681。
2. **重画或删除 clip 那张 ASCII 图，并给 clip 补一个数字例**（04 章）。例：A=1、ε=0.2、ratio=1.5 → surr1=1.5、surr2=1.2、min=1.2，loss 不再随 ratio 增长——一行就够。
3. **修数字一致性**：README 的"CPU 模式 ~2M"改为 ~0.1M；GRPO 例注明 std 是 N−1 口径；02 章给"masked > unmasked"断言加适用条件。

## 最喜欢 3 处

1. **04 章 GRPO 组内优势的数字例**（r=[1,0,0,1] → mean 0.5 / std 0.58 / adv ±0.87）：全教程唯一一处"敢把中间数全写出来"的公式讲解，我逐位复算全对（0.5774→0.58、0.8659→0.87），畏难学生跟着手算一遍就懂了"组内标准化=天然 baseline"。
2. **03 章 DPO 四步推导**（RLHF 目标 → 闭式解 → 反解奖励 → 代入 BT 后 Z 消掉）：每步只加一两句人话，"Z(x) 在减法中消掉了"那一下有真正的啊哈感；β 的作用在正文和课后题里重复强调了三次，正合畏差学生的需要。
3. **06 章量化的"公式→论文数字→本课实测→偏差解释"四层结构**：KV 显存表（1.07→0.27→0.13→0.03 GB）我能逐项手算复现；投机解码给出"理论 2.31 vs 实测 2.47"并解释差异来源；甚至预判了"CPU 欠训练时 Δ 变负"——我实跑果然得到 int8 Δ=−0.37、int4 Δ=−1.01。敢把自己的数字和论文数字并排放并解释差距，这是全套教程里最诚实也最有教学价值的一章。

## 费曼自检：DPO「直觉 → 公式 → 数字」

**直觉（不看书的自答）**：RLHF 要"奖励高但不跑远（KL 惩罚）"，得同时养 reward model + ref model + 在线采样。DPO 的洞察是这个带 KL 惩罚的目标有闭式最优解 π* ∝ π_ref·exp(r/β)，把它反解成 r = β·log(π/π_ref) + β·log Z，再塞回 Bradley-Terry 的 P(A>B)=σ(r_A−r_B)——两个 reward 相减时 log Z(x) 抵消，奖励模型就"解散"了，剩下对策略 log-prob 比值的一个 sigmoid 二分类。核对：与教程一致。

**公式（默写）**：L = −log σ( β·[ (log π(y_c|x) − log π(y_r|x)) − (log π_ref(y_c|x) − log π_ref(y_r|x)) ] )。β 小→敢偏离 ref（激进），β 大→被钉在 ref 附近（保守），常用 0.1~0.5。核对：一致；我另外注意到 β 的机制是"隐式奖励的缩放/温度"，教程 Q1 用"KL 惩罚轻重"来类比，含义对但措辞不严格。

**数字（自己代入并验证）**：
- 取 log π_c=−2.0, log π_r=−3.0, ref_c=ref_r=−2.5, β=0.1 → logits=0.1·[(−2.0+3.0)−0]=0.1 → loss=−log σ(0.1)≈0.644 < ln2，说明策略已比 ref 更偏好 chosen；隐式奖励 chosen=+0.05、rejected=−0.05，有区分度。
- 边界情形 policy==ref → loss=ln2=0.6931，我用代码验证：0.6931 ✓（这也是作业题 5 的验证标准）。
- 推广验证：ref 冻结是必要假设——若 ref 也更新，logits 里的 baseline 就失效，"锚点"说法成立。

结论：我能给一个没学过的同学讲明白 DPO 为什么不需要奖励模型。自检通过。

## 附录 A：数学逐抠记录

- **A1 SFT prompt masking 口径一致性**：mask 构造 `mask[0, prompt_len:] = 1`（prompt 含 `<|assistant|>` 标记为 0）；loss 内 `mask = loss_mask[:, 1:]` 与 `targets = tokens[:, 1:]` 同步 shift——mask 索引的是**目标 token** 位置，位置 t 的 logits 预测 t+1 的 token、用的是 mask[t+1]，口径严格一致。归一化 `ce.sum()/mask.sum().clamp(min=1)` 与公式 L=Σ_response/|response| 一致，clamp 防除零有交代。教程的 token-mask 对齐示意图逐列核对无误。**结论：一致，且是四步讲解里少有的"代码即公式"范式。**
- **A2 BT→DPO**：闭式解 π* = π_ref·exp(r/β)/Z(x) 正确（对目标 max E[r] − β·KL 加拉格朗日/配分函数即得）；反解 r = β·log(π/π_ref) + β·log Z 正确；代入 BT 后 Z 因同 prompt 相减消掉——推导链我逐步验算无误。缺数字例（见卡点 7）。
- **A3 PPO ratio/clip**：ratio=exp(new−old)、surr2=clamp(ratio,0.8,1.2)·A、loss=−min(surr1,surr2)——与论文一致（取 min 保守更新）；mask 加权平均正确。ASCII 图问题见卡点 2；无数字例。
- **A4 GAE**：δ_t = r_t + γ·V(s_{t+1}) − V(s_t) 中每个符号都有定义（r_t、V 两个值头输出、γ=折扣因子、λ 在"γλ 权衡"三条里定义、nonterminal=response mask 边界）。递归式 lastgae = delta + γλ·nonterminal·lastgae 与求和式 A_t=Σ(γλ)^l·δ_{t+l} 数学等价（我验证：展开递归即加权和）。γ=1.0 的"LLM 无终止状态"理由讲了。无数字例。
- **A5 GRPO 组内优势**：代码 view(-1, G) 组内标准化；数字例复算全对（std 为 N−1 口径，见卡点 4）。作业题 8 的验证标准（组内均值 0、标准差 1）与实现相符。
- **A6 k3**：公式 exp(d) − d − 1（d = log π_ref − log π）实现正确；非负性由 e^x ≥ 1+x 保证（教程未点破这一句）。无偏性我数值验证：π=(0.9,0.1), ref=(0.5,0.5) 时 E_π[k3]=0.3681=真实 KL ✓，且逐点 k3 值 (0.143, 2.391) 均 ≥0 而朴素 log-ratio 有一 1.609 的负值——这正是"低方差/非负"该讲而没讲的素材。教程缺推导与"低方差"论证（卡点 1）。
- **A7 量化（06 章）**：scale=max|W|/127、反量化误差 ≤ scale/2 正确；分组摊销 16bit/128≈0.125bit 正确；KV 公式复算 LLaMA-7B@2048 fp16：2×32层×32头×128×2048×2B=1.073GB ✓，GQA-8 → 0.268≈0.27 ✓，int8 → 0.135≈0.13 ✓；投机解码 E=(1−α^(γ+1))/(1−α)，α=0.6,γ=4 → 2.306≈2.31 与文一致 ✓。实跑（CPU，rc=0）：α≈0.56 → 理论 2.16；量化 Δ 为负但教程已预判（"CPU 50 步欠训练 Δ 变小甚至变负"）。
- **A8 ECE/语义熵/LoRA（07/08 章）**：ECE 10 桶定义与"置信度−准确率加权"口径正确；SE→ln(n) 上界说法正确；LoRA 的 B=0 ⇒ 起点无损、A=0 ⇒ 梯度恒 0（∂L/∂A∝B、∂L/∂B∝A）的对称性论证正确且精彩。

## 附录 B：脚本运行记录（scratch：/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S1_P8，MPLBACKEND=Agg，3 分钟超时）

| 脚本 | 结果 | 备注 |
|---|---|---|
| 01_gpt_model.py | rc=0 秒级 | 架构对比打印正常 |
| 02_pretrain.py | rc=0 <3min | loss 下降、生成正常 |
| 03_sft.py | rc=0 <3min | masked 2.7182 / unmasked 2.7361（与教程断言相悖，卡点 5） |
| 04_reward_model.py | 首跑 CUDA OOM；强制 CPU 后 rc=0 | 共享 GPU 显存被占满，非脚本 bug |
| 05_dpo_alignment.py | rc=0 <3min | DPO/ORPO/KTO 全流程通过 |
| 06_ppo_training.py | rc=0 <3min（CPU） | GAE/clipped loss 路径执行正常 |
| 07_grpo_training.py | rc=0 <3min（CPU） | 组优势/k3 路径执行正常 |
| 08_eval_and_chat.py | rc=0（带瑕疵） | GRPO ckpt size-mismatch FAIL、阶段表缺 ppo/grpo 行、准确率全 0%（卡点 6） |
| 09_quantize_and_serve.py | rc=0 <3min | 量化 Δ 为负（教程已预判）；α≈0.56→理论 2.16 |
| 09_reasoning_models.py | rc=0 <3min | SFT loss 0.68 / n=1:58%, n=4:70%（与教程引数字不同环境，已声明） |
| 10_lora_from_scratch.py | rc=0 秒级 | LoRA 对比实验通过 |
| 11_hallucination_safety.py | **time-blocked** | 脚本自述 GPU 约 5 分钟；共享 GPU 当前 OOM、0.5B 走 CPU 超 3 分钟预算（transformers/模型缓存均在，环境本身可跑） |
| 12_lm_eval_hands_on.py | **time-blocked** | lm_eval 0.4.x 已安装，但 0.5B×limit=100 在 CPU 超 3 分钟预算（依赖齐全，属算力受限非 env-blocked） |

**实跑小结**：13 个脚本中 11 个 rc=0 跑通（含全部秒级演示类与全部训练类核心路径），2 个因 3 分钟/显存预算 time-blocked（依赖本身齐全）；0 个因数据缺失 env-blocked，与 README"完全自包含"的承诺一致。

## 附录 C：作业题面对照

8 道题（Head/Block/SFT Loss/BT/DPO/GAE/PPO/GRPO）与教程章节一一对应，分值 100；三道观测题（量化实测、投机解码 γ 扫描、评估污染审查）正好落在 06/07 章。值得表扬：题面验证标准补上了教程正文缺的数字锚点（BT/DPO 的 ln2、GAE λ=0 退化式、ratio=1 → loss=−mean(A)、GRPO 组内均值 0 方差 1）。
