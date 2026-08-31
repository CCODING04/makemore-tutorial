# 课程学习路线图 v3 —— 每个节点的「内容安排 / 目标设定 / 学习验证」

> **生成**：2026-08-31。依据：全课程 14 个 Part 的实际内容盘点（教程章节 / 脚本 / 作业）+
> [llm_interview_guide.md](llm_interview_guide.md)（8 个需求簇、6 条讲故事链、硬数字清单、
> 手写代码 TOP8、三个项目故事）+ [paper_reading_guide.md](paper_reading_guide.md)（逐 Part 论文实战）。
>
> **与上一版的关系**：[course_roadmap_v2.md](course_roadmap_v2.md) 是**建设任务清单**（面向教师 /
> 维护者，T1-T7 已实施、T8 数据工程已落地为 Part 13、T9 归档，本文件不重复）；本文件是
> **学习者路线图**——每个节点回答三个问题：**学什么**（学习内容安排）、**学到什么程度**
> （学习目标设定）、**怎么证明学会了**（学习验证）。
>
> **用法**：按阶段顺序推进 → 每完成一个节点先走该节点的「学习验证」再进入下一节点 →
> 阶段五结束进入「毕业验收」→ 求职前按面试指南 §7 选赛道，决定是否走「应用线分叉」。

---

## 1. 全景图

```
节点 0  前置准备            环境 + Micrograd（可选）
  ↓
阶段一 神经网络地基        节点 1-5（Part 1-5）：Bigrams → MLP → BatchNorm → 手写反传 → WaveNet
  ↓
阶段二 现代架构            节点 6-7（Part 6-7）：Transformer/GPT → minimind 现代组件复现
  ↓
阶段三 后训练与对齐        节点 8（Part 8）：SFT → RM → DPO/PPO/GRPO + 推理服务 + 评估学
  ↓
阶段四 系统与工程          节点 9-10（Part 9-10）：CUDA 内核 → 分布式训练
  ↓
阶段五 工业实战四部曲      节点 11-14（Part 11-14）：verl 对齐 → LLaMA-Factory 微调 → 数据工程 → vLLM
  ↓
节点 15 毕业验收           三个项目故事 + 全链面试模拟
  ↓
分叉（按赛道）             算法赛道：缺口登记 → 补 scaling law / 评估实操 / 多模态
                           应用赛道：应用线 A1 最小 RAG → A2 Agent（姊妹系列，待开）
```

### 阶段 × 面试簇覆盖（簇编号对应面试指南 §1）

| 阶段 | 节点 | 主要覆盖 | 关键产出 |
|---|---|---|---|
| 一 | 1-5 | 支撑簇 A 的地基（反传/优化/诊断） | 能手推手写反向传播 |
| 二 | 6-7 | **A ★★★★★** + G（tokenizer/训练全流程） | 现代 LLM 组件全手写 |
| 三 | 8 | **B/C ★★★★★** + H（评估）+ F（06 章） | Pretrain→SFT→DPO/PPO/GRPO 全链 + 量化 |
| 四 | 9-10 | **F/G** | matmul 优化阶梯 + 分布式四件套 |
| 五 | 11-14 | B/C/G/F 的**实战化**（工具栈） | verl / LLaMA-Factory / Data-Juicer / vLLM |
| 毕业 | 15 | 全部 | 3 个简历项目故事 + 面试模拟 |

> **时效说明**：面试指南 §2 映射表写于 Part 11-14 建成之前，其中标注 ❌/🟡 的
> 「LoRA 实战 / RL 实战（verl）/ 数据工程 / vLLM 实战」现已由 Part 11-14 覆盖为 ✅
> （实战层）。仍然有效的缺口见 [§6 缺口登记](#6-缺口登记截至-2026-08-31)。

### 通用验证基线（每个节点默认执行，下文不再重复）

1. **作业**：`pytest assignments/assignment_X/` 全绿；卡壳先自己想，再对照
   [assignment_reference/](../assignment_reference/README.md)（14 套均已实测通过）。
2. **五项完成标准**（见根 README「如何判断学完一个 Part」）：不看笔记能解释核心概念 /
   独立跑通全部 scripts / 作业通过 / 能回答思考题 / 能向别人讲清楚（费曼检验）。
3. **论文**：[paper_reading_guide.md](paper_reading_guide.md) 对应篇目至少完成第①遍快读 +
   第②遍精读方法节；涉及公式用三遍阅读法第③步——数值验证（`tools/verify_paper_formulas.py`
   已含 RoPE / DPO / GAE / LoRA / MinHash / 流水线气泡 12 项断言，跑通即为及格线）。
4. **面试资产**：把本节点新增的「硬数字」（面试指南 §4）加入自己的复述清单；对应
   「讲故事链」（§3）走一遍，链上每一环自问"为什么"；涉及 TOP8（§5）的题目做到
   **不查资料白板默写**。
5. **答题框架**：每个核心概念按「直觉 → 公式 → 数字」三段复述（面试指南 §4），
   只会公式或只会直觉都算未达标。

### 硬件与环境

- **全程 CPU 可学**：Part 1-8 全部脚本有 toy 档；Part 10 全部脚本单进程可跑；作业 1-13
  除 Part 9 题 5 / Part 14 实验题外均可纯 CPU 完成。
- **GPU 强化节点**：节点 7（毕业指南 gpu-original 档 / 租卡成本见 05 章）、节点 9
  （必须 NVIDIA GPU + CUDA Toolkit，`cd courses/Part9_cuda_kernels/scripts && make`）、
  节点 11/14（Docker / 4090 实验题）。无 GPU 的降级路径已写在对应 Part 的 README 与作业里。
- **大文件数据**（minimind 语料 / HH-RLHF / GSM8K / FineWeb / 模型权重）：
  [datasets.md](datasets.md)。

### 参考时长

全程 ≈ 100-140 小时（首次学习、含跑脚本与实验；GPU 训练等待计入日历不计入有效学习）。
按每周 10-12 小时约 **10-12 周**。各节点标注的时长是参考值，已有 PyTorch 基础者可走
**加速通道**：节点 1-5 直接从作业开始做，全绿即通过（验证代替学习）。

---

## 2. 逐节点详解

### 节点 0 · 前置准备（≈3-6h）

**学习内容安排**
1. 环境自检：`uv venv && source .venv/bin/activate && uv pip install -r requirements.txt`；
   确认 `data/names.txt` 与 `data/input.txt` 就位。
2. 通读根 README 的「学习方式」「如何判断学完一个 Part」两节——本路线图所有节点的
   验证基线都建立在它上面。
3. **Micrograd**（Karpathy，约 2.5h，可选但强烈推荐）：从零实现自动微分引擎，
   为节点 4（手动反传）铺路。
4. 浏览三份配套文档的定位即可（不必现在精读）：[references_by_part.md](references_by_part.md)、
   [datasets.md](datasets.md)、本文件。

**学习目标设定**
- 能用自己的话说清链式法则与"计算图"的关系（Micrograd 的 `Value` 类为什么能 backward）。
- 环境、数据、依赖三项自检全部通过，知道遇到问题去哪里查（各 Part README 的环境自检节）。

**学习验证**
- `python courses/Part1_bigrams/scripts/01_explore_data.py` 正常输出数据统计。
- （若学 Micrograd）白纸默写 `Value` 类：`data/grad/_backward/topo` 四要素。

---

### 阶段一 神经网络地基（节点 1-5，Part 1-5）

> 本阶段不直接对应某个面试簇，但它是「训练链」（面试指南 §3 链 1）的根：初始 loss
> 诊断、初始化、归一化、手动梯度全部在这里建立。**不要跳**。

#### 节点 1 · Part 1 Bigrams：最简单的语言模型（≈3-5h）

**学习内容安排**
1. 教程 3 章：01 前置知识与课程预告 → 02 Bigram 模型（从统计到采样）→ 03 用神经网络
   重新实现 Bigram。
2. 脚本 01-07：探索数据 → 计数矩阵 → 可视化 → 概率采样 → NLL 损失 → 神经网络版 →
   梯度下降。
3. 作业 1：计数矩阵 / Laplace 平滑 / 概率与采样 / NLL / 神经网络等价版。
4. 论文（Part 1-2 合并）：Bengio et al. 2003《A Neural Probabilistic Language Model》
   ——重点读 §1"维度灾难"与 §5.1 结构。

**学习目标设定**
- 能解释：Bigram = 按前一个字的条件分布；为什么 loss 用负对数似然（NLL）；平滑解决
  什么问题；"统计版"与"神经网络版"为什么等价（softmax 行 = 归一化计数）。
- 能手写：计数矩阵 → 概率矩阵 → `torch.multinomial` 采样的完整三步。
- 建立"锚点数字"：均匀分布初始 loss ≈ ln(字符表大小)。

**学习验证**
- 脚本 07 梯度下降后 loss 从 ≈3.3 收敛到 ≈2.4-2.5，与计数版损失一致。
- `pytest assignments/assignment_1/` 全绿。
- 论文复现判定：说得出 assignment 2 将是这篇 2003 年论文的最小闭环（为什么）。

#### 节点 2 · Part 2 MLP：Embedding 与训练工程（≈4-6h）

**学习内容安排**
1. 教程 3 章：01 从 Bigram 到 MLP → 02 MLP 架构 → 03 训练与评估。
2. 脚本 01-07：上下文数据集 → Embedding → 前向 → minibatch 训练 → embedding 可视化
   → 采样。
3. 作业 2：block_size=3 数据集构建 / Embedding 查表 / forward+训练循环 /
   train-dev-test 三分 / 采样。
4. 论文：Bengio 2003 第③遍——只推一个公式：`onehot(w) @ C` 的梯度只流向 C 的第 w 行
   （embedding 梯度为什么天然稀疏）。

**学习目标设定**
- 能解释：embedding 表 = one-hot × 矩阵的乘法视图；为什么用 minibatch SGD；
  train/dev/test 各自回答什么问题；dev loss 开始回升意味着什么（过拟合，节点 3 深入）。
- 能手写：从原始文本构造 `(context, target)` 批量数据 + 完整训练循环。
- 面试钩子：讲出"嵌入的梯度是稀疏的"及原因（论文级细节，少有人讲）。

**学习验证**
- 脚本 05 训练收敛；脚本 06 的 embedding 散点图中元音聚簇。
- `pytest assignments/assignment_2/` 全绿。
- 论文进阶：block_size 3→8 重训，观察 dev loss 向 WaveNet（节点 5）逼近。

#### 节点 3 · Part 3 BatchNorm：训练诊断与优化（≈4-6h）

**学习内容安排**
1. 教程 3 章：01 训练诊断（你的模型生病了）→ 02 BatchNorm → 03 深层网络与诊断工具。
2. 脚本 01-06：初始 loss 诊断 → tanh 饱和 → Kaiming 初始化 → BatchNorm 实现 →
   深层网络 → 诊断工具。
3. 作业 3：初始 loss 诊断 / tanh 饱和统计 / Kaiming / **从零 BatchNorm1d
   （train/eval 双模式）** / 深层网络。
4. 论文：Ioffe & Szegedy 2015《Batch Normalization》——三个必推细节：方差用有偏
   （除 N）、反传中 dσ² 项、ε 加在方差里而非标准差里。

**学习目标设定**
- 能解释（面试指南训练链第 2-3 环）：初始 loss 明显高于 ln(V) 说明什么；tanh 饱和的
  直方图特征；Kaiming `gain/√fan_in` 管什么；BN 为什么能让深层网络可训；running stats
  与 train/eval 双模式为什么必须有。
- 能手写：BatchNorm1d 全量（含 `running_mean/var` 与 momentum）——面试常考手写题。
- 知识衔接：这里建立"归一化"坐标系，节点 7 的 RMSNorm 是它的减法版。

**学习验证**
- 脚本 01：未初始化时初始 loss 显著 > ln(V)；脚本 03：Kaiming 后各层激活 std 稳定；
  脚本 04：BN 后激活近似 N(0,1)。
- `pytest assignments/assignment_3/` 全绿（重点：双模式 BatchNorm1d）。
- 论文：不看笔记复述三个必推细节；（进阶）自己写的 BN 与 `nn.BatchNorm1d` 跑同输入，
  对比 running stats 轨迹（10 行）。

#### 节点 4 · Part 4 Backpropagation：手动反向传播（≈6-10h，本课程难度顶点）

**学习内容安排**
1. 教程 3 章（README 带难度星级表）：01 为什么要手写反向传播 → 02 前向+反向逐步推导 →
   03 简化公式与手动训练。
2. 脚本 01-06：前向逐步 → 反传逐步 → 梯度验证 → CrossEntropy 反传 → BatchNorm 反传 →
   手动训练。
3. 作业 4：Q1 forward_pass / Q2 backward_step / Q3 cross_entropy_backward /
   Q4-Q5 batchnorm_backward（🌟）/（拓展）手动梯度训练。
4. 参考：CS231n 反传笔记（论文指南 Part 4 篇目）。

**学习目标设定**
- 能手推：把 BatchNorm 拆成 5 个原子算子逐个反传；CrossEntropy 融合算子的梯度；
  `tanh` 用 `1−h²`（以输出表达，省一次重算）。
- 能解释：autograd 的 retain_grad 语义（为什么非叶子张量的 grad 默认不保留——
  这是作业与脚本的真考点）。
- 心理建设：这是全课程最难的推导，卡住就回 `scripts/02_backprop_step_by_step.py`，
  逐行对照。

**学习验证**
- 脚本 03：手动梯度 vs autograd 全项误差 < 1e-5。
- `pytest assignments/assignment_4/` 全绿（梯度阈值校准过，别调阈值凑答案）。
- 费曼：向别人讲清"局部导数 × 上游梯度"如何串成整条链。

#### 节点 5 · Part 5 WaveNet：层次化架构（≈3-5h）

**学习内容安排**
1. 教程 3 章：01 PyTorch 化（让代码更优雅）→ 02 WaveNet 架构（层次化融合）→
   03 训练与 Bug 修复。
2. 脚本 01-07：PyTorch 化分层 → 学习率曲线修复 → 上下文加长 → FlattenConsecutive →
   WaveNet 组装 → BN 3D 修复 → 放大版。
3. 作业 5：FlattenConsecutive / WaveNet 组装 / BatchNorm 3D bug 修复 / shape 流转验证，
   目标 loss < 2.0。
4. 论文：van den Oord 2016《WaveNet》——图 1（膨胀因果卷积感受野）一张图讲完全文。

**学习目标设定**
- 能解释：层次化融合 vs 直筒 MLP 的表达差异；FlattenConsecutive 与一维卷积的等价
  视角（卷积预览）；BatchNorm 3D bug 是"维度语义混淆"的典型样本。
- 能手写：FlattenConsecutive 层与 WaveNet 块组装。
- 工程习惯：shape 流转表（每层 B/T/C 怎么变）——调试一切序列模型的通用武器。

**学习验证**
- `pytest assignments/assignment_5/` 全绿，loss < 2.0，shape 流转验证通过。
- 论文：只用图 1 讲清"每一层看多远"；（对照）感受野与 Part 6 attention 的全局视野
  差异——这是节点 6 的引子。

---

### 阶段二 现代架构（节点 6-7，Part 6-7）

> 覆盖面试簇 A（★★★★★）。面试指南明确：Transformer 考察深度创新高，
> "为什么 Decoder-only 成为主流""GPT vs BERT"是高频题——本阶段是回答它们的弹药库。

#### 节点 6 · Part 6 Transformer/GPT：从零构建 GPT（≈6-10h）

**学习内容安排**
1. 教程 4 章：01 数据与 Tokenizer → 02 Attention 从零开始（含 6 条 attention 笔记）→
   03 Transformer Block → 04 超越 Transformer（nanoGPT 走读 + RLHF 预告）。
2. 脚本 01-07：数据 → bigram 基线 → attention trick → 单头自注意力 → 多头+FFN →
   LayerNorm+完整 Transformer → scale-up 与生成。
3. 作业 6（376 行）：tokenizer / get_batch / bigram 基线 / 单头 self-attention /
   🌟 完整 Block，带属性测试（shape 与不变量）。
4. 论文：Vaswani 2017《Attention Is All You Need》——至少精读方法节与表 1/2。

**学习目标设定**
- 能解释（高频面试题逐条对齐）：除以 √d 的方差论证；因果 mask 写在 attention 的
  哪一步、为什么；Decoder-only vs Encoder-Decoder 的取舍；残差 + pre-norm 为什么稳
  （衔接节点 3 的归一化坐标系）；Multi-Head 各头在学什么。
- 能手写：**TOP8 #1 多头因果自注意力**——写到默写级（causal mask 位置是判卷点）。
- 叙事资产：架构链（面试指南 §3 链 2）从这里起步。

**学习验证**
- 脚本 03/04：attention trick 的加权均值可视化；脚本 07：生成的 Shakespeare 文本
  可读，训后 ppl≈9-11（指南硬数字）；初始 loss ≈ ln(65)≈4.17。
- `pytest assignments/assignment_6/` 全绿（完整 Block 的属性测试全过）。
- 白板默写 TOP8 #1；自问架构链前半段每环的"为什么"。
- 论文：不看原文画出 decoder-only block 的数据流，标出 5 处与原论文不同之处
  （pre-norm 等）并解释为什么现代实现这么改。

#### 节点 7 · Part 7 Minimind 复现：现代 LLM 组件全手写（≈12-18h，里程碑 ①）

**学习内容安排**
1. 教程 5 章：01 BPE Tokenizer → 02 RMSNorm 与 RoPE → 03 GQA / KV Cache / SwiGLU /
   MoE → 04 Pretrain→SFT→DPO 流水线 → **05 复现 minimind 毕业指南（含面试直通车）**。
2. 脚本 01-11：BPE / RMSNorm+RoPE / GQA+KV Cache / SwiGLU+MoE / 全模型组装 /
   pretrain / SFT / DPO / 三阶段 eval demo / **MoE 负载均衡实验** /
   **RoPE scaling（naive/PI/NTK 外推实测）**。
3. 作业 7（7 题）：BPE 编码 / RMSNorm / RoPE / GQA repeat_kv / SwiGLU / DPO loss /
   🌟 KV Cache。
4. 论文：RoPE（Su et al. 2021）——推"平移不变性 → 只有相对位置差出现在内积里"；
   配 `tools/verify_paper_formulas.py` 的 RoPE 断言。
5. 数据与成本：`docs/datasets.md` 三个 t2t_mini 文件；3090 单卡 dense 全流程
   ≈2.3h / ≈¥3（毕业指南有镜像与断点续传的现实路径）。

**学习目标设定**
- 能解释（簇 A 后半全部转 ✅ 的关键节点）：BPE 的训练与选型；RMSNorm 砍掉了什么、
  权重绑定又省 12% 参数（答题框架示范）；RoPE 相对位置性质怎么证；GQA 省的是
  "KV 头数 × KV cache 显存"（配公式）；SwiGLU 的取舍；MoE 为什么会塌缩、aux loss
  公式每一项是什么。
- 能手写：**TOP8 #2 GQA 的 repeat_kv**、**#6 RoPE 的 precompute_freqs_cis + apply**；
  （进阶）KV Cache 推理循环。
- 能跑通：缩小版 Pretrain→SFT→DPO 三阶段（cpu-toy / gpu-course / gpu-original 三档）；
  毕业指南的配置放大表（课程缩小版 → 26M → 64M → MoE 198M）。
- 叙事资产：训练链（链 1）+ 架构链（链 2）完整讲通；简历项目故事 1 的主体。

**学习验证**
- 硬数字逐项复现（指南 §4）：脚本 10 的 MoE——α=0 时 gini 0.73、5/8 专家死亡，
  α=0.01 拉平但任务 loss 翻倍，minimind 用 5e-4；脚本 11 的 naive 外推 ppl 崩 vs
  PI/NTK 改善（长上下文三件套公式 PI：m→m/s、NTK：θ'=θ·s^(dim/(dim-2))）。
- 脚本 09 三阶段验收（预期输出样例已写死）：pretrain 流利但离题 / SFT 格式正确 /
  DPO 偏 chosen；held-out ppl 曲线齐全。
- 05 章毕业指南验收：对照 `trainer/` 四脚本 + `eval_llm.py`（lm-eval-harness 语义）；
  26M 配置复述：hidden 512 / 8 层 / 8 头 / kv 头 2 / vocab 6400。
- `pytest assignments/assignment_7/` 全绿；白板默写 repeat_kv 与 RoPE apply。

---

### 阶段三 后训练与对齐（节点 8，Part 8）

> 覆盖面试簇 B/C（★★★★★）与 H。指南：对齐链（链 3）是追问最密集的一条；
> RLHF 效果评估三维度（生成质量/安全性/对齐性）是高频题。

#### 节点 8 · Part 8 后训练全流程：从 SFT 到 GRPO（≈12-18h，里程碑 ②）

**学习内容安排**
1. 教程 8 章：01 GPT 与 pretrain → 02 SFT 与 chat（chat template / prompt masking）→
   03 奖励模型与 DPO（Bradley-Terry / ORPO / KTO）→ 04 PPO 与 GRPO（GAE+clip /
   GRPO / RLVR）→ 05 评估与部署（GSM8K 风格）→ 06 推理与服务（量化 / PagedAttention /
   连续批处理 / 投机解码 / vLLM 最小实操）→ 07 评估学（规则 / 人工 / LLM-judge、
   污染）→ 08 LoRA 与分类。
2. 脚本 01-10：GPT / pretrain / SFT / RM / DPO / PPO / GRPO / eval+chat /
   **09 量化与服务全家桶** / **10 LoRA 从零手写**。
3. 作业 8（8 题）：causal attention head / Pre-LN block / prompt-masked SFT loss /
   Bradley-Terry reward / DPO loss / 🌟 GAE / 🌟 PPO clipped / 🌟 GRPO。
4. 论文：DPO（Rafailov 2023）——公式推理五步法完整走一遍（边界检查：β→0 退化成
   什么）；配 `verify_paper_formulas.py` 的 DPO/GAE 断言。
5. 数据：`datasets.md` 的 HH-RLHF / GSM8K；SFT 支持真实数据开关（--original-data）。

**学习目标设定**
- 能解释：prompt masking 为什么用 ignore_index=-100；Bradley-Terry 是什么；DPO 推导
  哪一步消掉奖励模型；clip 与 GAE 各防什么；GRPO 为什么能去掉 Value Network；RLVR
  什么时候可用（配 minimind DPO lr=4e-8 / β=0.15 的实例）；RLHF 效果评估三维度。
- 能手写：**TOP8 #3 prompt-masked CE、#4 DPO loss、#5 GRPO 组内标准化优势**；
  （进阶）#7 absmax int8 逐通道量化、#8 投机解码接受判据（u < pt/pd）。
- 能跑通：三阶段训练 + 量化 + 评估全链；06 章的 vLLM 10 行最小实操。
- 叙事资产：对齐链（链 3）完整讲通；推理系统链（链 4）上层半段；简历项目故事 2 主体。

**学习验证**
- 硬数字逐项复现（脚本 09）：KV 显存 LLaMA-7B fp16 seq2048 = 1.07GB、GQA(kv=8) →
  0.27GB；PagedAttention 碎片 41%（简化模拟，论文语境 60-80%）→ <4%；投机解码
  α≈0.65 → 实测 2.81 tokens/cycle vs 理论 2.53；int8 量化 ppl Δ≈+0.38（500 步配置）。
- 07 章评估学验收：能说清 lm-eval-harness 两类请求（loglikelihood / generate_until）、
  GSM1k 显示 Mistral −8% / Phi −21% 过拟合（污染不是理论问题）、ppl 跨 tokenizer
  不可比。
- `pytest assignments/assignment_8/` 全绿（8 题，未实现题自动 SKIP 的机制先读懂）。
- 白板默写 TOP8 #3/#4/#5；对齐链自问"为什么"到每一环。
- 07 章面试直通车自测；若走了 GPU 路线，把 05 章三阶段行为对比截图存档（面试素材）。

---

### 阶段四 系统与工程（节点 9-10，Part 9-10）

> 覆盖面试簇 F/G。2025-26 面经的"明星话题"是推理优化；分布式是预训练岗硬门槛。
> 本阶段两条链：推理系统链（链 4）的底层半段 + 分布式链（链 5）整条。

#### 节点 9 · Part 9 CUDA 内核编程（≈8-14h；必须 NVIDIA GPU，无 GPU 走 CPU 作业路径）

**学习内容安排**
1. 教程 4 章：01 GPU 架构与第一个 kernel → 02 matmul 优化阶梯 → 03 profiling 与
   CUDA APIs → 04 Triton 与 PyTorch 扩展。
2. 先编译：`cd courses/Part9_cuda_kernels/scripts && make`；脚本 01-08：
   vector_add / 线程层级 / naive matmul / tiled matmul / atomics+streams /
   cuBLAS 对照 / Triton / PyTorch 扩展。
3. 作业 9：题 1-4 纯 CPU/纸笔（线程索引 / 主序与 CPU matmul / tiling 显存账本 /
   GFLOPS 报告），题 5 Triton 实战（无 GPU 跳过）。
4. 参考：siboehm 矩阵乘博客（论文指南 Part 9 篇目）。

**学习目标设定**
- 能解释：warp=32 / block≤1024 / SMEM 48KB（可配置更大）三件套；decode 为什么是
  memory-bound（roofline 直觉）；合并访存为什么差出一个数量级；拿到一个 kernel 怎么
  判断优化方向（算术强度 + ncu SOL）。
- 能手写：naive → tiled 两级 matmul kernel；说出每级提升的瓶颈从什么变成什么。
- 链路意识：Flash Attention 快在哪（tiling + fused，少一次 HBM 往返）——
  原理题从这里的长阶梯取材。

**学习验证**
- 硬数字逐项复现：4090 fp32 峰值 ~82 TFLOPS vs cuBLAS 实测 ~22 TFLOPS；手写阶梯
  553→8795 GFLOPS；合并访存做错 vs 做对差 ~9 倍；树形归约 vs 朴素 atomic 差 ~77 倍。
- `pytest assignments/assignment_9/` 全绿（题 5 按 GPU 有无走对应分支）。
- 面试自测：完整回答"给你一个 kernel 怎么判断优化方向"（链 4 收尾环）。

#### 节点 10 · Part 10 分布式训练（≈6-10h；全部脚本单进程可跑，双卡体验需 ≥2 GPU）

**学习内容安排**
1. 教程 4 章：01 为什么并行 + 集合通信原语 + 分布式 Hello World + 常见报错 FAQ →
   02 DDP 深入（桶、通信重叠、坑）→ 03 显存账本与 ZeRO/FSDP2（`fully_shard`，FSDP1
   已弃用）→ 04 TP/PP 与 3D 并行 + 工业 stack（nanotron / Ultra-Scale Playbook，
   **进阶可选**）。
2. 多卡体验：`torchrun --standalone --nproc_per_node=2 <脚本>`；脚本 01-06：
   collectives / DDP GPT / ZeRO 记账 / FSDP / 手写 TP（f/g 算子）/ GPipe-1F1B 流水线。
3. 作业 10（全 CPU 可完成）：all-reduce 平均语义 / 显存账本计算器 /
   DistributedSampler 不重不漏证明 / TP 分块数学 / 🌟 流水线 bubble 计算。
4. 论文：ZeRO（Rajbhandari 2019）§5.2——用"单调性检查"复算逐字节账本
   （`verify_paper_formulas.py` 的气泡断言一并跑）。

**学习目标设定**
- 能解释：多卡为什么不是插上就行；DDP all-reduce 是平均不是求和；16Ψ 显存账本；
  ZeRO 1/2/3 各切什么、通信代价；TP 的 f/g 算子为什么前向各一次 all-reduce；
  GPipe/1F1B bubble 公式；"给 32 台 8 卡机训 70B 怎么配"的推理链。
- 能手写：ZeRO 三阶段显存公式（zero1：4Ψ+12Ψ/N）；TP 列/行并行的分块数学。
- 叙事资产：分布式链（链 5）整条；简历项目故事 3 主体。

**学习验证**
- 硬标准复现（写死在脚本/教程里）：TP 前向与单卡误差 6e-07（< 1e-5 验收线）；
  PP 流水线 loss 与单进程完全一致（4.380254）；bubble 实测对上公式；ZeRO 记账可复算
  （zero1 vs zero2 谁更省取决于 N——能说清为什么）。
- `pytest assignments/assignment_10/` 全绿。
- 面试自测：分布式链每环的"为什么"；不看笔记写出 16Ψ → zero1/2/3 的显存表。

---

### 阶段五 工业实战四部曲（节点 11-14，Part 11-14）

> "手写 → 工具"双轨：每章先手写核心原理再用工具放大。作业 11-14 均含
> **面试直通车**（4 问）。**按方向调节奏**（面试指南 §7）：
>
> | 方向 | 本阶段优先级 |
> |---|---|
> | 对齐/后训练岗 | 节点 11（最急）→ 12 → 14 → 13 |
> | 预训练岗 | 节点 13 → 11 → 12 → 14（另补 scaling law，见缺口登记） |
> | 推理 infra 岗 | 节点 14 → 12 → 11 → 13（另深入 vLLM/SGLang 源码） |
> | 通用默认 | 11 → 12 → 13 → 14 |

#### 节点 11 · Part 11 对齐实战：verl 工业级 GRPO（≈6-10h；Docker 起步）

**学习内容安排**
1. 教程 2 章：01 从手写 GRPO 到 verl（概念桥接：HybridEngine / 三角色 / Ray）→
   02 verl 快速上手：0.5B GRPO 实战（CLI 实操，Docker → 双卡）。
2. 脚本：01_reward_and_bridge.py（RLVR 奖励函数 + 手写/工业桥接）。
3. 作业 11：3 编码题（稳健奖励函数三级抽取 / 组内优势 + 全同组退化 / KL k3 估计器
   + 预算护栏）+ Docker 实验题（PPO→GRPO 显存对比、reward hacking 观察）。
4. 论文：GRPO（DeepSeekMath）——组内基线为什么成立；配节点 8 的 GRPO 手写互证。

**学习目标设定**
- 能解释：GRPO vs PPO 的工程差异（Value Net 去除 → 显存/吞吐收益）；RLVR 何时可用、
  奖励函数怎么防 hacking；rollout 为什么是瓶颈（HybridEngine 的动机）；KL k3 估计器
  与预算护栏各管什么。
- 能手写：稳健 RLVR 奖励函数（三级抽取）、组内标准化优势、KL 预算护栏。
- 能跑通：verl 0.5B GRPO 训练（Docker 单卡起步 → 双卡）。
- 桥接能力：手写 GRPO 的每个量在 verl 里对应哪个角色/引擎——"手写→工具"的示范样板。

**学习验证**
- `pytest assignments/assignment_11/` 全绿；Docker 实验两个观察（显存对比曲线、
  reward hacking 现象记录）形成书面结论。
- **面试直通车 4 问**（GRPO vs PPO / RLVR / rollout 瓶颈 / KL 估计）自问自答成文。
- 论文：用五步法检查组内优势公式（N=1 时退化成什么？）。

#### 节点 12 · Part 12 微调实战：LLaMA-Factory（≈6-10h）

**学习内容安排**
1. 教程 2 章：01 手写 LoRA SFT 管线（yaml 字段逐行对照——理解每个字段在管线里的
   位置）→ 02 LLaMA-Factory 工作流：identity 小样 → WebUI → QLoRA 7B → export →
   DPO-LoRA。
2. 脚本：01_handwritten_sft_lora.py。
3. 作业 12：4 编码题（LoRA 参数账 / 合并数学 W'=W+BA / B 零初始化证明 / QLoRA 显存账）
   + 观测实验（QLoRA 7B 显存曲线、rank 8→64 对比）。
4. 论文：LoRA（Hu et al. 2021）；先跑 `Part8/scripts/10_lora_from_scratch.py`
   （3.4% 参数实证），再进 QLoRA。
5. 延伸仓库：unsloth（75.2k★，速度对照）。

**学习目标设定**
- 能解释：LoRA 为什么 B 零初始化（第一步等价原模型）、rank/alpha 缩放关系、参数账
  怎么算；QLoRA 的 NF4 + 双量化 + 分页优化器各省什么；什么时候该 LoRA、什么时候
  必须全参。
- 能手写：LoRA 线性层 + 合并数学。
- 能跑通：QLoRA 7B 微调 → export 合并 → DPO-LoRA 全流程（补齐簇 B 的实战缺口）。

**学习验证**
- `pytest assignments/assignment_12/` 全绿（参数账/合并数学/零初始化证明/显存账）。
- 两个观测实验形成数字结论：QLoRA 7B 显存曲线截图；rank 8→64 的显存/效果对比表。
- **面试直通车 4 问**自测；`verify_paper_formulas.py` 的 LoRA 断言全绿。

#### 节点 13 · Part 13 数据工程：MinHash/LSH → Data-Juicer（≈4-8h）

**学习内容安排**
1. 教程 2 章：01 手写 MinHash + 分带 LSH 去重（LSH 概率性质）→ 02 Data-Juicer
   YAML 管线、算子全家桶与审计。
2. 脚本：01_minhash_dedup.py（签名 → 分带 → 候选对 → 并查集聚类全流程）。
3. 作业 13：4 编码题（Jaccard/shingling / 签名一致率 / 分带 LSH 概率反推 /
   簇消解并查集）+ 观测实验（阈值 0.5→0.35 对比、Data-Juicer 追踪报告）。
4. 论文：FineWeb（Penedo et al. 2024）——流水线各步的"赚了多少"读实验表。
5. 背景数字：Gopher/C4 启发式规则、5-gram + 14 band × 8 row ≈ 阈值 0.7。

**学习目标设定**
- 能解释：Jaccard 与 shingling；MinHash 签名为什么保持相似度；分带 LSH 的 S^b 概率
  与"目标阈值 → band×rows"的反推；去重对训练数据污染的防御意义（衔接节点 8 的
  GSM1k 证据）。
- 能手写：MinHash+LSH 去重全流程（纯 Python，百行级）。
- 能跑通：Data-Juicer 管线并读懂审计/追踪报告。

**学习验证**
- `pytest assignments/assignment_13/` 全绿（重点：概率反推题——给定目标阈值算出
  band×rows 组合）。
- 阈值对比实验形成结论（0.5 vs 0.35 各留下多少、误杀/漏杀倾向）。
- **面试直通车 4 问**（含 FineWeb 配置题）自测；`verify_paper_formulas.py` 的
  MinHash 断言全绿。

#### 节点 14 · Part 14 推理部署：vLLM（≈4-8h；4090 实验题需 GPU）

**学习内容安排**
1. 教程 2 章：01 朴素基线与 TTFT/TPOT/吞吐测量（先把指标测出来，才知道 vLLM 赚在哪）
   → 02 vLLM 实战：离线推理 → 服务 → benchmark_serving → 量化 / n-gram 投机解码 →
   三行对比表。
2. 脚本：01_naive_generate_baseline.py。
3. 作业 14：3 编码题（serving 指标公式 / KV 容量账 / 静态批处理浪费率）+ 4090
   实验题（naive vs vLLM 三行对比表、prefix caching 对比）。
4. 论文：vLLM/PagedAttention（Kwon et al. 2023）——与节点 8 的分页模拟互证。
5. 延伸：llama.cpp（126.3k★，端侧线）、SGLang。

**学习目标设定**
- 能解释：TTFT/TPOT/goodput 的定义与测量方法；KV 容量账怎么算（衔接 1.07GB 锚点）；
  静态批处理的浪费率为什么高；prefix caching 的命中条件；量化位宽与 GGUF 命名规则。
- 能跑通：vLLM 服务 + benchmark，产出 naive vs vLLM 对比表（把簇 F 的 🟡 实战项转 ✅）。
- 收官意识：拿到一条线上延迟投诉，能按链 4 的框架定位瓶颈在 TTFT 还是 TPOT。

**学习验证**
- `pytest assignments/assignment_14/` 全绿（指标公式/KV 容量账/浪费率三题）。
- 实验产出一页"三行对比表"（TTFT / TPOT / 吞吐，注明硬件与参数——面试直接引用）。
- **面试直通车 4 问**自测。

---

### 节点 15 · 毕业验收：三个项目故事 + 全链面试模拟（≈4-8h）

**学习内容安排**
1. 三个项目故事写成一页纸各一（面试指南 §6 模板）：
   ① 从零复现 minimind（节点 7 全部 + 05 毕业指南）；② 手写推理优化实验（节点 8
   脚本 09 + 06 章）；③ 双卡分布式训练（节点 10）。
2. 六条讲故事链全部限时串讲（每条 10 分钟；应用链除外——那是应用线的事）。
3. 面试指南 §4 硬数字清单逐条"复现来源"标注（哪个脚本、什么现象）。
4. TOP8 白板默写终检（§5）。

**学习目标设定**
- 每个故事满足叙事公式：**"我跑过，现象是 X，原因是 Y"**——至少引用 3 个自己复现的
  实测数字，而不是"论文说"。
- 六条链中任意一环被追问三层"为什么"仍能接住。

**学习验证**
- 随机抽 5 个硬数字，10 秒内说出出处脚本与预期输出。
- TOP8 默写 8/8（找人对卷或隔天自评）。
- `pytest assignments/` 全库收尾绿；三份一页纸项目故事给非本课程的人讲一遍并提问。

---

## 3. 分叉：应用线（"大模型应用工程师/Agent 工程师"赛道，姊妹系列）

> 依据面试指南 §1 赛道澄清与 §7/§8：RAG/Agent 属应用赛道，与本课程算法主线相互独立。
> 状态：**待开**（见指南 §8 应用线表；载体为"先手写、后 harness"）。

#### 应用线 A1 · 最小 RAG（≈8-15h，待开）

- **学习内容安排**：手写分块 / embedding / 检索 / 重排 / 生成（约 200 行，呼应本课程
  "从零"哲学）→ 对比 ragflow（89.6k★）平台化能力与 llama_index（51.9k★）抽象。
- **学习目标设定**：能讲 RAG 全链路每步的失败模式（chunk 策略与召回率、混合检索、
  幻觉抑制）；能回答"什么时候不该用 RAG"。
- **学习验证**：交付一个可演示的问答 demo + 一页评测（召回率/命中率各一行数字）；
  链 6（应用链）前半段自问自答。

#### 应用线 A2 · Agent（≈8-15h，待开）

- **学习内容安排**：手写 agent loop（模型 + 工具表 + while 循环）→ 读 pi
  （earendil-works/pi，极简 harness：4 个工具 + <1000 token 系统提示词）→
  smolagents（HF，CodeAgent）→ OpenAI Agents SDK / Google ADK 按厂商栈选学 →
  LangGraph 只学图/状态机/checkpoint 概念（LangChain 本体仅作生态认知）。
- **学习目标设定**：能手写最小 function calling 循环并说清中断/重试/上下文管理三件事；
  能评价各框架的抽象收益与成本。
- **学习验证**：交付一个多步工具调用 demo（含一次失败恢复）；链 6 后半段自问自答。

---

## 4. 按赛道的"最后一公里"建议（引用面试指南 §7）

- **预训练岗**：主线节点 7/9/10 是重心；补 scaling law（见缺口登记）+ 节点 13 提前。
- **对齐/后训练岗**：节点 8 + 节点 11 是重心；节点 12 的 DPO-LoRA 段必做。
- **推理 infra 岗**：节点 9 + 8（06 章）+ 10 是重心；节点 14 之后建议进 vLLM/SGLang
  源码精读。
- **应用赛道**：主线节点 0-10 学到概念层即可提前分叉，重点转应用线 A1/A2。
- 求职期配套：面试指南 §9 的牛客面经 + wdndev/llm_interview_note（15.0k★）八股题库；
  毕业读物：llm.c（30.9k★）/ LLMs-from-scratch（104.0k★）源码精读。

## 5. 与上一版 roadmap 的关系（变更记录）

- v2（`course_roadmap_v2.md`）是**建设任务清单**：T1-T7 已实施验证；T8（数据工程）
  已落地为 Part 13；T9（多模态）维持归档；T10（面试直通车）已按"新章第一天就带
  面试怎么问"落地（Part 7 05 章、Part 8 07 章、作业 11-14）。任务清单使命完成，
  由本文件接棒。
- v3 新增：逐节点「内容 / 目标 / 验证」三段式、面试资产（链/数字/TOP8/直通车）到
  节点的映射、按赛道分叉的应用线与毕业验收节点、加速通道。

## 6. 缺口登记（截至 2026-08-31）

> 面试指南总评的缺口中，LoRA 实战 / RL 实战 / 数据工程 / vLLM 实战已由 Part 11-14
> 补齐；以下为**仍然存在**的缺口与拟开去向（按优先级）：

| 缺口 | 所属簇 | 拟开去向 | 状态 |
|---|---|---|---|
| RAG 全链路（算法侧认知） | D | 应用线 A1 | 待开 |
| Agent / function calling | E | 应用线 A2 | 待开 |
| scaling law | G | 独立小章（建议挂 Part 13 前） | 待定 |
| 幻觉 / 安全对齐专题 | H | Part 8 07 章扩节 | 待定 |
| lm-eval-harness 实操 | H | 毕业指南进阶实验 | 🟡（语义已讲，未实操） |
| 长上下文 YaRN 实验 | A | Part 7 脚本 11 扩展（PI/NTK 已实测） | 🟡 |
| Flash Attention 手写内核 | F | Part 9 进阶章 | 🟡（原理已讲） |
| 多模态 VLM | A 补充 | Part 15（可选，LLaVA / Qwen-VL 系） | 归档（v2-T9） |

## 修订记录

- **v3（2026-08-31）**：初版。依据课程 14 个 Part 全量盘点 + 面试指南 v2 + 论文阅读
  指南，把路线图从"建设任务清单"改写为"学习者路线图"：16 个主节点（0-15）+
  应用线分叉（A1/A2），每节点给出学习内容安排 / 学习目标设定 / 学习验证三部分。
