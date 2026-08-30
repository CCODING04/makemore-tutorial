# 论文阅读指南 — 算法工程师的"读论文、推公式、快复现"训练

> **定位**：论文/技术报告/技术博客的阅读能力，是算法工程师区别于"调参工程师"的分水岭。
> 本指南回答三个问题：**① 怎么快速读一篇论文？② 公式怎么推理和验证？③ 感兴趣的话怎么
> 用最小成本复现？** 每个 Part 给出一篇代表论文的完整实战。
>
> **为什么相信这套方法**：它在制定本课程的当天就抓到过一个真 bug——我们的 Part 10 教程
> 把 ZeRO-2 公式写成了 `8Ψ+4Ψ/N`，用"单调性检查"（ZeRO-2 是 ZeRO-1 的超集，显存不可能
> 更大）发现矛盾，再到 ar5iv 读 ZeRO 论文原文 §5.2 核实为 `2Ψ+14Ψ/N`。**不迷信任何
> 二手转述（包括本课程自己），永远回到原文+数值验证。**

---

## 1. 三遍阅读法（工程师版）

不是每篇论文都值得精读。三遍递进，每遍都有明确的"继续/放弃"判据：

| 遍 | 时长 | 读什么 | 判据：值得继续吗？ |
|---|---|---|---|
| **① 10 分钟浏览** | 10 min | 标题/摘要/图 1/结论/每段首句 | 它解决的问题我关心吗？方法属于我需要的谱系吗？ |
| **② 1 小时精读** | 1 h | 方法节全读 + 实验表 1/2 + 训练细节节 | 核心主张有实验支撑吗？消融证明"每部分都必要"了吗？ |
| **③ 深读+推导** | 数小时 | 全文 + 附录（往往藏着实现细节）+ 相关工作 | 我要把它讲给别人/写代码吗？ |

工程师版差异（对比学生版）：
- **先读实验表再看方法**：表 1 的数字告诉你这篇论文"赚了多少"，倒逼你带着问题读方法。
- **训练细节节（lr/batch/数据量）必须读**：这是复现成败的关键，学术读者常跳过。
- **相关工作可无限期推迟**；但引用的"前一篇奠基论文"要记入待读清单（拓扑排序你的阅读）。

**工具**：ar5iv.labs.arxiv.org（LaTeX 全文 HTML，公式可复制）读 arXiv 论文比 PDF 舒服；
用 `arxiv.org/abs/XXXX` 页面的 "Fields" 判断领域；GitHub 上的官方实现是"第 0.5 遍"——
看代码结构往往比读正文更快理解方法骨架。

## 2. 公式阅读与推理五步法

论文公式的默认读法是**猜**：先猜它想表达什么，再验证猜得对不对。五步：

```
① 符号表：把每个符号的 shape/含义写下来（论文常省略；shape 对不上=理解错了）
② 直觉：用一句话说出"这个公式在干蠢事还是聪明事"（例：DPO=给"好过差"的幅度发分）
③ 量纲/边界检查：代入极端值——λ=0/1、α→∞、T=1、N=1 时公式退化成什么？合理吗？
④ 单调性/不变量检查：这个量随参数应该单调吗？有守恒量吗？（我们抓 ZeRO bug 用的就是这步）
⑤ 数值验证：写 5-10 行 torch，把公式和 autograd/暴力实现对比（见 tools/verify_paper_formulas.py）
```

- 🔑 **③④ 是论文阅读的"验钞机"**：绝大多数转述错误、笔误、以讹传讹，都活不过这两步。
- 💡 **推导卡住时的三个万能起点**：a) 从损失函数出发倒推（一切公式都服务于某个目标）；
  b) 从最简单的特例推起（B=1、T=1、r=1）再推广；c) 从"这个公式 Must 满足什么性质"反推
  （例：位置编码必须满足"平移不变性"→ 只有相对位置差出现在内积里 → 推出 RoPE）。

## 3. 最小复现决策树与闭环原则

```
这篇论文值得复现吗？
├─ 影响面大（被主流模型采用）？—— 是 → 值得
├─ 核心主张能用 <100 行验证？—— 是 → 强烈值得
├─ 只有端到端大实验？—— 降级为"跑官方代码"或"复现它的一个组件"
└─ 复现什么：核心公式/机制的【最小可验证闭环】，不是论文的完整实验
```

**最小闭环三原则**：
1. **数据玩具化**：论文用 1T token，你用 100 条合成数据——机制一样，才能聚焦机制。
2. **规模缩小到单卡分钟级**：<30 秒最佳（本课程全部脚本的纪律）。
3. **必须有"对错判据"**：与暴力实现/autograd/已知性质的数值对比——没有验证的复现
   等于没复现（GPU 代码错了不报错，只会安静地给错答案，Part 9 01 章的老话）。

**本课程脚本 = 论文最小复现的活例**（你已经在复现了）：

| Part | 论文/报告 | 课程里的最小复现 |
|---|---|---|
| 3 | BatchNorm (Ioffe & Szegedy 2015) | `Part3/scripts/04_batchnorm_implementation.py` |
| 5 | WaveNet (van den Oord 2016) | `assignments/assignment_5`（膨胀因果卷积的 Flatten 版） |
| 6 | Attention Is All You Need (2017) | `assignments/assignment_6`（完整 decoder-only block） |
| 7 | RoPE (Su et al. 2021) | `Part7/scripts/11_rope_scaling.py`（PI/NTK 外推实验） |
| 8 | DPO (Rafailov 2023) / InstructGPT | `Part8/scripts/05_dpo_alignment.py` + assignment_8 |
| 9 | siboehm 矩阵乘博客 | `Part9/scripts/04_matmul_tiled.cu`（553→8795 GFLOPS 阶梯） |
| 10 | ZeRO (Rajbhandari 2019) | `Part10/scripts/03_zero_memory.py`（逐字节复算） |
| 11 | GRPO (DeepSeekMath) | `Part11/scripts/01_reward_and_bridge.py` + verl quickstart |
| 12 | LoRA (Hu et al. 2021) | `Part8/scripts/10_lora_from_scratch.py`（3.4% 参数实证） |
| 13 | FineWeb (Penedo et al. 2024) | `Part13/scripts/01_minhash_dedup.py`（MinHash+LSH 全流程） |
| 14 | vLLM/PagedAttention (Kwon et al. 2023) | `Part8/scripts/09` 分页模拟 + `Part14/scripts/01` 基线 |

---

## 4. 逐 Part 论文实战（每篇：快读路径 → 公式推导 → 最小复现）

### Part 1-2 — Bengio et al. 2003《A Neural Probabilistic Language Model》

- **为什么选它**：embedding + MLP 做语言模型的"创世论文"——Part 1-2 你写的东西它在 2003
  年就提出了，也是"用神经网络学分布式表示"思想的源头。
- **快读路径**：摘要 → §1 的"维度灾难"论述（为什么连续表示赢过 one-hot 统计）→ §5.1
  结构（其实你的 assignment 2 就是它）。§4 推导可跳。
- **公式推导引导**：核心只有一件事——`C(w)` 查表 = one-hot × 矩阵。自己推一遍：
  `onehot(w) @ C` 的梯度只流向 C 的第 w 行（为什么？），所以 embedding 的梯度天然稀疏
  ——这解释了 assignment 2 里 `C.grad` 大部分是零的现象。
- **最小复现**：`assignments/assignment_2` 就是它的 2003 版最小闭环。加深一步：
  把 block_size 从 3 加到 8，看 dev loss 逼近 assignment 5 的 WaveNet。

### Part 3 — Ioffe & Szegedy 2015《Batch Normalization》

- **快读路径**：摘要 → §3（算法框 + 为什么用 batch 统计）→ §3.2（Stochasticactivation）→
  实验 §4.1（ImageNet 收敛加速）即可。附录的梯度推导不用读——自己推是更好的练习。
- **公式推导引导**：核心公式 `y = γ·(x-μ_B)/√(σ²_B+ε) + β`。三个必推细节：
  ① 方差用**有偏**（除 N）——推理时 running stats 与训练口径一致；② 反传里 dσ² 项
  依赖 Σdh·(x−μ)（assignment 4 的 batchnorm_backward）；③ ε 在**方差**里加而非标准差里。
- **最小复现**：`Part3/scripts/04` 已实现；进阶——把你的 BatchNorm1d 与
  `nn.BatchNorm1d` 在同输入上对比 running_mean/var 的轨迹（10 行）。

### Part 4 — 技术博客：CS231n《Backprop uneasy》/ Karpathy Part 4

- **为什么选博客**：反传没有"原始论文"（Rumelhart 1986 太抽象）；CS231n 的链式法则笔记
  是工程师视角最清晰的推导范本。
- **公式推导引导**：以 `h = tanh(x)` 为例走"局部导数 × 上游梯度"：`dh/dx = 1−h²`
  （用 h 而不是 x 表达——省一次重算！）。再做 BN 的全推导：把 BN 拆成 5 个原子算子
  逐个反传（assignment 4 的 batchnorm_backward 一行公式就是 5 步压缩的结果）。
- **最小复现**：`assignments/assignment_4`（手动梯度 vs autograd 逐项对比）。
  卡住时回看 `Part4/scripts/02_backprop_step_by_step.py`。

### Part 5 — van den Oord et al. 2016《WaveNet: A Generative Model for Raw Audio》

- **快读路径**：摘要 → 图 1（膨胀因果卷积的感受野图——一张图讲完全文）→ §2.2 dilated
  convolutions → §3.1。音频相关章节跳过。
- **公式推导引导**：感受野随深度指数增长：RF = 1 + Σ(2^i × (k−1))。自己推：layer i 的
  输出依赖多少个输入？与 Part 5 的 FlattenConsecutive(2) 对照——课程用的是"每次融合 2 个"
  的树状版（T=8 → 4 → 2 → 1），WaveNet 原版是"每次跳 2^i"的带孔版。
- **最小复现**：assignment 5；进阶——把 FlattenConsecutive 换成 stride=2 的卷积并对比参数量。

### Part 6 — Vaswani et al. 2017《Attention Is All You Need》

- **快读路径**：摘要 → 图 1/2（架构）→ §3.2（attention 一页）→ 表 2（超参对照，复现必备）。
  §4 之外的训练细节在 §5.3（Adam warmup 公式 lr = d^−0.5 × min(step^−0.5, step·d^−1.5)）。
- **公式推导引导**：必推**为什么除以 √d_k**：设 q,k 各分量 iid N(0,1)，则 q·k 的方差 = d_k
  → 除以 √d_k 后方差回到 1，softmax 不会在初始化时饱和成 one-hot。
  （assignment 6 的 test_exercise_4 用实测 std 验证了这一点——3.93/√16 ≈ 0.98。）
- **最小复现**：assignment 6 + `Part6/scripts/`。进阶：实现 warmup 公式并画 lr 曲线。

### Part 7 — Su et al. 2021《RoFormer: Enhanced Transformer with Rotary Position Embedding》

- **为什么选它**：RoPE 是现代 LLM 与 GPT-2 的最大架构差异点，且论文的推导路径极具教学性
  （从"内积只依赖相对位置"的性质出发**反推**出复数旋转）。
- **快读路径**：摘要 → §3.1（复数形式的核心推导，2 页）→ 图 1（内外积几何）→ §3.4.2
  （与 LLaMA 用的实现等价性）。其余节按需。
- **公式推导引导**：三层递进——① 期望性质：⟨f(q,m), f(k,n)⟩ = g(q,k,m−n)（只依赖
  相对位置）；② 二维解：把 q,k 看成复数，f(z,m) = z·e^{imθ}，内积自动只含 (m−n)θ；
  ③ 推广到高维：两两分组、频率按 base^{−2i/d} 几何衰减。
  **课程已验证**：`scripts/11_rope_scaling.py` 数值证实"内积只依赖位置差"。
- **最小复现**：assignment 7 题 3 + 脚本 11（PI/NTK 外推对比）。进阶：实现 §3.4.2 的
  rotate_half 实数版并与复数版对数（数值一致即等价）。

### Part 8 — Rafailov et al. 2023《DPO: Your Language Model is Secretly a Reward Model》

- **快读路径**：摘要 → 图 1（RLHF vs DPO 流程对比）→ §4（核心推导，3 步）→ 图 2/3
  （合成数据上的机理实验——这篇的实验设计本身值得学）。
- **公式推导引导**（全课程最值得手推的一条链）：
  ① BT 模型：p(y_w ≻ y_l|x) = σ(r(x,y_w) − r(x,y_l))；
  ② 代入 RLHF 最优解 r*(x,y) = β·log(π(y|x)/π_ref(y|x)) + β·log Z(x)；
  ③ **Z(x) 配分函数在差分里被消掉**——奖励模型被解析地消去，得到只含 π 与 π_ref 的
  闭式损失。卡住时用"变量替换"视角：把 (y_w,y_l) 的 log-ratio 当作隐奖励。
- **最小复现**：assignment 8 题 5 + `Part8/scripts/05`。进阶：合成数据上画
  "chosen/rejected 的隐奖励差随训练步数"的曲线（论文图 3 的玩具版）。

### Part 9 — 技术博客：Simon Boehm《How to Optimize a CUDA Matmul Kernel to cuBLAS in 1 Hour》

- **为什么选博客**：把"kernel 优化"从玄学变成**可测量的阶梯**，每级都有数字——
  工程写作的范本；也是我们脚本 04 的直接蓝本。
- **公式推导引导**：两个必推——① naively 读 A 的每一行 N 次：全局读次数 = 2MNK，
  tiling 后 ÷T（assignment 9 题 3 的账本）；② roofline： achievable FLOPS =
  min(峰值 FLOPS, 算术强度 × 带宽)——算出 naive 的 ~0.25 FLOP/byte 在 4090 上意味着什么。
- **最小复现**：`Part9/scripts/04`（553→8795 GFLOPS）。进阶：实现 float4 向量化读。

### Part 10 — Rajbhandari et al. 2019《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》

- **快读路径**：摘要 → **表 1**（三阶段显存公式——全文核心，一张表）→ 图 3（通信量对比）
  → §7（与 DP/MP/PP 的组合）。其余是系统细节，用到再回读。
- **公式推导引导**：从 16 字节/参数的账本出发逐行推表 1——
  Pos = 16Ψ − 12Ψ + 12Ψ/N = **4Ψ + 12Ψ/N**；Pos+g = **2Ψ + 14Ψ/N**；Pos+g+p = **16Ψ/N**。
  ⚠️ 网上有大量转述写成"8Ψ+4Ψ/N"——**用单调性检查即可证伪**（ZeRO-2 ⊃ ZeRO-1 却更费）。
  我们的教训实录：本课程教程初稿也写错了，靠这条检查 + ar5iv 原文（§5.2 式 2Ψ+14Ψ/N）
  修正。**这就是"读原文 + 推导验证"的价值**。
- **最小复现**：`Part10/scripts/03_zero_memory.py`（逐字节记账与公式断言一致）。

### Part 11 — Shao et al. 2024《DeepSeekMath》（GRPO 的出处）

- **为什么选它**：GRPO 首次提出于此（R1 让它出圈）；论文同时给了"从 outcome-supervised
  RM 迁移到 RLVR"的完整叙事——与 Part 8 07 章评估学直接挂钩。
- **快读路径**：§2（RL 变体回顾，GRPO 公式在这里）→ 图 2（GRPO vs PPO 流程）→ 表 4/5。
  数学预训练章节可跳（除非关心数据配比）。
- **公式推导引导**：从 PPO 的优势 A_t = (r−b)/? 出发——PPO 用 Value 网络做基线 b=V(s)，
  GRPO 换成**组内均值**：A_i = (r_i − mean(r))/std(r)。推两个性质：① 组内全同时优势
  恒零（天然跳过已掌握题）；② 无需 critic → 省 16Ψ 账本里一整份 Ψ 的训练状态。
- **最小复现**：`Part11/scripts/01`（奖励→组内优势→KL）+ verl quickstart。
  进阶：用玩具策略梯度实现 GRPO 更新并与 PPO 对比样本效率。

### Part 12 — Hu et al. 2021《LoRA: Low-Rank Adaptation of Large Language Models》

- **快读路径**：摘要 → 图 1（reparametrization 一张图）→ §4.1/4.2（哪些层注入、r 怎么选）
  → §5 表（GLUE 对比）。§7 相关工作跳过。
- **公式推导引导**：三个必推——① 低秩假设从哪来（微调的 ΔW 秩 ≪ d：任务适配是"子空间
  移动"）；② B=0、A≠0 的初始化使起点 ΔW=0，且梯度不全零（都零则梯度恒零——对称性
  未打破）；③ 合并 W′=W+(α/r)BA 后推理零开销（Part 12 脚本 01 用 logits 逐元素一致
  验证过——还抓到"合并后忘停旁路→BA 算两次"的经典 bug）。
- **最小复现**：`Part8/scripts/10`。进阶：对同一任务扫 r=2/4/16 画"acc-参数量"曲线。

### Part 13 — Penedo et al. 2024《The FineWeb Datasets》

- **为什么选技术报告而非论文**：数据工程的"论文"形态就是带完整消融的技术报告——
  FineWeb 的每条规则都有 ablation 支撑，这是写/读数据报告的范本。
- **快读路径**：图 1（管线总览）→ §3（每步一个消融：先看数字再读做法）→ 附录的超参表
  （MinHash：5-gram、112 签名、14 band × 8 row、阈值 0.7——全部记录在案）。
- **公式推导引导**：MinHash 的核心恒等式 P[minhash 相等] = Jaccard（01 章 §2 有推导）；
  分带 LSH 的 P(候选) = 1−(1−J^r)^b——用这个公式反推"想要 99% 召回 J=0.7 的对需要几带"。
- **最小复现**：`Part13/scripts/01`（确定性强保证）。进阶：换 jaccard_threshold 0.5→0.35
  观察召回/误报权衡（Part 13 作业的实验题）。

### Part 14 — Kwon et al. 2023《Efficient Memory Management for LLM Serving with PagedAttention》

- **快读路径**：摘要 → 图 2（块表机制，一张图）→ §3.1（KV 浪费 60-80% 的来源）→
  表 3（端到端吞吐）。§4 系统细节用到再读。
- **公式推导引导**：两个账——① 浪费率：整块预留 max_len 时 E[浪费] = 1−E[len]/max_len
  （用你自己的请求长度分布算）；② 分页后浪费 ≤ 块大小/平均长度 → 选 block_size=16 的理由。
- **最小复现**：`Part8/scripts/09` 的分页模拟（41%→5%）；进阶——Part 14 脚本 01 的
  naive 基线 vs vLLM 实测，填完三行对比表。

---

## 5. 数值验证工具

[tools/verify_paper_formulas.py](../tools/verify_paper_formulas.py) 把上面多篇论文的核心
公式做**数值验证**（RoPE 正交性与相对位置不变性、DPO 在 π=ref 时 loss=ln2、GAE 的
λ=0/λ=1 退化、LoRA 合并前后 logits 一致、MinHash 的 P[相等]=Jaccard、流水线气泡公式）：
`python tools/verify_paper_formulas.py`——把它当模板：读任何论文时，把"④边界检查 +
⑤数值验证"写成一个这样的脚本，就是你的论文笔记。

## 6. 阅读清单的维护方式

- 每读完一篇：在本文档对应 Part 下追加一行"我读了 + 一句话收获 + 一个没看懂的问题"。
- 每月把"没看懂的问题"批量查一次（往往那时你已经从别的论文里学到了答案）。
- 论文→课程的映射反过来用也成立：**面试被问某篇论文时，你能引用自己跑过的实验数字**。

---

[← 返回课程总览](../README.md)
