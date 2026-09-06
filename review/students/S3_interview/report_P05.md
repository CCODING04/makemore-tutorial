# S3 学习报告 · P05（WaveNet / 层次化架构）— 面试冲刺视角

- 审计人：学生 agent S3（面试冲刺型）
- 日期：2026-09-04
- 材料：courses/Part5_wavenet/tutorial/（README + 3 章）、scripts/01-07、assignments/assignment_5/（题面实际为 README.md）、docs/course_roadmap_v3.md 节点 5
- 运行环境：仓库 .venv，CPU（32 核），OMP_NUM_THREADS=4，scratch=/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S3_P5/
- 声明：未读 assignment_reference/、未联网；所有训练为 CPU 复跑，与教程（视频原配置）可能存在种子/线程差异。

## 总分：7.5 / 10

一句话：架构叙事和白板素材一流，BN 3D bug 讲解是全套教程亮点；但两个硬数字失守（放大模型参数量 ~170K 实为 76,579；block8 展平 ~2.02 复跑 2.106），面试现场被追问就会翻车。

## 卡点清单（按严重度）

1. 【硬伤·数字】03_training_and_bugs.md 参数表"放大模型 ~170K"，实测（07 脚本自带打印）= 76,579 ≈ 76.6K，解析计算一致（648+6144+256+32768+256+32768+256+3456+27）。差 2.2 倍，不可辩护。
2. 【对不上·数字】02 章"扩大上下文窗口"表 block_size=8 → "~2.02"；复跑 03_increase_context.py（20K 步、seed42）验证 loss = 2.1064。几乎与 block3 基线 2.10 持平，直接削弱正文"仅靠更多上下文就能提升"的叙事。该 ~2.02 也写在脚本 03 的 docstring 里。
3. 【正文缺口】WaveNet 最后一层 FC(2) 后输出是 (B,1,128)，Linear 后是 (B,1,27)，训练时靠 `logits.view(-1, vocab)` 收成 (B,27)——脚本 05/07 均如此，但正文 02/03 章只字未提；作业题 4 思考题却明确问"(B,1,C) 的 squeeze/flatten 怎么变成 (B,C)"。正文答不出自己的作业。
4. 【措辞过强】03 章"层次融合…本质上等价于 Dilated Causal Convolution""完全等价"。准确说法：FlattenConsecutive+Linear 栈是 kernel=stride=2 的**非重叠**下采样融合，感受野同样指数增长，但 WaveNet 原文是 stride=1 滑窗+膨胀，重叠更多、每层输出 T 不减半。"感受野增长等价、实现不等价"才可辩护。
5. 【工程预期缺失】CPU 复跑 20K 步需 5-8 分钟（03: 4m52s；05: 8m06s，两进程并行抢核时更糟；默认 32 线程时 4 进程互相拖垮、全部超时）。教程/脚本未给任何运行时长预期，也未提供 --quick 之外的降档（07 有 --quick，03/05 没有）。
6. 【小】作业目录无 assignment.md，题面在 README.md；roadmap 节点 5 写"作业 5：FlattenConsecutive / …"未指明文件名，作业 README 自述"文件结构"里却列了 assignment.md（文件不存在）。
7. 【小·time-blocked】"WaveNet 放大 ~1.99 / 首次降到 2.0 以下"未能在 3 分钟限内复核（50K 步 CPU 估 10-15 分钟）。核法：完整运行 `python courses/Part5_wavenet/scripts/07_scaled_wavenet.py`（去 --quick），看"最终评估·验证集"。--quick（1000 步）sanity：dev 2.209，方向正常。

## 逐章评分

| 章节 | 分 | 评语 |
|---|---|---|
| README.md | 8.5 | 导航/路线图/"学完你能"清单齐全；"验证 loss 降到 <2.0"是全 Part 最强承诺，风险集中在 03 章 |
| 01_pytorchify.md | 8 | 模块化动机（字典→对象）真实有出处（Part3 的 05 脚本确用字典结构，已核对）；train/eval 双模式表格清晰。缺点：手动开关 training 的代码偏啰嗦，可一句 `model.train()/eval()` 对照精简 |
| 02_wavenet_architecture.md | 7 | 树状 ASCII 图 + "先消化 2 个再 4 个再 8 个"是最佳讲法；Linear 多维输入洞察到位。扣分：~2.02 表复跑对不上；缺 (B,1,C)→输出收尾；未给完整逐层 B/T/C 表 |
| 03_training_and_bugs.md | 7 | BN 3D bug 一节（buggy mean (1,T,C) vs fixed (1,1,C)）是本 Part 最强资产；卷积预览图直观但"完全等价"措辞过强。扣分：~170K 硬伤；~1.99 未核 |

## 学习目标达成表（roadmap 节点 5 逐条）

| # | roadmap 学习目标 | 正文覆盖 | 证据位置 | 判定 |
|---|---|---|---|---|
| 1 | 解释层次化融合 vs 直筒 MLP 的表达差异 | 是 | 02"动机"+树状图；作业 Q1 参数量对比、Q2 vs Attention | ✅ 覆盖好 |
| 2 | FlattenConsecutive 与一维卷积的等价视角（卷积预览） | 部分 | 03"卷积预览" ASCII 图 | ⚠️ 有内容但"完全等价"措辞过强（见卡点 4） |
| 3 | BatchNorm 3D bug 是"维度语义混淆"典型样本 | 是 | 03 章 bug/修复/验证三段 + 脚本 06 实测 | ✅ 全 Part 最佳 |
| 4 | 能手写 FlattenConsecutive 层与 WaveNet 块组装 | 是 | 02 代码块；脚本 04/05；作业题 1/2 | ✅ 覆盖好 |
| 5 | 工程习惯：shape 流转表（每层 B/T/C） | 半 | 作业题 4 verify_shapes；正文只有 FC 三步示例 | ⚠️ 正文缺完整逐层表与 (B,1,C) 收尾（卡点 3） |
| 验证 | pytest assignments/assignment_5 全绿、loss<2.0 | — | 实测：空骨架状态 3 通过 2 失败/跳过（测试器本身工作正常） | ⏳ 需学生完成作业后验证 |
| 验证 | 论文只用图 1 讲"每一层看多远" | 部分 | 03 章 ASCII 膨胀卷积图（未直接引用论文图 1） | ⚠️ 可用 |
| 验证 | 感受野 vs Part 6 attention 全局视野对照（节点 6 引子） | 半 | 正文仅 README 一句引子；作业 Q2 提示有对比 | ⚠️ 建议正文 03 章末补 3 行对照 |

## 白板默写自测（不看材料复述）

**FlattenConsecutive 解决什么问题、形状怎么变**
- 问题：直筒 MLP 把 8 个 embedding 展平成 80 维一次性混入 Linear，没有"先局部后全局"的结构先验，且首层参数随上下文线性爆炸（80×200=16,000）。
- 变形：(B,T,C) → (B,T//n,C*n)，把**相邻 n 个位置沿通道维拼接**（view 重解释 stride，零拷贝）。链：(B,8,10) →FC(2)→ (B,4,20) →(Linear+BN+Tanh)→ (B,4,68) →FC(2)→ (B,2,136)→…→ (B,1,274→68)，即 8 chars → 4 bigram → 2 fourgram → 1 eightgram。
- 细节：T 必须整除 n（作业要求 AssertionError）；view 要求内存连续；结尾 (B,1,C) 需 view/ squeeze 成 (B,C) 再进输出层（正文缺口，见卡点 3）。
- 自评：能默写 ✅。

**WaveNet 树状融合 vs 直接展平的权衡讲法**
- 参数账：展平首层 80×200=16,000；树状首层 20×200=4,000（作业 Q1 数据），小模型全量 22,397。每层输入维度减半增长、层数换宽度。
- 表达：树状强制"相邻先融合"，感受野按层翻倍（1→2→4→8），等价 kernel=stride=2 卷积栈（正文说"完全等价 dilated causal conv"过强——原版 WaveNet 是 stride=1 滑窗+膨胀、T 不减半；我们的是它的非重叠特例）。
- 代价：融合模式固定（只能看相邻对），无法动态加权——一句话引出 attention（O(n·L) 固定模式 vs O(n²) 动态全局），正是节点 6 引子。
- 自评：能讲 ✅，且能主动指出正文"完全等价"的偏差。

## 硬数字审计表

| # | 量化声明 | 出处 | 复跑结果 | 判定 / 面试可辩护性 |
|---|---|---|---|---|
| 1 | WaveNet 小模型参数 ~22K | 03 章表格；脚本 05 docstring | 实测打印 22,397；解析计算一致 | ✅ 对上；可辩护 |
| 2 | 放大模型参数 ~170K | 03 章参数表 | 实测（07 脚本 --quick 打印）76,579；解析一致 | ❌ 错 2.2 倍；不可辩护，必须改稿 |
| 3 | 放大表超参（n_embd 24 / n_hidden 128 / bs 128 / 50K 步） | 03 章表 | 与脚本 07 逐项一致 | ✅ |
| 4 | MLP(Part2) block3 dev ~2.10 | 02/03 章对比表 | 未复跑（属 Part2 范围） | ⚠️ 出处为 Part2/原视频；引用性数字，可辩护性中 |
| 5 | 深层 BN(Part3) block3 dev ~2.07 | 03 章对比表 | 未复跑（属 Part3 范围） | ⚠️ 同上 |
| 6 | block8 直接展平 dev ~2.02 | 02 章表格；脚本 03 docstring | **实测 2.1064**（20K 步，seed42，4m52s） | ❌ 对不上（+0.09）；实测与 block3 基线持平，叙事需改 |
| 7 | WaveNet 小模型 block8 dev ~2.07 | 03 章对比表；脚本 05 | 实测 2.0957（8m06s） | ⚠️ 接近（+0.03）；种子/线程噪声内勉强可辩护，宜写"≈2.1" |
| 8 | WaveNet 放大 block8 dev ~1.99、"首次 <2.0" | 03 章对比表 | time-blocked：50K 步 CPU 约 10-15 分钟超 3 分钟限；--quick 1000 步 dev 2.209（sanity 通过） | ⏳ 核法：完整跑 `07_scaled_wavenet.py`（去 --quick）看最终评估验证集 |
| 9 | BN 3D 修复后输出均值≈0、std≈1、running_mean 保持 (C,) | 03 章验证代码块 | 实测 -0.000000 / 1.000386 / True | ✅ 对上 |
| 10 | FlattenConsecutive 形状链 (4,8,10)→(4,4,20)→(4,2,40)→(4,1,80)；view 与 cat 结果一致 | 02 章 + 脚本 04 | 实测全部一致（allclose=True） | ✅ 对上 |

## 只改 3 件事

1. **改数字**：03 章"~170K"→"~77K"（脚本 07 实测 76,579）；同表顺带把 02 章 "~2.02" 改为 "≈2.11（seed42/20K 步实测；~2.02 需更长训练或调参，见视频）"，并在对比表给 2.07 补 "≈2.1" 的措辞。数字是面试的命，错一个全盘被怀疑。
2. **补收尾**：02 章"WaveNet 完整架构"后加 5 行："最后一层 FC 后 tensor 是 (B,1,C)，Linear 后 (B,1,27)，训练用 `logits.view(-1, vocab_size)` 收成 (B,27)；推理采样同理"——正文因此能接住作业题 4 的思考题。
3. **软化措辞 + 给预期**：03 章"完全等价"→"感受野增长等价：FC+Linear 栈是 kernel=stride=2 的非重叠特例，WaveNet 原文为 stride=1 膨胀滑窗"；每个脚本 docstring 顶部加一行"CPU 参考时长（03/05 约 5-8 分钟，07 约 10-15 分钟，--quick 1 分钟）"。

## 最喜欢 3 处

1. 03 章 BN 3D bug 三段式：buggy mean shape (1,T,C) vs fixed (1,1,C) 的对照打印（脚本 06 实测一致）——"batch 语义在序列模型里是 B×T"这一句值得背下来。
2. 02 章"先消化 2 个，再消化 4 个，最后消化 8 个"的叙事 + 树状 ASCII 图：30 秒讲清 WaveNet 的全部动机。
3. 02 章 view vs cat 零拷贝对比（脚本 04 实测 allclose=True）：把工程直觉（stride 重解释 vs 分配新内存）变成了可验证的实验。

## 面试资产清单（S3 冲刺包）

- **30 秒形状链**：(B,8,10)→FC(2)→(B,4,20)→Linear→(B,4,68)→FC(2)→(B,2,136)→…→(B,1,68)→Linear→(B,1,27)→view(-1,27)。被问"WaveNet 怎么实现的"先画这条链。
- **参数量两笔账**：展平首层 80×200=16,000 vs 树状首层 20×200=4,000；小模型总账 22,397（27×10 + 20×68 + 68 + 136×68 + 68×2×68×2… 按层报）。⚠️ 别引用教程的 170K。
- **BN 3D bug 一分钟**：症状（loss 降但差）、根因（dim=0 使每个时间步独立归一化，"batch"统计被切碎）、修法（ndim==3 时 dim=(0,1)，running stats 恒 1D，eval 时 unsqueeze 两次广播）。
- **卷积等价视角（纠偏版）**：我们的栈 = kernel=stride=2 非重叠融合，感受野 2^L；WaveNet 原文 = stride=1 膨胀因果滑窗，感受野同速、保留时间分辨率。差异点是高频追问点。
- **WaveNet vs Attention**（节点 6 引子）：固定局部模式、O(n) 每层、感受野需 L 层堆叠 vs 动态全局配对、一层全局但 O(n²)。
- **train/eval 双模式**：忘记切 eval → running stats 没更新/推理用 batch 统计 → 输出不确定；广播链 running(C,)→(1,C)/(1,1,C)。
- **运行成本实拍**（可直接当工程素养谈资）：20K 步 CPU 5-8 分钟；多进程并行时 torch 默认 32 线程互相拖垮——OMP_NUM_THREADS 限 4 后恢复正常，说明小模型上线程超订反而更慢。
