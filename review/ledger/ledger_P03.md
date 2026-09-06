# P03（Part 3 训练诊断与 BatchNorm）整改台账

> 整改人：T2｜日期：2026-09-05｜依据：plan_P03.md + 三学生报告（S1_math / S2_hands / S3_interview）+ T0 必修清单（10 项）
> 状态取值：fixed（已修复并验证）/ disputed（留 T0 裁量，本次不动）
> 来源：S1/S2/S3=学生报告；PLN=T1 主教计划（含 18 个 C1 缺口编号）；任务=T0 必修清单
> 关键实测：题1 loss=26.0063（seed 2147483647）/ 脚本01 口径=26.7891 / 脚本05 结构初始 loss=3.9119 / PyTorch running_var 存无偏 EMA（实测 True）

## P0（必修）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|---|---|---|---|---|---|---|---|
| P03-C01-S1S2S3-01 | P0 | fixed | S1🔴3+S2🟡4+S3 K4/硬数字#4；PLN 3.8；任务必修1 | 作业题 1 三方矛盾：README 预期 loss≈3.7~4.0，实测 26.0063，测试断言 >10（注释 ≈26.8） | assignments/assignment_3/README.md 题1预期 | README 改实测口径：loss≈**26**（本配置实测 26.01，20~30 均正常），注明测试断言 >10、脚本 01 同量级口径 ≈26.8 | scratch verify_claims [1]=26.0063；脚本01同款配置复算=26.7891，与 test 注释吻合 |
| P03-C02-S1S2S3-02 | P0 | fixed | S2卡点1+S1🟡9+S3；PLN 3.1；任务必修2 | 教程 03 网络末层带 `BatchNorm1d(vocab_size)`，脚本 05/06 输出层为纯 Linear，三方面冲突 | tutorial/03_deep_network.md 6层结构+ASCII图 | 教程改纯 Linear（脚本不动），新增 ⚠️ 框：两种都合法、**本课程以 scripts/05/06 为准**、初始 loss 实测 ≈3.91 略高于 3.29 属预期 | verify_claims [5]=3.9119（同配置全训练集 forward） |
| P03-C03-S1S2-03 | P0 | fixed | S2卡点1+S1🟡9；PLN 3.2；任务必修2 | 「初始化技巧 `layers[-1].gamma *= 0.1`」在脚本 05 中不存在，照抄拼不出脚本行为 | tutorial/03 初始化技巧节 | 重写：注明该步骤属 Karpathy「输出层带 BN」结构；本课程脚本无此步且无 gamma 可缩；警告勿混搭两套结构 | 文字项；run05_full.log 全量复跑与 S1 记录逐位一致 |
| P03-C04-S1S2S3-04 | P0 | fixed | S1🟡10+S2卡点2+S3 K5；PLN 1.2/审计要点1；任务必修3 | 饱和阈值分裂：教程 0.97 vs 脚本 02/03/06 与作业题 4 的 0.99（还有 0.999 档） | tutorial/01 诊断代码/练习2、tutorial/03 工具1 | 教程统一 **0.99**，保留一处注释说明 Karpathy 演示用 0.97 的历史口径；「健康标准 <5%」标明按 0.99 口径 | grep 全教程无残留 0.97 断言代码；示例数字换 06 实跑（见 C06） |
| P03-C05-S2-05 | P0 | fixed | S2卡点3；任务必修3 | 「我们跑了 200000 步」与脚本 04/05 的 max_steps=20000 矛盾 | tutorial/02 坑3 | 改为「脚本 04/05 跑 20000 步（早已稳定）；Karpathy 视频里跑 200000 步」；30-50 步收敛补 EMA 时间常数依据（≈1/momentum=10 步，3~5τ） | 文字项，与 run04_full.log（20000 步）一致 |
| P03-C06-S1S2S3-06 | P0 | fixed | S2卡点12+S1🟢14+S3硬数字9/10；PLN 3.3；任务必修3 | 教程诊断示例数字（layer 2/5/8…、2.78% 等学生笔记值）与脚本输出格式、阈值均对不上 | tutorial/03 工具1/2/3/4 示例输出 | 全部替换为 scripts/06 实跑 stdout（tanh_N/linear_N.W 格式、0.99 口径、1000 步），注明类写法与字典写法对应关系；工具4 代码片段改用脚本 06 实际的 ud_ratio_history 写法 | run06_full.log 实跑采集，逐行引用 |
| P03-C07-S1S3-07 | P0 | fixed | S1🔴1+S3 K1/论文三细节；PLN 2.1/2.2；任务必修4 | σ 三种口径（论文有偏 ÷N / torch.var 默认无偏 ÷N−1 / PyTorch running_var 存无偏）正文只字未提，且教程内两套代码口径并存 | tutorial/02 BN原理+类实现 | 新增「σ 用 1/n 还是 1/(n-1)？」三口径表小节；BN 原理补 LaTeX 前向公式（含 ε 在方差上的说明）；类实现与内联代码统一 `unbiased=False` | 实测 running_var==0.9+0.1·var(unbiased=True) 为 True、biased 为 False（verify_claims [2]） |
| P03-C08-S1S3-08 | P0 | fixed | S1🔴2+S3；任务必修5 | BN 反向传播推导完全缺席，且无 Part 4 衔接说明 | tutorial/02 新增小节 | 新增「预习：γ 和 β 的梯度其实很好推」（∂L/∂β=Σg、∂L/∂γ=Σg·x̂ + 平移/缩放直觉），显式指认 Part 4 · 02_forward_and_backward.md Step 12 与 03_simplified_and_training.md 一行公式为正式推导落点 | 链接目标 Part4 两节已核实存在（grep） |
| P03-C09-S1S3-09 | P0 | fixed | S1🟡8+S3 K2/默写②；PLN 1.4/审计要点2；任务必修6 | gain=5/3 无任何推导来源（只在练习提示），fan_in/fan_out 首次出现无定义 | tutorial/01 Kaiming 节 | 新增「5/3 是怎么来的？」：方差守恒 Var[y]=fan_in·σ_W²·σ_x² 的 LaTeX 推导 + tanh std≈0.63 补偿直觉 + ReLU 二阶矩口径 + 一行验证代码；fan_in/fan_out 就地定义 | verify_claims [4]：std(tanh)=0.6277、1/std=1.593、calculate_gain=1.6667 |
| P03-C10-S1-10 | P0 | fixed | PLN 1.5（事实错误）；T1 裁定；任务必修7 | 练习 3 题干方向反了：「为什么 tanh 的 gain 比 ReLU 的 √2 小」——5/3≈1.67 > √2≈1.41 | tutorial/01 练习3 | 题干改为「为什么 tanh 的 gain 反而比 ReLU **大**」，提示同步改为可自洽的压缩率口径（tanh std≈0.63 / ReLU 二阶矩减半 → √2） | 与 C09 新增推导、S1 实测数字一致 |
| P03-C11-S1S2-11 | P0 | fixed | S2卡点8（路径）；PLN G4-3；任务必修8 | 教程片段 savefig 到 ../images/ 与脚本实际保存目录（scripts/）不一致，照抄会写错位置 | tutorial/01 可视化片段 | 片段改存当前目录（单层直方图），注明仓库 images/ 下 cell 系列为 notebook 实跑存档 | 文字项；镜像脚本烟测无文件写向 images/ |

## P1（裁量修复）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|---|---|---|---|---|---|---|---|
| P03-C12-S1S3-12 | P1 | fixed | S1🟡5；PLN 2.3；任务必修8 | running 统计量两套约定并存：内联版存 std、类实现/脚本存 var，混用必错 | tutorial/02 内联+推理代码 | 统一存 **var**（bnvar_running），推理改 `torch.sqrt(bnvar_running+1e-5)`；加 ⚠️ 框说明视频 std 口径与本教程选择理由 | 02 diff；口径与 scripts/04-06、PyTorch 一致 |
| P03-C13-S3-13 | P1 | fixed | S3 K6 | 教程类实现与内联版 gamma/beta 均未设 requires_grad=True，照抄 backward 报错（作业题 3 硬性要求） | tutorial/02 类实现+MLP使用 | `torch.ones/zeros(dim, requires_grad=True)`，内联 bngain/bnbias 同步 | 与参考答案 batchnorm_exercises.py 写法一致，pytest 全绿 |
| P03-C14-S1-14 | P1 | fixed | S1🟢13；PLN 1.1 | `b2 = torch.randn(vocab_size) * 0` 怪写法；修复片段未交代 b1 | tutorial/01 怎么修 | 改 `torch.zeros(vocab_size)`；新增说明：本课只修输出层，b1 保持随机（脚本 01 口径），BN 之后会被 β 吸收 | 文字项 |
| P03-C15-S1S3-15 | P1 | fixed | PLN 2.4 | b1 三种处理（内联不写 / 脚本04 保留 / 作业题5 置零）无统一说法 | tutorial/02 BN 位置节 | 新增 💡 框：b1 会被 β 吸收属冗余，三种都合法，作业推荐 zeros | 文字项 |
| P03-C16-S1-16 | P1 | fixed | S1🟡4；任务必修8 | γ=2,β=3 示意图数字错（画成 -1..5 共 7 个刻度，对不上 y=2x+3） | tutorial/02 第二步 | 改为输入 -2,-1,0,1,2 → 输出 -1,1,3,5,7，并标明 y=2x̂+3 | 手算核对 |
| P03-C17-S1-17 | P1 | fixed | S1🟡7；S3 硬数字；任务必修8 | 「Layer 2: std=5.3 → Layer 3: std=28.0」来源不明（std/var 混排） | tutorial/01 每层方差变化 | 改理论链：1.0 → √30≈5.5 → 30·√30≈164（fan_in=30），附脚本 03 实测 5.57≈5.48，并注明旧数字混排更正 | run03_full.log + S1 实测 5.569 |
| P03-C18-S3-18 | P1 | fixed | PLN 3.4 | 工具 3 配图（cell017）实为权重梯度直方图，图注却写「参数梯度比率」，比率本身无图 | tutorial/03 工具3 | 修正图注为「权重梯度分布直方图」；嵌入新柱状图 03_grad_data_ratio.png（脚本 06 实跑数据 dump 重绘，与正文表格逐位一致） | dump06.json：log10 = -2.41/-1.99/-1.82/-1.85/-1.84/-1.80/-1.29 |
| P03-C19-S1-19 | P1 | fixed | PLN 3.5 | 工具 3 健康标准只有「各层接近、无极端值」，无数量级参考 | tutorial/03 工具3+总结表 | 量化：各层 log10(grad/data) 同数量级（实测 -2.4~-1.3）、无单层极端偏离；总结表同步 | 同上 |
| P03-C20-S1-20 | P1 | fixed | S1🟡6 | 坑 2 表述自相矛盾：先说「除 0 → NaN」，解药却是「加 eps」 | tutorial/02 坑2 | 重写：eps 只防除零（分母 ≥√1e-5≈0.003，输出放大约 300 倍，爆炸而非 NaN），根因是 1 样本统计无意义；补 PyTorch 对 batch=1 抛 ValueError | 文字项 |
| P03-C21-S2-21 | P1 | fixed | PLN 3.7 | README 文件结构自称 assignment.md，实际文件为 README.md；测试运行方式缺 pytest | assignments/assignment_3/README.md | 改 README.md + 补 `python -m pytest … -v` 用法 | 文字项 |
| P03-C22-S2-22 | P1 | fixed | S2卡点5/README-测试口径 | 题 5 承诺「steps=200000、dev<2.1」与测试实际（steps=10000、断言 <2.5）口径错位 | assignments/assignment_3/README.md 题5 | 双口径写明：测试口径 10k 步/<2.5（参考实现 batch=64 实测 ≈2.23，CPU 一两分钟）；200k 步/<2.1 为 Karpathy 口径自学挑战，不强制 | verify_claims [6]=2.2296@10000 步 |
| P03-C23-S3-23 | P1 | fixed | S3 硬数字#1 | ln(27)≈3.298 精度错（真值 3.2958，应报 3.296），作业 README/test/骨架共 10 处 | assignments/assignment_3/ 三文件 | 全量 3.298→3.296；test 仅动注释与打印，**断言零改动**；题 2 预期改 3.30~3.31（实测 3.3078） | diff 核对；修复后 pytest 5 passed |
| P03-C24-S2S3-24 | P1 | fixed | PLN 2.5/G4；S2 逐章评分 | 02 章（BN 主章）0 张图 | tutorial/02 + images/ | 新增 2 张实跑图：02_bn_standardization.png（标准化前后+γβ 三联分布，配 mean/var 数值表）与 02_bn_running_stats.png（running 收敛曲线）；均脚本 04 同款配置实跑 | make_figs 输出：pre 0.054/3.329 → norm 0.000/1.000 → γβ 0.010/1.009 |
| P03-C25-S1-25 | P1 | fixed | PLN 1.3/G4 | 01 章嵌入 cell015（深层网 5 条 Tanh 曲线）与本章单隐藏层语境错位 | tutorial/01 可视化 | 换实拍单层图 01_tanh_saturation_single.png（脚本 02 配置，±0.99 参考线），注明多层版本见 03 章；代码片段同步简化为单层版 | fig1 sat=65.05% 与 S1 实测 65.05% 逐位一致 |
| P03-C26-S1-26 | P1 | fixed | S3 硬数字#12 | 「标准化后大致 [-2,2]」缺严谨性（±2σ 仅约 95%） | tutorial/02 第一步 | 改「大致落在 [-2,2]（正态假设下约 95% 的样本）——大部分远离饱和区」 | 文字项 |
| P03-C27-S2-27 | P1 | fixed | PLN G10-1 | 「在 MLP 中使用」缺 import/g/Xb 等前置，需跨章拼凑 | tutorial/02 同名代码块 | 块首加最小上下文注释（F、g、n_embd/n_hidden/block_size、Xb/Yb 来源指引）；b2 怪写法顺带改 zeros | 文字项 |

## P2（轻量改进）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|---|---|---|---|---|---|---|---|
| P03-C28-S1S2-28 | P2 | fixed | PLN G8；S2 环境阻塞 | 脚本 03/04/05 全量 2-13 分钟、高载下易超时，无短程档 | scripts/01-06 | 统一 `STEPS` 环境变量短程档 + 周期打印 flush（默认档逐位不变：01/02=1000、03=10000、04/05=20000、06=1000）；docstring 注明烟测用法 | 全量档与 S1 记录逐位一致（03/04/05/06）；STEPS=100 六脚本烟测全过 |
| P03-C29-S1-29 | P2 | fixed | S1🟢11；PLN 审计要点5 | 脚本 03 为算 dev loss 把基线无谓重训 1 万步（`train_model(False)[1:]`） | scripts/03 | 复用首次训练返回的参数（同种子结果不变，耗时近乎减半）；脚本 05 死变量 `update_data_ratios` 一并删除 | run03_full.log 与 S1 记录一致；05 烟测 OK |
| P03-C30-S2S3-30 | P2 | fixed | S2核对表#5；PLN §6 | 教程未注明 seed=42（脚本）与 2147483647（作业）不能逐位对照 | tutorial/03 风格说明框 | 补「关于随机种子」说明：种子不同量级对照即可 | 文字项 |
| P03-C31-S2-31 | P2 | fixed | 对齐 Part2 体例 | tutorial/README 无脚本运行说明（STEPS/Agg） | tutorial/README.md | 新增「🛠️ 运行脚本」节（两条示例命令） | 文字项 |
| P03-C32-S2-32 | P2 | fixed | PLN G10-2 缓解 | 工具 4 教程片段用视频 notebook 的 `ud` 写法，脚本中无对应变量 | tutorial/03 工具4 | 已随 C06 改为脚本 06 实际 ud_ratio_history 写法（并入 C06 验证） | 06 实跑输出对照 |

## disputed（留 T0，本次未动）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 不动理由/建议 |
|---|---|---|---|---|---|---|
| P03-D01-PLN-D1 | P2 | disputed | PLN 3.6/§2 | README 无难度/时长星级（roadmap 节点 4 体例）；且 4-6h 总预算对 B+C+E 单元偏紧（实际 8-10h） | tutorial/README.md；docs/course_roadmap_v3.md | Part2 README 无星级表，Part3 单方面加会破坏体例一致性；属 roadmap 全局裁量，建议 T0 统一裁定（若加：单元 C/E ★★★、总时长改 8-10h） |
| P03-D02-PLN-D2 | P2 | disputed | PLN 审计要点4 | 脚本 06 诊断 2/3 用 `training=False` 的前向再 backward，采到 eval 路径梯度，与 Karpathy 训练模式采梯度口径不同 | scripts/06 L225 | 改动会变更脚本默认行为与既引数字（教程表格/图均按此口径）；已在 C06 标注「字典写法、实跑输出」来源。建议 T0 裁定是否切 training=True 并同步更新教程数字 |
| P03-D03-S3-D3 | P2 | disputed | S3 K3/K4（roadmap 部分） | 「BN 为什么有效」追问层（Santurkar 2018 平滑 loss landscape）与归一化家族坐标系（BN→LayerNorm→RMSNorm，为节点 7 铺路）正文零覆盖 | tutorial/02（未新增） | 属新增内容立项而非纠错，超出本次整改授权；σ 三口径（K1）已修，K3/K4 建议 T0 决定是否增设「进阶：BN 为什么有效 / 归一化家族」小节 |

计数：P0 11 条（全 fixed）/ P1 16 条（全 fixed）/ P2 5 条（全 fixed）/ disputed 3 条。
PLN 18 个 C1 缺口覆盖：1.1→C14、1.2→C04、1.3→C25、1.4→C09、1.5→C10、2.1→C07、2.2→C07、2.3→C12、2.4→C15、2.5→C24、3.1→C02、3.2→C03、3.3→C06、3.4→C18、3.5→C19、3.6→D01、3.7→C21、3.8→C01——18/18 逐条处理完毕。
