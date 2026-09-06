# 问题台账 · Part 5（WaveNet）· 全量批1

> 编号规则：P05-C{cc}-{SRC}-{nn}；cc∈{00=README,01,02,03,SC=scripts,AS=assignment_5}；SRC∈{S1,S2,S3,T,T1计划}
> 状态：fixed=已修复且教师侧验证；disputed=争议/跨 Part 待 T0 裁决（禁止 wontfix）
> 实测环境：CPU 32 核、torch 2.6.0、seed=42；完整档日志见 REVIEW/scratch/t2_P5/run*.log

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 | 复核 |
|------|--------|------|------|------|------|----------|----------|------|
| P05-C03-S1-01 / S2卡点D / S3卡点1 / T1(C1-11) | P0 | fixed | 三方命中+T1 判实 | 放大模型参数量写 "~170K"，实测 **76,579**（差 2.2 倍，不可辩护） | tutorial/03 参数表 | 改为"22,397（实测）/ **76,579（实测）**"，并新增「参数量怎么手算？」小节（逐层算式：Embedding+Σ Linear+2C×BN+输出层） | 07 完整档实测打印 76,579；教程算式逐层合计 76,579；05 实测 22,397 与算式一致 | 待用户 |
| P05-C02-S2-01 / S3卡点2 / T1(C1-5/C1-12) | P0 | fixed | 双命中+教师 | 性能叙事与数据自相矛盾："直接展平 8 个字符不如层次化融合"，但实测 flat-8 dev 2.1064 ≈ WaveNet 小 2.0957；"~2.02/~2.10/~2.07" 无锚点不可复现 | tutorial/02 扩大上下文节、tutorial/03 对比表、scripts/03/05/07 docstring 与尾部总结 | 叙事改诚实版："本预算下展平不吃亏（2.106 vs 2.096 打平），层次化的收益是首层参数 1/4 + 结构先验，loss 优势在放大后显现"；全 Part 性能数字替换为本机实测并逐个标注脚本+档位锚点；视频值（2.10/2.07/1.99）保留但显式注明"该步数档不可直接复现" | 完整档实跑：flat-8=2.1064（03）、WaveNet 小=2.0957（05）、放大=2.0004/test 1.9948（07）、P2 最小=2.3710（P2 ledger）、P3 深层=2.1625（本轮实跑）；03/05 默认档日志与原版逐位一致 | 待用户 |
| P05-C03-S3-03 / T1 | P0 | fixed | S3🔴+教师 | "验证 loss 首次降到 2.0 以下"过度承诺：实测 dev 2.0004（未严格 <2.0，test 1.9948） | tutorial/03 性能段、README 学完你能 | 改为"压到 ≈2.00（dev 2.0004 / test 1.9948）；稳定 <2.0 需更长训练或作业题 5 调参" | 07 完整 50K 步日志 run07_full.log | 待用户 |
| P05-C02-S2-02 | P0 | fixed | S2🔴（照抄崩） | 教程 02 的 FlattenConsecutive 缺 `parameters()`，照抄进 Sequential 后 `model.parameters()` 直接 AttributeError | tutorial/02 FlattenConsecutive 节 | 补全 `parameters()` 返回 `[]`，并加一句"为什么必须存在"（Sequential 逐层收集参数） | 与脚本 04/05/07 三处定义一致；学生可照抄跑通 | 待用户 |
| P05-CSC-S1-02 | P0 | fixed | S1🔴（黑屏） | 脚本 07 `--quick`（1000 步）打印条件 `%5000` 永不触发，训练阶段 0 行 loss 输出 | scripts/07 | 打印间隔改 `log_every = 5000 if max_steps>=5000 else max(1, max_steps//5)`；quick/短程档每 200 步打 train+dev loss；默认档仍为 5000 步 | `--quick` 实跑 5 行训练 loss（dev 2.2087 与 S2 实测一致）；默认档间隔 5000 不变 | 待用户 |
| P05-CSC-S1-03 / S2卡点5 | P1 | fixed | 双命中 | 脚本 04 演示 4 打印"结果相等: True"，注释却说"view 和 cat 的元素顺序不一定相同"——输出与注释自相矛盾（该写法严格相等） | scripts/04 L115-116 | 注释改为"对该写法两者严格相等（可 torch.equal 复核）；差别在代价：view 零拷贝，cat 要复制" | 实测 `torch.allclose=True`；教程 02 view vs cat 节同步改写并补"cat 配错切片顺序则语义不同"的边界 | 待用户 |
| P05-CSC-S1-04 / S2卡点6 | P1 | fixed | 双命中 | 脚本 06 `BatchNorm1dBuggy` 注释与现象矛盾：真实缺陷是 training 分支**从不更新 running stats**（且 eval 时 (1,C) 碰巧广播成功不报错），注释却说"可能被 3D squeeze 破坏" | scripts/06 类 docstring 与 L149 注释 | docstring 重写为两处缺陷（reduce 维度错 + 从不更新 stats + eval 碰巧广播的隐蔽后果）；L149 注释同步改正 | 与 S1 实验一致（buggy eval 不报错、stats 停在 0/1）；stdout 逐位不变 | 待用户 |
| P05-CSC-S1-05 | P1 | fixed | S1🔴 | 脚本 05 架构注释写 "Flatten → Linear(68,27) → 输出"，实际代码无 Flatten 层（Linear 直接作用在 (B,1,68)） | scripts/05 层次化结构注释 | 注释改为完整形状链 "(B,1,136)→Linear→(B,1,68)；Linear(68,27)→(B,1,27)，训练时 view(-1,27) 收成 (B,27)"，注明无需 Flatten 的原因 | 与 05 实测逐层形状打印一致 | 待用户 |
| P05-C01-S1-06 | P1 | fixed | S1🔴 | 教程 01 "封装了 view(0, -1) 操作"笔误（照抄即错），实际是 `view(x.shape[0], -1)` | tutorial/01 Flatten 节 | 改为 `view(x.shape[0], -1)` 并说明"保留 batch 维压平其余" | 与脚本 01 L147 一致 | 待用户 |
| P05-C01-T-01 / T1(C1-3)/G10 | P1 | fixed | T1 计划（S2 连带） | `BatchNorm1d`/`Tanh` 全 Part 正文未定义即用，照正文拼凑不可运行（G10） | tutorial/01 新增小节 | 新增「BatchNorm1d 和 Tanh 从哪来？」：完整 2D 类定义（与脚本 01 逐行一致）+ 归一化公式 + `unbiased=False` 说明，注明 03 章将升级 3D | check_latex 0 问题；定义与脚本 01 L86-126 一致 | 待用户 |
| P05-C03-T-02 / G10-3 | P1 | fixed | T1 计划 | 03 章 BN 修复代码块为半截摘录（截断于 xvar，无闭合、未标注） | tutorial/03 修复节 | 补全为完整 `__call__`（含 running stats 更新与 eval 双分支），注明"与 06 脚本一致、可直接替换" | 代码与 06 脚本行为一致；06 秒级实跑通过 | 待用户 |
| P05-C02-S3-04 / T1(C1-7) | P1 | fixed | S3🔴（正文答不出自己的作业） | (B,1,C)→(B,C) 收尾正文零覆盖，而作业题 4 思考题正考它 | tutorial/02 新增小节 | 新增「形状链：从 (B,8,27) 到 (B,27)」逐层表 + `logits.view(-1, logits.shape[-1])` 收尾讲解（训练与采样两处用法） | 形状链与脚本 05 实测打印逐行一致 | 待用户 |
| P05-C02-T-03 / T1(C1-6) | P1 | fixed | T1 计划（S1 连带） | 整除断言三处不一致：作业题 1 要求 AssertionError、脚本 04 有 assert、教程 02 与脚本 05/07 没有 | tutorial/02、scripts/05、scripts/07 | 教程 02 代码补 assert；05/07 的 FlattenConsecutive 补同一 assert；教程注明"这也是作业题 1 的要求" | 四处（教程/04/05/07/作业）一致；05/07 默认档数值输出不变 | 待用户 |
| P05-C03-T-04 / T1(C1-13) | P1 | fixed | T1 计划（S3 连带） | "本质上等价/完全等价 Dilated Causal Convolution"措辞过强：非重叠 stride-2 分组 ≠ 重叠滑窗膨胀卷积 | tutorial/03 卷积预览节 | 降格为"感受野等价（1→2→4→8），算子不等价"，补两点对比（T 减半折叠 vs 保留时间分辨率的滑窗）及适用场景 | 表述与 roadmap"等价视角"口径一致 | 待用户 |
| P05-CSC-T-05 / G8 | P1 | fixed | T1 计划 + S2/S3 工程预期 | 02/03/05 三个 20K 步脚本无短程档且 print 全缓冲，慢机/重定向下黑屏 | scripts/01/02/03/05/07 | 全部加 `STEPS` 环境变量档（默认档行为不变）+ `print=functools.partial(print, flush=True)`；07 保留 `--quick`；README 新增「如何运行脚本」节（用法+CPU 时长预期+OMP 建议） | STEPS=500/1000/2000 实跑全部出数；01/04/06 默认档输出与原版逐位一致；03 默认档逐位一致；05 默认档除 1.8428→1.8427 浮点噪声（线程数不同所致，同 P02-D4）外一致 | 待用户 |
| P05-C02-S1-07 / S3卡点3 | P1 | fixed | 双命中 | 教程 01 "自带 Kaiming 初始化"未解释 gain：randn/fan_in**0.5 是 gain=1 He normal，而 Part 3 用 gain=5/3，畏难学生无所适从 | tutorial/01 Linear 节、assignment_5/README 题 2 | 教程补一段"gain=1 vs 5/3 何时用哪个"（后面跟 BN → gain=1 即可）；作业题 2 补同款括号说明 | 与 S1 实测 std=1/√200 一致；两处口径统一 | 待用户 |
| P05-C00-T-06 / T1(C1-0) | P2 | fixed | T1 计划 | README 导航表漏列 02 章 view vs cat、上下文 3→8 两小节 | tutorial/README 导航表 | 02/03 行内容列补全（view vs cat、上下文 3→8、参数量手算、loss 平滑） | — | 待用户 |
| P05-C01-T-07 / T1(C1-2) | P2 | fixed | T1 计划（S2 连带） | "Part 3 的深层网络用字典管理层"与 P3 教程实际不符（P3 教程用类、脚本用字典） | tutorial/01 开头、scripts/01 docstring | 改为"P3 教程已用类；其配套脚本 05 用字典手动管理"，脚本 01 docstring 同步 | 与 P3 教程/脚本核对一致 | 待用户 |
| P05-C01-T-08 / T1(C1-4) | P2 | fixed | T1 计划（S2 连带） | 孤儿脚本 02_fix_lr_plot.py 全 Part 无教程引用；loss 平滑内容点在教程缺失 | tutorial/01 新增「延伸：把噪声 loss 曲线变平滑」小节 + 代码参考节收编 | 新增 view(-1,1000).mean(1) 平滑小节（脚本名历史遗留也注明），代码参考补链接与实测锚点 | 脚本 02 STEPS=2000 实跑通过（dev 2.1701 锚点写入 docstring） | 待用户 |
| P05-C00-T-09 / G4 | P2 | fixed | T1 计划（全 Part 零图） | 全 Part 教程 0 张插图 | images/ + tutorial/02、03 | 实测数据生成 2 张新图：`wavenet_loss_comparison.png`（五模型 dev loss 柱状图，含 2.0 参考线）、`wavenet_tree_fusion.png`（树状融合结构图）；均为英文图内标注 + 英中双语图注 + 正文数值表 | 图由 scratch/t2_P5/make_figures.py 用本轮实测数据生成，可复现 | 待用户 |
| P05-C03-T-10 / G4 | P2 | fixed | T1 计划 | 孤儿图片 cell011_output01.png（原 notebook 遗留，无任何引用） | tutorial/03 放大训练节 | 收编为"视频原配置 loss 曲线（仅示意趋势、无坐标轴标注）"正文插图，明示其局限 | 图片被教程引用，不再孤儿 | 待用户 |
| P05-C03-T-11 / T1(C1-14) | P2 | fixed | T1 计划 | 放大训练的 lr 三段调度只存在于脚本 07，教程只字未提 | tutorial/03 参数表+代码块后 | 参数表加"学习率 0.1→0.05@30K→0.01@40K"行，正文补一句 | 与脚本 07 一致 | 待用户 |
| P05-C03-T-12 / T1(C1-15) | P2 | fixed | T1 计划 | 03 章"验证"代码块未标出处、未演示 eval 分支 | tutorial/03 验证节 | 标注"出自 06 脚本、可独立运行"，补 eval 演示（冻结+广播不报错）与 buggy 版隐蔽后果对照 | 06 脚本实测一致（均值 -0.000000、std 1.000386、running 1D） | 待用户 |
| P05-C02-S1-08 | P2 | fixed | S1🟡（C1-9） | 上下文示例 "`...e`mma → `m`" 反引号错位、"....emma → n" 预测目标错误（emma 后是结束符） | tutorial/02 对比表 | 重写为 `emm → a` / `.....emm → a`（左补起始符，预测目标正确），表格增"脚本锚点"列 | 与 build_dataset 滑窗语义核对一致 | 待用户 |
| P05-CAS-S2-03 | P2 | fixed | S2🔴 + T1 | 作业题 2 隐性 gap：测试要求 2D 输出 (4,27)，但教程/骨架结构末端是 (B,1,27)，骨架 TODO 无任何提示（S2 靠预读测试才过关） | assignment_5/README 题 2、wavenet_exercises.py | README 题 2 显式写出"(B,1,C)→(B,C) 收拢"要求；骨架补 `Flatten` 类与 TODO 第 6 步提示 | 参考答案（末尾 Flatten）+ 骨架 pytest 5/5 | 待用户 |
| P05-CAS-T-13 | P2 | fixed | T1 计划（P2 同款） | 作业 README 文件结构自引 `assignment.md`，实际文件名 README.md | assignment_5/README.md | 改为 README.md | — | 待用户 |
| P05-C00-T-14 / G8 文档 | P2 | fixed | 教师（S2/S3 工程预期） | 脚本无任何运行时长预期；07 完整版 50K 步 CPU >15 分钟无提示 | scripts/01/02/03/05/07 docstring、tutorial/README | 各脚本 docstring 顶部加 CPU 参考时长；README「如何运行脚本」节汇总 | README 表格给出 1 min ~ 25 min 分档 | 待用户 |

## Disputed（跨 Part / 待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P05-D1 / T1(O5) | roadmap 节点 5 写"学习率曲线修复"，脚本 02 实际是 loss 曲线平滑、名字 fix_lr_plot 双重误导。本轮已在脚本 docstring 与教程注明"平滑的是 loss 不是 lr"，但 roadmap 措辞与脚本名二选一改名属跨 Part 文件 | 建议改名 `02_smooth_loss_plot.py` 或 roadmap 措辞改"loss 曲线平滑"；归 O5 横向一致性 |
| P05-D2 | P5 对比表引用的"MLP ~2.10 / 深层 BN ~2.07"为视频原配置（更长训练/更大模型）数字，本仓库 P2/P3 教程默认档只能复现 2.37/2.16 级别。本轮已在 P5 教程显式双列"实测 vs 视频参考"并注明不可直接复现；是否回写 P2/P3 各自教程补"视频参考值 vs 本仓库实测值"对照，属跨 Part 编辑 | 建议 T0 统一裁决：P2/P3/P5 三处性能叙事采用同一"双列"格式 |
| P05-D3 | `unbiased=False`（有偏方差）与 PyTorch 官方 BN 更新 running_var 用无偏估计的微妙差异，本轮已在教程 03 补一句说明；是否在 P3（BN 首次实现处）也展开，属 P3 篇幅取舍 | 建议 P3 整改批次同样补一句，保持口径一致 |
| P05-D4 | 05 脚本默认档 step 15000 loss 原版 1.8428 vs 新版 1.8427（±0.0001）：多线程浮点归约非确定性（原版跑于 OMP=6、验证跑于 OMP=8），非代码逻辑变化。教程统一用"数值随线程数/硬件有 ±0.01 浮动"措辞消化 | 无需行动；与 P02-D4 同源 |

## 本轮实测锚点（seed=42，CPU，默认档）

| 模型 | 脚本 | 档位 | dev loss | 日志 |
|------|------|------|----------|------|
| P2 MLP 最小（n_embd=2, n_hidden=100, block3） | Part2/05 | 20K | 2.3710 | P02 ledger（沿用） |
| P3 深层 BN（n_hidden=100, block3） | Part3/05 | 20K | 2.1625 | runP3_05_full.log |
| 展平 MLP（n_hidden=200, block8） | Part5/03 | 20K | 2.1064（test 2.1061） | run03_full.log |
| WaveNet 小（10/68, block8） | Part5/05 | 20K | 2.0957 | run05_full.log |
| WaveNet 放大（24/128, block8） | Part5/07 | 50K | **2.0004**（test 1.9948，train 1.7579） | run07_full.log |
| WaveNet 放大参数量 | Part5/07 | — | **76,579**（教程旧值 ~170K 错误） | run07_full.log |
| WaveNet 小参数量 | Part5/05 | — | 22,397 | run05_full.log |
