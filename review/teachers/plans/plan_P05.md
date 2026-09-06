# Part 5 (WaveNet) 分章审计计划

> 审计人：T1 主教（预审） · 日期：2026-09-04
> 仓库（只读）：/home/admin02/Code/WorkSpace/makemore-tutorial
> 对象：courses/Part5_wavenet/tutorial/（README + 3 章，共 4 个 .md，约 430 行）、scripts/（7 个 .py，1685 行）、assignments/assignment_5/（实际文件为 README.md + wavenet_exercises.py + test_wavenet_exercises.py，**无 assignment.md**）
> roadmap 锚点：docs/course_roadmap_v3.md 节点 5（≈3-5h，阶段一收尾）
> 受众锚点：有 DL/ML 基础、数学不强、实操薄弱的面试备战人群
> 格式规范依据：REVIEW/00_master_plan.md §10（G1-G11）

---

## 一、C1 标题-内容对应表（承诺 vs 实际 vs 缺口）

### README.md（Part 5 导航页）
| README 承诺 | 实际小节 | 判定 |
|---|---|---|
| 01 = Embedding/Flatten/Sequential 模块化、BatchNorm 训练/推理坑 | 均覆盖 | ✓ |
| 02 = 为什么要层次化、FlattenConsecutive、Linear 支持多维输入 | 均覆盖，另有 view vs cat、block_size 3→8 两节 | 缺口 C1-0（轻）：导航表漏列这两节 |
| 03 = BatchNorm 3D bug、放大训练、卷积预览 | 均覆盖 | ✓ |
| "学完你能"6 条（含 loss < 2.0） | 与 3 章内容一一对应 | ✓ |

### 01_pytorchify.md（# 01 — PyTorch 化：让代码更优雅）
| 标题/承诺 | 实际小节 | 缺口 |
|---|---|---|
| 从 Part 3 的问题出发（字典管理层） | 字典写法示例 + if/elif 痛点 | C1-2（中）：叙事与 P3 教程实际不一致，见第六节 |
| Embedding / Flatten / Linear / Sequential 四个类 | 均给出代码+一句封装说明 | C1-1（轻）：L57 "封装了 `view(0, -1)` 操作"为笔误，实际是 `view(x.shape[0], -1)` |
| 用 Sequential 重构网络 | 完整 model 定义 + requires_grad 一行开启 | ✓ |
| BatchNorm 的训练/推理坑 | 双模式对照表 + 忘记切换后果 + 手动切换代码 | C1-3（中）：模型里用了 `BatchNorm1d(n_hidden)` 但全 Part 正文从未给出该类定义（P3 脚本版为 2D-only），照正文拼凑不可运行 |
| 代码参考 | 链接 01 脚本 | C1-4（轻）：只链 01 一个脚本；**02_fix_lr_plot.py 全 Part 无任何教程引用（孤儿脚本）**；本章训练 2000 步无初始 loss sanity check（G7 附带） |

### 02_wavenet_architecture.md（# 02 — WaveNet 架构：层次化融合）
| 标题/承诺 | 实际小节 | 缺口 |
|---|---|---|
| 动机：为什么不能直接展平 | 80 维直混 vs 逐步融合 | ✓ |
| 树状融合结构 | ASCII 树（8-gram→4-gram→bigram→c1..c8） | ✓（可图化，见 G4） |
| Linear 支持多维输入 | (4,8,10)@(10,20)→(4,8,20) 示例 | ✓ |
| FlattenConsecutive 层 | 代码 + shape 示例链 (B,8,10)→(B,4,20)→(B,2,40)→(B,1,80) | C1-6（中）：教程代码**无 T%n 整除断言**，而作业题 1 明确要求"T 不能整除时抛 AssertionError"、脚本 04 有 assert——教程/作业/脚本三处不一致 |
| WaveNet 完整架构 | 3 层 [FC(2)→Linear→BN→Tanh] + 输出层 | C1-7（中）：止于 `Linear(n_hidden, vocab_size)`，输出是 (B,1,27)；**(B,1,C) 如何变 (B,C)/logits.view(-1,C) 完全未讲**，而作业题 4 思考题正考这一点——形状链条尾段缺失 |
| 扩大上下文窗口 | block 3→8 对照表（~2.10→~2.02） | C1-5（严重）+ C1-9/C1-10（轻），见下方清单 |
| view vs cat | view 零拷贝 vs cat 拷贝 | C1-8（中）：教程把 strided cat（偶/奇交错）当 view 的等价替代只谈效率；两者**语义不同**（cat 版配对隔位字符），脚本 04 实跑 allclose=False 但教程未点破 |

02 章缺口清单：
- **C1-5（严重，叙事 vs 数字自相矛盾）**：L110 断言"直接展平 8 个字符不如层次化融合效果好"，但本章表格 flat block8 ≈2.02 **优于** 03 章表格 WaveNet 小模型 ≈2.07；且两者口径不一（flat：n_hidden=200、20K 步，脚本 03；wavenet 小：n_hidden=68、20K 步，脚本 05）。需全量实跑核数，统一对比口径或改写叙事。
- C1-9（轻）：上下文示例 "`...e`mma → `m`" 反引号错位、排版混乱；"....emma → n" 预测示例费解。
- C1-10（轻）：性能表 "~2.10/~2.02" 无脚注来源（哪个脚本、几步、何 seed），G6 数字锚点缺失。

### 03_training_and_bugs.md（# 03 — 训练与 Bug 修复）
| 标题/承诺 | 实际小节 | 缺口 |
|---|---|---|
| BatchNorm1D 的 3D Bug | 问题（dim=0 reduce 得 (1,T,C)）+ 修复（ndim 分支 dim=(0,1)）+ 验证代码 | ✓ 讲解到位；验证代码块为半截摘录未标注（G10-3） |
| 放大训练 | 小模型 vs 放大模型参数表 + 模型代码 + 性能对比表 | C1-11（严重）：参数量 "~170K" 存疑；C1-12（中）：对比表数字无前序锚点 |
| 卷积预览 | 传统卷积/膨胀卷积 ASCII 示意 + "完全等价"论断 + WaveNet 得名 | C1-13（中）："完全等价"过强，见下方清单 |
| 代码参考 | 链接 06/07 脚本 | ✓ |
| 课后作业 | 链接 Assignment 5 | ✓ |

03 章缺口清单：
- **C1-11（严重，事实核查）**：放大模型（n_embd=24, n_hidden=128）参数量手算 ≈76.5K（27×24 + 48×128 + 136×128 + 256×128 + 136×128 + 256×128 + 136×128 + 128×27 = 76,552 加 BN 2×3×128=768 → **约 76.5K**），教程写 "~170K" 疑为错误（22K 一档经核算无误）。全量审计必须跑 `07_scaled_wavenet.py` 用打印的 `total_params` 实证修正。
- C1-12（中，跨 Part 口径）：对比表 P2=2.10 / P3=2.07 在 P2/P3 的教程与脚本中**均无数字锚点**（grep 证实）；且"WaveNet 小 2.07 vs flat-8 上下文 2.02"未解释（训练预算/容量差异），削弱"层次化更优"主线。
- C1-13（中，理论）：FlattenConsecutive+Linear 是**不重叠 stride-2 分组**，dilated causal conv 是**重叠滑窗**——感受野树结构等价，算子不等价，"完全等价"需降格为"感受野/视角等价"并给差异说明（roadmap 节点 5 本就要求"等价视角"而非等价算子）。
- C1-14（轻）：lr 三段调度（0.1→0.05→0.01 @30K/40K）只存在于脚本 07，教程放大训练表未提。
- C1-15（轻）："验证"代码块示例输出未标注"示意"，且未演示 eval 分支（仅脚本 06 覆盖）。

**C1 缺口合计：16 处（严重 2：C1-5/11；中 7：C1-2/3/6/7/8/12/13；轻 7：C1-0/1/4/9/10/14/15）**
（归并口径：C1-5 与 C1-12 同根——"性能数字口径与叙事统一"，整改时按一件事处理，但分列 02/03 两章位置。）

附带（作业侧，非 C1 计数）：assignment_5/README.md L38 文件结构图写 `assignment.md  # 本文件`，实际文件名是 README.md——与 plan_P02 发现的 P2 同款问题，归 G10/一致性清单。

---

## 二、章级学生单元划分与难度

按 master plan v1.3 细则第 1 条：P5 教程约 430 行（<2K），**学生单元 = 整个 Part**，3 名学生一次读完 3 章、报告按章分节。章级难度画像（供学生报告与 T2 排期参考）：

| 章 | 内容性质 | 难度 | 预计学生耗时 | 预判卡点 |
|---|---|---|---|---|
| 01 PyTorch 化 | 代码重构，无新数学 | ★☆☆ | 30-45 min | BatchNorm1d 类哪里来（C1-3）；training 标志手动切换 |
| 02 WaveNet 架构 | 新概念：3D 形状链、层次融合 | ★★☆ | 45-60 min | (B,1,C)→(B,C) 收尾（C1-7）；view/cat 语义差（C1-8）；作业题 1 的 assert 教程没教 |
| 03 训练与 Bug 修复 | bug 诊断 + 放大实验 + 卷积视角 | ★★☆ | 45-60 min | 为什么 dim=(0,1)；广播 unsqueeze 两次；~170K 数字与实跑不符会直接打击信任（C1-11） |

roadmap 预估 3-5h：教程 + 7 脚本（全量跑约 40-70 min CPU）+ 作业（4 基础 + 1 拓展 50K 步训练，CPU 上 20-40 min）。总尺度基本合理，无 O6 级偏差风险；脚本全量跑耗时占比偏高，见第四节档位建议。

---

## 三、本 Part 特有审计要点（全量章循环时逐项核验）

1. **PyTorch 化（Sequential 容器 / 自定义 Module）**
   - 核验 01 章 4 个类的 `__call__`/`parameters()` 接口与脚本 01 逐行一致（G10：函数签名与脚本一致）。
   - 核验 `model.layers[-1].weight *= 0.1` 的输出层缩放在教程中是否有交代（脚本有 L187-188，教程 01 章未提——整改候选）。
   - 演进链核验：P2 散装参数 → P3 类/字典混用 → P5 Sequential 容器；每步"为什么要这一步"是否有正文交代。
2. **FlattenConsecutive 与形状变化链**
   - 逐层 shape 表核验：(B,8,10)→FC(2)→(B,4,20)→Linear→(B,4,68)→…→(B,1,68)→输出 (B,1,27)；教程 vs 脚本 05 运行打印 vs 作业题 4 三方一致。
   - 教程 02 缺尾段收拢（C1-7），整改时补 `logits.view(-1, logits.shape[-1])` 或 squeeze 讲解。
   - 脚本 04 L43 注释说"view 成 (B, T//n, n, C) 然后 reshape"，实际代码是单步 view——注释/代码不符（脚本侧整改）。
   - 核验 n=2 的 view 拼接语义确为"相邻位置特征拼接"（[b,0,0:10]=x[b,0,:], [b,0,10:20]=x[b,1,:]），教程如补图须用此语义。
3. **层次化融合（树状结构）**
   - 树图（c1..c8 → bigram → 4-gram → 8-gram）与 3 层 FC(2) 的对应关系是否讲透（每层只融合相邻两个）。
   - 作业深度思考 Q1 的参数量论证（flat 80×200=16,000 vs wavenet 20×200=4,000）数值核验（数字本身正确）。
   - C1-5 矛盾的整改方向：要么同口径重训对比（同 n_hidden/同步数），要么把叙事改为"层次化以更少参数/更优扩展性逼近"。
4. **crop/blocks 概念**
   - 源头：原 notebook（makemore_part5_cnn1.ipynb）中 `context = context[1:] + [ix]  # crop and append` 即"裁剪滑动窗口"；block_size 即上下文长度。P5 三章正文把 build_dataset 当已知沿用但**从未解释 crop**——核验 P2 是否已讲；若 P2 未讲，P5 需补一句指引（跨 Part 衔接）。
   - 作业 README L31 数据路径 `../../data/names.txt`、脚本 `../../../data/names.txt` 两套相对路径基准不同（作业 cwd=assignment_5/，脚本 cwd=scripts/），实测两条都必须真实可达。
5. **卷积预览**
   - C1-13 的表述降格；ASCII 膨胀卷积示意建议图化（G4）。
   - 核验"感受野指数增长 1→2→4→8"与作业 Q2 提示（WaveNet O(n log n) vs Attention O(n²)）口径一致；O(n log n) 说法过简（层数 log，总量线性），整改时酌情加边界说明（W2：教结论更教边界）。

---

## 四、scripts 运行档位建议（7 个）

| 脚本 | 训练量 | CPU 预估时长 | 降级档 | 建议 |
|---|---|---|---|---|
| 01_pytorchify_layers.py | 2000 步（block3, nh=200） | ~20-40 s | 无需 | 学生全跑 |
| 02_fix_lr_plot.py | 20000 步 | ~3-5 min | ❌ 无 quick、无 flush | 加 G8 档（STEPS 环境变量）+ flush；学生用降级档即可看懂平滑逻辑 |
| 03_increase_context.py | 20000 步（block8） | ~4-6 min | ❌ 同上 | 同上；T2 完整档实跑核对 "~2.02"（G6/C3） |
| 04_flatten_consecutive.py | 无训练 | <1 s | — | 全跑；核 allclose=False 的打印与教程表述（C1-8） |
| 05_wavenet_architecture.py | 20000 步 | ~4-6 min | ❌ 无 quick、无 flush | 加 G8 档；T2 完整档核对 "~2.07" 与 ~22K 参数量 |
| 06_batchnorm_3d_fix.py | 无训练 | <1 s | — | 全跑；核 buggy/fixed mean shape 断言输出 |
| 07_scaled_wavenet.py | 50000 步 batch128 | ~20-40 min | ✅ `--quick`（1000 步，~1-2 min） | 学生验证用 `--quick`；T2 完整档跑一次核对 "~1.99" 与参数量（C1-11 实证）；**所有 print 补 flush=True** |

共性档位问题：02/03/05 三个 20K 步脚本无降级档且 print 无 flush（G8 附带违反，虽不在本次 G1/G2/G4/G10 计数内，须列入整改清单）。全部脚本固定 seed=42 ✓，数据路径经 REVIEW/data 软链可复用 ✓。

---

## 五、格式规范预检（G1/G2/G4/G10 违反位置清单）

机检与人工扫描结果（对 4 个教程 .md 全量执行）：

| 规范 | 违反数 | 位置清单 |
|---|---|---|
| G1 编号点换行 | **0** | README ①②③ 位于 ASCII 框图内各自成行，属排版图示不违规；正文编号点均独立成行 |
| G2 数学 LaTeX | **0** | check_latex.py 4 文件全过（本 Part 几乎无数学；注意全 Part 无 BN 归一化公式 x̂=(x−μ)/√(σ²+ε)，数学不强人群建议补行内公式——内容缺口非格式违规） |
| G4 能画则画 | **3** | ① 全 Part 教程 0 张插图，`courses/Part5_wavenet/images/cell011_output01.png` 为孤儿图片（原 notebook 遗留，无任何 md 引用，整改：删除或转为正文插图）；② 02 章树状融合/膨胀卷积仅 ASCII，应图化；③ 02/03 章两处性能对比表适合配柱状图 + G5 逐格数值表 |
| G10 正文代码可拼凑运行 | **4** | ① 01/02 章模型代码使用 `BatchNorm1d` 但全 Part 未定义该类（C1-3，拼凑断裂点）；② 01 章 L57 `view(0, -1)` 笔误（照抄即错，C1-1）；③ 03 章 BN 修复代码块为半截函数（截断于 `xvar = ...` 行，无闭合、未标"摘录"）；④ assignment_5/README.md 文件结构自引 `assignment.md` 名不副实（实为 README.md） |

---

## 六、跨 Part 一致性

1. **与 Part 3 的容器化改造衔接**
   - P5-01 开场称"Part 3 的深层网络用**字典**管理层"，但 P3 教程 03_deep_network.md L78 明确：教程用**类**、脚本用字典，并已注明两种写法等价。P5 叙事需改为"P3 脚本的字典写法"或承认"P3 教程已用类"，避免学生回头对不上（C1-2）。
   - 演进链完整性：P3 教程类的 BatchNorm1d（2D）→ P5-03 修复为 2D/3D 双模。P5 未以"从 P3 版本 diff"方式呈现增量大；整改时建议 03 章开头一句"在 P3 版本上只改 __call__ 的 reduce 维度"。
   - P5-01 的训练/推理切换代码与 P3 脚本写法（hasattr(layer,'training')）一致 ✓。
2. **与 Part 4 的衔接**
   - P4 为手写反传专题（自建组件 + requires_grad 手动管理），P5 未引用 P4，亦无直接冲突；唯 P5 全 Part 沿用"手动 parameters 列表 + p.data -= lr*grad"训练循环，属 P3 之前风格，P5-01 已自洽。无整改项，复验时确认即可。
3. **性能对比数字口径（重点）**
   - P5-03 对比表引用 P2=2.10、P3=2.07：**P2/P3 的教程与脚本中均无这两个数字的锚点**（grep 证实），属跨 Part 悬空引用。整改：a) 实测建立统一口径（seed/步数/超参注明），b) 回写 P2/P3 各自的最终 dev loss 锚点或在 P5 加脚注"来源：本次复现运行"。
   - Part 内部矛盾 C1-5（flat8=2.02 vs wavenet小=2.07）必须与上条一并处理，产出一张"统一口径性能总表"（MLP-3 / BN-3 / flat-8 / WaveNet-小 / WaveNet-放大），教程与脚本 07 尾部 f-string 同步（脚本尾部硬编码对比数字属 G6 风险点，实跑 diff 后统一）。
4. **roadmap O5 抽查**
   - 节点 5 描述"脚本 01-07：PyTorch 化分层 → **学习率曲线修复** → 上下文加长 → …"：脚本 02 实际内容是 **loss 曲线平滑**（view(-1,window).mean），名字 fix_lr_plot 与 roadmap"学习率曲线"双重误导——roadmap 措辞与脚本名/内容需三选一统一（登记给 O5 横向一致性）。
   - 节点 5 的作业描述（FlattenConsecutive/WaveNet 组装/BN 3D/shape 流转，目标 loss<2.0）与作业实际 4+1 题完全对应 ✓；论文指引（WaveNet 图 1）与 03 章卷积预览衔接 ✓。

---

## 七、整改优先级摘要（供 T2 任务书引用）

1. P0 级（事实/自洽）：C1-11（参数量 ~170K 实证修正）、C1-5+C1-12（性能口径统一与叙事修正，含脚本 07 尾部硬编码）。
2. P1 级：C1-3（BatchNorm1d 定义/来源）、C1-6（assert 三处一致）、C1-7（形状链尾段）、C1-8（view/cat 语义）、C1-13（"完全等价"降格）、G10 四处、G8 档位三脚本。
3. P2 级：C1-0/1/4/9/10/14/15、G4 三处图化、孤儿图片处置、孤儿脚本 02 的教程挂载或 roadmap 措辞统一、assignment README 文件结构自引。
