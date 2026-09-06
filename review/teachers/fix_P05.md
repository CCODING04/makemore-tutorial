# T2 整改报告 · Part 5（WaveNet）

> 整改人：T2 整改教师 · 日期：2026-09-05
> 输入：plan_P05.md + S1/S2/S3 三份 report_P05.md + REPO 教程/脚本/作业
> 原则：REPO 只读；全部修改写入 REVIEW 镜像；数字全部本机实跑；禁止 wontfix，争议记 disputed（4 条）
> 台账：ledger_P05.md（28 条：P0×5、P1×11、P2×12；另 disputed×4）

---

## 一、镜像产出清单

### tutorial（4 个 md，全部改动，check_latex 全 0 问题）
| 文件 | 主要改动 |
|------|---------|
| courses/Part5_wavenet/tutorial/README.md | 导航表补漏列小节；新增「如何运行脚本」节（STEPS 用法/CPU 时长/OMP 建议）；"loss < 2.0" 承诺改"≈2.17 压到 ≈2.0"（与实测一致） |
| courses/Part5_wavenet/tutorial/01_pytorchify.md | 字典叙事改为"P3 教程用类、脚本用字典"；view(0,-1) 笔误修正；Kaiming gain=1 vs 5/3 说明；**新增 BatchNorm1d/Tanh 完整定义小节**（G10）；weight×0.1 技巧补充；**新增 loss 曲线平滑延伸小节**（收编孤儿脚本 02）；代码参考补 02 脚本 |
| courses/Part5_wavenet/tutorial/02_wavenet_architecture.md | FlattenConsecutive 补 parameters() 与整除断言（照抄崩修复）；新增「形状链 (B,8,27)→(B,27)」小节 + (B,1,C) 收尾讲解；性能表改实测锚点（2.17/2.11）+ 诚实版"展平不吃亏"叙事；上下文示例反引号修正；view vs cat 补"严格相等、差在代价"与配错切片边界；树状融合配图 |
| courses/Part5_wavenet/tutorial/03_training_and_bugs.md | **~170K → 76,579（实测）** + 新增「参数量怎么手算？」逐层算式小节；性能对比表重建为五模型同口径实测表（含脚本锚点列 + 视频参考值注明不可复现）；"首次 <2.0" 改 "dev 2.0004/test 1.9948"；BN 修复代码块补全（含 eval 分支）；验证代码块标注出处 + eval 演示 + buggy 隐蔽后果；"完全等价"降格为"感受野等价、算子不等价"；lr 三段调度入表；loss 对比图 + 原 notebook 曲线收编引用 |

### scripts（7 个 py，全部改动）
| 文件 | 主要改动 |
|------|---------|
| 01_pytorchify_layers.py | STEPS 档 + flush；docstring 修正字典叙事、补时长预期 |
| 02_fix_lr_plot.py | STEPS 档 + flush + 平滑窗口自适应；docstring 注明"平滑的是 loss 不是 lr"（历史名遗留）+ 实测锚点 2.17 |
| 03_increase_context.py | STEPS 档 + flush；docstring 与尾部总结改实测数字（≈2.17→≈2.11）与诚实叙事 |
| 04_flatten_consecutive.py | 修三处注释（docstring "view+transpose"、L43 "两步 reshape"、演示 4 "顺序不一定相同"矛盾注释）；**stdout 逐位不变** |
| 05_wavenet_architecture.py | STEPS 档 + flush；FlattenConsecutive 补整除断言；"Flatten → Linear" 假注释改完整形状链；docstring/尾部数字改实测（22,397 / ≈2.10） |
| 06_batchnorm_3d_fix.py | Buggy 类 docstring 与注释重写（真实缺陷=从不更新 running stats + eval 碰巧广播）；**stdout 逐位不变** |
| 07_scaled_wavenet.py | **--quick 打印 bug 修复**（log_every 分档，训练中每 200 步出 train/dev loss）；STEPS 档 + flush；FlattenConsecutive 补整除断言；尾部硬编码对比表改实测锚点 |

### assignments/assignment_5
| 文件 | 主要改动 |
|------|---------|
| README.md | 文件结构 assignment.md→README.md；题 2 补 2D 输出要求与收拢方法；Kaiming gain=1 说明 |
| wavenet_exercises.py（骨架） | 补 Flatten 类；build_wavenet TODO 补第 6 步"收拢 (B,1,C)"提示（消除 S2 卡点 1 的隐性 gap） |

### images（G4：0 图 → 3 图）
| 文件 | 说明 |
|------|------|
| images/wavenet_loss_comparison.png | 新生成：五模型同口径 dev loss 柱状图（英文图内标注 + 英中图注 + 正文数值表） |
| images/wavenet_tree_fusion.png | 新生成：树状融合结构图（8 chars→4 bigrams→2 fourgrams→1 eightgram） |
| images/cell011_output01.png | 孤儿图片收编：03 章引用为"视频原配置 loss 曲线（仅示意趋势）" |

### 报告文件
- ledger/ledger_P05.md：28 条（P0×5 / P1×11 / P2×12）+ disputed×4
- teachers/fix_P05.md：本文件

---

## 二、验证记录（全部实跑）

### 1. 锚点数字（seed=42，CPU）
| 模型 | 脚本 | 档位 | dev loss |
|------|------|------|----------|
| P2 MLP 最小 | Part2/05 | 20K | 2.3710 |
| P3 深层 BN | Part3/05 | 20K | **2.1625**（本轮实跑） |
| 展平 MLP block8 | Part5/03 | 20K | **2.1064**（test 2.1061） |
| WaveNet 小 | Part5/05 | 20K | **2.0957** |
| WaveNet 放大 | Part5/07 | 50K | **2.0004**（test 1.9948，train 1.7579） |

参数量：小模型 **22,397**、放大模型 **76,579**（脚本打印 + 教程逐层手算双验证；旧教程 ~170K 错 2.2 倍）。
学生报告交叉核对：S2 实测 2.1064/2.0957/2.2087(quick) 与本轮逐位一致，学生数据可信。

### 2. 默认档不变性（G8 承诺）
- 04/06：新版 vs 原版 stdout **逐位一致**（仅注释变化）。
- 01：默认档（2000 步完整跑）stdout **逐位一致**。
- 03：默认档（20K 步完整跑）与原版日志 diff，**数值行全部一致**，唯一差异为有意改写的尾部总结文字。
- 05：默认档（20K 步完整跑）数值一致，仅 step15000 出现 1.8428→1.8427（OMP 线程数不同导致的浮点归约噪声，±0.0001；同 P02-D4 现象，记 disputed D4）。
- 07：--quick/STEPS 档实跑通过；默认档训练路径未动（仅打印分档与 flush）。

### 3. 短程档出数
STEPS=500（01）/1000（03、05）/2000（02）全部正常出数、提示与打印间隔按 max_steps//5 自适应；07 --quick 打印 5 行 train/dev loss（原版 0 行）。

### 4. 作业
参考答案（assignment_reference/assignment_05）+ 镜像 test：**pytest 5/5 PASSED**（128.99s，含题 5 训练 10000 步）。

### 5. 格式
check_latex.py 对 5 个改动 md（4 教程 + 作业 README）：**全部 0 问题**。

---

## 三、T1 计划 16 个 C1 缺口逐条对账

| C1 | 处置 | 台账 |
|----|------|------|
| C1-0 README 漏列两节 | fixed | P05-C00-T-06 |
| C1-1 view(0,-1) 笔误 | fixed | P05-C01-S1-06 |
| C1-2 字典叙事与 P3 不符 | fixed | P05-C01-T-07 |
| C1-3 BatchNorm1d 未定义 | fixed（教程补完整定义） | P05-C01-T-01 |
| C1-4 孤儿脚本 02 | fixed（01 章收编 + disputed D1 改名问题） | P05-C01-T-08 |
| C1-5 性能叙事矛盾 | fixed（实测重述） | P05-C02-S2-01 |
| C1-6 整除断言三处不一 | fixed（教程+05+07 补齐） | P05-C02-T-03 |
| C1-7 (B,1,C) 收尾缺失 | fixed（新增形状链小节） | P05-C02-S3-04 |
| C1-8 view/cat 语义 | fixed（明确"对该写法严格相等"+配错边界；T1 预判 allclose=False 被三方实测推翻，以实测为准） | P05-CSC-S1-03 |
| C1-9 上下文示例错位 | fixed | P05-C02-S1-08 |
| C1-10 数字无锚点 | fixed（全部实测+档位锚点） | P05-C02-S2-01 |
| C1-11 参数量 ~170K | fixed（76,579 + 手算小节） | P05-C03-S1-01 |
| C1-12 对比表无锚点 | fixed（五模型实测表） | P05-C02-S2-01 |
| C1-13 "完全等价"过强 | fixed（降格为感受野等价） | P05-C03-T-04 |
| C1-14 lr 调度未提 | fixed（入参数表） | P05-C03-T-11 |
| C1-15 验证块未标出处/无 eval | fixed | P05-C03-T-12 |

roadmap 类（O5：脚本 02 名实不符）→ disputed P05-D1；跨 Part 双列性能叙事 → disputed P05-D2。

---

## 四、通用问题候选（供 master plan 沉淀，≤3）

1. **"视频数字直搬"型失真**：~170K / ~2.02 / "首次 <2.0" 都是原视频配置的数字，未在本仓库配置下复核即写进教程，造成 3 处 P0。建议 master plan 增补规范：引用视频数字必须与"本仓库实测"双列并注明不可复现性。
2. **教程代码块"片段化"导致照抄崩**：FlattenConsecutive 缺 parameters()、BatchNorm1d 未定义、半截 __call__，同一 Part 内 3 处。建议规范：教程中出现的每个类/函数，要么给出完整定义，要么显式链接"定义所在章/脚本"。
3. **注释与输出/代码矛盾**：04 "顺序不一定相同"（实测相等）、06 "squeeze 破坏"（实际从不更新）、05 "Flatten 层"（不存在）、01 "字典管理层"（P3 教程用类）。建议规范：审计时"注释-代码-输出"三方对照，注释错误按事实错误计 P0/P1。

## 五、好写法候选（供推广，≤3）

1. **三方一致的自验证脚本**（04/06）：先 assert 再打印，秒级跑完、自带答案，整改后注释与输出严格一致——适合作为全部演示类脚本的模板。
2. **STEPS 短程档约定**：`max_steps = int(os.environ.get("STEPS", 默认值))` + `log_every = 5000 if max_steps>=5000 else max(1, max_steps//5)` + `print=functools.partial(print, flush=True)`，默认档逐位不变、短程档必出数——P2 首创、P5 全脚本推广，建议写入 scripts 编写规范。
3. **S1 的"参数量逐层手算表"**（648+6144+256+32768+256+32768+256+3483=76,579）：把"数字可自查"写进教程，比结论数字更抗错——建议后续 Part 的模型表都带一列"逐层算式"。

---

## 六、遗留与移交

- disputed 4 条（P05-D1 roadmap 改名、P05-D2 跨 Part 性能叙事格式、P05-D3 unbiased 口径回写 P3、P05-D4 浮点噪声）待 T0 裁决。
- 全部 28 条台账均为 fixed，"复核"列待用户验收。
- scratch 证据目录：REVIEW/scratch/t2_P5/（orig/ 原版留底、run*.log 完整档日志、new*_default.log 验证跑、make_figures.py、pytest_run/）。
