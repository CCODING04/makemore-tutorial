# S2 学习报告 · P05（WaveNet）

> 学生画像：S2（实操薄弱型）| 日期：2026-09-04 | 环境：Linux x64 / 32 核 CPU / Python 3.12.12 / PyTorch（.venv）
> 审计方式：教程代码块 ↔ 脚本逐段对照；7 个脚本全部实跑；作业先做后看（未读 assignment_reference / REVIEW）。

## 总分：8.2 / 10

内容质量高、可跑性总体好、作业测试设计防呆到位。主要扣分点：教程性能数字与脚本实测有出入（~2.02 未复现、~170K 参数实为 76.6K）、教程 02 的 FlattenConsecutive 代码块缺 `parameters()`（照抄会崩）、孤儿脚本 02_fix_lr_plot.py、07 完整训练 CPU 超 3 分钟无提示。

---

## 卡点清单（按遇到顺序）

| # | 卡点 | 影响 | 环节 |
|---|------|------|------|
| 1 | 测试要求 `model(x)` 输出 **2D** `(4, 27)`，但教程/脚本架构末端输出 3D `(B, 1, 27)`；README 题 2 的结构描述（`…×3 → Linear`）不含 Flatten/squeeze，也未提此事 | 学生照抄脚本 05 结构会挂 `test_build_wavenet`，需自行补 Flatten | 作业题 2 |
| 2 | 多脚本并发实跑时 torch 默认每进程吃满 32 线程，互相超订导致 5 进程全部 180s 超时；且脚本 stdout 全缓冲，timeout 后看不到任何进度 | 第一次跑 01/02/03/05/07 全部 EXIT=124，需 `OMP_NUM_THREADS=8 + python -u` 重跑 | 脚本运行 |
| 3 | 07 完整版 50000 步 × batch 128 在 CPU 上 >3 分钟（超任务时限） | 记 time-blocked，改用 `--quick`（1000 步）验证功能；教程/脚本未标注完整训练的预计时长 | 脚本 07 |
| 4 | 01（2000 步 MLP）在资源竞争下 170s 只跑到 step 1000/2000 | forward/各层 shape/收敛趋势已验证，训练段未跑完，记 time-blocked | 脚本 01 |
| 5 | 脚本 04 演示 4 打印"结果相等: True"，紧邻注释却说"view 和 cat 的元素顺序不一定相同" | 实测相等（交错切片沿 dim=2 拼接恰好等于连续 view），注释与输出矛盾，学生易困惑 | 脚本对照 |
| 6 | 脚本 06 的 `BatchNorm1dBuggy` **根本没有 running stats 更新代码**，3D 段打印 running_mean 仍为 `(10,)`，旁边注释却写"可能被 3D squeeze 破坏" | 演示没有展示出注释宣称的破坏现象；buggy 类的 bug 只有"reduce 维度错"，没有"squeeze 破坏" | 脚本对照 |

---

## 逐章评分

### README.md — 9/10
导航表、路线图、"学完你能…"清单清晰；作业链接正确。无运行指引（如何跑 scripts、预计耗时）是小缺憾。

### 01_pytorchify.md ↔ 01_pytorchify_layers.py — 8.5/10
- Embedding / Flatten / Linear / Sequential 四个代码块与脚本 01 **逐行一致** ✓
- training/eval 切换表格 + 双向后果讲解清楚，与脚本 01 评估段做法一致 ✓
- 小缺：BatchNorm1d/Tanh 类未给出（依赖 Part 3）；模型示例缺 `layers[-1].weight *= 0.1`（脚本有）；"Part 3 用字典管理层"的说法在本部分材料中无对应物可验证。

### 02_wavenet_architecture.md ↔ 03/04/05 脚本 — 7.5/10
- 动机（8 chars→4 bigrams→…）与树状 ASCII 图直观 ✓；"Linear 只在最后一维做矩阵乘法"演示正确 ✓
- **问题 A（会导致代码崩）**：教程版 `FlattenConsecutive` 缺 `parameters()` 方法。读者把它放进 Sequential 后调 `model.parameters()` 直接 AttributeError；脚本 04/05/07 都有该方法，教程代码块不完整。
- **问题 B**：view vs cat 小节只讲性能（零拷贝 vs 分配内存，本身正确），未说明两者元素顺序问题；脚本 04 实测 `allclose=True`，注释又自称"顺序不一定相同"，三方打架。
- **问题 C**：性能表称 block_size=8 → dev ~2.02；实跑 03 脚本 dev = **2.1064**（8 线程 CPU、seed 42、20000 步），未复现，差 ~0.09。
- FlattenConsecutive 形状示例表 (B,8,10)→(B,4,20)→(B,2,40)→(B,1,80) 全部正确 ✓

### 03_training_and_bugs.md ↔ 06/07 脚本 — 7.5/10
- 3D BN bug 的"问题→修复→验证"结构与脚本 06 完全对应：dim_reduce=(0,1)、running stats 恒 1D、eval 双 unsqueeze，实跑验证输出均值≈0 / 标准差≈1.0004 ✓
- 放大模型代码块（24/128、Linear 48/256/256→128、lr 三段）与脚本 07 一致 ✓
- **问题 D（数字错误）**：表格称放大模型 ~170K 参数；07 实测打印 **76,579**（Embedding 648 + 6144+256 + 32768+256 + 32768+256 + 3456 + BN≈768，手算吻合 76.6K），教程多算了一倍以上。
- 放大后 dev ~1.99：完整训练 time-blocked 无法验证（--quick 1000 步 dev 2.2087，仅证明流水线通）。
- 卷积预览（dilation 1/2/4 感受野）是很好的"另一种视角" ✓

### scripts/ 整体 — 8.5/10
7 个全部实跑通过（2 个部分跑完记 time-blocked）。统一结构（种子 42、数据路径 `../../../data/names.txt` 有效、训练打印节奏合理）、`--quick` 开关是亮点。缺点：07 完整版无时长预期；05 头部注释架构图写 "Flatten → Linear(68,27)" 但代码没有 Flatten 层（logits 实为 (B,1,27)，靠 `view(-1,…)` 兜底）；01–03/05/07 均 `import math` 未使用；02_fix_lr_plot.py 是**孤儿脚本**——没有任何教程章节引用它，loss 平滑这个内容点在教程中完全缺失。

### assignment_5 — 9/10
题量梯度合理（4 基础 + 1 拓展），TODO 注释给步骤不给答案，测试断言细（running stats 必须被更新、eval 不得更新、shape 恒 1D、T%n 报错）。缺点：README"文件结构"写 `assignment.md`，实际文件名 `README.md`；题 2 的 2D 输出要求与教程架构存在 gap（见卡点 1）。

---

## 一致性核对表（教程代码块 ↔ 脚本）

| 教程位置 | 对照脚本 | 结果 |
|---|---|---|
| 01 · Embedding/Flatten/Linear/Sequential 四代码块 | 01_pytorchify_layers.py | 一致 ✓ |
| 01 · Sequential 重构网络 + requires_grad | 01 脚本 L177-192 | 一致（教程缺 weight×0.1 提示，轻微）△ |
| 01 · training/eval 手动切换 | 01 脚本评估段 | 一致 ✓ |
| 02 · FlattenConsecutive 代码块 | 04 / 05 / 07 脚本 | **教程缺 parameters()，照抄会 AttributeError** ✗ |
| 02 · Linear 多维输入 (4,8,10)@(10,20) | 04 脚本演示 3 | 一致 ✓ |
| 02 · view vs cat | 04 脚本演示 4 | 教程只讲性能；脚本实测相等但注释自称"顺序不一定相同"，矛盾 ✗ |
| 02 · WaveNet 架构代码块 | 05 脚本 L187-197 | 一致 ✓（小模型参数 ~22K ↔ 实测 22,397 ✓） |
| 02 · 性能表 block3~2.10 / block8~2.02 | 02 实测 dev 2.1701 / 03 实测 dev 2.1064 | block3 基本吻合、block8 未复现 ✗ |
| 03 · BN 3D 修复代码块 | 06 脚本 BatchNorm1d | 一致 ✓（教程只贴 training 分支，eval 用文字说明，可接受） |
| 03 · 修复验证代码块（running_mean (10,)） | 06 脚本实测 | 一致 ✓（均值 -0.000000，标准差 1.000386） |
| 03 · 放大模型参数 ~170K | 07 实测 76,579 | **不符** ✗ |
| 03 · 放大配置 24/128、bs=128、50K 步、lr 0.1/0.05/0.01 | 07 脚本 | 一致 ✓（dev~1.99 因超时未能验证）△ |
| 03 · 脚本引用清单 | scripts 目录 | 06、07 被引用 ✓；**02_fix_lr_plot.py 无任何章节引用**（孤儿）✗ |
| README(作业) · 文件结构 assignment.md | 实际 README.md | 文件名笔误 ✗ |

---

## 作业元数据 + pytest 输出

- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_5`（data 软链可用）
- 做题顺序：题 1 → 3 → 2 → 4 → 5；仅编辑 `wavenet_exercises.py`（题 2 为满足 2D 输出在 Sequential 末尾自加了一个辅助 `Flatten` 层）；未读任何参考答案。

| 题 | 内容 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|------|---------|----------|------|------|
| 1 | FlattenConsecutive（含 T%n 报错） | 是 | 0 | ~2 min | PASS |
| 2 | build_wavenet | 是* | 0（靠预读测试代码发现 2D 输出 gap） | ~4 min | PASS |
| 3 | BatchNorm1d3D | 是 | 0 | ~5 min | PASS |
| 4 | verify_shapes | 是 | 0 | ~2 min | PASS |
| 5 | train_wavenet 拓展 | 是 | 0（首次 pytest 因多进程抢 CPU 超时，非代码问题） | ~10 min | PASS（dev_loss < 2.5 @10000 步） |

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.1.1
collected 5 items
test_wavenet_exercises.py::test_flatten_consecutive PASSED               [ 20%]
test_wavenet_exercises.py::test_batchnorm_3d PASSED                      [ 40%]
test_wavenet_exercises.py::test_build_wavenet PASSED                     [ 60%]
test_wavenet_exercises.py::test_verify_shapes PASSED                     [ 80%]
test_wavenet_exercises.py::test_train_wavenet PASSED                     [100%]
========================= 5 passed in 60.15s (0:01:00) =========================
```

**作业：5 题 5 PASS。**

### 脚本运行台账（OMP_NUM_THREADS=8、python -u，除注明外）

| 脚本 | 结果 | 关键输出 |
|---|---|---|
| 01 | 功能验证 OK；训练段 170s 未跑完 → time-blocked | 各层 shape 正确；step 1000 loss 2.53 正常下降 |
| 02 | PASS（76s） | train 2.1562 / dev 2.1701 |
| 03 | PASS（<170s） | dev 2.1064 / test 2.1061（教程称 ~2.02，未复现） |
| 04 | PASS（秒级） | allclose(view, cat)=True |
| 05 | PASS（<170s） | 参数 22,397；dev 2.0957（教程称 ~2.07，接近） |
| 06 | PASS（秒级） | 3D mean (1,4,10) vs (1,1,10)；输出均值≈0、std≈1.0004 |
| 07 | 完整版 50000 步 >3min → time-blocked；--quick PASS | 参数 **76,579**；quick 1000 步 dev 2.2087 |

---

## 只改 3 件事

1. **补全教程 02 的 FlattenConsecutive 代码块**：加上 `parameters()` 返回 `[]`（并顺带说明它必须存在才能被 Sequential 收集参数）。当前照抄教程的读者会在 `model.parameters()` 处直接 AttributeError。
2. **修正教程 03 的参数量与性能数字**：~170K → 实测 76,579；复核 02 章性能表（block8 实测 dev ≈2.11 而非 ~2.02），并为所有性能数字标注"seed 42 / CPU / 步数"等复现条件。
3. **收编孤儿脚本 02_fix_lr_plot.py**：在 01 或 03 章加一小节"loss 曲线平滑"（`view(-1, window).mean(1)`），并在脚本 07 头部注明"完整训练 CPU 需 >10 分钟，快速验证请加 --quick"；同时修复 04 的矛盾注释与 06 的 buggy 类注释（buggy 类实际不含 running stats 更新）。

## 最喜欢 3 处

1. 教程 02 的树状融合 ASCII 图 + 脚本 04 演示 3："Linear 只在最后一维做矩阵乘法"这一 WaveNet 得以成立的洞察，用一个 3 行例子讲透了。
2. 脚本 06 的 Buggy vs Fixed 对照实验：2D 两者一致、3D 打印 (1,4,10) vs (1,1,10)，bug 的本质（每个时间步独立归一化）一眼看穿。
3. 作业测试的防呆设计：断言"训练后 running_mean 必须被更新""eval 模式绝不能更新""T%n!=0 必须报错"，把学生最容易糊弄过去的角落全部钉死。

---

*S2 · 实操薄弱型学生审计 · 全程未联网、未读 assignment_reference 与 REVIEW 产物*
