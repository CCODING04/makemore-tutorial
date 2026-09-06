# S2 学习报告 · P02（Part 2: MLP）

> 审计人：S2（实操薄弱型学生：会调 API，没从零写过；不熟 shape/broadcasting/训练循环细节）
> 日期：2026-09-04 · 全程真实动手，未读任何参考答案/评审材料

---

## 一、总分：7.8 / 10

教程主线讲得清楚，`C[X]` 和 `view(-1)` 的详解对 shape 薄弱的学生非常友好；脚本可独立运行、路径稳健。扣分点：教程 02 有两处硬伤（view(-1) 元素数算错、"零拷贝"说法错误）、教程 03 引用的"完整代码"缺学习率搜索段、采样种子与示例输出误导；作业题 5 骨架注释里的初始化代码是一个会直接报错的真 bug，且 README 与骨架/测试在三处（签名、阈值、提示代码变量）互相对不上。

---

## 二、卡点清单（真实踩坑记录）

| # | 卡点 | 发生位置 | 耗时/后果 |
|---|------|----------|-----------|
| 1 | `C[X]` 的输出形状第一反应写成 (3,N,2)，不确定维度顺序 | 教程 02 / 作业题 2 | 回看教程"形状推导规则"确认：X 在前 → (N,3,2)。约 1 分钟 |
| 2 | `X.append(context)` 到底要不要拷贝？怕滑动窗口改掉已入列的行 | 作业题 1 | 想明白 `context[1:]+[ix]` 每次生成新 list 才安全。约 1 分钟 |
| 3 | 梯度清零用 `p.grad = None` 还是 `p.grad.zero_()`，不敢确定 | 作业题 3 | 骨架注释直接给了 None，照做；教程思考题有铺垫但没有正解 |
| 4 | **题 5 骨架注释初始化代码照抄必炸**：`torch.randn(..., requires_grad=True) * 0.1` 得到非叶子张量（实测 `is_leaf=False`），`p.grad` 恒为 None，train_step 里 `lr * p.grad` 直接 TypeError | 作业题 5 | 写之前起疑，写完做最小实验实证，改为 `(randn*0.1).requires_grad_(True)`（与 test 正确写法一致）。二次通过 |
| 5 | lr decay 阈值：README 提示写死 `i < 100000`，骨架注释写 `i < steps // 2`，两处不一致 | 作业题 5 | 选了 steps//2（对 steps=1000 的测试才合理） |
| 6 | **05/06/07 三个脚本在本机 CPU 上 20000 步均超 3 分钟**（timeout 180 被杀），且 print 无 flush，缓冲丢失 → 被杀后一条日志都没有 | scripts 05/06/07 | 全部记 time-blocked；只能自写 2000 步缩短版验证逻辑 |
| 7 | std=1 随机初始化时初始 loss ≈ 19.5（04 实测 19.5116），教程正文完全没提，第一次看到会被吓到 | 教程 02/03、脚本 04、作业 `__main__` | 作业 test_train_step 注释里解释了（CE≈18），教程缺这段 |
| 8 | 作业 README 自称 `assignment.md`，实际文件名是 README.md；其"文件结构"图对不上 | 作业 | 未影响做题，纯困惑 |
| 9 | test_tuning 占了 pytest 总时长的大头（全量 build_dataset 228146 样本 + 1000 步训练，总计 97s） | 作业测试 | 可接受，但迭代慢 |

---

## 三、逐章评分

### 3.1 教程

| 章节 | 分数 | 亮点 | 问题 |
|------|------|------|------|
| tutorial/README.md | 8.5 | 导航图清晰，学习路线一目了然 | 无 |
| 01_introduction.md | 8.5 | "emma 展开"表、滑动窗口强调、三划分 ASCII 图都很好 | ① 代码片段 `build_dataset(words[:n1])` 是单参闭包式，脚本里全是 `(words, stoi, ...)` 显式传参，学生照抄对不上；② `X[:5]` 示例配文"先看前 5 个名字"易误读成"5 行=5 个名字"（实际前 5 行全来自 emma，数值本身核对无误） |
| 02_mlp_architecture.md | 8.0 | **C[X] 形状推导规则 + view(-1) 详解是全教程最佳段落**；课后练习质量高 | ① **硬伤**：`emb.view(-1) → (64,)` 错，32×3×2=192；② **硬伤**："`C[X]` …零拷贝"说法错误，高级索引返回新张量（实测 data_ptr 不同）；③ 只写 `torch.randn((6,100))` 不提示初始 loss 会到 ~19 |
| 03_training_and_eval.md | 7.5 | train/dev loss 三种情况诊断图极好；minibatch 对比图直观 | ① 学习率搜索代码声称"完整代码见 05"，但 05 只有固定两段 lr decay，**没有** lr 搜索；② 采样种子 `2147483647` vs 脚本 07 的 `2147483647+10`，不一致；③ 示例输出 "mora/kiah/mel" 是原视频大模型（emb=10/hid=200）的输出，与教程自己的 2 维小模型对不上（我实测小模型输出是无意义词，见 §六）；④ 可视化代码 `plt.grid("minor")` 与 06 的 `plt.grid(True, alpha=0.3)` 不同 |

### 3.2 脚本（7 个全部实跑）

| 脚本 | 结果 | 备注 |
|------|------|------|
| 01_explore_data.py | ✅ 跑通（<1s） | 输出 32033 名字、27 字符，与教程一致 |
| 02_dataset_with_context.py | ✅ 跑通（<1s） | emma 展开演示清晰；**build_dataset 带 itos 参数，与 03-07（无 itos）签名不统一** |
| 03_embedding.py | ✅ 跑通（<1s） | C[X] 演示好 |
| 04_mlp_forward.py | ✅ 跑通（<1s） | 总参数 3481（教程"约 3500"吻合）；初始 loss 19.5116 无解释 |
| 05_minibatch_training.py | ⛔ **time-blocked**（timeout 180s 被杀，user 23m16s，无任何输出留存） | 20000 步 CPU 超预算；建议加步数开关/flush |
| 06_visualize_embedding.py | ⛔ **time-blocked**（同上，user 12m10s） | 仓库已有 8/30 生成的 embedding_visualization.png 可查 |
| 07_sampling.py | ⛔ **time-blocked**（同上，user 12m25s） | 种子与教程不一致（见上） |
| （自写）mini_train_check.py | ✅ 2000 步缩短版跑通 | 证明 05/06/07 逻辑正确：初始 19.57 → 500 步 2.97 → train 2.54 / dev 2.53 |

### 3.3 作业

设计良好：4 基础题 + 1 拓展题，测试断言合理（shape/dtype/不修改参数/loss<10/val<2.5），test 注释教学价值高。问题集中在题 5 的文档一致性（见 §四 #17-20）。

---

## 四、一致性核对表（教程 ↔ 脚本 ↔ 作业）

| # | 对照点 | 教程/文档 | 脚本/实现 | 判定 |
|---|--------|-----------|-----------|------|
| 1 | build_dataset 核心逻辑（`[0]*block_size`、`w+'.'`、`context[1:]+[ix]`、`torch.tensor`） | 01_introduction.md | 02-07 一致 | ✅ 一致 |
| 2 | build_dataset 签名 | `build_dataset(words)` 单参 | 02: `(words, stoi, itos, block_size)`；03-07: `(words, stoi, block_size)` | ⚠️ 不一致（教程简化 + 脚本间不统一） |
| 3 | mlp_forward 步骤 C[X]→view→tanh→W2 | 02 §Step1-4 | 04/05 逐步一致 | ✅ 一致 |
| 4 | view 写法 | 训练循环 `emb.view(-1, 6)` | `emb.view(emb.shape[0], -1)` | ✅ 等价（写法不同，初学会疑惑） |
| 5 | 训练步数/调度 | 1000 步固定 lr（简化段） | 05: 20000 步 + 15000 步衰减 | ⚠️ 教程简化版，无碍 |
| 6 | 学习率搜索代码归属 | "完整代码见 05" | 05 无此段 | ❌ 引用不对应 |
| 7 | 采样 generator 种子 | `manual_seed(2147483647)` | 07: `2147483647 + 10` | ❌ 不一致 |
| 8 | 采样示例输出 | "mora, kiah, mel..." | 本教程小模型实测输出无意义词 | ❌ 示例与模型配置不匹配 |
| 9 | 可视化实现/输出 | `C[:,0].data`、`grid("minor")`、`../images/cell031_output00.png` | `.detach().numpy()`、`grid(True, alpha=0.3)`、`scripts/embedding_visualization.png` | ⚠️ 等价但不同源；两图文件均存在 ✅ |
| 10 | view(-1) 展平示例 | "(64,)" | 32×3×2=192 | ❌ 教程算错 |
| 11 | "C[X] 零拷贝" | 02 教程声称 | 实测 data_ptr 不同（返回拷贝） | ❌ 说法错误 |
| 12 | 全量样本数 | 228146（"22万+"） | 实测 182625+22655+22866=228146 | ✅ 一致 |
| 13 | 参数量 | "约 3,500" | 04 实测 3481 | ✅ 一致 |
| 14 | 教程 X[:5]/y[:5] 数值 | [5,13,13,1,0] | 未打乱 words 前 5 行恰为 emma 展开 | ✅ 数值一致（配文易误读） |
| 15 | 作业文件名 | README 自称 assignment.md | 实际 README.md | ⚠️ 轻微 |
| 16 | 题 1-4 骨架签名 ↔ README ↔ test | `build_dataset(words, block_size=3)` 等 4 个 | 三方一致 | ✅ 一致（本次未见历史签名 bug） |
| 17 | 题 5 签名 | README 无 seed | 骨架与 test 均有 `seed=2147483647` | ❌ README 落后于实现 |
| 18 | 题 5 达标线 | README "< 2.2"；表格 A 配置却预期 "~2.3"（自相矛盾） | test "< 2.5"（10/100/1000步） | ❌ 三处三个数 |
| 19 | 题 5 提示代码变量 | `train_step(Xb, Yb, ...)`，Xb/Yb/batch_size 未定义 | 骨架注释用 `Xtr[ix]`/自己定义 batch | ❌ README 提示跳步 |
| 20 | **题 5 骨架注释初始化** | `torch.randn(..., requires_grad=True) * 0.1`（×4 行） | 实测 is_leaf=False → p.grad=None → TypeError；test 用 `(randn*0.1).requires_grad_(True)` | ❌ **注释代码是真 bug，照抄必炸** |

---

## 五、作业元数据 + pytest 输出

- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_2/`（内含复制的 README.md / mlp_exercises.py / test_mlp_exercises.py，仅编辑 exercises；data 软链 `../data` → 主仓 data，路径解析正确）
- 结果：**5 题全 PASS**（含拓展题）

```
$ .venv/bin/python -m pytest test_mlp_exercises.py -v
test_mlp_exercises.py::test_build_dataset PASSED                         [ 20%]
test_mlp_exercises.py::test_mlp_forward PASSED                           [ 40%]
test_mlp_exercises.py::test_train_step PASSED                            [ 60%]
test_mlp_exercises.py::test_evaluate PASSED                              [ 80%]
test_mlp_exercises.py::test_tuning PASSED                                [100%]
========================= 5 passed in 97.12s (0:01:37) =========================
```

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|----|---------|----------|------|------|
| 1 build_dataset | ✅ 一次过 | 回看 README 提示 1 次（确认 stoi 写法） | ~3 min | PASS（14 断言） |
| 2 mlp_forward | ✅ 一次过 | 骨架注释即答案，未看 README | ~1 min | PASS |
| 3 train_step | ✅ 一次过 | 骨架注释即答案 | ~2 min | PASS（loss<10、梯度非零） |
| 4 evaluate | ✅ 一次过 | 骨架注释即答案 | ~1 min | PASS（参数未被修改） |
| 5 tuning_experiment | ⚠️ 二次过（先识破注释 bug，实验实证后修正） | 骨架注释给得太全≈半答案 | ~5 min | PASS（val_loss < 2.5） |

佐证实验（学生自查）：
```
照注释写: is_leaf = False , requires_grad = True   ← 非叶子，p.grad 恒 None，必炸
正确写法: is_leaf = True , requires_grad = True
C[X] 与 C 共享内存(零拷贝)? False                     ← 教程"零拷贝"说法错误
```

---

## 六、补充实测（time-blocked 的替代验证）

05/06/07 原样跑超 3 分钟（记 time-blocked）。照抄其逻辑写 2000 步缩短版（`scratch/S2_P2/mini_train_check.py`）：

```
初始 mini loss: 19.5709  (ln(27)=3.2958)
step 0: 18.22 → step 500: 2.97 → step 1000: 2.19
Train loss: 2.5359 | Dev loss: 2.5332     ← 训练循环逻辑正确、能收敛
教程种子2147483647:  ['juonde', 'janasah', 'pastay', 'alin', 'koi']
脚本07种子+10:       ['eriaalmyadhiee', 'mad', 'ryal', 'rethrscend', 'lri']
```

结论：① 两个种子输出完全不同，教程与脚本种子不一致会直接导致学生复现不出教程结果；② 2000 步小模型采样是乱码，教程 "mora/kiah/mel" 示例绝非本教程配置所能产出。

---

## 七、只改 3 件事

1. **修作业题 5 的两处文档 bug**：骨架注释初始化改为 `(torch.randn(..., generator=g) * 0.1).requires_grad_(True)`（与 test 一致，照抄即过）；README 题 5 补 `seed` 参数、统一达标线（2.2 / ~2.3 / 2.5 三处矛盾取其一）、提示代码补上 `build_dataset` 与 `batch_size` 定义或删掉未定义变量。
2. **修教程 02 的两处技术硬伤**：`emb.view(-1) → (64,)` 改为 `(192,)`；"C[X] 直接查表，零拷贝"改为"C[X] 避免构造 one-hot 大矩阵、一次查出（注意它返回新张量，不是视图）"。
3. **让 05/06/07 在 CPU 上可跑完**：给 MAX_STEPS 加环境变量/argparse 缩短开关，print 加 `flush=True`；同时统一采样种子为教程的 `2147483647`（或教程标注脚本种子），教程 03 删除或补齐"完整代码见 05"指向的 lr 搜索段。

---

## 八、最喜欢的 3 处

1. **教程 02 的 "C[X] 高级索引详解"**：把 `X[1]=[0,0,5] → [C[0],C[0],C[5]]` 逐行展开再给"形状推导规则"，我这种 shape 盲读一遍就会了——全教程最有价值的段落。
2. **教程 03 的 train/dev loss 三种情况 ASCII 诊断图**：欠拟合/刚好/过拟合一图看懂，配合"参数 3500 vs 样本 22 万"的具体数字，过拟合判断不再玄学。
3. **作业 test_train_step 里关于初始化尺度的注释**（"std=1 实测 CE≈18，缩放后 ≈3.37≈ln(27)"）：用实测数字把 Part 3 的伏笔讲明白了，比教程正文还清楚——我跑 04 看到 19.5 时就是靠它释怀的。
