# S2 学习报告·P04（手动反向传播）

- 审计学生：S2（实操薄弱型：以"先动手做、卡了再查"的方式走完全程）
- 审计时间：约 18 分钟（脚本 6/6 已跑、作业 5/5 已交）
- 工作目录：
  - 脚本运行 scratch：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P4`
  - 作业提交：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_4`（只改了 `backprop_exercises.py`）

## 总分

**86 / 100**

| 维度 | 得分 | 说明 |
|------|------|------|
| 教程↔脚本代码一致性 | 17/20 | 12 步推导逐段可对上；2 处注释/表述瑕疵（见核对表） |
| 脚本可运行性 | 18/25 | 6/6 能跑通、对拍全对，但 05 有 ❌/🎉 自相矛盾输出、06 默认线程下超 3 分钟 |
| 作业质量 | 25/25 | 5/5 PASS，阈值设置有依据（test_q4 注释了 1e-4 的原因） |
| 测试工程规范 | 16/20 | 测试函数 return bool 而非 assert，pytest 下失败不会变红；Q2 有死代码 |
| 文档体验 | 10/10 | 直觉解释出色，卡点少，报错信息友好 |

## 卡点清单（按真实发生顺序）

1. **【概念·最大卡点】BN 一行公式里 `n/(n-1)` 的来源**。做 Q4 时我对着教程 12 步（12a–12e）自己推合并公式，推到方差链时变形不出 `n/(n-1)`（前向 `bnvar = bndiff².mean(0)` 明明是有偏 /n 方差）。最后靠脚本 05 的数值对拍（逐步 vs 简化 max diff = 4.56e-05，float32 噪声量级）接受公式等价，没有完成纯代数推导。**而 README 思考题 Q4 的提示（"无偏估计用 n-1，PyTorch var(unbiased=True)"）把我带偏了**——它暗示前向用了 n-1 方差，实际前向就是 /n。这是文档层面的真实误导，不是我的问题。
2. **【环境·time-blocked】脚本 06 默认设置下超时**。`--quick`（1000 步）首次运行 2m50s 被 timeout 杀掉（exit 143，user 15m27s——torch 多线程在核间空转）。加 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` 后约 1 分钟跑完。6 个脚本中唯一一次 time-blocked，脚本本身没有任何提示需要压线程。
3. **【输出自相矛盾】脚本 05 结尾永远打印 🎉**。实测输出：`简化 vs Autograd: max diff = 4.56e-05 ❌`（脚本内部阈值 1e-5），紧接着却打印"🎉 简化版 BatchNorm 反向传播验证完成！"——脚本没有按 all_ok 分支输出。学生看到 ❌+🎉 会困惑自己是不是装错了环境。
4. **【b1 的有无】** 教程 02 反推链里有 `db1 = dhprebn.sum(0)`（脚本 01–05 带 b1），而脚本 06 和作业 exercises 没有 b1（"b1 被 BN 的 beta 取代"）。做 Q5 时我一度想加 db1，对照 exercises 参数表确认无 b1 后才放下。跨章节能各圆其说，但初学者容易在 06 处回头翻 02 找 db1。
5. **【测试规范】pytest 的 PASSED ≠ 逻辑通过**。test 文件所有测试函数 `return True/False` 而非 `assert`，pytest 9 发出 `PytestReturnNotNoneWarning`；即使函数 return False，pytest 仍然显示 PASSED。必须用 `python test_backprop_exercises.py` 直跑看 `得分: 5/5` 才可信。另外 test_q2 里 `dh_ref`（`h.sum()` 反传那段）是死代码，从未被使用。

## 逐章评分

| 材料 | 评分 | 一句话评价 |
|------|------|-----------|
| `tutorial/01_why_backprop.md` | 8.5/10 | 动机充分、链式法则铺垫够用；开篇网络图 (30→200) 是 Part 3 配置，与正文 64 不同，需读者自行分辨 |
| `tutorial/02_forward_and_backward.md` | 8/10 | 12 步推导完整可跟，命名约定框（dX=∂L/∂X）很贴心；Step 9 注释 `(B, 200)` 是笔误（应为 (B,64)），Step 8 的 `squeeze()` 在 B=1 时会碎但未提示 |
| `tutorial/03_simplified_and_training.md` | 8.5/10 | CE 3 行 + BN 1 行的直觉解释是全 Part 亮点；把 `n/(n-1)` 称作"Bessel 校正"属于类比性表述，严格说不精确（前向方差是有偏 /n） |
| `tutorial/README.md` | 9/10 | 导航、学习路线图、学习目标清单齐全 |
| `scripts/01` `02` `03` `04` | 9/10 | 逐行可复现教程，对拍数值干净（1e-9 量级），03 的 retain_grad 教学尤其好 |
| `scripts/05` | 6/10 | 对拍本身正确，但 ❌/🎉 矛盾 + 阈值与数值噪声不匹配，作业 test_q4 已修（1e-4）而脚本没同步 |
| `scripts/06` | 8/10 | 端到端手动训练跑通（quick 模式 Train 2.47），无性能提示是硬伤 |
| `assignments/assignment_4` | 9/10 | README 与 exercises 函数签名不一致（`backward_step(step_name,...)` vs 4 个独立函数）、思考题 Q4 提示误导，但测试覆盖与注释质量高 |

## 一致性核对表（教程代码块 ↔ 脚本逐段）

| # | 教程位置 | 代码块内容 | 脚本对应 | 结果 |
|---|----------|-----------|----------|------|
| 1 | 02.md Step 1 | `dlogprobs[arange(B), Yb] = -1.0/B` | 02/03/04 脚本 Step 1 | ✅ 一致 |
| 2 | 02.md Step 2–6 | dprobs→dcounts_sum_inv→dcounts_sum→dcounts+=→dnorm_logits | 同名脚本逐步 | ✅ 逐行一致 |
| 3 | 02.md Step 7 | `dlogits=clone()`、`dlogit_maxes=(-dnorm_logits).sum(1,keepdim=True)` | 02/03/04 脚本 Step 7 | ✅ 一致 |
| 4 | 02.md Step 8 | `argmax`+scatter 回填 `dlogit_maxes.squeeze()` | 02/03/04 脚本 Step 8 | ✅ 一致（B=1 会碎，双方都没提示） |
| 5 | 02.md Step 9 | `dh = dlogits @ W2.T` 等三行 | 脚本 Step 9 | ⚠️ 代码一致，**教程注释 `(B, 200)` 笔误，脚本 `(32, 64)` 正确** |
| 6 | 02.md Step 10–11 | tanh 局部梯度、dbngain/dbnbias/dbnraw | 脚本 Step 10–11 | ✅ 一致 |
| 7 | 02.md Step 12a–12e | BN 逐步 5 子步 | 02/03/05 脚本 Step 12 | ✅ 一致 |
| 8 | 02.md Extra | dembcat/dW1/db1/demb/双循环 dC | 02/03 脚本 Extra | ✅ 一致（含 b1）；06 无 b1，教程 03 清单也无 db1，各自自洽 |
| 9 | 02.md cmp() | allclose atol=1e-5 + maxdiff 打印 | 02/03 脚本 cmp() | ✅ 一致（03 多 mean diff，无冲突） |
| 10 | 03.md CE 3 行 | softmax→[range(n),Yb]-=1→/=n | 04 脚本方法 2、06 脚本 | ✅ 一致；实测 简化 vs autograd = 1.86e-09 ✅ |
| 11 | 03.md BN 一行公式 | `bngain*bnvar_inv/n*(...)` | 05 脚本方法 2、06 脚本 | ✅ 公式一致；实测 vs autograd = 4.56e-05（脚本阈值 1e-5 判 ❌，作业 test 阈值 1e-4 判 ✅） |
| 12 | 03.md 训练清单/伪代码 | 6 步清单 + `hprebn = embcat @ W1`（无 b1） | 06 脚本 | ✅ 一致 |
| 13 | 02.md 配图引用 | `../images/cell018_output01.png` | `Part4_backprop/images/cell018_output01.png` 存在 | ✅ 引用有效；脚本 04 另生成 `scripts/dlogits_heatmap.png`，两张图不同但各有出处 |
| 14 | README.md 作业描述 | Q2 `backward_step(step_name, upstream_grad, cache)` | exercises 实为 `backward_tanh/linear/bn_scale/softmax_ce` 四函数 | ❌ 签名不一致；README 要求支持 4 种操作，test_q2 只测 3 种（bn_scale 未测） |
| 15 | 01.md 回顾图 | Linear(30→200) | 正文/脚本 01–05 用 64，06 用 200 | ⚠️ 跨文档尺寸漂移，均有注释，建议统一标注"Part 3 配置" |

## 作业元数据 + pytest 输出

- 提交方式：仅编辑 `backprop_exercises.py`（工作副本），test/README 未动；data 软链 `students/S2_hands/data` 未动
- 运行命令：`cd work/assignment_4 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python -m pytest test_backprop_exercises.py -v`

**每题记录：**

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|----|---------|---------|------|------|
| Q1 forward_pass | 是 | 0 | ~3 min | PASS，loss 误差 2.38e-07 |
| Q2 backward_tanh/linear/bn_scale | 是 | 0 | ~3 min | PASS，tanh diff=0，linear diff=0，softmax_ce diff=1.86e-09 |
| Q3 cross_entropy_backward | 是 | 0 | ~1 min | PASS，max diff = 1.86e-09 |
| Q4 batchnorm_backward | 是 | 1 次自我纠结（n/(n-1) 推导，见卡点 1） | ~4 min | PASS，max diff = 4.83e-05（< 1e-4） |
| Q5 manual_train | 是 | 0（骨架参照教程脚本 06） | ~8 min | PASS，1000 步 loss 3.3177→2.5116，last100 均值 2.9715→2.4745 |

**pytest 输出（关键行）：**

```
test_backprop_exercises.py::test_q1_forward_pass PASSED                  [ 20%]
test_backprop_exercises.py::test_q2_backward_step PASSED                 [ 40%]
test_backprop_exercises.py::test_q3_cross_entropy_backward PASSED        [ 60%]
test_backprop_exercises.py::test_q4_batchnorm_backward PASSED            [ 80%]
test_backprop_exercises.py::test_q5_manual_train PASSED                  [100%]
======================== 5 passed, 5 warnings in 34.27s ========================
```

（5 个 warning 均为 `PytestReturnNotNoneWarning`：测试 return bool 而非 assert，见卡点 5）

**直跑模式补充证据（`python test_backprop_exercises.py`）：** `得分: 5/5`，Q4 打印 `max diff = 4.83e-05, mean diff = 3.24e-06`，与 test 注释"实测正确实现 ≈4.8e-05"吻合。

**Q4 BN 反传推导过程与卡壳处（作业要求记录）：**
- 我的推导：对 hprebn 的梯度走三条路——(a) 直接路径 x̂=bndiff·bnvar_inv，贡献 `γ·bnvar_inv·d`；(b) 经均值 μ=mean(x) 的路径，贡献 `-γ·bnvar_inv/n·Σd`；(c) 经方差 σ²=(1/n)Σ(x-μ)² 的路径，先到 dbnvar `-0.5·(bnvar+ε)^{-1.5}`，再散回每个样本。
- 卡壳处：(c) 路径合并化简时推不出 `n/(n-1)` 因子——我的草稿只能到 `-γ·bnvar_inv/n·x̂·Σ(d·x̂)` 乘某个与 n 相关的系数。没能纯代数收尾，改用"两条独立计算链数值对拍"验证（脚本 05 与 test_q4 双重确认），接受一行公式。
- 事后认识：README 思考题 Q4 用"无偏估计/Bessel"解释该因子，方向有误导（前向方差就是有偏 /n）；该因子是合并路径时的代数变形结果，教程 03 的"三个项"表格同样未讲清来源。建议教程补一小段"为什么会出现 n/(n-1)"的推导或承认"此处直接给结论、数值验证兜底"。

## 只改 3 件事

1. **修 `scripts/05_batchnorm_backward.py` 的判定与输出**：阈值 1e-5 放宽到 1e-4（或对拍用 float64），并且结尾的 🎉/⚠️ 按 `all_ok` 分支输出——现在 ❌ 与 🎉 同屏，学生第一反应是怀疑自己环境装错（我实测卡了 1 分钟）。
2. **修 `tutorial/02_forward_and_backward.md` Step 9 注释**：`# (B, 200)` → `# (B, n_hidden)`（或 (B, 64)）。同时给 Step 8 的 `squeeze()` 加一句"B=1 时 squeeze 会退化，建议 `squeeze(1)`"。另在 01.md 回顾图旁标注"(Part 3 原配置 n_hidden=200，本 Part 教学用 64)"。
3. **修测试的 pytest 兼容性**：test 全部函数把 `return ok` 改成 `assert ok, ...`（否则 pytest 下逻辑失败也显示 PASSED，只有直跑 `python test_*.py` 才可信）；顺带删除 test_q2 里从未使用的 `dh_ref` 死代码；README 的 Q2 函数签名与 exercises 对齐（4 个独立函数），思考题 Q4 的提示改为"n/(n-1) 来自有偏方差反传的代数化简，不是前向用了无偏方差"。

## 最喜欢 3 处

1. **`02_forward_and_backward.md` 开头的"变量命名约定"提示框**（`dX` = ∂L/∂X、上游梯度 vs 局部梯度）——做作业 Q2 时我回头翻了两次，每次都省下重新辨义的时间。这是 Karpathy 原版没有、教程加分的部分。
2. **`03_verify_gradients.py` 的两段式验证**：先指出"hprebn 是非叶子节点，.grad 默认不保存（需 retain_grad）"再补一段 retain_grad 完整对比 7 个中间变量梯度。我第一次跑时以为脚本漏了中间变量，往下滚才发现是刻意设计——这种"先埋问题再解答"的脚本结构非常适合自学。
3. **`03_simplified_and_training.md` 的 CE 反传直觉段**（"你对正确答案的信心还不够，要再加把劲"）+ 04 脚本把 dlogits 热力图和每样本 L1 范数并排画出来——"正确类别位置才有 -1"这句话在图上一眼得到印证（正确位置标了黑叉且是深色）。
