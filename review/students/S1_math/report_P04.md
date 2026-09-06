# S1 学习报告·P04（手动反向传播）

> 学生画像：数学畏难型。逐式核验了全部手推梯度，实跑 6 个脚本，BN 口径问题做了独立数值裁决。

## 总分：7 / 10

一句话：12 步逐步推导（教程 02 + 脚本 02/03）数学上**全部正确**、对拍漂亮（max diff ≤ 5.6e-09）；但本章主打的"BN 一行简化公式"与自身前向的方差口径**不自洽**，脚本 05 对拍实测 **FAIL（4.56e-05 > 1e-5）**，且脚本失败后仍 exit 0 打印庆祝文案——恰好砸在它自称的"面试利器"上。

---

## 卡点清单（按严重度排序）

1. 【高·数学错误】**BN 简化反传公式口径错配**。教程 03 / 脚本 05 / 脚本 06 的公式第三项带 `n/(n-1)`，但前向是 `bnvar = bndiff2.mean(0)`（1/n 有偏方差），逐步反传（Step 12）也是 1/n 口径。我的四象限数值实验（n=8，对拍 autograd，见附录 B）证明：
   - 1/n 前向 + 系数 `1` → PASS（2.98e-08）
   - 1/n 前向 + 系数 `n/(n-1)`（教程用的）→ **FAIL（1.53e-02）**，batch 越小越糟
   - 1/(n-1) 前向 + 系数 `n/(n-1)` → PASS（3.73e-08）
   即 `n/(n-1)` 是**无偏方差口径**的产物，教程却把它套在 1/n 前向上。脚本 05 实测 n=32 时偏差 4.56e-05，超出 1e-5 阈值——不是浮点噪声，是系统性口径错误。
2. 【高·脚本行为】**脚本 05 对拍失败但仍"通过"**：`简化 vs Autograd: max diff = 4.56e-05 ❌` 打印之后，脚本 exit 0，结尾继续输出"🎉 验证完成！记住这个一行公式（面试利器）"。失败被庆祝文案淹没（06 同病：用的同一公式，结尾"训练成功！"）。
3. 【中·概念错误】**"Bessel 校正"解释是错误归因**（教程 03 表格、脚本 05"原理解释"、脚本 05 文件头）。前向根本没用无偏估计；`n/(n-1)` 的正解是"它属于 1/(n-1) 前向的反传公式，与 1/n 前向不配套"。assignment 思考题 Q4 同错："为什么方差的梯度要除以 n-1……PyTorch 的 var(unbiased=True) 也用 n-1"——BN training 归一化实际用有偏 1/n，反传口径必须跟前向走。
4. 【中·一致性】教程 02 Step 9 注释 `dh = dlogits @ W2.T  # (B, 200)` 与本教程计算图/脚本（n_hidden=64，实测打印 `(32, 64)`）矛盾；200 来自脚本 06 的配置，串写未改。
5. 【中·一致性】**网络定义三个口径**：01 教程图写 `Linear (30 → 200)`（Part 3 回顾），01–05 脚本 n_hidden=64 且带 b1，06 脚本 n_hidden=200 且无 b1（"b1 被 BN β 取代"）。教程 03 的手动反传清单也照 06 写（无 db1）。学生跨脚本对照时会困惑"为什么我的清单里没有 b1"。
6. 【低】教程 03 清单 6️⃣ 写 `dC[Xb] += demb (scatter 操作)`，但清单前一步没有定义 `demb`（应为 dembcat → demb → dC 两步）。
7. 【低】脚本 06 `--quick` 模式 1000 步但打印间隔 10000 → 只打印 Step 0，学生看不到 loss 下降过程。
8. 【低】脚本 03 在 `if hprebn.grad is None:` 处访问非叶子 `.grad` 触发 UserWarning（实测日志有）；脚本 05 第 123 行注释"（原脚本 bug）"是修 bug 的施工痕迹残留。

---

## 逐章评分

| 材料 | 分 | 依据 |
|---|---|---|
| tutorial/01_why_backprop.md | 8/10 | 动机、链式法则、`C=A@B` 的两个转置公式都对；(30→200) 与本 Part 64 不一致（卡点 5） |
| tutorial/02_forward_and_backward.md | 8.5/10 | **12 步 + Extra 逐式核验全部正确**（见下）；Step 9 形状笔误；BN 5 个子步骤只有代码+一句话，对畏难学生缺代数展开（尤其 12b 幂法则、12c 的 1/n、12e 两路合并） |
| tutorial/03_simplified_and_training.md | 5/10 | CE 简化式 `softmax - onehot → /n` 正确且直觉讲得好；BN 简化式口径错配 + "Bessel 校正"错误解释（卡点 1/3）——本章主菜出错 |
| tutorial/README.md | 8/10 | 导航、学习目标清晰；"手动推导 BN 的梯度（1 行简化版）"这条目标本身把学生引向带错的公式 |
| assignments/assignment_4（题面） | 6.5/10 | 分层题目好；Q4 签名带 `eps` 但 bnvar_inv 已给、eps 用不上；思考题 Q4 概念错 |
| scripts/01 | 9/10 | 形状逐行打印，loss 与 F.cross_entropy 对拍 3.32 一致 |
| scripts/02 | 9.5/10 | 7 参数对拍全 ✅（max diff ≤ 3.73e-09） |
| scripts/03 | 9/10 | 7 参数 + 7 中间变量双段验证，cmp() 给 max/mean 双 diff；UserWarning 小瑕疵 |
| scripts/04 | 9/10 | 三方法对比 + 热力图（Agg 正常） |
| scripts/05 | 4/10 | 逐步版 0.00e+00 ✅ 但简化版 4.56e-05 ❌，失败仍庆祝、exit 0、Bessel 错误解释 |
| scripts/06 | 6/10 | 完整训练闭环（quick 模式收敛 2.47/2.46/2.46）但继承口径错配，打印间隔问题 |

### 逐式核验明细（教程 02，全部对拍通过）

- Step 1：`dlogprobs[arange(B), Yb] = -1/B` ✓（mean→1/B、负号、scatter 三件事都对）
- Step 2：`dprobs = dlogprobs / probs` ✓（d log x = 1/x）
- Step 3：乘法两分支：`dcounts_sum_inv=(dprobs*counts).sum(1,keepdim)` + `dcounts=dprobs*counts_sum_inv` ✓（广播求和的"反传即求和"正确）
- Step 4：`dcounts_sum = dcounts_sum_inv * (-counts_sum**-2)` ✓（幂法则 -x⁻²）
- Step 5：sum 分支补 `+= ones * dcounts_sum` ✓（且教程强调了"累加"）
- Step 6：`dnorm_logits = dcounts * counts` ✓（exp 局部梯度=自身）
- Step 7：`dlogits = dnorm_logits.clone()` + `dlogit_maxes = (-dnorm_logits).sum(1,keepdim)` ✓（减法广播反传）
- Step 8：max 的梯度路由到 argmax 位置 `+=` ✓
- Step 9：`dh=dlogits@W2.T, dW2=h.T@dlogits, db2=dlogits.sum(0)` ✓（形状注释 (B,200) 笔误）
- Step 10：`dhpreact = dh*(1-h²)` ✓
- Step 11：gain/bias/raw 三分支 ✓
- Step 12（重点）：12a 乘法双分支 ✓；12b `d/dbnvar = -0.5·(var+eps)^-1.5` ✓；12c `dbndiff2 = dbnvar/batch_size`——**1/n 口径，与前向 `mean(0)` 自洽** ✓；12d `+= 2·bndiff·dbndiff2` ✓；12e `dbnmeani=-dbndiff.sum(0)`、`dhprebn = dbndiff + dbnmeani/n` ✓。**均值/方差两路的分子分母均无误，推导/代码/对拍三者一致（逐步版）**
- Extra：`dembcat=dhprebn@W1.T, dW1=embcat.T@dhprebn, db1=sum(0), dC 索引累加` ✓
- conv1d：本 Part 无此内容（Part 5 才涉及），符合范围
- 教程 03 CE 简化式 ✓（`∂L/∂logits_i = (p_i − 𝟙{i=y})/n`，脚本 04 实测 1.86e-09）

---

## 只改 3 件事

1. **修 BN 简化公式口径**（教程 03 公式块、脚本 05 一行公式区、脚本 06 训练循环）：在 1/n 前向下删去 `n/(n-1)`（第三项系数改 1），或保留公式但把前向/逐步反传统一改为 `1/(n-1)`。二选一，三处同步。
2. **修解释 + 失败要响**：教程 03 表格、脚本 05 文件头与"原理解释"里"Bessel 校正（无偏估计）"改为"系数必须与前向方差口径匹配：1/n 前向 → 系数 1；1/(n-1) 前向 → n/(n-1)"；assignment 思考题 Q4 同步改写。脚本 05 在任一 diff ≥ 1e-5 时以非零码退出、结尾明确写"验证失败"。
3. **修形状与网络口径的串写**：教程 02 Step 9 `(B,200)`→`(B,64)`；01 教程图注明"Part 3 示例用 200，本 Part 脚本用 64"；教程 03 清单补 `db1`/说明"脚本 06 无 b1"，并给 6️⃣ 补上 `demb` 的定义步骤。

## 最喜欢 3 处

1. **教程 02 开头的"变量命名约定"框**：明确 `dX = ∂L/∂X`、区分"上游梯度/局部梯度"——对我这种一看 `dx = dy * ∂y/∂x` 就发懵的人，这一个框把全文 12 步的阅读成本砍半，建议所有数学重的 Part 都抄这个做法。
2. **脚本 03 的两段式验证**：先 7 参数 cmp（带 max/mean 双 diff），再用 retain_grad 补齐 7 个中间变量（dlogits/dbnraw/dbndiff 都有实测数值）——"非叶子节点不存 .grad"这个坑是真实学习者必踩的，脚本正面演示了它。
3. **脚本 04 的三方法对比 + 热力图**：逐步 8 步 vs 简化 3 行 vs autograd 同屏对拍，再把 dlogits 热力图上用 `kx` 标出正确类位置——把"梯度只在正确类是 (p−1)/n、其余是 p/n"变成了看得见的东西。

## 费曼自检（先自答再核对的结果）

1. **CE 为什么是 softmax 减 onehot 再除 n？** 自答：L = −(1/n)Σ log p_y，p 对 logits 的雅可比是 diag(p) − p pᵀ，乘上 (e_y − p) 得 p − onehot。核对：教程 03 公式一致 ✓。
2. **BN 反传为什么绕不开整个 batch？** 自答：μ 和 σ² 是 batch 所有样本的函数，x_i 的梯度里必然含 Σdy 与 Σ(dy·x̂) 两个"全 batch 汇总项"。核对：教程 05 原理解释有一句"BatchNorm 的梯度依赖于整个 batch 的统计量" ✓。
3. **σ 用 1/n 还是 1/(n-1)？** 自答（核验后）：前向 `mean(0)` 是 1/n，与 PyTorch BN training 一致；逐步反传 12c 的 `dbnvar / batch_size` 也是 1/n——逐步版三者自洽 ✓。**简化式的 n/(n-1) 只属于 1/(n-1) 前向**，教程把它和 1/n 前向混装，这就是卡点 1。这一题教程的口头解释（Bessel/无偏）答错了，我的四象限实验给出正解。
4. **logit_maxes 的梯度为什么非零、只落在 argmax？** 自答：max 是分段线性，梯度路由到取 max 的那个位置，值等于上游和。核对：Step 8 ✓。
5. **为什么减 logit_maxes 不影响梯度结果？** 自答：softmax 对行内平移不变，但减法本身引入一条经 argmax 的额外梯度路，Step 7+8 的 `+=` 正是这条路的账。核对 ✓（这也是课后练习 3 考的点，出得好）。

诚实记录：核验中我自己的诊断脚本先后犯过 3 个错（漏乘 gain、`dvi` 误用 raw0 当 diff0、简化实验口径混杂），全部用有限差分仲裁定位修正——autograd 与有限差分一致（9.5e-05），教程逐步版与 autograd 一致（0.00），错的始终是我，最后才轮到教程的简化公式被判 FAIL。这轮"证明教程错"的代价让我真正学会了 BN 反传。

---

## 附录 A：6 个脚本实跑记录

环境：`/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python`，`MPLBACKEND=Agg`，scratch=`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S1_P4/`（日志 `log_*.txt`）。

| 脚本 | exit | 关键对拍（max diff） | 判定 |
|---|---|---|---|
| 01_forward_pass_steps.py | 0 | 手动 loss = F.cross_entropy（loss=3.3177，差异打印）| 通过 |
| 02_backprop_step_by_step.py | 0 | dC 3.73e-09 / dW1 3.73e-09 / db1 2.04e-09 / dbngain 1.16e-09 / dbnbias 1.86e-09 / dW2 0.00 / db2 0.00 | **全部通过** |
| 03_verify_gradients.py | 0 | 参数 7/7 ✅（≤3.73e-09）；中间变量 7/7 ✅（dembcat 2.79e-09, dhprebn 1.86e-09, dhpreact 1.63e-09, dh 2.33e-09, dlogits 5.59e-09, dbnraw 1.63e-09, dbndiff 1.86e-09）；有 UserWarning | **全部通过** |
| 04_cross_entropy_backward.py | 0 | 逐步 vs autograd 5.59e-09；简化 vs autograd 1.86e-09；逐步 vs 简化 5.82e-09；热力图已生成 | **全部通过** |
| 05_batchnorm_backward.py | 0 | 逐步 vs autograd **0.00e+00 ✅**；**简化 vs autograd 4.56e-05 ❌（>1e-5）**；逐步 vs 简化 4.56e-05 ❌；**脚本仍 exit 0 且结尾打印"验证完成/面试利器"** | **部分失败，被庆祝文案掩盖** |
| 06_manual_training.py --quick | 0 | 1000 步：loss 3.6944 → Train 2.4667 / Dev 2.4619 / Test 2.4581，采样 20 个名字；但训练用的 BN 简化式带口径错配（n=32 时每步梯度偏差 ~4.6e-05 量级，SGD 下未致命）| 跑通但公式带病 |

注：06 完整模式 200000 步为控时未跑，用 --quick（脚本自带开关）。

## 附录 B：BN 方差口径四象限独立实验

`bn_final_verdict.txt`（n=8, d=5, eps=1e-5，autograd 为基准，autograd 本身经有限差分交叉验证）：

| 前向方差口径 | 简化式第三项系数 | max diff | 判定 |
|---|---|---|---|
| 1/n 有偏（=教程前向） | 1 | 2.980e-08 | PASS |
| 1/n 有偏（=教程前向） | n/(n-1)（=教程公式） | 1.526e-02 | **FAIL** |
| 1/(n-1) 无偏 | 1 | 1.249e-02 | FAIL |
| 1/(n-1) 无偏 | n/(n-1) | 3.725e-08 | PASS |

结论：简化式系数与前向口径**一一配对**，不存在"给 1/n 公式打 Bessel 补丁"的说法。教程前向、逐步反传均为 1/n，唯独简化式（教程 03 / 脚本 05 / 脚本 06）带 1/(n-1) 口径的 n/(n-1)。
