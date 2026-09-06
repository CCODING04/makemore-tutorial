# 修改说明 · Part 4（手动反向传播）整改（T2）

> 整改人：T2 · 日期：2026-09-04
> 输入：plan_P04（T1 预审）+ S1/S2/S3 三份 report_P04 + 必修清单（T0 确认 7 项）
> 原则：REPO 只读；全部修改写镜像；无 wontfix（3 项 disputed）；文档口径修复必须以镜像实跑数值为据

## 一、镜像修改文件清单（16 文件 + 3 新图）

**教程（4）** `REVIEW/courses/Part4_backprop/tutorial/`
- `03_simplified_and_training.md` — **BN 一行公式删 `n/(n-1)` 改系数 1（P0）** + 「第三项系数由前向方差口径决定」警示框（替换错误"Bessel 校正"归因）+ Step 12→一行公式的代数来源；CE 补 3 行 LaTeX 推导（log-sum-exp 导数=softmax）；**新增「训练出了什么？」闭环小节**（实测数字表：初始 3.3177 vs ln27≈3.296、Train/Dev/Test 2.4665/2.4617/2.4579、采样示例）+ loss 曲线插图；清单 6️⃣ 补 `demb` 定义、开头注明"无 b1（被 BN β 取代）"；伪码补 `embcat` + 示意标注；作业表 Q2 改 4 函数实名；G2×4 全部 LaTeX 化；插 04 热力图与 05 对拍图（注明同源）
- `02_forward_and_backward.md` — **新增「前向传播参考代码」节**（12 步变量全部有定义，G10/C1-D）；**新增「广播的反传元规则」框**（三类用例）；**新增「autograd 只存叶子：is_leaf 与 retain_grad」小节**（坑复现 + 规范演示 + 指向脚本 03，C1-A）；**`(B,200)`→`(B,64)` 笔误修正** + 章首规模约定框（n_hidden 64 vs 200 统一叙事，T0 必修3）；G2×12：12 步全部改"LaTeX 前向关系+局部导数 → Python 代码"双层；Step 8 加 `squeeze(1)` 提示、Step 10 加"以输出表达省重算"动机、Extra 加 db1/脚本 06 差异说明
- `01_why_backprop.md` — G2×3（链式法则、计算图注释、矩阵求导全部 LaTeX 化）；架构图标注"Part 3/脚本 06 配置 200，本 Part 脚本 01-05 用 64"；loss 示例加示意标注
- `README.md` — 学习目标补 retain_grad 一条、BN 目标加"系数与方差口径配套"

**脚本（3 改 + 3 原样补齐）** `REVIEW/courses/Part4_backprop/scripts/`
- `05_batchnorm_backward.py` — **公式系数改 1（P0）**；docstring/原理解释/总结全部改口径配套表述；**判定分支修复**：按 all_ok 输出结论，超阈值打印"⚠️ 验证失败+口径提示"并 `sys.exit(1)`；新增对拍误差条形图存 `images/bn_backward_comparison.png`；删"（原脚本 bug）"施工痕迹
- `06_manual_training.py` — **训练公式同步删 `n/(n-1)`（P0）**；`print=functools.partial(print, flush=True)`（G8）；`STEPS` 环境变量短程档 + `log_every` 自适应（--quick 从只打 1 行变 10 行，默认档不变）；新增 loss 曲线存 `images/loss_curve_manual_training.png`（G4）；修复重复 `import sys`
- `04_cross_entropy_backward.py` — 热力图文字全英文（G4 字体风险）；产图迁 `images/dlogits_heatmap.png`（教程 03 插图同源）
- `01/02/03` — 原样纳入镜像（策略与 P02 一致：全量文件集），回归实跑全绿

**作业（3）** `REVIEW/assignments/assignment_4/`
- `test_backprop_exercises.py` — **return bool → 真 assert**（`_Checks` 收集 + `assert_all`，pytest 下逻辑失败必红；直跑模式捕获 AssertionError、跳过不计分、按结果定 rc）（P0）；**Q3/Q2 前置缺失 KeyError → 优雅跳过**（`_skip_unready`：pytest SKIP / 直跑"⏭️ 跳过不计入"）（P0）；删除 `dh_ref` 死代码；Q4 阈值 1e-4→**1e-5** 并更新注释（修后实测 9.31e-10，旧错公式 4.83e-05 会被正确判红）
- `README.md` — **Q1/Q2/Q4/Q5 签名与骨架逐一对齐**（Q2 展示 4 函数实名、Q4 删多余 eps）；Q4 要求改"系数与前向方差口径配套"完整表述；思考题 Q4 重写（正确口径答案）；思考题 Q2 误差量级改"1e-9~5e-5 随算子/维度而变"；运行方式补 pytest 说明
- `backprop_exercises.py` — Q4 docstring 公式删 `n/(n-1)` + 口径警示（题目代码本体未动）

**图（3 新 + 1 原样）** `REVIEW/courses/Part4_backprop/images/`
- `dlogits_heatmap.png`（04 生成，英文化）、`bn_backward_comparison.png`（05 生成）、`loss_curve_manual_training.png`（06 生成）、`cell018_output01.png`（原样）

**台账** `REVIEW/ledger/ledger_P04.md`（**22 条：P0×4 / P1×11 / P2×7**，全部 fixed；disputed×3）

## 二、验证记录（全部实跑，scratch/t2_P4/）

1. **对拍修复前后**：修前（T1/S1 实证）简化 vs autograd = **4.56e-05 ❌**（脚本仍 exit 0 打 🎉）；修后 = **9.31e-10 ✅**、逐步 vs autograd 0.00e+00、逐步 vs 简化 9.31e-10，rc=0。判定阈值 1e-5 显式打印。
2. **FAIL 分支**：注入旧 `n/(n-1)` 系数副本实跑 → 精确复现 4.56e-05 ❌ → "⚠️ 验证失败！…口径是否与前向一致" → **rc=1**。
3. **作业测试三态**：
   - 参考答案 + 修复 test：`pytest -W error::PytestReturnNotNoneWarning` → **5 passed 零警告**（Q4 max diff 9.31e-10）；直跑 → "得分: 5/5"；
   - 错误公式注入：Q4 **FAILED**（AssertionError，max diff 4.83e-05——与 S2 当年实测完全吻合，证明新阈值能抓住旧 bug）；
   - 骨架未作答：**3 failed + 2 skipped**，无 KeyError 崩溃（原状是 4 passed+1 failed 且 failed 是崩溃不是断言）。
4. **脚本回归**：01 rc=0（loss 3.3177）；02 rc=0（7 参数全 ✅ ≤3.73e-09）；03 rc=0（15 项 ✅）；04 rc=0（三方对拍 5.59e-09/1.86e-09/5.82e-09 + 热力图英文化产出）；06 --quick rc=0（Train/Dev/Test 2.4665/2.4617/2.4579）、STEPS=200 rc=0（10 行日志）。
5. **机检**：check_latex.py 对 5 个改动 md **全部 0 问题**（过程中抓出并修复 02 章 Step 8 公式内中文 2 处）。

## 三、Disputed 清单（详见台账）

- **D1 roadmap 节点 4**："Q4-Q5 batchnorm_backward/(拓展)手动训练"表述不准 + "pytest 全绿"需骨架初始态预期管理措辞 → docs 侧微修，待 T0 派发。
- **D2 脚本 06 默认档 200k 步未全量实跑**（约 12 分钟 CPU）：quick/STEPS 档已验、公式改动同源，最终 loss 数值建议 T0 安排一次后台全量回归。
- **D3 脚本 03 非叶子 .grad 访问 UserWarning**：倾向保留（警告即教学点，正文/脚本均有解释），是否消警留 T0 定夺。

## 四、通用问题候选（供课程级沉淀）

1. **同一公式三处拷贝、无单一事实来源**：BN 口径错配能存活，正因教程/脚本/作业各写一份公式且无人对拍——建议约定"公式唯一出处（教程）+ 脚本/作业引用"，或至少加一个 CI 级对拍脚本（本 Part 的 05 修后已可充当该角色）。
2. **判定型脚本的输出必须与判定一致**：❌ 之后打 🎉、exit 0，是三个学生共同的最大困惑源；建议全课程约定"有对拍的脚本必须按 all_ok 分支输出 + rc 退出"。
3. **作业测试的双模式约定缺失**：直跑给得分、pytest 给红绿是两种验收，return bool 会让后者静默失效——建议把本 Part 的 `_Checks + assert_all + _skip_unready` 模板沉淀为作业测试范式（P2 同类问题的终版答案）。

## 五、好写法候选（建议跨 Part 复制）

1. **教程 02「变量命名约定框」**（dX=∂L/∂X、上游 vs 局部梯度）：三名学生一致点名，数学重章标配。
2. **脚本 03「先埋坑再解答」两段式验证**：先让非叶子 .grad 为 None 触发警告，再 retain_grad 重跑——自学友好的最佳教学脚本结构。
3. **12 步推导「前向关系→局部导数→代码」三段式 + 累加⚠️警示**：S1 逐式核验零错，是全课程最可靠的推导模板。
