# 问题台账 · Part 4（手动反向传播）· 全量批1

> 编号规则：P04-C{cc}-{SRC}-{nn}；cc∈{00=README,01,02,03,SC=scripts,AS=assignment_4}；SRC∈{S1,S2,S3,T,SB=旧基线/T1计划}
> 状态：fixed=已修复且教师侧实跑验证；disputed=争议/跨 Part 待 T0 裁决（禁止 wontfix）
> 验证环境：REPO .venv（torch 2.6.0 / pytest 9.1.1），全部在 REVIEW 镜像上实跑，scratch/t2_P4/

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 | 复核 |
|------|--------|------|------|------|------|----------|----------|------|
| P04-C03-S1-01 / S2-05 / S3-06 / T-01 | P0 | fixed | 四方命中（T0 必修1） | BN 简化反传公式第三项系数 `n/(n-1)`（无偏口径产物）套在 1/n 有偏前向上：教程 03 / 脚本 05 / 脚本 06 / 作业 Q4 四处同错，对拍实测 4.56e-05 ❌（>1e-5） | tutorial/03、scripts/05、scripts/06、assignment_4/{README,exercises} | 按 T0 裁决：前向保持 1/n 有偏，公式删 `n/(n-1)` 改系数 1，四处同步；并补"系数与前向方差口径配套"说明 | 修后脚本 05 实测：逐步 vs autograd 0.00e+00，**简化 vs autograd 9.31e-10** ✅（修前 4.56e-05）；参考答案 pytest Q4 max diff 9.31e-10 | 待用户 |
| P04-CSC-S1-02 / S2-03 / S3-01 / T-02 | P0 | fixed | 三方命中（T0 必修1） | 脚本 05 对拍超阈值仍 exit 0 且结尾无条件打印"🎉 验证完成/面试利器"，失败被庆祝文案淹没 | scripts/05 对比/总结段 | 按 all_ok 分支输出：任一 diff ≥1e-5 → 打印"⚠️ 验证失败 + 口径检查提示"并 `sys.exit(1)`；阈值 1e-5 显式打印 | 注入旧系数实测：4.56e-05 ❌ → "⚠️ 验证失败" → **rc=1**；修后全绿 rc=0 | 待用户 |
| P04-CAS-S2-01 / T-03 | P0 | fixed | S2🔴 实证 + T0 必修4 | 测试函数全部 return bool 不 assert：pytest 下逻辑失败仍 PASSED + 5 个 PytestReturnNotNoneWarning，只有直跑才可信 | assignment_4/test_backprop_exercises.py | 重写为真 assert（_Checks 收集→assert_all），直跑模式捕获 AssertionError 计失败并按 n_pass/n_done 退出码；-W error::PytestReturnNotNoneWarning 下 0 警告 | 参考答案 5/5 PASSED 零警告；注入旧错误公式实测 Q4 **FAILED（AssertionError，4.83e-05）** | 待用户 |
| P04-CAS-S2-02 / S3-04 / T-04 | P0 | fixed | S2🔴 + S3🔴（T0 必修4） | Q3 测试直接取 `cache['logits']`：Q1 未实现时 KeyError 崩溃而非优雅跳过，学生第一跑见 traceback | test_q3（test_q2 同型隐患） | 前置检查 `cache is None or 'logits' not in cache` → `_skip_unready()`：pytest 下转 SKIP、直跑下计"跳过不计入"，不再崩溃 | 骨架未作答实测：**3 failed + 2 skipped**，无 KeyError；Q2/Q3 显示"⏭️ 跳过（前置题未完成）" | 待用户 |
| P04-C02-S3-02 / T-05 | P1 | fixed | S3🔴 roadmap 点名（T0 必修2） | retain_grad/is_leaf 语义（roadmap 真考点）教程正文 0 覆盖，仅藏于脚本 03 L236-275 | tutorial/02 验证节后 | 新增「autograd 只存叶子：is_leaf 与 retain_grad」小节：机制解释 + 坑复现 3 行 + retain_grad 规范演示 + 指向脚本 03 完整版；README 学习目标同步补条目 | check_latex 0 问题；正文与脚本 03 实测数值（中间变量最大 5.59e-09）互相引用 | 待用户 |
| P04-C02-S1-03 / S2-04 / S3-03 / T-06 | P1 | fixed | 三方命中（T0 必修3/5） | n_hidden 口径混乱：01 章图 (30→200)（Part 3 配置）、脚本 01-05 用 64、06 用 200；02 章 Step 9 注释 `(B, 200)` 笔误（实为 (B,64)）；03 清单无 b1 无说明 | tutorial/01 图、tutorial/02 Step9、tutorial/03 清单 | 01 图标注"Part 3/脚本 06 配置 200，本 Part 脚本 01-05 为 64"；02 章首加规模约定框；(B,200)→(B,64)（含 dW2 (64,27)）；Extra 加 db1/脚本 06 无 b1 说明；03 清单开头注明无 b1 缘由 | 全部 md 过机检；02 章前向参考代码实测形状 (32,64) 与脚本一致 | 待用户 |
| P04-C03-S1-04 / T-07 | P1 | fixed | S1🟡 概念错误（T0 必修1 连带） | "Bessel 校正（无偏估计）"解释是错误归因：前向根本没用无偏方差，混淆了前向统计量选择与反传系数来源 | tutorial/03 拆解表、scripts/05 文件头+原理解释、assignment README 思考题 Q4 | 全部改为"系数必须与前向方差口径配套：1/n 前向→系数 1；1/(n-1) 前向→n/(n-1)；混用=系统性偏差非浮点噪声"，并给从 Step 12 合并的代数来源 | 教程 03 / 脚本 05 输出 / 作业 README 三处口径表述一致；S1 四象限实验结论（配对关系）被正文采纳 | 待用户 |
| P04-CAS-S2-06 / T-08 | P1 | fixed | S2🟡 一致性核对表 #14 + S1🟡 | 作业 README 与骨架签名不一致：Q1 参数列表式 vs `forward_pass(params, Xb, Yb=None)`；Q2 `backward_step(step_name,...)` vs 实际 4 个独立函数；Q4 多余 `eps` 形参（bnvar_inv 已给，eps 用不上）；Q5 参数表不符 | assignment_4/README.md Q1/Q2/Q4/Q5 | 四题签名全部改为与 backprop_exercises.py 逐一对应；Q2 展示 4 函数真实签名；Q4 删 eps | README 与骨架逐参核对一致 | 待用户 |
| P04-C02-S3-05 / T-09 | P1 | fixed | S3🟡（T0 必修5） | "广播的反向 = 对广播维求和"元规则在 Step 3/5/7/9/12 隐式使用但从未显式总结 | tutorial/02 命名约定区 | 新增「广播的反传元规则」框：广播=复制→相加的反传=求和，列本 Part 三类用例（(B,1) 求和 / ones_like 加权 / bias sum(0)） | 与 Step 3/5/7/9/12c/12e 代码逐条对应 | 待用户 |
| P04-C02-T-10 / T1(C1-D) + G10 | P1 | fixed | T1 计划（T0 必修6） | 02 章标题承诺"前向"但正文只有 ASCII 图，12 步引用的 10+ 个前向变量在正文无定义，按序拼凑不可运行；03 章伪码 `embcat` 未赋值、`n/max_steps/Xb/Yb` 无出处；01 章 loss 示例无示意标注 | tutorial/02、tutorial/03、tutorial/01 | 02 章新增「前向传播参考代码」节（全变量、与脚本 01 同源、B=32/n_hidden=64）；03 伪码补 `embcat = emb.view(...)` 并整段标注"示意伪码，完整见脚本 06"；01 章补"（示意：logits/Yb 由上面网络前向得到）" | 参考代码变量与 12 步逐一对上；实测与脚本 01 前向一致（loss 3.3177） | 待用户 |
| P04-C03-T-11 / T1(C1-H) + G6/G7 | P1 | fixed | T1 计划（T0 必修7 连带） | "手动训练"缺闭环：评估（train/dev/test）与采样只在脚本 06；正文无任何锚点数字（初始 loss、最终 loss） | tutorial/03 训练节 | 新增「训练出了什么？」小节：实测数字表（全量初始 3.3177 vs ln27≈3.296、Step0 batch 3.6944、Train/Dev/Test 2.4665/2.4617/2.4579、采样示例）、loss 曲线插图、三档运行方式说明 | 数字全部来自本轮镜像实跑（run06_quick.log / run06_steps200.log），可复现 | 待用户 |
| P04-C03-T-12 / T1(C1-E) | P1 | fixed | T1 计划（T0 必修7） | CE 解析解直接给出无推导，对"数学最重章"断层 | tutorial/03 CE 节 | 补 3 行 LaTeX 推导：单样本 L=−log p_y → log-sum-exp 导数恰为 softmax → p−onehot → /n | check_latex 0 问题 | 待用户 |
| P04-CSC-SB-13 / T1 计划§五 | P1 | fixed | T1 机检（T0 必修6） | G2 ×20：数学推导全压代码块/反引号（01:3、02:12、03:4、作业:1） | 四个 md | 全部移出改 LaTeX（`$$` 单行、公式内无中文、列表内行内公式、Python 代码留代码块）；02 章 12 步改为"LaTeX 前向关系+局部导数 + Python 代码"双层结构 | check_latex 5 个 md 全部 0 问题（含修复 02 L192 公式内中文 2 处） | 待用户 |
| P04-CSC-SB-14 / T1(C1-G) | P1 | fixed | T1 机检（T0 必修6） | G4 ×3：脚本 06 有 lossi 无图；04 图内文字中文且有字体缺字风险、产图存 scripts/ 无 md 引用；cell018 与 04 产图关系未说明 | scripts/04/05/06、images/、tutorial/03 | 04 热力图英文化+改存 images/dlogits_heatmap.png+03 章插图；05 新增对拍误差条形图 images/bn_backward_comparison.png+03 章插图；06 新增 loss 曲线 images/loss_curve_manual_training.png+03 章插图；02 章 cell018 处注明"notebook 原始单图，04 生成更完整对比图见 03 章" | 3 张 PNG 均实跑生成于 REVIEW/courses/Part4_backprop/images/；04 rc=0（三方对拍 5.59e-09/1.86e-09/5.82e-09） | 待用户 |
| P04-CSC-SB-15 / S2-02 | P1 | fixed | T1 计划 + S2🔴 time-blocked（T0 必修6） | G8：脚本 06 全部 print 无 flush（长训练丢日志）；--quick 1000 步打印间隔 10000 只打 1 行；无短程档提示；L90-91 重复 `import sys` | scripts/06 | `print = functools.partial(print, flush=True)` 全量实时刷出；`STEPS` 环境变量自定义步数（默认档不变）；`log_every = 10000 if max_steps>=10000 else max(1, max_steps//10)`；docstring 写明三档用法；去重 import | STEPS=200 实测 rc=0（10 行日志，Train 2.7494）；--quick rc=0（10 行日志）；默认档 log_every=10000 与原版一致 | 待用户 |
| P04-C03-S1-06 | P2 | fixed | S1🟡 | 03 章清单 6️⃣ `dC[Xb] += demb` 但 demb 前一步未定义 | tutorial/03 清单 | 拆成两步：`demb = dembcat.view(emb.shape)` → `dC[Xb[i,j]] += demb[i,j]` | 与脚本 06 逐行一致 | 待用户 |
| P04-C02-S2-07 | P2 | fixed | S2🟡 | Step 8 `squeeze()` 在 B=1 时连 batch 维一起压导致形状错位，无提示 | tutorial/02 Step 8 | 加 💡 提示：稳妥写法 `squeeze(1)` 只压 max 维 | — | 待用户 |
| P04-C02-S3-07 | P2 | fixed | S3🟡 半覆盖 | tanh 局部导数用输出 h 表达"省一次重算"的动机未点破（roadmap 半覆盖项） | tutorial/02 Step 10 | 加 💡：$1-h^2$ 以输出表达免重算 tanh，"重算 vs 缓存"权衡即 autograd 常见做法 | — | 待用户 |
| P04-CSC-S1-07 | P2 | fixed | S1🟡 | 脚本 05 注释"（原脚本 bug）"为施工痕迹残留 | scripts/05 L123 | 改为教学表述"非叶子节点 .grad 默认不保存，必须 retain_grad()" | — | 待用户 |
| P04-CAS-S2-08 | P2 | fixed | S2🟡 | test_q2 内 `dh_ref`（h.sum() 反传段）从未使用的死代码 | test_q2 | 随测试重写删除 | 重写后测试不含未用变量 | 待用户 |
| P04-CAS-S2-09 / S3-05 | P2 | fixed | S3🟡 + P0 连带 | 思考题 Q2 把手动 vs autograd 误差解释为"~1e-7"，量级偏大 100 倍，且旧公式 4.56e-05 会被此条误导向"浮点噪声" | assignment README 思考题 Q2 | 改为"两条独立 float32 计算链，实测约 1e-9 ~ 5e-5 随算子/维度而变；>1e-3 更可能是推导错误" | 与修后实测（1e-09 量级）及旧 bug 实测（4.8e-05）两端吻合 | 待用户 |
| P04-C00-T-13 | P2 | fixed | S3🟡 | README 学习目标未列 retain_grad；"BN 1 行简化版"目标把学生引向旧错公式 | tutorial/README | 目标清单补 retain_grad 一条；BN 目标加"第三项系数与前向方差口径配套" | — | 待用户 |
| P04-CAS-T-14 / T1(C1-0) | P2 | fixed | T1 计划（核实类） | 作业入口文件名与其余 Part 惯例是否一致存疑（无 assignment.md） | assignment_4/ | 核实：P02 批1已裁决入口统一为 README.md（P02-CAS-S2-06），本 Part 已是 README.md，无需改动 | 与 ledger_P02 裁决一致 | 待用户 |

## Disputed（跨 Part / 待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P04-D1 | roadmap 节点 4 表述两处不准：作业描述"Q4-Q5 batchnorm_backward（🌟）/(拓展)手动训练"实为 Q4=BN、Q5=manual_train（非两个 BN）；"pytest 全绿"与骨架初始态（未作答=红/SKIP）需预期管理措辞 | roadmap 属 docs/，建议 T0 派发 docs 侧微修，不阻塞本 Part 关闭 |
| P04-D2 | 脚本 06 默认档 200000 步（约 12 分钟 CPU）本轮未实跑全量：--quick 与 STEPS=200 档已验、公式改动路径与 quick 完全同源，但最终 loss 数值（教程引用的是 quick 档）未经默认档复核 | 建议 T0 安排后台 12 分钟全量回归一次；预期 Train ≈2.4x（SGD 非确定性 ±0.01） |
| P04-D3 | S1 报告的脚本 03 非叶子 `.grad` 访问 UserWarning 未修：属 torch 行为演示的一部分（正文/脚本均有解释），若 T0 认为"教学演示不应报警告"可改为先 `is_leaf` 判断再访问 | 保持现状（警告即教学点）；如需消警是 3 行改动，留 T0 定夺 |

## 验证记录（全部实跑，scratch/t2_P4/）

1. **脚本 05 修后对拍**：逐步 vs autograd 0.00e+00、简化 vs autograd **9.31e-10**、逐步 vs 简化 9.31e-10，全 <1e-5，rc=0（run05_fixed.log）。
2. **FAIL 分支有效性**：注入旧 `n/(n-1)` 系数副本 → 复现 4.56e-05 ❌ → "⚠️ 验证失败" → **rc=1**。
3. **作业 pytest**：参考答案（scratch/ref）+ 修复后 test → **5 passed** 零警告；错误公式注入版 → Q4 **FAILED**（4.83e-05，AssertionError）；骨架初始态 → **3 failed + 2 skipped** 无 KeyError 崩溃；直跑模式 → "得分: 5/5"。
4. **脚本回归**：01（loss 3.3177）/02（7 参数全 ✅）/03（15 项 ✅）/04（三方对拍 ✅ + 热力图英文化迁 images/）全部 rc=0；06 quick（2.4665/2.4617/2.4579）与 STEPS=200 rc=0。
5. **机检**：check_latex.py 对 5 个改动 md 全部 0 问题（含 02 章 L192 公式内中文 2 处修复后复检）。
