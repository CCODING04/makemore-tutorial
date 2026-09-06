# 问题台账 · Part 2（MLP）· 全量批1

> 编号规则：P02-C{cc}-{SRC}-{nn}；cc∈{00=README,01,02,03,SC=scripts,AS=assignment_2}；SRC∈{S1,S2,S3,T,SB=旧基线/T1计划}
> 状态：fixed=已修复且教师侧验证；disputed=争议/跨 Part 待 T0 裁决

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 | 复核 |
|------|--------|------|------|------|------|----------|----------|------|
| P02-C02-S1-01 / S3-01 / T-01 | P0 | fixed | S1🔴+S3🔴 双命中 | `emb.view(-1) → (64,)` 数学错误，实为 32×3×2=192 | tutorial/02 view 详解 | 改 (192,)，补拍扁顺序说明，复查全章 shape 标注 | scratch 实测 `view(-1)=(192,)`；全章其余 shape 复核无误 | 待用户 |
| P02-C02-S1-02 / T-02 | P0 | fixed | S1🔴+S3 缺公式 | CrossEntropy 全章无公式无数值例 | tutorial/02 CE 节 | 补 softmax+NLL LaTeX 公式 + 3 类 toy 数值例表 | 手算 0.170 ≡ `F.cross_entropy=0.1698` 双路验证；check_latex 0 问题 | 待用户 |
| P02-C02-S1-03 / S2-07 / S3-05 / T-03 | P0 | fixed | S1🔴+S2🟡+S3🔴 三命中 | 初始 loss sanity check 缺失：ln(27)≈3.296 基线、randn≈19.5 现象无解释 | tutorial/02 CE 节后 | 新增「初始 loss sanity check」小节：理论基线 / 全零初始化恰为 ln27 / randn 方差大导致 19.5 / 与作业题5、Part3 衔接 | 04 脚本实测 19.5116；全零 logits 实测 3.2958；ln27=3.2958 | 待用户 |
| P02-C03-S3-02 / T-04 | P0 | fixed | S3🔴+教师 | 03 章 Q3 答案自相矛盾：定性过拟合却给"增大模型"、末尾又改欠拟合 | tutorial/03 练习 Q3 | 重写为三步诊断（绝对值→性质判断→按规则选动作），显式给出"过拟合不得加容量" | 重写稿与本章实测 gap≈0 证据自洽 | 待用户 |
| P02-CAS-S2-01 / T-05 | P0 | fixed | S2🔴 实证 + 教师 | 题5 骨架注释 `torch.randn(..., requires_grad=True) * 0.1` 非叶子张量 → grad None → TypeError | assignment_2/mlp_exercises.py | 注释改 `(randn*scale).requires_grad_(True)` 并加警示；test 补 `p.grad is not None` 回归断言 | 旧写法实测 TypeError（含 PyTorch 非叶子告警）；新断言可捕获；参考答案 + 新测试 pytest 5/5 绿 | 待用户 |
| P02-CAS-S2-02 / T-06 | P0 | fixed | S2🔴 + T1(C1-18/README 三处失配) | README 题5：缺 seed 参数、2.2/~2.3/2.5 阈值矛盾、提示代码 Xb/Yb/lr 未定义 | assignment_2/README.md | 签名补 `seed=2147483647`；目标统一为"完整 200k 步 <2.3，测试 1000 步 <2.5"；提示代码补 build_dataset/初始化缩放/mini-batch ix | 与 test_tuning 断言口径一致 | 待用户 |
| P02-C02-S2-03 / T-07 | P0 | fixed | S2🔴（data_ptr 实测）+ T1(C1-8) | "C[X] 直接查表，零拷贝"说法错误（高级索引返回拷贝） | tutorial/02 one-hot 关系注 | 改为"返回新张量、非视图；省的是 one-hot 大矩阵的构造与乘法"，并补 (N,3,27)@(27,2)→(N,3,2) 形状链 | data_ptr 实测不同；allclose 等价性实测 True | 待用户 |
| P02-C03-S2-04 / S3-01 / T-08 | P0 | fixed | S2❌+S3❌ 双命中 | 采样示例 "mora/kiah/mel" 不可复现（源自原视频大配置）；教程 seed 2147483647 vs 脚本 +10 | tutorial/03 采样节、scripts/07 | 种子统一为 2147483647；示例输出换为本配置实测；注明 mora/kiah/mel 出处 | 07 全量实跑：junide/janasar/prafay/adin/koi/...（scratch/t2_P2/run07_full.log） | 待用户 |
| P02-C03-S2-05 / S3-08d / S1🟡 / T-09 | P0 | fixed | 三方命中 | 05/06/07 20000 步 CPU 超时被杀且 print 无 flush，日志全丢 | scripts/05/06/07、tutorial/03 | `STEPS` 环境变量短程档 + 关键 print `flush=True`；教程/README 注明用法 | STEPS=2000/500 实跑通过；默认档与原版 2000 步输出逐位一致（2.5491/2.5999/2.5985/2.6021）；全量 20000 步实跑完成 | 待用户 |
| P02-C02-S3-03 / T-10 | P1 | fixed | S3🔴（roadmap 节点2 面试点）+教师 | "嵌入的梯度是稀疏的"教程零覆盖 | tutorial/02 C[X] 节后 | 新增「延伸（面试点）」小节：one-hot 求导 → scatter-add + 5 行实验 | 实测非零梯度行恰为 {0,5,13} | 待用户 |
| P02-C02-T-11 / T1(C1-5/C1-10) | P1 | fixed | T1 计划 | 正文参数初始化全缺 requires_grad、parameters 容器未定义，照抄跑不通 backward | tutorial/02 Step1-3 | C/W1/b1/W2/b2 全部补 requires_grad=True + Step3 后补 parameters 列表（3481） | 03 章训练代码可自洽运行；07 全量实跑通过 | 待用户 |
| P02-C02-T-12 / T1(C1-6) | P1 | fixed | T1 计划 | Loss 曲线图与生成代码放在 02 章，stepi/lossi 未定义（训练在 03 章才讲） | tutorial/02→03 | 图与代码移至 03 章 Minibatch 之后，训练循环补 stepi/lossi 记录 | 图文件路径不变，时序自洽 | 待用户 |
| P02-C03-S3-04 / T-13 | P1 | fixed | S3🔴 | 教程无任何真实训练锚点数字，面试无数字可背 | tutorial/03 Minibatch 节 | 加「本章实测数字」框：step0≈18.2、train 2.3749/dev 2.3710/test 2.3725 | 05 全量实跑（run05_full.log），数字可复现 | 待用户 |
| P02-C03-S3-05 | P1 | fixed | S3🟡 | "元音 (a,e,i,o,u) 聚在一起"以偏概全，u 是反例 | tutorial/03 可视化节 | 改口"a/e/i/o 明显成簇（可自算距离），u 是例外" | S3 实测元音均距 0.599<辅音 0.849、u 例外 | 待用户 |
| P02-C01-T-14 / T1(C1-2) | P1 | fixed | T1 计划（S1/S2 连带） | Part2 正文从未给 stoi/itos 构建代码，正文拼不出可运行程序 | tutorial/01 | 新增「字符映射」小节（与 Part1/作业统一写法） | 01-04 脚本实跑输出不变 | 待用户 |
| P02-C01-T-15 / T1(C1-3) / S2 签名表#2 | P1 | fixed | T1 计划 + S2🟡 | build_dataset 未定义即调用；脚本 02 带 itos 参、03-07 不带，签名不统一 | tutorial/01、scripts/02 | 01 正文给出与脚本一致的函数定义；scripts/02 去掉未用的 itos 参 | 四方（教程/02/03-07/作业）签名一致；02 脚本输出不变 | 待用户 |
| P02-C03-S3-06 | P1 | fixed | S3🟡 | `linspace` 注释"指数空间"措辞错误；为何指数采样无解释 | tutorial/03 lr 搜索 | 注释改"指数上线性取点，10** 后才是指数空间"；补"跨 3 个数量级，线性采样 99.9% 挤在大 lr 侧"两句 | 数学自洽 | 待用户 |
| P02-C03-S3-07 | P1 | fixed | S3🟡 | "一个 epoch 要算 228146 次前向+反向"措辞含糊 | tutorial/03 开头 | 改"一步全量梯度要同时处理全部 228146 个样本" | — | 待用户 |
| P02-CSC-SB-01 / 旧卡点7 | P1 | fixed(P2侧) | 旧基线 + T1 | Part1↔Part2↔作业 字符映射写法两种风格并存 | scripts/01-07、tutorial/01 | 选定 Part1/作业风格（a=1..z=26，'.'=0），Part2 教程与 7 个脚本全部统一；Part1 侧零改动 | 01-07 实跑输出逐位不变；disputed 见下 D1 | 待用户 |
| P02-CAS-S2-06 | P1 | fixed | S2🟡 + T1 | 作业 README 文件结构自称 assignment.md，实际 README.md | assignment_2/README.md | 改为 README.md | — | 待用户 |
| P02-C01-S2-07 | P1 | fixed | S2🟡 | `print(X[:5])` 配文"先看前 5 个名字"易误读成 5 行=5 个名字 | tutorial/01 | 配文改"前 5 行恰好全部来自第一个名字 emma" | 与实测一致 | 待用户 |
| P02-C00-T-16 / T1(C1-0) | P2 | fixed | T1 计划 | README 导航表 03 行漏列 Embedding 可视化 | tutorial/README.md | 内容列补"Embedding 可视化" | — | 待用户 |
| P02-C01-T-17 / T1(C1-1) | P2 | fixed | T1 计划 | 01 章 5 点承诺只讲 2 点，无章节指引 | tutorial/01 | 列表后补"第 2、3 点在 02 章、第 5 点在 03 章展开" | — | 待用户 |
| P02-C01-T-18 / T1(C1-4) | P2 | fixed | T1 计划 | 01 章名字数与 03 章 228146 样本数缺衔接 | tutorial/01 划分节 | 补一句"长度 L 的名字展开成 L+1 个样本，共 228146" | 与 02 脚本实测一致 | 待用户 |
| P02-C03-T-19 / T1(C1-11) | P2 | fixed | T1 计划 | lr 搜索片段 plt 未导入即使用 | tutorial/03 | 片段顶部补 `import matplotlib.pyplot as plt` | — | 待用户 |
| P02-C03-S3-08 / T1 | P2 | fixed | S3🟡 | 三种诊断情况数字（2.5/2.6 等）未声明是示意 | tutorial/03 诊断节 | 加"数字为示意值，非本章实测"声明；另补实测 gap≈0 佐证 | — | 待用户 |
| P02-CSC-S1-01 / T-20 | P2 | fixed | S1🟡 + 计划项9 | STEPS 短程档无任何文档说明 | tutorial/README 新增「学习方式」节 + tutorial/03 顶部提示 | 写明 `STEPS=2000` 用法与"默认档完全不变"承诺 | 实测默认档逐位一致 | 待用户 |
| P02-C02-S1-02b | P2 | fixed | S1🟡 | softmax 手算版 keepdim、`+b1` 广播未点破 | tutorial/02 Step2/Step4 | 各补一行注释（(N,1) 按行广播 / (100,) 广播到每行） | — | 待用户 |
| P02-C03-S2-08 | P2 | fixed | S2⚠️ | 教程图（images/cell031）与脚本产物（scripts/embedding_visualization.png）双图并存、网格参数不同 | tutorial/03 可视化节 | 教程注明 06 生成图"与下图同源"（S3 已核实坐标逐点一致）；是否合并单一来源记 disputed D3 | S3 复跑 27 坐标逐点一致 | 待用户 |

## Disputed（跨 Part / 待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P02-D1 | 字符映射统一方向：本台账选定 Part1/作业风格（`stoi={s:i+1}; stoi['.']=0`），Part2 七脚本+教程已改，Part1 与 assignment_reference 零改动即可达成三方一致 | 若 Part1 审计 agent 倾向反向统一（Part2 数据派生风格更稳健），需 T0 在批1收尾时二选一，两方向工作量均小 |
| P02-D2 | roadmap 节点 2 称"作业 2 含采样"实际无此题；"pytest 全绿"实际为 `python test_mlp_exercises.py` 独立脚本形式（pytest 亦可跑） | roadmap 属 docs/，建议 T0 派发 docs 侧微修（删"采样"二字即可），不阻塞本 Part 关闭 |
| P02-D3 | Embedding 可视化双图并存（images/cell031_output00.png 与 scripts/embedding_visualization.png） | 已核实同源一致，建议保留现状并在 G4-4 记"低优先"；若要单一来源，改 06 输出到 images/ 并更新教程引用 |
| P02-D4 | 全量 20000 步实测 train 2.3749 vs S3 报告 2.3766（±0.002） | 多线程浮点归约非确定性所致，非文档错误；教程已注明"可复现"指同机同配置量级复现 |
