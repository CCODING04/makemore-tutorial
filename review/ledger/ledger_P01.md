# P01（Part 1 Bigrams）整改台账

> 整改人：T2｜日期：2026-09-04｜依据：plan_P01.md + 三学生报告（S1_math / S2_hands / S3_interview）
> 状态取值：fixed（已修复并验证）/ disputed（留 T0 裁量，本次不动）
> 来源：S1/S2/S3=学生报告；PLN=T1 主教计划；OLD=旧基线（STUDENT_FEEDBACK）

## P0（必修）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|---|---|---|---|---|---|---|---|
| P01-C01-S1S2S3-01 | P0 | fixed | S1🟡3+S2🟡2+S3 K1 | "训练后 loss 收敛到约 2.47"不可复现（100 步实测 2.4901），脚本 07 尾部还硬编码打印"≈2.47"自相矛盾 | tutorial/03 §3 结论段；scripts/07 L68 | 教程改为实测口径（100 步+lr50+λ0.01 → 2.4901，含约 0.03 正则项；无正则 300 步 ≈2.46 逼近 2.454），并说明旧数字 2.47 对应无正则更多步设置；脚本 07 打印改为 ≈2.49 及其构成 | 镜像 07 实跑 EXIT=0，步骤99=2.4901，训练日志与原版逐行一致 |
| P01-C02-S1S3-02 | P0 | fixed | S1🔴2+S3 K2；PLN C1-3 | 锚点数字 ln(27)≈3.296 正文 0 次出现（仅作业思考题有）；randn 初始 3.76 无解读 | tutorial/03 §3；tutorial/01 | 03 §3 新增"三根基准线"表（瞎猜 3.2958=ln27（全零初始化可实测）/ randn 起点 ≈3.76 及为何更差 / 计数版 2.454）；01 章概率链处预告锚点 | 全零初始化 loss=3.295837=ln(27) 本机复算一致 |
| P01-C03-S1S3-03 | P0 | fixed | S1🔴3+S3 K5/D9/G4 | "训练后 W.exp() 和 N 几乎一模一样"实测不成立（corr 仅 0.34、量级差 89 vs 6763）；脚本 06 打印"W 最优值=log(N)"同样不精确 | tutorial/03 §4 与对照表洞察；scripts/06 等价性打印 | 教程 §4 重写为"分布等价"：附可运行对比代码（行归一化后 max 偏差 300 步 0.044 → 3000 步 0.005）；给出 softmax 行内平移不变 ⇒ W_i=log P_i+c_i 的准确表述；脚本 06 打印同步改为分布等价表述 | S1_probe3 实测数据引用；镜像 06 实跑新文案输出正常 |
| P01-C04-S2S3-04 | P0 | fixed | S2🟢8+S3 K4/D2 | "约 228,000 个字符"口径错误：字符实为 196,113，228,146 是 bigram 数 | tutorial/01 数据集段 | 改为"32,033 个名字 / 196,113 个字符 / 228,146 个 bigram"三数字并列并提示勿混用 | 本机复算：bigram=228146、纯字符=196113 |
| P01-C05-S2-05 | P0 | fixed | S2🔴1（代码隐患，任务必修5） | 教程/脚本用 set(words) 动态推字符集，小数据集上 stoi 错位翻车（作业题1首跑 FAIL 根因） | tutorial/02 §2 代码后；scripts/02 L20；assignment_1/README 题1思考 | 教程加 ⚠️ 警告段（生产词表必须固定）；脚本 02 加注释说明隐患与固定词表写法（默认行为不变）；作业题 1 思考新增一道字符集陷阱题 | 镜像 02 实跑输出与原版一致（27 字符/228146）；行为未变 |
| P01-C06-S3-06 | P0 | fixed | S3 K3（面试盲区，任务必修6） | 无"loss 接近 ≠ 分布接近"教学点：NN 版 NLL 2.4593 逼近闭式解 2.4544 但采样仍出乱码 | tutorial/03 §4 新增小节；scripts/07 尾部 | 03 §4 新增"面试考点"小节：计数版 vs NN 版采样并排表 + λ=0/300 步反例数字；脚本 07 总结加采样质量警告两行 | 镜像 07 采样输出 mria/mmyazzieelend/... 与教学点一致 |
| P01-C07-S1-07 | P0 | fixed | S1🟡7；PLN C1-7（任务必修7） | NLL 与 CrossEntropy 同式异名，无关系说明 | tutorial/03 §3 loss 定义处、§4 对照表 | §3 loss 代码后加术语说明（CE = softmax+NLL 合并算子，此处手写即 NLL 部分）；对照表损失函数列统一为 NLL 并加术语备注 | 文字项；check_latex 通过 |
| P01-C08-PLN-08 | P0 | fixed | PLN G2-1~G2-5；OLD 卡点9 残留 | 数学推导置于代码块/裸排：02 NLL 四步推导整块、broadcasting 分式演算、01 log 公式、03 softmax 无公式 | tutorial/02 §3/§5、tutorial/01 概率表、tutorial/03 §2 | 02 §5 NLL 推导改 LaTeX 分步推导（$$ 单行、无中文、$$ 前空行），每步保留"为什么"；broadcasting 分式演算改行内 LaTeX；01 log 恒等式改行内 LaTeX；one-hot/形状示意代码块保留（数据示意豁免） | check_latex.py 4 篇全部"未发现问题" |
| P01-C09-S1-09 | P0 | fixed | S1🔴1+S3 默写卡壳1 | one-hot × W 的"查表取行"直觉完全缺失（Karpathy 核心 aha 点） | tutorial/03 §2 | 新增 3×3 手算例子（[0,1,0]@W=第1行）+ "xenc@W 每行=抠出 W 对应行，W 每行=该字符的 27 个分数"点题，并与计数版"查 N 的行"对照 | 文字项；S1_probe 实验1 结论一致 |
| P01-C10-S1-10 | P0 | fixed | S1🟡4；PLN 审计要点3 | softmax 只给流程不给公式，"为什么 exp"仅一句类比 | tutorial/03 §2 | 补 softmax 公式（display LaTeX）+ exp 三理由（非负/保序/可导）+ "软版 argmax"点题 | check_latex 通过 |
| P01-C11-S1-11 | P0 | fixed | S1🟡1（任务必修8点名） | "归一化=条件概率"无公式，只有一句"每一行归一化" | tutorial/02 §3 | 补 $P[i,j]=N[i,j]/\sum_{j'}N[i,j']$ 公式并说明行和含义与 $P(j\mid i)$ 条件概率解读 | check_latex 通过 |
| P01-C12-S1S2-12 | P0 | fixed | S1🔴3；S2 ❌"承诺未兑现"；PLN C1-6 | 等价性小节只有 stub 注释，无可运行验证代码 | tutorial/03 §4 | §4 代码段改为可运行的分布对比（P_nn vs P_count，print max 偏差），附预期数值 | S1_probe3：0.044→0.005 数据引用 |
| P01-C13-S2S3-13 | P0 | fixed | S2 ❌ loss 定义不一致 + S3 | 教程 03 训练循环无正则项而脚本 07 有，读者对不上数字 | tutorial/03 §3 脚本指引处 | 教程明确注明"脚本 07 与本片段唯一差别是 loss 多加 0.01*(W**2).mean()，故打印值略高" | 镜像 07 实跑 2.4901 与教程新数字一致 |
| P01-C14-S1S3-14 | P0 | fixed | S1🟡8+S3 K7（作业部分） | 作业 README 文件结构自称 assignment.md，实际文件为 README.md | assignments/assignment_1/README.md 文件结构节 | 改为 README.md，并补 pytest 运行方式 | 镜像作业测试两种模式均跑通 |

## P1（裁量修复）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|---|---|---|---|---|---|---|---|
| P01-C15-S2-15 | P1 | fixed | S2🟡5 | NLL 代码先于平滑小节出现，先跑后读会 log(0) 炸 | tutorial/02 §5 代码后 | 代码后加 ⚠️ 注：此处 P 未平滑，对全量训练集不炸但遇未见 bigram（如 andrejq 的 jq）即 -∞，解法见下节，脚本 05 用 N+1 | 文字项 |
| P01-C16-S1-16 | P1 | fixed | S1🟡2 | "平均 NLL = NLL/总数"是推导链唯一没有"为什么"的一步 | tutorial/02 §5 第4步 | 推导第 4 步补理由：不同长度名字可比、量级不随数据规模漂移 | 文字项 |
| P01-C17-S1S3-17 | P1 | fixed | S1🟡5+S3 K6/D8 | L2≈平滑只有断言；且"等价"过强（不同先验） | tutorial/03 L2 小节；scripts/07 正则注释 | 补半步推导（W→0 ⇒ e^0=1 ⇒ 均匀分布，即 λ→∞ 极限）+ ⚠️ 类比非严格等价（L2=高斯先验 vs N+λ=Dirichlet）；脚本 07 注释同步 | 文字项；脚本注释随镜像 07 实跑验证 |
| P01-C18-S1-18 | P1 | fixed | S1🟡6 | "计数法隐式最大化似然"是断言无解释 | tutorial/03 §4 | 补一句：行归一化频率是多项分布 MLE 闭式解（行内似然最大的取值即频率） | 文字项 |
| P01-C19-S1-19 | P1 | fixed | S1🟢2 | "约 2.45"未区分平滑口径 | tutorial/02 §5 脚本指引 | 注明脚本 05 为平滑版（N+1）实测 2.4544 | 脚本 05 学生实测 + 本机向量化复算 2.4546（float32 累加口径差异内一致） |
| P01-C20-S3-20 | P1 | fixed | S3 K6 | 脚本 07 注释"L2 正则化等价于…（N+1）"过强 | scripts/07 L54 | 改为"类比于…（N+λ）：都是把分布往均匀拉（严格说是不同先验）" | 镜像 07 实跑通过 |
| P01-C21-S2-21 | P1 | fixed | S2🟢7 | 拓展题未实现时测试显示 PASSED 易误判 | assignments/assignment_1/test_bigram_exercises.py + README | 跳过分支文案明确提示"pytest 下显示 PASSED 实为跳过"；README 运行测试节加 ⚠️ 说明（不改跳过逻辑，保持直跑/pytest 双模式兼容） | 镜像测试直跑 5 通过 0 失败、pytest 5 PASSED |

## P2（轻量改进）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|---|---|---|---|---|---|---|---|
| P01-C22-S1S2-22 | P2 | fixed | S2🟡3+S2🟡4；PLN C1-5 | 教程可视化代码声称产出 ../images/*.png 且"完整脚本见 06"，与脚本实际（scripts/bigram_matrix.png；06 无绘图代码）脱节；照抄片段会往仓库写文件 | tutorial/02 §2、tutorial/03 §1 | 内嵌片段改为存当前目录；注明教程图为 notebook 存档、脚本 06 不含绘图代码；03 章 imshow 改 xenc[:100] 防全量绘制卡顿 | 文字项 |
| P01-C23-S2-23 | P2 | fixed | S2 ⚠️ | 教程 03 §2 W=randn 无种子，教程片段不可复现 | tutorial/03 §2 代码注释 | 注明脚本 06 固定 Generator().manual_seed(2147483647) | 文字项 |
| P01-C24-S2-24 | P2 | fixed | S2 ⚠️ | NN 版采样 seed（2147483647+10）教程未说明 | tutorial/03 §4 对比表 | 对比表中标注两版各自 seed | 文字项 |
| P01-C25-PLN-25 | P2 | fixed | PLN C1-2 | README 无环境自检节（roadmap 承诺） | tutorial/README.md | 新增"🧰 环境自检"节：torch 版本 + 数据文件行数（32033）两条命令 | 命令输出本机验证（32033） |
| P01-C26-S3-26 | P2 | fixed | S3 K7/V3；PLN C1-4 | 教程无任何论文钩子，Bengio 2003 无承接 | tutorial/03 预告 | 预告补 Bengio et al. 2003《A Neural Probabilistic Language Model》及"Part 2-3 是其最小复现" | 文字项 |
| P01-C27-S3-27 | P2 | fixed | S3 K5 延伸 | 脚本 06 注释"W…等价于 bigram 计数矩阵""exp 等价于 N 矩阵"在训练前不成立 | scripts/06 两处注释 | 改为"训练收敛后扮演计数角色/伪计数，收敛后行归一化分布才逼近频率分布" | 镜像 06 实跑通过 |

## disputed（留 T0，本次未动）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 不动理由/建议 |
|---|---|---|---|---|---|---|
| P01-D01-S1S2S3-D1 | P1 | disputed | PLN C1-1 + S1/S2 实测 | "每节大约 15-30 分钟"低估：02/03 章含推导+练习实测 45-60 min | tutorial/README.md 学习路线段 | 属时长口径类，按任务约定记台账不动；建议 T0 改为"02/03 章建议各预留 45-60 分钟" |
| P01-D02-S3-D2 | P1 | disputed | S3 D7/K2（roadmap 部分） | roadmap L134"脚本 07 loss ≈3.3→≈2.4-2.5"起点与 randn 实测 3.77 不符（3.296 仅全零初始化） | docs/course_roadmap_v3.md（docs/ 不动） | 教程侧已用"三根基准线"澄清；roadmap 措辞建议 T0 改为"≈3.3（均匀分布）/≈3.8（randn）→≈2.5" |
| P01-D03-S3-D3 | P2 | disputed | S3 K7（roadmap 部分） | roadmap/其他文档中"assignment.md"文件名引用（作业 README 侧已在 C14 修复） | docs/（不动） | 建议全局 grep assignment.md 统一为 README.md |
| P01-D04-S2-D4 | P2 | disputed | S2🟢6 | 02 §4 采样"版本差异"免责声明在当前环境多余（输出逐字一致） | tutorial/02 §4 | 保留：对未来 torch 版本仍是有效防呆，删除收益低、风险高，留 T0 裁量 |

计数：P0 14 条（全 fixed）/ P1 7 条（全 fixed）/ P2 6 条（全 fixed）/ disputed 4 条。
