# ledger_P17 — Part 17 Agentic RL 整改台账（T2）

- 日期：2026-09-04 ｜ 输入：plan_P17 + S1/S2/S3 report_P17 ｜ REPO 只读，改动全在 REVIEW 镜像
- 判定：按"≥2 命中或事实/代码/数学错误 → P0；禁止 wontfix，争议标 disputed"
- **统计：P0 = 2 ｜ P1 = 11 ｜ P2 = 9 ｜ 合计 22 条（21 已修 + 1 复核驳回 + 3 条 docs/联网侧转 outline_suggestions.md）**

## P0（双命中 / 承诺-兑现失配）

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| P0-1 | 01 章观察① + 脚本 + 作业 Q4 | "BC 砍到 2 个…卡 0.33 / 调软 40 步到 0.50"实测叙事**无可指认产出开关**（S2 卡点 3）且"6 个组合砍到 2 个"按字面执行（2 组合=4 条示范）复现得 **0.667 非 0.33**（S3 卡点 3 实证，T2 本机复跑证实）——双命中 | ✅ 已修：脚本加 `SHOW_NEGATIVE=1/2` 环境变量档（默认 0 行为逐位不变）；本机实跑 1 档 0.333×6 纹丝不动、2 档 0.344→0.500、4 条示范档 0.667；教程观察①改"12 条示范砍到 **2 条**"并附一键复现命令与 0.667 对照说明（覆盖组合数才是关键变量）；作业 Q4 同步表述+复现命令 |
| P0-2 | README 学习目标第 4 条 ↔ 02 章 | README 承诺"**配置 verl 的 multi-turn 训练**"，02 章实际只有选型表 + compute_score 练习，**无一行配置内容**（S1 卡点 7 + S3 卡点 2 双命中；导航"CLI 实操"名不副实加重误导） | ✅ 已修：02 章 §3 补"verl multi-turn 最小配置要点"（yaml 示意摘录：multi_turn.enable / max_assistant_turns / custom_reward_function.path + sanity batch 配套三动作 + 常设版本免责）；README 目标措辞降级对齐为"理解机制（配置要点）+ 配置奖励函数"；导航改"—（选型导览 + multi-turn 配置要点 + 练习）" |

## P1

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| P1-1 | 01 章"数学推导"节（G2×1，plan 违规 #2-1） | 整节公式放 ``` 代码块，规范点名词判例 | ✅ 已修：Step 1-4 全部转单行闭合 `$$` LaTeX（过 check_latex 0 问题），公式外中文说明 |
| P1-2 | 01 章 §3 + 02 章 Q3（G2×2） | 优势广播公式在代码块；02 章 Q3"GRPO 优势 = (r − mean) / std"行内纯文本 | ✅ 已修：§3 转 `$$`；02 章 Q3 改行内 `$…$` |
| P1-3 | 01 章 Step1-4（S1 卡点 1 + plan 缺口 #4） | Σ 对什么求和没写死、缺 eps 兜底、缺 1/#assistant_tokens 归一化——推导节/§3/脚本三口径不一 | ✅ 已修：推导节三个口径逐条写死（Σ=mask=1 的 assistant token；eps=1e-4 分母兜底；分母 Σ#assn），并与脚本 `A.expand_as(M)*M`/`M.sum()` 逐行对应 |
| P1-4 | 跨 Part（plan 缺口 #4，G15） | eps/std 口径四方漂移：P8 无偏+eps / P11 max 1e-6 / P17 有偏+1e-4 / 作业默认 1e-6，无任何声明 | ✅ 已修：01 章推导节加"跨 Part 口径声明（P17 收口）"段，四方口径并列 + 数值影响 ~1e-4 量级 + "要能说清自己代码用哪种"；std 有偏口径（少乘 √(G/(G-1))，G=8 差 ~7%）就地注明（S1 卡点 6） |
| P1-5 | 01 章全文（S1 卡点 3） | 单轮 GRPO 的 ratio/clip 与 KL 罚在轨迹级"消失"无一句交代 | ✅ 已修：新增"单轮→轨迹级 GRPO 组件对照表"（5 行）+ "clip 与 KL 去哪了"段：单步在线更新 ratio≡1 无 clip 对象、非预训练初始化无 KL 可守护——是教学设定省略而非理论删除，verl 实战两者照常在 |
| P1-6 | 01 章 Step3-4（S1 卡点 2） | 组内标准化无数字例，公式到代码之间断档 | ✅ 已修：补 5 行数字例（G=8，[1×5,0×3] → μ=0.625、有偏 σ≈0.484、A=+0.774/−1.291、全同组→0） |
| P1-7 | 01 章陷阱 3 + 02 章 §2（S1 卡点 4） | Echo Trap 只有症状/发现/缓解，无成因链；"多轮特有"无解释；clip-higher 直觉缺失 | ✅ 已修：两处各补恶性循环链（长轨迹 credit 噪声 → 梯度尖峰 → 分布变尖 → 熵塌 → 零方差组 → 无梯度逃不回探索）+ 单轮为何难塌 + clip-higher 直觉（放宽高概率 token 裁剪上界=给回低概率动作留梯度） |
| P1-8 | 01 章 L266/L317/Q3 + 02 章 §2 + 作业面试直通车（plan 缺口 #5，共 5 处） | StarPO-S 第三件统一写成"critic +…"，RAGEN 原件是 **critic revival（崩溃后回滚早期 checkpoint 重置）**，现表述与 GRPO 无 critic 卖点相抵触 | ✅ 已修：5 处统一改"critic 复活（revival）"+半句解释（不是"加个 critic"） |
| P1-9 | 全 Part（G4，plan 违规 #4） | 零图（无 images/） | ✅ 已修：新增 `REVIEW/courses/Part17_agentic_rl/images/mask_ablation.png`（双子图：左 RL 六轮曲线 A/B、右评测四格柱状，seed=7 RTX 4090 实测数据，含 round3=0.990/round5=0.948/B round4=0.979），01 章 §4 输出块后引用 |
| P1-10 | 01 章 L120-134/L264 + 02 章 L80/L97（G10×4） | `generate_until`/`tokenize`、`policy_distribution`、`demo_texts`、`MAX_STEPS` 均非脚本真实名，挂在"脚本核心"名下未标示意 | ✅ 已修：rollout 节改题"伪代码示意，非脚本 API"+说明对应内联循环；其余三处就地加"示意"注 |
| P1-11 | Part17 README（plan 缺口 #6，O4） | P17 部内 0 次提及 Part 19，姊妹篇分工声明单向缺失 | ✅ 已修："在 LLM 链路中的位置"节补分工声明（训练侧训得出 / Part 19 应用侧用得起） |
| P1-12 | 01 章性能表口径 + 脚本（G14） | 缺 torch 版本与线程数（plan #7-4）；脚本运行时也只打 device/seed | ✅ 已修：性能表口径行补"RTX 4090 / torch 2.6.0+cu124 / 纯 CPU torch 2.6.0 / 24 线程"；脚本 main() 参数行加 torch 版本与 threads 打印 |
| P1-13 | docs/course_roadmap_v3.md（plan 缺口 #7，大纲级） | roadmap v3 无节点 16/17，课程 README 却宣布 Part 1-17 走完（S3 卡点 1） | ➡️ 非部内可修：outline_suggestions.md 追加（T0 统筹） |

## P2

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| P2-1 | 01 章 §4 输出块（plan 缺口 #3） | 声明"逐字来自真实运行"实为节选（A 组摘 round 0/3/5、B 组摘 round 5） | ✅ 已修：改"以下为一次真实运行的**节选**"+注明完整 6 轮以脚本实跑为准 |
| P2-2 | README/01 章 Q2/02 章 §4（G13，S3 卡点 5） | Kimi 23 次、WebArena≈0%、star 数均无出处口径 | ✅ 已修：Kimi 标"官方博客口径，2025-06，转述未逐字核实"；WebArena 标"经验口径，未附出处"；star 数标题改"2026-08 快照，仅供量级参考，以仓库实际为准" |
| P2-3 | 作业题 2（S2 卡点 1） | docstring"eps: std 兜底"歧义：按"无条件相加"读法 `[[0.0,1.0]]` 得 0.999996 超容差 1e-6 必挂 | ✅ 已修：骨架 docstring 改"写 std = max(std, eps)（仅当 std=0 时回退），不要无条件 std+eps"；assignment.md 题 2 验收标准加同款说明——题面与骨架注释统一 |
| P2-4 | 01 章 §3/§4（S1 卡点 5） | BC 与 RL loss 同构（BC=优势恒正特例）未点破，两套代码被误读为两套数学 | ✅ 已修：§3 末补一段"BC 冷启动和 RL 更新是同一套数学"（移位/log-prob/mask/分母全同，差别只在优势取值） |
| P2-5 | README×2 + 01 章头部（S3 卡点 6） | CPU 时长三处口径微差（"约半分钟" vs "~20-60 秒"） | ✅ 已修：三处统一"约半分钟到一分钟" |
| P2-6 | 脚本 L172 + 作业 Q2（plan 缺口 #9） | 笔误"教程 01 章有专注 / '玩具判分漏洞'专注说明" | ✅ 已修：两处改"专门说明" |
| P2-7 | README 导航表（plan 缺口 #10） | 02 章标"—（CLI 实操）"与内容不符 | ✅ 已修：随 P0-2 改"—（选型导览 + multi-turn 配置要点 + 练习）" |
| P2-8 | 01 章 Q2 verl 键名 + 02 章练习 1（plan 缺口 #8，G19） | `rollout.mode=async` 等配置键名无版本免责；P11 二参 vs P17 四参 compute_score 签名漂移互不引用 | ✅ 已修（P17 侧）：Q2 加"配置键名以所用版本文档为准"；02 章 multi-turn 配置块带常设免责。P11 侧签名复核 → outline_suggestions.md 追加（cross-part） |
| P2-9 | 02 章 §3（S3 硬数字 #12） | "24GB **实操**：verl multi-turn…"无实操记录，与 SkyRL/AReaL 的"引用不实操"标注标准不一 | ✅ 已修：改"24GB 路线（纸面建议，本课未附实跑记录）" |

## disputed（2 条，无 wontfix）

| # | 争议 | 处置 |
|---|---|---|
| D1 | S3 卡点 4"docstring CPU 数字（31.2/39.6/28.1 vs 0/2.1/9.4）与本机不符、疑似异机产物"：T2 本机复跑（CPU，torch 2.6.0+cu124，24 线程，CUDA_VISIBLE_DEVICES=""）得 A 100.0/31.2/39.6/28.1 vs B 100.0/0.0/2.1/9.4——**与 docstring 逐字一致，指控不成立**；S3 复现出 GPU 数字恰证明 CPU 档对 torch 版本/线程敏感 | 数字保留不改；01 章设备说明 + 脚本 docstring 按 G14 补"随 torch 版本/线程波动、以当次复跑为准"口径（P1-12 连带）。已驳回但留痕 |
| D2 | 02 章 verl multi-turn yaml 键名（multi_turn.enable 等）离线不可核实，只能"示意+免责"处理 | 联网窗口对照官方 docs 复核键名；已加常设免责兜底 |

## 验证记录（2026-09-04，scratch/T2_P17/）

1. **原版基线（GPU，11.2s，exit 0）**：四格 99.0/12.5/43.8/25.0 vs 96.9/3.1/10.4/0.0，与教程引文**逐字一致**；curve A round3=0.990、round5=0.948，B round4=0.979。
2. **镜像脚本改动后基线**：四格与解读段与原版**完全一致**（仅新增 torch 版本/threads 参数行）→ 主表数字不变 ✅。
3. **新开关**：`SHOW_NEGATIVE=1` → 0.333×6 纹丝不动 + 反面结论段；`SHOW_NEGATIVE=2` → 0.344→0.500；补跑 4 条示范档 → 0.667（歧义修正的实证）✅。
4. **CPU 档复跑**：与 docstring"31.2/39.6/28.1 vs 0/2.1/9.4"逐字一致（disputed D1 依据）。
5. **作业**：assignment_reference 参考答案 + test（scratch）→ **pytest 4/4 passed**；镜像骨架 docstring 改动不影响测试。
6. **check_latex.py**：README / 01 / 02 / assignment.md 四个改动 md → **全部 0 问题**。
7. **G4 图**：images/mask_ablation.png 目检通过（实测数据标注、中文渲染正常）。
