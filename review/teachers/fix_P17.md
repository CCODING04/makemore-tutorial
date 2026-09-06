# fix_P17 — Part 17 整改报告（T2）

- 日期：2026-09-04 ｜ 依据：plan_P17（T1 预审）+ S1/S2/S3 report_P17 + T0 必修清单（8 项）
- 原则：REPO 只读；全部修改写入 REVIEW 镜像；脚本实跑先 cp 到 scratch（G16）
- 台账：`REVIEW/ledger/ledger_P17.md`（**P0=2 / P1=11 / P2=9，另 disputed 2 条**）

## 一、镜像改动清单

| 文件 | 改动摘要 |
|---|---|
| `REVIEW/courses/Part17_agentic_rl/scripts/01_toy_agent_grpo.py` | 新增 `SHOW_NEGATIVE` 环境变量开关（"1"=示范砍到 2 条 → 复现 0.333 死锁；"2"=再加 BC 40 步 → 0.344→0.500；默认 0 行为逐位不变）；main() 加 torch 版本/threads 打印（G14）+ 反面档结论段；docstring 补开关用法与 CPU 口径（torch/线程 + 环境敏感性）；笔误"有专注"→"有专门说明" |
| `REVIEW/courses/Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md` | "数学推导"节整体重写：代码块 → 单行 `$$` LaTeX（Step 1-4），Σ 范围写死（mask=1 的 assistant token）、eps=1e-4 兜底、有偏 std 注明、1/#assistant_tokens 归一化、**跨 Part eps/std 口径声明（P17 收口）**、**5 行数字例**（G=8 → +0.774/−1.291）；新增"单轮→轨迹级组件对照表"+ "clip 与 KL 去哪了"段；§3 公式转 LaTeX + BC/RL loss 同构点破；三重理由与"三个必讲的观察"G1 拆行；观察① 改"12 条砍到 2 条"+ 复现命令 + 0.667 表述坑；输出块标"节选"+ 设备说明补 torch/线程口径与"以复跑为准"；错误 3/陷阱 3/Q3 的 StarPO-S 改"critic 复活"并补 Echo Trap 成因链；性能表口径补 RTX 4090/torch 2.6.0/24 线程；rollout 速记标"伪代码示意，非脚本 API"；Q2 Kimi 出处口径 + verl 键名版本免责；引用 G4 实测图 |
| `REVIEW/courses/Part17_agentic_rl/tutorial/02_rewards_and_frameworks.md` | §3 补"verl multi-turn 最小配置要点"（yaml 示意 + 常设版本免责 + sanity batch 等配套三动作，与练习 1 闭环）；"24GB 实操"改"纸面路线（未附实跑记录）"；star 数标"快照参考，以实际为准"；WebArena≈0% 标"经验口径，未附出处"；§2 补 Echo Trap 成因链 + StarPO-S 三件改"critic 复活（revival）"；Q3 公式转行内 LaTeX；错误 2/3 代码块加示意标注 |
| `REVIEW/courses/Part17_agentic_rl/tutorial/README.md` | 学习目标第 4 条降级对齐（"理解 multi-turn 机制 + 配置奖励函数"）；导航 02 章"—（CLI 实操）"→"—（选型导览 + multi-turn 配置要点 + 练习）"；"在 LLM 链路中的位置"补 **Part 19 姊妹篇分工声明**；Kimi 23 次标"官方博客口径，2025-06，转述未逐字核实"；CPU 时长统一"约半分钟到一分钟"（×2） |
| `REVIEW/courses/Part17_agentic_rl/images/mask_ablation.png` | 新增 G4 图：左 RL 六轮曲线（A/B 实测）+ 右评测四格柱状（99.0/12.5/43.8/25.0 vs 96.9/3.1/10.4/0.0），seed=7 RTX 4090 torch 2.6.0 实测数据 |
| `REVIEW/assignments/assignment_17/assignment.md` | 题 2 验收标准补 eps 语义（`max(std, eps)` 零方差回退，勿无条件相加）；Q4 改"示范砍到 2 条（只覆盖组合 (1,1)）"+ SHOW_NEGATIVE 复现命令；面试直通车 StarPO-S 改"critic 复活"；"专注说明"→"专门说明" |
| `REVIEW/assignments/assignment_17/agentic_exercises.py` | 题 2 docstring eps 语义消歧（"eps: std 兜底"→"std = max(std, eps) 零方差回退，不要 std+eps"）；函数签名/测试零改动 |
| `REVIEW/ledger/ledger_P17.md`、`REVIEW/teachers/fix_P17.md` | 本报告与台账 |
| `REVIEW/outline_review/outline_suggestions.md` | 追加 3 条：roadmap v3 缺节点 16/17（S3 卡点 1，大纲级）、P11↔P17 verl compute_score 签名漂移 cross-part 复核、verl multi-turn 键名联网窗口核（disputed D2） |

## 二、必修清单 8 项逐项核销

1. **轨迹级 GRPO 数学补齐** ✅：符号展开+数字例（必修 1a）、clip/KL 消失/保留对照表（1b）、Echo Trap 成因链（1c）、BC/RL 同构点破（1d）、std 有偏口径+跨 Part 声明（1e）——全部落在 01 章，S1 的 7 个卡点清零。
2. **verl multi-turn 承诺兑现** ✅：02 章 §3 补最小配置要点（示意 yaml + 免责 + 配套动作），README 目标措辞对齐，导航"CLI 实操"改口——不再是只有名词。
3. **rollout 速记伪代码标注** ✅：改题"伪代码示意，非脚本 API"+ 与脚本内联循环的对应说明；连带 02 章 demo_texts/MAX_STEPS、01 章 policy_distribution 三处示意标注（G10 清零）。
4. **BC 0.33 反面实验** ✅：SHOW_NEGATIVE=1/2 开关（本机实跑 0.333×6 / 0.344→0.500）；"6 个组合砍到 2 个"歧义修正（2 **条**示范；4 条=0.667 已实测写进教程）；CPU 数字经复核与 docstring 一致（disputed D1 驳回），按 G14 补 torch 版本/线程口径替代改数。
5. **Kimi 23 次/WebArena≈0%/star 数** ✅：三处均补出处口径或"转述未核实"标注（离线无法给 permalink，已留 disputed 联网核）。
6. **作业题 2 eps 语义** ✅：assignment.md 验收标准与骨架 docstring 统一为"max(std, eps) 零方差回退、勿无条件相加"——S2 的"字面读法必挂"坑消除。
7. **格式规范 G1-G16** ✅：G1×2 拆行；G2×3 转 LaTeX（check_latex 全 0）；G4 新增实测双联图；G14 补 torch 版本/线程/具体型号；G13 三处出处口径。
8. **T1 十项内容缺口** ✅：#1→P0-2、#2→P0-1、#3→P2-1、#4→P1-3/4、#5→P1-8、#6→P1-11、#7→outline 追加、#8→P2-8、#9→P2-6、#10→P2-7，逐条落账。

## 三、验证结果（全部实跑，scratch/T2_P17/）

1. 原版脚本 GPU 基线 11.2s：四格与教程**逐字一致**；镜像脚本改动后基线输出与原版完全一致（仅多 torch/threads 参数行）→ **主表数字不变**。
2. 新开关：SHOW_NEGATIVE=1 → 0.333×6 纹丝不动；=2 → 0.344→0.500；4 条示范档 0.667（歧义实证）→ **开关生效**。
3. CPU 档复跑与 docstring 数字逐字一致（disputed D1 驳回依据，留痕）。
4. 作业：assignment_reference 参考答案 + test → **pytest 4/4 passed**（0.01s）。
5. check_latex.py：README / 01 / 02 / assignment.md → **全部 0 问题**。

## 四、通用问题（跨 Part）与好写法候选

**通用问题（≤3）：**
1. **实测叙事的"可复现开关"意识**：本 Part 唯一的重大翻车点（0.33 反面实验）不是数字错——数字是真的——而是"实测过"却没有产出开关，且表述歧义让按字面复现者得 0.667。凡教程写"我们实测过 X"，都应能做到"一条命令复现 X"（本轮已用 SHOW_NEGATIVE 兑现）；其他 Part 可对照自查。
2. **"承诺-兑现"清单化验收**：README 学习目标每条都是对读者的契约（verl multi-turn 这次半兑现被三个学生独立抓到）。建议各 Part 结稿时把学习目标逐条映射到"章节+小节锚点"，映射不到的要么补内容要么降级措辞。
3. **G13 外部数字的"口径三件套"不全**：数字+时点常有，但"出处可指认 + 未核实要声明"常缺（Kimi/WebArena/star 数三处同病）。建议统一句式："（来源，时点，转述未核实）"。

**好写法候选（≤3）：**
1. **"两种 mask，别混淆"警告框**（01 章）：把玩具开关与工业实现（观测进上下文 + loss-mask）的边界一句话讲透，还接住了作业 Q5——教学诚实度与概念区分度的双料样板，全课程可复用该框式。
2. **同 seed 四格消融 + "信号不在 train"叙事**（01 章 §4）：唯一变量、开卷/闭卷 × train/holdout 四格、holdout 探针任务，12 秒看到"泄漏买来的是依赖，不是能力"——实验设计本身就是面试故事（S3 面试资产 #7）。
3. **弹性题 SKIP 机制**（assignment 题 4/test）：未实现返回 None 时优雅 SKIP 而非 ERROR + 双运行模式兼容——stretch 题零挫败感设计（S2 最喜欢 #3）。
