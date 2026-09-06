# S2 学习报告 · Part 17 Agentic RL

> 审计者：学生 agent S2（实操薄弱型）· 日期 2026-09-04 · 用时约 18 分钟
> 材料范围：tutorial/ 3 篇 .md + scripts/01_toy_agent_grpo.py + assignment_17/ 三件（只读原件，作业在副本上做）

---

## 一、总分：8.8 / 10

| 维度 | 得分 | 一句话 |
|---|---|---|
| 01 章讲解质量 | 9.0 | 三问（多轮采样/观测 mask/奖励分配）结构清晰，消融数字逐字可复现 |
| 02 章讲解质量 | 8.5 | 奖励谱系表 + Echo Trap + 框架/评估选型密度高，但框架部分只能"读"不能"跑" |
| 教程 ↔ 脚本一致性 | 8.5 | 核心代码与输出一致；速记代码块未标"示意"，一处实测无法从脚本复现 |
| 作业设计 | 9.0 | 四题梯度合理、测试双模式贴心；题 2 的 eps 语义有一处歧义坑 |
| 可复现性 | 9.5 | 脚本 CUDA 实跑 11.8s，输出与教程引文**逐字一致** |

---

## 二、卡点清单（按遇到顺序）

| # | 卡点 | 位置 | 严重度 | 说明 |
|---|---|---|---|---|
| 1 | 题 2 首跑 FAIL：把 `eps=1e-6` 无条件加到 std 上，`[[0.0,1.0]]` 组优势 ≈0.999996，偏差 2e-6 超出测试容差 1e-6 | assignment 题 2 | 中 | docstring「eps: std 兜底」的"兜底"需理解为 **std=0 时的回退**而非无条件相加；改后 4/4 绿。骨架默认值与测试容差之间存在真实张力 |
| 2 | 教程 01 §2 的 rollout 速记代码用了 `generate_until` / `tokenize`，脚本中不存在（脚本内联生成循环） | 01 章"手写多轮 rollout" | 低 | 未标"伪代码示意"，学生对照脚本时会找不到函数 |
| 3 | "反面实验"（BC 示范砍到 2 组合、40 步、RL 卡 0.33 再推到 0.50）在脚本中**无对应代码/开关** | 01 章"三个必讲的观察"① | 中 | 教程说"我们实测过"，但学生无法一键验证这一关键论断 |
| 4 | 性能表"仅 BC"两行与"随机初始化"行需改脚本（RL_ROUNDS=0 / 直接评测）才能得到 | 01 章"性能数据" | 低 | 有 📊 复现说明，算合格但需学生自己动手改参数 |

---

## 三、分章评分与笔记

### 01 从单轮 RLVR 到 Agentic RL —— 9.0
- 轨迹解剖图（mask 标注）→ 形状追踪 ASCII 图 → rollout 速记 → GRPO 公式 → 真实消融输出，讲解链条完整。
- 🔎 **实测**：脚本 CUDA（seed=7）输出与教程引文**逐字一致**：A 组 99.0/12.5/43.8/25.0，B 组 96.9/3.1/10.4/0.0；曲线 round0/3/5 = 1.000/0.990/0.948（A）、B round5=1.000。全程 11.8 秒。
- 「玩具判分漏洞（诚实声明）」和「两种 mask 别混淆」是本章含金量最高的两段。
- 扣分点：卡点 2、3。

### 02 奖励设计与工业框架 —— 8.5
- 奖励谱系表（稀疏/塑形/ORM/课程）+ reward hacking 防线四件套 + Echo Trap/StarPO-S + 框架/评估两张选型表，信息密度高。
- 错误 1-3 的"解法"代码块均为示意（`group_rewards`、`demo_texts`、`MAX_STEPS` 不是脚本变量），概念正确但未标注。
- verl/slime/rLLM 部分是"读框架"性质（本机无 Docker 实操），对 S2 这类实操薄弱者无法验证，只能存疑记录。

### 作业 Assignment 17 —— 9.0
- 四题分别对应：loss mask / 轨迹级 GRPO / 工具解析 / Echo Trap 检测，与教程三问一一映射。
- 思考题 Q1-Q5 与教程呼应紧密（credit assignment 连坐、判分漏洞、多样性≠健康、BC 死锁链、两种 mask）。
- 扣分点：卡点 1（eps 语义歧义）。

---

## 四、一致性核对表（教程代码块 ↔ 脚本）

| # | 教程位置 | 教程内容 | 脚本对应 | 结论 |
|---|---|---|---|---|
| 1 | 01 §4 实测输出引文 | 四格消融数字 + 曲线摘录 | `main()` 打印格式与数字 | ✅ **逐字一致**（CUDA 实测） |
| 2 | 01 §2 rollout 速记 | `generate_until`/`tokenize`/`parse_call`/`run_tool`/`MAX_TURNS` | `rollout()` 内联生成 | ⚠️ 结构一致；`generate_until`/`tokenize` 为示意名，未标伪代码 |
| 3 | 01 §3 GRPO 公式 | `loss = −Σ logπ×A / #assistant` | `trajectory_grpo_step()` | ✅ 一致（优势×loss-mask，按 assistant 数归一）；grad-clip 1.0 教程未提（次要） |
| 4 | 01 §1 轨迹解剖注释 | user=0 / assistant=1 / 观测=0 | `rollout()` mask 构造 | ✅ 一致 |
| 5 | 02 错误 3 | "60 token 上限、闭卷探针 24 步上限" | `len(cur[0]) < 60` / `range(24)` | ✅ 一致 |
| 6 | 02 练习 2 / 作业实验题 | `run_experiment`/`curve`/`parse_call`/`real_ids` | 均存在 | ✅ 可操作 |
| 7 | 01 观察①反面实验 | BC 覆盖 2 组合、40 步、0.33→0.50 | **无对应代码路径** | ❌ 无法从脚本复现 |
| 8 | 01 性能表"仅 BC/随机初始化"行 | 100.0%/40.6%/36.5%/31.2%；10.4% | 需改参数复现 | ⚠️ 有复现说明，本次未逐一验证 |
| 9 | README/脚本运行时长 | GPU ~15 秒 / CPU 约半分钟 | docstring 写 ~10-15s / ~20-60s | ✅ 实测 11.8s，相符（措辞微差） |
| 10 | 设备差异声明 | CPU/GPU 数字波动 + CPU 参考值 | 脚本 docstring 同样声明 | ✅ 诚实，双向一致 |

---

## 五、作业元数据 + pytest 输出

工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_17/`（三件从原目录复制，仅编辑 exercises）

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|---|---|---|---|
| 题 1 轨迹 loss mask（30 分） | ✅ 一次过 | 0 | ~3 分钟 | PASS |
| 题 2 轨迹级 GRPO（30 分） | ❌ 二次过 | 1（重读"eps 兜底"语义） | ~6 分钟 | PASS |
| 题 3 工具调用解析（25 分） | ✅ 一次过 | 0 | ~4 分钟 | PASS |
| 题 4 🌟 Echo Trap（15 分） | ✅ 一次过 | 0 | ~2 分钟 | PASS |

**4/4 PASS**（3 题一次过；合计 ~15 分钟，远低于预期难度，纯 CPU 秒级）。

最终 pytest 输出：

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.1.1 -- .venv/bin/python
collected 4 items

test_agentic_exercises.py::test_ex1_mask PASSED                          [ 25%]
test_agentic_exercises.py::test_ex2_traj_grpo PASSED                     [ 50%]
test_agentic_exercises.py::test_ex3_parse PASSED                         [ 75%]
test_agentic_exercises.py::test_ex4_echo PASSED                          [100%]

============================== 4 passed in 0.00s ===============================
```

首次运行记录（题 2 失败现场，已留档 `pytest_output.txt` 被终版覆盖；失败摘要）：
`FAILED test_ex2_traj_grpo - assert (1.999996000034976e-06 < 1e-06)` —— eps 无条件相加所致。

---

## 六、只改 3 件事

1. **给 01 章 §2 rollout 速记代码块加标注**：「以下为伪代码示意，`generate_until`/`tokenize` 对应脚本内联的生成循环」——或直接改写成脚本真实结构。消除"对照脚本找不到函数"的迷惑（卡点 2）。
2. **把"BC 覆盖不足 → RL 死锁"反面实验做成脚本开关**（如 `BC_COVERAGE = 6 or 2`、`BC_STEPS = 120 or 40`），让"RL 六轮卡 0.33 → 调软后推到 0.50"这条关键论断可一键复现（卡点 3，这是本章"没有 BC 就没有 RL"的实证支柱）。
3. **作业题 2 docstring 消歧**：「eps: std 兜底」改为「eps: 仅当组内 std=0 时作分母兜底（请勿无条件加到 std 上）」。当前骨架默认 `eps=1e-6` 与测试容差 `1e-6` 组合下，"无条件相加"读法必挂（卡点 1）。

---

## 七、最喜欢的 3 处

1. **同 seed A/B 掩码消融的四格对照**（开卷/闭卷 × train/holdout）：泄漏组"复读观测"的崩塌机理讲得极透，且教程引文与脚本真实输出逐字一致——这在教程里非常少见，S2 实跑 11.8 秒就复现了整张表。
2. **「玩具判分漏洞（诚实声明）」**：主动交代"第二次观测=答案，可冒充最终答案"的教学设计取舍，并给出真实 RLVR 的三种堵法——诚实反而让消融结论更可信，还顺手教了 reward hacking。
3. **作业的弹性题设计**：题 4 未实现返回 `None` 时优雅 SKIP 而非 ERROR，测试同时兼容 `pytest` 与 `python 直跑`双模式，对基础薄弱学生非常友好。

---

*S2 · 2026-09-04 · 脚本日志：scratch/S2_P17/run_script01.log · 作业：students/S2_hands/work/assignment_17/*
