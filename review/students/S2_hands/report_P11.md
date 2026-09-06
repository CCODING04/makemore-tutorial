# S2 学习报告·P11（对齐实战 — verl 工业级 RL 后训练）

- 学生画像：S2 实操薄弱型（教程逐行读、代码亲手跑、作业先做后看）
- 审计范围：`courses/Part11_alignment_verl/tutorial/`（README + 01 + 02）、`../scripts/`（01、02）、`assignments/assignment_11/` 三件
- 审计日期：2026-09-04 ｜ 环境：Python 3.12.12 / torch 2.6.0+cu124（与教程 02 章口径一致）/ 2×RTX 4090 / Docker 受限

## 总分：8.3 / 10

一句话：01 章是"教程代码块 = 脚本 = 实测输出"三方逐行对得上的标杆章，作业一次全过；02 章因 Docker 环境受限无法实操验证，且 verl 配置键存在 2 处疑错、1 处键语义错位与数据路径缺口，是全章唯一的短板。

## 卡点清单

| # | 类型 | 卡点 | 影响 |
|---|------|------|------|
| 1 | env-blocked | Docker：`permission denied ... /var/run/docker.sock`（用户不在 docker 组），02 章全部实操（quickstart/GRPO/自定义奖励/双卡）无法执行 | verl 配置键只能纸面抽查，无法证真证伪 |
| 2 | time-blocked | 无（脚本 01 实测 0.011s、脚本 02 实测 1.33s，远低于 3 分钟阈值） | — |
| 3 | 学习卡点 | 02 章 Step 2 命令里 `data.train_files=gsm8k/train` 不是可用文件路径；quickstart 需要先跑数据预处理生成 parquet，教程没给预处理命令 | 即使 Docker 能跑，照抄命令也会 FileNotFoundError |
| 4 | 学习卡点 | 作业 docstring 说"比脚本 01 多两个工程要求"，但实现与脚本 01 的 `gsm8k_reward` 等价，"两个"指代不明 | 我对着脚本 01 复读了 3 分钟找"多出的两个"，没找到 |
| 5 | 学习卡点 | 测试用例 `\boxed{ 42 }`（花括号带空格）靠第 3 级"最后数字"兜底才通过——测试注释有解释，但教程 01 正文没提这个边界 | 初读会以为 1 级正则命中，理解偏差一次 |

## 分章评分

| 章节 | 得分 | 评语 |
|------|------|------|
| 01 手写 GRPO→verl 桥接 | 9.0/10 | 三段代码块与脚本 01 函数体逐行一致；三处"实测输出"块与本机 stdout 逐行一致（KL=0.0204、adv=±1.0、-0.58/1.73 全部复现）；手算验证框可自查。小瑕疵：核心循环块里 `skip_ids` 算了没用（教学简化可接受）、"2025: verl"年份存疑（HybridFlow 开源始于 2024） |
| 02 verl 快速上手 | 6.5/10 | 结构诚实（日志/性能表均标注"非本机实录"），但因 env-blocked 无法实操；配置键问题见核对表 #3-#6。CLI 抽查"可拼凑性"不过关：学生无法仅凭本章命令跑通 quickstart |
| README | 9.0/10 | 硬件矩阵（CPU/1×4090/2×4090）与"安装摩擦最高的一章"预警非常实用；导航、脚本对应列准确 |
| Assignment 11 | 9.0/10 | 4 核心 + 1 stretch 全 PASS；docstring 自足到几乎不用翻教程；单组/批量语义差异、`<=` 边界、eps 位置等坑都提前写了。小瑕疵见卡点 #4 与核对表 #9 |

## 一致性核对表

| # | 检查项 | 结果 |
|---|--------|------|
| 1 | 教程 01 代码块（gsm8k_reward / group_advantages / k3_kl）↔ 脚本 01 函数体 | 一致（教程 docstring 精简，函数体逐行相同） |
| 2 | 教程 01 三处"实测输出"块 ↔ 脚本 01/02 实际 stdout | 逐行一致（含 step 0/4/9/14/29/44/59 的 KL=0.413/0.341、零梯度组 [1,6,6,6,6,6]、BC 曲线） |
| 3 | 02 章 `algorithm.kl_penalty` 写成数值系数（0.01-0.1） | **存疑错位**：verl 中该键是字符串类型（如 low_var_kl），系数在 `algorithm.kl_ctrl.kl_coef`；教程最佳实践表和 01 章映射表（k3_kl→algorithm.kl_penalty）都按"数值系数"口径使用。建议作者对照源码确认 |
| 4 | 02 章 `actor_rollout_ref.actor.lr`（错误 3 解法 + 最佳实践表） | **疑错**：应为 `actor_rollout_ref.actor.optim.lr` |
| 5 | 02 章 `actor_rollout_ref.rollout.micro_batch_size` / `trainer.backend=gloo` | **存疑**：离线无法验证这两个键在 verl 主配置中是否存在；需容器实测 |
| 6 | 02 章 `data.train_files=gsm8k/train` | 缺前置数据预处理步骤，路径不可直接使用（见卡点 #3） |
| 7 | 02 章错误 4 把 `enable_gradient_checkpointing=true` 标注为"或使用 QLoRA" | 术语滑动：梯度检查点 ≠ QLoRA |
| 8 | 脚本 02 `d_t = ref_logp - logp  # (P, G) 可反传` | **注释 bug**：实际 shape 为 (G, P)（与上一行 182 行注释自相矛盾） |
| 9 | 教程 01 练习 3 `kl_budget_guard` ↔ 作业题 4 `kl_budget_ok` | 一致且教程明确声明"作业是简化版、测试不覆盖 guard"，处理到位 |
| 10 | 交叉引用锚点（#性能分析 / #陷阱-2-版本冲突 / #错误-1 / #错误-4 / #性能数据量级参考） | 全部有效 |
| 11 | assignment.md "骨架 alignment_exercises.py:20" 行号 | 正确（第 20 行即 `def math_reward`） |
| 12 | 测试双 runner：独立运行时 `return "skip"` 生效；pytest 下 "skip" 返回值不触发 skip（未实现会显示 PASSED 而非 SKIP） | 轻微行为不一致（本次已实现 stretch，未实际触发） |

## 作业元数据 + pytest 输出

- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_11/`（仅编辑 `alignment_exercises.py`，测试与 assignment.md 未动）
- 元数据：核心 4 题 + stretch 1 题全部一次过、0 次提示、总计约 8 分钟
  - 题1 math_reward：一次过 / 0 提示 / ~3 min（卡点 #4/#5 各耗一点时间）/ PASS
  - 题2 group_advantages：一次过 / 0 提示 / ~1 min / PASS（直接套脚本 01 公式，eps 位置作业已预警）
  - 题3 k3_kl：一次过 / 0 提示 / ~1 min / PASS
  - 题4 kl_budget_ok：一次过 / 0 提示 / ~1 min / PASS（注意 `bool(kl <= budget)` 保证 `is True` 断言）
  - 题5 zero_gradient_groups（stretch）：一次过 / 0 提示 / ~1 min / PASS（照脚本 02 `zero_adv_groups` 即可）
- pytest 输出：

```
platform linux -- Python 3.12.12, pytest-9.1.1
collected 5 items

test_alignment_exercises.py::test_ex1_reward PASSED                      [ 20%]
test_alignment_exercises.py::test_ex2_group PASSED                       [ 40%]
test_alignment_exercises.py::test_ex3_kl PASSED                          [ 60%]
test_alignment_exercises.py::test_ex4_budget PASSED                      [ 80%]
test_alignment_exercises.py::test_ex5_zero_gradient PASSED               [100%]

============================== 5 passed in 0.01s ===============================
```

- 独立 runner：`通过: 5/5 🎉`

## 只改 3 件事

1. **02 章命令进容器实测并补齐可跑闭环**：修正 `algorithm.kl_penalty`（系数应为 `algorithm.kl_ctrl.kl_coef`）与 `actor_rollout_ref.actor.optim.lr` 键路径，补上 quickstart 的数据预处理命令（生成 `train.parquet` 的前置步骤），把"核心行"升级成学生可原样粘贴跑通的完整命令。
2. **修脚本 02 的 shape 注释 bug**：`d_t = ref_logp - logp  # (P, G) 可反传` 改为 `(G, P)`——本章全程用 shape 追踪教学，一处错注释会让学生对着全景图自我怀疑。
3. **作业题 1 的"多两个工程要求"说清楚**：具体列出与脚本 01 的差异点（如测试新增的 `\boxed{ 42 }` 带空格兜底、自我纠正场景），或删掉该句；顺带把测试改为 `pytest.skip()` 让双 runner 的 skip 语义一致。

## 最喜欢 3 处

1. **01 章的"手算验证"引用框**：prompt0 的 ±1.0 和 KL=0.0204 都给了手工复算过程——把"信教程"变成"能验教程"，对我这种实操不自信的人是救命的。
2. **脚本 02 的可复现设计**：固定种子 42 让零梯度组现象每次逐字复现，且 torch 版与纯 math 版 k3 有逐位互验 assert（"同一公式、两种实现应逐位一致"）——工程品味极好，也是我作业题 3/题 5 的直接模板。
3. **"手写 ↔ verl 概念映射表"的三重复现**：01 章正文表格、脚本 01/02 的输出结尾、02 章每个 Step 的回链标注（如 `group_advantages() → adv_estimator=grpo`）三处一致——配置行背后的代码语义随时可查，02 章读起来不迷路。
