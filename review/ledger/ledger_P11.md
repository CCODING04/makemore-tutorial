# 问题台账 · Part 11（对齐实战 verl）· T2 整改批

> 编号规则：P11-C{cc}-{SRC}-{nn}；cc∈{00=README,01,02=章节,SC=scripts,AS=assignment_11,RM=roadmap}；SRC∈{S1,S2,S3,T=T1预审,T2=教师实证}
> 状态：fixed=已修复且验证；disputed=争议/待 T0 裁决（禁止 wontfix）
> 关键实证：①脚本 01/02 scratch 实跑 rc=0（0.010s/1.28s），修后与基线 diff 仅 2 行**有意修正**的 verl 映射打印行，全部训练数值（k3=0.0204、0.38→0.83、零梯度组 [1,6,6,6,6,6]、BC 1.00）逐行一致；②std 双口径实证：同组 [1,0,1,0] 总体 std(G)=0.5→±1.0 vs 样本 std(G−1)=0.5774→±0.866，S1 报告数字复核成立；③Docker 档 env-blocked（审计机无 daemon 权限），02 章 CLI 按 T1 降级口径做命令自洽性审计，未因跑不了 Docker 反推教程造假。

## Fixed（按严重度）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|------|--------|------|------|------|------|----------|----------|
| P11-C02/SC-01 | P0 | fixed | S1🔴 S2🔴 T1 缺口2 + T0 必修3 | `algorithm.kl_penalty` 键值错位（被当"系数 0.01-0.1"教）：verl 中该键是**惩罚类型**字符串键（kl/abs/mse/low_var_kl），系数在 `algorithm.kl_ctrl.kl_coef`（KL-as-reward）或 `actor.use_kl_loss`+`kl_loss_coef`（KL-as-loss）；S1 费曼自检实测"脱口而出配 kl_penalty"——口误真传导给读者 | 02 章最佳实践表、01 章映射提示+核心循环注释、脚本 01 映射框、脚本 02 docstring/L141/L178/L271 共 8 处 | 最佳实践表拆 3 行：kl_penalty=类型键 / kl_ctrl.kl_coef=系数 / use_kl_loss+kl_loss_coef=KL-as-loss 路线 + "常见误配"警告段；01 章/脚本全部映射点改 `algorithm.kl_ctrl.kl_coef` 并注"kl_penalty 是类型键" | grep 复核：全文 4 处 kl_penalty 引用均为"类型键"语义，0 处旧用法残留 |
| P11-C01-02 | P1 | fixed | S1🔴 T0 必修1 | 组内 std 分母双口径未点名：P8 04 章 torch.std（G−1，例 ±0.87）vs P11（1/G 总体 std，例 ±1.0），S1"第一反应是我又算错了，慌了 10 分钟"；eps 写法（max vs +）差异同样未对照；脚注"别的归一化口径"不点名 Part 8 | 01 章 ② 小节 | 手算验证后新增「分母口径对照：GRPO 论文原式 vs torch.std」小节：双口径对照表（公式/std/优势/谁在用，数字经 T2 复算）+ eps 两大流派说明（作业认 max）+ **G=1 退化**（A=0，组大小≥2，兼答 roadmap 五步法之问） | 复算：std(G)=0.500→±1.0、std(G−1)=0.5774→±0.866；G=1: A=(r−r)/ε=0 |
| P11-C01-03 | P1 | fixed | S1🟡 T0 必修2 | k3 推导跳步：`KL=E[exp(d)−d−1]` 缺关键步 E[exp(d)]=1（采样自 π_new 时 E[π_ref/π_new]=1），且 E 的取向未声明（取在 π_ref 下估计有偏）；手算例输入概率（log0.4/log0.6…）只在脚本里有，教程正文无来源 | 01 章 ③ 小节 | 推导重写为三步 LaTeX（KL 定义→令 d→**E[exp(d)]=1 关键步**）+ ⚠️ E 取向声明（学生自实现最常见翻车点）+ 手算验证标注输入出处（脚本 01 L226）+ 与 P8 `k3_kl` 签名漂移声明（参数序/聚合方式） | 手算链复算 0.0231/0.0177→0.0204 与脚本输出逐位一致 |
| P11-C02-04 | P1 | fixed | S2🟡 T0 必修3 后半 | `actor_rollout_ref.actor.lr` 键路径疑错（应为 `actor.optim.lr`；S2 离线核对存疑） | 02 章最佳实践表、错误 3 解法 | 两处改 `actor_rollout_ref.actor.optim.lr`，表内注明"键路径带 optim" | 离线降级核实口径已标注（见 Disputed D1） |
| P11-C02-05 | P1 | fixed | S2🔴 T1 缺口3 + T0 必修5（G10） | quickstart 命令不可拼凑：`data.train_files=gsm8k/train` 非真实路径（需先跑数据预处理生成 parquet）、缺预处理步骤、未标"节选"——学生照抄 FileNotFoundError | 02 章 Step 2/3/5 | Step 2a 补预处理步（examples/data_preprocess/gsm8k.py→~/data/gsm8k/*.parquet）+ Step 2b 命令改 parquet 路径 + 全部命令块标"**核心行节选**"+"为什么不能直接抄"说明框；Step 3/5 路径同步 | 命令间一致性核对通过（Step2/3/5 路径统一）；降级审计口径在正文如实注明 |
| P11-C02-06 | P1 | fixed | T1 缺口4（G10） | Step 4 称自定义奖励"你唯一必写的代码"，但缺 verl 侧接入配置行，端到端拼凑不出 | 02 章 Step 4 | 补 `custom_reward_function.path=my_reward.py` + `custom_reward_function.name=compute_score` 两键及"path 指文件、name 指函数名"说明 + 降级核实注 | my_reward.py 定义与配置行呼应（compute_score 函数名一致） |
| P11-CAS-07 | P1 | fixed | S2🟡 T1 缺口8 + T0 必修5（G18） | assignment 三处口径/机制问题：①练习 2 防除零两说（验收"max(std,eps)" vs 步骤提示"std<eps 返回全 0"）；②test stretch 跳过用 return "skip"，pytest 下显示 PASSED 而非 SKIPPED（G18）；③骨架"比脚本 01 多两个工程要求"指代不明（S2 找了 3 分钟） | assignment.md 练习 2、alignment_exercises.py 题1/题2 docstring、test_alignment_exercises.py | ①步骤提示统一为"std=max(std,eps) 兜底——全同组分子为 0 自然全 0"；②仿 P9/P10 修法 `_skip()`：检测 `PYTEST_CURRENT_TEST` 走 `pytest.skip`（Skipped 继承 BaseException，直跑 runner 捕不到，故用环境变量区分入口）；③docstring 点名两个边界（`\boxed{ 42 }` 带空格兜底 + 尾随小数点 rstrip） | 四象限全绿：pytest+参考 5 passed；直跑+参考 5/5；pytest 空跑 **4 failed + 1 skipped**；直跑空跑 4 失败 + 1 ⏭️——两入口 SKIP 语义一致 |
| P11-Cxx-08 | P1 | fixed | T1 §五（G2×10）+ T0 必修6 | 数学推导 10 处放代码块/docstring/纯文本：GRPO 三步推导块、②③ docstring 推导、两处手算验证、正文 A_i、ASCII 图内 loss 行、assignment 练习 2/3 推导块与正文 | 01 章、assignment.md | 全部 `$…$`/`$$…$$` 化（单行闭合、列表内行内公式）；docstring 推导段精简为引用教程正文（G15 单一事实源，教程与脚本 docstring 同步改）；ASCII 图保留、loss 行改伪码 + 图后补 display 公式与"单步 on-policy ratio≡1 故无需 clip"说明（兼答 S1 卡点 5） | check_latex.py 对 4 个改动 md 全部"未发现问题" |
| P11-Cxx-09 | P1 | fixed | T1 §五（G4）+ T0 必修6 | 全 Part 0 张图，3 组可画对象（训练曲线/零梯度组演化/PPO vs GRPO 成本） | images/（新增）+ 01/02 章引用 | 新增 3 张 PNG（图内英文、正文中文图注、相对路径）：grpo_vs_bc_curve（**脚本 02 实测**曲线，0.38→0.83 vs BC 1.00 + 盲区标注）、zero_gradient_groups（**实测** [1,6,6,6,6,6]）、ppo_vs_grpo_cost（02 章性能表推算值，图题注明 NOT measured） | scratch/t2_P11/collect_curve.py+make_figs.py 留档；曲线数据与脚本输出同源（seed 42 逐位）；逐张目检 |
| P11-Cxx-10 | P1 | fixed | S3🟡 T1 §五（G13×4）+ T0 必修6 | 外部数字裸奔：①性能表"官方 benchmark"无链接；②verl 23.2k★/slime 8.3k★ 无"截至"日期；③2026 证据表（GLM-5.3/DeepSeek-V4/Kimi）无来源口径；④rollout 60-80% 占比无出处（01/02 两处复用） | README、01 章、02 章 | ①去"官方"字样改"课程设计推算、无单一官方 benchmark 页可引"；②③补"截至 2026-09 课程编写时，引用前自行复核"；④标"经验量级、无正式论文出处、社区经验口径" | grep 复核"官方 benchmark"已清除 |
| P11-C00-11 | P1 | fixed | S3🟡 T1 §三-4 + T0 必修7 | "官方 ≥24GB" vs 性能表 "~8GB" 口径张力未圆场（S3：面试被追问会卡壳） | 02 章性能表脚注 | 补"显存口径圆场"段：24GB=含 vLLM KV cache 预留的保守整机需求 vs ~8GB=训练态推算占用——"前者是给你多少卡才稳，后者是账面上花了多少" | 与 S3 建议文本一致 |
| P11-C02-12 | P2 | fixed | S1🟢 S2🟡 T1 缺口6 + T0 必修4 | 脚本 02 L183 shape 注释 bug：`d_t = ref_logp - logp  # (P, G)`，实际 (G, P)（与上一行注释自相矛盾） | scripts/02 | 改 `# (G, P) 可反传`（行为不变） | 修后脚本实跑与基线 diff 逐行一致（仅映射行有意修正） |
| P11-C01/SC-13 | P2 | fixed | T1 缺口7 + S1 卡点7 | "同一条奖励链/Part 8 同款"过度声明：脚本 02 math_reward 与脚本 01 实现（正则/控制流）不同；P8↔P11 k3_kl 参数序/聚合漂移 | 01 章、脚本 01 | "同一条"→"同语义（脚本 02 内联实现，教学用例行为等价）"；k3 小节+脚本 01 补签名差异声明 | 01 章/脚本 01/脚本 02 三处措辞统一 |
| P11-C02-14 | P2 | fixed | S3🟢 T1 §三-3（缺口5）+ S3 卡点6 + T1 缺口11 | 02 章三处轻项：①三角色图为 GRPO 视角、与 Step2 PPO 必需的 critic 矛盾未声明；②错误 4 "或使用 QLoRA"注释与梯度检查点配置不匹配；③章末"回 Part 12"措辞误导 + P11 对 Part 17 零反向指引 | 02 章 | ①架构表后补"GRPO 视角三角色；PPO 需第四角色 critic，日志 `critic/...` 即它"声明；②注释改"开启梯度检查点（≠QLoRA；verl QLoRA 支持属实验性）"；③"回"→"下一步去"，补"多轮工具调用的 RL 见 Part 17" | 01 章性能摘要 QLoRA 处同步补实验性标注 |
| P11-Cxx-15 | P2 | fixed | T1 §五（G14） | 实测输出块口径不齐：组内优势/k3 两处"实测输出（脚本 01）"未带环境口径（首处已带） | 01 章 | 统一补"（Python 3.12 / CPU，同前口径）" | grep 复核 3 处实测块口径齐全 |
| P11-CRM-16 | P1 | fixed(改走 outline) | T1 缺口1 + S3🔴 + T0 必修7 | roadmap 节点 11 三处：①"3 编码题"计数漂移（实为 4 核心+1 stretch）；②脚本行漏列 02_grpo_toy_train.py；③"N=1"记号与 P10 卡数语境撞车 | docs/course_roadmap_v3.md L420-441（**REPO 只读不改 docs**） | 按战役规则追加 outline_suggestions.md 四条（含 P11 侧已补 G=1 退化内容的呼应说明） | outline_suggestions.md 末尾 P11 T2 四条在案 |

## Disputed（待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P11-D1 | verl 配置键离线不可终审：`algorithm.kl_penalty`=类型键、`algorithm.kl_ctrl.kl_coef`、`actor.optim.lr`、`custom_reward_function.path/name`、`trainer.backend=gloo`、`rollout.micro_batch_size` 等——教程审计机无 Docker/网络，修复按"教科书性错误高风险"与 verl 公开文档常识执行，已在正文标注"降级核实口径" | 终版前联网/容器窗口逐键复核；kl_penalty 错位是三源一致+S1 费曼实测传导，修复方向高置信 |
| P11-D2 | README "官方 quickstart 文档明确单卡 ≥24GB" 未附官方文档链接（24GB 数本身双源一致，但"官方文档明确"四字无从核对） | 保留数字+已加圆场段；终版补官方 quickstart 链接或软化"官方文档明确"措辞 |
| P11-D3 | P17 侧 README 用旧 org `volcengine/verl`，P11 全线 `verl-project/verl`——跨 Part 仓库名不统一，P17 侧不属本批修改范围 | 登记 P17 侧台账统一为新 org |
| P11-D4 | 01 章历史脉络把 verl 标为 2025（S2：HybridFlow 开源始于 2024；EuroSys'25 论文发表在 2025）——"2024 开源/2025 论文"两口径均可辩，未改动 | 终版可改"2024 开源（HybridFlow）、2025 EuroSys 论文"更精确 |
| P11-D5 | 01 章引言"Part 8 GRPO=…→clip 更新"与脚本 02 无 clip 的张力：已补"单步 on-policy ratio≡1 故无需 clip"说明（随 G2 整改落地）；若 T0 认为引言也需同步措辞再立轻项 | 观察整改后版本；如仍有困惑再补引言半句 |

## 统计

- **Fixed：16 条**（P0×1、P1×10、P2×5）
- **Disputed：5 条**（verl 配置键降级核实、24GB 出处、P17 旧 org、verl 年份、引言 clip 措辞）
- 验证：脚本两轮全绿（基线 01 rc=0/0.010s、02 rc=0/1.28s 与教程逐行一致；修后 diff 仅 2 行有意修正的映射打印，数值输出零变化）；作业四象限全绿（pytest+参考 5 passed、直跑+参考 5/5、pytest 空跑 4 failed+1 skipped、直跑空跑 4 失败+1 ⏭️）；check_latex 4 个改动 md 全部 0 问题
