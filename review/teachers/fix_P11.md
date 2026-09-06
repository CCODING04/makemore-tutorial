# Part 11（对齐实战 verl）T2 整改报告

- 整改日期：2026-09-04 · 执行：T2 整改教师
- REPO 只读；全部修改写入 REVIEW 镜像；脚本一律先 cp 到 `REVIEW/scratch/t2_P11/` 实跑（G16）
- 环境：RTX 4090 + 4090 D（2×24GB）· torch 2.6.0+cu124 · Python 3.12（REPO venv）· Docker env-blocked（daemon 权限受限，02 章 CLI 按 T1 降级口径做命令自洽性审计）

## 一、镜像清单（本次交付）

| 文件 | 改动 |
|---|---|
| `courses/Part11_alignment_verl/tutorial/README.md` | G13 时效数字口径（verl/slime star 数加"截至"、2026 证据表加来源口径） |
| `courses/Part11_alignment_verl/tutorial/01_handwritten_to_verl.md` | GRPO 推导块+两处手算+k3 推导 LaTeX 化（G2×6）、新增「分母口径对照」小节（必修1）、k3 推导补 E[exp(d)]=1+输入出处（必修2）、映射键 kl_penalty→kl_ctrl.kl_coef、`boxed{ 42 }` 兜底句、G=1 退化、loss 公式化+无 clip 说明、QLoRA/占比/性能摘要出处口径、P8 k3 签名漂移声明、实测块口径补齐、插入 2 图 |
| `courses/Part11_alignment_verl/tutorial/02_verl_quickstart.md` | kl_penalty 拆 3 行类型/系数/KL-as-loss + actor.optim.lr（必修3）、Step2 补数据预处理+parquet 路径+「节选」标注、Step4 补 custom_reward_function 接入配置、三角色图 critic 声明、错误 3/4 修正（optim.lr、梯度检查点≠QLoRA）、24GB vs ~8GB 口径圆场、性能表出处措辞、时间占比出处、章末"回→下一步"+P17 指引、插入 1 图、`$USER` 改写（check_latex） |
| `courses/Part11_alignment_verl/scripts/01_reward_and_bridge.py` | 两处 docstring 推导精简为引用教程（与教程代码块同步）+ std 分母/k3 签名漂移声明 + 映射框 kl_penalty→kl_ctrl.kl_coef（行为不变） |
| `courses/Part11_alignment_verl/scripts/02_grpo_toy_train.py` | **shape 注释 bug (P,G)→(G,P)（必修4）**、3 处 kl_penalty→kl_ctrl.kl_coef（行为不变） |
| `courses/Part11_alignment_verl/images/*.png` | 新增 3 图：grpo_vs_bc_curve（实测）、zero_gradient_groups（实测）、ppo_vs_grpo_cost（推算值，图题注明） |
| `assignments/assignment_11/assignment.md` | 练习 2/3 推导 LaTeX 化 + 正文公式 LaTeX（G2×3）、练习 2 步骤提示统一为 max(std,eps) 口径 |
| `assignments/assignment_11/alignment_exercises.py` | "多两个工程要求"指代明确化（boxed 带空格兜底+尾随小数点）、题 2 Steps 口径统一（注释级，行为不变） |
| `assignments/assignment_11/test_alignment_exercises.py` | G18：`_skip()` 检测 `PYTEST_CURRENT_TEST` 走 pytest.skip（两入口 SKIP 语义一致） |
| `assignment_reference/assignment_11/test_alignment_exercises.py` | 同步 G18 修复 |
| `ledger/ledger_P11.md` | 本 Part 台账（Fixed 16 / Disputed 5） |
| `outline_review/outline_suggestions.md` | 追加 P11 四条 roadmap 建议（3 编码题计数、漏列脚本 02、N=1 记号、24GB 口径） |
| `scratch/t2_P11/` | 实跑日志（base_/fix_ 两轮 diff）、collect_curve.py（数据收集+双口径验证）、curve_data.py、make_figs.py、asgn_ref/asgn_empty 四象限验证目录 |

## 二、必修清单执行情况（T0 七项全闭环）

1. **组内 std 分母双口径（必修1）**：01 章手算验证后新增「分母口径对照」小节——双口径表（GRPO 论文原式 1/G=±1.0 vs torch.std G−1=±0.87，T2 复算 0.5774 佐证）+ eps 两流派（max 下界兜底 vs + 更保守，作业认 max）+ 明确点名"Part 8 就是那个别的口径"+ G=1 退化（A=0，兼答 roadmap 五步法之问）。作业练习 2 验收与骨架 Steps 同步统一 max(std,eps) 口径（T1 缺口8 一并闭环）。✅
2. **k3 推导补关键步（必修2）**：推导重写为三步 LaTeX——先声明 **E 取在 π_new 下**（⚠️ 取在 π_ref 下估计有偏，学生自实现最常见翻车点），再补 `E[exp(d)] = E[π_ref/π_new] = 1` 关键步（求和相消一行写明），得 `E[exp(d)−d−1] = −E[d] = KL`；手算验证标注输入出处（logp_ref=[log0.4, log0.6] 等，出自脚本 01 L226）。✅
3. **02 章 verl 配置键错位（必修3）**：最佳实践表 `algorithm.kl_penalty` 行拆为类型键（kl/low_var_kl 字符串）/ `algorithm.kl_ctrl.kl_coef`（系数）/ `use_kl_loss`+`kl_loss_coef`（KL-as-loss 路线）3 行 + "常见误配"警告段；`actor.lr`→`actor.optim.lr` 两处（最佳实践表+错误 3）；01 章映射提示、脚本 01 映射框、脚本 02 三处注释全部同步（grep 复核 0 处旧语义残留）。✅
4. **脚本 02 shape 注释 bug（必修4）**：`d_t = ref_logp - logp  # (P, G)`→`(G, P)`；修后 scratch 实跑与基线 diff 逐行一致（仅 2 行有意的映射修正）。✅
5. **02 章节选/指代/兜底（必修5）**：Step2 补数据预处理步（gsm8k.py→parquet）+ 路径改 `$HOME/data/gsm8k/train.parquet` + 三处命令块标「核心行节选」+ "为什么不能直接抄"说明框；Step4 补 custom_reward_function 接入配置；作业骨架"多两个工程要求"点名两个边界（`\boxed{ 42 }` 带空格走第 3 级兜底、尾随小数点 rstrip）；01 章实测输出后补 `\boxed{ 42 }` 兜底路径句。✅
6. **格式 G1-G16 全量（必修6）**：G2×10 全 LaTeX 化（docstring 推导精简为引用教程正文，教程与脚本同步，G15 单一事实源；ASCII 图保留、loss 行伪码化+图后 display 公式）；G4×3 图（2 张实测数据、1 张推算值并在图题注明 NOT measured，图内英文/正文中文/相对路径）；G10×2（预处理步骤+接入配置）；G13×4（star 数"截至"、证据表口径、"官方 benchmark"去官方化、占比出处）；G14（实测块口径补齐）；G18（作业 skip 机制，`PYTEST_CURRENT_TEST` 方案——Skipped 继承 BaseException 直跑 runner 捕不到，环境变量区分入口是唯一两全解）；README `$USER` 改写消 check_latex 误报。✅
7. **T1 十一个 C1 缺口（必修7）**：1（roadmap 三处→outline 四条追加）、2（=必修3）、3/4（=必修5 的 G10 两处）、5（三角色图 critic 声明）、6（=必修4）、7（"同一条"→"同语义" + k3 签名漂移声明）、8（=必修1 的作业侧统一）、9（"官方 benchmark"措辞 + QLoRA 实验性标注）、10（=G13 的 star 数/证据表 + P17 旧 org 登记 ledger D3）、11（"回"→"下一步" + P17 反向指引）。11/11 逐条处理。✅

## 三、验证结果（必跑三项全绿）

1. **脚本**（scratch/t2_P11/，G16 先 cp 后跑）：基线 01 rc=0（0.010s）、02 rc=0（1.28s），输出与教程逐行一致（KL=0.0204、adv ±1.0/−0.58/1.73、0.38→0.83、零梯度组 [1,6,6,6,6,6]、BC 0.50→1.00）；修后复跑 diff 仅 2 行**有意修正**的 verl 映射打印行（kl_penalty→kl_ctrl.kl_coef），全部训练数值零变化。std 双口径验证脚本：std(G)=0.5→±1.0、std(G−1)=0.5774→±0.866，G=1 退化 A=0——新小节数字全部实弹。
2. **作业四象限**：pytest+参考答案 **5 passed**；直跑+参考答案 **5/5 🎉**；pytest 空跑（未实现骨架）**4 failed + 1 skipped**（G18 生效，题 5 不再假 PASSED）；直跑空跑 4 失败 + 1 ⏭️——两入口 SKIP 语义一致。
3. **check_latex.py**：4 个改动 md（01/02/README/assignment）全部"未发现问题"；过程中清掉一处 `$USER` shell 变量误报（02 章 L99，改 `<你的用户名>`）。

## 四、通用问题候选（≤3）

1. **"教程教错 → 学生内化 → 费曼复述复现"的传导链**：S1 费曼自检脱口而出"KL 系数配 algorithm.kl_penalty"——教程的最佳实践表、映射提示、脚本注释三处互相印证的错误口径被学生当成三重确认。教训：同一事实至少三处拷贝的写作模式（本课程"教程↔脚本↔作业"三方可逐行对齐的卖点）一旦源头错，对齐度越高错得越牢。建议终版对"外部工具事实"（配置键名/API 签名）建立单点事实源清单，三处只引用不复述。
2. **"别的口径"式提醒不点名**：01 章脚注"`adv=[0.71,...]` 出自别的归一化口径"、骨架"多两个工程要求"、roadmap"N=1"——三处都是作者知道有差异但没写差异对象，学生（S1 慌 10 分钟、S2 找 3 分钟、S3 自推补齐）各自付出搜索成本。提醒句必须写全"和谁、差在哪"。
3. **离线不可核的外部配置键裸写**：verl 键名/CLI 参数在无 Docker/无网络审计机上只能降级核实——但教程以确定语气书写（"0.01-0.1"填进类型键）。建议外部工具章固定加一行"键名以所用版本文档为准"的常设免责（本批已在关键处补，属可推广写法）。

## 五、好写法候选（≤3）

1. **脚本 02 双实现 k3 互验 assert**（三学生共同点名）：同一公式 torch 版与纯 math 版各算一遍逐位比对——"数学别信单次计算，要自证"的工程化表达，且让 shape 注释 bug 这类问题无所遁形（本次修注释后 diff 干净正得益于种子固定+输出可逐位复现）。
2. **"零梯度组的两种命运"叙事**：敢让 GRPO 停在 0.83、BC 到 1.00，用诚实的对照数字讲清 RLVR 适用边界再连到 DAPO dynamic sampling——一个现象讲出三代算法因果，是"无卡学生也能讲的实测故事"（S3 面试资产）。
3. **双层作业声明**：教程练习 3（kl_budget_guard 完整版）与作业题 4（kl_budget_ok 简化版）显式互指"测试不覆盖 guard"——同一概念的难度分层 + 验收边界诚实声明，值得全课程推广。
