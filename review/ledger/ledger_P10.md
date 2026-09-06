# 问题台账 · Part 10（分布式训练）· T2 整改批

> 编号规则：P10-C{cc}-{SRC}-{nn}；cc∈{00=README,01..04=章节,SC=scripts,AS=assignment_10,RM=roadmap}；SRC∈{S1,S2,S3,T=T1预审,T2=教师实证}
> 状态：fixed=已修复且验证；disputed=争议/待 T0 裁决（禁止 wontfix）
> 关键实证：①脚本 04 CPU 档实测 `RuntimeError: FSDP needs a non-CPU accelerator device`（rc=1，T2 本机复现）——P0-1 根因；②DDP 桶数双卡实测**随时点变**：首次 backward 后 1 桶、rebuild_buckets 后稳态 2 桶（首桶 ~1MB 默认上限）——教程"只建 1 个桶"断言失真；③LLaMA 2 论文实值 MFU 43.9%（联网核实），教程写 ≈46%。

## Fixed（按严重度）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|------|--------|------|------|------|------|----------|----------|
| P10-CSC/C00/AS-01 | P0 | fixed | S1🔴 S2🔴 S3🔴 T1 缺口2 + T0 必修1 | 脚本 04 CPU 兼容是假承诺：docstring"单进程兼容（gloo/CPU）"+README 总述句"全部脚本 CPU 也行"+roadmap"全部脚本单进程可跑"，三方被 README 环境表格（CPU 行只列 01/02/03/05/06）证伪；CPU 实测裸崩 RuntimeError 无指引 | scripts/04 docstring+main；README L65/L70；assignment L25；教程 03 章 §2；roadmap L71/L376（docs 侧走 outline） | 脚本 04 docstring 如实"需要 GPU（FSDP1 不支持 CPU）"+ main() 开头设备预检：无 CUDA 打印三条人话指引（跑脚本 03 记账/单卡 NO_SHARD 说明）后 sys.exit(1)；README 总述句改"除脚本 04 需 GPU 外…"+表格措辞对齐；assignment 完成方式句同步；教程 03 章脚本 04 引用处注明需 GPU+单卡读数；roadmap 两条见 outline_suggestions 追加 | 修后 CPU 档 rc=1 且输出友好指引（不再裸崩）；GPU 单进程/双卡 rc=0（11.8MB/12.7MB/3.005、6.0MB/2.981 与教程逐字） |
| P10-C04-02 | P0 | fixed | S1🔴 S3🟡 T1 缺口3 + T0 必修2 | 04 章 Q2 off-by-one（数学边界错误）：`m+p−1>70 → m≥63`，但 m=63 时 7/70 **恰=10.00%** 不满足严格"<10%"；且学习目标承诺"推导 bubble"而正文只有公式+数字例，无推导 | 04 章 Q2 答案、§2 | Q2 改"m>63 ⇒ 整数 m≥64（m=64 → 7/71≈9.86%）"+保留 m=63 恰 10% 边界注；§2 补 3 行排槽推导（总槽 2m+2(p−1)=2(m+p−1)、气泡槽 2(p−1)、约分得公式）+"m<p 也成立（p=4,m=1→75%）" | 复算脚本：m=62→10.14%、m=63→10.0000%、m=64→9.86%；assignment 测试 `isclose(f(8,64), 7/71)` 佐证 64 |
| P10-CAS-03 | P0 | fixed | S2🟡 S3🟢 + T0 必修6（G18） | SKIP 语义两入口不一致：`_Skipped` 自定义异常不被 pytest 识别，空跑 pytest 题 5 显示 FAILED（应为 SKIPPED），违反 assignment"⏭️ SKIP 不算失败"+"python/pytest 两入口均可"的承诺 | test_distributed_exercises.py `_skip`/main | 仿 P8 修法：`_skip()` 在 pytest 可用时抛 `pytest.skip(reason)`；新增 `_skip_exceptions()` 列两种异常类供 except 链使用 | 四象限全绿：pytest+参考答案 5 passed；直跑+参考答案 5/5；pytest+未实现=4 failed（题1-4 应失败）+ **1 skipped**；直跑+未实现=4 失败+1 跳过 |
| P10-C01-04 | P1 | fixed | T1 缺口1 | 01 章"三个'装不下/不够'、三种并行"（学习目标 L11+§1 标题 L27）vs 表格实为 4 行 vs 章末"四种并行"——数目口径三处自相矛盾 | 01 章 L11/L27 vs L98 | 学习目标与 §1 标题统一改"四个/四种"（表格 4 行为事实源） | grep 复核全章无"三种并行"残留；章末"四种并行"保持一致 |
| P10-C01-05 | P1 | fixed | T1 缺口9 | "LLaMA 2 70B = 8 路 TP × ..."以省略号截断，句子未写完 | 01 章 L36 | 改写为完整句：总卡数 = TP × PP × DP；LLaMA 2 70B 官方报告 2000+ 卡、拆法未完全公开，04 章教"读配置"推理链 | 与 04 章 §3 表述互指一致 |
| P10-C04/SC-06 | P1 | fixed | S1🟡 T1 缺口7/8 + T0 必修3 | TP 形状链无显式展开（X→H_r→Y_r 形状推理全靠脚本注释）；"按行切=列并行"术语地雷（Wᵀ 存法换算无辨析）；教程称脚本 05 体现 f/g 双算子但未声明 f 的 backward 是手工 all_reduce | 04 章 §1 | 新增形状逐跳表（数字例 8×16×256、tp=2 五步含通信标注）+「按行切=列并行」辨析框（列并行切输出维→nn.Linear 转置存法→切 dim0 行，口诀）+ f/g 实现差异声明（教学等价，Megatron 为共轭 autograd 算子） | 形状表与脚本 05 L59-77 逐跳核对一致（A_r/B_r (chunk,IN)、H_r (·,chunk)、Y_r (·,IN)） |
| P10-C02/SC-07 | P1 | fixed | S2🟡 T1 缺口10 后半 + T0 必修4 | 桶数验证缺位：教程让读者用私有 API `_get_zeros_like_grad_buckets()` 验证"1 个桶"，但脚本 02 无此调用（学生扑空）；且 T2 实测该断言本身失真——首 backward 后 1 桶、rebuild_buckets 后稳态 2 桶（首桶 ~1MB 默认上限重切），"只建 1 个桶"不成立 | 02 章 §3；scripts/02 | 桶数打印落进脚本 02（训练后输出实测值+私有 API/时点/版本敏感注释）；教程改为可行步骤"跑脚本 02 看打印行"+如实口径"个位数量级（实测 1→2 桶演化）"，教学结论（toy 桶间重叠无从体现）不变 | bucket_probe.py 逐 step 实测：step0/1=1 桶，step2 起=2 桶（covered=628,161 全覆盖）；脚本 02 双卡打印"梯度桶数（rebuild 后，torch 2.6 实测）: 2" |
| P10-CSC03-08 | P1 | fixed | S1🟢 S2🟢 T1 缺口6 + T0 必修5 | fp16/bf16 措辞漂移：教程 03 章账本表与 assignment 用 bf16，脚本 03 docstring/注释写 fp16（字节数同、口径应统一） | scripts/03 docstring+两行注释 | 统一为 bf16，注明"课程主线 bf16；fp16 同为 2 字节账目不变" | 教程/脚本/assignment 三方 grep 无 fp16 参数/梯度残留 |
| P10-C00-09 | P1 | fixed | T1 §五（G2×21）+ T0 必修7 | 数学公式 21 处纯文本/代码块承载：02 章六个推导块、03 章账本表+三阶段表+激活公式+恒等式、04 章公式链+bubble、README 两处、assignment 四条公式 | 五个 md 全局 | 全部改 `$…$`/`$$…$$`（单行闭合、公式内无中文、`\text` 用英文）；02 章 ASCII 时序图与 04 章 GPipe ASCII 迁移为 PNG+中文解读 | check_latex.py 对 6 个改动 md 全部 0 问题（04 章首检 5 处 CHINESE_IN_MATH 已清零） |
| P10-C00-10 | P1 | fixed | T1 §五（G4）+ T0 必修7 | 全 Part 0 张图：DDP 桶化重叠时序（02 章 ASCII）、GPipe 时间线（04 章 ASCII）、ZeRO 三阶段显存对比（03 章表格数字）均"可画未画" | images/（新增）+ 三章引用 | 新增 3 张 PNG（图内英文、正文中文解读、相对路径引用）：ddp_bucket_overlap（机制示意）、gpipe_timeline（机制示意，气泡高亮）、zero_memory_bar（**公式值** 112/38.5/26.25/14 GB，1e9 口径，图注注明） | matplotlib 生成，逐张目检（修正过标签压行缺陷）；数值与教程表格一致 |
| P10-Cxx-11 | P1 | fixed | T1 §五（G13×3） | 外部来源数字无出处/与论文不符：①激活公式 34+5as/h 无出处（2205.05198 在他章有引）；②LLaMA 2 70B "MFU≈46%"与论文实值 43.9% 不符（联网核实，46% 更接近 Llama 1）；③NVLink ~900GB/s、跨机 25-100GB/s 无规格出处 | 03 章 L61、04 章 §3/Q1 | ①激活公式标 Korthikanti et al. 2022 arXiv 2205.05198；②MFU 改 43.9% + Llama 2 论文（arXiv 2307.09288）链接；③NVLink 分代规格（4.0 双向 900GB/s / 3.0 600GB/s，NVIDIA 规格）+ InfiniBand 200-800 Gbps≈25-100GB/s | MFU 论文值经 WebSearch 核实（arXiv 2307.09288）；硬件规格为厂商公开规格页口径 |
| P10-Cxx-12 | P1 | fixed | T1 §五（G14×3）+ S1 卡点6/S3 卡点5 | 实测块缺复现口径：01 章（仅"双卡"）、03 章 FSDP（无 torch 版本、无单卡读数说明）、04 章（5.96e-07/4.380254 无 device 档位；单进程读数不同无预告） | 01/03/04 章实测块 | 统一补四元组（卡型×torch 2.6.0+cu124×NCCL×档位）；01 章标注"节选"+单进程读数差异预告；03 章补单卡 NO_SHARD 读数（12.7MB/3.005）+"数字随 world_size 变"；04 章补"单进程误差恒 0.00e+00（切分退化恒等）" | T2 本机三轮实跑留档（scratch/t2_P10/logs_、relogs_、fin_）；单进程/双卡读数与补注逐字一致 |
| P10-Cxx-13 | P1 | fixed | T1 §五（G10×6） | ①02 章五件套骨架 E/dataset 未定义未声明；②no_sync 片段 nullcontext 未注明导入；③README 示例命令假文件名 01_xxx.py；④桶数验证宣称可验证但脚本无落地（并入 #07）；⑤01 章实测输出摘录未标"节选"；⑥4.380254/5.96e-07 依赖运行时打印——实跑核对 | 02/01 章、README | ①骨架前加"教学骨架+可运行版见脚本"声明（含 gloo 回退/取模差异）；②片段补 `from contextlib import nullcontext`；③改 `01_distributed_basics.py`；⑤标"节选"+略段说明；⑥双卡实跑逐字核对通过 | 见本行各处；⑤⑥由 T2 三轮日志背书 |
| P10-Cxx-14 | P1 | fixed | T1 §五（G1×2） | 编号点挤行：03 章 Q3 答案"①激活②通信③缓冲"三点挤一段；assignment Q2 提示四点挤一段 | 03 章 Q3、assignment Q2 提示 | 每点独立成行（编号点换行） | 目检+引用块内分行渲染正常 |
| P10-CRM-15 | P1 | fixed(改走 outline) | T1 缺口4 + S3 裁决 + T0 必修8 | roadmap 三处与课程事实源矛盾/失真：①"zero1 vs zero2 谁更省取决于 N"（实为恒更省 Δ=2Ψ(1/N−1)≤0，N 决定省多少）；②"全程 CPU 可学/全部脚本单进程可跑"（04 需 GPU）；③"ZeRO/FSDP2（fully_shard）"措辞超前（脚本用 FSDP1，教程双写法） | docs/course_roadmap_v3.md L71/L376/L380/L399（**REPO 只读不改 docs**） | 按战役规则追加 REVIEW/outline_review/outline_suggestions.md 三条（含建议改写文本与裁决依据） | outline_suggestions.md 末尾三条 P10 T2 已在案；S3 数学裁决 Δ=2Ψ(1/N−1)≤0 复算无误 |
| P10-CSC03-16 | P2 | fixed | T1 缺口5 | 脚本 03 `zero_accounting_simulation` docstring 首句"模拟 ZeRO 三阶段"但代码只实现 ZeRO-1 布局（过度宣称） | scripts/03 | docstring 如实"落地 ZeRO-1 布局；ZeRO-2/3 扩展留作教程动手 2" | 与教程 03 章"ZeRO-2/3 复算留作动手 2"口径一致 |
| P10-CSC06-17 | P2 | fixed | T1 §三-8 附带 | 脚本 06 `torch.cuda.set_device(rank)` 未取模 device_count（01/02/04 均有），nproc>卡数时崩 | scripts/06 setup | 加 `% torch.cuda.device_count()`（与 01/02/04 同款）+注释 | py_compile+双卡实跑 rc=0 |
| P10-CSC01-18 | P2 | fixed | T1 §三-1 建议 | 脚本 01 all_gather 只打印不断言（broadcast/reduce_scatter 有 assert，语义验证不对称） | scripts/01 §3 | 补 `assert all(t.tolist() == [r]*2 ...)` 一行 | 单进程/CPU/双卡三档实跑 rc=0（断言通过） |

## Disputed（待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P10-D1 | DDP 桶数终版口径：原教程"只建 1 个桶"（S2 手动实验亦得 1）vs T2 双卡实测 rebuild_buckets 后稳态 2 桶（首桶 ~1MB 默认上限重切）。T2 已按"如实实测（1→2 演化）+ 个位数量级结论不变"修正，但"是否在正文展开 rebuild_buckets 时机"属课程深度决策 | 建议 T0 复核 02 章新口径；面试资产清单（S3）中"toy 只有 1 桶"表述宜同步为"1-2 个桶" |
| P10-D2 | LLaMA 2 MFU 教程原文 "≈46%" vs 论文实值 43.9%（T2 联网核实）：46% 与 Llama 1 报告值接近，疑为来源混淆；已按论文改 43.9% 并给出处。若 T0 认为单处外部数字偏差应升 P0 可升级，整改动作已完成故不再变动 | 已修；T0 确认分级即可 |
| P10-D3 | 激活公式 sbh(34+5as/h) 仅标注出处（Korthikanti et al. 2205.05198）未逐项复算（S3 同样留作论文口径）；逐项推导需引入 GQA KV-head 维度，属内容扩展 | 维持"出处+口径一致"现状；如终版要逐项推导再立新任务 |
| P10-D4 | roadmap L380 "03 …FSDP2（fully_shard，FSDP1 已弃用）"与脚本实际用 FSDP1 的措辞差（教程双写法自洽）——已在 outline_suggestions 给出改写建议，属 docs 侧终版统一 | 终版大纲裁决 |

## 统计

- **Fixed：18 条**（P0×3、P1×12、P2×3）
- **Disputed：4 条**（桶数口径、MFU 分级、激活公式深度、roadmap FSDP2 措辞）
- 验证：脚本三档三轮全绿（修后 CPU 01/02/03/05/06 rc=0、04 友好报错；GPU 单进程 6/6 rc=0；双卡 6/6 rc=0，验收数字 5.96e-07 / 4.380254 / 6.0MB / 2.981 / 3.612 逐字复现）；作业四象限全绿（参考答案 pytest 5 passed、直跑 5/5、空跑两入口 SKIP 一致）；check_latex 6 文件 0 问题
