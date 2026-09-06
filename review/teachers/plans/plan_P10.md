# Part 10（分布式训练）分章审计计划 — T1 主教预审

- 预审日期：2026-09-04
- 对象：`REPO/courses/Part10_distributed/`（tutorial 5 个 md ≈888 行 + scripts 6 个 .py 共 884 行）+ `assignments/assignment_10/`（assignment.md + 骨架 + 测试）+ `docs/course_roadmap_v3.md` 节点 10（L376-402）
- 预审方式：只读通读 + 手工复算交叉核对（参数量 628,161 / 6,298,624 / ≈2.96M 三个参数量已手工复算与教程一致），未运行任何脚本、未改任何课程文件
- Part 性质：全课程唯一"多卡"主题 Part；roadmap 承诺"全部脚本单进程可跑"，作业纯 CPU 可完成；含 ZeRO/TP/PP 三组硬公式，是公式-代码-实测三方口径审计的重镇

---

## 一、C1 标题-内容对应表（5 文件逐个）

| # | 文件（行数） | 标题 | 实际内容 | 对应脚本 | 章末作业映射（实况） | C1 判定 |
|---|---|---|---|---|---|---|
| 0 | README.md (105) | Part 10: 分布式训练 — 从单卡到多卡 | 学习目标（4 条）、章节导航表、前置知识（Part 8/9/7/6）、路线图、环境与"无多卡学习路径"表、四并行一表通、作业指引、资源 | 全部 6 个 | 题 1-4 必做 + 🌟题 5 可选 ✅ 与 assignment 一致 | 基本对应；**L65"全部脚本单进程可跑（CPU 也行）" vs L70 CPU 行只列 01/02/03/05/06（漏 04）vs 脚本 04 docstring 自称"单进程兼容（gloo/CPU）"——三方口径矛盾**（见 §三-8、§七-2） |
| 1 | 01_why_and_collectives.md (134) | 为什么并行 + 分布式 Hello World | 四痛点→四并行表、SPMD 心智模型、四集合通信原语表+脚本验证、torchrun 六类报错 FAQ、彩蛋坑（NCCL p2p）、思考题 3 | `01` | 题 1（all-reduce 语义）✅ 映射正确 | 基本对应；**学习目标 L11 与 §1 标题 L27 均说"三个'装不下/不够'、三种并行"，但表格实为 4 行（DDP/ZeRO/TP/PP），章末 L98 又说"四种并行"——三/四数目口径自相矛盾** |
| 2 | 02_ddp.md (318) | DDP：数据并行深入 | DDP 五件套、三步推导"平均==大 batch"+三前提表、桶化 all-reduce 时序（ASCII）、no_sync 数学、双卡实测、动手 1（batch×accum 扫描）/动手 2（平均的平均验证） | `02` | 题 3（DistributedSampler 证明）✅ | ✅ 对应（本 Part 质量标杆章：推导完整、口径声明最全）；遗留"1 个桶"断言的实证链问题（§三-2） |
| 3 | 03_memory_zero_fsdp.md (193) | 显存账本与 ZeRO / FSDP | 16Ψ 五项构成表、7B/70B 对照表、ZeRO 三阶段表、激活值账外声明、FSDP1/FSDP2 双写法、决策树、动手 1（填表）/动手 2（stage 扩展） | `03` `04` | 题 2（显存账本计算器）✅ | ✅ 对应；**公式在 md 表格/正文均为纯文本未 LaTeX（G2 重灾）；FSDP 实测块缺复现口径（G14）** |
| 4 | 04_tp_pp_and_beyond.md (138) | 张量并行、流水线并行与工业栈（进阶可选） | Megatron 列/行切分+f/g 算子、GPipe 时间线（ASCII）、bubble 公式、3D 并行、工业参考栈表、毕业小结 | `05` `06` | 题 4（TP 分块数学）+ 🌟题 5（bubble）✅ | 基本对应；**Q2 答案 bubble 边界 off-by-one：m≥63 应为 m≥64（m=63 时 bubble 恰=10%，不满足"<10%"；assignment 测试 `isclose(f(8,64), 7/71)` 佐证正确答案是 64）** |

- **导航链完整性 ✅**：README 末"上一章 Part 9 / 下一章 Part 11"双向链接实存；01→02→03→04 章末"下一步"全部贯通；04 章末有"上一章 + README"回链。无 P08 式断链。
- **G17 章末作业指引 ✅ 全对**：01→题1、02→题3、03→题2、04→题4+题5，与 assignment 题 1-5 语义逐一对上（对照 P08 三处错链，本 Part 干净）。
- **README 导航表脚本列 ✅**：01/02/03 04/05 06 与 6 个脚本一一对应，无重号。

## 二、学生单元划分（三学生 × 三单元）

| 单元 | 文件 | 体量 | 难度画像 | 审计重点 |
|---|---|---|---|---|
| **单元 A** | README + 01 + 02 | 105+134+318 ≈ 557 行 + 脚本 01/02 (266 行) | 01 ★★（心智模型为主）、02 ★★★★（全 Part 推导顶峰：三步推导+no_sync 数学+桶化时序） | 01 章"三/四"口径；四原语语义 vs 脚本 01 输出（含 reduce_scatter 单进程退化分支）；02 章推导逐步核对（尤其 all-reduce 同步对象是 `.grad` 当前值这一句）；"1 个桶"断言裁决；动手 1/2 验收数字溯源；README 全局口径矛盾（主教统裁） |
| **单元 B** | 03 + assignment（题 1-3 + 思考题 Q2/Q3/Q4） | 193 行 + 脚本 03/04 (290 行) | 03 ★★★★（数字口径最密集：16Ψ/ZeRO 三档/N=1 锚点/逐字节模拟） | 16Ψ 五项 vs 脚本 03 逐行（**教程 bf16 vs 脚本 docstring fp16 术语漂移**，字节数相同但口径应统一）；ZeRO 公式三处（教程表/脚本打印/assignment 题 2）一致性；N=1 全退化锚点；动手 1/2 验收数字（26.25/262.5、+233/−103/−131、50.4MB）复算；FSDP1 弃用声明与 FSDP2 写法事实性；**脚本 04 CPU 单进程可跑性裁决（P0 候选）** |
| **单元 C** | 04 + assignment（题 4/5 + 思考题 Q1/Q5） + roadmap 节点 10 对照 | 138 行 + 脚本 05/06 (328 行) | 04 ★★★★（公式边界敏感：bubble off-by-one 实锤在此；TP 形状链靠脚本注释补全） | f/g 算子语义 vs 脚本 05 实现（f 的 backward 是手工 all_reduce 非 autograd.Function）；TP 形状链逐跳标注核对；GPipe 时间线 vs 脚本 06 backward 反序；loss 汇总"/world 会砍半"注释 vs assignment Q4 呼应；4.380254 与 5.96e-07 两个硬数字的复跑核对预案；`tools/verify_paper_formulas.py` verify_bubble 断言三方一致（教程/作业/工具） |

- 学生 1→A、学生 2→B、学生 3→C。README 由主教统裁（导航/环境承诺/四并行表是全局口径源）。
- 三人共同动作：assignment.md 全文对读（本 Part 作业与教程咬合度极高，题 2 的"恒等式预警"、题 3 的"补齐样本不撞车"证明都要回指教程）。

## 三、本 Part 特有审计要点（任务点名八项逐一落地）

1. **集合通信原语讲解**（01 章 §3 ↔ 脚本 01）：
   - 四原语表（broadcast/all_reduce/all_gather/reduce_scatter）语义列与"训练中的用途"列事实性核对；"平均不是求和"在脚本 01 L70-75 有 SUM/world 演示 ✅。
   - 脚本 01 实测输出块（教程 L74-79）逐行核对：world=2 时 [2] 行 expect=3、`[1.5,...]`、[4] 行 rank0 得 [1,3]——预审已手工复算全部正确；执行时实际跑单进程+双进程各一遍留档。
   - 审计暗礁：教程只摘录 [2]/[4] 两段输出未标"节选"（G10 轻项）；all_gather 断言缺失（脚本只打印不断言 [3] 语义，broadcast/reduce_scatter 有 assert）——确认是否算教学取舍并评估补 assert 建议。
2. **DDP 桶化 all-reduce 与 no_sync**（02 章 §3/§4 ↔ 脚本 02）：
   - 三步推导（L64-90）逐行核对：梯度线性→按 rank 分组→DDP SUM÷N；三前提表（等大分片/均值 loss/同起点）✅ 预审复核无误。
   - no_sync 数学（L173-188）：关键句"第 K 步 all-reduce 同步的对象是 `.grad` 缓冲的当前值"——这是与 PyTorch 实际行为（Reducer 对 `.grad` 累积值整体 all-reduce 后 ÷world）一致的表述，执行时对照 `dist.reducer` 文档确认表述无歧义。
   - **"DDP 实际只建了 1 个桶"断言（L147-149）**：理由是 628,161 参数 fp32 ≈2.5MB < 桶上限 25MB——但 DDP 首桶默认 1MB、首次 backward 后 rebuild_buckets 才合并；结论对（rebuild 后单桶）但教程把推理链简化成"2.5MB<25MB 故 1 桶"，且验证手段 `len(ddp.reducer._get_zeros_like_grad_buckets())` 是私有 API、脚本 02 中并无此代码。执行时：①核对该私有 API 在 torch 2.6 存在性；②裁决"首桶 1MB"细节是否需要在教程加半句（防止较真学生复现出 2 桶）。
   - `find_unused_parameters` / `broadcast_buffers` / "gloo 不支持 ReduceOp.AVG"（脚本 02 L136 注释 ✅）三条陈述核对。
3. **显存账本数字口径**（03 章 §1 ↔ 脚本 03 ↔ assignment 题 2 三方）：
   - 16Ψ 五项构成：教程表 bf16 参数 2 + bf16 梯度 2 + fp32 master 4 + 动量 4 + 方差 4；**脚本 03 docstring 写"fp16 参数 2 + fp16 梯度 2"——bf16/fp16 术语漂移**（合计不变，但课程主线是 bf16 且 fp16 有 loss-scaling 语义差异），判级并统一。
   - 7B/70B 表与 N=8 档：112/38.5/26.25(动手补)/14 GB 与 1120/385/262.5/140 GB——预审复算全对；GB 口径 1e9 已声明 ✅。
   - ZeRO 恒等式：(2Ψ+14Ψ/N)−(4Ψ+12Ψ/N)=(2/N−2)Ψ<0 恒成立、N=1 四式全等 16Ψ——教程/assignment/脚本注释三方一致 ✅；**唯 roadmap L400 写"zero1 vs zero2 谁更省取决于 N"，与三方矛盾（省的"幅度"取决于 N，"谁更省"不取决于 N）→ roadmap 侧错误**。
   - TinyGPT Ψ=6,298,624 与 stage3@N=2≈50.4MB：预审手工复算完全一致 ✅（执行时脚本跑一遍留档即可）。
4. **ZeRO 1/2/3 划分**：三阶段表（切什么/每卡/通信代价）事实性：ZeRO-1/2 通信与 DDP 相同、ZeRO-3 ≈1.5×——与 ZeRO 论文口径一致；脚本 03 Part A 只打印 ZeRO-1 列（教程声明一致 ✅）；**脚本 03 docstring 首句"模拟 ZeRO 三阶段"但代码只实现 ZeRO-1 布局**（fp16 参数全量+fp16 梯度全量+三类分片）——docstring 过度宣称，判 P2 修 docstring 或补 stage 参数。
5. **FSDP**（03 章 §2 ↔ 脚本 04）：FSDP1 弃用提示 ✅、FSDP2 `fully_shard` 先子模块后根模块写法 ✅（torch 2.6 导入路径 `torch.distributed.fsdp.fully_shard` 核对）；"整体包裹 vs auto_wrap_policy"差异声明 ✅（脚本 04 确未传 auto_wrap_policy，注释里有进阶提示 ✅）；Q2 的 `reshard_after_forward=False ⇒ SHARD_GRAD_OP/ZeRO-2 语义` ✅；11.8MB/6.0MB/loss 2.981 实测块缺 torch 版本口径（G14）。
6. **张量并行切分形状链**（04 章 §1 ↔ 脚本 05）：
   - Megatron 列/行并行、f/g 共轭、"每层每方向恰一次 all-reduce"事实性 ✅。
   - **缺口：教程正文无显式形状链**（X:(B,S,IN) → H_r:(B,S,chunk) → Y_r:(B,S,IN) 部分和），形状推理散落在脚本 05 注释（含"B 是 W2 转置存法"的关键澄清）；审计判定是否补一段形状标注表（任务点名要点，倾向补）。
   - **实现差异未声明**：教程称脚本 05 体现 f/g 双算子，实际 g 是 `AllReduceSum`(autograd.Function)、f 的 backward 是脚本 L89 手工 `all_reduce(X_tp.grad)`——教程未声明这一简化。
   - "Attention QKV 按头切=天然列并行，呼应 Part 7 GQA"事实性核对。
7. **GPipe/1F1B 气泡公式**（04 章 §2 ↔ 脚本 06 ↔ assignment 题 5 ↔ tools/verify_paper_formulas.py 四方）：
   - bubble=(p−1)/(m+p−1)、p=2/m=4→20%、1F1B 激活驻留 m→p：四方一致 ✅（工具 verify_bubble 用工作量定义独立模拟，测试 `f(8,64)=7/71`）。
   - **实锤：04 章 Q2 答案"(p−1)/(m+p−1)<0.1 → m+p−1>70 → m≥63" off-by-one**——m>63 ⇒ 整数解 m≥64（m=63 时 7/70 恰=10% 不满足"<10%"）；assignment 测试取 m=64 佐证。判 P1 数学边界错误。
   - GPipe 时间线图（F1-F4 连排+反序 backward）vs 脚本 06 `reversed(range(n_micro))` ✅；loss 汇总"不能 /world"（脚本 L187-188 注释）与 assignment Q4 提示互指 ✅。
   - 脚本 06 两个 opt 均声明但从未 step（教学只验证 loss 一致）——确认教程/脚本是否需要一句声明。
   - NCCL p2p 卡死是"4090+4090D 混合机型"特有：审计机若为同型双 4090 复现不出属正常，验证 gloo 解法代码路径（L119-120 new_group(backend="gloo")）即可，不得反推教程造假。
8. **单进程可跑性（README 承诺）——本 Part 最高优先裁决项**：
   - README L65"全部脚本单进程可直接跑（…CPU 也行）"vs L70 CPU 行"01/02/03/05/06"（漏 04）vs 脚本 04 docstring"单进程兼容（gloo/CPU）"vs README L71"1 张 GPU：以上全部 + 04 的 FSDP（单卡分片…能跑通）"vs roadmap L376"全部脚本单进程可跑"vs assignment L25"先跑一遍脚本 01-06（单进程即可）"。
   - 矛盾核心：FSDP1 在 CPU/gloo 上能否跑。执行时实跑 `python 04_fsdp_gpt.py`（先按 G16 复制到 scratch）裁决：能跑 → 修 README 表格（CPU 行补 04）；不能跑 → README L65/roadmap/assignment 三处承诺全假，判 P0 并定统一口径（04 标注"需 ≥1 GPU"）。
   - 附带核查：脚本 06 `torch.cuda.set_device(rank)` 未取模 device_count（01/02/04 均有取模），nproc>卡数时会崩——健壮性备注；README L73 示例命令用假文件名 `01_xxx.py`（G10）。

## 四、scripts 运行档位建议（审计执行时按档验证，先复制到 scratch，G16）

| 脚本 | 档位（docstring/代码实据） | 预计耗时 | 审计动作 |
|---|---|---|---|
| 01_distributed_basics.py | 单进程 CPU/gloo；torchrun 双进程；双卡 NCCL | 秒级 | 三档各跑一遍，留存输出与教程 L74-79 比对 |
| 02_ddp_gpt.py | 单进程兼容；torchrun 2 卡推荐（教程 §5 数字档）；CPU 双进程不要求 GPU | GPU 秒级~0.2s wall；CPU 约 1-3 分钟 | 复核 §5 吞吐带 77k-95k 仅双卡档可比；CPU 档记录实际时长（G8 备注：无 STEPS toy 档/无 flush，但脚本本身 toy 规模，预检不判违规）；验证打印行"batch×accum×world 变量拼接"与动手 1 描述一致 |
| 03_zero_memory.py | 纯记账 CPU 单进程（**不 import dist，天然单进程**） | 秒级 | 直接跑，核对 Part B 断言与 Ψ=6,298,624 打印 |
| 04_fsdp_gpt.py | docstring 称 CPU 单进程兼容；显存对比需 GPU | 秒级 | **P0 裁决档：先跑 CPU 单进程**；再双卡档复现 11.8/6.0MB 与 loss 2.981 |
| 05_tensor_parallel.py | 单进程=稠密参照（误差应为 0）；双进程=分片验证（5.96e-07 档） | 秒级 | 双档各跑；单进程档误差应为 0.00e+00（教程未写单进程读数，顺带补） |
| 06_pipeline_parallel.py | 单进程=整模型参照（4.380254 核对点）；双进程=GPipe 流水线 | 秒级 | 双档各跑比对 loss 一致到 1e-6；同型双卡若 p2p 不卡死，验证 gloo 分组代码路径存在性即可 |

- 双卡档说明：本机 4090+4090D 混合机型正是教程"彩蛋坑"的发生环境，教程口径（RTX 4090×2 + torch 2.6.0+cu124 + NCCL）与审计机一致，双卡档数字具备可比性；run-to-run 波动大的项（02 吞吐）只判量级带不判点值。

## 五、格式规范预检（G1/G2/G4/G10/G13/G14）违反位置清单

> 按战役规范原文（00_master_plan §10）归类；执行学生对照定义复核定级。预检合计 **36 处实锤 + 5 处待核**。

- **G1（编号点必须换行）— 2 处**
  1. 03 章 L139（Q3 答案）："还要算：① 激活…；② 通信是否成为瓶颈…；③ NCCL 通信缓冲…"三点挤同一段。
  2. assignment.md L127-128（思考题 Q2 提示）："① 有效 batch 变大…② 忘了 set_epoch…③ lr…④ 打印的是本 rank…"四点挤同一段。
- **G2（数学一律 LaTeX；禁止代码块承载推导）— 21 处，重灾区**
  1. 02 章 §2 四个推导代码块：L64-66（L(θ) 定义）、L71-73（∇L）、L76-83（分组重排，**块内含中文标注"g_r：rank r 的本地 batch 梯度"**）、L88-90（最终等式）。
  2. 02 章 §4 两个推导代码块：L173-176（∇L_big/ĝ_k）、L185-188（all-reduce 展开）。
  3. 03 章 L61：激活公式 "sbh×(34 + 5·a·s/h)" 正文纯文本无 `$`；L160：恒等式 "(2Ψ+14Ψ/N) − (4Ψ+12Ψ/N) = (2/N−2)·Ψ < 0" 纯文本；L38-41 与 L49-54 两个表格内公式未 LaTeX。
  4. 04 章 L32：`Y = gelu(X·W1ᵀ)·W2ᵀ` 用反引号代码承载公式；L80-84：TP×PP×DP 关系式放代码块；bubble 公式纯文本 L14/L71/L98；Q2 推导链纯文本 L113-114。
  5. README L18（ZeRO 公式链）、L85（表格内 bubble 公式）。
  6. assignment.md L45-47（四条 ZeRO 公式放代码块）、L64、L137-138（"zero2 = zero1 − (2−2/N)Ψ ≤ zero1"）。
  - 执行要求：整改统一走 `$…$`/`$$…$$` 并过 `check_latex.py` 机检（0 问题过关）；注意"$$ 单行闭合、列表/引用块内只用行内公式"。
- **G4（能画则画）— 3 组对象缺图 + 3 处 ASCII 候选迁移**
  1. 全 Part 无 `courses/Part10_distributed/images/` 目录、0 张 PNG。
  2. 可画未画：① DDP 桶化 all-reduce 与 backward 重叠时序（02 章 L123-137 现为 ASCII 图）；② GPipe 流水线时间线（04 章 L58-65 ASCII）；③ ZeRO 三阶段每卡显存对比柱状图（03 章表格数字现成）。
  3. 候选（轻）：03 章 L104-110 决策树 ASCII。若画图：图内文字英文、正文中文解读、相对路径引用（G4 原文）。
- **G10（正文代码可拼凑运行；引用真实；输出标注）— 5 处实锤 + 1 处执行核对**
  1. 02 章 L30-53 五件套骨架：`E`、`dataset` 未定义（骨架性质需显式声明或注"见脚本对应行"）。
  2. 02 章 L159-167 no_sync 片段：`nullcontext` 片段内未导入（脚本 02 L17 有导入，片段应注明）。
  3. README L73：示例命令 `01_xxx.py` 为假文件名。
  4. 02 章 L147-148：桶数验证私有 API `_get_zeros_like_grad_buckets()` 在脚本 02 无落地（宣称"可验证"但学生无处运行）。
  5. 01 章 L74-79 实测输出只摘 [2]/[4] 未标"节选"。
  6. 执行核对：4.380254（脚本 06）、5.96e-07/5.82e-11（脚本 05）两处"实测"数字脚本内无硬编码、依赖运行时打印——执行档位实跑后核对（G6/G10 交叉；seed 固定 1337/42+7，CPU 与 CUDA 可能差末位，跨设备差异要标注）。
- **G13（外部来源数字双列制）— 3 处**
  1. 03 章 L61 激活公式 sbh(34+5·a·s/h)：源自 Korthikanti et al.（2205.05198），该论文在 04 章 L54 与 README L101 出现但 03 章未注出处。
  2. 04 章 L83："LLaMA 2 70B 官方报告用 2000+ 卡、MFU≈46%"——无论文链接、无双列（README 参考资源也未列 Llama 2 论文）；执行时核对论文实值。
  3. 04 章 L108（Q1 答案）："NVLink ~900GB/s vs 跨机 ~25-100GB/s"硬件规格数字无出处。
- **G14（实跑日志复现口径）— 3 处（README 全局声明可部分兜底，判级留执行）**
  1. 01 章 L74-79 实测块仅"（双卡）"，无 torch 版本/device。
  2. 03 章 L72-76 FSDP 实测块（11.8/6.0MB、loss 2.981）无 torch 版本。
  3. 04 章 L46-48（有 2×4090 无 torch 版本）与 L69（4.380254 无 device，CPU/CUDA 差异未标注）。
  - 正面标杆：02 章 §5 及动手 1/2 验收数字口径齐全（RTX 4090×2, torch 2.6.0+cu124, NCCL / CPU fp32），可作为另两章整改模板。
- **附带（非本组 G 项，登记备查）**：G15 相关——ZeRO 公式在教程/脚本/作业三处拷贝且已发现 roadmap 第四处口径漂移（见 §六-1），整改时以教程 §1 表为唯一事实源；G8 相关——02 脚本 CPU 时长未标注。

## 六、跨 Part 一致性

1. **与 roadmap_v3 节点 10（L376-402）**：
   - ✅ 教程 4 章/脚本 01-06 清单/torchrun 命令/作业 5 题/硬数字（TP 6e-07<1e-5、PP 4.380254、bubble 对公式）全部对得上；`tools/verify_paper_formulas.py` 气泡断言实存（verify_bubble，公式同款）。
   - ❌ **"ZeRO 记账可复算（zero1 vs zero2 谁更省取决于 N——能说清为什么）"与教程/assignment"ZeRO-2 恒 ≤ ZeRO-1（幅度随 N 变）"矛盾——roadmap 侧表述错误**，修 roadmap 或改题意（C1 缺口 #4）。
   - ⚠️ roadmap L380 "03 显存账本与 ZeRO/FSDP2（fully_shard，FSDP1 已弃用）"——教程标题实为"ZeRO/FSDP"、脚本 04 用 FSDP1+教程双写法；口径可自洽但 roadmap 措辞略超前，登记待核。
2. **与 Part 8**：README L3/前置 L35 引 Part8 01 章（路径实存 ✅）；"DDP/FSDP 改变的只是梯度从哪来、状态存在哪"的分工声明清晰，无重叠冲突。
3. **与 Part 9**：01 章前置 + 02 章 L26 引 Part9 01 章"内核异步执行"（路径实存 ✅），作为桶化重叠的前提声明恰当。
4. **与 Part 7**：04 章 L28/50 引 Part7 03 章 GQA（路径实存 ✅）；**02 章 L245"呼应 Part 7 RMSNorm 一章"无路径链接**——RMSNorm 实为 Part7 02 章（02_modern_components.md），建议补相对路径与其他引用统一（轻，缺口 #10）。
5. **与 Part 6**：README L42 可选引 Part6 03 章（路径实存 ✅）。
6. **与 Part 11**：README 末"下一章：Part 11 verl 对齐实战"链接实存 ✅；Part10 定位"手写原理"、Part11"工业框架"，与 P08→P11 双轨模式一致（Part 11 侧 reciprocal 措辞留该 Part 审计，不读其内容）。
7. **assignment ↔ 教程**：题 3"补齐样本不撞车"的断言预审已验证数学上成立（shuffle 后补齐尾部、位置 i 与 n+i 模 world 不同余）——但证明依赖 torch 实现"先 shuffle 后补齐"，执行时以 `sampler_indices` 骨架行为核对；题 5 SKIP 机制（返回 None → ⏭️）与 README/assignment 声明一致 ✅。

## 七、C1 缺口汇总（预审实锤 10 项，执行学生按清单定级）

1. **01 章"三个'装不下/不够'、三种并行"vs 表格 4 行 vs 章末"四种"——数目口径三处自相矛盾**（L11/L27 vs L29-34/L98）。
2. **README L65"全部脚本 CPU 单进程可跑" vs L70 CPU 行漏 04 vs 脚本 04 docstring"CPU 兼容"——三方矛盾，FSDP1 CPU 可跑性需实测裁决（P0 候选）**。
3. **04 章 Q2 bubble 边界 off-by-one：m≥63 应为 m≥64**（assignment 测试 7/71 佐证；P1 候选）。
4. **roadmap 节点 10"zero1 vs zero2 谁更省取决于 N"与教程/assignment"恒更省"矛盾**。
5. 脚本 03 docstring"模拟 ZeRO 三阶段"实际仅实现 ZeRO-1 布局（过度宣称，P2）。
6. 教程 bf16 vs 脚本 03 docstring fp16 术语漂移（字节数同、口径应统一）。
7. TP 形状链（X/H_r/Y_r shape）教程正文无显式标注，全靠脚本注释补全。
8. 教程称脚本 05 体现 f/g 双算子，未声明 f 的 backward 是手工 all_reduce（实现差异）。
9. 01 章 L36"LLaMA 2 70B = 8 路 TP × …"以省略号截断（疑似未写完或刻意悬念，需裁决改写）。
10. 02 章 L245"Part 7 RMSNorm 一章"无路径链接（实为 Part7 02 章）；02 章"只建 1 个桶"断言的实证链（私有 API + rebuild_buckets 时机）待执行复核（并入 §三-2）。

## 八、学生执行清单（各单元通用动作）

1. 逐章 C1：标题/小节 vs 内容 vs 对应脚本 vs 作业映射四栏核对（§一表为底稿），§七 10 项逐一定级（P0/P1/P2）并给修法。
2. 公式-代码互证：按 §三 八项逐条打开脚本行号比对（允许读脚本全文与 tools/verify_paper_formulas.py，禁止改课程文件；运行前先 cp 到 scratch，G16）。
3. 实测数字溯源：教程每个"实测"数字标三态（脚本硬编码 / 运行时打印可复现 / 无出处）；02 章吞吐带与波动免责句保留判级。
4. 档位实跑：按 §四 档位表执行，04 脚本 CPU 档结果即为缺口 #2 的裁决证据，最先做。
5. 格式扫描：按 §五 位置清单复核定级；G2 整改务必过 `check_latex.py`（0 问题过关），G4 出图遵守"图内英文/正文中文/相对路径/随交付附绝对路径"。
6. 产出：单元审计报告（缺口/格式/一致性/加分点 四类；加分点提示：本 Part 章末作业映射全对、参数量三处手工可复算全对、p2p 彩蛋坑记录诚实，是全课程口径质量上游水平），交主教合并。
