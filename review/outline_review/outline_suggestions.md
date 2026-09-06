# 大纲/路线图类建议清单（阶段 3 终版处理，不自动改 docs/）

> 全量期间各 Part 的 disputed 中属 roadmap/docs/大纲类的事项汇总于此，终版大纲报告时统一裁决。

| 来源 | 建议 | 影响 | 状态 |
|---|---|---|---|
| P01 disputed | roadmap/教程"15-30 分钟"时长低估（实际含脚本运行约 1-2 小时） | O6 时长修订 | 待终版 |
| P01 disputed | roadmap 措辞 2 处与教程不一致（环境自检节、承诺锚点） | O5 一致性 | 待终版 |
| P02 disputed D2 | roadmap 称"作业含采样"但作业无采样题 | O5 | 待终版 |
| P02 disputed D1 | 字符映射统一方向已按"Part2 侧统一"执行（Part1 零改动）；Part1 侧写法如需跟进在全量批2 一并看 | 跨 Part 一致性 | 已裁决：接受 T2 方案 |
| R2 deferred | 02 脚本 docstring 引 Anthropic 数字 1.9%（官方 2.1%）需对照原文 | C02 审计时核 | 批5 处理 |
| P08 T2 | roadmap_v3 节点 8 投机解码行过时："α≈0.65 → 实测 2.81 vs 理论 2.53" vs 06 章/脚本实测 "α≈0.60 → 2.47（121/49）vs 理论 2.31"（两组各自自洽，冲突在引用了哪一次运行）；建议改为 06 章口径或加注"α 随种子波动，以当次脚本输出为准" | O5 一致性 | 待终版 |
| P08 T2 | roadmap_v3 节点 8 "教程 8 章 01-08 / 脚本 01-10" 清单过时：实际 9 章（含 09 推理模型，脚本重命名为 13）+ 13 个脚本（11 幻觉安全、12 lm-eval）；v3.1 增量表已补但节点 8 正文未同步 | O5 | 待终版 |
| P08 T2 | roadmap_v3 "prompt masking 为什么用 ignore_index=-100" 超前于教程实现：教程/脚本用乘法 mask（无 -100），两种等价实现需对齐口径（建议 roadmap 改为"掌握 response-only loss 的两种等价实现"） | O5 | 待终版 |
| P08 T2 | roadmap "pytest assignments/assignment_8 全绿"已随 T2 修复成立（8 passed / 未实现时 8 skipped 无 failed）；建议终版抽查 | O5 | 已修复待复核 |
| P08 T2 | roadmap/教程 PagedAttention 口径：roadmap "41%→<4%" 是论文值，教程实测为 "41%→5%（含尾部半块的模拟值）"；建议 roadmap 注明"论文值"字样（教程侧已修） | O5 | 待终版 |
| P07 disputed | roadmap minimind 节点口径与教程 05 章事实源对齐（26M=8Q/2KV/θ1e6） | O5 | 待终版 |
| P07 disputed | minimind 官方 master 配置需联网窗口复核（MoE 参数量 198M 等） | C2 事实核 | 终版前联网窗口处理 |
| P08 disputed | 07 章三处"疑似乱码"经逐字复核为误报 | - | 已仲裁：关闭 |
| P07 disputed | temp/ 旧 ckpt 清理决策 | 仓库卫生 | 合回时由维护者决定 |
| P09 T2 | roadmap_v3 L600 技能验收线 "Triton 内核 ≥ SDPA 最优后端 **105%**" 与教程 05 章 "≥50% 合格 / >85% 优秀"（主表最慢场景 94.6%）三方口径冲突：S3 独立实测 6 场景仅 2 个 ≥105%（最慢 90.3%），105% 只在教程自注的 4090 D 空闲卡那次运行成立，而教程 4.4 已声明"跨卡数字不可直接比"——建议改为 85%（优秀线）并注"共享卡实测 90%~128%，空闲卡可超 100%（卡况敏感，见教程 4.4）" | O5 一致性 | 待终版 |
| P09 T2 | roadmap_v3 节点 9 "学习内容安排"第 1 条仍写"教程 4 章：01→04 / 脚本 01-08"，实际 v3.1 已扩为 5 章（+05 Flash Attention 毕业内核，543 行全 Part 之最）+ 9 个脚本（+09）；roadmap L607 自己都说"Part 9 新增手写 Triton FA 内核章"，§6 增量表已更新但节点 9 正文未同步 | O5 | 待终版 |
| P09 T2 | roadmap 节点 9 两条学习目标在教程正文 0 覆盖：①"SMEM 48KB/块"（4090 可经 `cudaFuncSetAttribute` opt-in 至 100KB——全教程 0 命中）；②"decode 为什么 memory-bound"（batch=1 单 token：权重读一次只用一次，算术强度 ≈1 FLOP/byte ≪ 82 → 带宽墙；02 章给了 roofline 工具但没出这道题）——建议 01/02 章各补 3 行（S3 报告已给出可写内容），或放宽 roadmap 目标措辞 | O5/O6 | 待终版 |
| P10 T2 | roadmap_v3 节点 10 学习验证 "ZeRO 记账可复算（zero1 vs zero2 **谁更省取决于 N**）" 表述错误：Δ = zero2−zero1 = 2Ψ(1/N−1) ≤ 0 对任意 N 恒成立（N=1 相等），N 决定的是"省多少"（幅度 (2−2/N)Ψ 随 N 增大趋近上界 2Ψ），不决定"谁更省"。教程 03 章动手 1 与 assignment 题 2/题 3 均为"恒更省"口径。建议改为："ZeRO 记账可复算（ZeRO-2 恒 ≤ ZeRO-1，N=1 相等；省多少 (2−2/N)Ψ 取决于 N——能说清为什么）" | O5 一致性 | 待终版（S3 已裁决：教程/作业对，roadmap 错） |
| P10 T2 | roadmap_v3 L71 "全程 CPU 可学：Part 10 **全部脚本**单进程可跑" 与节点 10 标题 "全部脚本单进程可跑" 均失真：脚本 04（FSDP1）在 torch 2.x 需要 GPU 加速器，纯 CPU 实测直接 RuntimeError（T2 本机复现，rc=1）。README 环境表格（CPU 行只列 01/02/03/05/06）是对的。建议 L71 改 "Part 10 除脚本 04（需 1 GPU）外全部脚本单进程可跑"；节点 10 标题同步。教程 README/脚本 04/assignment_10 的同款承诺已由 T2 在课程侧修正 | O5 一致性 | 课程侧已修，docs 侧待终版 |
| P10 T2 | roadmap_v3 节点 10 学习内容 "03 显存账本与 ZeRO/**FSDP2**（fully_shard，FSDP1 已弃用）" 措辞超前：脚本 04 实际用 FSDP1 API（教程 03 章为 FSDP1/FSDP2 双写法并注明弃用提示）。建议 roadmap 改为 "03 显存账本与 ZeRO/FSDP（脚本用 FSDP1，教程并给 FSDP2 fully_shard 写法）" | O5 措辞 | 待终版 |
| P14 T2 | roadmap_v3 节点 14 写"作业：**3 编码题**（指标公式/KV 容量账/浪费率）+ 4090 实验题"，实际 assignment_14 为 **4 编码题**（4×25=100，多出题 4 投机解码数学题）+ 题 5 stretch 调度模拟器（选做）。建议 roadmap 改"4 编码题 + 1 stretch 模拟器（选做）" | O5 一致性 | 待终版 |
| P14 T2 | roadmap_v3 节点 14 写"面试直通车 **4 问**自测"，assignment 🎯 面试直通车实为 **5 条**（多"投机解码什么时候负收益"）。建议 roadmap 改"5 问"或"4-5 问" | O5 一致性 | 待终版 |
| P14 T2 | roadmap_v3 节点 14 "三行对比表"术语 vs 教程 01 §3/脚本打印表实为 4 行（含 KV 显存行）：教程侧已加命名口径注（"三行"= 前三行定量指标，KV 行为定性补充）；roadmap 终版时可同步一句 | O5 | 教程侧已修，docs 侧待终版 |
| P13R T2 | roadmap_v3 节点 13 "学习内容安排"正文仍写"教程 2 章：01 → 02 / 脚本：01_minhash_dedup.py"，未含 00_scaling_laws（章节导航/学习地图/README 均已含 00 章且标"建议先读"，节点内部自相矛盾；L414/L596 增补行已登记 00 章）。建议节点 13 改"教程 3 章：00 scaling law 开篇（建议先读）→ 01 → 02 / 脚本：00（fit/scan/epoch 三模式）+ 01"，并在学习验证中补一条 scaling 口径断言（如"给定拟合参数复算 t/p(C) 或反算 R_eff"） | O5 一致性 | 待终版 |
| P13R T2 | assignment_13 五道题 100% MinHash/LSH，无 scaling law 编码题，学习者照节点+作业清单走会整体跳过 00 章（S3 硬卡点 1）。建议作业补 1 道 scaling 小题（如给定 E/A/α/B/β 与 C 算 N_opt/D_opt/t/p，或由 epoch 表反算 R_eff 折扣），与 roadmap 节点 13 同步修 | O5/O6 | 待终版 |
| P11 T2 | roadmap_v3 节点 11 "作业 11：**3 编码题**（稳健奖励函数三级抽取 / 组内优势+全同组退化 / KL k3 估计器+预算护栏）" 与 assignment 实际 **4 核心题（math_reward / group_advantages / k3_kl / kl_budget_ok，各 25 分）+ 1 stretch（zero_gradient_groups）** 计数漂移（把 k3+护栏合并计 1、stretch 未计）。建议改："作业 11：4 编码题（三级抽取奖励 / 组内优势+全同组退化 / KL k3 估计器 / KL 预算护栏）+ 1 stretch（零梯度组检测）"，与 assignment.md 分值表（4×25）一致 | O5 一致性 | 待终版（课程侧 assignment 无需改） |
| P11 T2 | roadmap_v3 节点 11 "脚本：01_reward_and_bridge.py" 漏列 `02_grpo_toy_train.py`——它是 01 章核心资产（三零件装配成会学习的 GRPO 循环）与最佳面试故事素材（0.38→0.83 零梯度组盲区），教程/README/作业多处引用。建议脚本行补 "01、02" 两个脚本 | O5 完整性 | 待终版 |
| P11 T2 | roadmap_v3 节点 11 论文验证项 "五步法检查组内优势公式（**N=1** 时退化成什么？）" 记号撞车：P10 语境 N=卡数（world size），此处实指组大小；且教程/作业全线用 G 记号。建议改 "组大小 **G=1** 时退化成什么？"（答案：mean=r、std=0、经 max(std,eps) 兜底 A=0——单样本零信号，故 G≥2；该结论已由 T2 补进教程 01 章"分母口径对照"小节） | O5 记号一致性 | 待终版（教程侧已补 N=1 内容） |
| P11 T2 | roadmap/README "官方 quickstart 明确单卡 ≥24GB" 与 02 章性能表 "~8GB" 的口径张力（S3 卡点：面试被追问"你到底占多少显存"会卡壳）——教程侧已补圆场句（24GB=含 vLLM KV cache 预留的保守整机需求 vs ~8GB=训练态推算占用）；roadmap 若引用 24GB 建议同步口径注 | O5 口径 | 教程侧已修，docs 侧待终版 |
| P12 T2 | roadmap_v3 节点 12 "作业 12：4 编码题（LoRA 参数账 / 合并数学 **W'=W+BA** / B 零初始化证明 / QLoRA 显存账）" 漏 α/r 缩放：教程 01 章、脚本 merge_lora（L229 `(alpha/r)*B@A`）、assignment 题 2 三方事实源均为 `W'=W+(α/r)·BA`。建议 roadmap 改为 "合并数学 W'=W+(α/r)·BA" | O5 一致性 | 待终版 |
| P12 T2 | roadmap_v3 节点 12 学习验证 "**面试直通车 4 问自测**" 在 P12 无落地栏目（各章只有"概念检验"3 题，S3 学习目标达成表同样标 ❌）。建议：教程侧加"面试直通车"栏目（可由 02 章 §5/概念检验扩写），或 roadmap 放宽为"概念检验 + 面经整理" | O5/O6 | 待终版 |
| P12 T2 | roadmap 节点 12 "QLoRA 的 NF4 + 双量化 + 分页优化器各省什么"：教程侧已由 T2 补齐（02 章分页优化器小节 + `optim: paged_adamw_8bit` 字段 + 01 章 QLoRA 表），roadmap 本身无需改 | - | 教程侧已修（备查） |
| P15 T2 | roadmap_v3 缺口登记表 L601 "多模态 VLM ｜ A 补充 ｜ Part 15（可选）｜ **归档（v2-T9）**" 与事实矛盾：`courses/Part15_vision_language/` 已建成（2 章 + 2 脚本 + 4+1 题作业，pytest 4 绿）；且与 L590 "多模态维持归档"、L32 全景图"多模态待补"三处互相冲突（S3 卡点 2）。建议 L601 状态改"✅ 已建（Part 15：LLaVA 两阶段 + CLIP/SigLIP 对齐 + 作业 15）"，L590 括注同步，全景图"多模态"待补字样删除 | O5 一致性 | 待终版 |
| P15 T2 | roadmap_v3 **节点 15（毕业验收）与 Part 15（VLM 课程）同号错位**无任何提示：节点 15 = "三个项目故事 + 全链面试模拟"，通篇不提多模态；缺口表落地去向又写 "Part 15"。按节点推进的学习者不会遇到已建成的 VLM 课程。建议节点 15 处加一句显式说明（如"注意：节点 15 ≠ Part 15 课程目录，多模态 VLM 见 courses/Part15_vision_language（选修补充）"），并在毕业验收的"硬数字清单"里登记多模态一条链（576/1017/1,856→29,620 均为可复算数字） | O5 路标 | 待终版 |
| P16 T2 | roadmap_v3 **无节点 16**（C1-1，重大）：L610 自述"16 个主节点（0-15）"逐节点详解止于节点 15（毕业验收），L601 缺口登记表把"多模态 VLM → Part 15"标**归档（v2-T9）**、Part 16 全文未提；而课程目录 Part 15/16/17 实体齐建，P16 03 章自述"生成线毕业（Part 1-16）"并指向 Part 17——路线图与已建成实体脱节，"归档"与"已建成"互相矛盾，按路线图自学会整体跳过两章。建议：v3 增补节点 15.5/16（多模态理解+生成）或在缺口登记表改口径"已建成 Part 15/16，节点详解另立"，并同步 L610 主节点总数 | O5 一致性 | 待终版 |
| P16 T2 | 设计文档（part15_16_multimodal_design.md）作业清单中"**SNR 与 β schedule**""**动态分辨率 token 估算（联动 a15）**"两题未落地且无参考答案支持（assignment_reference/assignment_16 仅 4 题，P16 disputed D1/D2）；若终版裁决补题需同步参考答案+test+分值表（现 30/25/25/20 已满 100，需重排） | O5/O6 | 待终版（T0 裁决） |
| P16 T2 | 视频生成无作业/自动验证落点（C1-4）：assignment 定位"全部纯 CPU"与视频实操（GPU）冲突，仅 03 练习 2 操作型题。建议在 assignment"实验题（观测型）"补一条视频观测题（无需参考答案），或 roadmap/设计文档放宽"操作型练习视作验证落点" | O5/O6 | 待终版 |
| P16 T2 | 设计文档 part15_16_multimodal_design.md 脚本 01 运行时长写 "<60s"，教程/脚本统一口径为 "<30 秒"（本机 CPU 实测 6.4s）——docs 侧同步 | O5 口径 | 教程侧已修，docs 侧待终版 |
| P17 T2 | **roadmap_v3 无节点 16/17**：逐节点详解止于节点 15，Part 16（图像/视频生成）与 Part 17（Agentic RL，全课程结业章）双双缺席；而课程 README 宣布"Part 1-17 全部走完"、Part 17 README 自称"Part 1-17 结业章"——规划文档与课程内容失配，"下一个 Part 学什么"是断头路（S3 卡点 1）。建议 v3 增补节点 16/17（或在分叉节显式登记"算法线延伸章：16 图像视频生成 / 17 Agentic RL 训练侧 + 18 RAG / 19 Agent 应用侧"），并把 P17 的"姊妹篇分工"（训练侧 17 / 应用侧 19）写进两节点 | O5/O6 | 待终版（T0 统筹） |
| P17 T2 | P11↔P17 verl `compute_score` 签名漂移：P17 02 章练习 1 教四参签名 `compute_score(data_source, solution_str, ground_truth, extra_info=None)`（与 verl 实际一致），P11 02 章 L155 教二参简化签名 `compute_score(response, ground_truth)`，两章互不引用。建议 P11 侧复核 verl 实际签名并补"简化签名，实际为四参"声明（P17 侧已加常设版本免责） | 跨 Part 一致性 | 待终版 |
| P17 T2 | 02 章 verl multi-turn 配置键名（`multi_turn.enable` / `max_assistant_turns` 等）离线只能"示意+常设免责"处理（ledger disputed D2）：联网窗口对照 volcengine/verl 官方 docs 复核键名与示例可运行性 | C2 事实核 | 终版前联网窗口处理 |
| P19 T2 | **联网核对清单（降级核实移交）**：① Pi 博客 URL/日期（mariozechner.at 2025-11-30）与 smolagents"千行代码"官方口径；② arXiv 编号-标题对应（τ²-bench 2506.07982 / GiGPO 2505.10978 / AgentRL 2510.04206 / ReAct 2210.03629）；③ τ-bench 上游评分 bug 的具体 PR/commit（教程+脚本 03 两处现为"社区流传口径（转述待核）"）；④ SWE-bench Verified 官方榜现值（教程 80.9% 为 2026-09 转述口径）；⑤ MCP protocolVersion 现行版本（教程只提 2024-11-05，后续有 2025-03-26/2025-06-18）；⑥ **脚本 02 未知工具错误码裁决**：现回 -32601（method not found），S1 主张按规范应为 -32602（Invalid params）——ledger_P19 T-21 disputed，对照 MCP 错误码表后定，若改码需同步教程 2.2 实录 | C2 事实核 | 终版前联网窗口处理 |
| P19 T2 | 脚本 01/03 健壮性顺手项（plan §四附注）：print 无 `flush=True`、无短程环境档——纯 CPU 学生（慢 20-40 倍）中断即零输出。建议两脚本 print 加 flush 或提供 `DEMOS=0`（只跑 Section 0/0.5）/`R=1` 环境变量档 | O5 可运行性 | 待终版 |
| P19 T2 | 脚本 01 "每轮调用数上限"目前只在脚本 03 以 `calls[:3]` 最小实现（教程陷阱 3 已注明归属）。若终版愿意补齐教学闭环，可在脚本 01 加 `MAX_CALLS_PER_TURN` 常量（默认 None 关闭，不改变 Demo 轨迹） | O5/O6 | 待终版（T0 裁决） |
| P18 T2 | roadmap_v3 应用线 A1 明列"对比 ragflow（89.6k★）平台化能力与 llama_index（51.9k★）抽象"，教程侧此前零兑现（三生 + T1 三方确认）；本轮已在 02 章末补"工具选型速览"半页（RAGFlow=平台化 / LlamaIndex=分层抽象 / Data-Juicer=语料处理 / vLLM=推理底座，各一句定位+与手写件对应，标注"未实测、社区口径"）。建议 roadmap A1 行加"（02 章末'工具选型速览'已兑现，未实测口径）" | O5 一致性 | 教程侧已修，docs 侧待终版 |
| P18 T2 | Anthropic contextual retrieval 官方口径已 T0 联网定谳：5.7%→3.7%(−35%)→2.9%(−49%)→1.9%(−67%，含 rerank)，教程三处统一用的 1.9% 与官方一致且 (5.7−1.9)/5.7≈66.7% 自洽；旧台账"官方 2.1%"系误记（2.1% 对应 −63%，与 −67% 不能同真）。docs 侧任何引用处勿再写 2.1% | C2 事实核 | 已定谳关闭，docs 侧备查 |
