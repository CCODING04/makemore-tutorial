# S3 学习报告·P19（Agent 与 Function Calling · 应用线 A2 · 面试冲刺型审计）

> 审计人：学生 agent S3 ｜ 日期：2026-09-04 ｜ 材料范围：P19 tutorial 全部 .md + scripts 01-03 + 作业 19 题面 + roadmap v3 应用线 A2（只读，未看 assignment_reference）
> 实测环境：RTX 4090 / .venv（python3.12, transformers 4.57.x, torch 2.6.0+cu124）/ **离线环境（代理不可达）**

---

## 总分：9.2 / 10

一句话：P19 是"推理侧用 agent"的完整闭环——三个脚本我全部亲手复现，脚本 01 的四条 Demo 轨迹与教程**逐字一致**（贪心解码可复现的声明成立），脚本 03 我复现出 pass^1=0.00 与教程口径吻合；扣分点集中在离线跑法未写明、一处耗时口径不一致、框架认知停留在纸面。

---

## 卡点清单（按严重度）

| # | 卡点 | 严重度 | 说明 |
|---|---|---|---|
| 1 | **离线跑法缺失**：脚本 01/03 `from_pretrained` 未设 `local_files_only`，离线环境直接 ProxyError 崩溃（我首跑 Demo 全没跑）；模型 `Qwen2.5-0.5B-Instruct` 明明已在 `~/.cache/huggingface/hub`，加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 即过。README「📦 环境」节没有这一行 | 高 | 环境卡点，非脚本逻辑缺陷；但"全程 CPU 可学"承诺过一部分（02 秒级可跑，01/03 需模型） |
| 2 | **「MCP 三层」口径歧义**：教程的"三层"= 协议三层（MCP/A2A/AGENTS.md）；面试官问"MCP 三层"也可能指 MCP 自身的协议/客户端/服务器角色划分。两个口径都必须会答（见白板自测 3） | 高（面试向） | 02 章只教了前者，被问后者时容易愣住 |
| 3 | **框架认知是纸面的**：四流派只有一张对比表 + 选型直觉，没上手任何框架。面试追问"你用过 LangGraph 吗"只能答"读对比 + 写过裸 loop"——诚实但弱。roadmap 用词是"框架认知"，算兑现，但资产含金量有限 | 中 | 面试话术要提前备好："我有裸 loop 全部件在手，框架表是我选型的地图" |
| 4 | 耗时口径不一致：脚本 03 头部注释"GPU 实测 **~12-25 秒**" vs README/02 章"**~15-25 秒**" | 低 | 数字审计时被发现 |
| 5 | τ-mini 用户模拟器是**脚本化剧本**而非 LLM 扮演——教程 5.1 表格明确标注了取舍，但面试时必须主动说清，否则"我复现过 τ-bench"会被追打到破防 | 中 | 主动披露反而是加分项 |
| 6 | 作业 `agent_exercises.py` 为空白模板（4 failed + 题 5 正确 SKIP），本次时间所限未实现——4/5 必做题未动手 | 中（学生侧） | 测试框架与 SKIP 机制本身验证正常 |
| 7 | SWE-bench"Opus 4.5 80.9% 首个破 80%"是转述（Anthropic 公告口径），离线无法核实 | 低 | 教程已带"以 swebench.com 实时榜单为准"的免责 |

---

## 分章评分

| 章 | 分 | 理由 |
|---|---|---|
| README | 9.0 | 导航/前置/学习地图完整；"0.5B 的成功与翻车都是教材"定位诚实；环境节缺离线一行 |
| 01 手写 Agent Loop | 9.5 | 五部件逐行 + 轨迹实测逐字可复现（我验证）；Demo 3 失败恢复、Demo 4 幻觉验收、nudge 教训全是真实工程颗粒；陷阱/性能/概念检验齐 |
| 02 协议、框架与评测 | 9.0 | MCP 踩坑实录（notification）+ 三方辩论 + 榜单方法论（锁版本/脚手架口径）是面试纯金；框架仅表格级认知；耗时口径小瑕 |
| scripts 01-03 | 9.5 | 三脚本全跑通；Section 0/0.5（沙箱直测 + 假模型验证终止条件）让循环逻辑零 GPU 可测——模型无关性的自证 |
| 作业 19 题面 | 9.0 | 4+1 结构、题 3 宽口径 vs 脚本严口径的对照是有意设计；面试直通车五问 + 4 思考题带提示；本次未实现故不验正向通过 |

---

## 学习目标达成表（对照 README 六条 ✅）

| 目标 | 达成 | 证据（我实测/自测） |
|---|---|---|
| 手写不依赖框架的 agent loop | ✅ | 白板默写通过（见下）；脚本 01 五部件对照源码确认 |
| 实现两种格式解析器 + 兜底 | ✅ | `<tool_call>` content 格式 vs API `tool_calls`（arguments 是字符串要二次解析）；兜底返回 `[]` 的理由能讲（最小化影响半径） |
| 复刻 mini-MCP 三方法 + 说清三层分工 | ✅（带口径警示） | 脚本 02 四步全跑通；协议三层能画；client/server 角色划分需自我补课后也能答 |
| 实测 τ-bench 式评测 + 解读 pass^k 与榜单 | ✅ | 我复现 pass^1=0.00（与教程一次运行一致）；pass^k 指数放大能推 |
| 复述三方辩论并落到"上下文隔离" | ✅ | Anthropic 15×/90.2%、Cognition 两定律、LangChain 调和——结论一句话能讲 |
| 说清中断/重试/上下文管理三件事（roadmap A2 目标） | ✅ | nudge（全报错才推）、裁剪（保 system+user0+尾部 12 条、防孤儿 tool 消息）、循环检测（同参×3 vs 同工具×3 口径差异） |

---

## 白板默写自测（不查资料，写后对照）

**1. agent loop 三终止条件 —— 能写 ✅**
① 无 tool_calls：正常结束（答案或放弃），防"永远循环烧钱"；② max_turns：防"永远再查一下"（脚本默认 8，假模型验证时显式传 4）；③ 循环检测：同工具**同参数**连续 ×3，防复读机（宽口径"同工具×3"是作业题 3，会误伤连续读三个不同文件的正常探索）。补生产第四类：预算上限（token/钱/墙钟）——"模型不知道自己该停"的三类 + 钱袋。

**2. function calling 消息往返完整序列 —— 能写 ✅**
`[system, user]` → ① `apply_chat_template(messages, tools=TOOL_SPECS)`（工具 schema 注入 system 的 `<tools>` 段）→ ② `model.generate` 出原始文本 → ③ `parse_tool_calls` 解析 `<tool_call>{...}</tool_call>`（可多个，按序；失败返回 `[]` 不抛异常）→ ④ `messages.append(assistant{content:"", tool_calls:[…]})`（**content 留空**，否则模板双重渲染）→ ⑤ 本地 `execute_tool(name, args)` → ⑥ `messages.append(tool{name, content:结果})`（Qwen 模板渲染成 user 角色的 `<tool_response>` 块）→ ⑦ 回到 ①。无调用轮：assistant 文本即 final answer。API 格式差异：`finish_reason="tool_calls"`、`arguments` 是 JSON 字符串需 `json.loads` 二次解析。

**3. MCP 三层各管什么 —— 半通过 ⚠️（如实记）**
- 教程口径（协议三层）：**MCP = agent↔工具**（JSON-RPC 2.0：initialize 握手 / tools/list / tools/call；管"手"）；**A2A = agent↔agent**（Agent Card 能力名片 + JSON-RPC/SSE；管"嘴"）；**AGENTS.md = agent↔代码库**（静态 Markdown 规矩；管"地图"）。三层不互替。→ 能默写 ✅
- 任务口径（MCP 自身角色）：**协议** = 消息格式与规则（带 id 请求/响应、notification 无 id 且**不得回应**、工具级错误走 `isError:true` vs 协议级走 `error` 对象如 -32601）；**客户端** = 拉起 server（subprocess/stdio）→ 握手 → tools/list 发现 → 把 schema 喂给模型（模型无感）→ 代发 tools/call；**服务器** = 持有工具实现与 dispatch，返回 `content` 数组 + `isError`，对 notification 静默。→ 第一反应是教程口径，此口径需回忆后才能答全——**面试验_item：两种问法都要接得住**。

**4. pass^1 与 pass^k —— 能写 ✅**
pass^1 = 单次通过率 = 通过次数/总次数（能力与运气混合；空样本返回 None 不是 0）；pass^k = 同一任务独立跑 k 次、**k 次全过才算过**（度量一致性/可靠性）。独立假设下 pass^k = (pass^1)^k，指数放大不稳定：pass^1=0.7 → pass^8≈0.057。pass^1 高 pass^8 低 = 能力强但不稳定；生产要"每次都对"。τ-mini 我复现：T1/T2/T3 各 3 次、temperature=0.7，全 False → OVERALL pass^1=0.00。

---

## 硬数字审计表

| 数字 | 出处 | 口径清楚? | 面试可辩护? |
|---|---|---|---|
| 脚本 01 全程 18.4s（README ~20s） | 01 章 §3 头 | ✅ 标明 4090/fp16/贪心/11 次生成+1 次 10s 超时演示 | ✅ 环境+构成齐全 |
| 脚本 03 耗时 | README/02 章 ~15-25s vs 脚本头注释 ~12-25s | ⚠️ 两处不一致 | ⚠️ 引用前先统一口径 |
| 模型加载 ~8s / 每轮 ~1-2s / CPU 慢 20-40 倍 / 0.5B ~1GB 显存 | 01 章 §4.2 | ✅ 环境限定（4090 fp16） | ✅ |
| 白名单 {ls,cat,grep,python} / 超时 10s / 截断 400 字符 / MAX_TURNS=8 / NUDGE_BUDGET=2 / KEEP_TAIL=12 / 复读 ×3 | 脚本 01 常量（我 grep 核对） | ✅ 代码即口径 | ✅ |
| 裁剪 22→15 条；假模型 max_turns=4 | 01 章 §3.5 | ✅（注意 4 是显式传参，默认 8） | ✅ |
| MCP protocolVersion=2024-11-05 / error -32601 / add(23,47)=70 | 脚本 02（我复现一致） | ✅ 版本号是示例常量（教程注明真实 server 会协商） | ✅ |
| **Anthropic 多 agent：token ~15×**（对照单 agent 4×）、**高 90.2%**（Opus lead + Sonnet subagents vs 单 Opus）、lead 并行 3-5 个 subagent | 02 章 §4 | ✅ 标明工程博客原文口径，90.2% 附"依赖 orchestrator 提示工程"的限定 | ✅ 引用时带"Anthropic 内部研究评测口径" |
| SWE-bench Verified 500 题；Opus 4.5 **80.9%** 首个破 80%（Claude Code 脚手架口径，截至 2026-09） | 02 章 §5.2 | ✅ 脚手架口径 + 时效免责双带 | ✅ 但"首个破 80%"为转述，离线未核实 |
| **τ-mini pass^1=0.00**（3 任务 × 3 次全败）；另一次运行 OVERALL=0.11（T3=1/3，1/9≈0.111） | 02 章 §5.1（我复现 =0.00，吻合） | ✅ 教程明示 9 样本置信区间无统计意义，报数必附次数+温度 | ✅ **把方差当教学内容讲反而加分** |
| temperature=0.7 top_p=0.9 / 每任务 R=3 / 政策 ~300 字 | 脚本 03（我 grep 核对） | ✅ | ✅ |
| pass^1=0.7 → pass^8≈0.057 | 02 章 Q3 | ✅ 独立假设下 | ✅ |
| 论文号：ReAct 2210.03629 / τ²-bench 2506.07982 / GiGPO 2505.10978 / AgentRL 2510.04206 | README/02 章 | ✅ | ✅ |
| ragflow 89.6k★ / llama_index 51.9k★ | roadmap 549 行 | ⚠️ **属 A1/Part 18 承诺，不属 A2**（P19 全文 grep 无此二词，见下） | 归 A1 审计 |
| roadmap A2：≈8-15h / 脚本 01-03 / 作业 19（4+1）/ 链 6 后半段 | roadmap 555-564 行 | ✅ | ✅ |

---

## Roadmap 应用线 A2 逐条对照

| roadmap A2 承诺 | 兑现? | 证据 |
|---|---|---|
| 手写 agent loop（模型+工具表+while 循环+三终止条件+沙箱白名单） | ✅ | 脚本 01 + 01 章；沙箱=白名单+超时+截断三道闸 |
| mini-MCP（JSON-RPC 2.0 三方法） | ✅ | 脚本 02：initialize/tools/list/tools/call 全实现，另附协议级错误与线上字节流 |
| τ-bench 微缩（政策+用户模拟器+pass^1） | ✅ | 脚本 03：政策 prompt + 脚本化用户剧本 + verify() 三查；pass^1 实测（卡点 5 的取舍已标注） |
| 框架认知（LangGraph / OpenAI Agents SDK / smolagents / 极简派 Pi） | ✅（概念级） | 02 章 §3 四流派表（心智模型/适合/不适合）+ Q4 升级信号；无实操（卡点 3） |
| 目标：能手写最小 function calling 循环并说清中断/重试/上下文管理三件事 | ✅ | 01 章 §2.4 三小节逐一对应；白板自测通过 |
| 目标：能评价各框架的抽象收益与成本 | ✅ | "框架的抽象是有税的（调试路径变长、可控性下降）"+ 选型直觉（先裸 loop 后升级） |
| 验证：跑通脚本 01-03（含一次失败恢复轨迹） | ✅ | Demo 3 报错→nudge×2→自纠，我亲手复现成功（final 含 1081：True） |
| 验证：作业 19（4+1） | ✅（题面侧） | 5 题结构与分值表齐；模板未实现（卡点 6） |
| 验证：链 6 后半段自问自答 | ✅ | 作业"面试直通车"五问即链 6 后半段的落地 |
| **ragflow / llama_index** | **不属 A2** | roadmap 549 行明确写在 **A1·Part 18**（"对比 ragflow 平台化能力与 llama_index 抽象"）；P19 未承诺也未出现——**归属澄清，不构成 A2 缺口** |

---

## 只改 3 件事

1. **README「📦 环境」加一行离线跑法**：`离线机器：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python 01_agent_loop.py`（模型已缓存时）。我首跑 ProxyError 崩掉四个 Demo，这一行能救所有离线学习者。
2. **统一脚本 03 耗时口径**：脚本头注释 ~12-25s vs README/02 章 ~15-25s，改一致（面试引用数字最怕两处口径）。
3. **02 章 §1 加一个口径澄清框**：「面试官问 'MCP 三层'有两种含义——协议三层（MCP/A2A/AGENTS.md）与 MCP 自身角色（协议/客户端/服务器）——各自答案如 §1 表与 §2.2」。这是我自己被问住的真实坑。

## 最喜欢 3 处

1. **Demo 4 幻觉验收**：`磁盘实况 /tmp/agent_demo2.txt = '(missing)'；final_answer 声称含 1081：True` 一行把"为什么 τ-bench 用 DB 终态判分"从教条变成目击——合规评测的是行为不是话术。
2. **02 章 §2.2③ notification 踩坑实录**："多写一条响应与少写一条请求同样是致命错误——对端没有任何办法把多余的响应归位"。这是只有真写过协议栈才写得出的句子。
3. **τ-mini pass^1=0.00 不藏丑**：把 0.00 与 0.11 两次运行的方差、9 样本置信区间无统计意义全部摆上台面——"0.5B 的失败是教材"不是口号，是教学设计。

---

## 面试资产清单（S3 冲刺复述卡）

- **三句话讲 agent**：while 循环 + 工具结果回填上下文 + 三终止条件；其余全是工程加固（解析兜底/沙箱/裁剪）。
- **0.5B 诚实成绩单**：单步工具调用合格（Demo 1/2），多步规划+失败自纠+政策遵循不可用（Demo 4 / τ-mini pass^1=0.00 我亲手复现）——这个差距正是 agentic RL（Part 17）要弥合的。
- **幻觉验收故事**：模型嘴上 "computation was successful"，磁盘无文件 → 环境终态验收 → τ-bench DB 判分 / SWE-bench 测试判分同源。
- **nudge 的教训**：无条件 nudge 会把已完成的模型拽去瞎折腾（实测踩坑）→ "全报错才推一把"是保守可用的启发式；"判断任务做没做完"本身是开放问题。
- **错误信息是写给模型看的 prompt**："file not found" vs "use X instead" 决定弱模型自纠还是道歉。
- **MCP 一分钟**：把 01 章"schema+执行器"层标准化；tools/list 之于 TOOL_SPECS、tools/call 之于 execute_tool，loop 一行不改；两级错误分开（isError vs -32601）；notification 不得回应。
- **多智能体三段论**：Anthropic 15× token / 90.2% 买的是什么 → 上下文隔离；Cognition 两定律（传递有损/行动改变结果）；判据是任务"广度型可自包含"，不是"更多 agent=更多智能"。
- **pass^k 指数放大**：0.7^8≈0.057；小样本报数三件套 = 锁 commit + 锁模型版本（含采样参数）+ 报运行次数；分数必须连脚手架口径一起读。
- **轨迹双视角**（与 P17 互指）：推理侧拼接上下文，训练侧切割 loss（观测 mask=0 防复读观测）。
- **实测留痕**：我的复现日志在 `/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S3_P19/`（run_01_offline.log 逐字吻合 / run_02.log / run_03_offline.log pass^1=0.00）。

## 待办（S3 后续）

- [ ] 实现 `assignments/assignment_19/agent_exercises.py` 题 1-4（本次未做，pytest 当前 4 failed + 1 skipped）
- [ ] 把"MCP 两种三层口径"练成条件反射；补读 A2A Agent Card 一个真实示例
- [ ] 自选：跑作业实验题（NUDGE_BUDGET=0 / WHITELIST+echo / R=6 方差）攒自己的数字
