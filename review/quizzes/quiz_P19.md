# 测验 · Part 19（Agent 与 Function Calling：Agent Loop、MCP 协议与 τ-bench 评测）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** 不依赖框架的 agent loop 由哪五个部件组成？三种终止条件各防哪一类"模型不知道自己该停"？

- **答案**：每轮五步：① apply_chat_template(messages, tools=TOOL_SPECS) 把工具 JSON schema 注入 system prompt → ② model.generate 生成 → ③ parse_tool_calls 解析（无调用 = 最终答案，循环结束）→ ④ execute_tool 本地执行（计算器/文件/白名单 bash）→ ⑤ messages 追加 assistant(tool_calls) 与 tool(result) 回填，回到①。三种终止条件：无 tool_calls（正常结束：答案或放弃）、max_turns（防"永远再查一下"）、循环检测（同工具同参数连续 3 次判复读机强制停止）。分别对应三类失败：不知道做完了、不知道做不完了、不知道自己在重复；生产还要加第四类——预算上限（token/钱/墙钟时间）。agent 的失败模式一半是"不会做事"，另一半是"停不下来"。
- **锚点**：教程 01_agent_loop.md §1 问题引入 + §2.5 终止条件：为什么恰好是这三个

**Q2（诊断）** 模型输出了一段截断的 JSON，解析器抛 json.JSONDecodeError 把整个 loop 炸掉。为什么兜底策略必须是"返回 []"而不是抛异常、或捕获后重跑整个任务？

- **答案**：0.5B 的畸形输出（截断 JSON、尾逗号、```json 围栏包裹）是常态而非异常。返回 [] 的语义是"这轮没调工具"，循环已有处理路径（当最终答案结束，或注入 nudge 给模型自纠机会）——已完成的工具副作用全部保留。抛异常则整个 loop 崩溃，一次畸形输出毁掉整段任务；重跑整个任务更糟：① 之前轮的副作用重放（文件写两遍、退款发两次）；② 成本翻倍；③ 贪心解码下畸形输出是确定性的，重跑得到同样畸形 → 无限重试。实现上宽容解析：剥 markdown 围栏 → 修尾逗号 → 全失败返回 []。原则：解析层的失败要最小化影响半径，决策上移给 loop。
- **锚点**：教程 01_agent_loop.md §2.2 解析层：parse_tool_calls + §4.1 陷阱 4；Assignment 19 思考题 Q3

**Q3（对比）** MCP / A2A / AGENTS.md 三层协议各连接谁、用什么机制？mini-MCP 复刻暴露的两条协议纪律是什么？

- **答案**：MCP = agent↔工具/数据源（JSON-RPC 2.0，initialize 握手 + tools/list + tools/call——管"手"）；A2A = agent↔agent（Agent Card 能力名片发现 + JSON-RPC/SSE 传输——管"嘴"）；AGENTS.md = agent↔代码库（仓库根目录静态 Markdown 说明：构建/测试命令、禁区——管"地图"）。三层不互替，生产系统可同时在场；01 章工具层（TOOL_SPECS / execute_tool）与 MCP 的 tools/list / tools/call 同构，换成连 MCP server 后 loop 一行不改。两条纪律：① notification（无 id 消息）不得回应——mini-MCP 第一版给 notifications/initialized 回了 error，client 把多余响应读成下一条结果，两条消息读串、协议栈错位（修复：`if "id" not in req: continue`）；② 工具级错误走 isError:true（协议正常、业务失败，client 可换参数重试），协议级错误走 JSON-RPC error 对象（code -32601 方法不存在），混用会让重试逻辑失效。
- **锚点**：教程 02_protocols_and_frameworks.md §1 协议三层分工表 + §2.2 逐块讲解（对照真实规范的四个要点）

**Q4（数字）** τ-mini 实测 Qwen2.5-0.5B-Instruct 的 pass^1 是多少？为什么判分必须看 DB 终态而不是让裁判读对话？pass^1 与 pass^k 各度量什么？

- **答案**：三个任务（T1 合规退款 / T2 已发货改址须拒绝 / T3 超期退款须拒绝）runs 全为 [False, False, False]，OVERALL pass^1 = 0.00（诚实登记；另一次完整运行得 T3=1/3、OVERALL 0.11——9 个样本的置信区间宽到没有统计意义，报数字必须附运行次数与温度）。必须看终态：T2 里模型嘴上说"很抱歉"手上却已 update_address 落库——话术与行为脱钩，LLM 裁判读对话会被话术骗；DB 终态客观、可复现（判分器是确定性代码），政策合规（"不该做的没做"）天然落在调用序列与终态上。pass^1 是单次通过率（能力与运气混合），pass^k 是 k 次独立全过（可靠性/一致性）：pass^1=0.7 时若各次独立，$0.7^8\approx 0.057$——不稳定被指数级放大；生产要"每次都对"，不是"偶尔对"。
- **锚点**：教程 02_protocols_and_frameworks.md §5.1 τ-bench 家族 + τ-mini 实测

**Q5（对比）** 复述多智能体三方辩论（Anthropic / Cognition / LangChain）的立场与关键数字，并给出本课结论。

- **答案**：正方 Anthropic（编排者-执行者）：lead agent 分解任务、一次并行拉起 3-5 个 subagent 检索再汇总；多 agent 系统 token 消耗约为普通对话的 15×（单 agent 约 4×），换来内部研究评测上比单 agent Opus 高 90.2%（依赖有效的 orchestrator 提示工程）。反方 Cognition《Don't Build Multi-Agents》两条定律：① 上下文传递有损——subagent 拿到的是压缩过的二手上下文；② 行动会改变结果——并行 agent 基于旧世界观行动，冲突无法事后合并。调和派 LangChain：多 agent 不是"更多智能"，而是"上下文工程"的一种手段——广度型任务（可并行独立检索）值得分，深度型任务（长程依赖、世界状态会变）不分。本课结论（面试标准答案）：subagent 的收益不来自数量，来自**上下文隔离**——给子任务一个干净、自包含的上下文，把结果（而非过程）带回；子任务做不到自包含时，分兵只会放大传递损耗。
- **锚点**：教程 02_protocols_and_frameworks.md §4 多智能体：三方辩论专节

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 手写不依赖框架的 agent loop（五部件 + 三种终止条件） | Q1 | loop 每轮五步与三类"不知道该停"（沙箱边界见闪卡） |
| 实现兼容两种格式的解析器及兜底策略 | Q2 | 畸形输出常态化 → 返回 [] 的语义与"不重跑"的三条理由 |
| 复刻 mini-MCP 并说出 MCP/A2A/AGENTS.md 三层分工 | Q3 | 三层机制与 mini-MCP 两条协议纪律（notification、两级错误） |
| 实测 τ-bench 式评测并解读 pass^1/pass^k 与榜单 | Q4 | pass^1=0.00 诚实登记与运行间方差；DB 终态判分理由（SWE-bench 脚手架口径与第三方榜单污染见闪卡） |
| 复述多智能体三方辩论并落到"上下文隔离才是收益来源" | Q5 | 15× token / +90.2% / Cognition 两定律 / 广度型 vs 深度型 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| agent loop 与 Part 17 训练循环的关系 | 同一条轨迹 [user | tool_call | observation | answer]；推理侧只管执行（拼回上下文），训练侧管梯度（loss mask、轨迹级奖励）——"推理侧拼接上下文，训练侧切割 loss" |
| bash 工具三道闸 | 白名单 {ls, cat, grep, python}（只看第一词，防 rm -rf 式破坏）+ 超时 10s（防挂死/资源占用）+ 输出截断 400 字符（防上下文爆炸）；模型输出是不可信输入 |
| 两种工具调用格式 | Qwen（本地 HF generate）：调用写在 content 的 <tool_call> 块里；OpenAI 兼容 API：finish_reason="tool_calls"，调用在 message.tool_calls，arguments 仍是 JSON 字符串要二次解析 |
| 回填 messages 的两个坑 | assistant 消息 content 留空（调用只放结构化 tool_calls 字段，否则模板渲染两遍、模型以为调了两次）；tool role 由 Qwen 模板渲染成 user 角色的 <tool_response> 块 |
| nudge 的实测教训 | 只在【所有工具结果都是错误】时注入——无条件 nudge 会把已给出正确答案的模型拽回去没事找事（调 echo 被拒 → 道歉循环） |
| 循环检测的口径 | 课程脚本：同工具**同参数** ×3（宽口径"同工具 ×3"会误杀"依次读 3 个不同文件"的正常探索；但参数微变的复读与 A/B 乒乓循环两种口径都拦不住） |
| Demo 4 的三个失败模式 | ① 并行调用的依赖错误（第 2/3 个调用依赖第 1 个的结果 → 占位符照抄任务文本；有依赖必须分轮）；② 绕开专用工具（有 calculator 却用 bash+echo 算术）；③ 幻觉验收（文件没写成功却声称完成）→ 必须环境终态验收 |
| 0.5B 诚实成绩单 | 单步工具调用合格（Demo 1/2 干净利落）；失败集中在多步规划、失败自纠、自我校验；全脚本 18.4s（RTX 4090，贪心解码可复现） |
| 错误信息是写给模型的 prompt | "file not found ... use X instead" 带修复提示 + harness nudge 两层配合，才把 0.5B 从"道歉反问用户"拉到稳定自纠（Demo 3） |
| 上下文裁剪 | 保 system + 第一条 user + 最近 12 条，砍中间老观测；裁剪后首条不能是孤儿 tool 消息（破坏模板渲染）；实测 22 → 15 条 |
| MCP 三步与消息规则 | initialize（交换协议版本与 capabilities）→ tools/list → tools/call；请求带 id 响应原样带回；无 id 的是 notification，协议禁止回应；JSON-RPC error -32601 = 方法不存在 |
| τ-mini 对 τ-bench 的三要件复刻 | 政策文档进 agent prompt（退换货政策 ~300 字）/ LLM 用户模拟 → 脚本化剧本（确定性、零方差）/ DB 终态 + 调用序列判分（不信 agent 话术） |
| agent 评测判分器 | 目标达成 ∧ 调用序列合规 ∧ DB 终态合规，三查齐全才放行——只查任务完成会奖励"嘴上拒绝、手上照办" |
| SWE-bench 榜单读法 | 只认 swebench.com 官方榜；分数必须连同脚手架（scaffold）口径一起读（同模型换 scaffold 差十几分是常态）；截至 2026-09 Claude Opus 4.5 以 80.9% 居首（Claude Code 口径，首个破 80%） |
| 复现 agent 榜单的正确姿势 | 锁基准仓库 commit hash + 锁模型版本（含采样参数）+ 报运行次数；τ-bench 上游修过评分 bug、榜单随之变化——评测基准也是代码 |
| 框架四流派 | LangGraph 状态机派（人审中断/检查点/复杂分支）、OpenAI Agents SDK 派（handoff + guardrails）、smolagents 代码即动作派（模型写 Python 当动作）、Pi 极简派（loop + 个位数打磨好的工具）；无升级信号就别升级——框架抽象有税 |
| τ-mini → agentic RL | 奖励 = verify() 通过布尔值、rollout = run_task、训练循环 = Part 17 轨迹级 GRPO——"用任务评测当奖励"的最小闭环 |
