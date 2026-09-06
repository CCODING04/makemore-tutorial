# Assignment 19：Agent 与 Function Calling

> 对应 Part 19 教程（[01 手写 Agent Loop](../../courses/Part19_agents/tutorial/01_agent_loop.md) /
> [02 协议、框架与评测生态](../../courses/Part19_agents/tutorial/02_protocols_and_frameworks.md)）。
> 五题全部纯 CPU、零模型依赖；题 5 为 🌟 弹性题（不实现返回 None，测试自动 SKIP ⏭️）。
> 与课程脚本同名同签名的接口：`parse_tool_calls` / `TOOL_SPECS`（见
> [课程脚本 01](../../courses/Part19_agents/scripts/01_agent_loop.py)——作业不加载模型，
> 只实现它的"可单测"部件）。

## 📊 分值表

| 题号 | 主题 | 分值 | 对应测试 |
|------|------|------|----------|
| 1 | tool_spec（函数 → JSON schema） | 25 | `test_ex1_tool_spec` |
| 2 | parse_tool_calls（四类输入解析） | 25 | `test_ex2_parse_tool_calls` |
| 3 | should_stop（终止状态机） | 20 | `test_ex3_should_stop` |
| 4 | pass_at_1（一次通过率） | 15 | `test_ex4_pass_at_1` |
| 🌟 5 | mini_mcp_call（JSON-RPC 握手 mock，Stretch） | 15（总分封顶 100） | `test_ex5_mini_mcp_call`（未实现自动 SKIP ⏭️） |

## 题目（实现 `agent_exercises.py`）

### 题 1：`tool_spec`（25 分）——函数 → OpenAI tools JSON schema

给定一个 Python 函数（用 `inspect` 读签名、docstring 首行做 description）或一个
描述 dict（`{"name","description","params": {参数名: {"type","description"}}}`），
生成一个合法的 OpenAI tools schema 元素：
`{"type": "function", "function": {"name", "description", "parameters": {...}}}`。
参数无默认值 → required；有默认值 → 可选。

**验收标准：**
- [ ] 函数输入：`name` 取函数名、非空；docstring 首行为 `description`（无 docstring
      时为非空占位串）
- [ ] `required` 恰好包含**无默认值**的参数（顺序不限），有默认值的参数不进 required
- [ ] 每个参数的 `type` ∈ {string, integer, number, boolean, array, object}；
      描述 dict 输入非法 type 时回退为 "string"
- [ ] 返回结构可直接塞进 `TOOL_SPECS` 列表（与课程脚本 01 的 calculator 条目同构）

### 题 2：`parse_tool_calls`（25 分）——解析模型输出（与脚本 01 同名同语义）

从文本中解析全部 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
（可能多个，按出现顺序），返回 `list[dict]`。**解析失败绝不抛异常**：
畸形 JSON（截断/尾逗号可修则修）与无调用的文本都返回 `[]`。

**验收标准：**
- [ ] 合法单调用：`<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>`
      → `[{"name": "calculator", "arguments": {"expression": "1+1"}}]`
- [ ] 合法多调用：一条文本两个 `<tool_call>` 块 → 按序解析出 2 个
- [ ] 畸形 JSON：`<tool_call>{"name": "calculator", "arguments": {"expr` （截断）
      → `[]`；尾逗号 `{"name":"a","arguments":{"x":1,}}` → 能修复解析出来
- [ ] 无调用：普通文本 `"The answer is 42."` → `[]`（不是 None）

### 题 3：`should_stop`（20 分）——终止状态机

输入 state 字典：`{"turns": int, "max_turns": int, "last_has_calls": bool,
"tool_history": [str, ...]}`（`tool_history` 是按序的工具名列表）。返回
`(stop: bool, reason: str | None)`。判定优先级：**max_turns > loop_detected >
no_tool_calls**：

- `turns >= max_turns` → `(True, "max_turns")`
- `tool_history` 末尾 3 个为同一工具 → `(True, "loop_detected")`
- `last_has_calls` 为 False（模型不再调工具）→ `(True, "no_tool_calls")`
- 其余 → `(False, None)`

**验收标准：**
- [ ] `{"turns": 4, "max_turns": 4, ...}` → `(True, "max_turns")`
- [ ] `tool_history=["file_read","file_read","file_read"]` → `(True, "loop_detected")`
- [ ] `tool_history=["file_read","bash","file_read"]`（同工具但没连着 3 次）→ 不算循环
- [ ] `last_has_calls=False` 且其余条件不触发 → `(True, "no_tool_calls")`
- [ ] 正常进行中 → `(False, None)`

### 题 4：`pass_at_1`（15 分）——一次通过率（与脚本 03 同名同语义）

输入布尔列表（每次独立运行的 pass 与否），返回 pass^1 = 通过次数 / 总次数；
**空列表返回 None**（无样本时估计量无定义）。

**验收标准：**
- [ ] `[True, True, True, True]` → `1.0`；`[False]` → `0.0`
- [ ] `[True, True, False]` → `2/3`（容差 1e-9）
- [ ] `[]` → `None`（不是 0.0——区分"全挂"与"没跑"）
- [ ] 数学性质：结果恒等于 `sum(runs)/len(runs)`；把 N 次运行拆成两组，两组均值
      的加权平均 == 整体 pass^1（脚本 03 的逐任务/总体汇总用的就是这条性质）

### 题 5：🌟 `mini_mcp_call`（15 分，弹性题）——JSON-RPC 三步握手（mock transport）

给定一个 mock transport（提供 `send(dict) -> dict` 发请求收响应、
`notify(dict) -> None` 发通知），完成 MCP 三步：① 发 `initialize` 请求（带 id）→
② 发 `notifications/initialized`（**无 id** 的通知）→ ③ 发真正的 `method` 调用
（新的 id）。返回第 ③ 步的 `result`；若响应带 `error` 返回 `None`。

**验收标准：**
- [ ] transport 收到恰好 2 条请求（id 互不相同）+ 1 条通知（无 id 字段）
- [ ] 返回第 ③ 步响应的 `result` 字段（dict）
- [ ] 第 ③ 步响应含 `error` 时返回 `None`（不抛异常）
- [ ] 未实现时保持 `return None` —— 测试将 SKIP ⏭️ 而非 ERROR

## 实验题（观测型）

- 跑[脚本 01](../../courses/Part19_agents/scripts/01_agent_loop.py)：把 `NUDGE_BUDGET`
  从 2 改成 0，观察 Demo 3（失败恢复）还能不能自纠——nudge 在哪一步起的作用？
- 把 `WHITELIST` 里加上 `echo`，重跑 Demo 4：模型的 `echo "$(cat ...)"` 链路会
  走到哪一步？对照教程 01 章"白名单为什么必须"。
- 跑[脚本 03](../../courses/Part19_agents/scripts/03_tau_mini.py) 两次，对比两次的
  pass^1 汇总——小样本评测的方差有多大？把 `R` 改成 6 再看。

## 🎯 面试直通车

- "agent loop 的核心是什么？"——while 循环 + 工具结果回填上下文 + 三种终止条件
  （无调用/轮数上限/循环检测）；剩下的全是工程加固（解析兜底、沙箱、裁剪）
- "为什么 bash 工具要白名单+超时？"——模型输出是不可信输入；白名单防破坏性
  命令、超时防挂死与资源占用、截断防上下文爆炸（Echo Trap 的防御面）
- "MCP 是什么？和 A2A 什么关系？"——MCP=agent↔工具（JSON-RPC：initialize/
  tools/list/tools/call），A2A=agent↔agent（Agent Card 发现），AGENTS.md=
  agent↔代码库；三层不互替
- "subagent 的收益来自数量还是上下文隔离？"——来自上下文隔离（Anthropic 15×
  token 买的是隔离；Cognition 指出有损传递；任务可自包含才值得分兵）
- "怎么评 agent？"——DB/环境终态判分 + 政策合规 + pass^1/pass^k；榜单必须锁
  基准版本、看脚手架口径（τ-bench 评分 bug 的社区教训）

## 🤔 思考题

**Q1：** subagent 的收益来自"数量"还是"上下文隔离"？一个任务拆给 3 个各带
干净上下文的 subagent，什么情况下优于 1 个带全部上下文的 agent？什么情况下相反？

<details>
<summary>💡 提示</summary>

来自**上下文隔离**，不是数量。隔离是净化时（子任务自包含、互不依赖、各自产出
只需汇总结论——如并行检索），多 agent 优于单 agent 膨胀上下文（Anthropic 深度
研究系统，代价 ~15× token）；隔离是有损压缩时（子任务要反复回问主上下文、
世界状态会被并行行动改变——Cognition 两条定律），分兵放大损耗，不如给单个
agent 做好上下文管理（压缩历史而非分兵）。判据是任务的"可分解性"，不是
"更多 agent = 更多智能"。见教程 02 章第 4 节三方辩论。

</details>

**Q2：** 题 3 的 `should_stop` 用了"同工具连续 3 次"的宽口径，课程脚本 01 用
"同工具**同参数**连续 3 次"的严口径。各举一个被对方口径误伤/漏放的场景。

<details>
<summary>💡 提示</summary>

宽口径（只看工具名）误伤：agent 连续读 3 个**不同**文件（file_read×3，路径各
不同）是正常探索，会被误停——设想"依次核对 a/b/c 三个配置文件再汇总"的任务，
第 3 个 file_read 会被拦在半路（教程 01 章 Demo 3 只有错→对两笔 file_read，
宽口径拦不到它，恰是反例）。严口径（同名同参）漏放：参数里有微小变化的复读（比如
每次重试多一个空格）永远拦不住；以及"轮流调两个工具"的乒乓循环
（A→B→A→B）两种口径都拦不住——真正稳健的循环检测要看"状态是否在演进"
（教程 01 章 2.4 节③）。

</details>

**Q3：** 题 2 的兜底策略是"畸形输出返回 []"。如果把 `[]` 的语义改成"抛异常让
上层重试整个任务"，agent 系统的行为会差在哪？

<details>
<summary>💡 提示</summary>

返回 []：把"这轮没调工具"交给循环逻辑处理——可以是最终答案、也可以 nudge
一次给模型自纠机会，**已完成的工具副作用都保留**（文件已写、订单已查）。抛
异常重跑整个任务：① 之前轮的副作用全部重放（文件写两遍、退款发两次——τ-mini
里就是真金白银的事故）；② 成本翻倍；③ 若畸形输出是确定性的（贪心解码），
重跑会得到同样的畸形——无限重试。原则：**解析层的失败应该最小化影响半径**，
决策上移给 loop（教程 01 章陷阱 4）。

</details>

**Q4：** τ-mini 的 T2/T3（违规请求须拒绝）里，agent"什么工具都不调、只反复反问
用户"也能拿 pass。这个判分漏洞怎么堵？

<details>
<summary>💡 提示</summary>

现状：拒绝类任务的 pass 只要求"没做违规操作"（真空也满足）——按兵不动即可
通过（脚本 03 的 T3 有一次正是"反复反问用户要信息直到剧本用完"拿的 pass，
属于不作为通过）。堵法：
① 判分加**交互质量**维度（是否明确告知用户"政策不允许"——关键词/裁判模型
检查 final 回复）；② 加"用户追问后仍守住"的剧本轮（T2 已这么做：用户二次
施压）；③ 任务集配比平衡：拒绝类任务的"假阴性"（漏放）与执行类任务的
"假阳性"（误判完成）对冲。本质：单一终态判分覆盖不了"沟通合规"，τ-bench
真实实现同样在判分器里混用终态+对话检查（教程 02 章 5.1）。

</details>

## ✅ 完成自检

```bash
python test_agent_exercises.py   # 或 pytest test_agent_exercises.py
```

- [ ] 题 1-4 全绿（✅）
- [ ] 题 5 实现则 ✅、未实现则 ⏭️ SKIP（均不算失败）
- [ ] 能不看笔记回答"面试直通车"五问
