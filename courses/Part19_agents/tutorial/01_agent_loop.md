# 01 — 手写 Agent Loop：while 循环就是 Agent 的全部骨架

> 🧭 [Part 17](../../Part17_agentic_rl/tutorial/README.md) 把"多轮工具调用轨迹"当作
> **训练数据**整条采出来做 RL；本章是同一条轨迹的**推理侧**——不训练、只执行：
> 模型发工具调用 → 本地执行 → 结果回填上下文 → 继续生成，一个 `while` 循环转到底。
> 跑 [scripts/01_agent_loop.py](../scripts/01_agent_loop.py)（GPU 实测 ~20 秒），
> 0.5B 模型的成功与翻车都是教材。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** 一个不依赖任何框架的 agent loop（渲染→生成→解析→执行→回填）
- ✅ **实现** `parse_tool_calls`：兼容 Qwen `<tool_call>` content 格式与 API
  `tool_calls` 结构化格式，并设计畸形输出的兜底策略
- ✅ **说出** agent 的三种终止条件与各自的必要性（无调用/轮数上限/循环检测）
- ✅ **配置** bash 工具的最小安全边界（超时 + 命令白名单）并解释为什么必须
- ✅ **解剖** 0.5B 模型的真实失败轨迹（并行调用占位符、幻觉验收、报错后放弃）

## 📖 前置知识

**必须掌握：**
- **Part 8 02 章**：chat template 与 SFT（工具调用协议就是 chat template 的扩展——
  `apply_chat_template(messages, tools=...)` 把工具 JSON schema 注入 system prompt）
- **Part 17 01 章**：多轮轨迹格式 `[user | tool_call | observation | ... | answer]`
  （本章轨迹 = 它的训练数据格式，一对一对应）

**建议掌握：**
- **Part 12**：微调实战（见过 Qwen 系 chat template 的 `<|im_start|>` 结构更佳）

**可选：**
- **Part 14**：vLLM 部署（把本章的 HF `generate` 换成 vLLM/OpenAI 兼容接口时需要）

## 1. 问题引入：为什么需要 Agent Loop？

Part 8 的 chat 模型只能"一问一答"：你问"23*47 是多少"，模型要么自己算（0.5B 大概率
算错），要么说"我不能执行代码"。痛点有两个：

1. **能力缺口**：模型的算术/文件/-shell 操作能力远不如一个解释器或内核；
2. **信息缺口**：模型不知道你的磁盘上有什么、订单库里有什么——这些信息在环境里，
   不在参数里。

Agent loop 的解法朴素到令人发指：**把工具的说明书（JSON schema）塞进 prompt，
模型每轮要么"说话"要么"填一张工具调用单"，harness 执行后把结果贴回对话，循环**。

> 💡 **类比**：agent loop 就像给一位很聪明但被关在隔音房里的专家一部电话和一本
> 黄页（工具列表）。专家说"帮我查一下 X"（tool_call），你查完把结果念给他
> （tool role 回填），他再决定下一步。全部智能在"决定下一步"里，其余是电话线路。

```
        ┌────────────────────────── agent loop（每轮）──────────────────────────┐
        │                                                                      │
 user ──┤  ① apply_chat_template(messages, tools=TOOL_SPECS)                   │
 query  │        ↓ 注入工具 schema（system prompt 里多出 <tools> 段）           │
        │  ② model.generate（贪心/采样）→ 原始文本                              │
        │        ↓                                                             │
        │  ③ parse_tool_calls(text) ──无调用──→ 最终答案，循环结束 ✅           │
        │        ↓ 有调用                                                       │
        │  ④ execute_tool(name, args)（本地：计算器/文件/白名单 bash）          │
        │        ↓                                                             │
        │  ⑤ messages += [assistant(tool_calls), tool(result)]  ──→ 回到 ①    │
        │                                                                      │
        │  终止条件：①无 tool_calls（答案/放弃）②max_turns ③同调用复读 3 次    │
        └──────────────────────────────────────────────────────────────────────┘
```

与 Part 17 的关系一句话：**训练侧关心的观测 mask、轨迹级优势，推理侧统统不需要**——
推理时我们只是"把工具结果拼回上下文"；训练时才需要决定"哪些 token 算 loss"。

## 2. 逐件拆解：五个核心部件

运行 `python 01_agent_loop.py` 对照以下讲解（输出见第 3 节实录）。

### 2.1 工具层：TOOL_SPECS（给模型看）+ EXECUTORS（给 harness 用）

```python
# TOOL_SPECS：OpenAI tools JSON schema —— apply_chat_template 会把它注入 system prompt
{"type": "function", "function": {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression with + - * / ( ). Example: '23*47'.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "Arithmetic expression, e.g. 23*47"}},
        "required": ["expression"]}}}
```

- **schema 是给模型的"菜单"**：`description` 写得越具体，模型选工具和填参数越准
  （实测：给 bash 的 description 里直接写"Allowed commands: ls, cat, grep, python"，
  0.5B 也会主要在白名单内出命令——虽然它仍会忍不住调 `echo`，见 3.4 节）。
- **执行器是 harness 的私事**：模型永远只能"下单"（name + arguments JSON），
  执行细节（超时、白名单、路径检查）对模型不可见——**安全边界必须放在执行侧**，
  因为模型输出是不可信输入。

**bash 的最小安全边界（本课教学点，不是装饰）**：

```python
WHITELIST = {"ls", "cat", "grep", "python"}   # 只看第一个词

def exec_bash(command):
    first = command.split()[0]
    if first not in WHITELIST:                          # ① 白名单
        return f"Error: command '{first}' not allowed. ..."
    r = subprocess.run(command, shell=True, capture_output=True, text=True,
                       timeout=BASH_TIMEOUT, env=env)   # ② 超时 10s
    return (r.stdout + r.stderr).strip()[:400]          # ③ 输出截断（防上下文爆炸）
```

三道闸各自防一类事故：白名单防"rm -rf /"式破坏性命令；超时防"sleep 9999"式资源
占用（也防 agent 自己卡死在长任务里）；截断防一条 `ls -R /` 把上下文吃光。

> ⚠️ **Echo Trap 回顾（Part 17 02 章的攻击面，这里是防御面）**：Part 17 讲过奖励
> 可被"调一个回显 prompt 的工具"hack；推理侧同理——**任何接受模型生成的字符串并
> 交给解释器/内核的工具都是注入入口**。白名单不是可选项：生产 agent 的事故复盘里
> "模型自己 curl 了一段恶意内容并执行"是真实类型。更严格的方案还有容器沙箱、
> 只读文件系统、命令审计日志——白名单只是第一道门。

### 2.2 解析层：parse_tool_calls（两种格式 + 兜底）

本地 HF `generate` 没有 `finish_reason`——Qwen2.5 的工具调用就写在 **content** 里：

```
<tool_call>
{"name": "calculator", "arguments": {"expression": "23*47"}}
</tool_call>
```

而 OpenAI 兼容 API 的返回是结构化的：`finish_reason="tool_calls"`，调用在
`message.tool_calls` 字段（`arguments` 还是 JSON **字符串**，要二次解析）。解析层
两种都要认识，换成 vLLM 服务时循环代码一行不改：

```python
def parse_tool_calls(text):
    # 格式②：API 结构化 message（本地推理不出现，vLLM/OpenAI 客户端会出现）
    if isinstance(text, dict) and text.get("tool_calls"):
        return [{"name": tc["function"]["name"],
                 "arguments": json.loads(tc["function"]["arguments"])}   # 字符串→dict
                for tc in text["tool_calls"]]
    # 格式①：<tool_call>{...}</tool_call>（finditer：一条消息多个调用全部按序解析）
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        ...  # _try_json：先原样 json.loads，再修尾逗号（0.5B 常见畸形）
    # 兜底：剥 markdown 围栏 → 裸 JSON → 全失败返回 []（绝不抛异常）
```

**兜底策略为什么是"返回 [] 而不是抛异常"**：0.5B 会生成截断的 JSON、带尾逗号的
JSON、甚至把调用包进 ```json 围栏。解析器一崩溃，整个 loop 就死了——模型连
"自纠一次"的机会都没有。返回 `[]` 的语义是"这轮没调工具"，交给循环逻辑处理
（最终答案，或 nudge 重试，见 2.4）。**静默容错 + 上层策略，是 harness 稳定性的第一课。**

### 2.3 回填：messages 的两条追加

```python
messages.append({"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": ..., "arguments": ...}}]})
messages.append({"role": "tool", "name": "calculator", "content": "1081"})
```

两个细节都有坑：

- **assistant 消息的 `content` 留空**：工具调用已经放在结构化 `tool_calls` 字段里，
  Qwen 的 chat template 会自己渲染成 `<tool_call>` 块。如果你把原始文本（内含
  `<tool_call>`）同时塞进 content，模板会渲染两遍——模型下一轮看到"自己调了两次"。
- **`tool` role 由 template 渲染成 `<tool_response>`**：Qwen2.5 的模板把 tool 消息
  包进一个 `user` 角色的 `<tool_response>` 块。真实渲染（脚本实测）：

```
<|im_start|>assistant
<tool_call>
{"name": "file_read", "arguments": {"path": "/tmp/agent_demo.txt"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Error: file not found: /tmp/agent_demo_v2.txt. ...
</tool_response><|im_end|>
```

### 2.4 中断/重试/上下文管理：三件"循环之外"的事

**① nudge（重试）**：0.5B 高发的失败模式是工具结果回来后"光叙述不行动"——
"let's proceed to step 2..." 然后就停了。harness 注入一条 user 消息把模型推回
轨道。但 nudge 有实测教训（第一版脚本踩的坑）：

> ⚠️ **nudge 只在【所有工具结果都是错误】时注入**。第一版无条件 nudge，结果把
> 已经给出正确答案的模型拽回去"继续任务"——它没事找事去调 `echo`，被白名单
> 拒绝后陷入道歉循环，正确答案变成了"我无法继续"。**判断"任务做没做完"本身
> 是开放问题**；"全报错才推一把"是保守可用的启发式。

**② 上下文裁剪**：长任务的 tool 结果会撑爆上下文。本脚本的策略：

```python
# 保 system + 第一条 user（任务本体）+ 最近 KEEP_TAIL=12 条，砍中间老观测
head + [{"role": "user", "content": "(older tool results trimmed)"}] + tail
```

裁剪后不能让第一条是孤儿 `tool` 消息（没有配对的 assistant `tool_calls` 会破坏
模板渲染）——脚本里显式跳过。这是 Part 17 02 章"上下文管理"三件套（截断/摘要/
partial rollout）的推理侧简化版。

**③ 循环检测**：同工具**同参数**连续调用 3 次 → 判定卡死，强制停止。注意区分：
同工具不同参数（连续读三个文件）是正常探索，放行；只有"复读机"才拦。
（作业 19 题 3 的 `should_stop` 用更简单的"同工具连续 3 次"规则——两种口径的
差异见作业提示。）

### 2.5 终止条件：为什么恰好是这三个

| 条件 | 防什么 | 没有它会怎样 |
|---|---|---|
| 无 tool_calls | 正常结束（答案/放弃） | 永远循环烧钱 |
| max_turns | 模型永远"再查一下" | 一个任务跑 1000 轮 |
| 循环检测（同调用 ×3） | 复读机（0.5B 高发） | 同样的错误无限重试 |

三个条件对应三类"模型不知道自己该停"：不知道做完了、不知道做不完了、不知道
自己在重复。**生产 agent 还要加第四类：预算上限（token/钱/墙钟时间）**——
agent 的失败模式一半是"不会做事"，另一半是"停不下来"。

## 3. 实测输出（逐字来自真实运行）

> 📊 环境：RTX 4090 24GB / Qwen2.5-0.5B-Instruct fp16 / transformers 4.57.6 /
> torch 2.6.0+cu124 / 贪心解码（do_sample=False）/ 全脚本 18.4 秒。
> 贪心解码在同设备上可复现同样轨迹；换设备（CPU/fp32）浮点差异会让轨迹分岔。

### 3.1 沙箱直测（Section 0，无模型，确定性）

```
── Section 0: 工具执行器直测（沙箱安全）──
  calculator('23*47')        → '1081'
  calculator('rm -rf /')     → "Error: invalid expression 'rm -rf /'. Only numbers and + - * / ( ) are allowed."
  bash('echo hi')            → "Error: command 'echo' not allowed. Whitelist: ['cat', 'grep', 'ls', 'python']"   ← 白名单拦截
  bash(sleep 30)             → 'Error: command timed out after 10s.'
```

`echo` 被 intercept 不是疏忽——后文 Demo 4 里模型真的会去调它。

### 3.2 Demo 1/2：成功基线（各 2 轮，干净利落）

```
── Demo 1: 单工具成功基线（calculator 23*47） ──
  [turn 0] CALL calculator({"expression": "23*47"})
          → '1081'
  [turn 1] ✅ final_answer: 'The result of computing 23*47 is 1081.'
  ⇒ stop_reason=no_tool_calls, n_tool_calls=1

── Demo 2: bash 工具（cat 预写文件） ──
  [turn 0] CALL bash({"command": "cat /tmp/agent_demo.txt"})
          → '1081'
  [turn 1] ✅ final_answer: "The file '/tmp/agent_demo.txt' contains the number 1081."
  ⇒ stop_reason=no_tool_calls, n_tool_calls=1
```

- 🔑 **轨迹结构**：`[user] → [assistant: tool_call] → [tool: 1081] → [assistant: 答案]`
  ——这正是 Part 17 轨迹 `[user | call | obs | answer]` 的逐 token 版本。Part 17
  在这条轨迹上算 loss 时只对 assistant 段算（观测 mask=0）；本章推理侧只是把它
  跑出来，没有任何 loss。
- 📝 **0.5B 的诚实成绩单**：单步工具调用（schema 清晰、参数就位）0.5B 是合格的；
  它的失败集中在多步规划与失败恢复（下两节）。

### 3.3 Demo 3：失败恢复（报错 → 放弃 → nudge → 自纠）

```
── Demo 3: 失败恢复（错误路径 → 报错带修复提示 → nudge → 正确路径） ──
  [turn 0] CALL file_read({"path": "/tmp/agent_demo_v2.txt"})
          → 'Error: file not found: /tmp/agent_demo_v2.txt. Existing file you should read instead: call...'
  [turn 1] ⚠️ 模型未调工具（"I'm sorry, but there seems to be an issue with the"...）→ 注入 nudge
  [turn 2] ⚠️ 模型未调工具（"Sure, let's try this time. The error message indic"...）→ 注入 nudge
  [turn 3] CALL file_read({"path": "/tmp/agent_demo.txt"})
          → '1081'
  [turn 4] ✅ final_answer: 'The file `/tmp/agent_demo.txt` contains the number 1081.'
  ⇒ stop_reason=no_tool_calls, n_tool_calls=2
```

解剖这条恢复链路，两个工程决策缺一不可：

1. **错误信息自带修复提示**（工具层）：`file not found ... call file_read with
   path='/tmp/agent_demo.txt'`。没有提示时（开发时实测），0.5B 只会道歉并反问
   用户"Could you check the location?"——**错误信息是写给模型看的 prompt**，
   "Error: file not found"与"Error: ... use X instead"对模型的可用性天差地别。
2. **nudge 注入**（harness 层）：turn 1/2 模型连续两轮"想试但没出手"（注意
   turn 2 的开头 "Sure, let's try this time"——它已经理解了该做什么，只是没生成
   `<tool_call>`）。nudge 把"意图"翻译成"行动"。两次 nudge 后模型终于照着错误
   里的提示改了路径。

- 💡 **为什么"读不存在的文件"是本课的失败恢复案例**：它是真实 agent 最高频的
  错误类型（路径拼错/文件未创建/权限），且错误信息可以结构化地带修复建议——
  两层（工具层提示 + harness 层 nudge）配合才把 0.5B 的自纠成功率从"几乎为零"
  （只有错误提示、无 nudge 时模型直接反问用户）拉到"稳定成功"。

### 3.4 Demo 4：能力边界（多步写文件——0.5B 的真实水平）

任务：计算 23\*47 → 用 bash 把结果写入 /tmp/agent_demo2.txt → 读回验证。

```
── Demo 4: 能力边界（计算→写文件→读回验证，0.5B 高概率翻车） ──
  [turn 0] CALL calculator({"expression": "23*47"})
          → '1081'
  [turn 0] CALL bash({"command": "echo \"$(cat /tmp/agent_demo2.txt\")\""})
          → "Error: command 'echo' not allowed. Whitelist: ['cat', 'grep', 'ls', 'python']"
  [turn 0] CALL file_read({"path": "/tmp/agent_demo2.txt"})
          → "Error: file not found: /tmp/agent_demo2.txt. ..."
  [turn 1] ✅ final_answer: 'The computation was successful using the calculator. The result of 23*47 is 1081. However, there see'
  ⇒ stop_reason=no_tool_calls, n_tool_calls=3
  磁盘实况 /tmp/agent_demo2.txt = '(missing)'；final_answer 声称含 1081：True
  ⚠️ 幻觉检测阳性：模型声称任务完成，磁盘证据不支持——0.5B agent 必须外置验收
```

这条 2 轮轨迹浓缩了三个失败模式（教程作者注：全部来自真实运行，未挑选）：

1. **并行调用的依赖错误**：turn 0 模型一口气排了 3 个调用（calculator、bash、
   file_read）。但第 2、3 个调用**依赖第 1 个的结果**（文件还不存在）——并行
   排队时它没法填入还没算出的值，于是 bash 的命令里出现了照抄任务文本的
   占位符。OpenAI 语义允许并行调用，但**有依赖的步骤必须分轮**；0.5B 分不清。
2. **绕开专用工具**：明明有 calculator，模型偏用 bash 算术（还选了白名单外的
   `echo`）。工具选择的稳定性是小模型弱项。
3. **幻觉验收**：文件根本没写成功，final_answer 却说"computation was
   successful"——**模型的"完成"声明不可信，必须用环境终态验收**（这里是磁盘
   内容；τ-bench 用数据库终态，见 [02 章](02_protocols_and_frameworks.md)与
   [脚本 03](../scripts/03_tau_mini.py)）。

> 🔑 **0.5B 与生产级模型的差距不在"会不会调工具"（单步合格），而在**：
> ① 多步依赖规划（并行/串行判断）；② 失败后的策略（放弃 vs 自纠）；③ 对自身
> 行为的校验（声称完成 vs 环境证据）。这三条也正是 agent 评测（τ-bench/SWE-bench）
> 实测拉开差距的地方。

### 3.5 终止条件的确定性验证（Section 0.5，假模型）

真模型轨迹长且依赖 GPU；用"按剧本出牌"的假模型可以秒级、确定性地验证循环逻辑——
这也是 agent loop"模型无关"的证据（换模型，循环代码零改动）：

```
── Section 0.5: 假模型验证终止条件（无 GPU）──
  同参数复读 ×5        → stop=loop_detected（期望 loop_detected）
  异参数连调 10 次      → stop=max_turns（期望 max_turns，max_turns=4）
  畸形输出+放弃         → stop=no_tool_calls（期望 no_tool_calls；nudge 用尽）
  22 条消息裁剪         → 22 → 15 条（触发=True，保 system+user0+尾部）
```

## 4. 工程实践

### 4.1 常见陷阱（症状/原因/解法）

#### 陷阱 1：模型"光说不做"（narrate without acting）

**症状**：工具结果回来后，模型输出 "Now let's proceed to step 2..." 然后结束，
没有 `<tool_call>`；任务停在半路。

**原因**：0.5B 级模型把"叙述计划"与"发起调用"当成两件事，且不清楚 harness 只认
`<tool_call>` 标签。

**解法**：harness 层注入 nudge（"emit a tool_call for the next step"）+ 系统提示
加 "Do not narrate without acting"。注意 2.4 节的教训：只在全部结果都是错误时
nudge，否则会把已完成的模型拽偏。

#### 陷阱 2：assistant 回填时 content 与 tool_calls 重复渲染

**症状**：模型下一轮"以为"自己调了两次同样的工具；或模板渲染出嵌套的
`<tool_call>`。

**原因**：把含 `<tool_call>` 的原始文本塞进 `content`，同时 `tool_calls` 字段又
让模板渲染了一遍。

**解法**：回填时 `content=""`，调用只放结构化 `tool_calls` 字段。

#### 陷阱 3：白名单/超时缺失导致的事故

**症状**：模型生成 `rm -rf node_modules && npm install`（合理意图、灾难命令）；
或 `python -c "while True: pass"` 把 agent 挂死。

**原因**：把模型输出当可信输入直接交给 `shell=True`。

**解法**：白名单（第一词检查）+ `timeout=10` + 输出截断 400 字符 + 每轮调用数
上限。生产再加容器沙箱与审计日志。

#### 陷阱 4：解析器对畸形 JSON 抛异常，整个 loop 崩溃

**症状**：`json.JSONDecodeError` 直接炸掉脚本；或轨迹里一条畸形输出毁掉整段任务。

**原因**：0.5B 会输出截断 JSON/尾逗号/围栏包裹的 JSON。

**解法**：宽容解析（剥围栏 → 修尾逗号）+ 失败返回 `[]`，把决策交给上层
（nudge 或终止）。

### 4.2 性能分析

| 项 | 实测（RTX 4090，fp16） | 说明 |
|---|---|---|
| 模型加载 | ~8 s | 每脚本一次；0.5B 占显存 ~1 GB |
| 每轮生成（≤200 token） | ~1-2 s | HF generate，贪心 |
| 脚本 01 全程（4 demo + 自测） | 18.4 s | 11 次生成 + 1 次 10s 超时演示 |
| 纯 CPU 同流程 | ~慢 20-40 倍 | 0.5B CPU ~3-5 token/s |

- 🚀 生产化第一刀：把 HF `generate` 换成 vLLM（Part 14）——同一循环、API 化的
  `tool_calls` 格式（解析层已兼容），吞吐高一个量级。
- 复杂度：轮数 × 每轮上下文长度 × 每 token 计算。**上下文裁剪不是优化而是
  正确性的一部分**（裁剪阈值决定长任务能否完成）。

### 4.3 最佳实践

1. **工具 description 当 prompt 写**：参数含义、合法取值、示例全部写进 schema
   （bash 的 description 直接写白名单，实测能约束 0.5B 的命令选择）。
2. **错误信息当 prompt 写**：带修复建议的错误（"use X instead"）让弱模型也能
   自纠；裸 "Error" 只会触发道歉。
3. **一切模型输出不可信**：白名单/超时/截断/验收（环境终态）四件套。
4. **循环逻辑与模型解耦**：假模型可测（本脚本 Section 0.5），换模型零改动。

## 5. 概念检验

<details>
<summary>Q1: agent loop 与 Part 17 的训练循环，对"同一条轨迹"各自关心什么？</summary>

A: 轨迹格式相同：`[user | tool_call | observation | ... | answer]`。推理侧（本章）
只关心**执行**：解析调用、执行、回填上下文、终止条件。训练侧（Part 17）关心
**梯度**：哪些 token 是 assistant（loss mask=1）、观测段不算 loss（防"复读观测"）、
整条轨迹的奖励怎么广播（轨迹级 GRPO）。一句话：推理侧拼接上下文，训练侧切割
loss。

</details>

<details>
<summary>Q2: 为什么 parse_tool_calls 解析失败要返回 [] 而不是抛异常？</summary>

A: `[]` 的语义是"这轮没有工具调用"，循环已有处理路径（最终答案/nudge 重试），
模型保留自纠机会；抛异常则整个 loop 崩溃，一次畸形输出毁掉整段任务。0.5B 的
畸形输出（截断/尾逗号/围栏）是常态而非异常——harness 的稳定性预算要按"模型
一定会犯错"来设计。

</details>

<details>
<summary>Q3: 循环检测为什么判"同工具同参数 ×3"而不是"同工具 ×3"？</summary>

A: 同工具不同参数往往是正常探索（连续读三个不同文件）；同工具**同参数**复读
才是死循环（同样的错误无限重试）。口径更宽的"同工具 ×3"会把正常探索误杀：
设想一个"依次核对 a/b/c 三个配置文件再汇总"的任务，第 3 个 file_read 会被
拦在半路。本章 Demo 3 恰好两笔 file_read（错→对），宽口径拦不到它——作业 19
题 3 用宽口径练手，实现时想想你会在哪类任务里被误伤。

</details>

<details>
<summary>Q4: Demo 4 里模型"声称完成"但磁盘没有文件。生产上怎么防？</summary>

A: ① 环境终态验收：任务的判定条件写成对环境的检查（文件内容/DB 字段/测试
通过），不解析模型的自然语言声明；② 任务分解：让每步的可验证性提高（写文件后
强制读回比对）；③ 这正是 τ-bench 用 DB 终态判分、SWE-bench 用测试通过判分的
原因——见 [02 章](02_protocols_and_frameworks.md)评测一节。

</details>

## 6. 动手实践

### 练习 1：给 agent 加第四个工具 `file_write(path, text)`

**验收标准：**
- [ ] schema 合法（required 含 path/text），description 说明行为
- [ ] 执行器写文件成功且返回确认信息；路径非法返回 Error 字符串
- [ ] 用"把 hello 写入 /tmp/x.txt 再读回"任务跑通完整轨迹

**步骤提示：**
```python
TOOL_SPECS.append({"type": "function", "function": {
    "name": "file_write", "description": "Write text to a file.",
    "parameters": {...}}})
EXECUTORS["file_write"] = exec_file_write   # 仿照 exec_file_read 写
```

### 练习 2：把 nudge 条件改成"关键词检测任务完成"

**任务：** 用任务关键词（如 query 中的文件名是否出现在 final_answer）判断任务
是否完成，决定是否 nudge。对照 2.4 节的"全报错才 nudge"启发式，各跑 Demo 1-4，
比较误 nudge 次数与任务成功率。

**验收标准：**
- [ ] Demo 1/2 零误 nudge（已完成的任务不被拽回）
- [ ] Demo 3 仍能恢复（该 nudge 时 nudge）
- [ ] 用一段话总结你的判据在什么任务上会失效

## 🧭 扩展思考

**思考 1：并行工具调用。** OpenAI 语义允许一条消息多个 tool_calls。什么时候
安全（无依赖）、什么时候危险（Demo 4 的占位符问题）？你会怎么改 schema/
系统提示，让模型显式声明"这些调用互不依赖"？（提示：MCP 与 OpenAI 的
parallel tool calls 设计差异。）

**思考 2：nudge 与 RL 的边界。** nudge 本质是 harness 替模型做了"没做完要继续"
的决策。Part 17 告诉我们这个决策可以被训练进模型（RL 的 credit assignment）。
列出三种"该 nudge 还是该训练"的判断依据（任务频率？模型规模？可观测性？）。

## 参考资源

- 脚本：[../scripts/01_agent_loop.py](../scripts/01_agent_loop.py)
- Part 17 轨迹与观测 mask：[../../Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md](../../Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md)
- Qwen2.5 工具调用模板：[Qwen2.5 官方仓库](https://github.com/QwenLM/Qwen2.5)（chat template 与 tool call 格式文档）
- ReAct 原始论文：ReAct: Synergizing Reasoning and Acting in Language Models (arXiv 2210.03629)
- OpenAI Function Calling 文档格式（tools JSON schema 与 finish_reason="tool_calls"）

## 学完本章你能...

- ✅ 手写不依赖框架的 agent loop（渲染→生成→解析→执行→回填）
- ✅ 实现兼容两种格式的 parse_tool_calls 与畸形输出兜底
- ✅ 说出三种终止条件并解释各自的必要性
- ✅ 给 bash 工具配最小安全边界（白名单+超时+截断）
- ✅ 解剖 0.5B 的真实失败轨迹（并行占位符/幻觉验收/报错后放弃）

## 下一步

工具层自己写、协议各发明的日子到头了——下一章把工具层标准化（MCP）、agent 间
通信标准化（A2A），再对照框架生态（LangGraph/OpenAI Agents SDK/smolagents/Pi）
与多智能体的三方辩论，最后用 τ-mini 把"agent 评测"跑出数字。

👉 [02 — 协议、框架与评测生态](02_protocols_and_frameworks.md)

---

[← Part 19 README](README.md) | [下一章：02 协议与框架 →](02_protocols_and_frameworks.md)
