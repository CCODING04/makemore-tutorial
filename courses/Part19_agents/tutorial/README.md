# Part 19: Agent 与 Function Calling — 手写 Agent Loop、MCP 协议与 τ-bench 评测（应用线 A2）

> 🧭 [Part 17](../../Part17_agentic_rl/tutorial/README.md) 是本部分的**训练侧姊妹篇**：
> 17 把多轮工具调用轨迹当训练数据做 RL（**训 agent**），19 把同一条轨迹在推理侧
> 跑起来（**用 agent**）。一个 `while` 循环 + 三工具 + 三种终止条件就是 agent 的
> 全部骨架（[脚本 01](../scripts/01_agent_loop.py)，GPU 实测 ~20 秒）；在此之上
> 补齐工程生态：工具层协议 MCP 手写复刻（[脚本 02](../scripts/02_mini_mcp.py)，
> 秒级）与 agent 评测 τ-bench 微缩（[脚本 03](../scripts/03_tau_mini.py)，GPU ~15-25 秒，
> Qwen2.5-0.5B 实测 pass^1）。0.5B 模型的成功与翻车都是教材——失败轨迹逐条解剖。

## 学习目标

完成本部分后，你将能够：

- ✅ **手写** 不依赖任何框架的 agent loop（渲染→生成→解析→执行→回填 + 三种终止条件）
- ✅ **实现** 兼容 Qwen `<tool_call>` 与 API `tool_calls` 两种格式的解析器及兜底策略
- ✅ **复刻** mini-MCP（JSON-RPC 2.0：initialize / tools/list / tools/call）并说出
  MCP/A2A/AGENTS.md 三层分工
- ✅ **实测** τ-bench 式评测（政策合规 + 用户模拟 + DB 终态）并正确解读 pass^1/pass^k
  与 SWE-bench 榜单（脚手架口径、第三方榜单污染）
- ✅ **复述** 多智能体三方辩论（Anthropic / Cognition / LangChain）并落到
  "上下文隔离才是收益来源"

## 理论背景（导览）

| 概念 | 一句话 | 详见 |
|------|--------|------|
| Agent loop | while 循环：模型发调用单 → harness 执行 → 结果回填上下文 | [01 章](01_agent_loop.md) |
| 终止条件 | 无调用 / max_turns / 循环检测（复读机）——"停不下来"是真实故障类 | [01 章](01_agent_loop.md) |
| 沙箱边界 | 白名单 + 超时 + 截断：模型输出是不可信输入 | [01 章](01_agent_loop.md) |
| 协议三层 | MCP=agent↔工具、A2A=agent↔agent、AGENTS.md=agent↔代码库 | [02 章](02_protocols_and_frameworks.md) |
| 多智能体辩论 | 收益来自上下文隔离而非数量（token 成本 ~15× 是乘数） | [02 章](02_protocols_and_frameworks.md) |
| Agent 评测 | DB 终态判分 + 政策合规 + pass^k；榜单必须锁版本看脚手架 | [02 章](02_protocols_and_frameworks.md) |

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 Agent Loop](01_agent_loop.md) | 五部件逐行 / 轨迹解剖（对照 Part 17）/ nudge 与上下文裁剪 / bash 白名单 / 0.5B 失败轨迹解剖 | `01_agent_loop.py` |
| 02 | [协议、框架与评测生态](02_protocols_and_frameworks.md) | MCP/A2A/AGENTS.md 三层 / mini-MCP 讲解 / 框架四流派 / 多智能体三方辩论 / τ-mini 实测 / agentic RL 去向 | `02_mini_mcp.py`、`03_tau_mini.py` |

## 🧰 前置知识

- **必须掌握**：
  - [Part 8 02 章 SFT 与 chat 模型](../../Part8_post_training/tutorial/02_sft_and_chat.md)——
    chat template（工具调用协议就是它的扩展：`apply_chat_template(messages, tools=...)`）
  - [Part 17 01 章](../../Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md)——
    多轮轨迹格式 `[user | tool_call | observation | answer]`（本部分轨迹 = 它的训练数据）
- **建议掌握**：[Part 18 RAG](../../Part18_rag/tutorial/README.md)——同为应用线
  （A1/A2 姊妹篇）：RAG 给 agent 提供"检索"这个最重要的工具类别之一
- **可选**：[Part 14 vLLM 推理部署](../../Part14_inference_vllm/tutorial/README.md)——
  把 01 章的 HF generate 换成 vLLM/OpenAI 兼容接口（`tool_calls` 结构化格式）

## 🔗 在 LLM 链路中的位置

```
Part 17（Agentic RL：训 agent）──┐
                                 ├──→ 【本部分: 用 agent——loop/协议/评测】→ 应用线（A1 RAG + A2 Agent）收尾
Part 18（RAG：应用线 A1）───────┘
```

会写 agent loop ≠ 会有可用的 agent：0.5B 单步工具调用合格，多步规划与失败自纠
不可用（本部分实测）——这个差距正是 Part 17 RL 要弥合的对象，也是评测（τ-bench/
SWE-bench）要度量的对象。

## 📦 环境

- 脚本 01/03 需要 `Qwen/Qwen2.5-0.5B-Instruct`（~1GB 显存；纯 CPU 可跑但慢 20-40 倍）
- 脚本 02 纯 CPU、零模型、秒级（只依赖标准库 + 子进程）
- 全部脚本 `MPLBACKEND=Agg python 0X_*.py` 直跑；01 章贪心解码可复现教程轨迹
  （同设备），03 章温度采样（pass^1 的方差本身就是教学内容）

## 📈 学习地图

```
while 循环 + 工具结果回填（agent 全部骨架）   ← 点：剥掉框架看本质
   ↓ 解析层（两种格式 + 畸形兜底）与沙箱边界（白名单/超时/截断）
失败恢复（错误带修复提示 + harness nudge）      ← 线：0.5B 的真实水平与抢救手段
   ↓ 协议标准化（MCP 三步握手 → 工具层解耦；A2A/AGENTS.md 分层）
评测（τ-mini：政策合规 + DB 终态 + pass^1）     ← 面：榜单怎么读、系统怎么评
   ↓ agentic RL 去向（评测当奖励 = Part 17 闭环）← 回环
```

## 📝 课后作业

👉 [Assignment 19](../../../assignments/assignment_19/)

## 🔗 相关资源

- 📄 ReAct: Synergizing Reasoning and Acting in Language Models (arXiv 2210.03629)
- 📄 τ²-bench (2506.07982) · GiGPO (2505.10978) · AgentRL (2510.04206)
- 🌐 [MCP 官方规范](https://modelcontextprotocol.io) · [A2A 官方站](https://a2a-protocol.org) ·
  [SWE-bench 官方榜](https://www.swebench.com)
- 🐙 [Qwen2.5（工具调用模板）](https://github.com/QwenLM/Qwen2.5) ·
  [verl-agent（GiGPO 官方）](https://github.com/langfengq/verl-agent) ·
  [badlogic/pi-mono（极简派 Pi）](https://github.com/badlogic/pi-mono)
- 📝 Anthropic [多 agent 研究系统工程博客](https://www.anthropic.com/engineering/built-multi-agent-research-system) ·
  Cognition [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents)

---

[← 上一部分：Part 18 RAG（应用线 A1）](../../Part18_rag/tutorial/README.md) | [返回总览](../../../README.md)

🎓 **应用线（A1/A2）到此建成**：RAG（Part 18）解决"知识从哪来"，Agent（Part 19）
解决"行动怎么执行"。带上你的实操记录（本课每章的真实数字与失败轨迹）进入
[面试指南 §7b 方向深挖](../../../docs/llm_interview_guide.md)；训练侧回
[Part 17](../../Part17_agentic_rl/tutorial/README.md)。
