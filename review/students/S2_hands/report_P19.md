# S2 学习报告 · Part 19（Agent 与 Function Calling）

- 审计身份：学生 agent S2（实操薄弱型：以"跟着教程跑脚本 + 做作业"的视角审计）
- 审计日期：2026-09-04
- 环境：RTX 4090 24GB / torch 2.6.0+cu124 / transformers 4.57.6 / Python 3.12.12 / Qwen2.5-0.5B-Instruct（本地缓存）——与教程 01 章声明的环境一致
- 材料：教程 2 章 + README、脚本 3 个（01/02/03）、作业三件（只读教程/脚本/作业，未读 assignment_reference）
- 跑脚本 scratch：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P19/`（s2_01/02/03_output.txt）
- 作业 work：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_19/`

## 总分：9.2 / 10

一句话：这是我跟过以来"实测可复现度"最高的教程——01 章 4 个 Demo 的模型轨迹与我跑出的**逐字一致**；扣分集中在小事：三处耗时口径不一、节选代码/实录未标"节选"、离线环境第一跑会崩。

## 卡点清单（按踩坑顺序）

| # | 卡点 | 类型 | 影响 | 解法/耗时 |
|---|------|------|------|-----------|
| 1 | 脚本 01 首跑 ProxyError 崩溃：transformers 4.57.6 加载 tokenizer 时 `_patch_mistral_regex` 会访问 HF Hub API（**模型已在本地缓存也照样访问**），本机代理 192.168.0.105:7890 不可达 → 整个脚本挂掉 | 环境 | 首跑即崩，58 秒后报错；离线/内网学生第一个就摔 | 加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 后一次通过。README"环境"节只说"直跑"，没给离线开关 |
| 2 | 脚本 03 耗时三处口径不一：README 与 02 章正文说 "~15-25s"，脚本 03 docstring 自己说 "~12-25s"，我实测 10.6s（低于两者下限） | 文档 | 数字对不上时学生先怀疑自己跑错了 | 无实质影响，建议统一并注明"因机器而异" |
| 3 | assignment.md 开头说同名接口是 "`parse_tool_calls` / `TOOL_SPECS`"，但作业骨架里没有 `TOOL_SPECS` 要实现（题 1 实际是实现生成 schema 的 `tool_spec` 函数） | 文档 | 第一眼找不到 TOOL_SPECS 要干嘛，多花了 2 分钟确认题意 | 把 `TOOL_SPECS` 改成 `tool_spec` 即可 |
| 4 | 题 1 函数路径：Python 参数如何映射 JSON type 教程/作业都没说（`inspect` 只给名字不给类型） | 认知 | 约 3 分钟犹豫；测试恰好不查函数路径的 type，默认 "string" 就过 | 建议题目里点一句"推不出就 string" |
| 5 | 02 章脚本 02"真实输出"实录是节选：实际 Step 4 打印 **两条**错误（resources/list 之外还有 multiply→unknown tool），后面还有"线上字节流（前 5 条）"整段，教程都没贴且未标注省略 | 文档 | 对照输出时以为漏跑了什么；确认逐字一致后排除 | 实录标注"节选"即可 |

无超过 3 分钟被中断的运行（最长单次：脚本 01 离线重跑 15.7s；唯一失败是卡点 1 的 58s 崩溃）。

## 脚本实测（对照教程数字）

| 脚本 | 教程声明 | 我的实测 | 结果 |
|------|---------|---------|------|
| 01_agent_loop.py | 18.4s（README ~20s，"贪心解码同设备逐字一致"） | **15.7s** | ✅ Section 0/0.5 + Demo 1-4 输出与教程 3.1-3.5 **逐字一致**（含 Demo 3 两次 nudge 链路、Demo 4 幻觉检测阳性、22→15 条裁剪） |
| 02_mini_mcp.py | <1s | **0.036s** | ✅ Step 1-3 与教程逐字一致（Step 4 教程为节选，见卡点 5） |
| 03_tau_mini.py | ~15-25s（docstring ~12-25s） | **10.6s** | ✅ T1=0.00 / T2=0.00 / T3=0.33 / **OVERALL pass^1=0.11**，与教程"另一次运行 T3=1/3、OVERALL 0.11"吻合；T1 第 1 次运行模型真把 "10 days ago" 的 10 当金额 `refund(1002, 10)`，与教程描述的失败模式逐字对上 |

## 分章评分

| 章节/材料 | 得分 | 评语 |
|-----------|------|------|
| 01 手写 Agent Loop | 9.5/10 | 五部件讲解与脚本一一对得上，真实失败轨迹可复现、有解剖有对策（nudge 只在全报错时注入的教训写得极好）。扣 0.5：2.1/2.2 节的节选代码块未标"示意/节选"（如 exec_bash 片段省了 PATH 注入与空命令检查） |
| 02 协议、框架与评测生态 | 8.8/10 | mini-MCP 的"notification 回应读串"bug 复盘、三方辩论、榜单读法都到位；τ-mini 数字诚实（附方差）。扣分：实录节选未标注、政策"~300 字"不准（实际约 570 字符/~120 词）、脚本 03 耗时口径与 README 不一致 |
| README（Part 19） | 9/10 | 导航/前置/学习地图清晰，数字基本准（脚本 03 的 15-25s 与正文一处出入） |
| 脚本 01/02/03 | 9.7/10 | 断言齐全（沙箱行为、解析五类输入、三种终止条件全有 assert）、假模型设计让无 GPU 学生也能验证循环逻辑、03 的 verify() 判分器三查结构清晰。唯一笔误：02 docstring "resources/prompts/prompts" 多打一个 prompts |
| 作业 assignment_19 | 9/10 | 五题纯 CPU 秒级、测试质量高（题 5 未实现优雅 SKIP 的设计很贴心）、实验题与思考题跟教程呼应紧密（Q2 宽/严口径与教程 2.4③ 预警一致）。扣分：卡点 3/4 的表述瑕疵 |

## 一致性核对表（教程代码块 ↔ 脚本）

| 对照项 | 教程位置 | 脚本位置 | 一致性 |
|--------|---------|---------|--------|
| calculator 的 TOOL_SPECS schema | 01 章 2.1 | 脚本 01 L53-59 | ✅ 逐字一致 |
| bash 白名单/超时/截断片段 | 01 章 2.1 | 脚本 01 L103-119 | ✅ 语义一致；⚠️ 教程片段为节选（省略空命令检查、"(no output)"、PATH 注入），**未标"示意"**（片段内有 "..." 提示） |
| parse_tool_calls 两种格式 + 兜底返回 [] | 01 章 2.2 | 脚本 01 L139-205 | ✅ 逻辑一致（finditer + 修尾逗号 + 剥围栏 + []）；片段明显省略，未标"示意"但有注释指路 |
| 回填两条件（assistant content=""/tool role） | 01 章 2.3 | 脚本 01 L290-297 | ✅ 一致 |
| 上下文裁剪 head+占位+tail、KEEP_TAIL=12 | 01 章 2.4② | 脚本 01 L209-220 | ✅ 一致（22→15 实测复现） |
| 循环检测"同工具同参数 ×3" | 01 章 2.4③/2.5/Q3 | 脚本 01 L304-307 | ✅ 一致；且教程明确预警作业题 3 是宽口径——两处口径差异**有互相呼应，不冲突** |
| 三种终止条件 | 01 章 2.5 表 | 脚本 01 run_agent + Section 0.5 | ✅ 一致，假模型断言全过 |
| Section 0 / 0.5 / Demo 1-4 实测输出 | 01 章 3.1-3.5 | 我实测 s2_01_output.txt | ✅ **逐字一致**（15.7s vs 18.4s） |
| 耗时表（加载 ~8s/全程 18.4s/CPU 慢 20-40 倍） | 01 章 4.2 | — | ✅ 同数量级（我的机器更快） |
| mini-MCP 三步握手/isError vs -32601 | 02 章 2.2 | 脚本 02 dispatch/serve | ✅ 一致（消息示例的 id=3 与脚本真实 id 序号也对得上） |
| mini-MCP"真实输出"实录 | 02 章 2 | 我实测 s2_02_output.txt | ⚠️ Step 1-3 逐字一致；Step 4 教程只贴 1/2 条错误、缺"线上字节流"段，**节选未标注** |
| τ-mini 判分三要件表 + 实测数字 | 02 章 5.1 | 脚本 03 verify()/TASKS | ✅ 一致；pass^1=0.11 与教程方差声明吻合 |
| 政策文档"~300 字" | 02 章 5.1 表 | 脚本 03 POLICY | ⚠️ 实际约 570 字符 / ~120 英文词，"300 字"两个口径都不符（小瑕疵） |
| 脚本 03 耗时 | README & 02 章 15-25s | docstring 12-25s | ⚠️ 两处口径不一；实测 10.6s |
| 作业接口同名声明 | assignment.md 开头 | 骨架 agent_exercises.py | ⚠️ "`TOOL_SPECS`"名不符实（实际是 `tool_spec` 函数）；`parse_tool_calls`/`pass_at_1` 同名同语义 ✅ |
| 作业题 3 宽口径 vs 脚本严口径 | 教程 01 章 2.4③ 预警 | assignment.md Q2 提示 | ✅ 预警到位，互相呼应 |

## 作业元数据 + pytest 输出

工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_19/`（三件齐全，只编辑了 agent_exercises.py）

| 题号 | 一次过? | 提示次数 | 耗时 | 结果 |
|------|---------|---------|------|------|
| 1 tool_spec | ✅ 一次过 | 0（卡过一次：函数参数 type 映射自己决定默认 string，见卡点 4） | ~8 min | PASSED |
| 2 parse_tool_calls | ✅ 一次过 | 0（照教程 2.2 的兜底策略直接写） | ~3 min | PASSED |
| 3 should_stop | ✅ 一次过 | 0（优先级顺序题目写得很清楚） | ~2 min | PASSED |
| 4 pass_at_1 | ✅ 一次过 | 0 | ~1 min | PASSED |
| 🌟 5 mini_mcp_call | ✅ 一次过 | 0（犹豫点：error 检查放逐步还是只查第三步，选了逐步） | ~5 min | PASSED（先跑时正确 SKIP ⏭️，验证了弹性机制） |

最终 pytest 输出（`python -m pytest test_*.py -v`）：

```
test_agent_exercises.py::test_ex1_tool_spec PASSED                       [ 20%]
test_agent_exercises.py::test_ex2_parse_tool_calls PASSED                [ 40%]
test_agent_exercises.py::test_ex3_should_stop PASSED                     [ 60%]
test_agent_exercises.py::test_ex4_pass_at_1 PASSED                       [ 80%]
test_agent_exercises.py::test_ex5_mini_mcp_call PASSED                   [100%]
============================== 5 passed in 0.00s ===============================
```

**5 题 5 PASS**（题 5 弹性题也实现并通过；中途验证过未实现时优雅 SKIP 生效）。

## 只改 3 件事

1. **README"环境"节加一行离线开关**：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（transformers 4.57.x 加载 tokenizer 会探测 Hub，模型已缓存也会因代理失效崩掉——离线/内网学生的第一坑）。
2. **统一脚本 03 的耗时口径**：README/02 章说 "~15-25s"、docstring 说 "~12-25s"、实测 10.6s——统一成"约 10-25s，因机器而异"，三处同步。
3. **给节选内容标"节选"**：01 章 2.1/2.2 的代码片段、02 章的脚本 02"真实输出"（补上或注明省略了 multiply 错误行与"线上字节流"段）；顺手修 assignment.md 的 `TOOL_SPECS`→`tool_spec`、02 docstring "resources/prompts/prompts" 笔误、政策"~300 字"→"~120 词"。

## 最喜欢 3 处

1. **Demo 3-4 的真实失败轨迹逐字可复现**：我跑出的 nudge 两次链路、"Sure, let's try this time"的未出手叙述、Demo 4 幻觉验收阳性（final_answer 声称含 1081 / 磁盘 missing）与教程一字不差——"失败也是教材"不是口号，是可验证的事实。
2. **Section 0.5 假模型**：用脚本化 FakeTok/FakeGen 秒级确定性验证三种终止条件与裁剪——既是"循环与模型解耦"的证据，也顺手教了 agent 代码该怎么测（无 GPU 学生完全不被排除）。
3. **τ-mini 的 verify() 判分器**：三查（get_order 前置 / 退款金额 / DB 终态）让 T2"嘴上抱歉、手上落库"无所遁形；我实测 T1 里模型真把"10 days ago"的 10 当金额退了——教程预言的失败模式当场重现，评测"只认行为不认话术"这一课印象极深。
