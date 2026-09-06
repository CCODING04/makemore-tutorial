# 问题台账 · Part 19（Agent / Function Calling）

> T2 整改（2026-09-04）。基线：脚本 01=15.3s / 02=0.03s / 03=8.6s（HF_HUB_OFFLINE=1，rc=0）；
> 修后：01 stdout **逐字一致**（diff 空，唯一例外=stderr 白名单警告行，见 T-01）、02 逐字一致、
> 03 rc=0（pass^1=0.00，温度采样方差与教程双口径叙事相符）；pytest 参考答案 5/5；
> check_latex 4 个改动 md 全 0 问题。合计 **21 条：P0×1 / P1×7 / P2×12 + disputed×1**。

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 | 复核 |
|------|--------|------|------|------|------|----------|----------|------|
| P19-T-01 | P0 | fixed | S1🔴+S2/S3+必修1 | bash 白名单只查第一词 + `shell=True`，`;`/`&&`/`\|`/命令替换可穿透（`cat x && rm -rf /` 第一词是 cat 照跑）；教程反复讲"模型输出不可信"却在执行器上漏了 shell 元字符这一课，陷阱 3 的例子（第一词 rm 被拦）反而误导学生以为白名单能挡 `&&` 链 | 01 章 2.1/陷阱3 + 脚本 01 `exec_bash` | F01 ①01 章 2.1 新增"白名单为什么不够"小节（绕过反例 + `shell=False`/argv、allowlist 解析、容器隔离三条生产出路）；②陷阱 3 补"症状例子恰是低级形态"辨析；③脚本 `exec_bash` 命中元字符时向 **stderr** 打 ⚠️（不改变判定，保 Demo 轨迹） | 实测 `exec_bash("cat /tmp/agent_demo.txt && echo PWNED")` → 打警告且返回 `'1081PWNED'`（绕过实证）；修后脚本 01 stdout 与基线 `diff` 逐字一致，唯一警告行来自 Section 0 超时演示（`import time; time.sleep(30)` 命中 `;`，stderr） | 待用户 |
| P19-T-02 | P1 | fixed | S1🟡+必修2 | pass^k 乘法原理（含 0.7^8≈0.057 数值例）只藏在折叠答案 Q3，正文只有文字定义；且全课未提 HumanEval `pass@k` 方向相反 | 02 章 5.1 | F02 正文补乘法原理段：LaTeX `$\text{pass}^k = p^k$` + `$0.7^2{=}0.49 \to 0.7^4{≈}0.24 \to 0.7^8{≈}0.057$` 数字例 + k 翻倍=平方直觉 + pass@k 记号方向对照框 | check_latex 0 问题；G4 曲线图同款数字互证 | 待用户 |
| P19-T-03 | P1 | fixed | S1/S2/S3 实测+必修3 | 离线/代理失效环境首跑必崩：transformers 4.57.x 加载 tokenizer 仍探测 Hub（模型已缓存也访问），ProxyError 崩溃；README"环境"节只字未提离线开关 | README 环境 + 01 章 §3 + 02 章导语 | F03 三处补 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线跑法 | 三脚本在 `HF_HUB_OFFLINE=1` 下全部 rc=0（01=15.3s/03=7.9s） | 待用户 |
| P19-T-04 | P1 | fixed | S1🟡+必修4 | τ-mini `verify()` 只查调用"出现过"，**不强制 get_order→refund 时序**（先退款后补查也 pass），政策第 1 条的时序语义未进判分器且正文未声明 | 02 章 5.1 要件表 | F04 判分行补"当前实现只查出现过、不强制时序"声明 + 指向练习 2 思考题 | 目检（脚本 03 verify 与声明一致） | 待用户 |
| P19-T-05 | P1 | fixed | S3🟡+必修4 | "MCP 三层"歧义：教程口径=协议三层（MCP/A2A/AGENTS.md），面试另解=MCP 自身角色（协议/客户端/服务器），只教了前者 | 02 章 §1 | F05 表后新增歧义辨析框：两种问法各自的答案落点（§1 表 vs §2.2/脚本 02） | 目检 | 待用户 |
| P19-T-06 | P1 | fixed | 教师(plan缺口#1)+必修5 | 陷阱 3 解法列"每轮调用数上限"，脚本 01 循环无此实现（脚本 03 的 `calls[:3]` 才有）——承诺-实现不符 | 01 章 陷阱3 | F06 注明"脚本 03 `calls[:3]` 是最小实现；脚本 01 教学 loop 未加（讲解先行、实现留白）" | 目检（脚本 01 L295 无上限 / 脚本 03 L207 `calls[:3]`） | 待用户 |
| P19-T-07 | P1 | fixed | 教师(G10×4)+S2+必修5 | 讲解用节选未标"节选/示意"（4 处同根：exec_bash 片段含未定义 `env`/`BASH_TIMEOUT`、parse_tool_calls `...` 空体、回填省略号 dict、裁剪伪表达式）+ 02 章脚本 02"真实输出"实录为节选未标（省第二条错误与"线上字节流"段） | 01 章 2.1-2.4、02 章 §2 | F07 各代码块首行加"节选/示意 + 完整实现见脚本 L 行"标注；02 章实录注明省略内容 | 目检；02 章省略内容与实测 base_02.txt 对读吻合 | 待用户 |
| P19-T-08 | P1 | fixed | 教师(G1×2)+必修8 | ①②③ 连排未拆行：01 章 3.4 差距 blockquote、02 章 §4 Cognition 两定律 | 两处 | F08 拆为逐行编号（blockquote 内加空行 / 独立列表） | 目检 | 待用户 |
| P19-T-09 | P2 | fixed | S2🟢+必修5 | "同一 JSON schema 格式"抹平外壳差异：OpenAI `function.parameters` vs MCP 顶层 `inputSchema`（脚本 02 用的就是后者） | 02 章 2.1 表 + 脚本 02 docstring/注释 | F09 表格说明列点破"内核同、外壳键名不同"；脚本 02 两处注释同步 | 目检 | 待用户 |
| P19-T-10 | P2 | fixed | S1/S2+必修5 | 政策文档"~300 字"与实际不符（实测 **614 字符 / 100 词**） | 02 章 5.1 表 + 脚本 03 L40 | F10 改"约 614 字符 / ~100 英文词"（python 实测计数） | `len(POLICY)=614`、`len(POLICY.split())=100` | 待用户 |
| P19-T-11 | P2 | fixed | S2/S3+必修5 | 脚本 03 耗时三口径漂移：README/02 章 ~15-25s vs docstring ~12-25s vs 实测 10.6s | README L8 + 02 章 L9 + 脚本 03 docstring | F11 三处统一为"约 10-25 秒（因机器而异）" | 三处 grep 一致 | 待用户 |
| P19-T-12 | P2 | fixed | S2🟢+必修5 | assignment.md 同名接口声明写 `TOOL_SPECS`，作业骨架实际实现的是 `tool_spec` 函数（名实不符） | assignments/assignment_19/assignment.md 头注 | F12 改为"`parse_tool_calls` 同名 + `tool_spec` 生成单条 schema 元素可塞进脚本 01 的 `TOOL_SPECS`" | 目检 + pytest 5/5 | 待用户 |
| P19-T-13 | P2 | fixed | S1🟢+必修6 | 22→15 条裁剪无数值分解（15 的构成要学生自己默算） | 01 章 3.5 | F13 补对账：head 2（system+user0）+ 裁剪标记 1 + tail 12 = 15 | 算术复核 | 待用户 |
| P19-T-14 | P2 | fixed | S1🟢+必修6/G13 | 90.2% 未注明是相对提升还是百分点差、基数未公布；15×/90.2% 无披露时点 | 02 章 §4 | F14 注明"相对提升（rel. improvement）、绝对基数未公布、引用注明口径"+ 披露时点 2025-06 | 目检 | 待用户 |
| P19-T-15 | P2 | fixed | S1🟢+必修6 | 评测方差"指出问题没给出路"（9 样本 CI 无意义之后无可操作下文） | 02 章 5.1 方差框 | F15 补三条出路：加大 R（比例型 R≥30 再谈区间 / Wilson·Clopper-Pearson 并报区间宽）、改 R=6 复跑（作业实验题）、降级定性结论 | 目检 | 待用户 |
| P19-T-16 | P2 | fixed | S3+必修7 | ragflow/llama_index 对比属 roadmap A1/Part 18 范围，P19 姊妹篇处无互指（S3 曾误判为 A2 缺口） | README 前置知识 Part 18 条 | F16 加一句互指"（ragflow/llama_index 等对比属 Part 18/A1 范围，见其章节）" | 目检 | 待用户 |
| P19-T-17 | P2 | fixed | 教师(G4)+必修8 | 全 Part 零配图（无 images/ 目录） | courses/Part19_agents/images/ | F17 新增 `pass_at_k_curve.png`（p∈{0.5,0.7,0.9} 的 pass^k 衰减曲线 + 0.7^8=0.057 标注），02 章 5.1 正文嵌入；生成脚本 `scratch/T2_P19/make_pass_at_k_fig.py` 可复现 | 图已生成并被 md 引用 | 待用户 |
| P19-T-18 | P2 | fixed(转述待核) | 教师(降级核实 4 项) | Pi 博客 URL/日期、smolagents"千行"口径、arXiv 编号（τ²/GiGPO/AgentRL）、τ-bench 评分 bug 出处、SWE-bench 80.9% 现值——离线均不可核 | 02 章参考资源/§5 + 脚本 03 两处 | F18 参考资源顶部加"转述待核"总声明；评分 bug 三处标"社区流传口径（转述待核）"；SWE-bench 注明"转述、撰写时点口径、现值未离线复核" | 标注齐备，联网窗口补核（见 outline） | 待用户 |
| P19-T-19 | P2 | fixed | S2🟢 | 脚本 02 docstring "resources/prompts/prompts" 笔误（多一个 prompts） | 脚本 02 docstring | F19 改"resources/prompts" | 目检 | 待用户 |
| P19-T-20 | P2 | fixed | 教师(G14 轻项) | 02 章 τ-mini 实测口径缺 transformers/torch 版本（01 章样板有、02 章漏） | 02 章 5.1 实测行 | F20 补 "fp16，transformers 4.57.6 / torch 2.6.0+cu124" + 全程约 10-25s | 目检 | 待用户 |
| P19-T-21 | P2 | disputed | S1 附录 | 脚本 02 未知工具回 -32601（method not found），S1 按 MCP 规范口径主张应为 -32602（Invalid params）——"未知工具/未知方法"的错误码归属两种读法各有支持，离线无法裁 | 脚本 02 dispatch | 教学主干无碍（关键在"两级错误分开"）；联网窗口对照 MCP 规范错误码表后裁决，若改码需同步教程与实录 | 留证：base_02.txt `unknown tool: multiply → -32601` | T0 裁决 |
