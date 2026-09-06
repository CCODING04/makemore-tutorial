# fix_P14 — Part 14 整改报告（T2）

- 日期：2026-09-04 ｜ 依据：plan_P14（T1 预审）+ S1/S2/S3 report_P14 + T0 必修清单
- 原则：REPO 只读；全部修改写入 REVIEW 镜像；脚本实跑先 cp 到 scratch（G16）
- 台账：`REVIEW/ledger/ledger_P14.md`（P0=2 / P1=6 / P2=9）

## 一、镜像改动清单

| 文件 | 改动摘要 |
|---|---|
| `REVIEW/courses/Part14_inference_vllm/tutorial/02_vllm_serving.md` | P0-1 分页算式改错+非整除例（520→8 token 碎片）；新增"KV 显存账"小节（LaTeX 公式+0.5B/7B 数字例）；新增块表 ASCII 图+prefix sharing 数字例；新增投机解码验证数学（α^k(1−α)、E=(1−α^(γ+1))/(1−α)+数字例）；新增 §4.5 GGUF 命名规则；benchmark 路径 3 处补获取方式；错误 1 换 PyPI 直装+Python 3.9+ 口径；性能表显存列回填脚本实测 1.85/1.87 GiB+更正记录；Q3 量化精度补"经验值"出处口径；speculative_config 版本注 |
| `REVIEW/courses/Part14_inference_vllm/tutorial/01_naive_baseline.md` | "数学推导"+"可操作定义"转 LaTeX（G2×2）；形状追踪框公式外移；TPOT 双口径点破块；基线注补单次运行口径/CPU 降档方向预期/HF_HUB_OFFLINE 提示；基线输出补显存峰值两行+2026-09-04 复跑核对行；"三行对比表"命名口径注；引用 G4 实测图 |
| `REVIEW/courses/Part14_inference_vllm/tutorial/README.md` | 重复"数学推导"块删除改指向 01 章（G2+G15）；70%+ 补行业经验估计口径；方案 B 代价栏写明 torch 降级重装；HF_HUB_OFFLINE 提示；star 数标"撰写时点参考"（2 处） |
| `REVIEW/courses/Part14_inference_vllm/scripts/01_naive_generate_baseline.py` | 加 `torch.cuda.max_memory_allocated()` 显存峰值输出（逐请求/静态批各一处，注释写明口径）；docstring 补离线提示；`line_buffering=True`（flush 问题）；末尾面试话术删"三行"字样 |
| `REVIEW/courses/Part14_inference_vllm/images/throughput_naive_vs_batch.png` | 新增 G4 图：naive 158 vs 静态批 1071（实测）vs vLLM ~3000（预期，斜纹标注），4090 实测数据 |
| `REVIEW/assignments/assignment_14/assignment.md` | G1：Q5 参考答案 ①②③ 逐条成行 |
| `REVIEW/ledger/ledger_P14.md`、`REVIEW/teachers/fix_P14.md` | 本报告与台账 |
| `REVIEW/outline_review/outline_suggestions.md` | 追加 3 条：roadmap"3 编码题"→4+1 stretch、"面试直通车 4 问"→5 条、"三行对比表"术语同步（docs 侧） |

## 二、必修清单逐项核销

1. **02 章显存列"✅ 实测"失实** → 补可产出实测：脚本加显存打印，scratch 实跑真值 1.85/1.87 GiB 回填，表注留更正记录 ✅
2. **benchmark_serving.py 3 处悬空** → 3 处均补 clone 获取方式 + `vllm bench serve` 等价命令 ✅
3. **错误 1 cu118 过时/矛盾** → PyPI 直装与 README 方案 A 统一，Python 口径 3.9+，官方安装页为唯一权威 ✅
4. **分页算式错步** → 公式改 $\lceil s/B\rceil\times B - s$，主例 520→33 块→8 token ≈1.5%，512 标整除特例 ✅
5. **公式补齐（G2 LaTeX）** → KV 显存 $2 L H_{kv} d_{head} s b$ 与投机解码 $\alpha^k(1-\alpha)$、$E=(1-\alpha^{\gamma+1})/(1-\alpha)$ 均提进正文并配数字例 ✅
6. **块表图/TPOT 双口径/重复推导删/HF_HUB_OFFLINE** → 四项全落 ✅
7. **GGUF 命名小节** → 02 章 §4.5（Q4_K_M 拆解+位宽速查+面试话术），roadmap 悬空目标兑现 ✅
8. **格式规范** → G2×4 转 LaTeX（check_latex 全 0 问题）；G4 新增实测对比图；G13 两处无来源数字补出处口径+star 时点；G1 Q5 逐条成行 ✅
9. **T1 的 4 个 C1 缺口** → C1-2 教程补 GGUF；C1-4 教程侧加命名口径注；C1-1/C1-3 属 docs → outline_suggestions.md 追加 ✅

## 三、验证结果（全部实跑）

1. naive 脚本（scratch，HF_HUB_OFFLINE=1，4090，exit 0）：**150 tok/s / 8.0ms / 6.6ms / 1005 tok/s** vs 教程 158/7.5/6.2/1071，偏差 −5~7%（±7% 内）→ 教程数字与口径声明成立；显存真值 1.85/1.87 GiB。
2. 作业：reference 参考答案 + test（scratch）→ **pytest 5/5 passed**。
3. check_latex.py：三个改动 md 两轮复跑 → **0 问题**。

## 四、通用问题（跨 Part）与好写法候选

**通用问题（≤3）：**
1. "实测"标注缺产出机制：性能表数字一旦没有对应脚本的打印项支撑就会腐化（本 Part 显存列即例）——建议全课程约定"每个 ✅ 实测数字必须能指认产出它的脚本行"。
2. 装机类教程的 CLI 引用需写"获取方式"：`benchmarks/benchmark_serving.py` 这类不随 pip 分发的脚本，相对路径引用必悬空；凡引用外部仓库文件应给 clone/子命令双路径。
3. 公式推导放代码块是课程级惯性（本 Part 4 处 + 形状追踪框），建议把"G2 公式必须 LaTeX"写进生成模板的硬约束。

**好写法候选（≤3）：**
1. 01 章"吞吐口径修正"自曝段（早期把 TTFT 探测计入分母、系统性低估 5-10%）——把错误摆上台面并给修正方向，是测量教学范本（三学生共同首选）。
2. 脚本 01 的"分子分母同口径"设计 + 7 处 synchronize 每处写明原因——口径声明与实现逐行对得上，可复现性强的直接原因。
3. 02 章 §6"手写 → 工具"总账对照表——把 P7/P8/P9/P14 串成"懂原理 + 会工具"一条线，是跨 Part 互证的样板结构。
