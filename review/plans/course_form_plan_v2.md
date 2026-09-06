# 课程形态优化计划 v2（形态规范 + 落地任务）✅ 已完成

> **版本**：v2.1（2026-09-05，T1-T10 执行完毕；T1 站点构建待联网装 mkdocs，其余全交付）｜ **上位计划**：[00_master_plan.md](00_master_plan.md)（审计战役 v1.3，已完成）
> **背景**：审计战役（468 条发现）收官后，基于"Gemini 优质课程清单 × 审计实证"的对照讨论，
> 确立课程形态优化方向。**本计划是新的执行锚点**：约束（F 规则）+ 落地任务（T1-T10）。
> **工作区**：不变 —— REPO 只读，一切写入 REVIEW（makemore-tutorial-review）。

---

## 一、约束裁决（Gemini 清单 × 审计实证）

| 来源 | 裁决 | 说明 |
|---|---|---|
| 规范 1（黑盒破壁） | ✅ 采纳（模板化） | 现状脚本先行已达标；新增 F4"3 分钟 Run 先行"框模板 |
| 规范 2（封杀孤立公式） | ✅ 已达标并强化 | G2/W6：公式必须 LaTeX + 数字算例 + shape 链；W21 图先于代码 |
| 规范 3（脚手架代码） | ✅ 已达标；Jupyter 双轨可选 | F8：.py 单一源，jupytext 同步 ipynb 为可选产物（不反向） |
| 规范 4（面试场景化） | ✅ 已达标；补话术框架 | F5：面试直通车升级为"结论→原理→工程边界"总分总话术卡 |
| **红线 1（推导 ≤3 步）** | ❌ **实证否决** | S1 的全部 🔴 都是"跳步"而非"推导长"；本课验收含白板手推。规则定为 F7：**推导不限步数，但每步必须三段式+数字例**（W9/G2） |
| 红线 2（史前技术） | ✅ 已达标 | 课程无 RNN/LSTM；Part1-5 每步为后续手写承重墙 |
| 红线 3（环境地狱） | ✅ 强共鸣并已制度化 | G8 toy 档/flush、G16、降级路径即消融（W5）、P10 FSDP 假承诺 P0；新增 F6 环境诚实 |
| 红线 4（工程思维） | ✅ 已达标；补算力经济学 | P9-P14 为核心阶段；新增"推理成本一页账"候选（P14 扩展） |

## 二、课程形态规范（F 规则，约束所有新形态产物）

| # | 规则 |
|---|---|
| F1 | **Markdown 单一事实源**。站点/HTML/动画全是渲染产物；任何交互件必须在对应教程 md 中有锚点引用与文字说明（无孤件） |
| F2 | **三层渐进**：L0 = Mermaid 图 + 静态站（零/低成本）；L1 = 单文件 HTML 交互小部件（iframe 嵌入）；L2 = 英雄动画（HTML canvas 优先，Manim 列备选，仅限 3-5 个全课程核心概念） |
| F3 | 交互件的说明文字同样遵守 G 系（G2 公式/G14 口径/数字例）；数据来自脚本实测的必须注明生成脚本 |
| F4 | 场景化引入模板：工具链章开头给"痛点场景（OOM/超时/算爆预算）→ 本章解法"，替代"今天我们讲 X" |
| F5 | 话术卡模板：每条面试直通车按「结论一句话 → 原理两三句 → 工程边界/代价一句」重写 |
| F6 | 环境诚实：Colab/一键运行 badge 只挂"免费档（CPU/单卡 16-24G）可完整跑通"的内容；其余给最低配置声明，宁缺毋假 |
| F7 | 推导规则：步数不限，三段式（前向关系→局部导数→代码）+ 每公式一个代入数字的算例（W9/W6）——数学为直觉服务，严谨性靠数字例落地 |
| F8 | 单步调试双轨：scripts/*.py 是唯一源；ipynb 只能由 jupytext 从 .py 派生，禁止手改 notebook 作为事实源 |
| F9 | 交互件零依赖：单文件、JS/CSS 全内联、无 CDN，离线可开（与 F6 同源） |
| F10 | 更新成本预算：每类新形态入课程前必须评估"内容变更时的同步成本"；数据图/交互件的数据优先由脚本生成（可复现，G6） |

## 三、落地任务与执行顺序

### P0 零成本（纯标准库 / GitHub 原生渲染）

| 任务 | 内容 | 产物 | 状态 |
|---|---|---|---|
| T1 站点 | **已交付（自研兜底）**：网络三路均阻（代理/镜像/直连），改用纯 stdlib 渲染器 tools/build_site_lite.py（110 页，KaTeX/Mermaid 浏览器端渲染、无网优雅降级）+ serve_site.sh（0.0.0.0 局域网服务，本机 IP 192.168.0.126:8000）。mkdocs-material 路线保留：联网后 `uv pip install mkdocs mkdocs-material` + tools/build_site.py 可升级 | tools/build_site_lite.py + tools/serve_site.sh | ✅ 完成（lite 版在线服务中；mkdocs 版待联网可选升级） |
| T2 Mermaid 试点 | 3 张高频 ASCII 图升级为 ```mermaid（GitHub/MkDocs 原生渲染，可 diff 可版本控制） | P18-01 管线图、P13-01 四阶段、P19-01 loop 时序 | ✅ 完成 |
| T3 面试直通车结构化 | 解析 19 套作业的"面试直通车"→ 结构化 YAML + CSV（为 T9 Anki 备料） | interview/cards.yaml + cards.csv + tools/extract_interview_cards.py | ✅ |

### P1 低成本（单文件 HTML / stdlib 脚本）

| 任务 | 内容 | 产物 | 状态 |
|---|---|---|---|
| T4 交互小部件 ×5（已完成，另加 T5 热力图） | ①LSH S 曲线（b/r 滑条）②softmax 温度 ③DDPM β/ᾱ schedule ④流水线气泡率计算器 ⑤KV Cache 显存增长 | widgets/*.html（单文件零依赖）+ 教程锚点 | ✅ |
| T5 注意力热力图 | P6 toy 模型注意力交互热力图（纯 canvas/内联数据，不依赖 BertViz 安装） | widgets/attention_heatmap.html | ✅ |
| T6 教材 CI 三件套 | check_latex（已有）+ check_links + check_assignment_refs（章末指引↔题号）+ GitHub Actions 模板 | tools/check_*.py + ci/workflow 模板 | ✅ |
| T7 话术框架全量 | 面试直通车"总分总"话术卡 | 19/19 Part、92 张卡（4-5 卡/Part），全部 check_latex 0 问题 | ✅ 完成 |

### P2 高成本（HTML 动画 / 结构化导出）

| 任务 | 内容 | 产物 | 状态 |
|---|---|---|---|
| T8 英雄动画 ×3（已完成，HTML canvas） | ①QK^T 注意力流动 ②DDPM 前向加噪/反向去噪 ③KV Cache 显存增长（HTML canvas 逐帧；Manim 列为维护者本地可选路线，成本理由见 F2/F10） | widgets/anim_*.html + 教程锚点 | ✅ |
| T9 Anki 导出 | 92 张话术版卡（正面=【Pxx】问法，背面=结论/原理/边界）；.apkg 为可选项 | interview/{cards.yaml,cards.csv,anki_import.csv} | ✅ 完成（话术版） |

## 四、验证与门禁

- 小部件/动画：`node --check` 提取的内联 JS 可过 + 文件单文件自洽（无外链）+ 对应教程 md 有锚点；
- Mermaid：mkdocs/GitHub 渲染器原生支持，语法用 mermaid.live 规则自查；
- 结构化数据：19/19 Part 覆盖、字段完整（part|question|answer_conclusion|answer_principle|answer_boundary|source）；
- CI 脚本在 REPO+REVIEW 合并树上跑通并给出首个基线报告；
- 全部产物登记 manifest.md；**REPO 依旧零写入**。

## 五、执行记录补充（2026-09-05）

- T7 全量执行时发现并修复一类新问题：**新写内容必须基于修复后底稿**——首批铺卡 agent 读了 REPO 未修复教程，继承陈旧数字（2.47/170K/3.7~4.0 语境错位）并覆盖了 P1-P5 已修复的作业 README（回退事故）；已按 fix 文档重放 18 个修复点 + 逐卡对照 REVIEW 口径修正。此教训并入 G6/G13 执行细则。
- 抽取器（tools/extract_interview_cards.py）支持话术卡/旧条目双格式；REVIEW 覆盖优先；前缀碰撞（-review vs -tutorial）已修。

## 五、未决/拿不准（已按默认执行，可随时推翻）

1. 站点栈默认 MkDocs Material（网络恢复即装）；若维护者偏好 Docusaurus/VitePress 可换壳，F1 不变。
2. 英雄动画默认 HTML canvas（本环境零依赖可交付）；Manim 效果上限更高但需维护者本地装环境，列为可选路线。
3. Anki 默认 CSV 导入（零依赖）；genanki 生成 .apkg 列为可选。
4. Jupyter 双轨（F8）未执行——等维护者确认是否需要 ipynb 产物再启动。
