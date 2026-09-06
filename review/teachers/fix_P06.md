# 修改说明 · Part 6（Transformer/GPT）整改（T2）

> 整改人：T2 · 日期：2026-09-05
> 输入：plan_P06（T1 预审）+ S1/S2/S3 三份 report_P06 + 必修清单（T0 确认 11 项）
> 原则：REPO 只读；全部修改写镜像；脚本 07 默认档行为不变；无 wontfix，3 项 disputed

## 〇、关键实证先行（决定了修法方向）

**教程数字没有造假，是设备口径问题。** 用 seed=1337 在 CPU 单线程下实跑脚本 04：教程引用的亲和力矩阵 `[0.5141, 0.4859] / [0.3271, 0.3374, 0.3356] / [0.1136, …]` 及全部训练日志**逐位一致**（scratch/t2_P6/probe_affinity.py、cpu_04.log）；三位学生报告的 `[0.3141, 0.6859]` 是 `cuda:0` 路径数字（CUDA 的 randn/multinomial 序列与 CPU 不同）。因此必修 4 的修法是 **保留 CPU 数字 + 全教程标注实跑口径 + 更新真正漂移的日志**（见下），而非替换成 CUDA 数字。

## 一、镜像修改文件清单（12 个文件 + 2 张新图）

**教程（5）** `REVIEW/courses/Part6_transformer/tutorial/`
- `README.md` — 前置知识"均分梯度"措辞修正；**路线图终点改 Part 7 组件升级线** + Part8 出口段（C1-1）；演进表 LayerNorm 行 2.23→2.24、缩小型 2.80→2.79（对齐实跑）；**挂 loss_evolution.png**；表下补 ppl=exp(loss) 换算注（2.24→9.4 即指南 9-11）；学完清单补 ppl 条目
- `01_data_and_tokenizer.md` — 词汇表节补"动态词表换数据需重建"⚠️（A1①）；BPE 对比挂 [Part 7·01]；**get_batch 节补 device/block_size/batch_size 定义**（G10）+ 📌 节选说明；新增全教程"实跑口径声明"（CPU 单线程、seed=1337、torch 2.6）；bigram 训练日志按实跑更新（4.7707→2.5002）；200 字生成样例换实跑；作业映射补**题 3（Bigram）**（C1-3）
- `02_attention_from_scratch.md` — v3 节补全 -inf 行 NaN ⚠️；Head 前补 📌 全局变量说明（白板/作业版预告）；`tril[:T,:T]` 切片必要性（A6①）；**新增 `scaled_dot_product_affinity(q,k)` 模块级函数段**（C1-4，与作业题 4(a) 同签名）；亲和力演示加设备口径注；scaled attention 节**补三步 LaTeX 方差推导**（"为什么开平方"）+ **softmax_scaling.png** + 脚本 04 数值验证引用（Var 31.84→0.995）；笔记 6 回链推导；Multi-Head 补整除 ⚠️（A3①）；**新增「各头在学什么」观察实验小节**（4 头画像表，实测）；多头价值段挂 Part7·03（GQA/KV Cache）；300 字生成样例换实跑；作业映射改为只挂题 4（C1-3）
- `03_transformer_block.md` — 前置知识/正文/Q1/学完清单的**残差措辞 4 处修正**（"原样复制，不是除以 2"）；Phase2/Phase3 数字按实跑更新（2.5017/2.2299）；LN 数值例归因改"**测量口径**"（有偏=1.0 / 无偏测量=1.118，LaTeX）；LN 节挂 Part7·02（RMSNorm）；**新增「pre-norm 为什么稳？」小节**（两条机制 + 16 层实测：第 1 层入口梯度 pre 0.0316 vs post 0.0069）；FFN 挂 Part7·03（SwiGLU）；06 日志 step800/1199 更新、400 字生成样例换实跑；**超参表改"两种档位"**（0.112M/1.9s vs 10.789M/约 6.5 分钟）+ **SMALL=1 用法** + 零输出缓冲提示；缩小型日志/样例按实跑更新（终值 2.7885）；**新增「从 loss 到困惑度 ppl」小节**（6 行换算表 + math.exp 验证，C1-2）；生成节挂 KV Cache 钩子（Part7·03）；学完清单同步
- `04_beyond_transformer.md` — nanoGPT 走读后**新增「与原论文的 5 处不同」对照表**（每条标出处+面试记法）；RLHF 三步后挂 [Part 8] 出口（C1-5）；总结与展望加 **6 行 Part6→Part7 组件升级对照表** + Part7/Part8 链接；"一路降到 2.23（CPU 缩小型）"语义修正（2.24 / 缩小型 2.79）；完结段加下一站链接；学完清单补"5 处不同"

**脚本（3）** `REVIEW/courses/Part6_transformer/scripts/`
- `04_self_attention.py` — "为什么除以 sqrt(head_size)"段新增数值验证打印（独立 Generator，**不扰动主训练 RNG**：训练日志与原版 diff 逐位一致）；docstring ⑥ 补说明（11 行）
- `05_multihead_feedforward.py` — docstring 残差措辞 1 行（与教程统一，代码零改动）
- `07_scaleup_generate.py` — **`SMALL=1` 环境变量强制缩小型**（默认档 `CPU_MODE=not cuda` 判定不变）；docstring 写明三种模式；关键 print 全部 `flush=True`；GPU 完整版头部加一行 SMALL=1 提示（stdout 唯一差异，见 disputed D2）

**作业（1）** `REVIEW/assignments/assignment_6/assignment.md`
- 题 5 残差措辞同步修正（"原样复制，不是除以 2"）。test/exercises 骨架零改动（参考答案 + 测试实测 7 passed）

**图（2，新增）** `REVIEW/courses/Part6_transformer/images/`
- `loss_evolution.png`、`softmax_scaling.png` — 由 scratch/t2_P6/make_plots.py 用实测数据生成

**台账** `REVIEW/ledger/ledger_P06.md`（**19 条 fixed + 3 条 disputed**；P0=5 / P1=10 / P2=4）

## 二、验证记录（全部实跑，scratch/t2_P6/）

1. **脚本 01-03**（原版，改教程数字用）：CPU 单线程实跑通过；01 章统计/encode 输出逐位吻合；02 bigram 1500 步、03 三版本 allclose True×3。
2. **修改版 04**（CPU）：新增验证段输出 Var=31.84/std=5.643 → 0.995/0.998；**训练日志与原版 diff 逐位一致**（独立 Generator 不扰动 RNG）。
3. **修改版 07 SMALL=1**（CPU）：与原版 CPU 缩小型输出**逐位一致**（仅耗时行 1.8s/1.9s 差异）——112,193 参数、4.1655/4.1661 → 2.7885。
4. **07 默认档**（GPU 探针，25s 超时中断）：仍自动跑完整版（10,788,929 = 10.789M、5000 步），头部输出带 flush 立即可见；超参选择/训练代码 diff 确认未变。
5. **pre-norm 实验**：16 层/std=0.02/同输入反传——第 1 层入口梯度 pre-norm 0.0316 vs post-norm 0.0069（4.6×），exp_prenorm.py 可复现。
6. **各头观察实验**：脚本 05 Phase1 配置训练 400 步 + 200 val batch 统计——H1 自我 4.02×基线/前字符 0.03×，H3 前字符 1.86×/元音 1.30×，exp_heads.py 可复现。
7. **作业 pytest**：参考答案 + 镜像测试 → **7 passed in 0.78s**（S2 基线 7 passed 保持）；exercises 骨架已还原未改动。
8. **check_latex.py**：5 篇教程 + assignment.md 全部 **0 问题**。
9. **相对链接**：按 REPO 真实结构校验教程全部链接（Part7/Part8/脚本/互相引用）通过；2 张图片仅存在于镜像（合入 REPO 即解析）。
10. **py_compile**：修改的 3 个脚本全部通过。

## 三、Disputed 清单（详见台账）

- **D1** 教程日志统一为 CPU 基准+设备声明后，CUDA 学生仍有 ±0.03 漂移/生成文本不同（设备本质决定）：是否建议学生用 `CUDA_VISIBLE_DEVICES="" python -u` 复现教程数字，属课程口径，请 T0 定。
- **D2** 脚本 07 GPU 完整版模式新增一行 SMALL=1 提示 print（超参/训练/seed 逐位不变）：对"默认行为不变"口径的边界说明，不接受删一行即可。
- **D3** `for iter in range(...)` 遮蔽内置 `iter`（S2 风格项）：整改需触碰 7 脚本+教程全部训练循环且破坏"教程=脚本节选"对照，建议列入全局编码规范与 Part1-5 一并统一。

## 四、遗留（不在本 Part 范围）

- **Part 4 的"加法均分梯度"措辞**（S3 引为"最佳叙事"的源头）：Part6 侧已全部改正；Part 4 `02_forward_and_backward.md` 属另一 Part，建议其 T2 同步（回链处 README 前置已按新表述写）。
- 教程旧日志与当前实跑的漂移根因（01 章 bigram 日志、05 Phase2/3、06 后半段、全部生成样例）疑为历史 torch 版本的 randint/multinomial 序列差异，无法追溯原作者环境；已按"当前环境 CPU 实跑"统一对齐。
- roadmap L253"脚本 03/04 加权均值可视化"现为打印级：G4 两图已覆盖 loss 演进与 softmax 对比；亲和力热图属弱候选（T1 豁免判定项），未画。
