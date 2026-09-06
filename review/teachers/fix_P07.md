# 修改说明 · Part 7（minimind 复现）整改（T2）

> 整改人：T2 · 日期：2026-09-05
> 输入：plan_P07（T1 预审）+ S1/S2/S3 三份 report_P07 + 必修清单（T0 确认 9 项）+ T1 C1 缺口 10 项
> 原则：REPO 只读（终检确认 09 mtime 与 temp/ 均未动）；全部修改写镜像；脚本默认档行为不变；无 wontfix，3 项 disputed

## 〇、关键实证先行（决定修法）

1. **脚本 09 bug 双实证并复现**：scratch 中拷贝 REPO temp 旧 ckpt（max_position_embeddings=128/73/256 混杂，SFT=73）跑原版 09，逐字复现 S2/S3 报告的 `RuntimeError: shape '[128, 1, 48]' is invalid for input of size 3504`（3504=73×48）。修法 = 评测窗口与位置表对齐（`seq=min(seq, config.max_position_embeddings)`），不是重建 freqs_cis。
2. **"CPU 缩小版初始 loss"取决于 temp 里的 tokenizer**：REPO temp 的 bpe_tokenizer.json 已是 6400 词表（S1 实跑产物），CPU toy 档 06 实测 step0 loss **8.7616≈ln6400**（不是 S1 推测的 5.5）。教程 CPU 档按"≈ln(当前词表)：65→4.2 / 258→5.5 / 6400→8.8"表述，避免写死一个数再被打脸。
3. **S3 的实测数字全部可信**：11（ppl 四方案逐位一致）、12（1.07/0.27/0.08GB、7.0%、NSA 2.3101）本轮复跑再次逐位命中——我的整改未破坏任何实测资产。

## 一、镜像修改文件清单（16 个文件）

**教程（7）** `REVIEW/courses/Part7_minimind/tutorial/`
- `README.md` — 导航表：04 行挂 `09`、05 行挂 `10`、06 行标"选修"（C1-1/6a 课程侧）；数据行改"01–09、11 读 input.txt；10 随机张量、12/13 合成"（C1-7）；演进表 GQA 行双口径 + 表下新增「minimind 配置口径统一约定」块（26M=8Q/2KV/θ1e6/eps1e-6/intermediate1600 为唯一事实源，必修1）；loss 表头改"GPU 课程模板档" + ⚠️ 块拆 CPU toy 档/GPU 档（G14-2）；依赖表加 CUDA OOM/P7_STEPS 提示（S2 卡点8）
- `01_bpe_tokenizer.md` — **核心代码块改写为脚本真实代码**（2 个特殊 token、unk_token=None、ByteLevelDecoder、initial_alphabet）+ "老资料 3 token"警示（必修4/S1🔴2）；预期输出改 **im_start(0)/im_end(1)** + GPU 档声明 + 实测数字（325,208/3.430x）+ 📝CPU 档说明（G14-1）；chat 节"3 个特殊 token"改 2 个/id 0-1；平局规则拆训练侧/编码侧并补 **rank 定义**（S1🟢12，题 1 落点）；"To be"两版统一 18 字符（S1🟢15）
- `02_modern_components.md` — 学习目标落点改 05 章（C1-2）；**残句收束 + 补相对位置一步 LaTeX 证明**（G10-2，S1/S3 共同诉求）；RMSNorm 定义/旋转矩阵/freq 公式 LaTeX 化（G2）；rope θ 双口径（S1🟡8）；eps 双口径 📝（必修7）
- `03_gqa_and_ffn.md` — GQA 8Q/4KV 全部改"课程实现"口径 + 官方 26M=8Q/2KV 注（含缓存表补 8/2=17MB 行，C1-3/必修1）；KV Cache 复杂度改严格口径（K/V 重算 $O(N^2L)→O(NL)$、**定义 L**、注意力读取仍 O(N²) 提醒，必修6）+ 学完清单同步；**Flash Attention 重复引注删一处**（C1-8）；silu 三处"0~1"改"平滑软门控、不限幅"（必修6）；ceil/round 改"通常一致、512 时不等（1664 vs 1600）"（必修6）；**MoE 节整段对齐脚本 04**（top_k=2/intermediate_size/Switch aux loss LaTeX，官方 4 专家 top-1 归属 05 章，145M→**198M-A6xM**，必修2）
- `04_training_pipeline.md` — 预训练示意图去掉 `<|im_start|>`（C1-4）；AdamW 片段 lr 改 3e-4+档位注（G14-6）；**预期输出拆 CPU toy 档/GPU 模板档**（含实测 8.7616 锚点，G14-2）；`build_chat_sample`→**`make_chat_tokens`**（代码块按脚本 07 重写+span/mask 等价声明，必修5）；BT/L_DPO 公式 LaTeX 化（G2 优先裁决项）；DPO β 三口径 📝（G14-3）；`pretrain_hidden.pth`→`out/pretrain_512.pth`（S1🟢13）
- `05_reproduce_minimind.md` — **intermediate_size 算例改"512→1600（2432 是 hidden=768 的值）"**（必修1/C1-9）；配置表 moe 行加"教学脚本 top-2"注（必修2）；"字符级 256"口径注（G14-5）；**嵌入 `../images/output_long_context.png`**（G4，png 归位 images/）
- `06_attention_mla_nsa.md` — MLA 账本改"课程账本口径（LLaMA-7B 量级 32 头…非 V2 官方端到端数字）"（C1-10/必修6）；"**课后练习**"升 `##` 标题 + 📝 课后作业节声明选修章不挂 assignment_7（C1-5）

**脚本（8）** `REVIEW/courses/Part7_minimind/scripts/`
- `09_eval_demo.py` — **shape bug 修复**：`heldout_ppl` 的 `seq = min(seq, model.config.max_position_embeddings)`（评测路径 seq_len 与 freqs_cis 对齐）；docstring 教程落点 05 章→04 章末（C1-7a）。**默认行为（max_pos≥128 的 ckpt）完全不变**
- `06/07/08_*.py` — G8：`reconfigure(..., line_buffering=True)`（日志逐行即时刷新）+ 环境变量 `P7_STEPS` 覆盖步数（**默认档不变**）+ docstring 说明
- `03_gqa_kv_cache.py` — 复杂度打印与 docstring 两处改严格口径（与教程 03 章同步）
- `04_swiglu_ffn_moe.py` — docstring "minimind 用 8 experts/top-2"改官方归属声明（4 专家 top-1 见 05 章；本脚本教学 4/2）
- `12_mla_nsa_accounting.py` — Part A 注释改"latent 维取 V2 的 kv_lora_rank=512/rope 64；基线为 LLaMA-7B 级 32 头口径"

**作业（1）** `REVIEW/assignments/assignment_7/assignment.md`
- 题 2 加 eps 双口径 📝；题 4 背景/Q3/Q5 表格 8Q/4KV 全部改双口径（官方 26M=8Q/2KV）。`minimind_exercises.py`/`test_*.py` 零改动（参考答案+test 实测 7 passed）

**图（1，归位）** `REVIEW/courses/Part7_minimind/images/output_long_context.png`（自 REPO scripts/ 归位；合并回 REPO 时请将 png 移至 `courses/Part7_minimind/images/`）

**台账** `REVIEW/ledger/ledger_P07.md`（**31 条 fixed + 3 条 disputed；P0=12 / P1=16 / P2=3**）

## 二、验证记录（全部实跑，G16：先 cp 到 scratch 再跑）

1. **09 加载路径**（scratch/mirror，含旧 ckpt）：原版复现崩溃（错误串与 S2 报告逐字一致）→ 修后 **EXIT=0**，三阶段 ppl/生成/预期行为对照全部输出（旧 ckpt 权重质量差属 disputed D3，非脚本问题）。
2. **09 迷你路径**（scratch/mini，无 temp）：**EXIT=0**，三阶段现场迷你训练 ppl 10.24/10.75/12.21。
3. **06 短程档**：CPU（`CUDA_VISIBLE_DEVICES= P7_STEPS=10`，step0 loss 8.7616）与 GPU（`P7_STEPS=5`，63.9M、step0 8.9141≈教程锚点 ln6400=8.76）双档 **EXIT=0**，日志逐行落盘；07（P7_STEPS=20）/08（P7_STEPS=10）EXIT=0，08 打印可见 beta=1.0 演示口径与 SFT 正则 5.63。
4. **11/12 复跑**：ppl 四方案 **5.00/6.37、14.06/13.35、5.08/5.20、5.27/5.05 逐位一致**；12 的 1.07/0.27/0.08GB、7.0%、NSA max diff 2.3101 一致——教程 05/06 章实测资产未被整改破坏。
5. **pytest**：参考答案（assignment_07）+ 修复后 test → **7 passed in 0.60s**。
6. **check_latex.py**：8 个改动 md（README/01-06/assignment.md）全部 **0 问题**（列表内公式均已行内化、`$$` 单行闭合、无中文入式）。

## 三、通用问题（供全课程规范）

1. **"同一名词、两个模型"**：本 Part "CPU 缩小版"曾同时指 26M 叙事档与 0.3M toy 档。凡教程给"预期输出"，必须注明**档位**（哪个配置、多少步、什么设备）；跨档数字（8.8→2.0 vs 5.5→?）不可混排在同一表头下。其他 Part 预计同样存在（P06 已按设备口径处理过一轮）。
2. **教程代码块与脚本漂移无机制**：`build_chat_sample`/特殊 token/MoE 签名三案同源——教程块是早期版本，脚本后来改了，教程没跟上。建议：教程代码块头部统一加"节选自 scripts/0X（2026-09 核对）"，重大重构时 grep 教程引用的函数名。
3. **官方口径必须带版本限定词**：minimind 有 26M/64M/MoE 多个配置，"minimind 用 X"式断言在本 Part 造成 6 处冲突。建议全课程对"官方项目"表述默认加版本/规模限定（对 GPT-2/Llama 等同理）。

## 四、好写法候选（范本）

1. **05 章开篇"验证边界（诚实声明）"**：事实已核验/端到端未自动验证的边界声明 + 成本现实表——外部事实密集章节的最佳实践，建议推广到 06 章 MLA 数字与任何"官方数字"章。
2. **02 章"📝 脚本实现对照"框**：主动声明"教程用实数版讲原理、脚本用复数版、数学等价、作业两种都收"——本轮所有双口径标注都沿用此范式（MoE/eps/θ/KV 头数）。
3. **assignment.md 题 4 的 ⚠️ 维度约定说明**：明确作业与脚本的布局差异并指定以测试为准——"防 shape error 于未然"的出题范式（S2/S3 双重点名表扬）。

## 五、遗留与移交

- **disputed 3 项**（见台账）：D1 roadmap 导航口径（docs 范围）＋迷你版验收落差；D2 minimind master 口径未联网复核（05 章权威地位保持）；D3 temp 旧 ckpt 清理决策。
- **合并回 REPO 提醒**：①`scripts/output_long_context.png` → `images/`；②08 短程档跑动会在 `courses/Part7_minimind/temp/` 覆盖 `ckpt_sft.pt`（本轮验证全程在 scratch，REPO temp 已终检未动）。
