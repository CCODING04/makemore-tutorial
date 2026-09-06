# fix_P15 — Part 15 多模态理解（VLM）整改报告（T2）

- 日期：2026-09-04 ｜ 依据：plan_P15（T1 预审）+ S1/S2/S3 report_P15 + T0 必修清单
- 原则：REPO 只读；全部修改写入 REVIEW 镜像；脚本实跑先 cp 到 scratch（G16）
- 台账：`REVIEW/ledger/ledger_P15.md`（P0=2 / P1=7 / P2=11，无 disputed）

## 一、镜像改动清单

| 文件 | 改动摘要 |
|---|---|
| `REVIEW/courses/Part15_vision_language/scripts/01_vit_projector_pipeline.py` | P0-2：Stage 2 `trainable2` 收入 `patch_embed`（312 参数）→ "全部解冻"名副其实，实测打印 **29,620 / loss → 0.034**（Stage 1 与 loss 数值完全不变，可复现性未破坏） |
| `REVIEW/courses/Part15_vision_language/scripts/02_clip_siglip_alignment.py` | P1-2/P2-9：打印改"学到的 scale = 16.21（温度 τ ≈ 0.062）"（数值不变、语义诚实）；`logit_scale` 注释改"可学习 scale=1/τ；玩具初始 scale=10（CLIP 默认 τ=0.07 → scale≈14.3）" |
| `REVIEW/courses/Part15_vision_language/tutorial/01_handwritten_projection_vlm.md` | P0-2：形状账本框/实测块/性能表三处 29,308→**29,620**+组件分解；P1-1：练习 2 提示改 `Linear(d_v, d_l)`，推导节新增"mlp2x 的 2x = 2 层 MLP 非 2 倍宽"防坑块；P2-1：推导整段代码块转 LaTeX（含 Projector 复合函数公式）；P1-4：~2h/~10h 改"社区复现口径"+表注声明论文未报时长，时长列口径统一；P1-5：最佳实践表标"LLaVA 真实口径"+两档 lr 对照注；P2-7：实测块补 seed=1337/CPU 浮动（2.84→1.84）；P2-8：引用两张实测图 |
| `REVIEW/courses/Part15_vision_language/tutorial/02_alignment_losses_and_schemes.md` | P1-6：新增"数学定义"段（InfoNCE 行列 softmax 双向 CE、SigLIP $-\frac{1}{N^2}\sum\log\sigma(T_{ij}S_{ij})$ LaTeX 公式 + SigLIP bias 省略声明 + s=1/5/10 数字例）；P1-2：全文统一 scale=1/τ 记号（温度段重写/陷阱 2/学习目标）；P1-3：性能表后两行删"论文报告值 ~95%/1h/30min"改量级示意+表注（附 CLIP 256×V100×12 天参照）；P1-1/P2-5：细节差异点拆行；P2-2/P2-7/P2-9：行内代码转 $、seed+双设备声明、错误 1 补玩具初始化注；P2-8：引用损失曲线图 |
| `REVIEW/courses/Part15_vision_language/tutorial/README.md` | P2-4：datasets.md §5→§4；P2-6：star 数标"2026-09 撰写时点参考"；P2-7：实测声明补 seed 与"数字随设备略有浮动"（删"已逐项核对一致"的绝对化表述） |
| `REVIEW/courses/Part15_vision_language/images/`（3 张，新增） | G4：`two_stage_loss_curve.png`（两阶段 2.907→1.825→0.034，逐点实测）、`infonce_siglip_curve.png`（4.155→1.968 / 0.912→0.106）、`patch_projection_flow.png`（形状链+312/9,648/1,856/17,804 参数账） |
| `REVIEW/assignments/assignment_15/assignment.md` | P0-1：题 3 验收数字 21,043,712→**20,979,712**（test 动态算公式，不需改）；P1-7：实验题加"预期管理"块（玩具观察不到差异≠跑错 + log N 下界提醒）、Q3 参考答案改"理论预期+玩具实测诚实修正"双段；P1-2：Q4 的 τ≈16/8.5 改为 scale 口径并互指记号约定；P2-3：Q2 Σ 公式转 LaTeX |
| `REVIEW/ledger/ledger_P15.md`、`REVIEW/teachers/fix_P15.md` | 本报告与台账 |
| `REVIEW/outline_review/outline_suggestions.md` | 追加 2 条（P2-11/C1-1，docs 侧）：缺口表"归档"失真、节点 15 ≠ Part 15 同号错位 |

## 二、必修清单逐项核销（T0 十项）

1. **作业题 3 数字（P0）** → md 改 20,979,712 并附分解；test 经查为公式动态计算、不含硬编码，只修题面；修复后 pytest 4/4 实证 ✅
2. **参数账** → 选"脚本修复"路线：trainable2 收入 patch_embed，实测 29,620/0.034，教程三处同步组件分解；"全部解冻"现已名实相符 ✅
3. **练习 2 projector 口径** → 以脚本实测（1,856，d_l 隐层）为准统一；练习 2 提示改正 + "2x=两层"防坑块 ✅
4. **τ 双语义** → 全文统一 scale=1/τ 并显式声明约定（02 章温度段）；脚本打印/assignment Q4 对齐；log(1/0.07) vs log(10) 两档注明（O1 一并）✅
5. **公式体系补齐（G2）** → InfoNCE/SigLIP LaTeX 定义进 02 章正文 + bias 省略声明 + s=1/5/10 数字例（0.610/0.993/1.000，验算过）✅
6. **无出处数字与 lr 档位** → 02 章"论文报告值"改量级示意+声明非论文值；01 章 ~2h/~10h 改社区口径；lr 表标真实口径+玩具两档对照注（G14-2）✅
7. **两道实验题预期** → 预期管理块 + Q3 参考答案诚实修正（保留"谁动 vs 动多快"的教学内核）✅
8. **README §5 断链** → 改 §4 ✅
9. **G1-G16 全量** → G1×1、G2×5、G4（3 张实测图）、G10×3、G13×3、G14×2 全部落；O1/O2 一并处理 ✅
10. **C1 四缺口** → C1-2（题 3）、C1-3（练习 2）、C1-4（§5）课程侧已修；C1-1（roadmap）转 outline_suggestions.md 追加 2 条 ✅

## 三、验证结果（全部实跑）

1. **基线（G16）**：原版两脚本 scratch 实跑一次通过，9 个实测数字逐字复现。
2. **镜像脚本**：CUDA + CPU 双轮复跑——Stage 1/脚本 02 数值逐字不变，Stage 2 升为 29,620/0.034（两设备一致）；时长实测 2.24s / 1.5s 支撑性能表。
3. **作业**：reference 参考答案 + test → **pytest 4/4 passed**（题 3 数字修正后仍 PASS）。
4. **check_latex.py**：4 个改动 md 全部 **0 问题**。
5. **图片**：3 张图重生成两轮，曲线数据与脚本输出逐位对齐（脚本 02 数据生成前补 seed(1337) 修正过一次 RNG 漂移）。

## 四、通用问题（跨 Part）与好写法候选

**通用问题（≤3）：**
1. **Part 编号与 roadmap 节点编号是两套坐标系**（本 Part "节点 15=毕业验收"与"Part 15=VLM"同号错位，缺口表又写"归档"）——学习者按路线图推进将整体错过已建成的 Part。建议终版给 roadmap 加"节点≠Part 编号"对照说明并刷新缺口表状态。
2. **"论文报告值"无出处标注是惯性**（本 Part 02 章 ~95%/1h/30min 量级失真、01 章 ~2h/~10h 归因论文）——建议全课程约定：引论文的数字必须能给论文名+口径，给不出就标"社区复现口径/量级示意"，性能表禁用"论文报告值"字样兜底。
3. **练习提示代码与脚本/作业的口径漂移**（练习 2 的 d_l×2 全课程独一份，照抄即与题 3 参数账冲突）——练习是学生动手入口，建议约定"练习提示的结构必须与脚本同名类逐字对齐，并注'以脚本为准'"。

**好写法候选（≤3）：**
1. **形状账本 + 脚本内联 shape 断言**（三学生共同首选）：教程每步 shape 与脚本 assert 逐行互证，跑一遍等于对一遍账——本 Part 数字可复现性全课程领先的根本原因。
2. **InfoNCE vs SigLIP 双列对照表 + "N 选 1 单选题 vs N² 道独立判断题"直觉**：一页纸讲清选型与 batch 依赖，面试可直答（S3 白板默写全过的素材）。
3. **参数账叙事链**：1,856 → 29,620 → LLaVA 7B 投影器 ~21M 占 ~0.3% → 对照 Part 8 LoRA 3.4%——把"为什么拼接式便宜"讲成可复算的算术题，跨 Part 互证。
