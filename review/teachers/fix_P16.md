# fix_P16 — Part 16 整改报告（T2）

- 日期：2026-09-04 ｜ 依据：plan_P16（T1 预审）+ S1/S2/S3 report_P16 + T0 必修清单
- 原则：REPO 只读；全部修改写入 REVIEW 镜像；脚本实跑先 cp 到 scratch（G16）
- 台账：`REVIEW/ledger/ledger_P16.md`（P0=1 / P1=6 / P2=18 / disputed=4）

## 一、镜像改动清单

| 文件 | 改动摘要 |
|---|---|
| `REVIEW/courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.md` | Step 2 省略号兑现（t=2 两高斯合并 α₂β₁+β₂=1−ᾱ₂ + 数值例）；反向均值公式补后验配方 μ̃_t + x₀ 代回两行追溯；VLB→MSE 三步简化段（KL 闭式→ε 重参数化→"噪声加权谱"置 1）；「常见陷阱」节删除与「调试展示」合并（G15）；陷阱 2 重写为「设计选择：β_t vs β̃_t」（纠正"采样方差影响训练 loss"错误归因）；性能表分口径（线性=实测 / cosine=论文声称无对照 / 真实图像=论文引述 CIFAR-10 ~78 万步）；最佳实践表补 T=1000 vs 脚本 T=400 呼应注；下标 0/1-indexed 记号约定块；章头统一 "CPU <30 秒"；作业链接升级文件级；引用 G4 过程图 |
| `REVIEW/courses/Part16_image_video_generation/tutorial/02_t2i_i2i_pipelines.md` | "8× 空间压缩"四处改双口径（每边 8×/面积 64×，通道 3→4 增维，净 ≈48×）+ 口径提示块；练习 1 idx 映射去黑盒（两套时间线线性换算注释 + 教学近似声明）；pip 行钉 `diffusers>=0.31` + 口径注；§2 补 SD1.5 下载命令示例 + 回指 README 权重三步；作业链接升级文件级 |
| `REVIEW/courses/Part16_image_video_generation/tutorial/03_alignment_and_video.md` | §3 表格骨干列按三家分列（Latte 式 temporal attention / CogVideoX 3D full attention / Wan2.1 全注意力+flow matching）；"最小增量视角"标注仅对 Latte 式严格成立；新增「因果性在哪一层？」小节（VAE 层因果 ≠ attention 层因果）；概念检验 Q2 重写（因果分层，消除突袭）；练习 1 措辞改"用 [3] 号实验思想、(1,512) 布局复算" + 构造性结果定性说明（A-2）；"24GB 实测路径"改"选型路径（官方口径，本课未实测）"；练习 2 补 num_frames=49 的 4n+1 约定；作业链接升级文件级 |
| `REVIEW/courses/Part16_image_video_generation/tutorial/README.md` | 新增「权重下载三步」块（下载什么/命令/HF_ENDPOINT 镜像/缓存位置/datasets.md 链接）；学习目标"配置推理服务"降为"使用 diffusers 跑通推理"；star 数标"撰写时点参考"；pip 行钉版本；作业链接升级文件级 |
| `REVIEW/courses/Part16_image_video_generation/scripts/02_alignment_mechanisms.py` | 补 scale=0 解耦性直接断言（`assert torch.allclose(out_s0, txt_only, atol=1e-6)`）；尾注 temporal attention 表述加 Latte 式限定（与 03 章口径统一）；实跑数值行零回归 |
| `REVIEW/courses/Part16_image_video_generation/images/ddpm_forward_reverse.png` | 新增 G4 图：双月环真实数据前向（t=0→399 加噪，标 ᾱ 衰减）+ 反向（采样真实快照 t=399→0）两行十面板 |
| `REVIEW/ledger/ledger_P16.md`、`REVIEW/teachers/fix_P16.md` | 本报告与台账 |
| `REVIEW/outline_review/outline_suggestions.md` | 追加 4 条：roadmap v3 无节点 16（C1-1 重大）、SNR/动态分辨率两题无参考答案支持（C1-2/3）、视频无作业落点（C1-4）、设计文档运行时长口径（A-5 docs 侧） |
| `REVIEW/manifest.md` | 追加本 Part 逐文件条目 |

（assignments/assignment_16 三件**零改动**：作业经 S1/S2 双学生实测 4/4 一次通过、S1 评分 90，无需整改；T1 建议的补题因无参考答案支持记 disputed。）

## 二、必修清单逐项核销

1. **作业链接 404（P0）** → 复核为误报：`assignments/assignment_16/` 实际存在、四处相对路径 ls 全部可达（S3 审计材料缺 assignments 所致）；仍将四处目录级链接升级为文件级 assignment.md 消除渲染歧义 ✅；T1 建议补两题无参考答案支持 → disputed D1/D2 留 T0 ✅
2. **01 章前向闭式推导补步** → t=2 显式合并两三行 + 数值例；反向均值公式补后验配方来源两行；VLB→MSE 补三步简化一段 ✅
3. **陷阱 2 因果错配** → 重写为"设计选择：β_t vs β̃_t"（明确采样方差不参与训练）；「调试展示/常见陷阱」重复段合并为一 ✅
4. **权重下载三步指引** → README 主块（下载什么/命令/缓存位置）+ 02 章 §2 命令示例 + 挂 docs/datasets.md ✅
5. **02 练习 1 idx 去黑盒 / 03 章 temporal 因果统一 / Q2 重写 / 8× vs 48× / 100K+ 步出处** → 五项全落 ✅
6. **脚本 02 补 scale=0 直接断言** → 已加并实跑零回归 ✅
7. **下标约定挑明 + 性能表 cosine 行注明无对照** → 两项全落 ✅
8. **格式规范 G1-G16 全量 + G4 图** → 见下节；G4 新增真实数据过程图 ✅
9. **T1 的 5 个 C1 缺口逐条处理** → C1-1 → outline 追加；C1-2/C1-3 → disputed（无参考答案支持）；C1-4 → disputed + outline；C1-5 → 已修（链接+版本钉）✅

## 三、格式规范（G1-G16）全量结论

| 规则 | 结论 |
|---|---|
| G1 结构完备 | 三章+README 结构齐全，学习目标与正文措辞经本轮对齐（P2-10/P1-5）→ 通过 |
| G2 链接有效 | 13 条跨目录链接 ls 全可达（含新增 2 条）；check_latex 0 问题 |
| G4 图表 | 新增 1 张实测过程图（真实数据、seed 可复现、生成脚本留 scratch）；9 处 ASCII/公式裸围栏按豁免口径结案（P13/P14 先例） |
| G8/G10/G13/G14 | 作业链接 4 处文件级可达；概念检验（3/4/3 题）与动手实践齐；13 张表分隔行/列数完好 |
| G15 单一事实源 | 01 章重复双节合并；教程-脚本-作业三方签名口径既有"签名说明"块维持 ✅ |
| G16 | 本轮所有脚本执行均先 cp 到 scratch/T2_P16/（baseline/mirror 分离）✅ |

## 四、验证结果（全部实跑）

1. **基线**（REPO 原版 → scratch/T2_P16/baseline）：脚本 01 loss **1.1148→0.2098**、均值偏移 **[0.069, −0.011]**、方差比 **[1.039, 1.065]**——教程数字 100% 复现（逐位）；脚本 02 **1.252/2.505、1,600、0.998** 与教程/练习一致。
2. **镜像脚本 02**（scratch/T2_P16/mirror）：新断言通过、数值行 diff 零差异。
3. **作业**：参考答案 + test → **pytest 4/4 passed**。
4. **check_latex.py**：4 个改动 md → **0 问题**。
5. **链接**：ls 逐条核对全 OK（见台账验证记录 5）。

## 五、通用问题（跨 Part）与好写法候选

**通用问题（≤3）：**
1. 学生审计的材料包必须含 assignments/：S3 因拿不到题面把"目录不存在"误判为 P0、作业打 0/5 并拉低总分——材料完整性直接决定审计结论有效性（本 Part 最大教训）。
2. "实测"与"引述"混标是本课程惯性（本 Part 性能表/24GB 档位/星数三处同病）：凡非脚本可复现的数字应一律标口径（论文声称/官方 README/撰写时点），否则 S2/S3 类"来源存疑"卡点会反复出现。
3. 概念检验题在正文零铺垫处"突袭"（03 章 Q2 因果）：检验题的每个概念应能在正文找到同口径段落——写章时检验题应随正文同步改，而不是沿用旧版。

**好写法候选（≤3）：**
1. 01 章"签名说明"块（教程三参 vs 作业四参提前声明"数学完全相同"）——教程/脚本/作业三方接口对齐，S1/S2 双学生零踩坑，全教程最贴心的一段。
2. 02 章 strength="01 章闭式的第二次消费"叙事 + 可逐位复现的 √ᾱ 参考表——章节咬合与"直觉→公式→数字"三件套的满分示范（S1/S2/S3 三人共同点名）。
3. 脚本 02 的断言内联验收（shape/线性性/CFG 余弦阈值）——"教程声称的实测数字"全部有脚本行背书，是"每个 ✅ 必须能指认产出它的脚本行"这一约定执行的最好的 Part。
