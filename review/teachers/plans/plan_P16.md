# Plan P16 — Part 16（图像/视频生成）整体预审计划（T1 主教）

- 审计对象：`REPO/courses/Part16_image_video_generation/`（tutorial 4 个 .md 共 964 行 + scripts 2 个共 263 行）+ `assignments/assignment_16/assignment.md`（110 行）
- 参照基线：`docs/course_roadmap_v3.md`、`docs/part15_16_multimodal_design.md`（Part 16 节）、`docs/course_roadmap_v2.md`（生成侧进阶行）
- 预审方式：只读通读 + 数学逐式人工互证，不改文件不跑脚本

## 0. 审读覆盖记录（已完成）

| 文件 | 行数 | 状态 |
|---|---|---|
| tutorial/README.md | 93 | ✅ 通读 |
| tutorial/01_ddpm_from_scratch.md | 397 | ✅ 通读 + 数学逐式核对 |
| tutorial/02_t2i_i2i_pipelines.md | 239 | ✅ 通读 + strength 数学核对 |
| tutorial/03_alignment_and_video.md | 234 | ✅ 通读 + 对齐机制核对 |
| scripts/01_ddpm_from_scratch.py | 137 | ✅ 通读 + 与教程互证 |
| scripts/02_alignment_mechanisms.py | 126 | ✅ 通读 + 与教程互证 |
| assignments/assignment_16/assignment.md | 110 | ✅ 通读（目录内另有 generation_exercises.py / test_generation_exercises.py / __pycache__，未审内容） |
| docs/course_roadmap_v3.md | 节点 16 相关 | ✅ 已查（见 C1-1） |
| docs/part15_16_multimodal_design.md | Part 16 节 | ✅ 通读 |

## 1. C1 对应表（roadmap/设计文档 → 教材实体）

| 设计文档/路线图要求 | 教材落点 | 判定 |
|---|---|---|
| README + 01 手写 DDPM + 02 文生图工具链与 img2img + 03 对齐机制与视频（设计文档"章节"行） | 实体完全一致（视频并入 03，设计文档自洽） | ✅ |
| 脚本 01_ddpm_from_scratch.py（2D toy，CPU 可跑）+ 02_alignment_mechanisms.py（cross-attn/IP-Adapter 解耦 KV/CFG 手写） | 两个脚本均在，CPU 档 | ✅ |
| 作业：DDPM 闭式前向 | 题 1（q_sample 四参 + signal_ratio） | ✅ |
| 作业：CFG 公式 | 题 2 | ✅ |
| 作业：img2img strength→t₀ | 题 3 | ✅ |
| 作业：**SNR 与 β schedule** | 无对应题 | ❌ C1-2 |
| 作业：**动态分辨率 token 估算（联动 a15）** | 无对应题 | ❌ C1-3 |
| 01-04 四节锚点（设计文档按 01/02/03/04 列） | 实际 03 章承载"对齐+视频"两节 | ✅（末尾章节行已改口径，不算缺口） |
| roadmap v3 **节点 16** | **v3 逐节点详解止于节点 15（毕业验收）；Part 15 在缺口登记表标"归档（v2-T9）"、Part 16 全文未提** | ❌ C1-1（重大） |
| ControlNet（零卷积）/ IP-Adapter/InstantID/PuLID | 02 §4 / 03 §1 | ✅ |
| CogVideoX-2B / Wan2.1-1.3B / HunyuanVideo 概览与 24GB 档位 | 03 §3 + README 环境表 | ✅ |
| 设计文档引 docs/datasets.md 增补节（权重下载） | 教程未挂该链接，仅声明"独立 venv + 镜像 ID" | ⚠️ C1-5（弱） |
| 视频生成的作业/验证落点 | 仅 03 练习 2（操作型 GPU 题），assignment 纯 CPU 未覆盖视频 | ⚠️ C1-4（弱） |

### C1 学生单元划分（审计执行单元）

- **U1 扩散数学单元**：README + 01 章 + 脚本 01 —— 前向闭式/ε 预测/采样循环/β 调度，纯 CPU。验收锚点：脚本 01 内联判据（均值偏移<0.1、方差比 0.85-1.15）。
- **U2 工具链单元**：02 章 —— LDM 两跃迁/模型谱系/img2img strength/ControlNet。仅文档级（diffusers 代码内嵌，无脚本），练习 1 可 CPU 验证、练习 2 需 GPU venv。
- **U3 对齐与视频单元**：03 章 + 脚本 02 —— 解耦交叉注意力/CFG 外推/视频最小增量。CPU 手写可验；视频实操仅文档级。

## 2. 特有审计要点（逐项预核结论，正式审计需复核）

### 2.1 DDPM 数学逐式与代码互证（全部通过）

| 式 | 教程表述 | 代码落点 | 互证 |
|---|---|---|---|
| 前向闭式 x_t=√ᾱ_t·x₀+√(1−ᾱ_t)·ε | 01 §推导 Step1-3、q(x_t|x₀)=N(√ᾱ_t x₀,(1−ᾱ_t)I) | 脚本 01 L40-44 `q_sample`：s=alphas_cumprod[t].view(-1,1) → s.sqrt()*x0+(1-s).sqrt()*noise | ✅ 一致；ᾱ=∏(1−β) 定义一致（L36-37） |
| 训练目标 ‖ε−ε̂‖² | 01 §2（DDPM §3.2 VLB 简化） | L102-106：随机采 t + randn_like + mse_loss(eps_pred, noise) | ✅ 一致 |
| 反向采样（式 11） | x_{t−1}=1/√α_t·(x_t−β_t/√(1−ᾱ_t)·ε̂)+√β_t·z，t=0 时 z=0 | L68-74：mean/var/z 实现逐项一致；L69 t=0 用 zeros_like | ✅ 一致 |
| β/ᾱ 调度 | 线性 1e-4→0.02（论文 §4）+ cosine（Nichol&Dhariwal 2021） | L30-32 linspace；教程练习 3 cosine 公式与标准实现一致（f/f[0]、betas=1−ac[1:]/ac[:-1]、clamp≤0.999） | ✅ 一致 |
| ε 参数化 vs x₀ 预测 | 概念检验 Q3（等价但 ε 更稳） | 脚本仅实现 ε | ✅ 自洽 |
| 签名口径 | 教程 L134-136 显式声明脚本 3 参版 vs 作业 4 参版 `q_sample(x0,alphas_cumprod,t,noise)` | 作业题 1 签名一致 | ✅ 无歧义 |

### 2.2 img2img 强度插值

- t₀=⌊steps×strength⌋ 与 √ᾱ 表（02 练习 1 参考值 0.9204/0.6017/0.2737/0.1322，T=400 线性调度）数学自洽；作业 Q2（0.60/0.13）与之吻合 ✅。
- 审计注意：02 练习 1 的 `idx = min(int(t0/30*399),399)` 是"推理步→400 步时间线"的教学映射（非 diffusers 内部真实离散化），正式审计需确认该简化是否向学生言明（正文未明说 → 记为待核点 A-1）。

### 2.3 ControlNet / IP-Adapter / CFG 原理

- ControlNet 零卷积 = "LoRA B=0 同款设计模式"（02 §4、Q4）：类比成立 ✅（2302.05543 引用正确）。
- IP-Adapter：out = attn(Q,K_txt,V_txt) + scale·attn(Q,K_ref,V_ref)，**共享 Q、独立新 K/V**——脚本 02 `DecoupledCrossAttention` 与论文 2308.06721 一致 ✅；"22M 参数、scale=0 行为不变"可验证定义与题 4 验收一致 ✅。
- CFG 外推公式与 10% 条件置空的"训练侧配套"论述（03 §2、Q3）✅。
- 待核点 A-2：03 §2/练习 1 的"余弦随 w 收敛"实验，输入是随机两路 ε，cos(w) 上升是线性代数构造的必然（非模型行为），教程表述"外推方向收敛"略过度包装——正式审计建议加一句定性说明（不影响判分）。

### 2.4 CogVideoX / Wan2.1 概览（发现 1 处技术准确性问题）

- **A-3（重点）**：03 §3 表格与脚本 02 尾注均称视频模型 = "空间块间插入 temporal attention"（Latte 式因子化）；但 **CogVideoX 用 3D full attention、Wan2.1 用全自注意力 + flow matching + UMT5**，并非"插入式 temporal attention"。同章 Q2 又写"CogVideoX 用 3D 因果 VAE + 分层策略"——章内表述存在张力。建议正式审计定级为中等技术错误（教学简化可辩护，但"最小增量视角"应标注仅对 Latte 严格成立）。
- 权重/显存档位（fp16 ~4GB / int8 3.6GB / Wan 8.2GB / Hunyuan 60GB→量化 24GB）与 README 环境表一致 ✅；CogVideoXPipeline 示例参数（THUDM/CogVideoX-2b、num_frames=49、steps 50、guidance 6.0）与官方用法相符 ✅。

### 2.5 diffusers 权重依赖档位（分层清单，供正式审计对照）

| 档位 | 内容 | 教程声明位置 |
|---|---|---|
| L0 纯 CPU 零新依赖 | 脚本 01/02、题 1-4、02 练习 1、03 练习 1 | README 环境表、assignment 头部 ✅ |
| L1 GPU 可选 venv | SD1.5 fp16 ~2GB（⚠️ 镜像 ID `stable-diffusion-v1-5/stable-diffusion-v1-5`，runwayml 已删——三处口径一致 ✅）、SDXL ~7GB、FLUX fp8 ~12GB | README 表、02 §2/陷阱 2/练习 2 |
| L2 视频 | CogVideoX-2B、Wan2.1-1.3B（~4min/5s 480p） | 03 §3、练习 2 |
| L3 引述不实操 | HunyuanVideo 13B | 03 §3 ✅ 已声明 |

- 风险：`pip install …（latest）` 无版本钉——diffusers API 漂移（如 export_to_video 路径、pipeline 签名）无兜底说明 → 记待核点 A-4（弱）。

### 2.6 scripts 档位判定（toy 2D 扩散可跑性）

- 脚本 01：torch 单依赖、seed=1337 固定、CPU<30s、验收判据内联（mean<0.1 / std ratio ±0.15）→ **toy 2D 扩散可跑，L0 档成立** ✅。模块级 `alphas_cumprod` + main() 内 global 搬运到 device 的写法已由教程"签名说明"显式豁免 ✅。
- 脚本 02：CPU<5s、shape/行为断言内联（scale=0 等价性、CFG cos>0.99）→ 可跑 ✅。
- 口径小瑕疵 A-5：运行时长三处表述不一（01 章头"CPU 30 秒"、脚本头"<30 秒"、设计文档"<60s"）——不影响判分。

## 3. 格式规范预检（G1/G2/G4/G10/G13/G14）

| 规则 | 预检结果 | 违反清单 |
|---|---|---|
| G1 结构完备 | README：学习目标/导航/前置/位置/环境/学习地图/作业/资源齐；三章均有 学习目标/前置/理论/工程实践/学完你能/概念检验/动手实践/作业/下一步 | 0 |
| G2 链接有效 | 实测跨 Part 链接 4 条全部存在（P15 README、P17 README、P6 02 章、P8 06 章）；章内互链、docs/llm_interview_guide.md、paper_reading_guide.md 均存在；Part15 02 章确为"三大方案与对齐损失"，被引内容属实 | 0 |
| G4 代码块语言标注 | 无语言裸围栏共约 9 处（01 章 5：L42/59/89/101/143 附近的 ASCII 图与公式块；02 章 2：L23/63；03 章 2：L44/76）——均为 ASCII 示意图/公式块，若 G4 豁免此类则不违反 | 0 硬违反 + 9 处待裁决 |
| G10 课后作业链接 | 4 个文件均有 `assignment_16` 链接且路径正确 | 0 |
| G13 概念检验+动手实践 | 01：3Q+3 练；02：4Q+2 练；03：3Q+2 练（README 按 README 体例豁免） | 0 |
| G14 表格格式 | 抽查全部 13 张表：表头分隔行齐全、列数一致 | 0 |

**格式违规数（硬）：0**；待裁决项 1 类（G4 裸围栏 9 处，建议按"ASCII/公式块豁免"口径结案）。

## 4. 与 Part 15 的跨模态对齐主线衔接

- README 顶部"Part 15 理解侧 ⇄ Part 16 生成侧"主线声明 ✅；01/02/03 三章前置知识均回指 P15 02 章，且 P15 02 章实有"对齐损失 + 和生成侧的连接（Part 16 的地基）"反向外指 ✅——双向衔接闭合。
- 03 §4 收官总图（projector/adapter KV/scale-CFG 三件套共享设计模式）与脚本 02 尾注"统一视角"一致 ✅。
- 弱点 C1-4 的另一面：对齐主线在"理解侧"有作业闭环（a15），在"生成侧"只有 CPU 四题，视频/IP-Adapter 拉通能力无作业验证落点——正式审计可据此建议在 assignment 或 03 练习中补一个选做观测题（不强制）。
- 收尾链：03 章"生成线毕业（Part 1-16）→ Part 17 Agentic RL"与 P17 目录实体存在 ✅。

## 5. C1 缺口汇总（正式审计的评分输入）

| 编号 | 缺口 | 级别 |
|---|---|---|
| C1-1 | roadmap v3 无节点 16：逐节点详解止于 15，Part 15 标"归档 v2-T9"、Part 16 未收录——路线图与已建成实体脱节（含"多模态归档 vs 实际两 Part 齐建"矛盾） | 重大 |
| C1-2 | 设计文档作业清单中"SNR 与 β schedule"题未落地 | 中 |
| C1-3 | 设计文档作业清单中"动态分辨率 token 估算（联动 a15）"未落地 | 中 |
| C1-4 | 视频生成无作业/自动验证落点（仅操作型练习） | 弱 |
| C1-5 | 教程未挂 docs/datasets.md 权重下载指引链接；diffusers 版本无钉（A-4 同源） | 弱 |

**C1 缺口数：5（1 重大 / 2 中 / 2 弱）**

## 6. 技术准确性待核点（非 C1，供正式审计定级）

- A-1：02 练习 1 的"推理步→400 步时间线"idx 映射为教学简化，未向学生言明。
- A-2：CFG 余弦实验为随机向量下的构造性结果，"方向收敛"表述略过度包装。
- A-3：temporal attention 表述 vs CogVideoX/Wan2.1 的 3D full attention（章内自相张力）——**建议定级中等**。
- A-5：运行时长口径三处不一（30s / <30s / <60s）。
- A-6：01 陷阱 2"方差口径不一致 → loss 不下降"因果链可疑（std 估计口径影响评估统计量，通常不影响训练 loss 走向）——建议正式审计重写该陷阱的症状/原因配对。
- A-7：01 最佳实践表 T=1000 与脚本 T=400 未在表处呼应（错误 2 处已注明 T=400）。
- A-8：README 学习目标"配置 diffusers 的推理服务"措辞过强（教程无 serving 内容，仅推理两行调用）。

## 7. 正式审计执行清单（T2/T3 接手用）

1. U1：逐式复核 §2.1 六行互证表；实跑脚本 01 验证内联判据与教程实测数（loss 1.11→0.21、均值偏移/方差比）。
2. U2：复核 02 练习 1 参考数值（0.9204/0.6017/0.2737/0.1322）与题 3 floor 语义边界（strength=0.02→1、∈[0,steps]）。
3. U3：复核脚本 02 三个实验输出与教程引述数（scale=0 等价、[2] max 变化、CFG cos≈0.998→练习 1 表 0.7004/0.9803/0.9974/0.9994）。
4. A-3/A-6 两处技术表述按 §6 建议定级并记录修改建议（不改文件，仅登记）。
5. G4 的 9 处裸围栏按豁免口径裁决后结案。
6. C1-1 需升级至路线图 owner（v3 增补节点 15.5/16 或在缺口登记表改"归档"口径）。
