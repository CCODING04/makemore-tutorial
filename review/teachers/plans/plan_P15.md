# plan_P15 — Part 15 多模态理解（VLM）整体预审计划（T1 主教）

- 日期：2026-09-04 ｜ 主教：T1 ｜ 模式：只读预审（未改文件、未跑仓库脚本；算术用 python -c 验算）
- 范围：`courses/Part15_vision_language/`（tutorial 3 个 md 共 785 行 + scripts/ 2 个共 313 行）+ `assignments/assignment_15/`（assignment.md 187 行 + test/vlm_exercises 骨架各 1 个）+ `docs/course_roadmap_v3.md` 节点 15 与缺口登记表
- 环境事实：HF 缓存共 8 项，**无任何 VLM/CLIP 权重**（无 SmolVLM/Qwen2-VL/PaliGemma）；两脚本声明零新依赖 → 全部内容可 CPU 实跑，工业推理节天然走概念档

---

## 1. C1 对应表（roadmap ↔ 仓库实体）

> 特殊性：roadmap **节点 15 = 毕业验收**（三个项目故事+面试模拟），与目录 `Part15_vision_language` **同号不同物**；VLM 在 roadmap 仅出现于 §6 缺口登记表一行（"多模态 VLM ｜ A 补充 ｜ Part 15（可选）｜ 归档（v2-T9）"）。对应表按"缺口表行 + README 自我定位"双向核。

| roadmap / README 承诺 | 仓库实体 | 状态 |
|---|---|---|
| 缺口登记表：多模态 VLM，A 补充，Part 15 可选，归档（v2-T9） | 实际已建成：2 章 + 2 脚本 + 作业 15（4+1 题）+ 测试全绿痕迹（`__pycache__` 存在） | ⚠️ C1-1 缺口表状态行陈旧：未反映"已建成"现状；且"节点 15（毕业）"与"Part 15（VLM）"同号错位无提示语，学习者按路线图走到"节点 15"极易与 Part 15 目录混淆 |
| README 学习目标：手写四件套（PatchEmbed+ViT+Projector+拼接） | 01 章全文 + 脚本 01（断言内联）+ 作业题 1/3 | ✅ |
| README 学习目标：解释 CLIP/SigLIP 数学原理、batch 依赖 | 02 章 §2 + 脚本 02 + 作业题 2 + 思考题 Q2/Q4 | ✅ |
| README 学习目标：画出三大方案注入位置图并选型 | 02 章 §1 表格 + 思考题（文字可画） | ✅ 内容在（图缺失见 G4-1） |
| README 学习目标：识别静态图缓存/τ/batch 陷阱 | 01 错误 1 + 02 错误 1/2 + 脚本 01 L146-147 实证注释 | ✅ |
| 01 学习目标：标注每步 shape（形状账本） | 01 §1 + 脚本 01（自算复核 1,856/29,308 全对） | ✅ |
| 02 学习目标：估算动态分辨率 token 数 | 02 练习 3 = 作业题 4（同名同签名同默认值）+ 02 概念检验 Q2 | ✅（4070//4=1017 自算 ✓） |
| 作业 15 四题 + 🌟 题 4 优雅 SKIP + 实验题 + 面试直通车 + 思考题 | assignment.md：题 1/2/3/4🌟 + 2 实验题 + 4 条面试直通车 + Q1-Q5 | ✅（但题 3 数字硬伤，见 C1-2） |
| README "权重获取见 docs/datasets.md §5" | `docs/datasets.md` 实际仅 §1-4（权重在 §4） | ❌ C1-2' 引用断链（归入 G10-1） |
| README 导航：上一章 Part 14 / 下一章 Part 16 | 两 README 均存在 | ✅ |

**C1 缺口汇总 = 4**：
- **C1-1** roadmap 缺口表 Part 15 行定位陈旧 + "节点 15 vs Part 15"同号错位无提示（建议缺口表行加一句"已建成，见目录"或路线图加编号错位说明；P2）。
- **C1-2（P0 候选）** assignment.md 题 3 验收标准写死 `mlp2x_params(1024, 4096) == 21,043,712`，而同一行给出的公式 `(1024*4096+4096)+(4096*4096+4096)` 真值 = **20,979,712**（主教验算），差 64,000；test_vlm_exercises.py L58 用公式动态计算（= 20,979,712）→ **学生若照 md 硬编码 21,043,712 必 FAIL**。md 数字笔误，须改 md（test 不用动）。
- **C1-3（P1）** 01 章练习 2 步骤提示 `Linear(d_v, d_l*2) → GELU → Linear(d_l*2, d_l)`（隐层 d_l×2）与正文 L58、脚本 01 Projector、作业题 3 的 `Linear(v,l)→GELU→Linear(l,l)`（隐层 d_l）**口径矛盾**；若按练习 2 实现，题 3 参数公式随之改变，牵连作业口径（同根见 G10-2）。
- **C1-4** README L58 "docs/datasets.md §5" 断链（§5 不存在，应指 §4）（同 G10-1）。

**交叉引用抽验（通过）**：`Part6_transformer/tutorial/03_transformer_block.md`、`Part8_post_training/tutorial/02/07/08`、`Part14/Part16 README` 均存在；论文 arXiv 号全对（CLIP 2103.00020 / SigLIP 2303.15343 / LLaVA 2304.08485 / Qwen2-VL 2409.12191 / Flamingo 2204.14198）；558K/665K（LLaVA-1.5 数据）、CLIP batch 32768、τ=0.07、576 token（336²/14²）均为文献正确值。Part 6 03 章"同款 Block"与 Part 16"互为镜像"的**实现级**呼应留 B 阶段抽验。

## 2. 学生单元划分（3 学生并行；全文 <1.3K 行，不拆章组）

- **S1 形状账本+两阶段档**：01 章全文 + `scripts/01_vit_projector_pipeline.py` 实跑（CPU <10s）+ §3.1/3.2/3.4 清单 + 数字核对（1,856 / 29,308 / 2.907→1.825 / →0.034）。
- **S2 对齐损失+三大方案档**：02 章全文 + `scripts/02_clip_siglip_alignment.py` 实跑 + §3.5/3.6 清单 + 数字核对（loss 4.155→1.968 / 0.912→0.106、τ 16.21 / 8.53、检索 100%）。
- **S3 作业+交叉引用+格式档**：assignment_15 全部（跑 `python test_vlm_exercises.py` 或 pytest，预期全绿；专验 C1-2 题 3 数字）+ README + roadmap C1 对应 + G 码通检互查 + Part 6/8/16 交叉引用抽查。

## 3. 特有审计要点

### 3.1 patch embedding 形状链（重点，可复现）
- 链条：图像 (B,3,8,8) → Conv2d(k=s=2) → (B,24,4,4) → flatten+transpose → (B,16,24)，(8/2)²=16 ✓；脚本 L137 有内联 assert，教程"形状账本"与脚本逐行一致 ✓。
- B 阶段核对（S1/S3）：题 1 三组值 256/576/4070（74×55，主教已自算 ✓）与 `math.ceil` 非整除补齐口径一致。
- 玩具图 8×8 vs 真实 224² 的档位说明在脚本 docstring 有 ✓。

### 3.2 ViT 结构
- `ViTBlock` 为 pre-norm + 双残差：`x + a + mlp(ln(x+a))`，数学上等价标准 pre-norm 两子层展开 ✓；视觉塔双向（无 mask）、ToyLLM 文本段因果 mask（`triu` buffer 切片）✓，与概念检验 Q2"图像 token 在前缀内互可见"解释自洽 ✓。
- **与 Part 6 03 章"同款"声明是文字级**：两处实现写法不同（Part 6 待 S3 抽验是否逐行同构；不同不判错，但"同款"措辞建议核实）。
- 观察（P3）：`ToyLLM.forward` L85-88 手动重抄了 ViTBlock 的展开逻辑（为传 causal mask），与类定义重复——单点维护风险，建议 ViTBlock.forward 支持 attn_mask 参数复用（G15 精神）。

### 3.3 projector（MLP 拼接）实现
- 脚本 `Projector(24,32)`：Linear(24→32)+GELU+Linear(32→32)，参数 800+1,056=**1,856**（主教自算 ✓）；Stage 2 总参 vit 9,648 + llm 17,804 + projector 1,856 = **29,308**（自算 ✓）。
- **C1-3 练习 2 隐层口径矛盾必须整改**（d_l*2 vs d_l，三处对一处错）。
- 作业题 3 结构口径（Linear(v→l)+Linear(l→l) 均含 bias）与脚本一致 ✓；"LLaVA 7B 约 20M、占 ~0.3%"结论数量级 ✓（真值 20.98M / 7e9 ≈ 0.3%）。

### 3.4 LLaVA 两阶段训练口径
- Stage 1 冻结 ViT+LLM 只训 projector（生成式对齐，next-token CE）、Stage 2 全解冻——与 LLaVA 论文口径一致 ✓；labels 的 -100 prompt masking 与 Part 8 02 章呼应 ✓；`build_batch()` 每步重建图（静态缓存坑）实证注释 ✓。
- **G14-2 档位混用（P1）**：01 章最佳实践表给"真实口径"lr（Stage1 1e-3 / Stage2 2e-5），脚本实际用玩具 lr（3e-3 / 1e-3），正文未注明玩具版与真实口径的差异及理由——学生跑脚本对照表会困惑。建议表旁加一句"玩具版 lr 见脚本（3e-3/1e-3），真实 LLaVA 口径如下"。
- 数字核对（S1 实跑）：Stage1 loss 2.907→1.825、Stage2 →0.034；参数 1,856/29,308 已静态核过 ✓。

### 3.5 CLIP/SigLIP 对比（InfoNCE / Sigmoid loss 公式）
- InfoNCE：`0.5*(CE(logits)+CE(logits.T))` 对称双方向、标签=对角线 ✓ 与脚本一致；SigLIP：`targets=2*eye-1`、`-logsigmoid(targets*logits).mean()` ✓ 与脚本一致；mean 等价论文 1/N² 归一 ✓。
- **简化口径未声明（P3 建议）**：SigLIP 论文原式含逐对可学习偏置 b（`z_ij·(t·x·y+b)`）与 -1 系数细节，教程/脚本省略——建议加一句"玩具版省略可学习偏置"。**G2-3**：两损失的数学定义全文仅有 Python 代码块形态，无 LaTeX 公式（softmax 归一化式、sigmoid loss 原式缺位）。
- batch 依赖性论述（InfoNCE 负例来自 batch / SigLIP 逐对独立、论文 batch 1/4 持平）与文献一致 ✓；思考题 Q2 分母展开解释准确 ✓。
- **O1（G15 精神）**：温度初始化两口径——教程 02 错误 1 解法 `log(1/0.07)`（CLIP 默认 ≈14.29），脚本 02 实际 `log(10.0)`，未注明玩具初始化不同于 CLIP 默认。
- 温度机制（`exp(log_scale)` 保正 + clamp 100 + 可学习控锐度）讲解正确 ✓；τ 实测 16.21/8.53 留 S2 实跑核对。

### 3.6 三大方案（拼接 / 门控 / early-fusion）准确性
- (a) 拼接式主流名单（LLaVA/SmolVLM/Qwen-VL/InternVL/nanoVLM/minimind-v）✓；(b) Flamingo Perceiver Resampler + gated xattn，Q1"tanh 门控零初始化、与 LoRA B=0 同模式"解释与文献一致 ✓；(c) Fuyu-8B patch 线性投影直入无视觉编码器 ✓、Chameleon VQ token ✓；InternVL pixel shuffle 2×2 重排 → token ÷4 ✓；Qwen 系原生动态分辨率 + M-RoPE ✓。
- "GLM-5.3-Flash 原生多模态 / DeepSeek-V4 混合模态注意力"为 2026 时效断言：不判错，B 阶段标注"教学锚点、以官方发布为准"即可。
- 措辞小瑕（P3，不单列违规）：表头"Qwen2.5-VL"与正文"Qwen-VL 系/Qwen2-VL"混称；README L8 锚点 Qwen3-VL 与 01 章对照表 Qwen2.5-VL 并存，建议统一加版本号。

### 3.7 图像-文本对数字例（已全部自算）
| 数字 | 出处 | 主教自算 |
|---|---|---|
| (224/14)²=256、(336/14)²=576 | 作业题 1 | ✓ |
| ceil(1024/14)×ceil(768/14)=74×55=4070 | 02 概念检验 Q2 / 题 1 | ✓ |
| 4070//4=1017（pixel shuffle，未触发 2560 预算） | 题 4 / 02 Q2 | ✓ |
| projector 1,856 / Stage2 29,308 | 01 章 / 脚本 | ✓ |
| mlp2x_params(1024,4096) | 作业题 3 | md 写 21,043,712 ✗（真值 20,979,712，C1-2） |
| InfoNCE 4.155→1.968 / SigLIP 0.912→0.106 / τ 16.21, 8.53 / 检索 100% | 01/02 章"实测" | 留 S1/S2 实跑 diff |

### 3.8 权重依赖档位说明
- 两脚本：**toy 档、纯 CPU、零下载**（torch only，seed=1337 固定）——README"CPU 可跑、零新依赖"声明属实 ✓。
- 工业推理（SmolVLM-500M <1.5GB / Qwen2-VL-2B ~5GB）：README 已定位"进阶自练"，不进脚本、不进作业 ✓；本机缓存无此类权重 → B 阶段该节走概念档，不下载。
- pytest 无权重依赖，S3 可直接全绿验证；体积数字（<1.5GB/~5GB）标"外部来源、B 阶段不核"。

## 4. scripts 档位

| 脚本 | 行数 | 档位 | 依赖 | B 阶段动作 |
|---|---|---|---|---|
| 01_vit_projector_pipeline.py | 203 | toy（8×8 图、2+2 层、200+100 步、CPU <10s） | torch | S1 于 REVIEW/scratch 复跑（G16：先复制再运行），diff 4 个数字 |
| 02_clip_siglip_alignment.py | 110 | toy（4 概念×8 样本 N=32、300 步、CPU <10s） | torch | S2 同上，diff 5 个数字（含 τ 两位小数敏感性注明） |

- 两脚本均有内联 shape assert / 优雅打印，"失败仍庆祝"风险无（无判定分支）✓；G16 注意：脚本不写文件，风险低但仍走 scratch。

## 5. 格式规范预检违反清单（合计 15 条：教程 13 + 作业 2；另 2 观察项）

**G1（1 条）**
- G1-1：02 章 L34-36 "📌 细节差异点"——Qwen-VL 动态分辨率与 InternVL 像素洗牌两个并列要点以分号挤同一段落，应拆行。

**G2（5 条：教程 4 + 作业 1）**
- G2-1：01 章 L49-65 "数学推导：Projector 的作用"——Step 1-4 推导整段放代码块（禁代码块承载数学推导），应转 `$`/`$$`。
- G2-2：01 章 L46-47 `v ∈ R^{d_v}` 等 unicode 数学，应为 `$v \in \mathbb{R}^{d_v}$`。
- G2-3：02 章 L43-51 InfoNCE/SigLIP 核心公式仅以 Python 块呈现，全文无 LaTeX 数学定义（softmax 归一化式与 sigmoid loss 原式缺位）——本 Part 最重的 G2 项。
- G2-4：02 章 L59 `scale = exp(log_scale)` 行内代码承载数学。
- G2-5（作业）：assignment.md L123 InfoNCE 分母 `Σ_j exp(sim(i, j))` 行内代码。

**G4（1 条）**
- G4-1：全 Part 0 图、无 `images/` 目录。可画点至少 4 处：三大方案注入位置对比图（学习目标要求"画出"却无参照图）、patch 切分→token 序列示意、两阶段 loss 下降曲线（2.907→1.825→0.034）、CLIP vs SigLIP 相似度矩阵/温度锐度对比热图。

**G10（3 条）**
- G10-1：README L58 `docs/datasets.md §5` 断链（=C1-4，应指 §4）。
- G10-2：01 章练习 2 签名口径与脚本/正文/题 3 不一致（=C1-3）。
- G10-3（作业）：assignment.md L62 写死 21,043,712 与公式/test 动态值不一致（=C1-2）。

**G13（3 条）**
- G13-1：02 章 L147-148 "~95%、~1h、~30min"标注"论文报告值"——CLIP/SigLIP 论文报的是 ImageNet zero-shot 等口径，无"检索 ~95%"与训练时长此类数字，来源不可考，疑似杜撰口径（P1，整改时改标"示意/删除"或给真实出处）。
- G13-2：01 章 L198-199 7B "~2h/~10h" 混列"LLaVA 论文 + 本课开发机实测"——论文未报训练时长，须拆分双列或删归因。
- G13-3：README star 数（nanoVLM 5.0k / minimind-v 8.5k / Qwen3-VL 19.9k / LLaVA 25k）未标"截至 x 年 x 月"时点。

**G14（2 条）**
- G14-1：全部"实测"数字仅 device 级口径（"RTX 4090 实测/CPU 同量级"），缺 seed（脚本 1337 可直接补注）/torch 版本/线程声明（01 L115、02 L60、README L57-58 三处同根，算 1 条）。
- G14-2：档位混用——最佳实践表真实 lr（1e-3/2e-5）vs 脚本玩具 lr（3e-3/1e-3）未对照说明（=§3.4，P1）。

**观察项（不判违规）**
- O1：温度初始化两口径（教程解法 log(1/0.07) vs 脚本 log(10.0)）未注明玩具差异（G15 精神，见 §3.5）。
- O2：01 章 L78-84 形状账本伪代码块未标"示意"（有"运行脚本验证"引导，边缘合规）。

## 6. 总评与派工

**总评**：本 Part 体量小、工程质量高——形状账本/参数账/文献数字静态核验几乎全对，脚本断言内联、诚实档位声明是范本级；但有 1 个 P0（题 3 写死数字与公式/测试矛盾，学生会照 md 写错）、1 个 P1 口径矛盾（练习 2 隐层 d_l×2）与 2 个 P1 归因问题（G13-1 疑似杜撰"论文报告值"、G14-2 lr 档位混用），全部为局部可改，无结构性缺失。

**P0/P1 速览（供 T2 台账）**：C1-2/G10-3（题 3 数字，改 md 一处）、C1-3/G10-2（练习 2 口径，改提示一处对齐 d_l）、G13-1（改标示意或删）、G13-2（拆双列）、G14-2（加一句对照）。C1-1/C1-4 与 G 码 13 条为 P2/P3 批量小修。

**派工**：S1/S2 各带一张数字 diff 表（§3.4/§3.7 共 9 个实测数字）；S3 除 pytest 全额外，专项复核 C1-2 修复后 md-测试一致性。禁改清单与 G16（先复制再运行）对全体生效。
