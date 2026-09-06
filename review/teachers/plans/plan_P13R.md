# plan_P13R — Part 13 剩余章节预审计划（00_scaling_laws + 02_data_juicer + README）

> T1 主教 · 2026-09-04 · 范围：`courses/Part13_data_engineering/tutorial/` 下 **00 章、02 章、README.md**
> （01 章已完成审计，本轮不碰）。只读仓库 `/home/admin02/Code/WorkSpace/makemore-tutorial`（REPO）。
> 本文件是审计**计划**：先固定对应表与要点，正式审按 §6 顺序执行。

---

## 1. 审计对象与已核实的物理事实（预检阶段已验证，正式审直接引用）

| 载体 | 位置 | 规模 | 状态 |
|---|---|---|---|
| 00 章教程 | `REPO/courses/Part13_data_engineering/tutorial/00_scaling_laws.md` | 613 行 | 待审 |
| 02 章教程 | `REPO/courses/Part13_data_engineering/tutorial/02_data_juicer_pipeline.md` | 355 行 | 待审 |
| Part13 README | `REPO/courses/Part13_data_engineering/tutorial/README.md` | 183 行 | 待审 |
| 00 脚本 | `REPO/courses/Part13_data_engineering/scripts/00_scaling_laws.py` | 945 行 | 已通读 |
| 00 脚本产物 | `scripts/output_scaling_fit.png`、`scripts/output_scaling_epoch.png` | 均存在 | 已验证 |
| 数据依赖 | `REPO/data/input.txt`（tiny shakespeare，1,115,394 字节） | 存在 | 已验证 |
| 作业题面 | `REPO/assignments/assignment_13/assignment.md`（题 5 + 实验题，02 章相关部分） | — | 已读 |
| 路线图 | `REPO/docs/course_roadmap_v3.md` 节点 13（L467-490）+ L414/L596 的 v3.1 增补行 | — | 已读 |

**预检已验证为真的环境/事实**（正式审不必重跑）：
- 教程 00 章 L293-294 环境标注 "torch 2.6.0+cu124 · Python 3.12 · scipy 1.18.1" 与 `.venv` 实测**完全一致**（scipy 1.18.1 / torch 2.6.0+cu124 / Python 3.12.12）。
- 00 章引用的 7 个站内相对链接全部存在（Part7 README、Part7/03_gqa_and_ffn.md、Part8/01_gpt_and_pretrain.md、Part8/07_evaluation.md、Part14 README、Part12 README、01_dedup_from_scratch.md）。
- 跨 Part 事实声明属实：Part7·03 章确有 "MoE：把 FFN 换成一群专家"（L326）；Part8·01 确为 "LayerNorm + learned PE + MHA + ReLU" 经典款（L3）；Part8·07 确有 n-gram 去污染（L177）。
- **contextual retrieval 核对结论（台账项关闭）**：`scripts/02_contextual_retrieval.py` 在 **Part18_rag**，不在 Part 13。Part13 全目录（教程 3 文件 + 脚本 + 作业）grep `contextual / 1.9% / 2.1% / 上下文检索 / bm25` **零命中** → 02 章与作业**不含**该数字，**无失实问题**；"docstring 1.9% vs 官方 2.1%" 的修正只归属 Part 18 RAG 审计线。本计划不再列此项为缺口。

---

## 2. C1 对应表（内容 ↔ 载体 ↔ 路线图 ↔ 作业）

### 2.1 00 章（Scaling Law 开篇，一个大单元）

| 教程内容 | 载体/脚本 | 作业/路线图 | 对应判定 |
|---|---|---|---|
| 三项式推导（Kaplan→6ND→Chinchilla→Lagrange） | `chinchilla_loss`（脚本 L61-79）、`compute_optimal`（L158-170） | 无直接题；路线图 L596 "Chinchilla 拟合/isoFLOP/epoch 三实验" | ✅ 对应 |
| fit 模式：合成网格 + Huber 拟合 + isoFLOP 图 | `run_fit_mode`（L548-646） | 无 | ✅ 对应 |
| scan 模式：3×3 网格真训 + 病态拟合诊断 | `run_scan_mode`（L652-772） | 无 | ✅ 对应 |
| epoch 模式：R∈{1,2,4,8,16} 饱和实验 | `run_epoch_mode`（L778-920） | 无 | ✅ 对应 |
| 三坑复盘 / 三陷阱（LR horizon、参数边界、train/val） | `lr_at` 注释（L296-311）、make_resampled_corpus 注释（L379-383） | 无 | ✅ 对应 |
| **00 章整体** | 00 脚本 | **路线图节点 13 正文（L470 "教程 2 章"、L472 "脚本：01_minhash_dedup.py"）→ ❌ 未同步**；L414/L596 增补行已提 00 章 → 节点内部自相矛盾 | ❌ 缺口 C1-1、C1-2 |

### 2.2 02 章（Data-Juicer，一个单元）

| 教程内容 | 载体 | 作业/路线图 | 对应判定 |
|---|---|---|---|
| 最小 YAML（words_num_filter + document_minhash_deduplicator） | 无脚本（YAML/CLI 实操，README 导航已标 "—（YAML/CLI 实操）"） | 实验题 "装上 Data-Juicer，用 02 章 YAML 跑自己的 tiny_corpus" | ✅ 对应 |
| 手写↔Data-Juicer 对照表（5 行） | 无 | **题 5 引用 "02 章对照表第 4 行"**（keep-first→keep-longest）→ 核对：02 章 L94 第 4 行确为该内容 | ✅ 对应准确 |
| 追踪审计（stats/ 目录，L117-137） | 无 | 实验题 "读一遍 stats 追踪报告" | ✅ 对应 |
| FineWeb 配置表（L250-258） | 无 | 作业 Q4 + 面试直通车 "5-gram、14×8、≈0.7、3.3M CPU 小时" | ⚠️ 两处口径一致性待联网核对 FineWeb 原文（见 §4.2） |
| 02 章整体 | 无脚本 | 路线图节点 13 L470-471 ✅ 已含 | ✅ 对应 |

### 2.3 README

| README 内容 | 判定 |
|---|---|
| 章节导航表（3 行，含 00 章标 ⭐建议先读） | ✅ 已同步 00 章 |
| 学习地图（L158-166，含 "Scaling Law 开篇（00）"） | ✅ 已同步 |
| **学习目标 5 条（L13-17）：无一条覆盖 00 章 "预算语言/Scaling Law"** | ❌ 缺口 C1-3 |
| **环境与版本策略（L144-155）：只有 01 章与 02 章两行，无 00 章脚本的 torch/CUDA/matplotlib 档位** | ❌ 缺口 C1-4 |
| 前置知识 "必须掌握 Part 8·07 评估学" vs 00 章自身前置 "必须掌握 Part 7 预训练" | ⚠️ 口径差异（全 Part 视角 vs 单章视角），正式审判断是否需要一句互指说明 |

**C1 缺口计数：4 项**（C1-1 路线图节点 13 "教程 2 章" 缺 00 章；C1-2 路线图节点 13 脚本清单缺 00_scaling_laws.py 且作业描述 "4 编码题" 未含题 5 Stretch——与 C1-1 同源但分列；C1-3 README 学习目标缺 00 章条目；C1-4 README 环境节缺 00 章脚本档位）。
另记作业侧同步点 1 项：路线图 L473-474 "4 编码题" vs 作业实际 "4 必做 + 1 🌟 Stretch（题 5）"（并入 C1-2 计数）。

---

## 3. 学生单元划分（正式审按此切分逐段核）

- **单元 A（00 章，一个大单元）"Scaling Law = 数据预算语言"**：历史脉络 → 数学推导 6 步（含 Kaplan vs Chinchilla 的 LR horizon 坑、重复折扣、过训练）→ 两个附录（涌现之争、MoE 粒度）→ 代码三模式 → 实测三段 → 三坑复盘 → 三陷阱 → 练习（概念 3 + 动手 2 + 扩展 3）。
- **单元 B（02 章，一个单元）"Data-Juicer 工业放大"**：安装 → 最小 YAML → 手写对照表 → 完整管线 → 审计 → 错误调试 3 例 → 性能表 → 陷阱 3 例 → 最佳实践 + FineWeb 配置表 → 概念检验 3 + 练习 2。
- README 不单独成单元，作为导航/环境一致性核验对象（对应缺口见 §2.3）。

---

## 4. 特有审计要点（本轮预审的核心增量）

### 4.1 00 章 Chinchilla 公式与拟合数字口径

**已预核为正确的（正式审抽查即可，不必重推）**：
- 常数表 E=1.69 / A=406.4 / α=0.34 / B=410.7 / β=0.28（教程 L123-131、脚本 L55）= Hoffmann 2203.15556 Table 3 参数化拟合，来源标注正确，且注明 "N 为非 embedding 参数"。
- Lagrange 推导（L149-159）逐步验算正确：`-αA·N^(-α-1) + βB·(6/C)^β·N^(β-1) = 0` → `αA·N^-α = βB·D^-β` → N_opt/D_opt 闭式解与脚本 `compute_optimal` 实现一致；α/(α+β)=0.548≈0.55、β/(α+β)=0.452≈0.45 ✓。
- t/p 账全自洽：Chinchilla 70B/1.4T=20 ✓；Llama2-70B/2T≈29 ✓；Llama3-8B/15T=1875 ✓；90×=1875/20 ✓；"8B 的 Chinchilla 最优约 1600~2000 亿 token" = 20 t/p×8B ✓。
- fit 模式 isoFLOP 表（L315-319）三行 t/p 11.0/13.7/17.2/21.5 与正文 "11→21.5"（L167）一致；scan [5] 表 62.3/129.6/292.4 内部除法一致 ✓。
- epoch [5] 折扣表非单调波动（1.21x/0.69x/1.21x/1.44x/0.39x）已在 L409-410 诚实标注为拟合残差噪声 → 口径合格。
- "R≤4 近似免费、R>16 收益趋零"（L188-190）与 Muennighoff 2305.16264 论文措辞一致，且教程特意给出论文原话定位 ✓。

**正式审必查的疑点**：
1. **[P13-00-1] 符号口径失真**：教程 L304 "A: 真值 406.400 单次拟合 360.954 偏差 +11.2%"——360.954 < 406.4，真实偏差是 **-11.2%**。根因在脚本 L583：`abs(fv-tv)/tv*100` 套了 `abs()` 却用 `{:+.1f}` 带符号格式 → 恒显 "+"。审计裁定：改脚本格式串（去 abs 或去符号）或教程文字注明 "偏差为绝对值"。**这是本轮已锁定的第 1 个数字口径缺陷**。
2. **[P13-00-2] 耗时口径**：教程 L250 epoch "~31s" vs 脚本 docstring "~30-40s"、L251 标题 "~31s"。取实测单值可以，但建议与脚本 docstring 区间对齐（25-30/30-40）。低优先级。
3. **[P13-00-3] 待联网复核的声明**（离线不可判，逐条列给正式审）：Lilian Weng 博客 URL 日期 2026-06-24 是否真实存在（L603）；Jason Wei 博客 URL（L225/L601）；Kaplan "LR horizon 未逐 run 调度" 的归因表述（L169-176）是否过强——Chinchilla 论文的批评点还包括分布外 N 外推等，教程只讲了 LR horizon 一种机制，需裁定是否补一句 "机制之一"。
4. **[P13-00-4] --full 未实测**：脚本 L670 "预计 ~3x smoke（~80s）" 与教程练习 1（L537-539 "约 3 倍 smoke 时间"）均诚实标注 "未实测" ✓ 口径合格，正式审确认无遗漏即可。

### 4.2 02 章 Data-Juicer YAML 与算子表准确性

**已预核为合理/正确的**：
- `document_minhash_deduplicator` 参数 `tokenization/window_size/num_permutations/jaccard_threshold/num_bands/num_rows_per_band` 与 Data-Juicer 真实 API 相符；L79 特意标注 "num_rows_per_band（不是 num_rows）" ✓。
- 112 = 14×8 与 FineWeb "5-gram、14 band × 8 row、阈值≈0.7" 口径一致，且 README L177、作业 Q4、面试直通车三处同口径 ✓。
- `dj-process --config` ✓；`words_num_filter(lang,min_num,max_num)`、`language_id_score_filter(lang,min_score)`、`alphanumeric_filter(tokenization,min_ratio)`、`word_repetition_filter(rep_len,max_ratio)`、`clean_html_mapper/fix_unicode_mapper/remove_header_mapper` 均为真实算子名。
- 审计 JSON（L124-133）已双重免责标注 "教学示意，以 stats/ 实际产物为准"（L124、L135-137）✓；性能表已标注 "官方 benchmark 数字，未经本机复现，仅量级参考"（L207-208）✓。

**正式审必查的疑点**：
1. **[P13-02-1] 内容自相矛盾（本轮最重要的 02 章发现）**：概念检验 Q1 答案首句 "先去重可省后续计算（同文档只算一次）"（L273）vs 陷阱 1 "纯'先去重'在大语料上代价高：……等于给垃圾也建签名"（L220-223）。两处最终都推荐 "轻过滤→去重→重过滤"，但**理由陈述方向相反**。裁定：改 Q1 首句为 "先去重看似省验证/后续算力，但去重本身是重算子……" 与陷阱 1 对齐；同时检查作业面试直通车 "去重放在质量过滤前还是后？两流派都讲得出消融理由（02 章练习 Q1）" 是否受牵连（该表述本身中立，可不动）。
2. **[P13-02-2] `python -m data_juicer.list_ops`**（L177，错误 2 的解法）：data_juicer 包是否存在此 `__main__` 入口存疑（官方文档列算子的路径是 `dj-ops` CLI 或 ops 文档页）。待联网核对；若失实，改 `dj-ops` 或 "查官方 ops 文档"。
3. **[P13-02-3] `llm_quality_score_filter` 的 `is_hf_model`**（L194）：真实参数名疑似 `is_api_filter`（走 API 时置 true；`api_or_hf_model` 本身是真实参数）。待联网核对后裁定教程 YAML 是否失实。
4. **[P13-02-4] "200+ 算子（58 过滤 / 95 清洗改写 / 12 去重）"**（L5）：分项和 165 < 200，需注明含 Ray 分布式变体/selector 等才到 200+，或与官方 ops 总数对表。待联网核对官方当前数字（版本敏感，建议教程加 "以所装版本为准" 缓冲语）。
5. **[P13-02-5] FineWeb 配置表**（L250-258：0.7/0.65/0.5/50/100000）：与作业 Q4/面试直通车同源，待联网对 FineWeb 论文核对（尤其语言阈值 0.65 与质量阈值 0.5 的归属：0.5 是分类器分还是教育分；min_words 50 是否 FineWeb 而非 Gopher 规则值）。这是 02 章与作业**共享口径**，改则三处联动。

### 4.3 作业 02 章相关题面（题 5 / 实验题）核对结论

- 题 5 引用 "02 章对照表第 4 行" → 02 章 L94 实际第 4行确为 "keep-first 丢弃 | 簇消解策略 + 可选'保留文本最长的'" ✅ 对应准确。
- 实验题 "用 02 章 YAML 跑 tiny_corpus、读 stats 追踪报告" ↔ 02 章 §4 审计节 ✅。
- 作业 Q4 的 S 曲线拐点算术 `(1/14)^(1/8)≈0.72`、`(1/28)^(1/4)≈0.44` 手算复核正确 ✅。
- 联动项：若 P13-02-1 改 Q1 答案，检查作业面试直通车 "两流派（02 章练习 Q1）" 的指向仍成立。

### 4.4 scripts 档位

| 项 | 档位 | 备注 |
|---|---|---|
| 00 `--mode fit` | 零 GPU ~2s，CPU 笔记本可跑 | 硬依赖 matplotlib（Agg+150dpi×2 图）；教程已标 "~2s" ✓ |
| 00 `--mode scan` | 单卡 4090 smoke ~25-30s | bf16 autocast；CPU 有降级分支（T=128,B=16,N≤0.03M）但 "耗时会显著增加" |
| 00 `--mode epoch` | 单卡 ~30-40s | CPU 降级到 0.1M 参数、R≤8 |
| 00 `--full` | 未实测，预计 ~80s | 脚本与教程均诚实标注 ✓ |
| 02 | 无脚本（dj-process CLI，CPU 即可） | 与 README 环境表 "任何机器全部内容" 一致（但该表未列 00 章 → 见 C1-4） |

正式审动作：核对 README 环境节补 00 章档位行（"00 章 fit 零 GPU/scan、epoch 单卡分钟级/CPU 可降级"），并把 matplotlib、scipy 列入 00 章依赖说明。

---

## 5. 格式规范预检（G1/G2/G4/G10/G13/G14）违反清单

> 判定以规范原文为准；以下为预检候选 + 预检已排除项。

**候选违规（6 项）**：
1. **[G1][P13-F1] 02 章缺 "参考资源" 节**：00 章有 "参考资源"（L595-605），02 章从 "课后作业" 直接跳 "下一步"，Data-Juicer/FineWeb/Gopher 的引用散在正文与 README，无集中清单。
2. **[G1][P13-F2] 02 章前置知识无链接**："**01 章**：MinHash/LSH 四阶段"（L21）纯文字；00 章/README 的前置知识条目均带 markdown 相对链接。
3. **[G2][P13-F3] 02 章 "概念检验/动手实践" 用粗体段落而非 `###` 标题**（L267、L296），00 章同功能节用 `### 概念检验`/`### 动手实践`/`### 扩展思考` → 两章结构层级不一致。
4. **[G2][P13-F4] 收尾节措辞不一致**：02 章 "学完本部分你能..."（L260）vs 00 章 "学完本章你能..."（L586）。
5. **[G4][P13-F5] README Broder 链接疑似张冠李戴**（L140）：论文标题 "Similarity Estimation Techniques from Rounding Algorithms"（STOC 1998）但 URL 指向 `cs.brown.edu/.../ugrad/2005/broder.pdf`（2005 本科论文库路径），高度可疑死链/错链。待联网确认后换官方 PDF 或引文页。
6. **[G4][P13-F6] 两个外链离线不可判**（不计数，列待办）：Lilian Weng 博客 URL（00 章 L603）、Jason Wei 博客 URL（L225/L601）。

**预检已排除**：
- G10 图片：00 章 2 张 png 存在且被正确引用（相对路径 `../scripts/output_*.png`）；02 章/README 无图，无悬空引用。
- G13 表格：00/02/README 全部表格列数一致、表头规范，未发现错列。
- G14 标点/空格：R≤4、"t/p"、"nat/token" 等符号用法两章一致；未发现全半角混用热点。
- G4 站内链接：00/02/README 全部 11 个站内相对链接目标已逐一验证存在。

**格式违规计数：确认候选 5 项（F1-F5）+ 待联网 2 项（F6 不计入）。**

---

## 6. 跨 Part 衔接（00 章与 Part 7/8 训练叙事）

| 衔接点 | 声明位置 | 预检结论 |
|---|---|---|
| 00 章前置 "Part 7 预训练流程：final_loss 就是这个量" | 00 章 L27-29 | Part7 README 存在 ✅；正式审确认 Part7 叙事确有 next-token/cross-entropy 即可 |
| 00 章 "scan/epoch 的小 GPT 是 Part8·01 的缩小版（LayerNorm+learned PE+MHA+ReLU FFN 经典款）" | 00 章 L32-34、脚本 L174-176 注释 | Part8·01 L3 逐字匹配 ✅；脚本 "不跨 Part import、自包含" 声明与实现一致 ✅ |
| 00 章附录 "涌现能力之争衔接 Part 8·07 评估学：指标连续性" | 00 章 L216-231 | 07_evaluation.md 存在 ✅；正式审确认其确有指标选择/污染话题 |
| 00 章附录 "MoE 粒度衔接 Part 7·03（手写过 MoE）" | 00 章 L233-241 | 03_gqa_and_ffn.md L326 "MoE：把 FFN 换成一群专家" ✅ 属实 |
| README 前置 "Part 8·07 污染去重是本章算法底座" | README L34-37 | 07_evaluation.md L177 n-gram 去污染 ✅ 属实 |
| 00 章 → 01 章 "下一步：MinHash+分带 LSH ~60 行" | 00 章 L607-613 | 01 章已审，链接有效即可 |
| 02 章 → Part 14 "下一步 vLLM" | 02 章 L351-355 | Part14 README 存在 ✅ |

**衔接审计正式动作**：只需核对 Part7 README 是否被 00 章描述为 "预训练流程" 的确切性（next-token prediction / cross-entropy 词汇是否出现），以及裁定 README 前置（Part8·07 必须掌握）与 00 章前置（Part7 必须）是否需要一句 "00 章另见 Part 7" 的互指——倾向不改，在报告中说明差异合理性。

---

## 7. 正式审执行顺序（留给 T1 执行者）

1. 联网核对批次（一次性做完）：Chinchilla Table 3 常数；Muennighoff "4/16 epoch" 措辞；Llama 3 15T；Data-Juicer `dj-ops`/`list_ops`、`is_hf_model` vs `is_api_filter`、算子总数分档；FineWeb 配置表 5 值；3 个博客/PDF 外链。
2. 00 章正文逐段过：确认 §4.1 四疑点 + §6 表中两项措辞核对；产出引用行号清单。
3. 02 章正文逐段过：裁定 P13-02-1 矛盾修改方案；P13-02-2/3/4/5 待联网结果逐条落锤。
4. README/路线图同步缺口：C1-1~C1-4 逐条给出最小修改建议（改路线图节点 13 三行、README 学习目标加一条、环境节加 00 章一行）。
5. 格式清单 F1-F5 落锤（改或豁免，写明理由）；F6 联网后定。
6. 汇总产出正式 review 报告：C1 缺口 4 + 格式候选 5 + 内容缺陷 3（P13-00-1 符号失真、P13-02-1 自相矛盾、其余视联网结果）。

---

## 8. 预审总结（一句话）

00 章质量显著高于 02 章：常数、推导、t/p 账、实测数字、环境标注全部经得起复算，唯一硬伤是脚本打印符号失真被教程照抄；02 章有一个理念自相矛盾（Q1 vs 陷阱 1）和 4 处待联网核对的 YAML/算子口径；最大的一致性缺口在**路线图节点 13 正文与 README 未同步 00 章**（而路线图其他位置已登记 00 章，属内部失步而非漏建）。
