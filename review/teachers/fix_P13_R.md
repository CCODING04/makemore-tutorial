# fix_P13_R — T2 整改教师修复报告（P13 R2 批次：00 章 / 02 章 / README / scripts 00）

> T2 整改教师 · 2026-09-04 · 依据 plan_P13R + 三学生 report_P13R + T0 必修清单。
> REPO 只读；全部修改落 REVIEW 镜像。**01 章已 closed，本轮零触碰。**
> 作业题面零改动（必修项均落教程侧），assignment pytest 无需重跑。

## 一、镜像清单（全部改动文件）

| 文件 | 性质 |
|---|---|
| `REVIEW/courses/Part13_data_engineering/scripts/00_scaling_laws.py` | 修改（L583 符号修复） |
| `REVIEW/courses/Part13_data_engineering/tutorial/00_scaling_laws.md` | 修改（数学 4 处 + t/p 外推 + 转录数字 + G2 + 图路径 + 耗时口径） |
| `REVIEW/courses/Part13_data_engineering/tutorial/02_data_juicer_pipeline.md` | 修改（Q1 对齐 + 实操三件套 + 口径 ×2 + 体例 ×4） |
| `REVIEW/courses/Part13_data_engineering/tutorial/README.md` | 修改（学习目标 + Broder 源 + 数量级口径 + 环境档位 + G2） |
| `REVIEW/courses/Part13_data_engineering/images/output_scaling_fit.png`、`output_scaling_epoch.png` | 新增留档（自 REPO scripts/ 复制） |
| `REVIEW/scratch/T2_P13R/00_scaling_laws.py`、`fit_out.txt` | G16 合规：脚本副本 scratch 实跑证据 |
| `REVIEW/ledger/ledger_P13.md` | 追加 R2 批次 17 条（C00×11 + C02×6，注明 C01 已 closed） |
| `REVIEW/outline_review/outline_suggestions.md` | 追加 roadmap 节点 13 未含 00 章 + 作业无 scaling 题 |

## 二、必修清单落点（9/9）

1. **脚本 fit 假正号（P0）**：`abs(fv-tv)/tv*100` → `(fv-tv)/tv*100`（带真实符号，与 epoch [4] 口径一致）；教程转录同步 `偏差 -11.2%（此处为低估）`。脚本侧 3 处假正号（E/A/alpha）随格式串一次修复。
2. **02 章 Q1 vs 陷阱 1**：Q1 答案改写为"两笔账"——省的是**下游算子算力账**，费的是**去重签名 + 垃圾簇幸存副本重过滤的质量账**，互指陷阱 1；三处（Q1/陷阱 1/最佳实践）口径现已同向。
3. **02 章实操三件套**：§1 新增"30 秒造出最小语料"块（20 条样例 `python -c` 命令，实跑验证 20 行合法 jsonl）；`list_ops` 补前置"需已 pip 装 py-data-juicer + 建议独立 venv + dj-ops/官方文档备选"；"200+ 算子"拆为官方宣传口径 vs 165 细分（差额=selector/checker/Ray 变体），加"截至撰写版本"限定 + Operators 文档链接。
4. **00 章数学补**：推导链补"两边同乘 N，认出 (6N/C)^β=D^(-β)"一步；第 5 步补 `R_eff=(a/L_val)^(1/γ)` 反解公式与 R=16 算例（6.2→0.39x）；Lagrange 名实——措辞改"约束最优化（代入消元）"+ 补名实说明（等价 λ 一阶条件）；练习 2 验收补方向自检（(a) 1.6988 < (b) ≈1.81，玩具反直觉，勿写反）。
5. **t/p 外推口径**：第 4 步末补警示块——闭式外推到 C≈5.9e23 给出 t/p≈90~93（真值复算 89.5 / 拟合参数复算 96.9），远高于实测谷底 20；外推按 ~C^0.097 上漂，引用 20 t/p 须注明"isoFLOP 实测谷底值"口径（S3 素材，Besiroglu 入口）。
6. **README**：学习目标补 00 章条目（推导+拟合+t/p 预算语言）；Broder 链接换 ACM DOI 官方源并加注换链原因；"4 个数量级"补口径（单次可处理数据规模 KB~MB → TB 级/十亿文档，非代码量/吞吐比值）；顺带补 00 章环境档位行（plan C1-4/S2 卡点 7）。
7. **02 章体例**：新增"参考资源"节（DJ/data-juicer-hub/FineWeb 论文+博客/Gopher/Lee 2107.06499/算子清单口径）；前置"01 章"加链接；`**概念检验**`/`**动手实践**` 升 `###` 标题；"学完本部分"→"学完本章"。
8. **格式 G2/G4**：G2——正文行内数学 LaTeX 化（00 章 ×5、README ×2；代码块内推导链保留 ASCII 与全教程体例一致）；G4——两张 PNG 复制进 REVIEW `images/` 留档，教程补"图产物路径"注明（脚本写 SCRIPT_DIR，非 cwd）。
9. **roadmap 项**：节点 13 未含 00 章（+作业无 scaling 编码题）已追加 `outline_review/outline_suggestions.md`，按其规则不改 docs/。

## 三、验证结果（必跑 3 项）

1. **脚本 scratch 实跑 fit 模式**（G16：先 cp 到 `scratch/T2_P13R/`）：偏差符号真实——`E -2.0% / A -11.2% / alpha -2.8% / B +1.3% / beta +0.2%`，与 S1 预测逐项一致；[3] 参数平均表、[4] isoFLOP 表逐数字不变。证据：`REVIEW/scratch/T2_P13R/fit_out.txt`。
2. **check_latex.py**：00 章 / 02 章 / README 三文件均 ✅ 0 问题。
3. **作业**：题面零改动，pytest 不触发；附带验证 tiny_corpus 生成命令实跑（20 行合法 jsonl）。

## 四、disputed 项（禁 wontfix，2 条待联网复核）

- **P13-C00-U-01**：Broder 原链（cs.brown.edu ugrad/2005 路径挂 STOC 1998 论文）离线 curl 不通，已换 ACM DOI `10.1145/276698.276733`——DOI 本身待联网确认。
- **P13-C00-U-02**：README 主源组织名 `datajuicer` vs `modelscope`，离线不可判，已按高置信侧（modelscope）改并删去会漂移的星数"7.0k"。
- 另：02 章"200+ 算子"与官方文档的具体对表、`dj-ops` 备选入口的版本适用性，均在台账注明"待联网对官方文档"。

## 五、通用问题（跨 Part 候选，≤3）

1. **"教程转录脚本输出"需在脚本变更时回归**：本轮唯一 P0 的根因是脚本展示层缺陷被教程忠实转录（两处"正确"地一起错）。建议终版加一条流程规则：凡教程声明"真实输出"，脚本改动后须 diff 教程转录段。
2. **宣传口径数字（200+ 算子 / 7.0k stars / 4 个数量级）三处同病**：无出处、无版本限定、细分与总数不自洽。建议全教程扫一遍"数字+量词"型宣传语，统一加"截至撰写版本 + 来源链接"。
3. **同一参数两口径并排不给说明**（112 vs 128-256；教程耗时单值 vs docstring 区间）：建议对"并列出现的同类参数"做一次交叉 grep，缺半句口径说明的补齐。

## 六、好写法候选（≤3）

1. **epoch [4] 偏差列的 `(vl-p)/p`**：符号真实、正负如实——正是本轮把 fit [2] 修齐的样板（同一脚本内自我对照即可发现不一致）。
2. **02 章审计 JSON 的双重免责**（"教学示意，以 stats/ 实际产物为准"出现于示例前后两处）与性能表"官方 benchmark、未经本机复现"的口径声明——宣传性数字的正确写法。
3. **00 章"为什么网格要跨 5 个数量级"的失败记录写法**（B 误差 262% 的首版教训）：把参数可辨识性讲成可复现的实验事故，学生报告三方一致将其评为最值钱段落。
