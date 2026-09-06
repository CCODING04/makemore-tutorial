# S2 学习报告·P13(00/02章)

- 审计身份：学生 agent S2（实操薄弱型——先照抄跑通，再回头读理论）
- 审计材料：`tutorial/00_scaling_laws.md`、`tutorial/02_data_juicer_pipeline.md`、`tutorial/README.md`（01 章按任务跳过）、`../scripts/00_scaling_laws.py`
- 实操记录：fit / scan / epoch 三模式在本机全部跑通（RTX 4090，`.venv` 全路径，`MPLBACKEND=Agg`，raw 输出存于 `makemore-tutorial-review/scratch/S2_P13R/{fit,scan,epoch}_out.txt`）；未安装 Data-Juicer（按要求只做声明完整度评估）；未做 assignment_13

## 总分：8.5 / 10

00 章"教程↔脚本"一致性近乎完美（三模式输出与教程截选**逐值一致**，含所有 loss 小数位——种子固定，可复现性极强），扣分集中在：02 章"读得爽但跑不起来"（无样例数据、无配套脚本、口径无出处），以及 00 章/README 的环境依赖声明缺口。作为实操薄弱型学生，我的体感是：**00 章是我跟得最稳的一章，02 章是我第一步就会卡住的一章。**

## 卡点清单

| # | 严重度 | 位置 | 卡点 | 学生视角的后果 |
|---|--------|------|------|----------------|
| 1 | **高** | 02 章 §1 + README 导航表 | 02 章没有配套脚本/样例数据：`dedup_demo.yaml` 指向 `./tiny_corpus.jsonl`，但全文没有生成这条语料的命令；README 导航表 02 章对应脚本一栏明写 "—（YAML/CLI 实操）" | 照抄 YAML 后 `dj-process` 第一步就报找不到数据。错误 1 的 txt→jsonl 片段救不了一步——学生得自己编 10~20 条英文文本，实操薄弱型直接卡死在起点 |
| 2 | **中** | 02 章引言 / README | "200+ 算子（58 过滤 / 95 清洗改写 / 12 去重）"口径**无出处**：没链接 data-juicer 的 Operators 文档页；且 58+95+12=165，距 200+ 差 35+，缺口（selector/mapper 其他类/Ray 变体？）未解释，细分与总数不自洽 | 无法核实的宣传数字 + 自相矛盾的细分，学生引用到面试里被追问出处就露怯 |
| 3 | **中** | 02 章"错误 2 解法" | `python -m data_juicer.list_ops` 未声明前置条件=已成功安装 py-data-juicer；安装命令在 §1（远离此处），且全章未提示：无 Python 版本要求、data-juicer 依赖 torch 可能与课程 .venv 里的 torch 版本冲突、未建议独立虚拟环境 | 没装成功的同学跑这条得到 `No module named data_juicer`，与"错误 2 的算子参数错误"症状对不上，排障方向被带偏 |
| 4 | 轻 | 02 章 §3 | 标题承诺"照抄即用"，但第 6 行 `document_minhash_deduplicator: {...}` 是占位符，`{...}` 不是合法 YAML，必须回 §1 抄参数 | "照抄即用"承诺过强，第一次照抄直接 YAML 解析报错 |
| 5 | 轻 | 02 章 §4 | `cat output/audit.json` 是**不存在的产物路径**（教程自己承认是"教学示意"，真实产物是 `stats/` 目录，且有引言+脚注双重免责） | 免责做得好，但实操薄弱型的第一反应仍是先跑这条命令，`No such file` 后需要回读两条注释才能明白 |
| 6 | 轻 | 00 章 §代码实现 / 脚本 | 图保存到**脚本所在目录**（`SCRIPT_DIR/output_scaling_*.png`），不是当前工作目录；教程没写"图会生成在哪"，只给了 `../scripts/*.png` 的引用图 | 从 scratch 目录运行的学生跑完在 cwd 找不到图（我实测确认：两张 PNG 落在 `courses/Part13_data_engineering/scripts/` 下，教程引用相对路径本身解析正确） |
| 7 | 轻 | 00 章 / README | 00 章依赖未声明：脚本 import torch/scipy/numpy/matplotlib，教程只在"实测环境标注"里作为环境记录提及，README"环境与版本策略"只写了 01/02 章（"01 章无需任何安装"——但 00 章要装 scipy/torch 这件事没写） | venv 缺 scipy 的学生第一步 `import scipy` 就 ImportError，教程没有给一行 `pip install` 兜底 |
| 8 | 轻 | 00 章模式列表 | 耗时口径：教程写 scan ~25s / epoch ~31s；本机实测网格 31.1s / epoch 总 40.6s（同为 4090）。脚本 docstring 的 "scan ~25-30s / epoch ~30-40s" 才准确 | 量级无误导，但按 ~31s 设超时的学生可能误以为卡住 |
| 9 | 轻·脚本 bug | 脚本 fit [2] / 00 章实测转写 | 偏差符号恒为 "+"：`abs(fv-tv)/tv*100:+.1f%` 先取 abs 再强制正号格式，实测 A 拟合 360.954 < 真值 406.400（应为 **-11.2%**），打印成 "+11.2%"；教程原样转写了这个带误导的符号 | 方向信息丢失：学生看不出单次拟合偏大还是偏小（对"估计量方差"这个教学点无碍，但符号是假的） |

## 分章评分

| 章节 | 得分 | 一句话评价 |
|------|------|-----------|
| 00 章 + 脚本 00 | **9.5/10** | 三模式实测全通过、输出与教程逐值一致、三个坑复盘值钱；只扣环境声明与图位置说明、偏差符号小 bug |
| 02 章 | **7/10** | 概念地图与算子对照表质量高、免责声明诚实；但作为"YAML/CLI 实操"章，实操链条在"无数据、无脚本、口径无出处"三处断裂 |
| README | **8/10** | 导航/前置知识/位置图清晰；环境策略表对 00 章依赖只字未提，"200+ 算子"口径在此复述也无出处 |

## 一致性核对表

### 00 章 ↔ 脚本（实测核对，scratch 输出为证）

| 教程声明 | 脚本/实测 | 判定 |
|---|---|---|
| `--mode fit` 零 GPU ~2s | 实测 ~2s（拟合 0.5s），EXIT=0，✅ 全部参数 <5%（E 0.13%/A 2.43%/α 0.53%/B 3.32%/β 0.44%） | ✅ 一致 |
| fit [2] 单次抽取"A 360.954 +11.2%" | 逐字符一致；但符号应为 -11.2%（见卡点 9） | ⚠️ 转写忠实，符号源 bug |
| fit [4] isoFLOP "共 7 档 C，输出截选 4 档"，t/p 11.0→21.5 | 实际 7 档全打印，截选 4 档数值一致（11.0/13.7/17.2/21.5） | ✅ 一致 |
| `--mode scan` 单卡 ~25s | 实测 31.1s 网格 + 1.1s 生成 ≈ 33s（同 4090） | ⚠️ 量级一致、偏慢 ~25% |
| scan 语料池 24.0M / vocab=65 / E=2.616 / ln(65)=4.174 | 逐项一致 | ✅ 一致 |
| scan 网格表 9 行 val_loss（2.9687…2.7575） | 9 行逐值一致 | ✅ 一致 |
| 自由拟合 E=0.0000 A=1.04 α=0.100 B=4.18 β=0.031；固定 E 拟合 A=421.3 α=0.742 B=52.7 β=0.354，残差均值 +0.02% 最大 2.08% | 逐值一致 | ✅ 一致 |
| 学生结论 t/p≈62~292 | 62.3 / 129.6 / 292.4 一致 | ✅ 一致 |
| 练习 1 `--mode scan --full` = 4×3 网格、N 到 3M、D 到 60M、约 3 倍 smoke | 脚本 full 分支 N=[0.12M,0.26M,1M,3M]×D=[6M,20M,60M]，注释 ~3x/80s，且诚实标注"未实测" | ✅ 一致 |
| 形状追踪（B=32,T=256 → x,y=(32,255) int64；token 池 uint8） | ChunkSampler/E make_resampled_corpus 实现一致（epoch 侧 train_ids 为 int64，教程未标注但不冲突） | ✅ 一致 |
| epoch：R=[1,2,4,8,16]，R=16 train=0.9505 val=1.8849 反升；幂律 L=2.567·R^-0.169；R=16 折扣 0.39x | 逐值一致（实测总耗时 40.6s vs 教程 ~31s，见卡点 8） | ✅ 数值一致 / ⚠️ 耗时口径 |
| epoch [5] 表下的 R≤8 波动解释（1.21x 属拟合残差噪声） | 教程**额外补充**的诚实注释，脚本只有一行括注；两者不矛盾 | ✅ 教程增补合理 |
| 图片 `../scripts/output_scaling_fit.png`、`output_scaling_epoch.png` | 两文件真实存在且本次运行重新生成到 scripts/ 目录（相对路径从 tutorial/ 解析正确）；教程未说明生成位置（卡点 6） | ✅ 引用有效 / ⚠️ 位置未声明 |

### 02 章 YAML/命令 ↔ 可拼凑性核对（未装 Data-Juicer，静态审计）

| 项 | 教程给了什么 | 能拼凑吗 |
|---|---|---|
| 安装 | `pip install py-data-juicer`（§1 与 README 两处，包名与 PyPI 一致），注明重依赖为可选 extras、CPU 可跑 | ✅ 安装声明本身到位 |
| YAML 骨架 | `dataset_path`/`export_path`/`process` 三段齐全，注释逐行解释，`num_rows_per_band` 参数名有专门警告，14×8=112 自洽 | ✅ YAML 本身可照抄 |
| 输入数据 | 只有一行注释"每行一条 {"text": "..."}（jsonl，非 JSON 数组）"；无生成命令、无样例文件 | ❌ 不可（卡点 1） |
| `dj-process --config dedup_demo.yaml` | 命令给出，产物说明给了 | ✅（前提是装好+有数据） |
| `python -m data_juicer.list_ops` | 命令在"错误 2 解法"给出 | ⚠️ 未声明前置=已安装（卡点 3） |
| "200+ 算子（58/95/12）" | 引言与 README 复述两次，无任何出处链接，细分合计 165 | ❌ 不可核实（卡点 2） |
| §3 完整管线 | 6 个算子给了 5 个完整参数，末位 `{...}` 占位 | ⚠️ 半可（卡点 4） |
| §4 审计 | `cat output/audit.json` 为示意，已双注免责（真实产物 stats/，字段随版本变） | ⚠️ 已缓解但首跑必困惑（卡点 5） |
| 算子名/参数（words_num_filter、language_id_score_filter、alphanumeric_filter、word_repetition_filter、document_minhash_deduplicator、document_line_deduplicator、remove_header_mapper、llm_quality_score_filter 等） | 与本 agent 对 py-data-juicer 的既有知识记忆一致，**离线无法最终核实**；教程仅对审计 JSON 免责，未对算子名加"以 list_ops 输出为准"的提示 | ⚠️ 待装后验证 |

## "装 Data-Juicer 实操路径"可行性评估（不装，只评声明完整度）

- **已声明（好的部分）**：PyPI 包名正确（py-data-juicer）；CPU 即可跑小管线的承诺明确；重依赖（Ray/多模态/audio）为可选 extras 的说明出现两次（README + 02 章）；`dj-process` CLI 与 `python -m data_juicer.list_ops` 排障命令给了；错误 1/2 覆盖了"格式不对/参数名不对"两类最常见首跑失败。
- **未声明（缺口）**：① Python 版本门槛只字未提（data-juicer 对 Python/依赖有版本约束，课程环境标注 Python 3.12 是否兼容未确认）；② **torch 依赖冲突风险未提示**——data-juicer 会拉自己的 torch/datasets/pyarrow 版本，装进课程 `.venv` 有覆盖现有 torch 2.6.0 的风险，教程没建议"独立 venv"，这对本课程多 Part 共用 .venv 的结构是实际风险；③ 装完后的**最小验证命令**（如 `dj-process --help` 或 `list_ops` 的预期输出样例）没有给，学生无法判断"装好了没"；④ 配套样例数据缺失（卡点 1），装好后的第一步依然是断的。
- **结论**：声明完整度约 **60%**——"装得上"（安装命令真实），但"第一次跑通的最后一公里"（独立环境建议 + 验证命令 + 样例数据）缺三件。对 S2 这类实操薄弱型学生，02 章现状是"读完觉得会了，打开终端先卡 20 分钟"。

## 只改 3 件事

1. **02 章 §1 加"30 秒起步块"**：3 行造数据命令（`python -c` 写 ~20 条 `tiny_corpus.jsonl`）+ 1 行安装验证（`python -m data_juicer.list_ops | head` 或 `dj-process --help`）+ 1 句"建议独立 venv，避免覆盖课程环境的 torch"。一石三鸟：卡点 1、3 和环境冲突风险全消。
2. **给"200+ 算子"补口径出处**：引言处加一句出处链接（Data-Juicer 官方 Operators 文档/GitHub awesome-operations），或把口径改自洽："165+ 核心算子（58 过滤/95 清洗改写/12 去重），含分布式变体共 200+"——并注明"以下载版本的 `list_ops` 输出为准"。
3. **00 章"代码实现"节加 2 行环境说明**：`pip install torch scipy matplotlib numpy`（或"需 Part 7 的 venv"）+ "两张图输出到 scripts/ 目录（与教程引用路径一致）"；顺手把脚本 fit [2] 的 `abs(...):+.1f` 改成带真实符号的 `:.1f`。

## 最喜欢 3 处

1. **00 章"实验设计复盘：三个坑"**（语料坑/网格坑/步数坑）：与脚本注释逐条互相印证（`lr_at()` 的 Kaplan 偏差红线注释是全脚本最值钱的段落），"toy 实验不是缩小版的真实实验"这句值得裱起来。
2. **scan 模式敢把病态摆上台面**：自由拟合 E=0.0000（顶边界）原样打印，再演示"固定 E=可独立测量的熵下界"救活拟合（残差 2%）——"先展示失败再教解法"在教程里极罕见，且我实测复现了全过程。
3. **02 章"手写 ↔ Data-Juicer 逐步对照"表** + `num_rows_per_band`（不是 num_rows）的实战警告：一张表把 01 章的 60 行映射到工业算子，参数名警告一看就是真踩过坑的人写的；性能数据表主动标注"未经本机复现，仅量级参考"的诚实口径也是加分项。

---
*审计耗时 ~12 分钟；raw 输出：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P13R/`（fit/scan/epoch 三个 tee 文件）*
