# Part 12（微调实战：LLaMA-Factory）分章审计计划 — T1 主教预审

- 预审日期：2026-09-04
- 对象：`REPO/courses/Part12_finetune_llamafactory/`（tutorial 3 个 md 共 991 行 + scripts 1 个 .py 共 311 行）+ `assignments/assignment_12/`（assignment.md 140 行 + 骨架/测试）+ `docs/course_roadmap_v3.md` 节点 12（L442-468）
- 性质：全课程唯一"纯外部工具链" Part——教程大部分命令跑在 LLaMA-Factory 上游仓库，可复现性天生弱于手写脚本 Parts；审计重心 = **手写↔工具互证的数学口径** + **版本敏感声明的完备性**
- 预审方式：只读通读 + grep/手算交叉核对，未运行任何脚本、未改任何课程文件（合规：未读 REVIEW/assignment_reference 等）

---

## 一、C1 标题-内容对应表（4 文件逐个）

| # | 文件（行数） | 标题 | 实际内容 | 对应脚本/作业 | C1 判定 |
|---|---|---|---|---|---|
| 0 | README.md (179) | Part 12 微调实战 — LLaMA-Factory（LoRA/QLoRA/DPO 全流程） | 学习目标、2 章导航、前置（P8 08/02/03→P10→P11）、链路位置图、成本效果表、LoRA 低秩推导、演进史、venv 版本策略、硬件档位表、学习地图 | 脚本 `01`；Assignment 12 | ✅ 基本对应；**效果证据表（L48-53）无来源标注**；LoRA 推导在代码块（G2，见 §五） |
| 1 | 01_handwritten_sft_lora.md (417) | 手写 LoRA SFT：LLaMA-Factory 自动化的到底是什么 | 五步管线（预热→注入→SFT→推理→merge）、形状追踪图、五步↔yaml 逐字段对照表（核心产出）、手写vs工具差距表、3 错误调试、性能表、3 陷阱、LoRA/QLoRA 配置推荐、3 概念检验 + 3 练习 | scripts/01_handwritten_sft_lora.py ✅（~40 秒 CPU/GPU）；Assignment 12 | ✅ 对应；**宣称数字群漂移（~3秒/~250行）**、A 初始化口径、cutoff_len 映射松弛（见 §三/§七） |
| 2 | 02_llamafactory_workflow.md (395) | LLaMA-Factory 工作流：identity → QLoRA 7B → export → DPO-LoRA | 6 节命令流水线（环境/identity 0.5B/WebUI/QLoRA 7B/export+chat+api/DPO-LoRA）+ 总账表 + 4 错误调试 + 性能表 + 3 陷阱 + 配置推荐 + 3 概念检验 + 2 练习 | —（CLI 实操，README 已声明）✅ | ✅ 对应；**QLoRA 数学缺"分页优化器"（roadmap 要求）**、双量化 0.37GB 口径、DPO lr 待核（见 §三/§七） |
| 3 | assignment.md (140) | Assignment 12：微调实战 | 题 1-4 各 25 分（参数账/合并数学/零初始化/显存账）+ 题 5 stretch（+20 不计基础分，None→SKIP）+ 2 实验题 + 5 思考题带答案 | finetune_exercises.py + test（pytest） | ✅ 对应；**题 4 括号说明"fp32 参数+梯度+两动量"（=16B）与 12B/参数公式自相矛盾** |

**导航链**：README ↔ 01 ↔ 02 ↔ Assignment 12 链路完整；README 末尾 ← Part 11 / → Part 13 目标均存在 ✅。01/02 章末"去 Assignment 12"不带题号 → 无 G17 风险 ✅。

## 二、学生单元划分（三学生 × 三单元）

| 单元 | 文件 | 体量 | 难度画像 | 审计重点 |
|---|---|---|---|---|
| **单元 A** | README + 01 章 + scripts/01 逐行 | 596 行 md + 311 行 py | ★★★★（本 Part 数学顶峰：A/B 初始化、α/r、形状账、merge 精确加法） | 教程↔脚本逐行互证（6,144/200,664=3.1%、1,536/层、18 倍、loss 3.572→0.076、2/3、max\|Δlogits\|）；LoRA 数学三处口径（§三-1）；G2 代码块推导 |
| **单元 B** | 02 章全章 | 395 行 | ★★★★（版本敏感命令最密集；QLoRA/DPO 声明） | 6 节命令逐行对照安装版 examples/；QLoRA 声明四要素（§三-2）；Qwen 依赖档位（§三-3）；性能表来源列 |
| **单元 C** | assignment.md + finetune_exercises.py + test_finetune_exercises.py + roadmap 节点 12 对表 + 跨 Part 一致性 | 140 行 + 骨架/测试 + 交叉 | ★★★（纸笔题为主，口径矛盾藏在文字里） | 题 4 显存账自洽性；题 1-3/5 公式与教程/脚本三方同源（G15）；思考题 5 答案 vs 教程；roadmap 节点 12 逐句对表；P8 02/03/08 章口径抽查 |

- 学生 1 → A；学生 2 → B；学生 3 → C。单元 B/C 学生需回看 01 章对照表作为接口（02 章每个 yaml 字段都引它）。
- 每个学生通读 README 一次（版本策略/硬件档位表是全局口径源，问题由主教裁定）。

## 三、本 Part 特有审计要点（主教逐条给出预判，执行学生定级）

1. **LoRA 数学（单元 A 主责，教程↔脚本↔作业三方同源）**
   - 低秩分解：ΔW = B·A，B∈R^{d×r}、A∈R^{r×k}，参数量 r(d+k)；压缩比 256 = 4096²/(8·8192) ✅ 手算通过。脚本 L114-115 形状 (r,in_f)/(out_f,r) 与教程形状图 ✅ 一致。
   - α/r 缩放：前向 `y = Wx + (α/r)·BAx`、合并 `W' = W + (α/r)·BA`——教程/脚本 L127/L229/assignment 题 2 三处一致 ✅（G15 通过）。注意 **roadmap 节点 12 写"合并数学 W'=W+BA"漏 (α/r)** → 缺口（见 §七-7）。
   - 注入矩阵口径三层声明是否齐全：脚本只注 MLP 两个 Linear（L136-137）→ 01 章 L98 已声明"与脚本同口径" ✅；真实 LF 常用 q_proj/v_proj 或 all（L133、L150）✅。待核：01 章"手写 vs 工具"表未提**注入位置不同会使 3.1% 与 LF 实际可训练比例不可直接比**——是否需一句声明。
   - **A 初始化三处口径**：README L113"A 用高斯初始化"（未给分布）／01 章理论段 L56"A ~ N(0, σ²)"（σ 未定）／形状图 L112"N(0,1)/√r"／脚本 L114 `randn/√r`（=N(0,1/r)）。四处并存，P8 08 章为 N(0,1/r)——建议统一为"高斯，常用 std=1/√r"单一表述（G15）。
   - B 零初始化：起点 ΔW=0，assignment 题 3 用 Frobenius 范数表述"起点无损" ✅；思考题 Q2"若 A 也为 0 则双零死鞍点"梯度论证 ✅。
   - **merge 双加 bug 的信息倒挂**：脚本 L117/L218-219（is_merged 哨兵 + "Wx+2·BAx 实测 ≈2.9"）比教程详细——01 章错误清单（device/dtype/mask 三条）**没有** merge 双加条目。教程-脚本信息倒挂，执行时裁定是否补进 01 章调试清单（G15 精神）。
2. **QLoRA 声明四要素（单元 B 主责）**：NF4 ✅（02 章 L62/L112"为正态分布设计的 4bit 格式"）；双量化 ✅（L55-58/L114）但 **"额外节省 ~0.37GB"系 QLoRA 论文 65B 模型 / 0.37 bit-参数 的数字直接套到 7B**（7B 按同口径应 ≈0.32GB）→ 数字口径缺口；**分页优化器（paged optimizer）全章 0 提及，而 roadmap 节点 12 明确要求"NF4 + 双量化 + 分页优化器各省什么"** → C1 缺口；"A/B 保持 bf16 训练不被量化" ✅（L64、概念检验 Q1、assignment Q4 一致）。assignment 题 4 公式 `7×4/8 + 20M×12B = 3.74GB` 手算 ✅，但括号文字自相矛盾（§七-3）。
3. **Qwen 权重依赖档位说明（缺位）**：02 章 §1 用 Qwen2.5-0.5B-Instruct、§3-§5 用 Qwen2.5-7B-Instruct（下载 ~15GB），assignment 实验题跑"QLoRA 7B × identity"——但 README 环境节只讲 venv 安装，**全 Part 无一处声明模型下载量/磁盘占用/HF 镜像变量（HF_ENDPOINT）/断网时哪些步骤不可跑**。对照 P8 的做法（模型缺失优雅退出 rc=0 有声明），应登记为档位声明缺口。
4. **yaml 字段逐行对照准确性**：01 章对照表 6 行字段名（template/train_on_prompt/lora_target/lora_rank/lora_alpha/dataset/dataset_info.json/learning_rate/num_train_epochs/per_device_train_batch_size/cutoff_len/export）均为 LF 真实字段 ✅；`pad_batch() ↔ cutoff_len` 映射松弛（cutoff_len 是截断长度，不是 padding 机制；括号里 packing 说明反而准确）→ 建议改写映射措辞；01 章 Q1 的 `additional_target` ✅ 真字段。02 章命令字段 quantization_bit/double_quantization/quantization_type/pref_beta/pref_loss/export_size/export_legacy_format 均 LF 现行字段名 ✅，执行时逐个对安装版 schema 复核。
5. **版本敏感命令的可复现性声明**：现状 = README"跟随 latest（耦合轻）"策略 + 02 章 3 处文件名⚠️（qwen_lora_sft.yaml→qwen3_lora_sft.yaml L90-91、qwen3_lora_sft_otfq.yaml L111、qwen3_lora_dpo.yaml L136）——声明意识好但**无版本锚点**。审计要点：① 02 章错误 1 的模板清单路径 `extras/constants.py`（L174）同样版本敏感（新版模板注册已迁至 data/template.py）却**未**加⚠️ → 补；② 执行审计时 `llamafactory-cli version` 的实测版本号应写进审计记录，评估"跟随 latest"策略是否需要补一行"本文命令在 vX.Y 验证于 2026-0X"；③ 02 章 §5 叙述 DPO "lr 用 5e-6 量级" vs 所引官方 qwen3_lora_dpo.yaml 的默认 lr（需打开安装版核实是否一致，不一致则教程须注明" yaml 默认值为准"）。
6. **export/DPO-LoRA 流程**：§4 export 命令字段 ✅；§4→§3 的 output_dir 路径衔接（saves/qwen7b-qlora）✅ 前后一致；§5 pref_beta=0.1 与 P8 03 章 DPO β=0.1 默认 ✅ 同口径；rewards/margins 读法 ✅；概念检验 Q2"不合并场景=vLLM multi-LoRA"与 assignment Q3 ✅ 双向一致。
7. **脚本档位**：唯一脚本 `scripts/01_handwritten_sft_lora.py`——CPU/GPU 均可、无外部依赖（仅 torch）、docstring "~40 秒"、无 flush=True、无环境变量短程档。G8 按字面适用于 >1 分钟脚本，40 秒宣称处于边界——执行实测若超 1 分钟需按 G8 补；"~3 秒"（教程）vs "~40 秒"（脚本）冲突必须先裁决（§七-4）。02 章无脚本（CLI 实操）且 README 导航表已声明"—" ✅。

## 四、scripts 运行档位表

| 脚本 | 档位/耗时（docstring 实据） | toy 档可行性 | 备注 |
|---|---|---|---|
| 01_handwritten_sft_lora.py | ~40 秒，CPU/GPU 均可，零外部依赖（仅 torch） | ✅ CPU 直接跑 | 教程称 ~3 秒（冲突待裁决）；无 flush/G8 短程档（边界豁免待实测）；教程引用的全部输出数字（6,144/200,664、3.572→0.076、2/3、max\|Δlogits\|<1e-4）以复跑为准（G6/G14） |

02 章命令流水线（identity 0.5B / QLoRA 7B / export / DPO）依赖 LLaMA-Factory 独立 venv + Qwen 权重下载，审计执行按 G16 精神在 scratch 建 venv，默认只验证命令块字段自洽与声明完备性，不强求跑通 7B 档。

## 五、格式规范预检（G1/G2/G4/G10/G13/G14）违反清单

> 按战役规范归类；执行学生对照规范原文复核定级。

- **G2（数学一律 LaTeX，禁止代码块承载数学推导）— 3 处实锤 + 1 处待裁**
  1. README L89-109：LoRA 低秩分解"推导过程 Step 1-3"整体放在 ``` 代码块。
  2. 01 章 L54-72："数学推导：LoRA 的初始化和缩放"Step 1-3 在 ``` 代码块。
  3. 02 章 L46-59："QLoRA 的量化过程"Step 1-3 在 ``` 代码块。
  4. 待裁：01 章 L101-137 形状追踪 ASCII 图承载前向形状推导（属图示还是推导，主教裁定；倾向改绘为 G4 图 + 数值表）。
- **G4（能画则画）— 1 项结构性缺口（全 Part 无 images/ 目录）**：候选图位 ≥3——01 章五步 loss 曲线（3.572→0.076）、02 章 DPO rewards/margins 示意、assignment 实验题的 QLoRA 显存曲线（学生产物，教程侧应给示例图位）。执行时按规范决定补图清单。
- **G13（外部来源数字双列制）— 1 处 + 1 处并案**：README L48-53 效果证据表（~120GB/~6GB/90%+/1-2h）无来源列（01/02 章同款性能表都有"数据来源"行，README 漏）；README L6/L173 的 74.4k/75.2k star 数无截至日期（外部数字漂移源）并案处理。
- **G14（实跑口径声明）— 2 处**：① 01 章 L94"以上为脚本真实输出（RTX 4090 / CPU 均可复现，~3 秒）"——与脚本 docstring "~40 秒"冲突且无 device/seed/torch 版本口径（G6 同根，最高优先）；② 02 章 §1 标题"小时级内出结果" vs 同章性能表 0.5B "~10min" 轻漂移。
- **G1（编号点换行）— 0 实锤**：grep ①②③④ 无命中；各"根本限制/性质"均为标准列表 ✅。
- **G10（正文代码可拼凑运行）— 0 实锤，1 项低风险**：正文代码块均为"症状-解法"碎片（体例如此，非拼凑主线），变量名与脚本一致（labels[:n_prompt]、apply_lora、merge_lora）✅；练习骨架 TODO 与脚本签名同型 ✅。低风险项：练习 1 提示 `apply_lora(model, r=8, alpha=16)` 与脚本默认 (r=4, alpha=8.0) 取值不同（示例值，非冲突，可忽略）。

**格式违规计数：8 处（G2×3 实锤 + G2×1 待裁 + G4×1 + G13×1 + G14×2）。**

## 六、跨 Part 一致性（重点：与 Part 8 的 SFT/DPO 口径）

1. **LoRA 与 P8 08 章**：ΔW=(α/r)BA、B 零初始化、merge 精确加法、只注 MLP 的诚实声明——P12 与 P8 08 章完全同口径 ✅；P12 README L3"Part 8 用几十行手写了 LoRA"与 P8 10_lora_from_scratch.py（3.4% 实证）呼应 ✅，roadmap 节点 12"先跑 P8 脚本 10 再进 QLoRA"的学习路径两侧文档均有 ✅。唯一残留：A 初始化泛写（§三-1）。
2. **SFT/masking 与 P8 02 章——措辞口径缺口**：P12 README L29 与 01 章 L20 均把 "prompt masking（**labels=-100**）"归为 P8 02 章内容；grep 证实 P8 02 章**无 -100/ignore_index 字样**（其实现为乘法 mask 四步）。两种实现等价，但"引用出处措辞失实"——建议改为"prompt masking（P8 用乘法 mask，本 Part 脚本用 labels=-100，等价实现）"。
3. **DPO 与 P8 03 章**：β=0.1 ✅；(prompt, chosen, rejected) 语义同源 ✅；"越靠后 lr 越小"呼应 Part 7 05 章 ✅——双向引用链完整。
4. **与 roadmap_v3 节点 12 的偏差**：教程 2 章/脚本 1 个/作业 4+1 题/实验 2 项/unsloth 延伸——逐句对表全部吻合 ✅；两处措辞缺口：①"分页优化器"教程缺位（§三-2）；②"合并数学 W'=W+BA"漏 (α/r)（§三-1）。学习验证"verify_paper_formulas.py 的 LoRA 断言"指向仓库级脚本，执行时确认该文件存在且含 LoRA 断言。
5. **与 P10/P11/P13/14 接口**：多卡→P10 FSDP ✅、DPO 对照→P11 ✅、下一步→P13 ✅、multi-LoRA 部署话题留 P14（02 章 Q2 提 vLLM，衔接自然）——无断链。

## 七、C1 缺口汇总（预审实锤 10 项，执行学生按清单定级）

1. **QLoRA"分页优化器"缺位**：roadmap 节点 12 明确要求，02 章通篇无（grep 实证）。
2. **双量化"~0.37GB"口径**：论文该值对应 65B/0.37 bit-参数，7B 应 ≈0.32GB——数字出处错位。
3. **assignment 题 4 自相矛盾**："fp32 参数+梯度+AdamW 两动量"（=16B/参数）vs 公式 12B/参数（12=2+2+8 bf16 口径，01 章练习 3 同）；思考题 Q1"约 12 字节"承 12B 口径——三处需统一拆解文字。
4. **宣称数字群漂移**：教程"~3 秒" vs 脚本"~40 秒"；教程"~250 行" vs 脚本实长 311 行；02 章"小时级" vs 性能表 "~10min"。
5. **A 初始化四处口径**（README 泛写 / σ² 未定 / N(0,1)/√r / 脚本 N(0,1/r)）——统一为单一事实源（G15）。
6. **Qwen 权重依赖档位说明缺位**：下载量/磁盘/HF 镜像/断网降级路径全无（§三-3）。
7. **roadmap"合并数学 W'=W+BA"漏 (α/r) 缩放**——与教程/作业/脚本三方口径差。
8. **"labels=-100"归属 P8 02 章的措辞失实**（P8 实为乘法 mask）——跨 Part 引用措辞修正。
9. **02 章错误 1 模板清单路径 `extras/constants.py` 版本敏感未加⚠️**（新版在 data/template.py）——与 3 处文件名⚠️的声明标准不齐。
10. **DPO lr=5e-6 叙述 vs 官方 yaml 默认值待核**；连同 6 个命令块的"跟随 latest"策略是否补实测版本锚点，执行时一并裁决。

## 八、学生执行清单（各单元通用动作）

1. 逐文件 C1：标题/小节 vs 内容 vs 对应脚本/命令 vs 作业映射核对（本计划 §一 表为底稿；§七 10 项逐个定级 P0/P1/P2）。
2. 数学-代码三方互证：LoRA/QLoRA 每个公式在 教程↔脚本↔assignment 找齐三处并比对（允许读脚本全文，禁止运行课程文件；外部命令块在 scratch 验证）。
3. 实测数字溯源：教程引用的每个脚本输出数字（6,144/200,664、3.572→0.076、2/3、max|Δlogits|≤2.4e-06、3.74GB、256 倍、18 倍）标注"脚本声明/章内引用/无出处"三态；01 章数字群由单元 A 复跑脚本核对（G16：先 cp 到 scratch）。
4. 版本敏感清单：02 章全部命令块逐个标注"字段名/文件名/路径 三查"结果，补齐缺声明处（§七-9/10）。
5. 格式扫描：按 §五 清单位置复核定级；G2 三处代码块推导的改写方案（LaTeX 化）随审计报告给出。
6. 产出：单元审计报告（缺口/格式/一致性/加分点 四类；加分点候选：merge 双加 bug 进 01 章调试清单、vLLM multi-LoRA 与 P14 的显式衔接），交主教合并。
