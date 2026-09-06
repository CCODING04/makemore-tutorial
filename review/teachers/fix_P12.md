# Part 12（微调实战：LLaMA-Factory）T2 整改报告

- 整改日期：2026-09-04 · 执行：T2 整改教师
- REPO 只读；全部修改写入 REVIEW 镜像；脚本先 cp 到 `REVIEW/scratch/t2_P12/` 实跑（G16）
- 环境：RTX 4090（课程 .venv，torch 2.6.0+cu124）· Python 3.12
- 局限如实记录：02 章 LLaMA-Factory 工具链（identity 0.5B / QLoRA 7B / export / DPO）需独立 venv + ~16GB 权重下载，本次 **env-blocked 未实跑**（三份学生报告同样未跑）；整改以命令块字段自洽、声明完备性、数学口径修正为准

## 一、镜像清单（本次交付）

| 文件 | 改动 |
|---|---|
| `courses/Part12_finetune_llamafactory/tutorial/README.md` | LoRA 推导代码块→LaTeX+低秩动机、A 初始化统一 N(0,1/r)、效果表加来源列（G13）、star 数截至日期、下载依赖档位节（HF_ENDPOINT/离线边界）、masking"两种等价实现"措辞 |
| `courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.md` | 初始化/缩放推导→LaTeX、α/r 为什么除 r、注入矩阵 why（Wq,Wv→all、3.1% 不可直接比）、耗时口径注（GPU 2.4s/CPU 单线程 40s）、错误 4（merge 双加）、性能表架桥句、练习 3 显存公式改"只按可训练参数"+数字例+全参账警示、QLoRA 表补 `optim` 行、~250→~300 行、loss 实测曲线图 |
| `courses/Part12_finetune_llamafactory/tutorial/02_llamafactory_workflow.md` | **双量化账本重写（P0：单位+方向双错）→双列账本 3.5→3.94→3.61GB**、分页优化器小节（三件套补齐 3/3）、QLoRA 量化推导→LaTeX、DPO 一段式公式+margins 定义+读图、§3 显存架桥句（3.74 vs 6GB 差 2.26GB 四项拆解）、~20MB→~40MB、"小时级"→"~10 分钟量级"、constants.py 版本⚠️、显存阶梯图、DPO lr 免责 |
| `courses/Part12_finetune_llamafactory/scripts/01_handwritten_sft_lora.py` | docstring "~40 秒"→GPU/CPU 双口径实测值；`per_device_batch`→`per_device_train_batch_size` |
| `courses/Part12_finetune_llamafactory/images/*.png` | 新增 3 图：lora_sft_loss_curve（**实测数据**）、finetune_memory_ladder（公式值/官方量级，图注注明）、dpo_rewards_margins（标注 schematic） |
| `assignments/assignment_12/assignment.md` | 题 4 文字拆解 16B→12B（bf16 2+2 + fp32 4+4）+静态账 vs 官方 6GB 架桥；思考题 Q1 同步改拆解+补 16B/参数口径注（7B 全参≈112GB+激活≈120GB） |
| `ledger/ledger_P12.md` | 本 Part 台账（Fixed 16 / Disputed 3） |
| `outline_review/outline_suggestions.md` | 追加 P12 三条（roadmap W'=W+BA 漏 α/r；面试直通车 4 问无落地；分页优化器教程侧已补） |
| `scratch/t2_P12/` | 基线/修后脚本、参考答案+测试四象限运行件、绘图脚本 |

## 二、必修清单执行情况（T0 九项全闭环）

1. **QLoRA 双量化账本重写（P0）**：删"3.5+0.37≈3.87GB"，改论文口径/本例口径双列（0.373 bits/param、65B≈3GB；7B 常数 0.44→0.11GB、总账 3.94→3.61GB）+⚠️误读警示框。✅
2. **分页优化器补齐**：02 章新小节（省峰值风险非平均占用、`optim: paged_adamw_8bit`、面试一句话）+01 章 QLoRA 表+02 章 §3 命令注释，三件套 3/3。✅
3. **显存账本三处自洽**：练习 3 公式改"只按可训练参数"+数字例 16GB 对上性能表+全参 84/112/120GB 警示；20MB→40MB；3.74 vs 6GB 两章各补架桥句+显存阶梯图。✅
4. **α/r 除 r / 注入矩阵 why / 低秩动机**：各 1-3 句落位（README 推导段、01 章推导段+对照表）。✅
5. **耗时口径**：docstring→"GPU ~3 秒（实测 2.4s）；CPU 多线程 wall ~2.6s、单线程约 40s"；教程注明口径。✅（修后实测 2.327s）
6. **HF 下载依赖档位**：README 新增"模型/数据从哪来"节（三档下载量、镜像变量、离线只到脚本 01）。✅
7. **DPO 公式+margins 定义、README 来源列、脚本字段全名**：三项齐。✅
8. **G1-G16**：G2×3 推导 LaTeX 化（check_latex 4 文件 0 问题）；G4 三图落位（实测数据优先：loss 曲线来自 scratch 复跑逐点捕获）；G13 效果表来源列+双量化双列账；G14 耗时口径两处。✅
9. **T1 的 10 个 C1 缺口**：①分页优化器→必修 2；②0.37GB→必修 1；③题 4 16B/12B→必修 3；④数字群漂移（40 秒/250 行/小时级）→三处改；⑤A 初始化四处→统一 N(0,1/r)；⑥依赖档位→必修 6；⑦roadmap W'=W+BA 漏 α/r→**outline**；⑧labels=-100 措辞→"两种等价实现"；⑨constants.py→补⚠️；⑩DPO lr→加"以安装版为准"+disputed D1。✅ 全部处置；roadmap 两项按规则追加 outline_suggestions.md。

## 三、验证结果（必跑三项全绿）

1. **脚本 scratch 实跑（G16）**：修前基线 + 修后各一轮，GPU wall 2.405s / 2.327s（rc=0）；`[1]` 6,144/200,664（3.1%）、`[2]` 3.572→0.076、`[3]` 回声 2/3（w3w7→w19）、`[4]` max|Δlogits|=5.36e-07/1.07e-06/2.38e-06 全 <1e-4——与教程数字逐项一致；字段对照打印已是 `per_device_train_batch_size`。
2. **作业**：`assignment_reference/assignment_12/finetune_exercises.py` + 现行 test → pytest **5 passed in 1.14s**；独立运行模式 **5/5 🎉**。题 4 的 12B 公式未动（错在文字拆解，公式/验收本自洽）。
3. **check_latex.py**：README / 01 / 02 / assignment.md 四个改动文件全部 **0 问题**（02 章首检 2 处 CHINESE_IN_MATH/TEXT_WITH_CJK 已清零）。

## 四、通用问题候选（≤3）

1. **外部论文数字直接当本例数字用（单位/口径双漂移）**：QLoRA"0.37 bits/param"被写成"0.37GB"还加进总量——凡引用论文数字，建议强制走 G13 双列（论文口径列 + 本例口径列），并在同段给"怎么折算到我的模型"的一行公式。
2. **"总量公式"与"逐项拆解文字"分开维护导致打架**：题 4 公式（12B）与括号文字（16B）各改各的。账本类内容建议以"公式行为事实源、文字只复述公式"的单源方式维护（P10 fp16/bf16、P8 masking 措辞同根）。
3. **跨 Part 引用措辞失实**：把"labels=-100"记在 P8 02 章名下（实为乘法 mask）。引用其他 Part 的实现细节时宜写"两种等价实现/以源章为准"，终版可对全部 P→P 引用做一次 grep 抽查。

## 五、好写法候选（≤3）

1. **01 章"五步 ↔ yaml 字段对照表"**（三名学生不约而同列为最喜欢）：把工具黑盒逐字段透视回手写代码，是"手写教会映射、工具给工程完备性"课程论点的最佳载体——值得作为全课程"工具链 Part"的固定体例。
2. **脚本合并正确性的测量方法论**：逐元素 logits 比对 + "argmax 平局翻转不能作判据"注释 + 断言阈值 1e-4，教学价值被三份报告一致认可；本次已把它升格为 01 章"错误 4"，教程-脚本信息倒挂消除。
3. **诚实标注文化**："本机实测/官方量级参考，未逐行本机复现/文件名以安装版本为准"三层来源声明——本次整改只需补齐少数漏标处（效果表来源列、constants.py⚠️、DPO lr），说明该体例本身可扩展、可审计。
