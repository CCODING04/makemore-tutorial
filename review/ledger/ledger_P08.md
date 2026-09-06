# 问题台账 · Part 8（后训练全流程）· T2 整改批

> 编号规则：P08-C{cc}-{SRC}-{nn}；cc∈{00=README,01..09=章节,SC=scripts,AS=assignment_8}；SRC∈{S1,S2,S3,T=T1主教计划,T2=教师复核实证}
> 状态：fixed=已修复且验证；disputed=争议/待 T0 裁决（禁止 wontfix）
> 关键实证：`temp/ckpt_grpo.pt`、`ckpt_ppo.pt` 实测 **config 谎报**（config 写 n_embed=64/2 blocks，state_dict 权重实际是 (65,512)/12 blocks 的 GPU 档）——08 脚本按 config 建模必然 size mismatch 崩溃，这是 P0-3 的根因，非"GPU/CPU 档正常差异"。

## Fixed（按严重度）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|------|--------|------|------|------|------|----------|----------|
| P08-C03/C04-S2/S3/T-01 | P0 | fixed | S2🔴 S3🔴 T1 缺口1/2 + T0 必修1 | 章末作业指引错位：03 章"题 6（ORPO）"（题 6 实为 GAE，全作业无 ORPO 题）；04 章"题 7（GAE）、题 8（PPO Clipped）"全错位（实际 6=GAE、7=PPO、8=GRPO）且漏指 GRPO 题 | 03 章 L317、04 章 L362 | 03 章改"题 4（BT）、题 5（DPO）"并注明 ORPO/KTO 不设题；04 章改"题 6（GAE）、题 7（PPO Clipped Loss）、题 8（GRPO 组内优势）" | 与 assignment.md 题 1-8 逐一比对一致 |
| P08-C00/SC-02 | P0 | fixed | S2🔴 T1 缺口7 + T0 必修2 | 双 09 脚本编号冲突：09_quantize_and_serve.py 与 09_reasoning_models.py 并存，README 导航表两行均写 `09` | scripts/、README L18、09 章 | 重命名 `09_reasoning_models.py → 13_reasoning_models.py`（12 之后无冲突位）；同步 4 处引用：README 导航 `09`→`13`、09 章 L6 链接、09 章"脚本 09 的核心流程/三个实现细节"→"脚本 13"、脚本 docstring 自引用 | grep `09_reasoning` 全镜像仅剩教程文件名（章节号 09 不变）；13 脚本 scratch 实跑 rc=0 |
| P08-CSC-03 | P0 | fixed | S1🔴 S2🔴 + T0 必修3 | 08_eval_and_chat.py 按 ckpt 内 config 建模加载，混合档 ckpt（config=CPU 档、权重=GPU 档）直接 size mismatch 崩；且无任何档位提示 | scripts/08 L285-308；06/07 保存段；05 章 | ①08 新增 `infer_config_from_state_dict()`：以 tok_emb/pos_emb/blocks/heads 的实际张量形状反推配置加载，config 不符时打印 ⚠️；失败时打印"按 05 章运行顺序同档重跑 02→03→05/06/07"指引；②06/07 保存侧改为按被保存模型实际形状写 config（根治上游）；③教程 05 章"脚本运行顺序"补档位一致声明 + `SMALL=1` 用法 | scratch 实跑：修复前 GRPO ckpt 加载崩；修复后 5 个 ckpt 全部 [OK]（含两个 config 谎报的）rc=0；07 重跑后新 ckpt config=n_embed 512/12 blocks/8 heads 与权重一致（n_head 表达式对真实 sd 验证 =8） |
| P08-C00/C01/AS-04 | P0 | fixed | S1🔴 S2🟡 S3🔴 T1 缺口4 + T0 必修4 | 参数量四处三个数：README "CPU ~2M"（实测 0.112M，夸大 18 倍）、"GPU ~40M"（实测 89.6M）；01 章表 "~40M" 行（公式/实测 89.3M/89.6M）、"~6M" 行（实为 30.5M）、"~0.8M" 行（实为 1.04M）与自身公式矛盾；assignment "本课 2M 模型"；09 脚本输出"本课 2M 小模型/40M 模型" | README L81-84/92、01 章 L215-223、assignment.md L278、09_quantize L230/242 | README：CPU 行"~0.1M（实测 0.11M）"、GPU 行"~89M（实测 89.6M）"+ 口径注（untied、vocab 65/50304、公式见 01 章）；01 章表按公式重算（~0.13M/~1M/~30M/~89M/~405M）+ untied/weight-tying 注 + 实测例；assignment "2M 模型"→"~0.4M 模型（脚本 09 的量化实验模型）"；09 脚本两处文案同步 | 四档实测（01_gpt_model.GPT）：0.112M / 0.137M / 89,634,944 / 406,262,865；GPT-2 行 406M ✓ 保留 |
| P08-CAS-05 | P0 | fixed | S3🔴 + T0 必修5 | assignment 测试 pytest 路径 8 FAILED：`_Skipped` 自定义异常不被 pytest 识别为 skip；且 pytest 的 `Skipped` 继承 **BaseException**（非 Exception），except Exception 捕获不到——独立运行在装了 pytest 的机器上也会裸崩 | test L39-49、main | `_skip()`：pytest 可用时改抛 `pytest.skip(reason)`；新增 `_skip_exceptions()` 显式列出两种异常类供 except 与 `_is_skip` 使用；main 的 except 链改为 `except _skip_exceptions()` | 四象限全绿：pytest+参考答案=8 passed；pytest+未实现=8 skipped 无 failed；独立+参考答案=8/8 通过；独立+未实现=8 跳过 0 失败（py3.8/torch2.2.1/pytest8.3.5 与 py3.12 双环境） |
| P08-C08-S2-06 | P0 | fixed | S2🔴 + T0 必修6 | 08 章 LoRALinear 代码块漏 `self.r, self.alpha = r, alpha`，照抄会 AttributeError（forward 用 self.alpha/self.r） | 08 章 L37 | 补一行并注释"forward 的缩放要用，必须保存"；与脚本 10 L88 对齐 | 与 10_lora_from_scratch.py 逐行一致（脚本实跑 rc=0） |
| P08-C04-S1-07 | P1 | fixed | S1🔴 卡点1 + T0 必修7 | k3 KL 估计器只给"无偏/非负/稳定"结论不给理由，缺 E[exp(d)]=1 关键一步与"低方差"论证 | 04 章 k3 节 | 补两行 LaTeX 推导（E[k3]=1+KL−1=KL；归一一步 ∑π_ref=1）+ 非负性一句（e^x≥1+x）+ 手算数字例（π=(0.9,0.1), ref=(0.5,0.5)：KL=0.3681，逐点 k3=0.143/2.391，期望=0.3681 ✓） | 数字例 Python 复算逐位一致（0.9·0.1434+0.1·2.3906=0.3681） |
| P08-C04-S1-08 | P1 | fixed | S1🔴 卡点2 + T0 必修7 | PPO clip ASCII 图误导：不分 A>0/A<0、截断区位置画反、无分段平坦形状 | 04 章 clip 图 | 重画双联图（A>0 封顶平坦在 (1+ε)A；A<0 封底平坦在 (1−ε)A）+ "clip 只封过度更新一侧"两句解读 + 数字例（A=1,ε=0.2,ratio=1.5 → surr1=1.5, surr2=1.2, min=1.2 恒定） | 与 min(surr1,surr2) 语义逐步核对（A<0 时 ratio<1−ε 取 surr2=(1−ε)A） |
| P08-C04-S1-09 | P1 | fixed | S1🟡 卡点4 + T0 必修7 | GRPO 数字例 group_std=0.58 未注明是 torch.std 默认 N−1（无偏）口径，按总体 std 复算得 0.5 对不上 | 04 章 GRPO 数字例 | 加 📌 注：除以 N−1=3 → √(1/3)≈0.577→0.58；总体口径是 0.5 | 复算一致 |
| P08-C02/SC-S1-10 | P1 | fixed | S1🟡 卡点5 + T0 必修7 | "masked loss 通常比 unmasked 大"断言与脚本实跑相反（Unmasked 2.7361 > Masked 2.7182），未给适用条件 | 02 章 L160、03_sft.py 结尾声明 | 两处均改为"经验规律非数学保证"，注明欠训练+合成数据可相反（引用实跑数字），落点改为"训练目标正确性" | S1 实跑数字（scratch/S1_P8）引用 |
| P08-CSC-S1-11 | P1 | fixed | S1🟡 卡点9 + T0 必修7 | 09_reasoning 脚本笔误"非 NM"应为"非 NN"（神经网络） | 13 脚本 L152 | 改"非 NN" | grep 确认；教程原文"非神经网络 RM"本正确 |
| P08-C05-T-12 | P1 | fixed | T1 缺口3 + T0 必修8 | 05 章旧收尾"恭喜完成 Part 8 全部学习"（实际还有 4 章）且全章无作业指引 | 05 章 L258-264 | 收尾改"主线（01-05 章）学完 + 列出剩余四章"；新增"📝 课后作业"段（章内 details 练习 + 观测题 C 衔接 + Assignment 8 链接） | 导航行保留（上一章/下一章/README） |
| P08-C07-T-13 | P1 | fixed | T1 G13 + T0 必修9 | 07 章通篇半角直引号（代码块外 166 个），与其余 9 文件中文引号风格离群 | 07 章全文 | 脚本化流式配对替换为 “” （仅非代码块行；166 个全部成对，结束态配平） | 替换后代码块内 12 个引号原样保留（抽查 L84-89 python 语法完好） |
| P08-C07/06-T-14 | P1 | fixed | T1 G2 缺口5 + T0 必修9 | 07 章 L487 引用 `docs/course_roadmap_v2.md`（现行是 v3）；06 章"用简化模拟**量化**了这个叙事"病句（该实验是显存模拟非量化）且 41%→5% 未标模拟口径 | 07 章 L487、06 章 L100-101 | v2→v3；病句改"验证了"；补"论文 60-80%→<4% 是真实长尾负载口径，本课 5% 含尾部半块的模拟值，别混用" | — |
| P08-C07/08/09-T-15 | P1 | fixed | T1 缺口7（导航链三断）+ T0 必修9 | 导航链断裂：07 章末无"下一章"；08 章末只指 Part 12（拟开）不指 09 章；09 章无"上一章" | 07/08/09 章末 | 07 章导航补"下一章：LoRA 与分类微调"；08 章"下一步"改指 09 章（Part 12/11 降为延伸提及）+导航行补下一章；09 章补"← 上一章：LoRA 与分类微调" | Part12/15 目录存在性已核（链接非死链） |
| P08-C09-T-16 | P1 | fixed | T1 缺口8 + T0 必修12 | 09 章作业段残留"（Part 16 02 章的 img2img/ControlNet 作业使用 Assignment 16）"（跨 Part 复制残留）；GRPO 阶段无实测数字声明 | 09 章 L92-93、L26-33/40-48 | 残留句删除，作业段改写为可执行任务（跑脚本 13 记录 n=1/4/8 并解释）；实测段补口径（0.2177=GPU 档、CPU 档 0.68；n=1/4/8 两档数字并注"环境敏感"）；诚实声明"GRPO 简化阶段只落地奖励设计、未跑完整 RL 循环，动手用 04 章+脚本 07" | CPU 实跑 13 脚本：SFT loss 0.68、n=1/4/8=58/70/80%（scratch run_scripts/reasoning.log） |
| P08-C03-T-17 | P1 | fixed | T1 缺口10（C1）+ S3 白板素材 | DPO β 边界（β→0/β→∞）教程缺位；Q1"KL 惩罚"说法与推导的"无显式 KL 项"前后口径不一 | 03 章 L167-171 后新增 | 新增"⚖️ 边界情形"段：β→0 → loss→ln2 常数、梯度∝β→0 **停更不更新**（不是退化为 BT）；β→∞ → σ 饱和、隐式奖励放大 β 倍数值不稳、等效硬拉对齐易 reward hacking；并说明"β 大=KL 惩罚重"是源目标直觉、DPO 里 β 是隐式奖励温度 | 与 S3 白板自测结论一致 |
| P08-CSC-18 | P1 | fixed | G8 + T0 必修10 | 02/06/07 在 GPU 机器默认大档长跑，无 toy 档、训练 print 无 flush，3 分钟零输出像卡死 | 02/06/07 | 三脚本加 `SMALL=1` 环境变量强制 CPU 小档（默认行为不变）；训练进度 print 加 `flush=True`；docstring 补运行档位说明；教程 05 章补 SMALL=1 用法 | 02 rc=0（CPU 小档 step 0/10 loss 正常打印，flush 生效）；07 SMALL=1 rc=0 全流程含 ckpt 保存；06/07/02 py_compile 通过 |
| P08-C03-T-19 | P2 | fixed | T1 §三-1（KTO 简化未声明） | KTO 脚本 `clamp(min=0)` 是对论文 KL baseline 的课程自选简化，教程未声明 | 03 章 KTO 代码块后 | 补"与论文的差异声明"引用块（论文 KL 估计可为负；本课 clamp(min=0) 为简化，机制思想一致） | — |
| P08-C01/03/04-20 | P2 | fixed | G2（LaTeX 化） | DPO/GAE/TD/GRPO/k3 公式以 ASCII 代码块呈现，未 LaTeX 化 | 03 章 5 处、04 章 4 处 | 全部改 `$$` 单行公式（无中文），代码块只留实现 | check_latex.py 全部 0 问题 |
| P08-C00-21 | P2 | fixed | G14 | 实测数字口径缺失：README 参数量、09 章自一致性/SFT loss 无运行环境口径 | README/09 章 | 补"实测+档位+vocab 口径"注（README 规模表下、09 章两处数字块） | — |
| P08-C08/09-22 | P2 | fixed | G1 | 08/09 章"学完**本部分**你能"应为"本章" | 08 章 L84、09 章 L69 | 统一改"学完本章你能" | grep 确认 |

## Disputed（待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P08-D1 | T1 预审判定的三处"疑似乱码/病句"（07 章 L22-23 前置知识、07 章 L125-129 yaml 段、07 章 §7.2 Q2）经逐字复核**语句完整、非乱码**（S3 全文扫描亦无 U+FFFD）；06 章 L101 确为病句已修。预审 grep 命中系代码块/ASCII 图误报 | T0 确认"07 章乱码"项关闭；以本次逐字复核为准 |
| P08-D2 | 08 脚本修复后按权重形状可加载全部 ckpt，但 5 阶段准确率仍全 0%（CPU 字符级数据 + GPU 档模型、欠训练），教程 05 章"SFT 10-20%"预期区间 CPU 全流程达不到（教程已有免责声明）。是否在 05 章预期表加"CPU 档实测 0% 属正常"一句话，属课程口径决策 | 建议 T0 采纳加一句；本次已在 05 章运行顺序节注明档位风险 |
| P08-D3 | 09 章"下一步"指向 Part 15（多模态）而 README 末尾"下一章 Part 9"（课程主线）——两个出口并存已在 09 章注明"主线下一站 Part 9，见 README"；课程整体章节出口口径（Part 顺序 vs 专题跳转）建议终版统一 | 列入终版大纲裁决 |
| P08-D4 | 13_reasoning_models.py 的 RL 阶段只展示奖励设计未跑训练循环（教程已如实声明）；若 T0 希望"R1 四阶段全可跑"需扩脚本（超出本次整改范围） | 建议 roadmap 措辞对齐"两阶段落地+两阶段机制讲解" |

## 未修复即"确认为正确"项（复核结论）

- 06 章 "~0.4M 玩具模型" ✓（脚本 09 量化模型实测 0.43M，S3 复核一致）——保留
- 01 章 GPT-2 行 406M ✓（实测 406,262,865，untied 公式 405M 同级）——保留
- S1 附录 A1-A8 数学逐抠（mask shift 口径、BT→DPO 推导、GAE 递推、k3 实现、KV 公式 1.073GB、E=2.31、LoRA 对称性）全部复核通过，未改动
- README"完全自包含"声明 ✓（HH-RLHF 仅 docs 登记、脚本规则合成数据）
