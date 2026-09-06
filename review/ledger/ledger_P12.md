# 问题台账 · Part 12（微调实战：LLaMA-Factory）· T2 整改批

> 编号规则：P12-C{cc}-{SRC}-{nn}；cc∈{00=README,01/02=章节,SC=scripts,AS=assignment_12}；SRC∈{S1,S2,S3,T=T1预审,T2=教师实证}
> 状态：fixed=已修复且验证；disputed=争议/待 T0 裁决（禁止 wontfix）
> 关键实证：①脚本 01 修前/修后均 GPU wall ~2.4s（RTX 4090，.venv torch），教程数字群（6,144/200,664=3.1%、3.572→0.076、回声 2/3、max|Δlogits|=2.38e-06）逐项复现；②双量化复算：常数 0.5 bits/param×7e9/8=0.44GB、DQ 后 0.127×7e9/8=0.11GB ⇒ 总账 3.5→3.94→3.61GB、省 ~0.33GB@7B（论文 0.373 bits/param≈3GB 是 **65B** 口径）；③12B/参数拆解=bf16 参数+梯度(2+2)+fp32 双动量(4+4)——"fp32 参数+梯度+两动量"实为 16B；④02 章工具链 7B/0.5B 命令未实跑（需独立 venv+~16GB 下载，env-blocked，如实记录）。

## Fixed（按严重度）

| 编号 | 严重度 | 状态 | 来源 | 描述 | 位置 | 修复方案 | 修复证据 |
|------|--------|------|------|------|------|----------|----------|
| P12-C02-01 | P0 | fixed | S1🔴 S3🔴 S2🟢 + T0 必修1 + T1 缺口2 | 双量化账本单位+方向双错："额外节省 ~0.37GB，总计 3.5+0.37≈3.87GB"——0.37 是论文 bits/**param**（65B 才折 ~3GB）不是 GB，且"节省"被**加**进总量 | 02 章数学推导 Step 3 | 重写为 LaTeX 推导（0.5→0.127 bits/param）+ **论文口径/本例口径双列表**（G13）：65B 省 0.373 bits/param≈3GB；7B 3.94GB→3.61GB 省 ~0.33GB；补 ⚠️"常见误读：单位和方向都错了"警示框 | 复算：0.44/0.11GB、3.94/3.61GB 与 S1/S3 独立复算一致；check_latex 0 问题 |
| P12-C02-02 | P0 | fixed | S1🟡 S3🔴 + T0 必修2 + T1 缺口1 | 分页优化器（Paged Optimizers）全 Part 0 提及——QLoRA 三件套只教 2/3（roadmap 节点 12 点名） | 02 章（性质列表后）+ 02 章 §3 yaml 注释 + 01 章 QLoRA 配置表 | 补小节"QLoRA 三件套之三：分页优化器"（unified memory 页出动量、省峰值风险非平均占用、换页降速代价、面试一句话）；§3 命令注释加 `optim: paged_adamw_8bit` 行（"4 个字段"→"5 个字段"）；01 章 QLoRA 表补 optim 行 | grep "分页\|paged" 三章多点命中；与 roadmap 目标对齐 |
| P12-CAS-03 | P0 | fixed | S1🔴 + T0 必修3 + T1 缺口3 | 题 4 文字拆解 16B 自相矛盾："×12B/参数：**fp32 参数+梯度+AdamW 两个动量**"（=16B）vs 公式 12B/验收 3.74；思考题 Q1"fp32 副本+梯度+两动量（约 12B）"同病——照文字实现 16B 得 3.82≠3.74 测试红且无从排查 | assignment.md 题 4 题面+验收+思考题 Q1 | 统一为 12B 拆解"bf16 参数与梯度 2+2B + fp32 一阶/二阶动量 4+4B（LoRA 口径只算可训练参数）"；Q1 补"另计 fp32 master 副本即 16B/参数（7B 全参≈112GB+激活≈120GB）"口径注；公式与 3.74 验收不动 | 参考答案（12B 公式）pytest 5/5 两种模式全绿；16B 复算 3.5+0.32=3.82 与题面矛盾已消除 |
| P12-C01/02-04 | P0 | fixed | S1🔴 S2🟡 S3🟡 + T0 必修3 | 显存账本三处不自洽：①01 章练习 3 经验公式把梯度(2B)/优化器(8B)按 **model_params_B 全量**算——7B 得 ~84GB，与同章性能表"7B LoRA ~16GB"矛盾（LoRA 的梯度/动量只该算可训练 ~20M）；②02 章"LoRA bf16 (~20MB)"实为 20M×2B=**40MB**；③静态账 3.74GB vs 官方 ~6GB 差 2.26GB 无架桥 | 01 章练习 3 docstring+性能表注；02 章 L31-32 + §3 | ①练习 3 改"梯度和优化器状态只按可训练参数计"+数字例（14GB+20M×12B≈0.24GB+激活≈16GB，对上性能表）+⚠️"全量算是全参的账（≈112+激活≈120GB）"；②20MB→40MB（注 20M×2B）；③01 性能表+02 §3 各补架桥句（量化常数+激活+CUDA context+峰值波动）+ 显存阶梯图 | 数字例 0.24GB/16GB 复算通过；3.74→6 差 2.26GB 四项拆解两章一致；图与文案数字一致 |
| P12-C00/01/02-05 | P1 | fixed | T1 §五（G2×3）+ T0 必修4 | 数学推导 3 处用代码块承载：README LoRA 低秩分解、01 章 A/B 初始化与缩放、02 章 QLoRA 量化过程 | 三个 md 推导块 | 全部改 $…$/$$…$$（单行闭合、公式内无中文）；顺带并入必修 4 素材：README 补**低秩假设动机**（ΔW 内在秩低，Aghajanyan et al. 2020）、01/README 补**α/r 为什么除 r**（BA 尺度随 r 漂移、解耦后调 r 不必重调 α/lr、α=2r 惯例来由）、01 章对照表补**注入矩阵 why**（原论文 Wq/Wv 起家、QLoRA 附录 all 更好、玩具 3.1% 与真实 LF 比例不可直接比） | check_latex 4 文件全 0 问题；三处动机各 1-3 句落位 |
| P12-CSC-06 | P1 | fixed | S1🟢 S2🟡 S3🟢 + T0 必修5 + T1 缺口4 + G14 | 耗时口径冲突：教程 01 章"~3 秒（RTX 4090 / CPU 均可复现）" vs 脚本 docstring"~40 秒"；实测 GPU wall 2.405s（T2 复跑与 S1/S2/S3 三方一致）、CPU 多线程 wall ~2.6s/单线程 ~40s | scripts/01 docstring + 01 章 L94 | docstring 改"GPU wall 实测 ~3 秒（RTX 4090 上 2.4s）；CPU 亦可：多线程 wall ~2.6s、单线程约 40s"；教程注明"GPU wall ~2.4s / CPU 多线程 wall ~2.6s、单线程约 40s"口径 | 修后脚本 scratch GPU 实跑 2.327s，输出与教程逐项一致 |
| P12-C00-07 | P1 | fixed | S2🔴 + T0 必修6 + T1 缺口6 | 模型/数据下载依赖零说明：02 章直接 `Qwen/Qwen2.5-0.5B/7B-Instruct`+`identity,alpaca_gpt4_zh`，无下载量/磁盘/HF 镜像/断网降级路径 | README 环境节 | 新增"模型 / 数据从哪来（下载依赖档位）"小节：三档表（脚本 01=0 / 0.5B+数据 ~1GB / 7B ~15GB）、`export HF_ENDPOINT=https://hf-mirror.com`、离线边界="无网只能完成脚本 01" | grep 下载/HF_ENDPOINT/离线 命中；档位量级与 S2 建议一致 |
| P12-C02-08 | P1 | fixed | S1🟡 S3🟡 + T0 必修7 | DPO-LoRA 只有工程字段无数学：无 DPO 损失、无 margins 定义（概念检验 Q3 让读曲线却不给定义） | 02 章 §5 + 新增读图 | 补"一段式最小数学"：隐式奖励 $r_\theta=\beta\log\frac{\pi_\theta}{\pi_{ref}}$、$\mathcal{L}_{DPO}=-\log\sigma(r_\theta(y_w)-r_\theta(y_l))$、margins=两隐式奖励差；新增 DPO rewards/margins 示意图（标注 schematic） | 公式与 Part 8 03 章 β=0.1 同口径；check_latex 0 问题 |
| P12-C00-09 | P1 | fixed | T1 §五（G13）+ S3 硬数字表 #6 | README 效果证据表（~120GB/~6GB/90%+/1-2h）无来源列；L6/L173 star 数（74.4k/75.2k）无截至日期 | README 效果表+两处 star | 表加"来源与口径"列（官方 benchmark 量级参考/QLoRA 论文 99.3% 口径/截至 2026-09）；两处 star 补"截至 2026-09" | 逐行标注后与 01/02 章性能表的来源声明标准对齐 |
| P12-C00/01-10 | P1 | fixed | T1 缺口5（G15） | A 初始化四处口径并存：README"高斯初始化"泛写 / 01 章"N(0, σ²)"σ 未定 / 形状图"N(0,1)/√r" / 脚本 `randn/√r` | README 性质列表 + 01 章 Step 1 | 统一为"$\mathcal{N}(0, 1/r)$（实现即 randn/√r，本课脚本同口径）"单一事实源 | 三处 grep 一致；与 P8 08 章口径兼容 |
| P12-C00/01-11 | P1 | fixed | T1 缺口8 | "prompt masking（labels=-100）"归属 P8 02 章措辞失实：P8 实现为乘法 mask，无 -100 字样 | README 前置 + 01 章前置 | 改为"P8 用乘法 mask 实现，本 Part 脚本改用 labels=-100——两种等价实现" | 两处措辞与 P8 事实源一致 |
| P12-C01/02-12 | P1 | fixed | T1 缺口4（G14 并案） | 宣称数字群漂移：01 章"~250 行" vs 脚本实长 311 行；02 章 §1 标题"小时级内出结果" vs 同章性能表 0.5B "~10min" | 01 章 L3；02 章 §1 标题 | "~250 行"→"~300 行"；"小时级内出结果"→"~10 分钟量级出结果" | 与 311 行实长/性能表逐字对齐 |
| P12-C02-13 | P1 | fixed | T1 缺口9 | 错误 1 模板清单路径 `src/llamafactory/extras/constants.py` 版本敏感未加⚠️（新版迁至 data/template.py）——与 3 处文件名⚠️的声明标准不齐 | 02 章错误 1 解法 | 补"⚠️ 路径随版本演进——新版模板注册已迁至 data/template.py，以安装版为准" | 与章内既有⚠️体例一致 |
| P12-C01-14 | P1 | fixed | T1 §三-1（加分点转正）+ S1 最喜欢#2 | merge 双加 bug 信息倒挂：脚本有 is_merged 哨兵+"Wx+2·BAx≈2.9"详细警示，01 章调试清单却无此条 | 01 章调试展示 | 新增"错误 4：合并后忘停旁路（BA 被加两次）"：症状/原因（$Wx+2\frac{\alpha}{r}BAx$）/解法（is_merged 哨兵+逐元素 logits 比对才算证据） | 与脚本 L117/L218-232 口径一致 |
| P12-Cxx-15 | P1 | fixed | T0 必修8（G4） | 全 Part 0 张图（无 images/ 目录），可画未画 ≥3 处 | images/（新增）+ 三章引用 | 新增 3 张 PNG（图内英文、正文中文解读）：①`lora_sft_loss_curve.png`——**实测数据**（scratch 复跑脚本捕获 400 步逐点 loss，3.572→0.076 与教程一致）；②`finetune_memory_ladder.png`——公式值/官方量级四柱（120/16/3.74/6GB，log 轴，图注注明非逐次实测，含 2.26GB 差注释）；③`dpo_rewards_margins.png`——读图指南（标注 schematic） | matplotlib 生成逐张目检（修正标题截断/标注压线两轮）；loss 图与脚本实跑数字一致 |
| P12-CSC-16 | P2 | fixed | S2 卡点3 + T0 必修7 | 脚本末尾 yaml 对照打印用缩写 `per_device_batch`，教程表与 LLaMA-Factory 真实字段是 `per_device_train_batch_size` | scripts/01 末尾打印 | 规范化为全名 | 修后实跑打印行已为新字段名 |

## Disputed（待 T0 裁决，禁止 wontfix）

| 编号 | 描述 | 建议 |
|------|------|------|
| P12-D1 | DPO lr"5e-6 量级" vs 官方 `qwen3_lora_dpo.yaml` 默认值：需安装 LLaMA-Factory 打开 yaml 核实（T1 缺口10），本次 env-blocked 未核。已加"官方 yaml 默认口径，具体数值以安装版本为准"免责 | 联网/装机窗口终核；若不一致以 yaml 为准改写 |
| P12-D2 | 02 章性能表 0.5B 行"~4GB/~10min（本机实测）"无存档日志（S3 硬数字表 #11 无法复核）；7B 行"官方量级参考，未逐行本机复现"诚实标注可辩护 | 维持信任标注；建议终版补跑一次 0.5B 档留日志 |
| P12-D3 | 02 章"跟随 latest"策略是否补实测版本锚点（`llamafactory-cli version` 实测值+验证日期）：需独立 venv+安装，本次环境阻塞（G16 精神记 env-blocked）；6 个命令块的字段名/文件名已按既有⚠️体例声明 | 装机后补一行"本文命令在 vX.Y 验证于 2026-0X"； roadmap 类衍生项见 outline_suggestions |

## 统计

- **Fixed：16 条**（P0×4、P1×11、P2×1）
- **Disputed：3 条**（DPO lr 版本锚、0.5B 实测行无日志、latest 版本锚点）+ outline 侧 3 条（roadmap W'=W+BA 漏 α/r、面试直通车 4 问无落地、分页优化器教程侧已补）
- 验证：修后脚本 GPU 实跑 rc=0（wall 2.327s，输出与教程数字逐项一致）；作业参考答案 pytest **5 passed** + 独立运行 **5/5**；check_latex 对 4 个改动 md 全部 **0 问题**
