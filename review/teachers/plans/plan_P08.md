# Part 8（后训练全流程）分章审计计划 — T1 主教预审

- 预审日期：2026-09-04
- 对象：`REPO/courses/Part8_post_training/`（tutorial 10 个 md 共 2642 行 + scripts 13 个 .py）+ `assignments/assignment_8/assignment.md` + `docs/course_roadmap_v3.md` 节点 8
- 性质：Part 8 是全课程最大 Part（后训练主线 + v3.1 扩的幻觉安全/评估学/推理服务），按三学生 × 三单元分工
- 预审方式：只读通读 + grep 交叉核对，未运行任何脚本、未改任何课程文件

---

## 一、C1 标题-内容对应表（10 文件逐个）

| # | 文件（行数） | 标题 | 实际内容 | 对应脚本 | 章末作业映射（实况） | C1 判定 |
|---|---|---|---|---|---|---|
| 0 | README.md (155) | Part 8 从零训练 LLM：后训练全流程 | 章节导航、两源仓库分工表、前置知识、规模对照表、演进路线、脚本 11 自备数据说明 | 全部 | — | 基本对应；**规模表 "~2M" 与 01 章 "~0.1M" 矛盾**（见 §五-3） |
| 1 | 01_gpt_and_pretrain.md (393) | GPT-2 架构与预训练 | Head/MHA/MLP/Pre-LN、forward_hidden()、参数量公式、AdamW/cosine+warmup/bf16/grad-accum/clip/checkpoint | 01, 02 | 题 1、题 2 ✅（映射正确） | ✅ 对应 |
| 2 | 02_sft_and_chat.md (232) | SFT 与 Chat Template | ChatML 简化模板、Prompt Masking 四步（shift→逐 token CE→乘 mask→除 mask.sum）、与 Part 7 SFT 对比 | 03 | 题 3 ✅ | ✅ 对应 |
| 3 | 03_reward_and_dpo.md (325) | 奖励模型与对齐算法：DPO、ORPO、KTO | BT 模型+损失、RewardModel（末 token/零初始化）、DPO 五步推导、ORPO/KTO、三算法对比、隐式奖励 | 04, 05 | 题 4、题 5 ✅；**"题 6（ORPO）"❌——assignment 题 6 是 GAE，全作业无 ORPO 题** | ❌ 缺口 1 |
| 4 | 04_ppo_and_grpo.md (370) | 强化学习：PPO 与 GRPO | 策略梯度、Clipped Surrogate、GAE（γλ 折中）、ValueHead、PPO 循环、GRPO 组内优势、k3 KL、RLVR、PPO vs GRPO | 06, 07 | **"题 7（GAE）"❌（题 7 实为 PPO Clipped）、"题 8（PPO Clipped Loss）"❌（题 8 实为 GRPO）且漏掉真正的 GRPO 题** | ❌ 缺口 2（两处错链+漏一题） |
| 5 | 05_eval_and_deploy.md (264) | 评估与推理部署：GSM8K、生成策略、完整流水线回顾 | GSM8K mini 流水线、temperature/top_k/top_p/repetition penalty、KV Cache 简介、全流水线回顾、运行顺序 | 08 | **全章无 Assignment 链接**（01-04 章均有） | ❌ 缺口 3 |
| 6 | 06_inference_and_serving.md (188) | 推理与服务 | memory-bound 主线、RTN/GPTQ/AWQ、KV 显存公式+KIVI、PagedAttention/连续批处理、投机解码、TTFT/TPOT/goodput、vLLM 实操 | 09(quantize) | 实验 A/B（assignment 观测题）✅ | 基本对应；**§3 "用简化模拟量化了这个叙事（64 请求）"为病句/疑似乱码**；"41%→5%" vs 论文 "<4%" 口径需核对脚本实测 |
| 7 | 07_evaluation.md (491) | 评估学：怎么科学地给模型打分 | 三范式、lm-eval-harness 实操+自定义 task、HELM、GSM1k 污染、幻觉（SE/温度/ECE）、安全（refusal direction/GCG/HarmBench）、合规四件套 | 11, 12 | 观测题 C ✅ | 基本对应；**前置知识节引号/断句异常（L22-23）；§2 自定义 task 段与 §7.2 Q2 疑似乱码待逐字复核；末尾引用 docs/course_roadmap_v2.md（应为 v3）** |
| 8 | 08_lora_and_classification.md (125) | LoRA 与分类微调 | 全参微调显存账、LoRALinear 从零（A 高斯/B 零/α-r）、分类微调 4 步范式、与 Part 12 分工 | 10 | 自定义三实验（非 Assignment 题，合理） | ✅ 对应；**"下一步"跳 Part 12(拟开) 而非 09 章** |
| 9 | 09_reasoning_models.md (99) | 推理模型与 test-time compute | R1 四阶段、cold-start SFT + GRPO 简化、self-consistency 实测 | 09(reasoning) | "无独立作业"（声明明确，可接受）；**但作业段突兀提及 "Part 16 02 章 img2img/ControlNet 作业"（残留句）**；GRPO 阶段无实测数字（只有 SFT loss 0.2177） | ❌ 缺口 4（残留句+缺实测） |

**章节导航链断裂**：07 章→08 章无"下一步"链接（仅"上一章/README"）；08→09 断（跳 Part 12）；09 章无"上一章"链接且"下一步"跳 Part 15；README 末尾"下一章 Part 9 CUDA 内核"与 09 章去向（Part 15）不一致——学生按顺序读完 05 章时会被"恭喜完成 Part 8 全部学习"误导（05 章末原话，实际还有 4 章）。

## 二、学生单元划分（三学生 × 三单元）

| 单元 | 文件 | 行数 | 难度画像 | 审计重点 |
|---|---|---|---|---|
| **单元 A** | README + 01 + 02 + 03 | 155+393+232+325 ≈ 1105 | 01 ★★（Part 6 复习为主）、02 ★★★（masking 逐行）、03 ★★★★（DPO 推导是本单元顶峰） | 架构陈述与脚本 01/02 互证；bf16 表格事实；DPO 五步推导逐步核对；题 4/5 映射 |
| **单元 B** | 04 + 05 + 06 | 370+264+188 ≈ 822 | 04 ★★★★★（全 Part 数学顶峰：GAE/k3/clip）、05 ★★（叙事为主）、06 ★★★★（工程实测数字密集） | GAE 递推 vs 06_ppo 脚本；k3 方向与无偏性；脚本③④实测数字三口径（教程/roadmap/脚本）；作业观测题 A/B |
| **单元 C** | 07 + 08 + 09 | 491+125+99 ≈ 715 | 07 ★★★★（491 行 v3.1 新扩，实验数字最多）、08 ★★★、09 ★★ | 07 章逐字扫乱码；SE/ECE/温度表 vs 脚本 11 实测；lm-eval 坑位 vs 脚本 12；LoRA 公式 vs 脚本 10；合规四件套事实核查 |

- 学生 1 → 单元 A；学生 2 → 单元 B；学生 3 → 单元 C。单元 B/C 的学生需回看 03 章结尾作为接口（DPO→PPO 衔接叙事）。
- 每个学生除本单元外，通读 README 一次（导航表/规模表/演进路线是全局口径源，README 的问题由主教统一裁定）。

## 三、本 Part 特有审计要点（公式推导 ↔ 代码实现互证清单）

1. **Bradley-Terry / DPO 目标函数推导**（03 章 L110-165）：
   - 链条：RLHF 目标 `max E[r] − β·KL(π‖π_ref)` → 闭式解 `π* = π_ref·exp(r/β)/Z(x)` → 反解 `r = β·log(π/π_ref) + β·log Z` → Z 在 chosen−rejected 中消去 → `L = −logsigmoid(β·[(logπ_c−logπ_r)−(logπ_ref_c−logπ_ref_r)])`。逐行核对推导无跳步、符号一致；与 `05_dpo_alignment.py` 的 `dpo_loss` 及隐式奖励 `β(logπ−logπ_ref)` 互证。
   - roadmap 要求的边界检查（β→0 / β→∞ 各退化成什么）教程正文与思考题均未覆盖 → 建议登记为内容缺口（论文五步法在 roadmap 属"学习目标"，教程只给了直觉）。
   - **ORPO**（L208-227）：`log_odds = (chosen_mean − log1mexp(chosen_mean)) − (rejected_mean − log1mexp(rejected_mean))`，与 `05_dpo_alignment.py` 实现比对 `_log1mexp` 数值稳定性；NLL 项 per-token 归一化口径。
   - **KTO**（L246-262）：脚本 L319 `kl = cat([chosen, rejected]).mean().clamp(min=0).detach()` 是对 KTO 论文 KL baseline 的**简化**（论文用 KL(π‖π_ref) 的样本估计，教程未声明 clamp(min=0) 是课程自选简化）→ 审计是否需要一句"与论文实现差异"声明。
2. **SFT prompt masking**（02 章 L98-149 ↔ `03_sft.py`）：四步实现（shift、reduction='none'、乘 mask、除 `mask.sum().clamp(1)`）逐行互证；**mask 也 shift `[:, 1:]`** 这个易错点必须核；教程声明与 roadmap "ignore_index=-100" 口径不一致（脚本用乘法 mask，grep 确认脚本无 -100）→ 列为口径缺口。
3. **PPO / GAE**（04 章 L93-219 ↔ `06_ppo_training.py` L237+）：
   - `δ_t = r_t + γV(s_{t+1}) − V(s_t)`、`A_t = δ_t + γλ·A_{t+1}`；γ=1.0、λ=0.95 默认。
   - 脚本 `nonterminal = resp_mask[:, t+1]`（用 response mask 充当终止信号）与教程伪代码一致性；`values_next` 的构造（shift or 独立张量）必须读脚本确认。
   - 教程 PPO 循环伪代码 `kl = old_logp − ref_logp`（可正可负的近似）与 GRPO 的 k3 口径差异是否向学生声明清楚。
   - value loss（MSE with clip）、entropy bonus（coeff 0.01）与脚本互证。
4. **GRPO 组内优势**（04 章 L234-263 ↔ `07_grpo_training.py`）：`A=(r−mean)/(std+eps)`、eps=1e-4 与 DeepSeekMath 一致；assignment 题 8 验证标准"组内 std=0 时 advantage 全为 0"（分子 0，eps 不影响，自洽）；`kl_coef=0.04` 与 DeepSeekMath 一致。
5. **k3 KL 估计器**（04 章 L269-279 ↔ 脚本 L211+）：`exp(log_ref−log_new) − (log_ref−log_new) − 1`，期望 = KL(π_new‖π_ref)，无偏、非负、不需 clamp——教程三优点表述正确；审计脚本注释 "KL(π‖π_ref)" 的 π 指代是否写清是 new policy。
6. **bf16 混合精度声明**（01 章 L288-308 ↔ `02_pretrain.py`）：指数 8/尾数 7 bit 表述正确；"CPU 不支持 bf16"的说法不严谨（torch CPU 可 autocast bf16，课程是"不用"而非"不支持"）→ 登记措辞问题；autocast 只包 forward 的写法与 GradScaler 讨论。
7. **量化**（06 章）：RTN `scale=max|W|/127` 逐通道；GPTQ（事后 Hessian 误差补偿）vs AWQ（事前激活感知缩放）一句话对比的事实性；int4 不分组"翻车"归因离群通道；KIVI（K 按通道/V 按 token + attention-sink 窗口）表述核查。**教程无 GPTQ/AWQ 代码实现（概念层）——属声明合理的取舍，确认章节内有明确声明即可。**
8. **数据集依赖（HH-RLHF 等）**：
   - `05_dpo_alignment.py` 用**规则合成偏好对**，`04_reward_model.py` 同理——**HH-RLHF 不在本地也未被任何脚本引用**（仅 `docs/datasets.md` L37 登记为外部资源）；GSM8K 同样只在叙事层（脚本用合成算术）。README "完全自包含"声明与实况一致 ✅。
   - 唯二外部依赖：`03_sft.py --original-data`（Alpaca，需联网+datasets，默认关）、脚本 11/12 的 Qwen2.5-0.5B(-Instruct)（HF 缓存，缺失优雅退出 rc=0）。审计 README L74-75 依赖表是否与上述逐条吻合。
9. **LoRA**（08 章 ↔ `10_lora_from_scratch.py`）：`ΔW=(α/r)BA`、A~N(0,1/r)、B=0、合并 `W += (α/r)BA`；实测 "230,226 vs 7,872 (3.4%)" 与脚本打印核对；"注入 MLP 两层 vs 原论文 Wq/Wv"的诚实声明已存在。
10. **07 章实验数字群**（vs `11_hallucination_safety.py` docstring 与正文实测）：AUROC 0.864、ECE 0.171→0.198、温度表四行、course_quiz 0.0%、arc_easy 58/59%——脚本与教程已内置"复跑漂移"免责声明（写得好），审计确认每个数字都有出处行。

## 四、scripts 运行档位建议（审计执行时验证 toy 档可行性）

| 脚本 | 档位/耗时（docstring 或代码实据） | toy 档可行性 | 备注 |
|---|---|---|---|
| 01_gpt_model.py | 纯定义 | ✅ 无训练 | import 级验证即可 |
| 02_pretrain.py | CPU <30s（字符级 vocab=65）；GPU tiktoken | ✅ CPU | GPU 路径 tiktoken 需装库（可选依赖） |
| 03_sft.py | CPU 合成数据；pretrain_steps CPU 20/GPU 50 | ✅ CPU | `--original-data` 需联网，审计默认不跑 |
| 04_reward_model.py | CPU 20 步预训练底座 | ✅ CPU | 同上 |
| 05_dpo_alignment.py | 同上（L379） | ✅ CPU | DPO/ORPO/KTO 三合一 |
| 06_ppo_training.py | 765 行；CPU 可跑但 RL rollout 慢 | ⚠️ 建议标注实测时长 | 审计时记录实际耗时 |
| 07_grpo_training.py | 687 行；合成加减法任务 | ⚠️ 同上 | 自带 15 分钟导读注释 |
| 08_eval_and_chat.py | 加载 02-07 各 ckpt | ⚠️ 链式依赖 | **必须按 05 章"脚本运行顺序"先跑 02→03→05(或06/07)**；审计验证缺 ckpt 时的行为（是否优雅提示） |
| 09_quantize_and_serve.py | CPU 全跑 2-3 分钟；GPU 500 步训练路径用于量化对比 | ✅ CPU（⚠️ 教程声明 CPU 50 步欠训练时 Δ 会失真，对比实验须 GPU） | 量化数字只在 GPU 档可信——审计执行时按此档位 |
| 09_reasoning_models.py | GPU ~30s / CPU ~2min | ✅ | 与上一脚本同名前缀 09，**双 09 命名冲突**（README 导航表两行都写脚本列 `09`），登记命名问题 |
| 10_lora_from_scratch.py | ~30s CPU/GPU | ✅ | |
| 11_hallucination_safety.py | GPU 4-6min；CPU 15-25min；需 Qwen2.5-0.5B + matplotlib；缺模型 rc=0 优雅退出 | ⚠️ 有 GPU 则 GPU 档 | refusal_direction 段数据读者自备（设计如此，rc=0） |
| 12_lm_eval_hands_on.py | 每档 1-3min；需 `lm_eval[hf]` + 联网 arc_easy；缺依赖 rc=0 | ⚠️ 依赖联网/安装 | |

GPU 需求总结：README "单卡 4090 全课程可跑"的声明与 13 个脚本实况一致；406M 原版规模属可选延伸（README 已给 batch=4+梯度累积步骤）。审计执行时 CPU toy 档优先覆盖 01-05、09(两个)、10；06/07 记录时长；08 验证 ckpt 链；11/12 验证优雅退出路径。

## 五、格式规范预检（G1/G2/G4/G10/G13/G14）违反位置清单

> 按战役格式规范归类；执行学生对照规范定义复核后再定级。

- **G2（链接/引用有效性与指向正确性）— 4 处实锤 + 2 处待核**
  1. 03 章 L317："题 6（ORPO）"→ assignment 题 6 = GAE，无 ORPO 题（错链）。
  2. 04 章 L362："题 7（GAE）"（实为 PPO）、"题 8（PPO Clipped Loss）"（实为 GRPO）——两处错链 + 漏题 8 GRPO 未被任何章引用。
  3. 07 章 L487：引用 `docs/course_roadmap_v2.md`——仓库现行 roadmap 是 v3。
  4. 08 章 L121："下一步"指向 `Part12_finetune_llamafactory`（README 标注"拟开"，确认目录是否存在，不存在则死链）。
  5. 待核：09 章 L92-93 提及 Part 16 作业、L97 Part 15 链接目标存在性；05 章 L118 与 README 各相对链接抽查。
- **G1（标题-内容/结构宣称一致性）— 3 处**
  1. 05 章 L260"恭喜你完成了 Part 8 的全部学习"——其后还有 06/07/08/09 四章（v3.1 扩章后未回改旧收尾语）。
  2. 05 章标题含"推理部署"但 KV Cache 仅 2 段简介、正文自述"本章是 Part 8 的收尾"——与 06/07 章分工声明需在章首补一句（07 章首已有衔接，05 章没有）。
  3. 08 章 L83"学完**本部分**你能"（应为"本章"）；09 章 L61 同款。
- **G4（代码块语言标注）**：架构图/流程图围栏普遍无语言标注，重点文件：01 章 L26/49/84/110/129/148/155/206-208/270-272/283/354-358；02 章 L38/74/145；04 章 L75-87/150/169/182；05 章 L64/150/168；06 章 L72/107。执行时按规范决定是否统一补 `text`。
- **G13/G14（标点/引号一致性）**
  1. 07 章通篇使用半角直引号 `"..."`（L3/5/16/32/38/100/109/241/296-303/396-397 等），其余 9 个文件用中文引号 —— 全章风格离群。
  2. **疑似乱码/病句（最高优先级逐字复核）**：06 章 L101"用简化模拟量化了这个叙事（64 请求）"；07 章 L22-23 前置知识断句残缺；07 章 L125-129 mytasks yaml 段语句错乱（"运行时在系统临时目录生成一份绝对路径版 yaml…再交给 TaskManager"）；07 章 §7.2 Q2 答案段（L395-397 附近）需整段通读。
- **G10（表格/编号一致性）**
  1. README 导航表脚本列：06 行 `09` 与 09 行 `09` 重号（09_quantize_and_serve.py / 09_reasoning_models.py 双 09 命名）。
  2. README 规模表"CPU 模式 ~2M" vs 01 章同配置 "~0.1M" vs 06 章 "~0.4M" vs assignment "2M 模型"——参数量四处口径（见 §六-1）。
  3. 03 章 DPO 代码块把"隐式奖励计算"放进 `dpo_loss` 返回值——与 assignment 题 5 签名（只返回 loss）不完全一致，执行时核对。
- **章节导航链**（归 G1/G2 复合）：07→08 无"下一步"；08→09 断链；09 无"上一章"；README 末"下一章 Part 9"与 09 章去向（Part 15）矛盾。

## 六、跨 Part 一致性

1. **与 roadmap_v3 节点 8 的偏差**（节点 8 正文未随 v3.1 扩章回改）：
   - roadmap："教程 **8 章**：01→08"——实际 9 章（缺 09_reasoning_models）。
   - roadmap："脚本 **01-10**"——实际 12 个（缺 11 幻觉安全、12 lm-eval；§6 缺口登记表两行已承认 11/12 归属 Part 8，正文未同步）。
   - **投机解码硬数字矛盾**：roadmap "α≈0.65 → 实测 2.81 vs 理论 2.53" vs 06 章 "α≈0.60 → 实测 2.47（121 token/49 前向）vs 理论 2.31" vs 脚本（运行时打印）。两次运行的数字被两文档各取其一，需统一为同一份实测并注明运行日期。
   - PagedAttention：roadmap "碎片 41% → <4%" vs 06 章 "41% → 分页 5%"（教程注明 5% 是"含尾部半块"的模拟值，<4% 是论文值）——口径可自洽但 roadmap 未区分。
   - roadmap 学习目标 "prompt masking 为什么用 **ignore_index=-100**" vs 教程/脚本乘法 mask（脚本无 -100）——两种等价实现，roadmap 表述超前于教程，需对齐。
   - int8 Δ≈+0.38（roadmap）≈ +0.4（06 章/assignment）✅ 一致；KV 1.07GB/0.27GB ✅ 一致。
2. **与 Part 11（verl 实战）分工声明**：已存在且双向明确——04 章 L339-341"Part 11 用工业框架 verl 跑同一批算法，一原理一工程双视角"+ README L29 同义。审计执行时抽 Part 11 README 侧是否 reciprocal（本次不读 Part 11 内容，仅登记待查项）。
3. **与 Part 7 DPO/SFT 口径**：README L46 声明"Part 8 与 Part 7 独立、重合内容视为复习"；02 章 L179-185 给出 Part7 vs Part8 SFT 三行对比表（全 token vs response-only、模板差异、masking 有无）——口径自洽 ✅。Part 7 minimind DPO（roadmap 提 lr=4e-8/β=0.15）与 Part 8 DPO β=0.1 默认属不同实现档位，执行时确认 03 章有无"与 Part 7 数值不同是正常的"免责句。
4. **内部前后呼应**：09 章前置知识声明依赖"04 章 GRPO + 07 章规则奖励"✅；06 章前置依赖"Part 7 03 章 KV/GQA + Part 9 02 章 memory-bound"（向前引用 Part 9，属"建议掌握"级，可接受但需确认措辞不会让读者以为必须先学 Part 9）。

## 七、C1 缺口汇总（预审实锤 10 项，执行学生按清单定级）

1. 03 章"题 6（ORPO）"错链；2. 04 章"题 7/题 8"双错链且漏 GRPO 题；3. 05 章无 Assignment 链接 + 旧收尾语；4. 参数量四处口径（~2M/~0.1M/~0.4M/2M）；5. 06 章病句 + 41%/5% 口径；6. 07 章 roadmap_v2 引用 + 疑似乱码三处待逐字复核；7. 章节导航链三处断裂 + 双 09 命名；8. 09 章 Part 16 残留句 + GRPO 阶段缺实测数字；9. roadmap 节点 8 正文未随 v3.1 回改（章数/脚本数/投机解码数字/ignore_index 口径）；10. DPO β 边界检查（β→0/∞）教程缺位（roadmap 要求的论文五步法深度未达）。

## 八、学生执行清单（各单元通用动作）

1. 逐章 C1：标题/小节 vs 内容 vs 对应脚本 vs 作业映射四栏核对（本计划 §一 表为底稿）。
2. 公式-代码互证：按 §三 清单逐条打开对应脚本行号比对（允许读脚本全文，禁止运行）。
3. 实测数字溯源：教程出现的每个"🧪 实测"数字，标注"脚本内声明 / 章内引用 / 无出处"三态。
4. 格式扫描：按 §五 清单位置逐个复核定级（G 编号以战役规范原文为准）。
5. 产出：单元审计报告（发现项按 缺口/格式/一致性/加分点 四类），交主教合并。
