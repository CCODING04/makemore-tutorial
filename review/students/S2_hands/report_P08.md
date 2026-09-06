# S2 学习报告·P08（Part 8 后训练全流程）

**审计学生**：S2（实操薄弱型）｜**日期**：2026-09-05｜**审计范围**：tutorial/ 9 章 + README、scripts/ 13 个、assignment_8/ 三件（只读材料，未读 assignment_reference 与 REVIEW 产物）

---

## 一、总分：84 / 100（B+）

| 维度 | 得分 | 权重 | 说明 |
|---|:---:|:---:|---|
| 教程可读性 | 17/20 | 20% | 章节导航、参考来源标注、面试直通车都很好；扣分在章末作业指引两处错位 + 两章缺指引 |
| 代码可跑性 | 15/20 | 20% | 13 脚本中 7 个 RC=0；02/06/07 time-blocked、08 checkpoint 崩、12 env+超时；11 优雅退出值得表扬 |
| 教程↔脚本一致性 | 18/20 | 20% | 抽查 6 处核心代码（Head/tril、sft_loss、BT/DPO、GAE、GRPO、LoRA 数字）基本一致；LoRA 代码块漏 1 行 |
| 作业质量 | 19/20 | 20% | 8 题 docstring 带公式+提示、SKIP 机制友好、与脚本同构；扣分在 03/04 章指引与题号错位 |
| 实操体验（S2 视角） | 15/20 | 20% | 作业一晚能做完；但跑脚本链路（02→08）在 GPU 环境下断裂，学生容易卡住 |

---

## 二、卡点清单（按严重度排序）

1. **[高] 章末作业指引与实际题号错位（03、04 章）**：03 章指引"题 6（ORPO）"，实际题 6 是 GAE，ORPO 在作业中根本不存在；04 章指引"题 7（GAE）和题 8（PPO Clipped）"，实际题 6=GAE、题 7=PPO、题 8=GRPO——且 04 章漏掉了真正的题 8（GRPO），而 GRPO 恰是 04 章主讲内容。学生按指引翻作业会找不到北（推测作业题序调整后教程末尾未同步）。
2. **[高] 脚本编号冲突：两个 09**：`09_quantize_and_serve.py`（06 章）与 `09_reasoning_models.py`（09 章）并存，README 导航表里都写 `09`。且最后是 `12_lm_eval_hands_on.py`，中间 10（LoRA）对应 08 章、11 对应 07 章——脚本序号与章节序号完全错开，学生"第 N 章→脚本 N"的心智模型失效（README 表有列对应脚本，算是缓解，但双 09 仍是隐患）。
3. **[高] 08_eval_and_chat.py 加载 checkpoint 直接崩**：实测报 `size mismatch for pos: (512,) vs (64,)`、`tok_emb.weight (65,512) vs (65,64)`，另附大量 Unexpected keys——checkpoint 是 GPU 模式（n_embed=512、12 blocks）训的，脚本按另一模式建模型，跨模式加载无防护。02（预训练）又被 time-blocked 没能产出匹配 checkpoint，"预训练→评估"主链路在本机断裂。
4. **[中] 参数量数字矛盾**：06 章正文说"本课 ~0.4M 玩具模型"，README 规模表说 CPU 模式 ~2M；本机检测到 cuda 走 GPU 模式（~40M），学生对照教程数字会更晕。
5. **[中] 02/06/07 在 180s 内跑不完**：GPU 模式 40M 模型训练量大（README "<30s" 是对 CPU 小模式说的，预期管理没错，但实操学生超时挫败感强）；07_grpo 同类。
6. **[中] 12_lm_eval 首跑被 HF 网络重试拖死**：离线/代理不可达环境下 Retry 5 次超过 3 分钟（rc=124）。11 的"MISS→打印指引→rc=0"优雅退出是好范本，12 应对齐。
7. **[低] 05、07 章无"📝 课后作业"段**：05 章只有章内 `<details>` 练习；07 章的观测题 C（评估污染审查）其实存在于 assignment.md，但章末没有指引指向它。
8. **[低] 教程 08 章 LoRALinear 代码块漏一行**：`__init__` 未保存 `self.r, self.alpha = r, alpha`（脚本 10 第 88 行有），照抄教程会 AttributeError。

---

## 三、分章评分（10 分制）

| 章 | 文件 | 对应脚本 | 分 | 一句话评 |
|---|---|---|:---:|---|
| 01 | 01_gpt_and_pretrain.md | 01 02 | 9 | Head/tril 代码与脚本逐行一致，GPT-2 vs 现代架构对照清晰；02 脚本 time-blocked |
| 02 | 02_sft_and_chat.md | 03 | 9 | sft_loss 四步讲解是全包最清楚的 prompt masking 教程；03 脚本 3 分钟内跑通 |
| 03 | 03_reward_and_dpo.md | 04 05 | 7 | BT/DPO/ORPO 公式与脚本 05 一致；**章末题 6=ORPO 指引错误** |
| 04 | 04_ppo_and_grpo.md | 06 07 | 7 | GAE/GRPO 讲解+对比表极好；**章末题 7/8 指引错位、漏 GRPO 题**；06/07 均超时 |
| 05 | 05_eval_and_deploy.md | 08 | 7 | top_p 明示"留作练习"诚实；**缺章末作业指引**；08 实测 checkpoint 崩 |
| 06 | 06_inference_and_serving.md | 09_quantize | 9 | "decode 是 memory-bound"一个原理打全部；四实验全可复现（RC=0）；~0.4M 数字与 README 矛盾 |
| 07 | 07_evaluation.md | 11 12 | 8 | 评估学+幻觉安全+合规四件套，覆盖面全包最强；观测题 C 未在章末指引 |
| 08 | 08_lora_and_classification.md | 10 | 8 | 教程数字与脚本实测完全一致（0.955/0.924）；代码块漏 1 行 |
| 09 | 09_reasoning_models.md | 09_reasoning | 8 | R1 四阶段表+self-consistency 实测（56%→46%→58%）诚实呈现"n 大不一定更好" |
| README | — | — | 9 | 导航/参考来源双表+规模对照表+多卡备注，全课程 README 模板水准；唯双 09 未解释 |

---

## 四、一致性核对表（教程代码块 ↔ 脚本逐段抽查）

| # | 教程位置 | 脚本位置 | 核对内容 | 结论 |
|---|---|---|---|---|
| 1 | 01 章 L72-80 | 01_gpt_model.py L70-79 | `register_buffer('tril',...)` + `masked_fill(tril==0,-inf)` | ✅ 逐行一致 |
| 2 | 02 章 L98-115 | 03_sft.py L180+ | sft_loss：shift→逐 token CE(reduction='none')→乘 mask→除 mask 和 | ✅ 结构一致（教程四步注释更细） |
| 3 | 03 章 L54/L158/L219 | 05_dpo_alignment.py L217/L273 | `-F.logsigmoid(r_ch-r_rj).mean()`、DPO `-logsigmoid(beta*logits)`、ORPO log_odds | ✅ 公式一致 |
| 4 | 04 章 L117-124 | 06_ppo_training.py | GAE 反向递推（教程用 values_next+nonterminal 变体，脚本等价） | ✅ 等价 |
| 5 | 04 章 L281-293 | 07_grpo_training.py L183+ | grpo_loss（clip+k3 KL+resp_mask）、`group_advantages(rewards, group_size, eps=1e-4)` | ✅ 一致（与作业题 8 同签名） |
| 6 | 05 章 L32-95 | 08_eval_and_chat.py | evaluate_model(temperature, top_k)、top-k 截断 | ✅ 一致（top_p 双方均未实现，教程已声明） |
| 7 | 06 章 §1 表格 | 09_quantize_and_serve.py 头注释 | int8 Δ/int4 g128/int4 无分组三行量化数字 + speculative gamma | ✅ 对应（四实验一一对齐） |
| 8 | 07 章 L58/L198 | 12 / 11 脚本 | §2 lm-eval 实操→脚本 12；§6/7 幻觉安全→脚本 11 实验 A/B/C | ✅ 对应 |
| 9 | 08 章 §2 代码块 | 10_lora_from_scratch.py L77-89 | LoRALinear | ❌ **教程漏 `self.r, self.alpha = r, alpha`（脚本 L88）**；实测输出数字（230,226/7,872/0.955/0.924）与教程引用完全一致 |
| 10 | 09 章 §2 | 09_reasoning_models.py | 玩具加法 SFT→self-consistency n=1/4/8 | ✅ 一致 |

### 章节-作业映射表（重点核对项）

| 章 | 章末指向题号 | assignment_8 实际题号 | 对? |
|---|---|---|:---:|
| 01 | 题 1（单头注意力 Head）+ 题 2（Pre-LN Block） | 题 1 Causal Head、题 2 Pre-LN Block | ✅ |
| 02 | 题 3（Prompt-Masked SFT Loss） | 题 3 SFT Loss | ✅ |
| 03 | 题 4（Bradley-Terry）+ 题 5（DPO）+ **题 6（ORPO）** | 题 4 BT ✅、题 5 DPO ✅、**题 6 = GAE，ORPO 无对应题** | ❌ |
| 04 | **题 7（GAE）+ 题 8（PPO Clipped）** | **题 6 = GAE、题 7 = PPO、题 8 = GRPO**（04 章未提 GRPO 题） | ❌ 全错位 |
| 05 | 无"📝 课后作业"段（仅章内 `<details>` 练习） | — | ⚠️ 缺指引 |
| 06 | "综合题不变" + 跑通 09_quantize 四实验 | 实验 A（量化实测）、实验 B（投机解码扫描）均在 assignment.md | ✅ |
| 07 | 无"📝 课后作业"段 | 观测题 C（评估污染审查）对应 07 章 §4，但章末未指引 | ⚠️ |
| 08 | 跑通脚本 10 + 三个对比实验（非题号） | 无对应自动测试题（符合"观测型不进测试"设计） | ✅ |
| 09 | 明示"无独立作业" | — | ✅ |

---

## 五、作业元数据 + pytest 输出

- **文件**：`assignment_8/`{assignment.md, post_training_exercises.py(12.5KB), test_post_training_exercises.py(17.4KB)}
- **题量/分值**：8 题 = 5 基础×12 分 + 2 拓展×15 分 + GRPO×10 分 = 100 分；另有实验 A/B + 观测题 C（不进自动测试）
- **做题方式**：先做后看（写完 8 题才首次运行 pytest）；编辑仅 exercises 文件

**逐题记录**：

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|:---:|:---:|---|---|
| 1 Causal Head | ✅ | 0 | ~2min | PASS |
| 2 Pre-LN Block | ❌ 第 2 次过 | 0（自悟） | +3min | PASS（首版单 Head 输出 head_size=8 与残差 32 不匹配；docstring 只说"MultiHeadAttention（用题 1 的 Head）"未明说需 ModuleList 并联拼接，题面小坑） |
| 3 SFT Loss | ✅ | 0 | ~2min | PASS |
| 4 BT Reward | ✅ | 0 | <1min | PASS |
| 5 DPO Loss | ✅ | 0 | ~2min | PASS（返回三元组 docstring 已写明） |
| 6 GAE | ✅ | 0 | ~3min | PASS |
| 7 PPO Clipped | ✅ | 0 | ~2min | PASS |
| 8 GRPO Advantage | ✅ | 0 | ~2min | PASS |

**最终 pytest 输出**（第 2 次运行；第 1 次 7 passed / 1 failed）：

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.1.1
collected 8 items

test_post_training_exercises.py::test_exercise_1_causal_head PASSED      [ 12%]
test_post_training_exercises.py::test_exercise_2_preln_block PASSED      [ 25%]
test_post_training_exercises.py::test_exercise_3_sft_loss PASSED         [ 37%]
test_post_training_exercises.py::test_exercise_4_reward_loss PASSED      [ 50%]
test_post_training_exercises.py::test_exercise_5_dpo_loss PASSED         [ 62%]
test_post_training_exercises.py::test_exercise_6_gae PASSED              [ 75%]
test_post_training_exercises.py::test_exercise_7_ppo_loss PASSED         [ 87%]
test_post_training_exercises.py::test_exercise_8_group_advantages PASSED [100%]

============================== 8 passed in 0.62s ===============================
```

**作业结论：8/8 PASS（一稿 7 过 1 挂，二稿全过；全程 0 求助提示，总用时 ~17 分钟）**

---

## 六、脚本运行台账（13 个，scratch 日志在 review/scratch/S2_P8/）

| 脚本 | 结果 | 证据 |
|---|---|---|
| 01_gpt_model.py | ✅ RC=0 | 架构打印+与 Part 7 对比正常 |
| 02_pretrain.py | ⏱ time-blocked（rc=124 @180s） | GPU 模式 40M 训练量大 |
| 03_sft.py | ✅ RC=0 | 180s 内完整跑完 |
| 04_reward_model.py | ✅ RC=0 | BT 训练完成 |
| 05_dpo_alignment.py | ✅ RC=0 | DPO 对齐完成 |
| 06_ppo_training.py | ⏱ time-blocked（rc=124 @180s） | 同 02 |
| 07_grpo_training.py | ⏱ time-blocked（rc=124 @180s） | 同 02 |
| 08_eval_and_chat.py | ⚠️ env/state-blocked（rc=124，实际秒崩） | checkpoint 规模不匹配（512 vs 64），见卡点 3 |
| 09_quantize_and_serve.py | ✅ RC=0 | 四实验全跑完 |
| 09_reasoning_models.py | ✅ RC=0 | self-consistency n=1/4/8：56%/46%/58% |
| 10_lora_from_scratch.py | ✅ RC=0 | acc 0.955 vs 0.924（与教程一致） |
| 11_hallucination_safety.py | 🌐 env-blocked 但优雅退出 rc=0 | HF 模型无缓存+代理不可达，打印指引后跳过 |
| 12_lm_eval_hands_on.py | 🌐 env-blocked + ⏱（rc=124） | HF HEAD 请求 Retry 5 次拖过 180s |

**重复编号确认**：`09_quantize_and_serve.py` 与 `09_reasoning_models.py` 两个 09 并存（无内容冲突但编号冲突）；最大编号 12，缺一个"09 之后顺延"的 13 号位。

---

## 七、只改 3 件事

1. **同步章末作业指引**（03/04 章各一行）：03 章改为"题 4（Bradley-Terry）和题 5（DPO）"（删 ORPO 或注明"ORPO 不设作业题"）；04 章改为"题 6（GAE）、题 7（PPO Clipped）和题 8（GRPO）"。成本 2 分钟，消除学生最大困惑点。
2. **教程 08 章 LoRALinear 代码块补回 `self.r, self.alpha = r, alpha`**（对齐脚本 10 L88），并在 06 章把 "~0.4M" 与 README "~2M/~40M" 统一口径。
3. **解决双 09 + 加载防护**：把 `09_reasoning_models.py` 重命名为 `13_reasoning_models.py`（或 README 明示"09x 双脚本"约定）；给 08_eval_and_chat.py 加 checkpoint 规模自检——size mismatch 时打印"checkpoint 为 GPU 模式（n_embed=512），请先跑通 02 或设 CPU_MODE 强制重建"而非抛长堆栈。

## 八、最喜欢 3 处

1. **作业 docstring 的"公式+提示+验证标准"三段式**：做题全程零外部求助，`_skip` 机制让未做题不报错——这是我做过的作业里对自学者最友好的一份。
2. **教程数字与脚本实测严格对齐**：08 章引用的 230,226/7,872 参数、0.955/0.924 acc 与我跑出的输出逐字一致；06 章量化三行数字与 09 脚本输出对应——"看趋势别死记数字"的声明下仍给了可复现实测，诚意足。
3. **README 的"参考来源标注"双列表**：明确两个源仓库各管哪章哪些模块（主源跑通流程、rasbt 做扎实），加上规模对照表和"超参因果"四条——把"抄哪个仓库、什么时候"讲成了方法论。

---

*S2（实操薄弱型）：作业部分近乎满分体验，扣分集中在"跑通全流程"链路——教程写得好，但 02→08 这条主线在我机器上断了两处。*
