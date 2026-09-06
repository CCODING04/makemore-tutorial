# S2 学习报告 · Part 7 minimind（实操薄弱型学生视角）

- 审计人：学生 agent S2（动手优先，先做后看）
- 日期：2026-09-04
- 工作目录：scratch `/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P7`；作业 `/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_7`
- 范围：`courses/Part7_minimind/tutorial/` 7 个 .md + `scripts/` 13 个 + `assignments/assignment_7/` 三件

## 总分：8.4 / 10

组件讲解质量高、脚本绝大多数能跑、作业题与教程衔接顺。扣分点集中在：脚本 09 有一个学生必踩的 shape bug、MoE 一节教程代码块与脚本三处不一致、README 一处宣称被证伪、04 章引用了脚本里不存在的函数名。

## 作业结果（先做后看）

7 题全部 PASS，一次通过 7/7，未使用任何教程外提示（docstring 的分步提示已足够照写）。作业确实如 assignment.md 所说"纯随机张量、不需要数据"，`../../data/names.txt` 路径存在但做题用不到。

| 题 | 内容 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|---|---|---|---|---|
| 1 | BPE 编码（rank 最小 + 最靠左合并） | ✅ | 0 | ~2 min | PASS |
| 2 | RMSNorm | ✅ | 0 | ~1 min | PASS |
| 3 | RoPE（view_as_complex 复数乘法） | ✅ | 0 | ~3 min | PASS |
| 4 | repeat_kv（expand+reshape） | ✅ | 0 | ~1 min | PASS |
| 5 | SwiGLU 三投影 | ✅ | 0 | ~1 min | PASS |
| 6 | DPO loss（logsigmoid） | ✅ | 0 | ~1 min | PASS |
| 7 | KV Cache（拓展） | ✅ | 0 | ~1 min | PASS |

pytest 实录（`/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python -m pytest test_minimind_exercises.py -v`）：

```
test_exercise_1_bpe_encode PASSED
test_exercise_2_rmsnorm PASSED
test_exercise_3_apply_rope PASSED
test_exercise_4_repeat_kv PASSED
test_exercise_5_swiglu PASSED
test_exercise_6_dpo_loss PASSED
test_exercise_7_kv_cache PASSED
============================== 7 passed in 0.60s ===============================
```

作业唯一的"卡"感：题 3 的 `view_as_complex` 要求 reshape 成 `(..., hd//2, 2)`，作业 docstring 第 4 步写"变回 (..., head_dim//2, 2) 再 reshape 回原形"，照做即可；`freqs_cis` 需要 `unsqueeze(0).unsqueeze(2)` 广播到 `(1,T,1,hd//2)`，这一步 docstring 只说"广播到该形状"，对实操薄弱的学生是唯一需要自己想一下的地方（约 1 分钟）。

## 脚本运行台账（13 个）

| 脚本 | 状态 | 用时 | 说明 |
|---|---|---|---|
| 01_bpe_tokenizer | ✅ | 2.2s | 真 BPE（6400 词表），压缩率 3.43x（教程宣称 ≈3.5x，吻合） |
| 02_rmsnorm_rope | ✅ | ~1s | CPU 模式自动降配 |
| 03_gqa_kv_cache | ✅ | ~1s | |
| 04_swiglu_ffn_moe | ✅ | ~1s | |
| 10_moe_load_balance | ✅ | 15.9s | 随机张量，**不读 input.txt**（与 README 宣称冲突） |
| 12_mla_nsa_accounting | ✅ | 0.8s | |
| 05_full_model | ✅ | <180s | |
| 07_sft_training | ✅ | <180s | |
| 08_dpo_alignment | ✅（CPU 串行） | <180s | 首次 7 脚本并行抢 GPU 报 CUDA OOM，属我的运行方式，非脚本问题；CPU 下正常 |
| 11_rope_scaling | ✅ | <180s | 用 input.txt |
| 13_long_context_eval | ✅ | <180s | 合成任务，符合 README 宣称 |
| 06_pretrain_pipeline | ⏱ time-blocked | 180s 超时 | 训练类，教程未给单脚本预期时长，学生无法判断"在算还是卡死" |
| 09_eval_demo | ❌ broken | — | **可稳定复现的 bug**（CPU/GPU 均崩），见下 |

### 09_eval_demo.py 的 bug（最大卡点）

```
File ".../scripts/09_eval_demo.py", line 80, in apply_rotary_pos_emb
    fc = freqs_cis[:T].view(T, 1, half)
RuntimeError: shape '[128, 1, 48]' is invalid for input of size 3504
```

`freqs_cis` 预计算了 73 个位置（3504/48），生成时却需要 128 个位置。与权重、GPU 无关，CPU 下同样崩——学生按流水线跑到"三阶段验收"必然卡死。（09 的修改时间 8/30 17:01 晚于其余脚本的 13:11，疑似后续改动引入。）

## README 宣称抽查（2 条）

1. **"数据 `data/input.txt` 已在仓库内，脚本 01–11 都用它（含 09 三阶段验收、10 MoE、11 RoPE 外推）" → ❌ 证伪**。`grep input.txt` 结果：01/09/11 确实使用；**10_moe_load_balance.py 全程 `torch.manual_seed(42)` 随机张量，不读任何数据文件**。宣称应改为"01/09/11 用 input.txt；10 用随机数据"。
2. **"未安装 tokenizers 时自动回退字符级分词" → ✅ 属实**。脚本 01 确有 try-import 与清晰提示逻辑；本环境已安装，真 BPE 跑通。附带验证：教程 01 章宣称压缩率 ≈3.5×、CPU 十几秒，实测 3.43×、2.2s（数量级吻合，"十几秒"偏保守但无害）。

## 一致性核对表（教程代码块 ↔ 脚本）

| 组件 | 教程代码块 | 脚本实际 | 结论 |
|---|---|---|---|
| BPE 训练（01章） | `BpeTrainer(vocab_size=6400, special_tokens=3个)` + ByteLevel | 01 脚本一致，真跑通过 | ✅ 一致；⚠️ 教程"预期输出"写 `<\|endoftext\|>(0), <\|im_start\|>(1), <\|im_end\|>(2)`，实测日志是 `im_start=0, im_end=1`，特殊 token id 对不上 |
| RMSNorm（02章） | `eps=1e-5`，`weight*(x/rms)` | 02 脚本一致（多 fp32 `.type_as` 处理） | ✅ 一致；⚠️ 作业题 2 docstring 默认 `eps=1e-6`，与教程/脚本 1e-5 不同（测试不查，但对照时会疑惑） |
| RoPE（02章） | 用实数版 `rotate_half` 讲原理，**并明确注明**脚本用复数版（`torch.polar`+`view_as_complex`） | 02/05 脚本均为复数版 | ✅ 教程自我声明差异，处理得当；签名参数名 `end/rope_base` vs `max_seq_len/theta` 属已声明的原理版改写 |
| GQA repeat_kv（03章） | `(B,T,n_kv,hd)` + expand | 03 脚本逐字一致 | ✅ 一致；作业题 4 用 `(B,n_kv,T,hd)`，assignment.md 已加 ⚠️ 说明维度差异，好 |
| KV Cache（03章） | `cat(dim=1)`（B,T 布局） | 03 脚本一致 | ✅；作业题 7 dim=2 是 (B,heads,T,hd) 布局，自洽 |
| SwiGLU（03章） | gate/up/down 三投影 | 04 脚本一致 | ✅；hidden dim 教程已注明脚本用四舍五入版 `int(π·h/64+0.5)*64`，等价 |
| **MoE（03章）** | `__init__(hidden, num_experts=4, top_k=1)`，正文"minimind 用 4 专家/top-1"；aux_loss=`(load*scores.mean()).sum()*E*coef` | 04 脚本 `__init__(hidden_size, intermediate_size, num_experts=4, top_k=2)`；aux_loss=`E*Σ(f_i·P_i)`（Switch 公式） | ❌ **三处不一致：top_k 1 vs 2；签名缺 intermediate_size；aux_loss 公式两种写法**（脚本 10 注释的 Switch 公式才是脚本真实实现） |
| SFT masking（04章） | 引用 `build_chat_sample(question, answer, tokenizer)` + a_start/a_end | 07 脚本实际是 `make_chat_tokens(enc, user_text, assistant)` + `collate_batch` + `masked_loss` | ⚠️ **函数名对不上**（按教程名字在脚本里搜不到），逻辑等价 |
| DPO（04章） | `loss=-F.logsigmoid(beta*(reward_w-reward_l)).mean()`，ref 冻结 | 08 脚本 `dpo_loss` 同公式；作业题 6 同 | ✅ 三方一致 |
| 杂项 | 03 章 Flash Attention 处的 "Part 9 CUDA" blockquote | — | ⚠️ 同一段引用块**重复出现两次**（L230-232 与 L247-249），编辑残留 |

## 卡点清单（按严重度）

1. **[阻断] 脚本 09 shape bug**：`09_eval_demo.py:80` `freqs_cis[:T].view(T,1,half)`，73 个预计算位置 vs 需要 128，CPU/GPU 均稳定崩。三阶段验收跑不了。
2. **[困惑] MoE 三处不一致**：教程代码块 top_k=1 且声称"minimind 用 top-1"，脚本 04 是 top_k=2、签名多 intermediate_size、aux_loss 公式不同。学生逐行对照会怀疑自己抄错。
3. **[困惑] 04 章引用不存在的函数名** `build_chat_sample`（实际 `make_chat_tokens`），按名字搜索找不到。
4. **[时间] 06 预训练 180s 跑不完**（time-blocked）。教程说"CPU 缩小版"但没给每个脚本预期时长，实操学生无法区分"正常训练中"与"卡死"。
5. **[文档] README 宣称"脚本 01–11 都用 input.txt"被脚本 10 证伪**。
6. **[文档] 教程 01 章"预期输出"特殊 token id（0/1/2）与实测（im_start=0, im_end=1）不符**。
7. **[轻微] 作业 RMSNorm eps 默认 1e-6 vs 教程/脚本 1e-5**；03 章引用块重复两次。
8. **[环境提示] 脚本默认 `cuda` 且不检查显存**，多个脚本并行会 OOM（单人串行无碍；教程可加一句"脚本会自动用 GPU，显存不足时 CUDA_VISIBLE_DEVICES= 强制 CPU"）。

## 分章评分

| 章节 | 分 | 评语 |
|---|---|---|
| README | 8 | 导航/前置/演进表优秀；"01–11 都用 input.txt"一处失实 |
| 01 BPE | 9 | 手推例子（aaabdaaabac）极好；预期输出的特殊 token id 过时 |
| 02 RMSNorm/RoPE | 9.5 | 实数/复数版差异主动声明 + 手算例子，全教程最佳实践 |
| 03 GQA/FFN/MoE | 8 | GQA diff 对照、参数/缓存表优秀；MoE 小节三处不一致 + 引用块重复 |
| 04 训练流水线 | 8 | masking 对齐与 DPO 冻结讲得清楚；函数名引用失配 |
| 05 复现指南 | 8.5 | 脚本 11/13 运行通过，实验设计与作业实验题呼应 |
| 06 MLA/NSA | 8.5 | 篇幅短、脚本 12 秒过 |
| 作业 assignment_7 | 9.5 | 7/7 可一次过、属性测试友好、维度差异标注到位；仅 eps 默认值与教程不一致 |

## 只改 3 件事

1. **修 09_eval_demo.py 的 freqs_cis 长度**：预计算长度与生成最大步数对齐（如 `precompute_freqs_cis(dim, max_seq_len=max_gen_len)` 或生成时按需扩表），让"三阶段验收"能跑完。
2. **统一 MoE 小节**：教程 03 章代码块改为与脚本 04 一致（`top_k=2`、补 `intermediate_size`、aux_loss 用 `E*Σf_i·P_i`），或在代码块上方加一句"教学简版为 top-1，脚本默认 top-2"。
3. **修正两处引用失实**：04 章 `build_chat_sample` → `make_chat_tokens`；README "脚本 01–11 都用 input.txt" → 改为"01/09/11 用 input.txt，10 用随机数据"。

## 最喜欢 3 处

1. **02 章的"📝 脚本实现对照"框**：主动告诉读者"教程用实数版讲原理、脚本用复数版，两者数学等价"——这正是实操型学生最需要的"为什么我抄脚本和书上不一样"的答案。
2. **assignment.md 题 4 的 ⚠️ 维度约定说明**：明确指出作业 `(B, n_kv, T, hd)` 与脚本 03 `(B, T, n_kv, hd)` 的顺序差异并让学生按测试约定实现——避免了最常见的"照抄报 shape error"。
3. **03 章 GQA 参数/KV 缓存对照表**（MHA 524K/67MB、GQA 262K/33MB、MQA 65K/8MB）：每个数字都能用 hidden=512/8 头/8 层手工复算，"看得懂还能算得出来"。

---
*作业产物：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_7/minimind_exercises.py`（7/7 PASS，pytest 0.60s）*
