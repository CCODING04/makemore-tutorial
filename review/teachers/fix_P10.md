# Part 10（分布式训练）T2 整改报告

- 整改日期：2026-09-04 · 执行：T2 整改教师
- REPO 只读；全部修改写入 REVIEW 镜像；脚本一律先 cp 到 `REVIEW/scratch/t2_P10/` 实跑（G16）
- 环境：RTX 4090 + 4090 D（2×24GB，混合机型）· torch 2.6.0+cu124 · Python 3.12（REPO venv）

## 一、镜像清单（本次交付）

| 文件 | 改动 |
|---|---|
| `courses/Part10_distributed/tutorial/README.md` | 总述句 CPU 承诺修正、表格措辞、假文件名、3 处 LaTeX |
| `courses/Part10_distributed/tutorial/01_why_and_collectives.md` | "三/四"口径统一、LLaMA 句补全、实测块"节选"+口径 |
| `courses/Part10_distributed/tutorial/02_ddp.md` | 6 个推导块 LaTeX、ASCII 时序图→PNG+中文解读、桶数验证修正（实证 1→2 桶）、五件套/nullcontext 声明、RMSNorm 链接 |
| `courses/Part10_distributed/tutorial/03_memory_zero_fsdp.md` | 账本/三阶段/恒等式 LaTeX、ZeRO 柱状图、激活公式出处、FSDP 实测口径+需 GPU 注、Q3 拆行 |
| `courses/Part10_distributed/tutorial/04_tp_pp_and_beyond.md` | Q2 off-by-one 修正、bubble 3 行推导、TP 形状链表+术语辨析框、GPipe 图、TP×PP×DP LaTeX、MFU 46%→43.9%+出处、NVLink/IB 规格出处、Q1/Q2 LaTeX |
| `courses/Part10_distributed/scripts/01_distributed_basics.py` | all_gather 补 assert |
| `courses/Part10_distributed/scripts/02_ddp_gpt.py` | 训练后打印实测桶数（私有 API 注明+时点语义） |
| `courses/Part10_distributed/scripts/03_zero_memory.py` | fp16→bf16 措辞统一、docstring 如实（仅 ZeRO-1 布局） |
| `courses/Part10_distributed/scripts/04_fsdp_gpt.py` | docstring 如实"需 GPU" + main() 开头设备预检友好报错 |
| `courses/Part10_distributed/scripts/06_pipeline_parallel.py` | set_device 取模（与 01/02/04 一致） |
| `courses/Part10_distributed/images/*.png` | 新增 3 图（ddp_bucket_overlap / gpipe_timeline / zero_memory_bar） |
| `assignments/assignment_10/assignment.md` | 完成方式句 04 需 GPU、Q2 提示四点拆行、ZeRO/bubble 公式 LaTeX |
| `assignments/assignment_10/test_distributed_exercises.py` | SKIP 语义修复（pytest.skip + _skip_exceptions） |
| `outline_review/outline_suggestions.md` | 追加 P10 三条 roadmap 建议（不改 docs） |
| `ledger/ledger_P10.md` | 本 Part 台账 |
| `scratch/t2_P10/` | 实跑日志（logs_/relogs_/fin_ 三轮）、生成图脚本、探测脚本 |

## 二、必修清单执行情况（T0 八项全闭环）

1. **脚本 04 CPU 假承诺（P0）**：实测 CPU 档 `RuntimeError: FSDP needs a non-CPU accelerator device`（rc=1）复现；裁决=README 表格对、脚本 docstring 与 README 总述句错。修：脚本 docstring 如实 + `main()` 开头设备预检（打印人话指引后 `sys.exit(1)`）；README 总述句改"除脚本 04 需 GPU 外…"；assignment 完成方式句同步；教程 03 章（脚本 04 的落点章）§2 注明"需要 GPU"+单卡 NO_SHARD 读数。✅
2. **04 章 Q2 off-by-one（P0）**：复算 m=63 → 7/70 恰 10.00% 不满足严格 <10%；修正为 m≥64（m=64 → 7/71≈9.86%），并补 3 行排槽推导（总槽 2(m+p−1)、气泡槽 2(p−1)、约分）+ "m<p 也成立（p=4,m=1→75%）"边界。✅
3. **TP 形状链 + 术语地雷**：04 章新增形状逐跳表（数字例 b=8,s=16,h=256,4h=1024,tp=2：X→H_r→Y_r→all-reduce）+「按行切=列并行」辨析框（W 转置存法换算口诀）+ f 的 backward 为手工 all_reduce 的实现差异声明。✅
4. **桶数验证修正**：私有 API 落进脚本 02（训练后打印实测桶数）；教程给可行验证步骤=跑脚本 02 看打印行。**升级实证**：双卡实测首次 backward 后 1 桶、rebuild_buckets 后稳态 **2 桶**（首桶 ~1MB 默认上限重切），教程原"只建 1 个桶"断言失真——改为"个位数量级"如实口径，教学结论（toy 桶间重叠无从体现）不变。✅
5. **fp16/bf16 统一**：脚本 03 docstring/注释两处改 bf16，注明"课程主线 bf16，字节数相同"。✅
6. **作业 SKIP 语义（G18）**：仿 P8 修法，`_skip()` 在 pytest 可用时抛 `pytest.skip`，新增 `_skip_exceptions()` 供直跑入口与 except 链使用。✅
7. **格式 G1-G16 全量**：G2×21 全 LaTeX 化（02 章六块推导、03 章表格/恒等式/激活公式、04 章公式链、README、assignment）；G1×2 拆行（03 章 Q3 答案、assignment Q2 提示）；G4×3 图（DDP 重叠时序=机制示意、GPipe 时间线=机制示意、ZeRO 柱状图=公式值 112/38.5/26.25/14 GB，1e9 口径，图注已注明）；G10×6；G13×3；G14×3。✅
8. **T1 十个 C1 缺口**：1（三/四口径）、2（=必修1）、3（=必修2）、4+roadmap 两处（outline 追加）、5（脚本 03 docstring）、6（bf16）、7/8（TP 形状链+差异声明）、9（LLaMA 截断句补全）、10（RMSNorm 链接+桶数，见上）逐条处理。✅

## 三、验证结果（必跑三项全绿）

1. **脚本**（`scratch/t2_P10/`，三轮日志）：
   - 修后 CPU 档：01/02/03/05/06 rc=0；04 rc=1 但为**友好报错+指引**（不再裸崩 RuntimeError）；
   - 修后 GPU 单进程：6 个 rc=0（04 单卡 NO_SHARD：11.8MB/12.7MB/loss 3.005）；
   - 双卡 torchrun×2：01/02/03/04/05/06 全 rc=0；验收数字逐字复现——TP 5.96e-07、PP 4.380254、FSDP 6.0MB+2.981、DDP loss 3.612（带 3.6 内）、桶数打印=rebuild 后 2。
2. **作业**：参考答案 + 修后 test 四象限全绿——pytest 5 passed；直跑 5/5"全部通过"；空跑 pytest=4 failed（题 1-4 未实现应失败）+ 1 skipped（题 5 ⏭️ 不算失败，两入口行为一致）。
3. **check_latex.py**：6 个改动 md 全部"未发现问题"（0 问题）。

## 四、通用问题候选（≤3）

1. **"总述承诺 vs 明细表格 vs 脚本 docstring"三源口径漂移**：README 一句总述扫全场（"全部脚本 CPU 也行"），表格精确但没人读（漏 04 恰是对的），docstring 第三源说反——同一承诺至少三处拷贝，改一处漏两处。建议全课程固定"环境表格为唯一事实源，总述句只做复述"。
2. **实测数字不带档位四元组**：脚本 04 单进程 12.7MB/3.005 vs 双卡 6.0MB/2.981、脚本 05 单进程 0.00e+00 vs 双卡 5.96e-07、DDP 桶数 1（首 backward）vs 2（rebuild 后）——同一量在不同 world/档位/时点读数不同，教程引用时必须带（world × device × torch 版本 × 时点/档位），否则学生复现不出就以为自己的错。
3. **外部数字裸奔**（G13 的根因）：MFU "46%"（论文 43.9%）、NVLink "~900GB/s"（分代规格）、激活公式 34+5as/h（有出处未标）——外部来源数字一律"链接 + 论文值/本课实测值分列"。

## 五、好写法候选（≤3）

1. **02 章三步推导 + 动手 2 的"翻车-救回"**：每步只引入一个事实（线性→分组→对照 DDP），再用 10 行代码让"不等大分组"误差跳 7 个数量级再加权救回——把推导前提做成可亲眼看到的实验。
2. **03 章"公式 + 逐字节记账断言"双轨互证**：公式印在明面，脚本 03 Part B 逐张量累加并 assert 与 4Ψ+12Ψ/N 一致——面试被追问"12Ψ/N 怎么来的"时有实弹。
3. **真实 war story 三呼应**：4090+4090D 混合机型 NCCL 集合通信正常但 p2p 卡死 → 点对点单独 gloo 组 + CPU 中转，在 01 章 FAQ、04 章 §2、脚本 06 注释三处互指且本机实测有效——"分布式问题不总是逻辑 bug"的可复现教学。
