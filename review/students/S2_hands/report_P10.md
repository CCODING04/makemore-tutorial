# S2 学习报告 · Part 10 分布式训练（实操薄弱型学生视角）

- 审计人：S2（学生 agent，实操薄弱型：以"照着跑、照着做"为主，理论推导少抠）
- 日期：2026-09-04
- 环境：RTX 4090×2（4090+4090D 混合机型）· torch 2.6.0+cu124 · Python 3.12 · NCCL/gloo
- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P10`（脚本实测）、`.../students/S2_hands/work/assignment_10`（作业）

---

## 总分：92 / 100

一句话：教程-脚本-作业三方数字几乎全部逐字复现（含 5.96e-07、4.380254、2.981、6.0MB 等验收数字），
是我审过的部分里"实测可复现度"最高的一章；扣分集中在**一处承诺虚假**（脚本 04 CPU 档必崩，
三方口径互相矛盾）和**测试入口行为不一致**（pytest 下题 5 的 SKIP 失效）。

---

## 卡点清单（按严重度）

| # | 严重度 | 卡点 | 证据 |
|---|---|---|---|
| K1 | 🔴 高 | **脚本 04 CPU 档必崩，三方矛盾实锤**。脚本 04 docstring 自称"单进程兼容（gloo/CPU）"，README 总述句（L65）说"全部脚本单进程可直接跑……CPU 也行"，但 README 环境表格（L70）CPU 档只列 01/02/03/05/06。实测 `CUDA_VISIBLE_DEVICES= python 04_fsdp_gpt.py` → `RuntimeError: FSDP needs a non-CPU accelerator device`（FSDP1 在 torch 2.6 根本不支持 CPU）。**结论：表格是对的，总述句和脚本 docstring 是错的** | scratch 实测输出；`04_fsdp_gpt.py` L16；`README.md` L65 vs L70 |
| K2 | 🟡 中 | **pytest 入口下题 5 的"SKIP 不算失败"失效**。未实现时空跑 `pytest -v` → 题 5 显示 **FAILED**（自定义 `_Skipped` 异常不是 `pytest.skip()`），只有 `python test_distributed_exercises.py` 直跑入口才打印 ⏭️ SKIP。而 assignment.md"完成方式"同时写了 `python test_...py # 或 pytest test_...py`——两个入口行为不一致，纯 pytest 学生会被迫以为选做也挂了 | 空跑 pytest 输出：`5 failed`（含题 5） |
| K3 | 🟡 中 | **教程 02 §3 让读者"可验证桶数=1"，但脚本 02 里没有这段验证代码**。`len(ddp.reducer._get_zeros_like_grad_buckets())` 是私有 API，教程给了名字没给位置；我手动跑了一遍：API 存在且确实返回 1（结论为真，但学生按教程去找会扑空） | 手动验证输出 `bucket count: 1`；脚本 02 全文无此调用 |
| K4 | 🟢 低 | **教程 02 §1 五件套代码块是"纯 NCCL 简化版"**：`init_process_group(backend="nccl")` + `device_ids=[rank]`。无 GPU 学生照抄会 init 即崩；脚本的真实写法（gloo 回退 + `rank % device_count()` + `device_ids=None`）教程没提。教学简化可以接受，但对"只有 CPU"的目标读者是个暗坑 | 教程 02 §1 vs `02_ddp_gpt.py` L29-39/L108 |
| K5 | 🟢 低 | **fp16 vs bf16 措辞漂移**：教程 03 §1 账本表和作业题 2 写 bf16 参数/梯度，脚本 03 注释写"fp16 参数/fp16 梯度"。16Ψ 账本数字不受影响，但逐字对照时会疑惑 | 教程 03 §1 表 vs `03_zero_memory.py` L72-73 |
| K6 | 🟢 提示 | 我的实测吞吐单卡 178k tokens/s（教程写 180k–185k）、双卡 93.5k（教程带 77k–95k）、单卡 loss 3.293（教程 ≈3.3）、双卡 loss 3.675（教程 ≈3.6，动手 1 验收带 3.607–3.687）——**全部落在教程声明的波动带内，复现成功**，不算卡点；仅提醒 run-to-run 波动大，教程自己也声明了 | torchrun ×2 实测 |

---

## 分章评分

| 章节 | 分 | 评语（实操视角） |
|---|---|---|
| README | 88 | 学习路线、四并行对照表、"无多卡学习路径"表格都很好；但 L65 总述句与 L70 表格、脚本 04 三方矛盾（K1），且"本课开发机验证环境"段把验收数字全押在 4090×2 上，纯 CPU 读者要自己发现 04 跑不了 |
| 01 为什么并行 + 集合通信 | 96 | "01 章不卡人"目标达成：单进程 0.86s 跑通，四原语输出与教程引用逐字对应（单进程 all_reduce 得 [1,1,1,1]，双卡即教程的 [3,3,3,3]）；torchrun 六类报错 FAQ 是真金白银 |
| 02 DDP | 95 | 三步推导 + 三前提表 + 动手 1/2 设计极佳；实测全部复现。扣分：K3（桶数验证缺位）、K4（简化代码块对 CPU 读者不友好） |
| 03 显存账本与 ZeRO/FSDP | 94 | 16Ψ 账本 + ZeRO 三阶段公式讲解清楚；脚本 03 纯 CPU 0.76s 跑通，Ψ=6,298,624、100.8MB 与教程动手 2 的数字逐字一致；04 双卡实测 6.0MB / loss 2.981 **逐字复现教程引用**。扣分：K5 |
| 04 TP/PP 与工业栈 | 93 | "进阶可选"定位准确。脚本 05 双卡 5.96e-07 / 5.82e-11、脚本 06 loss 4.380254 == 4.380254 全部逐字复现；NCCL p2p 卡死 → gloo 组 + CPU 中转的工程解法在本机（正是 4090+4090D）真实有效，torchrun ×2 跑 06 一次通过 |

---

## 一致性核对表（教程代码块 ↔ 脚本逐段）

| 对照点 | 教程位置 | 脚本位置 | 结论 |
|---|---|---|---|
| DDP 封装：init/set_device/DDP 包装 | 02 §1 五件套 | 02 `setup()` L29-39、L108 | ✅ 一致；教程为 NCCL 简化版，脚本多 gloo 回退+取模（K4） |
| DistributedSampler + `shuffle=False` + `drop_last` | 02 §1 与 §2 前提表 | 02 L101-103 | ✅ 一致 |
| `sampler.set_epoch(epoch)` | 02 §1 ④ | 02 L117 | ✅ 一致（教程 FAQ 表也覆盖） |
| no_sync 梯度累积（`is_last`/`nullcontext`/`(loss/accum).backward()`） | 02 §4 代码块 | 02 L120-129 | ✅ 逐行同款；脚本多 `clip_grad_norm_`（教程未画，属简化非矛盾）；"accum=2 通信减半"与脚本默认 accum=2 一致 |
| 桶化 all-reduce、桶数=1 | 02 §3 | 脚本 02 无验证代码 | ⚠️ 结论对（手动验证=1）但 K3：教程引用了脚本外的私有 API |
| ZeRO 公式 16Ψ / 4Ψ+12Ψ/N / 2Ψ+14Ψ/N / 16Ψ/N | 03 §1 表 | 03 Part A 打印 | ✅ 逐字一致（含"ZeRO-2 列教程没列、留动手 1"的自洽安排） |
| Part B 逐字节模拟只断言 ZeRO-1 | 03 §1（明确说明） | 03 L114 断言 | ✅ 一致，教程如实交代"ZeRO-2/3 留动手 2" |
| TinyGPT Ψ=6,298,624、16Ψ=100.8MB | 03 动手 2 验收 | 03 实测输出 | ✅ 逐字一致 |
| fp16 vs bf16 措辞 | 03 §1 表（bf16） | 03 注释（fp16） | ⚠️ K5 措辞漂移 |
| FSDP 整体包裹（教学版不传 auto_wrap_policy） | 03 §2 专门说明 | 04 L127-130 | ✅ 一致，教程主动交代了与工业写法的差异 |
| FSDP 双卡数字 6.0MB / loss 2.981 | 03 §2 引用 | 04 双卡实测 | ✅ **逐字复现**（我实测 6.0 MB / 2.981） |
| TP：W1 列并行（按输出维）/ W2 行并行 / f-g 算子 | 04 §1 | 05 L61-77、L89 | ✅ 一致；脚本 B 为 W2 转置存法故按 dim0 切（注释解释清楚）；f 在脚本中用 backward 后手动 all-reduce 实现而非 autograd.Function（等价） |
| TP 双卡 5.96e-07 / 5.82e-11 | 04 §1 引用 | 05 双卡实测 | ✅ **逐字复现** |
| PP：stage 切分（emb+blocks[:2] / blocks[2:]+head）、GPipe 反序 backward | 04 §2 图示 | 06 L152-186 | ✅ 一致 |
| PP loss 4.380254 == 单进程 | 04 §2 引用 | 06 双卡+单进程实测 | ✅ **逐字复现** |
| NCCL p2p 卡死 → 点对点 gloo 组 + CPU 中转 | 01 彩蛋坑、04 §2 | 06 L71-73、L119-120 | ✅ 一致且本机实测有效 |
| bubble (p−1)/(m+p−1)，p=2,m=4→20% | 04 §2 | 06 输出 | ✅ 一致 |
| loss 汇总"不能 /world"的坑（rank0 贡献为 0） | 作业 Q4 提示 | 06 L187-191 注释 | ✅ 教程-脚本-作业三方呼应 |
| README"无多卡学习路径"表：CPU 档 01/02/03/05/06 | README L70 | 实测 | ✅ 表格对：5 个脚本 CPU 强制档全 PASS（02 CPU 也 1.6s 跑通，loss 3.278）；❌ 04 崩 → 但 README L65 总述句"全部脚本……CPU 也行"与自家表格矛盾（K1） |

**脚本实测台账**（全部在 scratch/S2_P10 下执行）：

| 脚本 | 启动方式 | 结果 | 关键输出 | 耗时 |
|---|---|---|---|---|
| 01 | 单进程 / CPU 强制档 | ✅ PASS | 四原语验证通过 | 0.86s |
| 02 | 单进程 GPU / CPU 强制档 | ✅ PASS | loss 3.293(GPU)/3.278(CPU)，178k tok/s | 1.6s |
| 02 | torchrun ×2 | ✅ PASS | loss 3.675，93,486 tok/s（带内） | 3.2s |
| 03 | 单进程 CPU | ✅ PASS | Ψ=6,298,624，ZeRO-1=63.0MB，断言过 | 0.76s |
| 04 | 单进程 GPU | ✅ PASS | 12.7MB / loss 3.005 | 1.6s |
| 04 | **CPU 强制档** | ❌ **FAIL** | `FSDP needs a non-CPU accelerator device` | 1.2s |
| 04 | torchrun ×2 | ✅ PASS | 6.0MB / loss 2.981（=教程逐字） | 3.1s |
| 05 | 单进程 / CPU 强制档 | ✅ PASS | 误差 0.00e+00（world=1 稠密参照） | 0.96s |
| 05 | torchrun ×2 | ✅ PASS | 5.96e-07 / 5.82e-11（=教程逐字） | ~3s |
| 06 | 单进程 / CPU 强制档 | ✅ PASS | 单进程 loss 4.380254 | 0.98s |
| 06 | torchrun ×2 | ✅ PASS | 流水线 loss 4.380254（== 单进程） | 3.2s |

无脚本超 3 分钟，无 time-blocked 项。

---

## 作业元数据 + pytest 输出

- 目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_10`
- 只编辑了 `distributed_exercises.py`（exercises），题目与测试文件未动
- 做题顺序：先读 assignment.md 题面自己推导 → 实现全部 5 题 → 一次跑 pytest
- 命令：`cd .../work/assignment_10 && /home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python -m pytest test_distributed_exercises.py -v`

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|---|---|---|---|
| 1 all_reduce 平均语义 | ✅ 一次过 | 0 | ~1 min | PASS |
| 2 显存账本计算器 | ✅ 一次过 | 0 | ~2 min | PASS |
| 3 DistributedSampler 不重不漏 | ✅ 一次过 | 0（骨架 docstring 四步算法≈半送分；仅手推 n=5,world=4"补齐不撞车"花约 1 min） | ~3 min | PASS |
| 4 TP 分块数学 | ✅ 一次过 | 0 | ~2 min | PASS |
| 5 🌟 流水线气泡（选做） | ✅ 一次过 | 0 | ~1 min | PASS |

```
test_distributed_exercises.py::test_exercise_1_allreduce_mean PASSED     [ 20%]
test_distributed_exercises.py::test_exercise_2_memory PASSED             [ 40%]
test_distributed_exercises.py::test_exercise_3_sampler PASSED            [ 60%]
test_distributed_exercises.py::test_exercise_4_tp_math PASSED            [ 80%]
test_distributed_exercises.py::test_exercise_5_bubble PASSED             [100%]
============================== 5 passed in 0.60s ==============================
```

**5 题（含选做题 5）全 PASS**。实现质量自评：题 3 的 `with_torch=False` 纯 Python 回退、题 2/5 的未知值 ValueError 属骨架要求之外我主动补的防御，测试均通过。空跑基线（实现前）：题 1-4 FAIL、题 5 在 pytest 下 FAILED（见 K2）。

---

## 只改 3 件事

1. **修 K1（04 的 CPU 谎言）**：`04_fsdp_gpt.py` docstring 的"单进程兼容（gloo/CPU）"改为"需要 GPU（FSDP1 在 torch 2.6 不支持 CPU 加速器）"，并在 `main()` 开头加 `if not torch.cuda.is_available(): log("脚本 04 需要 GPU…"); return`；README L65 总述句同步改为"除脚本 04 需 GPU 外，全部脚本单进程可直接跑"。
2. **修 K2（pytest 入口的 SKIP 失效）**：`test_distributed_exercises.py` 的 `_skip()` 在检测到 pytest 在场时改用 `pytest.skip(reason)`（或把 `_Skipped` 挂进 `pytest.ExceptionInfo` 可识别的异常链），保证 `python test_*.py` 与 `pytest` 两个入口行为一致——兑现 assignment.md"两个入口均可"的承诺。
3. **修 K3（桶数验证缺位）**：把教程 02 §3 的桶数验证落进脚本 02（训练后加两行：`len(ddp.reducer._get_zeros_like_grad_buckets())` 打印并注明是私有 API、版本敏感），或在教程里直接给可复制的验证代码——别让读者对着一个脚本里不存在的调用猜。

## 最喜欢 3 处

1. **02 章三步推导 + 动手 2 的"翻车-救回"设计**：先推导"all-reduce 平均 == 大 batch"，再用不等大分组 4+12 让 max|Δ| 跳 7 个数量级、加权救回——把"等大分片"这个前提从背诵变成了亲手撞过的墙。
2. **脚本 06 的两处"真踩坑"注释**：`4.380254 == 4.380254` 是"按层切开不改变数学"的最硬证据；"loss 汇总不能 /world，否则恰好减半、数字看起来合理实则错了"这种把自己的事故写进注释的诚实，比十页理论都管用。
3. **README"无多卡学习路径"表格 + 01 章 torchrun 报错 FAQ**：0 GPU 学生被当成一等公民（CPU 双进程同样演示分布式语义），六类报错"症状→原因→解法"能真省一晚上——除了 04 那一处自己没兑现的承诺（K1）。
