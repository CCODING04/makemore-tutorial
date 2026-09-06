# S2 学习报告 · P09（CUDA 内核编程）

> 审计人：学生 agent S2（实操薄弱型）。审计方式：全文读教程 6 个 .md → 逐段对照 9 个脚本 → 在副本里从零编译+运行 → 以学生身份独立做作业（先做后看，全程未读 assignment_reference）。
> 环境：RTX 4090 ×2 / driver 550.120 / torch 2.6.0+cu124 / triton 3.2.0 / 系统 gcc 12.3 / 机器有 cuda-11.8 + cuda-12.4 共存（`/usr/local/cuda` 软链指向 11.8）。

## 总分：9.2 / 10

一句话：这是全套课程里"教程-代码-实测数字"三方对齐度最高的一部分——README 承诺的每一条（make 一次过、多版本 CUDA 自动选择、GFLOPS 阶梯、FA 验收线）都在我机器上复现了；扣掉的 0.8 分主要在"仓库里带了预编译 bin/ 导致 make 空转"和"题 5 在 pytest 下显示 FAILED 而非 SKIP"这类新手会撞的毛边。

## 卡点清单（按遇到顺序）

| # | 卡点/发现 | 严重度 | 详情 |
|---|---|---|---|
| 1 | **G16 复制 scripts 连带 bin/，`make` 直接空转** | 中 | `cp -r scripts p9scripts` 把预编译 `bin/`、`__pycache__/` 一起复制过来，首次 `make` 输出"对 'all' 无需做任何事"——我一度以为自己复制错了。后 `make clean && time make` 从零重编：3.2 秒全过，自动选中 `/usr/local/cuda-12.4`（与 README 声称的"选版本最高"一致），gcc 12.3 + CUDA 12.4 无需 `-ccbin`（与 Makefile 注释一致）。仓库分发编译产物对新手是双刃剑：能直接跑，但容易误判"我编译过了" |
| 2 | **pytest 下未实现的题 5 显示 FAILED 而非 ⏭️ SKIP** | 中 | 作业 md 写"题 5 未实现或无 GPU 会 ⏭️ 跳过（不影响其他题）"——独立运行 `python test_cuda_exercises.py` 确实如此；但按作业推荐的 `pytest test_cuda_exercises.py` 跑时，未实现的题 5 抛自定义 `_Skipped` 异常，pytest 不认识，显示 **1 failed, 4 passed**。我做题 1-4 后先跑了一次 pytest 亲眼看到红色 FAILED，作为初学者会以为自己弄坏了什么。无 GPU 同学（README 明确支持的群体）100% 会撞上 |
| 3 | **README 说脚本 09"约 2-4 分钟，大头是 autotune 编译"——实测 13.5 秒** | 低 | 共享 GPU（两块卡各有 5-6GB 被占）上 `time python 09_flash_attention_triton.py` 全程 13.5s，含 autotune。说法过于保守，新手会预留过量时间或以为程序卡住 |
| 4 | **脚本 08 运行时打印 TORCH_CUDA_ARCH_LIST UserWarning** | 低 | 教程 04 章"编译坑"清单没列这条。无害（编译成功），但实操型初学者会停下来排查 |
| 5 | whoami 打印顺序与教程示例形态不同 | 备注 | 教程已声明"顺序 NOT guaranteed"，本次实测按 block 分组倒序输出，与示例的交错形态不同——不算不一致，仅提示别拿输出逐字对 |

除此之外**零编译错误、零运行错误**：6 个 .cu + 3 个 .py 一次全过。

## 分章评分

| 章节 | 分 | 依据（实测证据） |
|---|---|---|
| README | 9.0 | 环境三步自检全部复现；"make && make run"路径真实可用；坑点清单（gcc 上限/CUDA_HOME 缓存/rpath）事后验证条条属实。扣分：bin/ 随库分发（卡点 1）、09 耗时过时（卡点 3） |
| 01 GPU 与第一个内核 | 9.0 | 内核/五步曲代码块与 01 号脚本一致；实测 CPU 0.183ms / GPU 0.007ms / 26.5x（教程 0.230/0.019/12.2x，已声明浮动）；"够用的 C"对零基础友好 |
| 02 matmul 优化阶梯 | 9.5 | naive 内核代码块与脚本逐字一致；实测阶梯 613.8→4868.1→5959.5→7084.5→8716.0 GFLOPS 与教程表逐级吻合（coalesced ≈9 倍的震撼点复现）；算术强度 4 行输出与脚本逐字对应 |
| 03 Profiling/Atomics/Streams/cuBLAS | 9.5 | atomic 1.396ms 与教程**完全相同**；racy 版实测 3.05（教程 3.88，同为非确定）；"重叠反而更慢"教训复现（0.882 vs 1.020ms）；cuBLAS OP_N/OP_N 代码块与脚本 06 逐字一致，实测 20953.7 GFLOPS |
| 04 Triton 与 PyTorch 扩展 | 9.0 | 07/08 直接跑通；扩展实测 0.0042 vs 0.0111ms（教程 0.0042/0.0110）；backward=(2x+1) 验证过；`scalar_type()` 坑位注释准确。扣分：ARCH_LIST warning 未列入坑清单（卡点 4） |
| 05 Flash Attention 毕业内核 | 9.5 | 09 一次跑通；数值验收 4 组 PASS 且 max\|Δ\| 与教程表**逐字节相同**（固定种子）；泄漏检查 OK；本次性能 89.4%~127.1%（教程 94.6%~127.1%，共享 GPU 浮动，最慢场景仍过"优秀"线）；5 行核心/autotune 网格/diag_lo 对齐/-1.0e6 全部与脚本逐字核对一致 |

## 一致性核对表（教程代码块 ↔ 脚本逐段对照）

| 教程位置 | 对照脚本 | 结论 |
|---|---|---|
| 01 章 vector_add 内核 + 五步曲（2 个代码块） | 01_vector_add.cu | ✅ 一致（教程为节选；公式/3907 blocks/边界守卫全对应） |
| 01 章 whoami 输出示例 | 02_thread_hierarchy.cu | ✅ 一致（顺序不保证，教程已声明） |
| 02 章 naive matmul 内核代码块 | 03_naive_matmul.cu | ✅ 逐字一致（含行主序注释） |
| 02 章 L3 SMEM 代码块 | 04_matmul_tiled.cu | ✅ 教学示意版（省略边界处理），语义一致；L1-L5 内核名与运行输出吻合 |
| 02 章 阶梯 GFLOPS 表 | 04 运行输出 | ✅ 五级全部复现，量级/排序一致 |
| 03 章 sum_naive_wrong / atomicAdd 代码块 | 05_atomics_streams.cu | ✅ 一致；实测数字吻合（atomic 1.396ms 完全相同） |
| 03 章 cuBLAS OP_N/OP_N 调用代码块 | 06_cublas_sgemm.cu L76-81 | ✅ 逐字一致（含 d_B, N / d_A, K 换序） |
| 04 章 Triton add_kernel 代码块 | 07_triton_kernels.py | ✅ 语义一致；教程把 y 的 load 内联进 store（紧凑节选），脚本分写 |
| 04 章 softmax 内核代码块 | 07（softmax_kernel） | ✅ 节选一致（减最大值/other=-inf/mask） |
| 04 章 polynomial C++ 代码块 | 08_pytorch_extension.py CUDA_SRC | ✅ 一致（模板/__restrict__/AT_DISPATCH scalar_type()） |
| 05 章 online softmax"5 行核心" | 09 L148-150 附近 | ✅ 逐字一致 |
| 05 章 autotune 16 组合网格 | 09 L160-170 | ✅ 一致（BM/BN∈{64,128} × w∈{4,8} × s∈{2,3}） |
| 05 章 diag_lo 按 BLOCK_N 对齐 / -1.0e6 / stage 编码 | 09 L210/142-145/242 | ✅ 逐字一致 |
| 05 章 数值验收表、泄漏检查 | 09 段 3 输出 | ✅ 逐字节相同（固定种子，确定性输出） |
| README FA 实测表 | 09 段 4 输出 | ✅ 趋势/量级一致（共享 GPU 浮动，教程已两处声明"看趋势"） |

未发现任何"教程说 A、代码做 B"级别的实质不一致。教程中的实测数字块全部带有"数字随硬件浮动，看趋势"的免责声明，且经复测属实。

## 作业元数据 + pytest 输出

做法：先独立实现（只参考作业 md 与 docstring 内提示，未看任何参考答案），再跑测试。工作目录 `/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_9/`，仅编辑 `cuda_exercises.py`。

### 逐题记录

| 题 | 类型 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|---|---|---|---|---|
| 1 全局线程索引（30 分） | CPU（纯 Python） | ✅ | 0（docstring 公式足够） | ~3 min | PASS |
| 2 行/列主序 + CPU matmul（30 分） | CPU | ✅ | 0 | ~2 min | PASS |
| 3 tiling 账本（20 分） | CPU（纸上推导） | ✅ | 0（公式在注释里，T²/T=T 化简一眼出） | ~2 min | PASS |
| 4 GFLOPS 报告（10 分） | CPU | ✅ | 0 | ~2 min | PASS |
| 5 Triton softmax（10 分） | GPU | ✅ | 1（作业"最常见的坑"折叠框——内核必须模块顶层；我据此把 @triton.jit 放顶层并用 try 包住 import triton） | ~4 min | PASS |

卡点实录：题 5 有一个真实的小决策——模块顶层的 `import triton` 在无 GPU 机器上可能失败，所以用 `try/except` 包住并让 `triton_softmax` 在无 triton/GPU 时返回 None（与测试约定吻合）。其余题无任何卡壳。

### pytest 输出（最终，5/5）

```
$ /home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python -m pytest test_cuda_exercises.py -v
test_cuda_exercises.py::test_exercise_1_global_index PASSED              [ 20%]
test_cuda_exercises.py::test_exercise_2_indexing_and_matmul PASSED       [ 40%]
test_cuda_exercises.py::test_exercise_3_tiling_reads PASSED              [ 60%]
test_cuda_exercises.py::test_exercise_4_gflops PASSED                    [ 80%]
test_cuda_exercises.py::test_exercise_5_triton_softmax PASSED            [100%]
============================== 5 passed in 0.97s ===============================
```

独立运行方式：`通过: 5/5, 失败: 0, 跳过: 0 🎉 全部通过！`

（做题中途的状态：题 1-4 完成时 pytest 显示 **4 passed, 1 failed**——题 5 未实现被 pytest 判为 FAILED 而非 SKIP，见卡点 2。）

### 编译/运行实测汇总（G16 副本 `/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P9/p9scripts/`）

- `make clean && make`：3.2s，自动选 `/usr/local/cuda-12.4/bin/nvcc`，rpath 焊入 lib64，无警告无错误
- `make run`：01（26.5x）/02（1D+2D square CORRECT）/03（4675.1 GFLOPS）/04（L1→L5 全 OK）/05（racy WRONG ✓、tree+atomic 0.012ms、重叠更慢 ✓）/06（20953.7 GFLOPS）全部符合教程描述
- `07`：vecadd/softmax 对 torch 全对；`08`：扩展编译 OK（CUDA_HOME 自动 12.4）、数值/梯度对；`09`：13.5s 全流程，验收"优秀"线达标

## 只改 3 件事

1. **修题 5 的 pytest skip 语义**（`assignments/assignment_9/test_cuda_exercises.py`）：把自定义 `_Skipped` 换成 `pytest.skip(allow_module_level=False)` 或在无 GPU 分支用 `pytest.mark.skipif(not _HAS_TRITON, ...)`，让"未实现/无 GPU"在 pytest 下显示 SKIPPED 而非 FAILED——这正是 README 承诺"无 GPU 也能完成作业"的群体第一眼会看到的东西。
2. **别把 `bin/`、`__pycache__/` 放进 scripts/ 分发**（或 README"环境要求"节加一句：若 scripts 是从别处复制来的，先 `make clean` 再 `make`）。否则新手第一次 `make` 看到"无需做任何事"会困惑自己是否编译成功、改源码也不会触发重编。
3. **更新 README 中脚本 09 的耗时预期**：'约 2-4 分钟' → 实测参考"首次运行约 15-60 秒（autotune 每次进程内重跑但很快）"；顺手把 04 章编译坑清单补一行 `TORCH_CUDA_ARCH_LIST is not set` 警告属正常现象。

## 最喜欢 3 处

1. **03 章"重叠反而更慢"的诚实教训**：教程不惜展示自己的反直觉实测（双流 1.026ms > 串行 0.942ms）并给出原因分析——我复测同样形态（1.020 > 0.882ms）。"直觉必须经过 benchmark 检验"这门课的元方法论，比任何一个知识点都值钱。全篇"看趋势，别死记数字"的口径一以贯之，实操派友好。
2. **05 章陷阱节（症状→原因→解法）**：`-inf` 产生 NaN、bf16 存累加器、误差扎堆在 BLOCK_M 倍数附近的定位法（先 amax/argmax 再猜）、`sdpa_kernel` 写进被计时 lambda 的不公平——四条全是真实工程里会撞的坑，且给出了可照抄的排查代码。这是我唯一见过把"debug 方法论"写进教学内核章节的教程。
3. **Makefile 的多版本 CUDA 自动选择 + rpath + gcc-11 降级**：零配置 3.2 秒编译成功，且注释把每个 `ifeq` 背后的"为什么"（unsupported GNU version、ldconfig 找不到新版 lib64）讲透——它是教程"编译坑点"清单的 executable 版，README 声称的每条防御都真实生效。
