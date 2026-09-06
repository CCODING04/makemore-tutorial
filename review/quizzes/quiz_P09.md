# 测验 · Part 09（CUDA 内核编程：从 vector add 到 Flash Attention）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** 画出 grid / block / thread 三级执行层级到硬件的映射；warp 是什么？为什么 blockDim 一般取 128/256/512 而不是 100？

- **答案**：thread → CUDA core（最小编排单位）、block → SM（一个 SM 同时驻留多个 block，同 block 线程共享 SMEM 与 barrier）、grid → 整块 GPU（4090 有 16384 个 CUDA core、128 个 SM）。warp = 32 个连续编号线程组成的"小队"，锁步执行同一条指令，是 SM 真正的调度单位。blockDim=100 = 3 个完整 warp + 4 个空座，最后一个 warp 的 lanes 大半浪费；取 32 的倍数让每个 warp 满员。同一 warp 内线程走不同分支（warp divergence）时两条分支要串行各跑一遍，并行度减半。
- **锚点**：教程 01_gpu_and_first_kernel.md §"GPU 的执行层级（对应原课程 05 课）"、课后练习 Q1

**Q2（数字）** 合并访存（coalescing）为什么值约 9 倍速度？报出 matmul 优化阶梯（4090，512³ fp32）从坏映射到 cuBLAS 的 GFLOPS 数字，并说出与 cuBLAS 差距的构成。

- **答案**：一个 warp 的 32 个线程地址连续时，硬件合并成 **1 次** 128 字节内存事务；地址分散则拆成 32 次事务，带宽浪费 32 倍。同一算法只改线程→数据映射：L1 uncoalesced **553.6** GFLOPS → L2 coalesced **4844.9**（≈9×）。继续攀升：L3 SMEM tile 5905.4 → L4 1D blocktile 6967.5（每线程 8 输出，寄存器复用）→ L5 2D blocktile 8795.2（4×4 微 tile，手写尽头）；cuBLAS **22163.1**（约 2.5× 差距 = Tensor Core + autotune + 向量化 float4 + double buffering）。注意 512³ 全塞进 72MB L2 会低估 tiling 收益，benchmark 要用目标规模测。
- **锚点**：教程 02_matmul_optimization.md §"优化前必修：合并访存（coalescing）"、§"优化阶梯"

**Q3（诊断）** ncu 的 SOL 表显示 naive matmul：Memory Throughput 92%、L1/TEX Cache Throughput 94%、DRAM Throughput 3%、Compute (SM) 92%。这个内核卡在哪堵墙？下一步该优化什么？nsys 时间线又给了什么额外提醒？

- **答案**：卡在**内存墙**，且绑的是 **L1/TEX 缓存管线**而非 DRAM——naive 的 2MNK 次重复读全砸在缓存管线上；DRAM 只有 3% 是因为 512³ 三块矩阵（1MB×3）全装在 4090 的 72MB L2 里。方向是减少重复的全局读（SMEM tiling → 寄存器复用），不是加算力/上 Tensor Core。nsys 的提醒：6 次内核每次约 54µs，而 H2D 2×50µs + D2H 58µs ≈ 158µs——**数据搬运是内核的 3 倍**，脚本只计内核时间会误导（"小任务上 GPU 反而慢"的定量版）。纪律：先测量、再优化；L5 内核 ncu 提示 grid 只有 0.1 waves，先扩大规模再谈优化。
- **锚点**：教程 03_profiling_and_cuda_apis.md §"实测 ncu：SOL 表直接指认瓶颈"、§"实测 nsys：数据搬运比内核还贵"

**Q4（对比）** 用"谁操心线程"说清 Triton 与手写 CUDA 的分工与性能边界；自己写的 CUDA 内核接进 PyTorch 需要哪三个关键件，为什么还要 `torch.autograd.Function`？

- **答案**：CUDA = scalar program + blocked threads（你逐线程写标量代码、自己组织 block、自己管 tiling/mask/SMEM）；Triton = blocked program + scalar threads（你按"一块数据"写向量视角代码，编译器替你铺线程、做 mask 与向量化访存）。代价：softmax 上百行 vs ~20 行；性能上限 CUDA 最高（最后 10-20%：warp 级 mma/shuffle、寄存器与 SMEM 精确布局、double buffering 是 Triton 放弃的控制权），elementwise/reduction 类几乎打平——生态分工是 GEMM 给 cuBLAS/CUTLASS、长尾融合内核给 Triton。扩展三件套：`AT_DISPATCH_FLOATING_TYPES`（按 dtype 实例化模板）、`__restrict__`（承诺指针不重叠、启用激进优化）、pybind 绑定（C++ 函数暴露为 Python）。`load_inline()` 编出的是裸函数、无 autograd，必须用 `torch.autograd.Function` 包一层手写 backward（本例 `2x+1`）。
- **锚点**：教程 04_triton_and_extensions.md §"设计哲学：CUDA vs Triton"、§"PyTorch CUDA 扩展：三个关键件"、课后练习 Q1

**Q5（数字）** Flash Attention 实测：T=4096 causal 下 naive 与手写 Triton FA 各多少 ms？验收线是什么、实测落在哪？FA 省了什么、没省什么？online softmax 维护哪三个状态？

- **答案**：naive 6.198 ms（物化 3 份 (B,H,T,T) 矩阵）→ 手写 Triton FA **0.279 ms**（22.2×，123 TFLOPS）；SDPA 最优后端（cudnn）0.308 ms → 手写版 ≈111%；full 场景 3.579 → 0.436 ms（8.2×）。验收线：≥ SDPA 最优后端 50% 合格、>85% 优秀，实测 95-127% 全部"优秀"（同卡同负载才可比，独占/共享是公平性变量）。FA **不省 FLOPs**（精确算法，QK^T 与 PV 乘加不变），省的是 **HBM 读写**：从不物化 (T,T) 矩阵，显存 $O(T^2) \to O(T)$。online softmax 三状态：行最大值 m、分母滚动和 l、输出累加 acc；新块到来先更新 m，再用 $\alpha = \exp_2(m_{old} - m_{new})$ 把历史 l/acc 折算到新基准后累加（换底系数折进 qk_scale 用 exp2 提速）。causal 三阶段里整块被 mask 的段直接跳过，是 causal 比 full 快 ~1.6× 的来源。
- **锚点**：教程 05_flash_attention.md §"一、为什么需要 Flash Attention"、§"二、online softmax：逐步推导"、§"3.3 causal 三阶段分解"；README §"实测参考"

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 画出 GPU 执行模型（SM / 线程层级 / 存储层次），解释内核如何跑起来 | Q1 | CUDA 五步曲（分配→拷入→启动→同步→拷回）与"启动异步、不同步读到垃圾"见闪卡 |
| 攀登手写 matmul 优化阶梯（合并访存 → SMEM tiling → block tiling），用算力墙/内存墙定位瓶颈 | Q2 | 算术强度判据与 naive 0.25 FLOP/byte 见闪卡 |
| 使用 nsys / ncu 做 profiling，用测量数据决定下一步 | Q3 | 含 atomics 树形归约 77× 与 streams"重叠反而更慢"教训（见闪卡） |
| 编写 Triton 内核（融合算子），说清与手写 CUDA 的取舍 | Q4 | 含 `@triton.jit` 必须模块顶层定义的坑（见闪卡） |
| 封装自定义算子为 PyTorch 扩展，接入训练代码 | Q4 | 三件套 + autograd.Function；编译坑（gcc 版本、CUDA_HOME 缓存）见闪卡 |
| 手写 Flash Attention 前向（online softmax + causal），对照 SDPA 四后端验收 | Q5 | FA1→FA3→Flex→Sage 演进与"4090 无 WGMMA 用不了 FA3"见闪卡 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| 全局索引公式与边界守卫？ | `i = blockIdx.x * blockDim.x + threadIdx.x`（"第几班 × 每班人数 + 班内学号"）；线程数向上取整到 block 倍数，多出的必须 `if (i < n)` 拦住 |
| CUDA 程序五步曲？ | cudaMalloc 分配 → cudaMemcpy 拷入 → `<<<blocks, threads>>>` 启动 → cudaDeviceSynchronize 同步 → cudaMemcpy 拷回；启动是异步的，忘同步就计时永远 0.001ms |
| vector add 实测加速？ | N=100 万：CPU 0.230 ms vs GPU 0.019 ms（仅内核）→ 12.2×；验证方法论：先写 CPU 参照版再逐元素对照 |
| 算术强度怎么判断瓶颈？ | 算术强度 = 总 FLOPs / 总内存字节；matmul 理想 85.33 FLOP/byte 高于 4090 屋顶线约 82（理论 compute-bound），naive 不复用数据实际约 0.25 → memory-bound |
| `__syncthreads()` 是什么级别的屏障？ | block 内屏障：所有线程到齐才放行；只有同 block 线程能通信（SMEM + barrier），跨 block 只能等内核结束 |
| L3→L4→L5 各省的是什么？ | L2 合并访存省事务数；L3 SMEM 省重复的全局读；L4 每线程 8 输出用寄存器省 SMEM 读；L5 4×4 微 tile 把复用推满——即 cuBLAS/CUTLASS 的骨架 |
| 100 万个数求和：atomicAdd vs 树形归约？ | 直接 `+=` 有竞争条件（输出 3.88 vs 正确 500006）；atomicAdd 正确但 1.396 ms；block 内 SMEM 树形归约 + block 间 atomicAdd 只 3907 次 → 0.018 ms，快 77× |
| streams 重叠实测教训？ | 4 块 1MB 双流重叠 1.026 ms 反而比串行 0.942 ms 慢：每块太小、async 调用开销大于收益；streams 真正赢在 GB 级数据或持续流式输入——直觉必须过 benchmark |
| cuBLAS 列主序怎么零成本适配？ | 行主序缓冲按列主序读 = 转置；要算 C=A@B 就让 cuBLAS 算 B^T@A^T（指针换序），写回即行主序 C；参数写错不报错只给错答案，靠 CPU 参照验证 |
| Triton 内核的 mask 与嵌套坑？ | `mask = offsets < n_elements` 一次判断整块，`tl.load/store` 带 mask 自动挡越界；`@triton.jit` 函数必须定义在模块顶层，否则 `NameError: tl is not defined` |
| Triton softmax 为什么快？ | 一个 program 管一行：整行载入片上、减最大值、exp、求和、除法一遍完成——内核融合，省掉 eager 里 4 个内核的 4 次显存往返 |
| 自定义 activation 扩展实测？ | `x²+x+1`：custom CUDA 0.0042 ms vs torch eager 0.0110 ms（2.6×，eager 拆 3 个内核 + 中间张量）；工程顺序：先 torch.compile，不够再手写 |
| PyTorch 扩展常见编译坑？ | CUDA 11.x 最高 gcc-11（传 `-ccbin g++-11`）；只改 `os.environ["CUDA_HOME"]` 无效，需同时覆盖 `torch.utils.cpp_extension.CUDA_HOME`；JIT 需 ninja；缓存在 `~/.cache/torch_extensions/` |
| online softmax 两块合并的 α 从哪来？ | $l_{new} = l_1 \cdot \alpha + l_2$，$\alpha = \exp(m_1 - m_{new})$——分母按新基准重缩放而非重算；首次迭代 m=−inf 使 α=0 自动清零历史 |
| FA 内核里哪些是 bf16、哪些是 fp32？ | 进 `tl.dot` 的 qk、p、v 是 bf16（Tensor Core 要求低精度输入）；状态 m/l/acc 全程 fp32 累加——"bf16 autocast 几乎不掉点"的硬件根基 |
| causal 三阶段分解？ | 阶段 1 带外块无 mask 快速路径；阶段 2 对角块逐元素 mask；阶段 3 整块被 mask 直接跳过——causal 0.279 vs full 0.436 ms（~1.6×，非 2× 因固定成本与对角块不省） |
| 为什么 4090 用不了 FA3？ | FA3 靠 Hopper 专属 WGMMA（SM90）+ TMA 异步 + FP8；4090 是 SM89 只有 mma 路径，编译都过不了；天花板是 FA2 类实现（SDPA flash/cudnn）+ SageAttention 量化路线（INT8 QK^T、INT4 PV，~3× FA2） |
| FlexAttention 是什么？ | 用 Python 函数写 mask_mod/score_mod，`create_block_mask` 编译成块级 Bitmap、全 0 块不进内核，torch.compile 生成 Triton；官方约 FA2 前向 90%，本机 causal T=4096 为 flash 后端的 94% 耗时 |
