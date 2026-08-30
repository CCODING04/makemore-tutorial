# 03 — Profiling、Atomics、Streams 与 CUDA 库

> 🧭 会写、会优化之后，还差三块工程拼图：**怎么测量**内核时间花在哪（profiling）、
> **线程之间怎么协作**（atomics / 归约）、**任务之间怎么重叠**（streams），
> 以及**什么时候不该自己写**——直接调 cuBLAS / cuDNN。

## 📖 前置知识

- **02 章**：内存墙/算力墙、warp、SMEM、`__syncthreads()`
- **Part 3**：诊断工具的思路（先测量、再下结论）——本章就是 GPU 版的"给内核做体检"

## Profiling：先测量，再优化（对应原课程 05 课 03 节）

手搓计时（cudaEvent / `clock_gettime`）只能给你总时间。想知道**时间花在哪**，
用 NVIDIA 的两件工具：

| 工具 | 干什么 | 一句话用法 |
|---|---|---|
| **Nsight Systems (nsys)** | 时间线全景：内核、memcpy、CPU-GPU 交替 | `nsys python train.py` → 看"谁在等谁" |
| **Nsight Compute (ncu)** | 单个内核的深入剖析：SM 占用率、内存吞吐、瓶颈判定 | `ncu --set full ./bin/04_matmul_tiled` → 看 SOL（Speed Of Light）表 |

原课程 03 Profiling 课还演示了 **NVTX**：在代码里插 `nvtxRangePush("forward")` /
`nvtxRangePop()`，时间线上就有彩色的阶段标记，一眼分清 forward/backward/数据搬运。
（PyTorch 的 `torch.profiler` 底层就是这套设施 + CUPTI。）

> 🔑 **profiling 的纪律**（呼应 Part 3 的"诊断→治疗"）：优化前先看数据。
> ncu 的 SOL 表直接告诉你这个内核卡在 Compute 还是 Memory——对应 02 章的两种墙，
> 别凭感觉优化。

## Atomics：一百万个线程抢一个数（[scripts/05_atomics_streams.cu](../scripts/05_atomics_streams.cu) Part A）

任务：GPU 上对 100 万个数求和。第一直觉的写法是错的：

```cuda
__global__ void sum_naive_wrong(float *x, float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[0] += x[i];        // ❌ 读-改-写三步会被别的线程插队 → 竞争条件
}
```

实测它每次输出都不一样（我们跑出过 3.88，正确答案是 500006）。修法是原子操作：

```cuda
atomicAdd(result, x[i]);      // ✅ 读-改-写不可分割；冲突线程被硬件串行化
```

正确了，但 **100 万次原子写互相排队，实测 1.396ms**——比整个 naive matmul 还慢。
标准解法是**分层归约**：

```
第一层（block 内）：shared memory 树形归约
   256 个数两两相加 → 128 → 64 → ... → 1，共 log2(256)=8 步
   （相邻线程加相邻线程，无 bank 冲突；每步 __syncthreads()）
第二层（block 间）：每个 block 的最终和 atomicAdd 到全局
   原子操作次数从 1,000,000 降到 3,907（block 数）
```

```
atomic         : 500004.22   (1.396 ms)
tree + atomic  : 500006.62   (0.018 ms)  <- 正确且快 77 倍
```

- 🔑 这个"**先局部、再全局**"的模式是 GPU 上一切归约（sum/max/norm）的原型。
  PyTorch 里 `tensor.sum()` 背后就是这样的多层树形归约内核。
- 💡 排队/竞争这个词在多线程 CPU 编程里也见过（`+=` 不是原子的）——概念完全同构，
  只是 GPU 的"线程数"大了四个数量级，问题被放大到肉眼可见。

## Streams：把"搬运"和"计算"重叠（[scripts/05_atomics_streams.cu](../scripts/05_atomics_streams.cu) Part B）

默认情况下，你的所有操作排在一一条**默认流**里**串行**执行：
拷贝 → 算 → 拷回 → 拷下一块 → ……（PCIe 搬运时 GPU 计算单元闲着）。

**Stream = 一条独立队列**。两条流里的任务可以并行：

```
串行（默认流）:  [copy1][calc1][back1][copy2][calc2][back2] ...
两条流交错    :  stream0: [copy1][calc1]      [back1]
                stream1:        [copy2][calc2][back2]  ← copy2 和 calc1 重叠
```

关键 API 只有三个：`cudaStreamCreate(&s)`、启动时把流当第四个参数
`kernel<<<grid, block, 0, s>>>(...)`、`cudaMemcpyAsync(..., s)`（异步版拷贝）。

> ⚠️ 演示简化说明：脚本里 4 个 chunk 共用同一块 `d_x` 缓冲——只测时延形态、
> 不校验数值；正式实现应每 chunk 独立缓冲。

⚠️ **诚实的实测教训**：我们的脚本里 4 块 1MB 数据双流重叠后是 1.026ms，
反而比串行 0.942ms 慢一点。为什么？① 每块太小，拷贝只有微秒级，
内核更是纳秒级——重叠收益小于 async 调用本身的额外开销；② 现代 GPU 的拷贝引擎
和小内核原本就能部分重叠。**streams 真正赢的场景**：单块数据量 GB 级、
或"持续输入流式处理"。这条教训本身就是本课要教的东西：**直觉必须经过 benchmark 检验**
（原课程方法论：always verify, always benchmark）。

## cuBLAS：把 matmul 交给库（[scripts/06_cublas_sgemm.cu](../scripts/06_cublas_sgemm.cu)）

手写阶梯的尽头是 8795 GFLOPS（L5），cuBLAS 同一块卡上是 22163 GFLOPS——
还有 2.5 倍差距（Tensor Core、autotune、向量化、double buffering）。
**工程上永远直接调库**，手写 matmul 是为了"知道库在做什么、什么时候库不是最优"。

cuBLAS 最著名的坑：**列主序（column-major）**。它是 Fortran 血统，矩阵默认"一列挨一列"存；
而 C/PyTorch 是行主序。我们的脚本用了一个漂亮技巧零成本适配：

```
🔑 关键恒等式：一块"行主序存储"的缓冲区，按列主序去读，读到的正好是它的转置。

要算行主序 C = A @ B：
  列主序视角下，三块缓冲区"读出来"分别是 A^T、B^T、C^T
  而 C^T = (A@B)^T = B^T @ A^T
  → 让 cuBLAS 算 "B^T @ A^T"（指针换序，不搬运任何数据），写回的内存就是行主序 C

cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, d_B, N,      // ← B 的位置，按列主序读 = B^T
            d_A, K,              // ← A 的位置，按列主序读 = A^T
            &beta, d_C, N);
```

实测我们第一次用 OP_T/OP_T 的参数组合，结果全错（max_err=37）——
**cuBLAS 的参数写错不报错，只会安静地给你错的答案**，CPU 参照验证再一次救场。

- 💡 cuBLAS 家族：`cuBLAS`（单卡常规 GEMM）、`cuBLASLt`（带启发式的进阶接口，
  可选融合 epilogue——把 bias/激活等附加计算拼进 GEMM 尾部）、`cuBLASXt`（多卡拆分矩阵）。PyTorch 的 `@` 最终落到它们。
- **cuDNN** 同理，是 DL 算子的全家桶（conv/RNN/attention/activation），
  原课程 06 课演示了 cuDNN 版 Tanh 和 Conv2d，并与 torch 结果对照。

## 关于"错误处理"的成人礼

脚本里我们略去了错误检查让代码短一点，但真实工程每个 CUDA 调用都该包一层：

```c
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); exit(1); \
    } } while (0)

CUDA_CHECK(cudaMalloc(&d_a, size));
```

⚠️ GPU 错误是**异步**的：出错的位置可能在很久之后的下一个调用才暴露。
`compute-sanitizer` 工具能抓越界访问——调内核 bug 的最后手段。

## 学完本部分你能...

- ✅ 说出 nsys 和 ncu 各解决什么问题，以及 SOL 表怎么指导优化方向
- ✅ 写出树形归约，解释"原子操作为什么对但慢"以及分层归约快 77 倍的原因
- ✅ 解释 stream 的重叠原理，并复述我们"重叠反而更慢"的实测教训
- ✅ 用行主序/列主序恒等式手动推出 cuBLAS 的参数写法
- ✅ 说明什么场景该用库、什么场景才值得手写内核

**课后练习**

<details>
<summary>Q1: 树形归约里如果 `if (threadIdx.x < s)` 写成 `if (threadIdx.x % 2 == 0)`，
除了慢还有什么问题？</summary>
A: 交错配对（thread 0 加 thread 1，thread 2 加 thread 3...）依赖固定的 bank 布局，
在 SMEM 上会产生大量 bank conflict，且这种 stride 访问模式和 warp 调度不友好。
相邻配对（x[s] += x[s+idx]）是标准写法。更细的优化还有 warp shuffle
（`__shfl_down_sync`，warp 内不经过 SMEM 直接交换寄存器）——这是 CUB 库干的事。
</details>

<details>
<summary>Q2: PyTorch 里 `x @ W` 你可以指定 `torch.cuda.Stream()` 让它和别的计算并行。
结合本节，说说什么情况下值得这么做？</summary>
A: 典型场景：① 梯度交换/数据预取与计算重叠（DDP 的通信流）；
② 权重加载/量化转换与推理流水线重叠；③ 多路互不依赖的 batch 并行。
不值得的场景：算子间有强依赖（A 的输出是 B 的输入）、内核大到已经吃满 GPU——
排队重叠毫无收益还添同步复杂度。
</details>

<details>
<summary>Q3: 为什么 cuBLAS 写错参数不报错？这对你 review 别人的 GPU 代码有什么启示？</summary>
A: cuBLAS 是"参数进、地址上读写"的底层库，它不知道你的"意图"（你要 A@B 还是 B@A），
任何参数组合在它看来都是合法请求。启示：GPU 代码必须配"黄金参照"（CPU 实现或
小规模已知答案）+ 形状/数值断言；盲信"能跑 = 对"是 GPU 编程第一大坑。
</details>

## 📝 课后作业

本章概念在 Assignment 9 的题 3/4 中延伸（访存账本与 GFLOPS 报告是 profiling 的纸面版）：

👉 [Assignment 9](../../../assignments/assignment_9/)

## 下一步

手写 CUDA 你已经走完全程。最后一步：看看**工业界怎么让"写内核"这件事变简单**——
Triton（Python 写内核）和 PyTorch 自定义扩展（把自己的内核接进 PyTorch），
并给整个课程画一张"继续往哪走"的地图。

👉 [04 — Triton 与 PyTorch 扩展：通向 llm.c](04_triton_and_extensions.md)
