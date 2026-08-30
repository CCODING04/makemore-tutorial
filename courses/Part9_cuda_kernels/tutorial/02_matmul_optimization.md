# 02 — matmul 优化阶梯：从 naive 到 cuBLAS

> 🧭 深度学习 90% 以上的浮点运算花在 matmul 上。本章把"最朴素能跑"的 GPU matmul
> 当起点，走一遍业界标准的优化阶梯，每一级用实测 GFLOPS 验证收益，
> 并回答一个面试高频问题：**为什么 GPU 快、快在哪、瓶颈到底是什么**。

## 📖 前置知识

- **01 章**：grid/block/thread 层级、全局索引公式、CUDA 五步曲、2D 启动
- **Part 1-6**：`x @ W` 在训练里出现的频率（每个 Linear 层、每个 attention 的 q/k/v 投影）
- **Part 7 03 章**：提过 Flash Attention 是"内核级优化"——本章给出它的直觉

优化阶梯的出处：Simon Boehm 的博客
[*How to Optimize a CUDA Matmul Kernel to cuBLAS in 1 Hour*](https://siboehm.com/articles/22/CUDA-MMM)
（cuda-course 07 课的主线材料），我们沿用它的 kernel 1→5 命名。

## 先跑基准：naive matmul（[scripts/03_naive_matmul.cu](../scripts/03_naive_matmul.cu)）

每个 GPU 线程负责输出矩阵 C 的**一个元素**：先算出自己的 (row, col)，再沿 K 做点积。

```cuda
__global__ void matmul_gpu_naive(const float *A, const float *B, float *C,
                                 int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
            sum += A[row * k + l] * B[l * n + col];   // 行主序：A[i][k]=A[i*k+k]...
        C[row * n + col] = sum;
    }
}
```

**实测（4090，512×512×512，fp32）**：

```
GPU naive : 0.058 ms  ->  4651.1 GFLOPS
CPU       : 52 ms   (one shot)
Arithmetic intensity: 85.33 FLOP/byte -> memory-bound! (see script 04)
```

CPU 要 52 秒毫秒级（52ms），GPU 0.058ms——快了 **900 倍**。但先别高兴：
这离这块卡的潜力差着 4-5 倍。上面最后一行是关键——

## 🔑 核心概念：算力墙 vs 内存墙（roofline 直觉）

每个内核都在两种瓶颈中占一种：

```
算力墙（compute-bound）：计算单元在满转，数据早就备好
   → 提升 = 更多 FLOPS（Tensor Core、更精密的指令）

内存墙（memory-bound）：计算单元大部分时间在【等数据】从内存赶来
   → 提升 = 更少的内存读写（tiling、融合、缓存）
```

判断方法叫**算术强度**（arithmetic intensity）：

```
算术强度 = 总 FLOPs / 总内存字节数
naive matmul:  2MNK FLOPs，但要读 2MNK 次数据（每个输出沿 K 读 A 行+B 列）
             ≈ 1 FLOP : 1 次全局读 → 强度太低 → 内存墙
```

- 🔑 **LLM 推理（生成阶段）几乎总是 memory-bound**：每生成一个 token 要把全部权重读一遍，
  计算只占一小部分时间。这就是为什么 Part 7 的 KV Cache 有效（避免重复算）、
  为什么量化有效（读的每个数变小）、为什么 batching 能提升吞吐（一次权重读喂多个请求）。
- ⚠️ 面试里说"GPU 快是因为核多"只对了一半；**完整的说法是：核多 + 分级内存 +
  海量线程把访存延迟藏起来**。接下来三招全在围绕"内存"做文章。

## 优化前必修：合并访存（coalescing）

一个 warp（32 线程）同时发起内存请求时，如果它们的地址**连续**，硬件把它们合并成
**一次**内存事务（128 字节）；如果地址分散，就拆成 32 次独立事务——**带宽浪费 32 倍**。

naive 版里藏着两种访问模式（warp = 同一行的 32 个相邻 col 线程）：

```
A[row * k + l]   ：32 个线程读【同一个】地址 → 广播，免费 ✓
B[l * n + col]   ：col 相邻 → 地址相邻，合并成 1 次事务 ✓
```

我们脚本 03 的线程映射（x→col）**已经**是合并的。脚本 04 的 L1 故意把映射写坏
（x→row），让 B 的读取跨行跳跃：

```
L1 uncoalesced   553.6 GFLOPS   ← 坏映射，比 L2 慢 ~9 倍
L2 coalesced    4844.9 GFLOPS   ← 同样的计算量，只是"读的方式"对了
```

> 🔑 **同一个算法，仅仅是线程到数据的映射方式不同，差 9 倍**——这是本课最震撼的一组数字，
> 也是"GPU 编程 = 访存编程"的最好证据。

## 优化阶梯（[scripts/04_matmul_tiled.cu](../scripts/04_matmul_tiled.cu)）

阶梯全貌（4090 实测，512³ fp32）：

```
kernel               time(ms)       GFLOPS   优化的是哪堵墙
---------------------------------------------------------
L1 uncoalesced          0.485        553.6   （对照组：访存做错）
L2 coalesced            0.055       4844.9   合并访存
L3 smem tile            0.045       5905.4   shared memory 复用
L4 1D blocktile         0.039       6967.5   寄存器复用（每线程 8 输出）
L5 2D blocktile         0.031       8795.2   寄存器复用最大化（4x4 微tile）
cuBLAS（脚本 06）       0.012      22163.1   Tensor Core + autotune + 向量化
```

### L3：shared memory（SMEM）—— 每线程仍 1 个输出

观察浪费：输出 tile 的所有元素都依赖 A 的同一批行、B 的同一批列。
naive 让每个元素**各自**去全局内存读 → 同一份数据被重复读 N 次。

做法：每个 block 把需要的 A/B 小块（tile）**一次性搬进 SMEM**（每 SM 独享的高速
片上内存，~100KB 级，延迟接近寄存器），之后整个 K 循环都在 SMEM 里进行：

```cuda
__shared__ float As[BM][BK];          // block 内所有线程共享
__shared__ float Bs[BK][BN];
for (int t = 0; t < k; t += BK) {
    // 合作搬运：block 里每线程搬 1 个 A 元素 + 1 个 B 元素
    As[ty][tx] = A[aRow * k + aCol];
    Bs[ty][tx] = B[bRow * n + bCol];
    __syncthreads();                  // 等 tile 全部落位（block 内屏障！）
    for (int l = 0; l < BK; l++)
        sum += As[ty][l] * Bs[l][tx];
    __syncthreads();                  // 算完再搬下一块，防止有人提前改写
}
```

- 🔑 `__syncthreads()` 是 **block 内**的屏障：所有线程到齐才放行。它是 block 这个抽象
  存在的根本原因——**只有同 block 的线程能通信**（SMEM + barrier），跨 block 只能等内核结束。
- 💡 这就是 Part 7 提过的 **Flash Attention 的核心思想**：把 Q/K/V 分块（tile）搬进 SMEM，
  在片上算完一部分 attention 再换下一块，全局内存里从不出现巨大的注意力矩阵。
  FA 的额外难点是 softmax 需要"整行"信息，所以用了 online softmax 的递推技巧。

### L4：1D block tiling——寄存器接力

L3 之后 SMEM 读成了新瓶颈。解法：让每个线程一次算 **8 个输出**（同一列相邻 8 行），
结果全存**寄存器**（比 SMEM 还快一个量级）：

```
读 1 次 Bs[l][tx]（SMEM）→ 喂给 8 次乘加   # SMEM 读次数 ÷ 8
sums[0..7] 活在寄存器里，最后一次性写回
```

### L5：2D block tiling——把复用推满

每线程算 **4×4 = 16 个输出**的微 tile，行方向和列方向的 SMEM 读都被摊薄。
这一版的结构（块级 tile → 线程级微 tile → 寄存器累积）**就是 cuBLAS/CUTLASS 高性能
GEMM 的骨架**。

### 通向 cuBLAS 还差什么（原课程 07 课的延伸层）

| 手段 | 一句话 |
|---|---|
| 向量化访存 | `float4` 一次搬 128 bit，减少指令数（原课程 07 课 unrolling_example.cu） |
| `#pragma unroll` | 展开循环，让编译器排更多指令填等待空隙（可用 `nvcc -ptx` 验证是否生效） |
| Autotuning | BM/BN/BK/TM/TN 的最优组合随 GPU 架构变化，grid search 自动选（siboehm 用脚本搜参数） |
| Double buffering | 搬下一块 tile 与算当前 tile 重叠（异步拷贝） |
| Tensor Core | 专用矩阵乘加单元（fp16/bf16/tf32 输入），一次算 4×4 矩阵块——cuBLAS 快的真正大头 |

> 💡 **occupancy**（占用率）这个词在原课程 07 课出现过：SM 上活跃 warp 数 ÷ 最大可容纳
> warp 数。寄存器太多、SMEM 太大的 kernel 会降低 occupancy——优化常常是三者间的权衡，
> `nvcc -Xptxas -v` 能看到每个内核的寄存器用量。

## 和我们课程的关系（面试向）

| Part 9 概念 | 前面课程出现的位置 | 面试问法 |
|---|---|---|
| memory-bound | Part 7 KV Cache、Part 8 推理加速 | "LLM 推理为什么是 memory-bound？怎么优化？" |
| tiling / SMEM | Part 7 Flash Attention 提及 | "Flash Attention 为什么快？省了什么？" |
| bf16 / Tensor Core | Part 8 autocast(bf16) | "混合精度训练为什么几乎不掉点还更快？" |
| GFLOPS / 算术强度 | 本课新引入 | "给你一个 kernel，你怎么判断它该往哪个方向优化？" |

## 学完本部分你能...

- ✅ 写出 naive matmul 内核，并算出它的算术强度、判断瓶颈类型
- ✅ 解释合并访存为什么值 9 倍速度，检查一个内核的 warp 访问模式
- ✅ 说出优化阶梯每一级"省的是什么"：coalesced→事务数、SMEM→重复读、寄存器→SMEM 读
- ✅ 把 Flash Attention / KV Cache / bf16 的加速原理用"内存墙"语言重新讲一遍
- ✅ 用 GFLOPS 对比手写内核与 cuBLAS，说出差距的构成

**课后练习**

<details>
<summary>Q1: 512³ 的矩阵只有 1MB×3，全塞得进 4090 的 72MB L2 cache——我们的实测阶梯
里 L2→L5 的提升因此偏小。换成 4096³（192MB×3）重测，哪几级提升会变大？</summary>
A: 全部以"减少全局读"为卖点的级（L3/L4/L5）差距会拉大：小矩阵时 L2 把 naive 的重复读
都缓存住了，tiling 的优势被掩盖；大矩阵重复读全打到 HBM，SMEM/寄存器复用才是硬收益。
教训：**benchmark 要用目标规模测**（原课程反复强调 verify + realistic size）。
</details>

<details>
<summary>Q2: 为什么 block tile 取 64×64、BK 取 8 这类数？取 96×96 行不行？</summary>
A: 约束有一堆：每 block 线程数 ≤1024；SMEM 用量 BM*BK + BK*BN 个 float 要 ≤ 每 SM 上限；
寄存器用量（每线程 TM*TN 个累加器）不能把 occupancy 压死；BK 取 8/16/32 与共享内存
bank 数（32）和 warp 大小对齐友好。96×96 未必不行——**没有普适最优解，所以工业界用
autotuning 按显卡实测搜参数**（这正是 siboehm 的 autotune 一章做的事）。
</details>

<details>
<summary>Q3: attention 的 Q@K^T 是 (T,d)@(d,T)——它在 LLM 里为什么也是"大 matmul"？
seq=4096、d=128、heads=32 时 FLOPs 是多少？</summary>
A: 每个 head 一次 Q@K^T 是 T×T×d 的 matmul（2*T*T*d FLOPs），32 个 head 并行。
4096² ×128 ×2 ×32 ≈ 137 GFLOPs——一次 forward 仅这一项就上百 GFLOPs；
再加上 V 加权和（同量级）和 FFN（通常是 attention 的 2-4 倍），单层单次 forward
就是数百 GFLOPs。所以"内核效率"直接等于"训练账单"。
</details>

## 📝 课后作业

完成本章后，去 Assignment 9 完成题 3（tiling 访存账本）和题 4（GFLOPS 报告）：

👉 [Assignment 9](../../../assignments/assignment_9/)

## 下一步

内核会写了、会优化了，但还没回答：**怎么测量**（profiling）、**怎么协作**
（atomics / streams）、**什么时候不该自己写**（cuBLAS / cuDNN）。
下一章补齐这三块工程拼图。

👉 [03 — Profiling、Atomics、Streams 与 CUDA 库](03_profiling_and_cuda_apis.md)
