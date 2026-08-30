# 01 — GPU 架构与第一个 CUDA 内核

> 🧭 前八课里，`tensor @ tensor` 一直是个黑盒。从这章起我们打开它：GPU 到底是什么、
> 怎么把一句"c[i] = a[i] + b[i]"变成一百万个并行线程，以及为什么 GPU 能快 10 倍以上。

## 📖 前置知识

本章需要你已经掌握：

- **Part 6**：self-attention 里 `q @ k^T`、`softmax` 这些"算子"的概念（我们只会用到"算子"这个词，不需要推导）
- **Part 7 03 章**：Flash Attention、KV Cache 被提过是"内核级优化"——本章之后你就懂这个词
- **Part 8**：bf16 autocast——混精度为什么省显存/加速，学完 Part 9 会更具体
- **够用的 C**：下面会现场补，零基础也能跟上

> 💡 没有 GPU？本章的**概念**全部可以纸上学习；`scripts/` 里的 `.cu` 需要
> NVIDIA GPU + nvcc 编译，可以放到 [Google Colab](https://colab.research.google.com/)（免费 T4）
> 或 Kaggle（每周 30h P100）上跑。作业 9 的题 1-4 纯 CPU 可完成。

## 环境自检（对应原课程 02_Setup）

```bash
nvidia-smi                 # 能看到显卡 → 驱动 OK
nvcc -V                    # 能看到 release 11.x/12.x → 编译器 OK（没有的话见本课 README）
python -c "import torch; print(torch.cuda.is_available())"   # True → PyTorch 侧 OK
```

三者互相独立：`nvidia-smi` 有但 `nvcc` 没有 = 只装了驱动没装 CUDA Toolkit；
`nvcc` 有但 torch 说 False = torch 装成了 CPU 版。这三种坑都常见。

## 深度学习生态里的 CUDA（对应原课程 01 课）

你在 Part 1-8 写的每一行 PyTorch，实际执行的路径是：

```
你写的 Python:        out = x @ W + b
                          ↓  ATen（PyTorch 的 C++ 算子库）
库调用:               cublasSgemm(...) / conv kernel / ...
                          ↓  CUDA Runtime / 驱动
硬件:                 GPU 上的几万个线程同时做乘加
```

- 🔑 **CUDA** 是 NVIDIA 的并行计算平台：一套 C/C++ 扩展语法 + 编译器（nvcc）+ 运行时。
- **cuBLAS / cuDNN** 是 NVIDIA 的预写内核库（矩阵乘 / 卷积等），PyTorch 的大部分算子直接调它们。
- 所以"PyTorch 慢"几乎从来不是 Python 慢，而是**内核选择/访存模式**的问题——这就是为什么
  有时候手写一个融合内核能快 10 倍（Part 8 的 GRPO 采样循环就是典型受益者）。

原课程把 CUDA 课程的目标定为"为读懂 Karpathy 的 **llm.c** 打基础"——llm.c 就是用
纯 CUDA+C++ 复刻 GPT-2 训练，不依赖 PyTorch。学完本课你会具备读它的第一块拼图。

## CPU vs GPU：两种哲学（对应原课程 04 课）

| | CPU | GPU |
|---|---|---|
| 设计目标 | **低延迟**：尽快算完这一件事 | **高吞吐**：同时算完一大堆事 |
| 核心数 | 几个~几十个大核，主频高 | 几千~上万个"小核"（4090：16384 个 CUDA core） |
| 每核能力 | 分支预测、乱序执行、大缓存 | 极简，靠数量取胜 |
| 擅长 | 逻辑复杂、分支多的串行任务 | 大规模的**规则**并行计算 |
| 典型弱项 | 一万个数相加（逐个来太慢） | `if/else` 密集的串行逻辑 |

> 💡 GPU 不是"更快的 CPU"，而是完全不同的机器。把 CPU 比作 4 位教授，GPU 是 16384 个小学生：
> 教授解奥数题完胜；但让 16384 个小学生每人算一页乘法口诀，瞬间完成。
> 深度学习的本质 = 海量矩阵乘加（小学生型任务）→ 这就是 GPU 统治 DL 的原因。

⚠️ **数据要来回搬运**：CPU 内存（主机内存）和 GPU 显存是两个世界，中间隔一条 PCIe 总线。
数据搬运慢、计算快，所以"小任务上 GPU 反而更慢"——拷贝开销吃掉了收益。这是后面
streams 一节（03 章）要解决的问题。

## GPU 的执行层级（对应原课程 05 课）

CUDA 把并行组织成三层，**这是整门课最重要的一张图**：

```
Grid（网格）＝ 一次内核启动的全部线程
┌────────────────────────────────────────────┐
│  Block (0,0)    Block (1,0)    Block (2,0) │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │ t0 t1 t2│   │ t0 t1 t2│   │ t0 t1 t2│   │
│  │ t3 t4 t5│   │ t3 t4 t5│   │ t3 t4 t5│   │   ...
│  └─────────┘   └─────────┘   └─────────┘   │
│  Block (0,1)    Block (1,1)    ...         │
└────────────────────────────────────────────┘
   硬件映射：
   thread → CUDA core（最小编排单位）
   block  → SM（Streaming Multiprocessor，一个 SM 同时驻留多个 block）
   grid   → 整块 GPU
```

再加一个必须知道的概念——**warp**：

- 🔑 **warp = 32 个连续编号的线程组成的"小队"，它们锁步（lock-step）执行同一条指令**。
- SM 真正的调度单位是 warp，不是单个线程。`block_size` 必须取 32 的倍数（128/256/512 常见），
  否则最后一个 warp 有空座，纯浪费。
- ⚠️ 同一 warp 里线程走不同分支（如 `if (i % 2)`）时，两个分支要**串行各跑一遍**
  （warp divergence，分支发散），并行度减半。内核里少写依赖数据的分支。

## 够用的 C（对应原课程 03 课，10 分钟版）

| 概念 | 代码 | 一句话解释 |
|---|---|---|
| 指针 | `float *p` | p 存的是"地址"，`p[i]` 等价于"从 p 指向的位置数第 i 个" |
| 数组即指针 | `A[i * K + l]` | 二维数组按行摊平成一维后，`(i,l)` 元素在第 `i*K+l` 格（row-major） |
| 动态内存 | `malloc / free` | 主机内存版；GPU 显存用 `cudaMalloc / cudaFree` |
| 宏 | `#define N 1000000` | 编译前文本替换，定义常量 |
| 编译 | `nvcc a.cu -o a` | nvcc = gcc + CUDA 编译器：`.cu` 里的 CPU 代码交给 gcc，GPU 代码（kernel）编译成 PTX/SASS |

`nvcc` 的编译流水线（读 llm.c / PyTorch 编译报错时你会认得这些词）：
`.cu` →（nvcc 前端）→ **PTX**（虚拟汇编）→（ptxas）→ **SASS**（具体架构的机器码）。
`nvcc -ptx a.cu` 可以导出 PTX 人工阅读——原课程 07 课用它验证循环是否被展开。

## 第一个内核：vector add（跑 [scripts/01_vector_add.cu](../scripts/01_vector_add.cu)）

任务：`c[i] = a[i] + b[i]`，i 取 0..999999。CPU 版就是 for 循环；GPU 版分两半：

**① 内核（在 GPU 上跑的函数）**：

```cuda
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 我是谁？
    if (i < n) {                                     // 边界守卫
        c[i] = a[i] + b[i];
    }
}
```

- `__global__` 声明"从 CPU 启动、在 GPU 执行"的函数（另两个兄弟：`__device__` 只在 GPU 内调用、`__host__` 只在 CPU）。
- **每个线程执行同一份代码**（SIMT），靠"我是谁"算出各自处理的下标——这就是 CUDA 的核心思想。
- 🔑 全局索引公式 `i = blockIdx.x * blockDim.x + threadIdx.x`：
  把它读成"**第几班 × 每班人数 + 班内学号**"。比如 blockDim=256 时，block 3 的 17 号线程是 `i = 3*256+17 = 785`。
- `if (i < n)` 边界守卫：线程总数要向上取整到 block 的倍数（3906.25 → 3907 个 block），
  多出来的线程必须拦住，否则越界读写显存。

**② 主函数（CPU 侧的五步曲）**：

```cuda
// 1. 分配显存                      2. 拷输入到显存
cudaMalloc(&d_a, size);             cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
// 3. 启动内核：<<<block数, 每block线程数>>>
vector_add_gpu<<<3907, 256>>>(d_a, d_b, d_c, N);
// 4. 等 GPU 真正做完（启动是异步的！）5. 拷结果回来
cudaDeviceSynchronize();            cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
```

- ⚠️ **内核启动是异步的**：`<<<...>>>` 这一行瞬间返回，GPU 在后台慢慢算。
  不 `cudaDeviceSynchronize()` 就去读结果，读到的是没算完的垃圾。PyTorch 用户对应的坑：
  忘了 `torch.cuda.synchronize()` 就计时，测出的"耗时"永远是 0.001ms。
- `h_`/`d_` 前缀是社区惯例：host（主机）/device（设备）。

**实测输出（4090，N=100 万）**：

```
CPU avg: 0.230 ms
GPU avg: 0.019 ms (kernel only, no memcpy)
Speedup: 12.2x
Verification: CORRECT (0 errors)
```

> 💡 **验证方法论**（原课程反复强调）：先写 CPU 版当标准答案，GPU 算完逐元素对比。
> GPU 代码错了不会报错，只会安静地给出错答案——CPU 参照物是唯一可靠的防线。
> 这也是我们课程 Part 1 的老传统：计数版 bigram 对照神经网络版。

## 线程层级实操（跑 [scripts/02_thread_hierarchy.cu](../scripts/02_thread_hierarchy.cu)）

这个脚本让每个线程打印自己的坐标：

```
=== whoami: grid(2,2) x block(2,2), 16 threads, order NOT guaranteed ===
block(1,1,0) thread(0,0,0)
block(1,1,0) thread(1,0,0)
block(0,1,0) thread(0,0,0)
block(0,0,0) thread(1,1,0)
...（顺序每次都不一样！）
```

- 💡 **打印顺序乱**：GPU 不保证 block 的启动顺序——16 个线程真的在并行执行。
  这是学生第一次"亲眼见到"并行的地方。
- 2D/3D 启动：`dim3 grid(2,2), block(2,2)`，索引公式对 x/y/z 各算一遍。
- 同一份数据，用 1D 启动和 2D 启动算 square，结果必须一致（脚本验证了这一点）——
  **"1D/2D/3D"只是给线程编号的方式，数据本身永远是一维线性内存**。
  2D 的价值在 matmul：row/col 两个坐标天然对应输出矩阵的两个维度（下一章主角）。

## 学完本部分你能...

- ✅ 画出 grid/block/thread/warp 层级，说出它们到硬件（core/SM/GPU）的映射
- ✅ 手写全局索引公式（1D/2D），解释为什么需要边界守卫
- ✅ 完整说出 CUDA 程序五步曲：分配显存 → 拷入 → 启动内核 → 同步 → 拷回
- ✅ 解释"为什么 GPU 快"和"什么时候 GPU 反而慢"（搬运开销、分支密集）
- ✅ 用 CPU 参照实现验证 GPU 内核的正确性

**课后练习**

<details>
<summary>Q1: 为什么 blockDim 一般取 128/256/512，而不是 100？</summary>
A: warp = 32 线程锁步执行。SM 按 warp 分配调度槽位，100 = 3 个完整 warp + 4 个空座；
这 4 个线程所在的 warp 只用了一半多一点的 lanes，浪费硬件资源。取 32 的倍数让每个 warp 都满员。
</details>

<details>
<summary>Q2: 把 vector add 的 `if (i < n)` 删掉，N 恰好等于线程总数时，程序对吗？值得删吗？</summary>
A: 这次恰好对（没有多余线程），但这是脆弱的巧合——N 一变就崩。守卫只花一次整数比较，
相比内存读写几乎免费，永远保留。安全 > 一点点微小的性能。
</details>

<details>
<summary>Q3: 一个 block 最多 1024 个线程。想用 4096 个线程怎么办？为什么要限制？</summary>
A: 拆成 4 个 block（4096/1024），block 之间由 GPU 调度到各个 SM 并行执行。
限制 block 大小是因为一个 block 的所有线程必须驻留在**同一个 SM** 上（共享同一块 SMEM、
共用 barrier 同步），SM 的寄存器/SMEM 数量物理上装不下太多线程。
</details>

## 📝 课后作业

完成本章后，去 Assignment 9 完成题 1（索引数学）和题 2（行/列主序）：

👉 [Assignment 9](../../../assignments/assignment_9/)

## 下一步

会写"最简单的内核"了。但深度学习的心脏是 **matmul**——下一章我们写它，
测出第一份 GFLOPS 报告，然后一梯一级把它从 500 GFLOPS 优化到 8000+，
并搞清楚每一级优化到底在省什么。

👉 [02 — matmul 优化阶梯：从 naive 到 cuBLAS](02_matmul_optimization.md)
