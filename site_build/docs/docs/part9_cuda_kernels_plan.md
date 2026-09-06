# Part 9 设计方案：CUDA 内核编程 — 打开深度学习的引擎盖

> 参考：[infatoshi/cuda-course](https://github.com/infatoshi/cuda-course)（FreeCodeCamp CUDA Course，MIT）
> 硬件验证环境：RTX 4090（本机 admin02）+ torch 2.5.1+cu121 + triton 3.1.0（项目 venv）
> CUDA 工具链：多版本并存——`/usr/local/cuda-11.8`（`/usr/local/cuda` 默认软链）与
> `/usr/local/cuda-12.4`（后装，驱动 550 原生支持）。Makefile/扩展脚本自动选最高版本，
> 不覆盖已有 CUDA；CUDA 11.x 需 `-ccbin gcc-11` 规避 gcc 上限，12.4 原生支持 gcc-12

## 定位

Part 1-8 里 `tensor @ tensor` 一直是黑盒：Part 7 提到的 Flash Attention、KV Cache、Part 8 用的
bf16 autocast，都是"GPU 内核层面"的优化。Part 9 打开这个黑盒：**从第一个 CUDA 内核写到手写
matmul 优化阶梯、Triton 内核和 PyTorch CUDA 扩展**，并用速度实测解释"为什么 GPU 快、快在哪、
瓶颈是什么"。

```
Part 8 (后训练全流程：模型是"用"出来的)
    │
    │  "模型跑在 GPU 上，但 GPU 怎么算 matmul / softmax / attention？"
    ▼
Part 9: CUDA 内核（模型是"算"出来的）
    ① GPU 架构 + 第一个内核    — 线程层级 / vector add CPU vs GPU
    ② matmul 优化阶梯          — naive → coalesced → shared memory → block tiling
    ③ 进阶 CUDA + 库           — atomics / streams / profiling / cuBLAS / cuDNN
    ④ Triton + PyTorch 扩展    — 写得起、接得上、通向 llm.c
```

与原仓库的对应（完整覆盖其核心 lecture）：

| cuda-course lecture | Part 9 覆盖 |
|---|---|
| 01 Deep Learning Ecosystem | 教程 01 开篇（为什么速度=钱） |
| 02 Setup | 教程 README「环境自检」+ 教程 01 附录 |
| 03 C/C++ Review | 教程 01「够用的 C」一节（指针/数组/宏/编译） |
| 04 Gentle Intro to GPUs | 教程 01（CPU vs GPU、SM、warp） |
| 05 Writing your First Kernels | 脚本 01/02/03/05 + 教程 01/03 |
| 06 CUDA APIs (cuBLAS/cuDNN) | 脚本 06 + 教程 03 |
| 07 Faster Matmul | 脚本 04 + 教程 02（siboehm 优化阶梯） |
| 08 Triton | 脚本 07 + 教程 04 |
| 09 PyTorch Extensions | 脚本 08 + 教程 04 |
| 10 Final Project (MLP MNIST in CUDA) | 教程 04「毕业项目」路线图（原仓库 README-only，本课给设计图纸与实现顺序） |
| 11 Extras | 教程各章速查（内存层级图、nvcc 流水线、cheatsheet 链接） |

## 文件树

```
courses/Part9_cuda_kernels/
├── scripts/
│   ├── Makefile                       # 一键编译所有 .cu（nvcc -O2 -arch=native）
│   ├── 01_vector_add.cu               # 第一个内核：CPU vs GPU 向量加法 + 正确性验证 + 基准
│   ├── 02_thread_hierarchy.cu         # 网格/块/线程：1D/2D/3D 索引 + square 内核 + 边界守卫
│   ├── 03_naive_matmul.cu             # naive matmul：CPU vs GPU + GFLOPS 初体验
│   ├── 04_matmul_tiled.cu             # 优化阶梯：coalesced → shared memory → 1D/2D block tiling
│   ├── 05_atomics_streams.cu          # atomicAdd 归约 + streams 重叠拷贝与计算
│   ├── 06_cublas_sgemm.cu             # cuBLAS：库调用 vs 手写内核的速度对照
│   ├── 07_triton_kernels.py           # Triton：vector add + softmax（对照 torch）
│   └── 08_pytorch_extension.py        # PyTorch CUDA 扩展：polynomial activation（JIT 编译）
└── tutorial/
    ├── README.md                      # 导航 + 环境自检 + 与 Part 1-8 的关系
    ├── 01_gpu_and_first_kernel.md     # GPU 架构 / 够用的 C / 第一个内核
    ├── 02_matmul_optimization.md      # 内存层级 / 合并访存 / shared memory / block tiling
    ├── 03_profiling_and_cuda_apis.md  # profiling / atomics / streams / cuBLAS / cuDNN
    └── 04_triton_and_extensions.md    # Triton / PyTorch 扩展 / 通向 llm.c 与毕业项目

assignments/assignment_9/
├── assignment.md
├── cuda_exercises.py                  # 纯 Python 可做（索引数学/tiling 模拟/GFLOPS），Triton 题 GPU 可用时自动验证
└── test_cuda_exercises.py
```

## 脚本设计要点

- **忠实原仓库**：内核代码结构、命名（vector_add / matmul / atomicAdd / polynomial_activation）
  与 cuda-course 一致，注释本地化并补充"对应 lecture"标记。
- **规模缩小**：默认 N=1e6（原 1e7）、matmul 512×512（原 256×512×256 保持）、迭代次数减半，
  保证每个脚本 <30s；全部保留 CPU 参照实现 + 逐元素正确性验证（cuda-course 的核心方法论：
  先写 CPU 版，GPU 版与之对照）。
- **基准规范统一**：warm-up → 计时 → 平均 → 输出 speedup 与 GFLOPS，print 全英文（图表/终端
  输出英文惯例），教程里解释输出。
- **编译**：Makefile 统一 `nvcc -O2 -arch=native`（CUDA ≥11.6）；教程给手敲 nvcc 命令版本。
- **Python 脚本**：仅依赖 torch + triton（GPU 机器随 torch 安装）；脚本开头 `torch.cuda.is_available()`
  自检并给出可读报错。PyTorch 扩展用 `torch.utils.cpp_extension.load()`（JIT），同时附
  原仓库 `setup.py` 打包方式说明；⚠️ 标注 nvcc 与 torch CUDA 版本不匹配的坑。

## 教程章节要点

### README
章节导航 / 前置知识（Part 6 的 attention、Part 7 的 KV Cache 与 Flash Attention 提及处、Part 8
的 bf16——只要求"见过这个词"）/ 环境自检命令（nvidia-smi、nvcc -V、torch.cuda）/ 与原仓库
lecture 对照表 / 无 GPU 同学怎么办（作业 1/2/3/5 纯 CPU 可完成；Colab/Kaggle 免费 GPU 跑脚本）。

### 01 GPU 与第一个内核
- 为什么 DL 生态关心内核（生态图：PyTorch → ATen/cuDNN → CUDA → 硬件）
- CPU vs GPU：延迟优化 vs 吞吐优化、核心数对比、什么时候 GPU 反而慢
- 硬件映射：core/thread、SM/block、device/grid；warp（32 线程锁步）
- 够用的 C：指针与数组、`malloc/free`、宏、`nvcc` 编译流水线（对照 gcc）
- 第一个内核 vector add：线程层级、全局索引公式 `i = blockIdx.x*blockDim.x + threadIdx.x`、
  边界守卫、内存拷贝、验证 + 基准（脚本 01/02）
- 2D 索引与 grid-stride loop（脚本 02）

### 02 matmul 优化阶梯
- 内存层级：register/local/shared/L1/L2/global/DRAM + 带宽数字直觉
- row-major vs column-major（呼应 cuBLAS）
- naive matmul 为什么慢：每个输出 2K 次全局读
- 阶梯：① coalesced（转置访问 B）② shared memory tiling ③ 1D block tiling ④ 2D block tiling
  ⑤ 向量化访存（float4/128-bit）⑥ autotuning ⑦ 对标 cuBLAS（脚本 04）
- GFLOPS 与"算力墙 vs 带宽墙"（roofline 直觉）；LLM 推理是 memory-bound 的直觉解释
  （呼应 Part 7 KV Cache 为什么能省时间）
- 与 Flash Attention 的关系：fused + tiled 内核（点到为止，给延伸资料）

### 03 进阶 CUDA 与库
- Profiling：nvtx 打标 / nsys / ncu 怎么读（脚本 05 配套）
- Atomics：atomicAdd 做归约、为什么快不起来（冲突串行化）
- Streams：拷贝与计算重叠、双缓冲直觉
- cuBLAS（sgemm / cublasLt / cublasXt 一句话定位）与 cuDNN（tanh/conv 例）
- occupancy 与 `#pragma unroll`（原 lecture 的两个问答，浓缩成一节）

### 04 Triton 与 PyTorch 扩展
- Triton 设计哲学：CUDA=scalar program + blocked threads；Triton=blocked program + scalar threads
- vec add（`tl.program_id / arange / mask` 与 CUDA 逐行对照）+ softmax（数值稳定技巧——
  呼应 Part 1 的 softmax max 技巧）（脚本 07）
- PyTorch CUDA 扩展：`AT_DISPATCH_FLOATING_TYPES` / `__restrict__` / pybind11 绑定 / 编译两条
  路（load() vs setup.py）（脚本 08）
- 通向 llm.c 与毕业项目：用 CUDA 写 MLP+MNIST 的路线图（对应原仓库 10_Final_Project）；
  延伸：GPUMODE、PMPP 书、siboehm 博客、CUTLASS

## 作业设计（5 题，GPU 可选）

| # | 题目 | 验证 |
|---|------|------|
| 1 | `global_index_1d/2d`：纯 Python 复现 CUDA 索引数学 + 边界守卫逻辑 | 精确索引值 / 范围 |
| 2 | `row_major/col_major` 线性索引 + `matmul_cpu` | 数值 vs torch.matmul |
| 3 | `tiled_matmul_ops`：模拟 tiling 的分块循环，统计"全局内存访问次数"，验证相对 naive 的节省倍数 | 访问计数 / 数值 |
| 4 | 🌟 `gflops_report`：由时间与规模算 FLOPs/GFLOPS/达到 cuBLAS 的百分比 | 数学关系 |
| 5 | `triton_softmax_rows`：补全 Triton softmax 内核（GPU 可用→allclose 验证；否则 pytest.skip） | 条件跳过 |

> 无 GPU 学生可完成 1-4；题 5 代码补全也无需 GPU（仅验证需要），与课程"自包含"哲学一致。

## 验证计划

```bash
# Phase B.1 编译与运行（4090 本机）
cd courses/Part9_cuda_kernels/scripts && make && for b in 01 02 03 04 05 06; do ./bin/$b; done
python courses/Part9_cuda_kernels/scripts/07_triton_kernels.py
python courses/Part9_cuda_kernels/scripts/08_pytorch_extension.py

# Phase B.2 作业测试（含无 GPU 路径检查）
python assignments/assignment_9/test_cuda_exercises.py
```

## 关键设计决策

1. **不引入新数据文件**：全部合成数据，与 Part 8 的 CPU 模式哲学一致。
2. **.cu 保真 + 规模缩小**：内核写法与原仓库同构，改的只有规模/迭代/输出格式。
3. **CPU 参照物贯穿**：每个 GPU 内核都有 CPU 版对照（cuda-course 方法论，也呼应 Part 1
   "计数版 vs 神经网络版"的对照传统）。
4. **面试/工程出口**：04 章给 llm.c、GPUMODE、PMPP 等延伸路径，作为整个课程的"往下走"地图。
