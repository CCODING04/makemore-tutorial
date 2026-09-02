# Part 9: CUDA 内核编程 — 打开深度学习的引擎盖

> 🚀 Part 1-8 里 `tensor @ tensor` 一直是黑盒。这一部分我们亲手写 CUDA 内核：
> 从第一个 vector add，到手写 matmul 优化阶梯（对比 cuBLAS），再到 Triton 和
> PyTorch 自定义扩展——最后把它们组装成"毕业内核"：手写 Flash Attention。
> 参考：[infatoshi/cuda-course](https://github.com/infatoshi/cuda-course)（FreeCodeCamp CUDA Course）

## 🎯 学习目标

完成本部分后，你将能够：

- ✅ **画出** GPU 的执行模型（SM / 线程层级 / 存储层次），解释一个内核是怎么跑起来的
- ✅ **攀登** 手写 matmul 的优化阶梯（合并访存 → SMEM tiling → block tiling），用
  "算力墙 / 内存墙"给每一级定位瓶颈
- ✅ **使用** nsys / ncu 做 profiling，用测量数据（而非直觉）决定下一步优化什么
- ✅ **编写** Triton 内核（融合算子），说清它与手写 CUDA 的取舍
- ✅ **封装** 自定义算子为 PyTorch 扩展，接入现有训练代码
- ✅ **手写** Flash Attention 前向内核（online softmax + causal），对照 SDPA 四后端验收

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [GPU 架构与第一个内核](01_gpu_and_first_kernel.md) | CPU vs GPU、线程层级、够用的 C、CUDA 五步曲 | `01` `02` |
| 02 | [matmul 优化阶梯](02_matmul_optimization.md) | 算力墙/内存墙、合并访存、SMEM tiling、block tiling、cuBLAS 对照 | `03` `04` |
| 03 | [Profiling 与 CUDA 库](03_profiling_and_cuda_apis.md) | nsys/ncu、atomics 归约、streams 重叠、cuBLAS（cuDNN 概念带过） | `05` `06` |
| 04 | [Triton 与 PyTorch 扩展](04_triton_and_extensions.md) | Triton 内核、自定义算子接入 PyTorch、通向 llm.c 的路线图 | `07` `08` |
| 05 | [Flash Attention 毕业内核](05_flash_attention.md) | online softmax 推导、causal 三阶段、SDPA 四后端实测、FlexAttention/SageAttention | `09` |

## 🧰 前置知识

**必须掌握：**

- **C 的最低子集**：能看懂 `for` 循环和 `a[i]` 数组下标——01 章有"够用的 C"速成节
  （指针/数组/宏/编译，10 分钟版），零基础也来得及

**建议掌握：**

- **Part 6**：Transformer / attention（[Part 6 教程](../../Part6_transformer/tutorial/README.md)）——
  知道"`q @ k^T` 是个算子"这个层面就够了，02 章会分析它为什么是性能大头
- **Part 7 03 章 / Part 8**：见过 Flash Attention、KV Cache、bf16 autocast 这些词
  —— Part 9 会把它们的"内核原理"补上（02/04 章的连接点）

**可选：**

- **Part 3**：诊断工具的思路（先测量、再下结论）——03 章的 GPU profiling 是它的镜像，
  没学过也不影响
- 无 GPU 也能学：概念全可读，`.cu` 脚本可上 Colab/Kaggle 跑（见下方环境表），
  作业题 1-4 纯 CPU 可完成

> 💡 本部分与前面所有部分**代码完全独立**，随时可以直接开始；但学过 Part 7/8 的同学
> 会在"内存墙 → KV Cache / Flash Attention / bf16"这些连接点上获得双倍回报。

## 🗺️ 学习路线图

```
Part 1-8 (PyTorch 视角：模型是"用"出来的)
    │
    │  "GPU 到底怎么算 matmul / softmax / attention？"
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Part 9: CUDA 内核（模型是"算"出来的）                          │
│                                                              │
│  ① GPU 架构 + 第一个内核  — 线程层级 / vector add            │──→ 01_gpu_and_first_kernel.md
│  ② matmul 优化阶梯       — naive → coalesced → SMEM → tiling │──→ 02_matmul_optimization.md
│  ③ 进阶 CUDA + 库        — atomics / streams / cuBLAS        │──→ 03_profiling_and_cuda_apis.md
│  ④ Triton + PyTorch 扩展 — 融合内核 / 自定义算子 / llm.c     │──→ 04_triton_and_extensions.md
│  ⑤ Flash Attention       — online softmax / causal / SDPA 验收│──→ 05_flash_attention.md
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 📦 环境要求（与其他部分不同！）

本部分脚本需要 **NVIDIA GPU + CUDA Toolkit**：

```bash
nvidia-smi        # ① 驱动 + 显卡
nvcc -V           # ② CUDA Toolkit（nvcc 编译器）
python -c "import torch; print(torch.cuda.is_available(), __import__('triton').__version__)"
                  # ③ torch(GPU 版) + triton（torch 自带，Linux 下无需单独安装）
```

| 缺什么 | 怎么办 |
|---|---|
| 都有 | `cd courses/Part9_cuda_kernels/scripts && make && make run`，然后跑 07/08/09 三个 .py（09 需独占 GPU，约 2-4 分钟，大头是 autotune 编译） |
| 没有 GPU | 概念照学（教程全可读）；`.cu` 脚本上 [Colab](https://colab.research.google.com)/Kaggle 免费卡跑；**作业题 1-4 纯 CPU 可完成** |
| 有 GPU 没 nvcc | 装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)（或对应 PyTorch 版本），Colab 也可 |

**多版本 CUDA 共存（本课开发机的真实配置）**：

机器上可以同时装多个 CUDA Toolkit（例如 `/usr/local/cuda-11.8` 和 `/usr/local/cuda-12.4`），
互不覆盖：

- `PATH` 和 `/usr/local/cuda` 软链接保持指向旧版 → 已有环境（其他项目、脚本）完全不受影响
- 本课的 Makefile 和 PyTorch 扩展脚本会**自动选择版本最高的** `/usr/local/cuda-*`
- 想手动指定某个版本：`make NVCC=/usr/local/cuda-11.8/bin/nvcc`，或在 Python 里
  `os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"`
- 安装新版本到独立目录（不动软链接）：
  ```bash
  sudo sh cuda_12.4.1_*.run --silent --toolkit --override --toolkitpath=/usr/local/cuda-12.4
  # 装完检查软链接，若被改向新版本，恢复指回旧版：
  sudo ln -sfn /usr/local/cuda-11.8 /usr/local/cuda
  ```

**编译相关坑点**（都来自真机实测，撞到时回来看）：

- `unsupported GNU version! gcc versions later than 11...`：CUDA 对宿主 gcc 有版本上限
  （11.x 最高 gcc-11；12.1+ 原生支持 gcc-12/13）。选中 11.x 时 Makefile/扩展脚本会自动
  加 `-ccbin gcc-11`，选 12.x 则什么都不用做
- **多版本共存时的运行时库**：编译用的 libcublas 等也要在运行时被找到，而系统 ldconfig
  通常只配了旧版路径。Makefile 已把所用版本的 `lib64` 用 rpath 焊进二进制（`ldd bin/06_cublas_sgemm`
  可验证解析到了哪个版本），无需配 `LD_LIBRARY_PATH`
- **PyTorch 扩展的 CUDA_HOME 缓存**：torch 在 `import` 时就把 CUDA_HOME 解析并缓存成
  模块全局，之后只改 `os.environ` 无效——要同时覆盖 `torch.utils.cpp_extension.CUDA_HOME`
  （脚本已处理）
- PyTorch 扩展 JIT 编译报 `Ninja is required`：`pip install ninja` 并确保其可执行文件在 PATH
- torch 扩展编译产物缓存在 `~/.cache/torch_extensions/`，异常时清掉重编

**与原课程的对应**：本部分完整覆盖 cuda-course 的核心 lecture
（01 生态 / 03 C 复习 / 04 GPU 简介 / 05 first kernels / 06 APIs / 07 faster matmul /
08 Triton / 09 extensions），映射表见 [docs/part9_cuda_kernels_plan.md](../../../docs/part9_cuda_kernels_plan.md)。

## 📈 实测参考（RTX 4090，fp32，512³ matmul）

我们的脚本在你机器上跑出来会是类似这样（数字随硬件浮动，**看趋势**）：

```
L1 uncoalesced     553.6 GFLOPS      ← 合并访存做错（对照组）
L2 coalesced      4844.9 GFLOPS      ← 只是"读的方式"对了
L3 smem tile      5905.4 GFLOPS      ← shared memory 复用
L4 1D blocktile   6967.5 GFLOPS      ← 寄存器复用 x8
L5 2D blocktile   8795.2 GFLOPS      ← 手写阶梯的尽头
cuBLAS           22163.1 GFLOPS      ← Tensor Core + autotune（库的天花板）
```

> ⚠️ 小矩阵（512³ 全部塞进 L2 cache）会低估 tiling 的收益；教程 02 章解释了为什么
> 大矩阵下差距会拉大。**看趋势，别死记数字。**

**Flash Attention（脚本 09，RTX 4090，torch 2.6.0+cu124 / triton 3.2.0，
bf16，B=2 / H=8 / D=64，前向，预热 10 + 测 50；2026-09-02 共享 GPU 实测）：**

```
T=4096 causal:  naive 6.198 ms → 手写 Triton FA 0.279 ms (123 TF, 22.2x)
                SDPA 最优（cudnn）0.308 ms (112 TF) → 手写版 111%
T=4096 full:    naive 3.579 ms → 手写 Triton FA 0.436 ms (158 TF, 8.2x)
                SDPA 最优（cudnn）0.414 ms (166 TF) → 手写版 95%
验收线: 教学版 >= SDPA 最优后端 50% 合格、>85% 优秀 → 实测 95-127%，全部"优秀"
（空闲整卡上多轮运行曾实测 105%~163%；计时的"独占 vs 共享"是公平性变量，见教程 4.4）
```

> 📝 教学版只做前向（不物化 logsumexp、无 backward），SDPA 要为 autograd 额外写
> LSE，小形状下手写版可能反超；比较内核快慢须同卡同负载（独占/共享数字会整体漂移）。详见
> [05 章](05_flash_attention.md)的实测小节。

## 📝 课后作业

每章末尾有思考题（`<details>` 折叠答案）。全部学完后：

👉 [Assignment 9](../../../assignments/assignment_9/)
（题 1-4 纯 CPU 可完成：索引数学 / 行列主序 / tiling 账本 / GFLOPS 报告；题 5 是 Triton 实战）

## 🔗 相关资源

- 🐙 [infatoshi/cuda-course](https://github.com/infatoshi/cuda-course) — 本部分参考的项目（FreeCodeCamp）
- 📝 Simon Boehm：[How to Optimize a CUDA Matmul Kernel to cuBLAS in 1 Hour](https://siboehm.com/articles/22/CUDA-MMM) — matmul 阶梯的出处
- 📁 [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) — 阶梯完整代码
- 🐙 [karpathy/llm.c](https://github.com/karpathy/llm.c) — 本课程的"毕业读物"（纯 C/CUDA 训练 GPT-2）
- 📖 [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — 官方手册
- 📚 PMPP（*Programming Massively Parallel Processors*）— GPU 编程圣经
- 🐙 [triton-lang/triton](https://github.com/triton-lang/triton) — Triton 官方仓库 + tutorials（fused attention 等）
- 📺 [GPUMODE](https://www.youtube.com/@GPUMODE) — 每周 GPU 内核讲座（原课程推荐）

---

[← 上一章：Part 8 后训练全流程](../../Part8_post_training/tutorial/README.md) | [下一章：Part 10 分布式训练 →](../../Part10_distributed/tutorial/README.md)
