# 04 — Triton 与 PyTorch 扩展：通向 llm.c

> 🧭 最后一章回答两个实际问题：① 不写 C++，能不能写内核？（Triton——Part 7 提过的
> Flash Attention、多数现代开源内核的语言）② 自己写的内核怎么接进 PyTorch 训练管线？
> 最后给出整个课程的"毕业去向"地图。

## 📖 前置知识

- **01-03 章**：线程层级、SMEM/tiling、合并访存（Triton 帮你管的正是这些）
- **Part 1**：softmax 的"减最大值防上溢"技巧（Triton 一节会再见到它）
- **Part 7/8**：Flash Attention、KV Cache 出现的位置（本章把它们和内核语言连起来）

## Triton：Python 写内核（对应原课程 08 课）

### 设计哲学：CUDA vs Triton

原课程 README 里这张对照是精髓：

```
CUDA   = scalar program + blocked threads
         你逐线程写代码（标量视角），把线程组织成 block（你来操心）
Triton = blocked program + scalar threads
         你按"一块数据"写代码（向量视角），线程怎么划分（编译器替你操心）
```

| | CUDA | Triton |
|---|---|---|
| 语言 | C/C++ | Python（装饰器 + `triton.language`） |
| 谁管 tiling/mask/SMEM | 你 | **编译器** |
| 性能上限 | 最高（最后 10-20%） | 接近（elementwise/reduction 类几乎打平） |
| 写一个 softmax 的代码量 | 上百行 | ~20 行 |
| 典型用户 | NVIDIA、CUTLASS | OpenAI/FlashAttention、Unsloth、torch.compile 的后端 |

### vector add：逐行对照 CUDA（[scripts/07_triton_kernels.py](../scripts/07_triton_kernels.py)）

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)             # CUDA: blockIdx.x（这次一个"程序"管一块）
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # 这一块的 1024 个下标（向量！）
    mask = offsets < n_elements             # CUDA: if (i < n)，但一次判断 1024 个
    x = tl.load(x_ptr + offsets, mask=mask) # load/store 带 mask，越界自动挡
    tl.store(output_ptr + offsets, x + tl.load(y_ptr + offsets, mask=mask), mask=mask)
```

- 🔑 没有 threadIdx：**你写的是"一块数据怎么算"**，编译器自动把它铺到合适的线程/warp 上，
  并自动做向量化访存（你手写 `float4` 才能做到的事，这里默认就有）。
- `tl.constexpr` 的 BLOCK_SIZE 是编译期常量——不同 BLOCK_SIZE 生成不同内核（autotune 的抓手）。
- ⚠️ 本章作业（题 5）马上会踩的坑：`@triton.jit` 函数必须定义在**模块顶层**，
  否则 `NameError: tl is not defined`。

### softmax：整块加载 + 数值稳定（呼应 Part 1）

```python
row = tl.load(row_start_ptr + col_offsets, mask=..., other=-float('inf'))
row_minus_max = row - tl.max(row, axis=0)      # Part 1 的老朋友：先减最大值
numerator = tl.exp(row_minus_max)
softmax_output = numerator / tl.sum(numerator, axis=0)
```

一个 program 负责矩阵的**一行**：整行载入 SMEM（`tl.load` 自动处理）、
片上完成 softmax、一次写回——**没有中间的全局内存往返**。这就是"内核融合"
（kernel fusion）的最小样本：torch eager 里 `max → exp → sum → div` 四个内核
四次显存往返，融合后一遍完成。

**实测（4090）**：

```
[vecadd]  triton 0.007 ms vs torch 0.004 ms   (effective BW ~1.70 TB/s -> memory-bound)
[softmax] triton 0.010 ms vs torch 0.012 ms
```

> 💡 诚实解读：elementwise 上 Triton 和 torch 内核互有胜负（都是带宽上限附近）；
> Triton 的价值不在"比 cuBLAS 快"，而在**用 1% 的代码量写出"够快"的融合内核**——
> Flash Attention 原始实现、 Unsloth、绝大部分 SOTA 开源内核都是 Triton 写的。

## PyTorch CUDA 扩展：自己的内核接进训练管线（对应原课程 09 课）

场景：你写好了一个 CUDA 内核（比如算子融合的 activation、自定义 attention），
想让 PyTorch 张量直接进出。原课程 09 课用 `x²+x+1`（polynomial activation）演示了
完整闭环，我们的 [scripts/08_pytorch_extension.py](../scripts/08_pytorch_extension.py) 用 `load_inline()` 现场编译同款：

```cpp
template <typename scalar_t>                       // scalar_t = float 或 double
__global__ void polynomial_activation_kernel(
    const scalar_t* __restrict__ x,                // __restrict__: 承诺不重叠 → 放心优化
    scalar_t* __restrict__ output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = x[idx];
        output[idx] = val*val + val + 1;
    }
}

torch::Tensor polynomial_activation_cuda(torch::Tensor x) {   // C++ 包装：张量进张量出
    int threads = 1024, blocks = (x.numel() + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(x.type(), "...", ([&] {        // 按 dtype 实例化模板
        polynomial_activation_kernel<scalar_t><<<blocks, threads>>>(...);
    }));
    return output;
}
```

**三个关键件**（原课程 README 逐个讲过）：

1. `AT_DISPATCH_FLOATING_TYPES`：同一份内核代码，自动支持 fp32/fp64（泛型分发）。
2. `__restrict__`：向编译器承诺两个指针不重叠，启用激进优化。原课程给了反例：
   `add_arrays(data, data+3, 7)` 这种重叠调用在 `__restrict__` 下行为未定义。
3. pybind 绑定：`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("polynomial_activation", ...); }`
   ——把 C++ 函数暴露成 Python 函数。（`load_inline()` 的 `functions=[...]` 参数
   自动生成这段；`setup.py` 打包路线需要手写，见原仓库 `09_PyTorch_Extensions/setup.py`。）

**实测**：

```
[speed] custom CUDA : 0.0042 ms  (1 个融合内核)
[speed] torch eager : 0.0110 ms  (中间张量 + 3 次内核启动)
[grad]  手写 backward 与 (2x+1) 完全一致
```

- ⚠️ `load_inline()` 编译的是**裸函数**，没有 autograd 支持——反向传播不会自动工作。
  工程做法：`torch.autograd.Function` 包一层，`backward()` 手写（本例一行：`2x+1`）。
  这正是理解"PyTorch 算子 = forward 内核 + backward 内核"的最佳练习
  （呼应 Part 4 手动反向传播！）。
- ⚠️ 编译坑（我们在 4090 机上真实撞到）：CUDA 对宿主 gcc 有版本上限（CUDA 11.x 最高
  gcc-11），报 `unsupported GNU version` 时给 nvcc 传 `-ccbin g++-11`；CUDA 12.1+ 则原生
  支持 gcc-12/13，无需任何参数。机器上多版本 CUDA 共存时，脚本会自动选版本最高的
  `/usr/local/cuda-*`——注意**只改 `os.environ["CUDA_HOME"]` 没用**：torch 在 `import`
  时就把 CUDA_HOME 缓存成了模块全局，必须同时覆盖 `torch.utils.cpp_extension.CUDA_HOME`
  （不动全局环境、不覆盖旧版本——本课开发机就是 11.8 与 12.4 并存）；JIT 编译还需要
  `ninja`；产物缓存在 `~/.cache/torch_extensions/`，改了名字/源码不生效时先清缓存。

## 毕业去向：这门课之后学什么（对应原课程 10/11 课）

原课程的 final project 是"用 CUDA 从零写 MNIST 的 MLP"（仓库只给了架构图，
留作练习）。结合我们的课程体系，给三条进阶路线：

**路线 A：把内核学深（系统方向）**
1. 原课程 Final Project：CUDA 写 MLP forward + 手写 softmax/CE backward（把 Part 4 的
   手动反传翻译成 CUDA，是极佳的综合练习）
2. PMPP 书（*Programming Massively Parallel Processors*，GPU 编程圣经）
3. Karpathy 的 **llm.c**——本课开篇说的目标：现在你能读懂它的 matmul/attention 内核了
4. CUTLASS：NVIDIA 开源的 GEMM 模板库（02 章阶梯的工业完全体）

**路线 B：把内核用起来（Deep Learning 方向）**
1. Triton 官方 tutorials：fused-softmax、matmul、**flash-attention**（把 02 章的直觉落地）
2. torch.compile：看看 Inductor 给你的模型生成了什么 Triton 内核
3. GPUMODE 社区（原课程推荐）：每周内核优化讲座 + 竞赛

**路线 C：回到课程主线**
Part 7/8 训练过的模型，现在你知道：`@` → cuBLAS → Tensor Core；`autocast(bf16)` →
内存带宽减半；KV Cache → 显存里的持久 buffer；DPO/GRPO 的慢 → 采样循环的内核
launch 次数。**回去把超参改一改、用 nsys 看看时间线**——训练器视角和内核视角合璧，
才算真正打通"从 tensor 到 SM"。

## 学完本部分你能...

- ✅ 用"Triton=CUDA 的编译器换你操心线程"解释两者的分工
- ✅ 写出带 mask 的 Triton elementwise / 行归约内核，并绕开"嵌套定义"的坑
- ✅ 说清 PyTorch 扩展三件套（dispatch / restrict / pybind）各干什么
- ✅ 用 `torch.autograd.Function` 给自定义内核补 backward
- ✅ 给自己画出 Part 9 之后的三条进阶路线

**课后练习**

<details>
<summary>Q1: Triton 为什么"写不出"cuBLAS 顶级的 matmul？它放弃/隐藏了哪些控制权？</summary>
A: 你无法手控 warp 级原语（shuffle/mma 指令）、寄存器分配、shared memory 的精确布局、
double buffering 的调度——这些正是 GEMM 最后 20% 性能的来源（Tensor Core 的 mma 指令
排布尤其需要）。Triton 的赌注是：99% 的内核不是 GEMM，把 tiling/mask/向量化自动化
收益远大于损失。所以生态分工：GEMM 给 cuBLAS/CUTLASS，长尾融合内核给 Triton。
</details>

<details>
<summary>Q2: 你的自定义 activation 快了 2.6 倍（0.0110→0.0042ms）。为什么 torch eager 慢？
如果用 torch.compile 会怎样？</summary>
A: eager 把 x*x、+x、+1 拆成 3 个内核：各读写一遍 4MB 张量（12MB 流量）+ 3 次启动开销；
自定义内核一遍完成（8MB 流量 + 1 次启动）。torch.compile（Inductor）会生成 Triton
融合内核，性能通常接近手写——所以工程顺序是：先 compile，不够再手写。
</details>

<details>
<summary>Q3:（综合 8 课）训练 LLM 时收到"GPU util 只有 40%"的报告，列出至少 4 个
可能原因和对应工具。</summary>
A: ① 数据加载慢（CPU 瓶颈）→ nsys 看 GPU 等待间隙，DataLoader 加 workers/prefetch；
② 小 batch/小模型 → kernel launch 开销占比大，batch 拼大或 CUDA Graphs；
③ 频繁 CPU-GPU 同步（.item()/print loss）→ 异步化、少同步；
④ 内存墙算子多（优化器、embedding 查表）→ 融合优化器（fused adamw）、
减少 host-device 拷贝。工具链：nsys 时间线定位 → ncu 看单内核 → 改架构/融合。
</details>

## 📝 课后作业

👉 [Assignment 9](../../../assignments/assignment_9/)（题 5：亲手写 Triton softmax）

## 课程回顾

```
Part 1-5  用神经网络写人名      →  学会"训练"这件事本身
Part 6    从零搭出 GPT          →  学会 Transformer 骨架
Part 7    复现 minimind         →  学会现代 LLM 的零件（RoPE/GQA/SwiGLU/MoE）
Part 8    后训练全流程          →  学会 SFT→RM→DPO→PPO→GRPO
Part 9    CUDA 内核             →  学会这一切跑在什么机器上、为什么快、还能更快
```

下一步见 ✋

---

[← 上一章：Part 8 后训练全流程](../../Part8_post_training/tutorial/README.md)
