# Assignment 9：CUDA 内核编程

> 对应 Part 9 教程：[courses/Part9_cuda_kernels/tutorial/](../../courses/Part9_cuda_kernels/tutorial/README.md)
> 参考项目：[infatoshi/cuda-course](https://github.com/infatoshi/cuda-course)（FreeCodeCamp CUDA Course）

## 🎯 作业目标

CUDA 内核的核心难点**不在 C 语法，而在三件事**：

1. **索引数学**——把"一个线程"对应到"数据的一个位置"（全局索引公式）
2. **访存账本**——算清楚一个内核要读/写多少次内存（tiling 为什么有效）
3. **性能语言**——用 GFLOPS 描述内核快慢，和峰值比、和 cuBLAS 比

这份作业刻意让你**不写一行 CUDA** 也能把这三件事练会（题 1-4 纯 Python / 纸上推导）；
题 5 是给有 GPU 的同学准备的 Triton 实战。

## 📋 完成方式

1. 打开 `cuda_exercises.py`，找到每个函数里的 `TODO`
2. 实现后运行测试：
   ```bash
   python test_cuda_exercises.py        # 独立运行
   # 或
   pytest test_cuda_exercises.py       # pytest 运行
   ```
3. 题 1-4 未实现会 ❌；题 5 未实现或无 GPU 会 ⏭️ 跳过（不影响其他题）

## 📝 题目列表

### 题 1：全局线程索引（30 分）——`global_index_1d` / `global_index_2d` / `launch_config`

用纯 Python 复现 CUDA 的索引数学。GPU 上"每个线程是谁"全靠这套公式：

```
1D:  i = blockIdx.x * blockDim.x + threadIdx.x
2D:  col = blockIdx.x * blockDim.x + threadIdx.x
     row = blockIdx.y * blockDim.y + threadIdx.y
     idx = row * n_cols + col          # row-major 展平
启动: num_blocks = (n + block_size - 1) // block_size   # 向上取整！
```

**要求**：
- `global_index_1d(block_idx, thread_idx, block_size)` → 全局编号
- `global_index_2d(block_idx, thread_idx, block_size, n_cols)` → `(row, col, idx)`
- `launch_config(n, block_size)` → `(num_blocks, total_threads)`，
  注意 `total_threads` 可大于 `n`——这就是内核里 `if (i < n)` 存在的原因

<details>
<summary>💡 提示</summary>

2D 就是把 1D 公式对 x/y 各算一遍，再按行主序展平。对照脚本
`courses/Part9_cuda_kernels/scripts/02_thread_hierarchy.cu` 的 `square_2d`。

</details>

### 题 2：行主序 / 列主序 + CPU matmul（30 分）——`row_major_index` / `col_major_index` / `matmul_cpu`

- `row_major_index(row, col, n_cols)`：`A[row][col]` 在内存里的下标（C/PyTorch 方式）
- `col_major_index(row, col, n_rows)`：同上，但按 cuBLAS/Fortran 的列主序
- `matmul_cpu(A, B)`：三重循环 matmul（输入输出都是 list of lists）

<details>
<summary>💡 提示</summary>

行主序：`row * n_cols + col`；列主序：`col * n_rows + row`。
区别只有"谁乘步长"——但它正是 cuBLAS 那套"行主序当列主序读 = 读到转置"魔法的原因。
`matmul_cpu` 对照脚本 03 的 `matmul_cpu`，注意先累加到局部变量再写入。

</details>

### 题 3：tiling 的"账本"（20 分）——`count_global_reads` / `tiled_speedup_ratio`

不跑 GPU，**算出** shared memory tiling 省了多少全局读：

- naive：每个输出元素沿 K 读一整行 A + 一整列 B → 共 `M*N*2*K` 次读
- tiled(T)：每个 T×T 输出 tile 对应的 block 把 A 的 T 行 + B 的 T 列各读一次
  → 共 `(M/T) * (N/T) * 2*T*K` 次读

**要求**：两个函数实现后，`tiled_speedup_ratio` 应对任意 T 恰好返回 T
（测试会检查 8/16/32/64 四种 tile）。

<details>
<summary>💡 这道题在说什么？</summary>

这就是"内存墙"的算术：matmul 的浮点运算是固定的 `2MNK`，但 naive 要做 `2MNK` 次
全局读（1 FLOP : 1 读），tiling 后变成 `2MNK/T` 次。GPU 的 FLOPS 远快于全局内存带宽，
所以把读次数除以 T，速度上限就乘以 T。脚本 04 里 L3→L5 的 GFLOPS 提升就是这道题的实测版。

</details>

### 题 4：🌟 GFLOPS 报告（10 分）——`gflops_report`

内核优化的日常动作：跑 → 测时间 → 算 GFLOPS → 和峰值比。

- matmul 的浮点运算次数 `≈ 2*M*N*K`（乘 + 加各算一次）
- `GFLOPS = 2*M*N*K / (time_ms / 1000) / 1e9`
- 传入 `peak_gflops` 时，另算达成百分比

跑完脚本 04/06 后，用真实数字填一张表：你的 L5 内核达到 cuBLAS 的百分之几？
（4090 上我们的参考答案：L5 ≈ 8800 GFLOPS，cuBLAS ≈ 22000 GFLOPS，约 40%——
剩下 60% 靠向量化访存、autotuning、Tensor Core，见教程 02 章"通往 cuBLAS"。）

### 题 5：Triton softmax（10 分，需要 GPU）——`triton_softmax`

把脚本 07 的 softmax 内核自己再写一遍，对照 `torch.softmax` 验证。

<details>
<summary>⚠️ 最常见的坑</summary>

`@triton.jit` 函数必须定义在**模块顶层**。定义在别的函数内部时，Triton 编译器
找不到 `tl`（它只扫描模块 globals），报 `NameError: tl is not defined`。
这是 Triton 初学者最常撞的墙之一。

</details>

## 🤔 思考题

**Q1：`launch_config(1000, 256)` 返回 `total_threads=1024 > 1000`。这 24 个"多余"线程
在内核里会发生什么？如果没有 `if (i < n)` 会怎样？**

<details>
<summary>💡 提示</summary>

它们会被启动、执行同样的代码。没有边界守卫时，它们会读写 `x[1000..1023]`——
越界访问未分配的显存。轻则结果错乱（读脏数据），重则 CUDA error 崩掉整个 context
（且往往在之后的某次调用才爆出来，非常难查）。

</details>

**Q2：为什么 4090 的 fp32 "理论峰值" 有 ~82 TFLOPS，而 cuBLAS 单精度 GEMM 实测只跑出
~22 TFLOPS？列出至少 3 个原因。**

<details>
<summary>💡 提示</summary>

① 82 TFLOPS 是"乘加器全速"的纸面值，实际还有访存等待、同步、occupancy 损耗；
② 理论峰值通常按 2×FMA 计数口径，和 2MNK 的口径未必一致；
③ 深度学习负载的 GEMM 还要写回输出；④ 小矩阵（512³）喂不满 GPU，kernel 启动/
尾效应占比大；⑤ cuBLAS 这么快本身已经用了 Tensor Core（TF32）+ 深度调优。

</details>

**Q3：题 3 的账本假设了"每个 tile 从全局读一次"。但 naive 的读其实会命中 L2 cache
（512³ 的 A/B 才 1MB，4090 的 L2 有 72MB）。那 tiling 在真实大矩阵上为什么更重要？**

<details>
<summary>💡 提示</summary>

小矩阵缓存把 naive 的重复读"洗白"了——所以脚本 04 里 512³ 各内核差距不算悬殊。
真实 LLM 的权重矩阵是 GB 级，远超 L2，重复读全部打到 HBM；此时 tiling 的
`1/T` 读次数直接决定速度。这也解释了"LLM 推理是 memory-bound"和 Part 7 KV Cache
为什么有效。测内核永远要在真实规模下测。

</details>

## ✅ 提交检查清单

- [ ] `python test_cuda_exercises.py` 题 1-4 全部 ✅（有 GPU 的话题 5 也 ✅）
- [ ] 能不看笔记说出全局索引公式和向上取整启动公式
- [ ] 能用自己的话说清"tiling 为什么把全局读除以 T"
- [ ] 跑过脚本 04 和 06，知道你的机器上手写内核 vs cuBLAS 的差距
