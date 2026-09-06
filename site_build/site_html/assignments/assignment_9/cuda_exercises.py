#!/usr/bin/env python3
"""
Part 9 作业：CUDA 内核编程

设计原则：不用 GPU 也能完成题 1-4（索引数学 / tiling 模拟 / GFLOPS 分析
都可以在 CPU 上"纸上跑"）；题 5 需要 GPU + triton，未实现或无 GPU 时测试自动跳过。

每道题的 TODO 注释是步骤级提示，几乎等于伪代码 —— 先自己想，卡住了再看。
"""

import math

# torch 只有题 2/3/5 需要；纯 CPU 也能 import（题 1/4 完全不用）
try:
    import torch
except ImportError:      # 极端情况：没装 torch 也允许做题 1/4
    torch = None


# ═════════════════════════════════════════════════════════════════════
#  题 1：全局线程索引 —— 用纯 Python 复现 CUDA 的索引数学（30 分）
# ═════════════════════════════════════════════════════════════════════

def global_index_1d(block_idx, thread_idx, block_size):
    """
    复现 1D CUDA 内核的第一行：int i = blockIdx.x * blockDim.x + threadIdx.x;

    Args:
        block_idx:  block 在 grid 中的编号（0 起）
        thread_idx: 线程在 block 内的编号（0 起）
        block_size: 每个 block 的线程数（blockDim.x）

    Returns:
        全局线程编号 i（int）
    """
    # TODO: 一行乘法 + 一行加法
    return None


def global_index_2d(block_idx, thread_idx, block_size, n_cols):
    """
    复现 2D CUDA 内核的索引（脚本 03 的 matmul 就用它）：
        col = blockIdx.x * blockDim.x + threadIdx.x
        row = blockIdx.y * blockDim.y + threadIdx.y
        idx = row * n_cols + col          # row-major 展平

    Args:
        block_idx:  (bx, by) —— block 的 x/y 编号
        thread_idx: (tx, ty) —— 线程在 block 内的 x/y 编号
        block_size: (bx_size, by_size) —— blockDim.x / blockDim.y
        n_cols:     矩阵列数（展平时的行宽）

    Returns:
        (row, col, idx) 三元组
    """
    # TODO:
    #   1. 算 col 和 row（同 1d 公式，各算一遍）
    #   2. idx = row * n_cols + col
    #   3. return (row, col, idx)
    return None


def launch_config(n, block_size):
    """
    复现 CUDA 启动时 block 数量的"向上取整除法"：
        num_blocks = (n + block_size - 1) // block_size

    并回答：为什么要向上取整？多出来的线程靠什么不越界？（见 docstring 答案折叠）

    Args:
        n: 总元素/线程数
        block_size: 每个 block 的线程数

    Returns:
        (num_blocks, total_threads) —— total_threads = num_blocks * block_size，
        注意它可能 > n！多出来的线程就是内核里 `if (i < n)` 拦下来的那些。
    """
    # TODO: 向上取整算 num_blocks，再算 total_threads
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 2：行主序 / 列主序 + CPU matmul（30 分）
# ═════════════════════════════════════════════════════════════════════

def row_major_index(row, col, n_cols):
    """行主序线性下标：A[row][col] 存在 A[row * n_cols + col]（C 语言 / PyTorch 的方式）"""
    # TODO: 一行
    return None


def col_major_index(row, col, n_rows):
    """列主序线性下标：A[row][col] 存在 A[col * n_rows + row]（cuBLAS/Fortran 的方式）"""
    # TODO: 一行 —— 注意和行主序的差别：谁乘步长、谁是加数
    return None


def matmul_cpu(A, B):
    """
    纯 Python 三重循环 matmul（对应脚本 03 的 matmul_cpu）。

    Args:
        A: list of lists，形状 (M, K)
        B: list of lists，形状 (K, N)

    Returns:
        C: list of lists，形状 (M, N)，C[i][j] = sum_l A[i][l] * B[l][j]
    """
    # TODO:
    #   1. M = len(A), K = len(B), N = len(B[0])
    #   2. 三重循环：外两层 i/j，内层 l 累加
    #   3. 别忘了 C[i][j] = sum（先累加到局部变量再写入）
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 3：tiling 的"账本"—— 全局内存读取次数分析（20 分）
# ═════════════════════════════════════════════════════════════════════

def count_global_reads(M, N, K, tile):
    """
    数一数两种 matmul 各要从（慢的）全局内存读多少次数据。SMEM 免费不计。

    naive（脚本 03）：每个输出元素 C[i][j] 都要沿 K 读一整行 A + 一整列 B
      -> 每个输出 2*K 次读，共 M*N 个输出
    tiled（脚本 04 L3）：每个 T×T 的输出 tile，对应的 block 只需把
      A 的 T 行 + B 的 T 列 各完整读一次 -> 每 tile 2*T*K 次，共 (M/T)*(N/T) 个 tile
      （M、N、K 均能被 tile 整除；能整除时公式才精确成立）

    Args:
        M, N, K: 矩阵规模
        tile: shared memory tile 的边长 T

    Returns:
        (naive_reads, tiled_reads) —— 都按"元素个数"计（不乘 4 字节）
    """
    # TODO:
    #   naive_reads = M * N * 2 * K
    #   tiled_reads = (M // tile) * (N // tile) * 2 * tile * K
    #   return 两者
    return None


def tiled_speedup_ratio(M, N, K, tile):
    """
    返回 naive_reads / tiled_reads，用 count_global_reads 的结果算。
    化简后它应该正好等于 tile —— 这就是"tiling 把全局读次数除以 T"的直觉来源。
    """
    # TODO: 调用 count_global_reads，返回比值（float）
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 4：GFLOPS 报告 —— 内核好不好，用数字说话（10 分，必做）
# ═════════════════════════════════════════════════════════════════════

def gflops_report(time_ms, M, N, K, peak_gflops=None):
    """
    给定内核耗时，产出"性能报告"。这是内核优化的日常动作：跑 -> 测 -> 算 GFLOPS。

    Args:
        time_ms: 内核平均耗时（毫秒）
        M, N, K: matmul 规模（FLOPs ≈ 2*M*N*K：乘加各算一次浮点运算）
        peak_gflops: 硬件理论/实测峰值（可选），传入时计算达成百分比

    Returns:
        dict: {
            'flops':   总浮点运算次数（float）,
            'gflops':  GFLOPS = flops / 秒 / 1e9,
            'pct_of_peak': 达成峰值百分比（peak_gflops 为 None 时本键为 None）
        }
    """
    # TODO:
    #   1. flops = 2 * M * N * K
    #   2. gflops = flops / (time_ms/1000) / 1e9
    #   3. pct = gflops / peak_gflops * 100（peak 为 None 则 None）
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 5：🌟 Triton softmax 内核（10 分，可选实战，需要 GPU；未实现/无 GPU 时跳过）
# ═════════════════════════════════════════════════════════════════════

def triton_softmax(x):
    """
    用 Triton 实现按行 softmax（对照脚本 07；呼应 Part 1 的"减最大值防上溢"）。

    Args:
        x: torch.Tensor，形状 (n_rows, n_cols)，CUDA 设备

    Returns:
        y: torch.Tensor，同形状，每行 softmax 后的结果
           未实现 / 无 GPU 时返回 None（测试会跳过）

    提示（伪代码）：
        ⚠️ @triton.jit 内核必须定义在【模块顶层】（tl 要能从模块 globals 解析；
        定义在函数内部会 NameError: tl is not defined —— Triton 最常见初学坑之一）
        BLOCK = triton.next_power_of_2(n_cols)
        内核里：
            row = tl.load(base + tl.arange(0, BLOCK), mask=..., other=-inf)
            e = tl.exp(row - tl.max(row, axis=0))
            out = e / tl.sum(e, axis=0)
            tl.store(..., out, mask=...)
        启动：kernel[(n_rows,)](output, x, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK)
    """
    # TODO: 实现内核与包装函数；无 GPU 时直接 return None
    return None
