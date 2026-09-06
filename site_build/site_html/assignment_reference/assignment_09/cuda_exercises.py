"""Part 9 作业参考答案：CUDA 内核编程（索引数学 / 访存账本 / GFLOPS / Triton）"""
import math

import triton
import triton.language as tl

# ⚠️ @triton.jit 内核必须定义在模块顶层（tl 要能从模块 globals 解析；
#    定义在函数内部会 NameError: tl is not defined —— Triton 最常见初学坑之一）


def global_index_1d(block_idx, thread_idx, block_size):
    """1D 全局线程索引：i = blockIdx.x * blockDim.x + threadIdx.x"""
    return block_idx * block_size + thread_idx


def global_index_2d(block_idx, thread_idx, block_size, n_cols):
    """2D：col/row 各按 1D 公式算一遍，再 row-major 展平 idx = row*n_cols + col"""
    bx, by = block_idx
    tx, ty = thread_idx
    bsx, bsy = block_size
    col = bx * bsx + tx
    row = by * bsy + ty
    return (row, col, row * n_cols + col)


def launch_config(n, block_size):
    """向上取整除法：(n + bs - 1) // bs；total 可能 > n（多余线程靠 if (i<n) 拦截）"""
    num_blocks = (n + block_size - 1) // block_size
    return (num_blocks, num_blocks * block_size)


def row_major_index(row, col, n_cols):
    """行主序：A[row][col] 存于 A[row * n_cols + col]"""
    return row * n_cols + col


def col_major_index(row, col, n_rows):
    """列主序（cuBLAS/Fortran）：A[row][col] 存于 A[col * n_rows + row]"""
    return col * n_rows + row


def matmul_cpu(A, B):
    """纯 Python 三重循环 matmul：C[i][j] = Σ_l A[i][l]*B[l][j]"""
    M, K, N = len(A), len(B), len(B[0])
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for l in range(K):
                s += A[i][l] * B[l][j]
            C[i][j] = s
    return C


def count_global_reads(M, N, K, tile):
    """naive: M*N*2*K；tiled: (M/T)*(N/T)*2*T*K"""
    return (M * N * 2 * K, (M // tile) * (N // tile) * 2 * tile * K)


def tiled_speedup_ratio(M, N, K, tile):
    naive_reads, tiled_reads = count_global_reads(M, N, K, tile)
    return naive_reads / tiled_reads


def gflops_report(time_ms, M, N, K, peak_gflops=None):
    """GFLOPS = 2MNK / 秒 / 1e9（乘加各算一次）"""
    flops = 2.0 * M * N * K
    gf = flops / (time_ms / 1000) / 1e9
    return {'flops': flops, 'gflops': gf,
            'pct_of_peak': None if peak_gflops is None else gf / peak_gflops * 100}


# @triton.jit 内核必须定义在【模块顶层】（tl 从模块 globals 解析）
@triton.jit
def _softmax_kernel(o_ptr, i_ptr, is_, os_, nc, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    co = tl.arange(0, BLOCK)
    row = tl.load(i_ptr + r * is_ + co, mask=co < nc, other=-float('inf'))
    e = tl.exp(row - tl.max(row, axis=0))          # 减最大值防上溢（Part 1 技巧）
    tl.store(o_ptr + r * os_ + co, e / tl.sum(e, axis=0), mask=co < nc)


def triton_softmax(x):
    """按行 softmax（Triton）。无 GPU 时返回 None（测试会跳过）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        n_rows, n_cols = x.shape
        BLOCK = triton.next_power_of_2(n_cols)
        out = torch.empty_like(x)
        _softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK=BLOCK)
        return out
    except Exception:
        return None
