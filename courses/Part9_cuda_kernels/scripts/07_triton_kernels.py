#!/usr/bin/env python3
"""
Part 9 - 脚本 07: Triton 内核 —— vector add 与 softmax
目标：用 Triton 写两个和前 6 个脚本同样的内核，逐行对照"CUDA 写法 vs Triton 写法"，
      并和 PyTorch 官方实现比对正确性、测速度。

对应 cuda-course: 08_Triton/01_vec_add.py + 02_softmax.py
运行（需要 GPU + torch 自带的 triton）:
    python 07_triton_kernels.py
"""

import torch
import triton
import triton.language as tl


# ============ 内核 1: vector add（对照脚本 01 的 CUDA 版） ============
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,          # 编译期常量：每个"程序"处理多少元素
):
    # CUDA:  int i = blockIdx.x * blockDim.x + threadIdx.x;
    # Triton: 一个 program 处理一整块（BLOCK_SIZE 个）元素，块内向量式操作
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # 这块元素的 [start, start+BLOCK) 下标

    mask = offsets < n_elements        # CUDA:  if (i < n)  —— 但这里是"整块一起"判断
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    # 启动网格：多少个 program（类似 CUDA 的 block 数），cdiv = 向上取整
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============ 内核 2: softmax（对照 Part 1 的 softmax / Part 8 生成时用到的） ============
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一行（一行 = 一个 token 的 logits）
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 整行一次载入 SRAM；BLOCK_SIZE 是 2 的幂 > n_cols，越界位置填 -inf
    row = tl.load(row_start_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))

    # 数值稳定技巧：先减最大值再 exp（Part 1 教程里讲过为什么：防 exp 上溢）
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(out_row_start_ptr + col_offsets, softmax_output, mask=col_offsets < n_cols)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](output, x, x.stride(0), output.stride(0), n_cols,
                              BLOCK_SIZE=BLOCK_SIZE)
    return output


# ============ 验证 + 基准 ============
def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA GPU required. Part 9 Python scripts need a GPU "
                         "(CPU readers: still do assignment 9 exercises 1-4).")

    torch.manual_seed(42)

    # ---- vector add：正确性 ----
    n = 1_000_000
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    torch.testing.assert_close(triton_add(x, y), x + y)
    print(f"[vecadd] correct vs torch (n={n:,})")

    # ---- vector add：速度（和脚本 01 的 C 版对照：同样是"带宽受限"操作） ----
    def bench(fn, iters=50):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    t_triton = bench(lambda: triton_add(x, y))
    t_torch = bench(lambda: x + y)
    bw = 3 * n * 4 / (t_triton / 1e3) / 1e12   # 读2写1个 float32 -> 有效带宽 TB/s
    print(f"[vecadd] triton {t_triton:.3f} ms vs torch {t_torch:.3f} ms "
          f"(effective BW ~{bw:.2f} TB/s -> memory-bound, same as CUDA version)")

    # ---- softmax：正确性（对照 torch.softmax，Part 6/7/8 生成 logits 后都做这一步） ----
    logits = torch.randn(4096, 1024, device='cuda')
    torch.testing.assert_close(triton_softmax(logits), torch.softmax(logits, dim=1))
    print("[softmax] correct vs torch.softmax (rows=4096, cols=1024)")

    t_triton = bench(lambda: triton_softmax(logits))
    t_torch = bench(lambda: torch.softmax(logits, dim=1))
    print(f"[softmax] triton {t_triton:.3f} ms vs torch {t_torch:.3f} ms "
          f"(torch may win: cuDNN/ATen kernels are highly tuned; "
          f"try BLOCK_SIZE/shape for fun)")

    print("\nKey takeaway: Triton ~= CUDA performance for elementwise/reduction kernels"
          "\nwith 10x less code; CUDA gives you the last 10-20% on matmul-class kernels.")


if __name__ == '__main__':
    main()
