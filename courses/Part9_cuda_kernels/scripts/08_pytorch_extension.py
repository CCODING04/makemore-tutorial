#!/usr/bin/env python3
"""
Part 9 - 脚本 08: PyTorch CUDA 扩展 —— 自己的算子接进 PyTorch
目标：把一个 CUDA 内核（polynomial activation: x^2 + x + 1，来自 cuda-course 09 章）
      用 torch.utils.cpp_extension.load() 现场编译，然后当普通 PyTorch 函数调用，
      与 torch 参考实现做正确性验证 + 速度对比。

对应 cuda-course: 09_PyTorch_Extensions/{polynomial_cuda.cu, setup.py, polynomial_activation.py}
运行（需要 GPU + nvcc，第一次编译约 1-2 分钟）:
    python 08_pytorch_extension.py
"""

import os
import torch
from torch.utils.cpp_extension import load

# CUDA 源码：和仓库的 polynomial_cuda.cu 相同，用 C 字符串内联写在脚本里
#（课程哲学：单脚本自包含可运行；正式项目请用独立 .cu + setup.py，见教程 04 章）
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 模板：scalar_t 会被 AT_DISPATCH 替换成 float / double
// __restrict__：承诺两个指针不重叠，编译器可以放心优化（教程 04 章有反例）
template <typename scalar_t>
__global__ void polynomial_activation_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = x[idx];
        output[idx] = val * val + val + 1;   // x^2 + x + 1
    }
}

// C++ 包装：张量进张量出，负责选 launch 配置 + 类型分发
torch::Tensor polynomial_activation_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int threads = 1024;
    int blocks = (x.numel() + threads - 1) / threads;

    // AT_DISPATCH_FLOATING_TYPES: 按 x 的 dtype 实例化对应模板（float/double）
    // ⚠️ 传 x.scalar_type()（返回 caffe2::TypeMeta 的 ScalarType，torch 1.8+ 稳定可用）。
    // 老教程常见的 x.type() 传法在 torch>=2.4 已无法编译（DeprecatedTypeProperties
    // 到 TypeMeta 的隐式转换被移除）——升级 torch 后扩展报编译错就是它。
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "polynomial_activation_cuda", ([&] {
        polynomial_activation_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            x.numel()
        );
    }));
    return output;
}
// 注：cuda-course 原版这里还有一个 PYBIND11_MODULE 块做 Python 绑定。
// 本脚本用 load_inline()，它会根据 functions=["polynomial_activation_cuda"]
// 自动生成绑定 —— 两种方式等价，原版写法见教程 04 章（setup.py 打包路线）。
"""


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA GPU required.")

    print("Compiling extension (first run ~1-2 min, then cached in ~/.cache/torch_extensions)...")

    # ── 多版本 CUDA 共存时的选择（本课机器：11.8 与 12.4 并存，互不覆盖）──
    # torch 的扩展编译器按 CUDA_HOME 找 nvcc；这里优先选 /usr/local/cuda-* 里最高的一个，
    # 只设置本进程的 CUDA_HOME，不动全局 PATH / /usr/local/cuda 软链接。
    import glob
    import re
    import shutil
    import subprocess
    homes = sorted(glob.glob("/usr/local/cuda-[0-9]*"))
    cuda_home = homes[-1] if homes else os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if homes:
        os.environ["CUDA_HOME"] = cuda_home
        # ⚠️ 坑点：torch 在【import 时】就把 CUDA_HOME 解析并缓存成了模块全局
        # （cpp_extension.py: CUDA_HOME = _find_cuda_home()），之后再改环境变量不会生效，
        # 必须把模块全局也覆盖掉。这就是"只设 os.environ 不起作用"的原因。
        import torch.utils.cpp_extension as _cpp_ext
        _cpp_ext.CUDA_HOME = cuda_home
    ver_out = subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "--version"],
                             capture_output=True, text=True).stdout
    cuda_major = int(re.search(r"release (\d+)\.", ver_out).group(1))

    # ⚠️ 坑点（与本课 Makefile 同款）：CUDA 对宿主 gcc 有版本上限——
    # CUDA 11.x 最高吃 gcc-11（报 "unsupported GNU version" 时用 -ccbin 指定旧版）；
    # CUDA 12.1+ 原生支持 gcc-12/13，什么都不用加。
    extra_cuda_cflags = ["-O2"]
    if cuda_major == 11 and shutil.which("g++-11"):
        extra_cuda_cflags += ["-ccbin", "g++-11"]

    from torch.utils.cpp_extension import load_inline
    ext = load_inline(
        name="polynomial_activation_part9",   # 编译产物名（缓存 key 之一）
        # cpp_sources：声明；cuda_sources：实现（两个字符串会被分别写进 .cpp / .cu）
        cpp_sources="torch::Tensor polynomial_activation_cuda(torch::Tensor x);",
        cuda_sources=[CUDA_SRC],
        functions=["polynomial_activation_cuda"],   # 自动生成 pybind 绑定
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,                         # 改 True 看完整编译命令（排查 nvcc 版本坑用）
    )
    print(f"Compiled OK (CUDA_HOME={cuda_home}, nvcc {cuda_major}.x).\n")

    # ---- 正确性：CUDA 内核 vs 纯 torch 表达式（CPU/GPU 都能算） ----
    torch.manual_seed(42)
    x = torch.randn(1_000_000, device='cuda')
    ref = x * x + x + 1
    out = ext.polynomial_activation_cuda(x)
    torch.testing.assert_close(out, ref)
    print(f"[correct] matches x*x + x + 1 (n=1,000,000)")

    # ---- 速度：自定义内核 vs torch 融合写法 ----
    def bench(fn, iters=100):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t_cuda = bench(lambda: ext.polynomial_activation_cuda(x))
    t_eager = bench(lambda: x * x + x + 1)          # torch eager：3 个内核逐个跑
    print(f"[speed] custom CUDA : {t_cuda:.4f} ms  (1 fused kernel)")
    print(f"[speed] torch eager : {t_eager:.4f} ms  (temp tensor + 3 kernel launches)")

    # ---- 梯度：为什么这是"自定义算子"而不是"自定义函数"？----
    # load() 编译的是"无 autograd 支持"的裸函数 —— 反向传播不会自动得到梯度。
    # 工程做法二选一：① 数学上可分解时，用 torch.autograd.Function 包一层手写 backward；
    #                ② 用 torch.compile / Triton autograd 友好路径。
    # 演示 ①：x^2+x+1 的导数是 2x+1，手写 backward 一行搞定。
    print("\n[demo] wrapping in torch.autograd.Function with handwritten backward:")

    class PolynomialActivation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return ext.polynomial_activation_cuda(x)

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out * (2 * x + 1)       # d/dx (x^2+x+1) = 2x+1

    xa = torch.randn(8, device='cuda', requires_grad=True)
    PolynomialActivation.apply(xa).sum().backward()
    torch.testing.assert_close(xa.grad, 2 * xa + 1)
    print("[grad] custom backward matches (2x + 1)")

    print("\nDone. PyTorch 自带算子（以及 FlashAttention 这类扩展）就是这样发布的："
          "\nCUDA 内核 + C++ 包装 + pybind11 绑定。你现在会从头到尾走一遍了。")


if __name__ == '__main__':
    main()
