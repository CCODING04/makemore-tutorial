#!/usr/bin/env python3
"""
Part 10 - 脚本 5（进阶）: 张量并行 —— 手写 Megatron 式 MLP
目标：把一个 MLP 的权重切开到 2 个进程上算（列并行 + 行并行），验证：
      ① 前向输出与单卡稠密计算完全一致（max_err < 1e-5）
      ② 梯度也一致（all-reduce 之后）
这就是 Megatron-LM 张量并行的最小骨架：f/g 两个共轭通信算子。

原理（对应教程 04 章）：
  Y = gelu(X @ A^T) @ B        （A: out×in 按行切 = 列并行；B: in×out 按列切 = 行并行）
  forward: 各 rank 算自己的 H_r = gelu(X@A_r^T)（无通信）→ Y_r = H_r@B_r → all_reduce 求和
  backward: autograd 算各 rank 局部 dX_r → all_reduce 求和 = 完整 dX
  （Megatron 的 f = forward恒等/backward all-reduce；g = forward all-reduce/backward 恒等）

运行：
  torchrun --standalone --nproc_per_node=2 05_tensor_parallel.py   # 分片验证（推荐）
  python 05_tensor_parallel.py                                     # 单进程 = 稠密参照
"""

import os
import sys
import torch
import torch.distributed as dist

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(42)


class AllReduceSum(torch.autograd.Function):
    """forward 求和（g 算子），backward 恒等 —— 数学上等价于"先求和再反传"。"""

    @staticmethod
    def forward(ctx, x):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad  # g 的 backward 是恒等


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        os.environ.update({"RANK": "0", "WORLD_SIZE": "1",
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"})
        dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None
    is_root = rank == 0

    # 全量权重：所有 rank 用同一个种子生成 → 切片前大家"看到"同一个 A、B
    IN, OUT, BATCH, SEQ = 256, 512, 8, 16
    gen = torch.Generator().manual_seed(7)
    A = torch.randn(OUT, IN, generator=gen) / IN ** 0.5     # 列并行：按 dim0（行）切
    B = torch.randn(OUT, IN, generator=gen) / OUT ** 0.5    # 行并行：按 dim1（列）切
    # （B 语义是第二层权重 W2 的"转置存法"：Y = H @ B，H:(·,OUT), B:(OUT,IN) → Y:(·,IN)）
    X = torch.randn(BATCH, SEQ, IN, generator=gen)
    A, B, X = A.to(dev), B.to(dev), X.to(dev)

    chunk = OUT // world
    lo, hi = rank * chunk, (rank + 1) * chunk
    # 列并行：W1=A 按输出维（dim0）切；行并行：B 按输入维 H4（也是 dim0，因为 Y=H@B）切
    # → A_r:(chunk,IN)，B_r:(chunk,IN)，H_r:(·,chunk)，Y_r:(·,IN)=完整输出的一部分和
    A_r, B_r = A[lo:hi, :], B[lo:hi, :]

    # ── 张量并行前向 + 反向 ──
    X_tp = X.clone().requires_grad_(True)
    H_r = torch.nn.functional.gelu(X_tp @ A_r.T)     # 无通信（f：forward 恒等）
    Y_r = H_r @ B_r
    Y_tp = AllReduceSum.apply(Y_r)                    # g：forward all_reduce
    loss_tp = Y_tp.pow(2).mean()
    loss_tp.backward()

    # ── 单卡稠密参照（所有 rank 都本地算一份，root 负责对比）──
    X_dense = X.clone().requires_grad_(True)
    Y_dense = torch.nn.functional.gelu(X_dense @ A.T) @ B
    loss_dense = Y_dense.pow(2).mean()
    loss_dense.backward()

    if world > 1:
        # f 的 backward = all_reduce：把各 rank 局部 dX 求和成完整 dX
        dist.all_reduce(X_tp.grad, op=dist.ReduceOp.SUM)

    if is_root:
        fwd_err = (Y_tp - Y_dense).abs().max().item()
        grad_err = (X_tp.grad - X_dense.grad).abs().max().item()
        print("═══ 张量并行（Megatron 式 MLP）═══")
        print(f"  world_size={world}, 隐藏维 H4={OUT} 按 2 切: A、B 各取 [{lo}:{hi}] 行")
        print(f"  前向 max |Y_tp - Y_dense| = {fwd_err:.2e}  → {'✅' if fwd_err < 1e-5 else '❌'}")
        print(f"  梯度 max |dX_tp - dX_dense| = {grad_err:.2e}  → {'✅' if grad_err < 1e-5 else '❌'}")
        print(f"  loss: TP={loss_tp.item():.6f} vs dense={loss_dense.item():.6f}")
        print(f"""
  通信记账（每层 MLP，world=N）：
    forward : 1 次 all-reduce（Y_r 求和）     ← g
    backward: 1 次 all-reduce（dX_r 求和）    ← f
  这就是 Megatron 论文里"每层每方向恰好一次 all-reduce"的来历；
  Attention 同理：QKV 投影按头切（列并行），输出投影行并行。
  更进一步：all-reduce 还能拆成 reduce-scatter + all-gather（sequence parallelism）。""")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
