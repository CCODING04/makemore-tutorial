#!/usr/bin/env python3
"""
Part 10 - 脚本 1: 分布式 Hello World —— 进程组与集合通信原语
目标：建立分布式训练的心智模型（rank/world_size/进程组），亲手用四个最常用的
      集合通信原语（broadcast / all_reduce / all_gather / reduce_scatter），
      并用单进程数学验证每个原语的语义。

对应教程：tutorial/01_why_and_collectives.md

两种运行方式（本课所有脚本都兼容）：
  单进程（CPU 也行）： python 01_distributed_basics.py
  多进程（推荐 2 卡）： torchrun --standalone --nproc_per_node=2 01_distributed_basics.py
"""

import os
import sys
import torch
import torch.distributed as dist

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def setup():
    """初始化进程组：单进程走 gloo；torchrun 注入的环境变量自动被 env:// 读取。
    ⚠️ 常见报错 FAQ：
      - "Connection reset / timeout"：MASTER_PORT 被占 → 换 --master_port=29501
      - "NCCL error"：容器/驱动问题 → 先用 backend='gloo' 排除 NCCL 因素
      - 卡在 init 不动：某个 rank 忘了调用 collectives，其他 rank 在死等（集合通信是对齐的！）"""
    if "RANK" in os.environ:                      # torchrun 已注入 RANK/WORLD_SIZE/MASTER_*
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:                                         # 裸 python：伪装成 1 卡世界
        os.environ.update({"RANK": "0", "WORLD_SIZE": "1",
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"})
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()


def main():
    rank, world = setup()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    is_root = (rank == 0)

    if is_root:
        print("═══ Part 10 - 分布式基础：集合通信原语 ═══")
        print(f"  world_size={world}, 每个进程一个 rank（本机 {world} 个）")
        print("  核心心智模型：同一份代码被 N 个进程同时执行，各自拿一个 rank 号，")
        print("  靠'集合通信'交换数据 —— 没有主从消息传递，全是 N 个进程对齐的集体操作。\n")

    # ── 1) broadcast：rank0 的数据 → 所有人 ──────────────
    x = torch.arange(4, device=device, dtype=torch.float32)
    if not is_root:
        x = torch.zeros_like(x)                   # 非 root 先清零
    dist.broadcast(x, src=0)
    assert x.tolist() == [0, 1, 2, 3]
    if is_root:
        print(f"[1] broadcast(src=0): rank0 的 [0,1,2,3] → 所有人 = {x.tolist()}")

    # ── 2) all_reduce：所有 rank 各持一份，做规约后人人有结果 ──
    x = torch.full((4,), float(rank + 1), device=device)   # rank i 持有 [i+1,i+1,i+1,i+1]
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expect = sum(range(1, world + 1))              # 1+2+...+world
    assert x.tolist() == [expect] * 4
    if is_root:
        print(f"[2] all_reduce(SUM): rank i 持 i+1 → 人人得到 {x.tolist()}")

    grad = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad /= world
    if is_root:
        print(f"    all_reduce(SUM)/world = 梯度平均 {[round(v,1) for v in grad.tolist()]}")
        print("    ⭐ DDP 的梯度同步就是这个：allreduce 出来的默认是【平均】不是求和！")

    # ── 3) all_gather：人人有数据，拼出所有人的 ───────────
    local = torch.full((2,), float(rank), device=device)   # rank i 持 [i, i]
    out = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(out, local)
    if is_root:
        print(f"[3] all_gather: rank{rank} 持 {local.tolist()} → 收齐 = "
              f"{[t.tolist() for t in out]}")

    # ── 4) reduce_scatter：规约后"切片分发"（FSDP 的核心原语）──
    x = torch.arange(world * 2, device=device, dtype=torch.float32)  # 每人同持 [0..2W)
    x += rank                                       # 弄成各不相同
    y = torch.zeros(2, device=device)
    if world == 1:
        y = x.clone()          # 单卡世界：reduce_scatter 退化为"规约(自己)+切给自己"
    else:
        dist.reduce_scatter(y, list(x.chunk(world)), op=dist.ReduceOp.SUM)
    expect_seg = (torch.arange(world * 2, dtype=torch.float32) * world
                  + sum(range(world))).chunk(world)[rank].tolist()
    assert torch.allclose(y.cpu(), torch.tensor(expect_seg)), (y, expect_seg)
    if is_root:
        print(f"[4] reduce_scatter(SUM): 规约后按 rank 切片，rank{rank} 得 {y.tolist()}")
        print("    FSDP = reduce_scatter(梯度) + all_gather(参数)，见脚本 04。")

    dist.barrier()
    if is_root:
        print("\n═══ 全部原语语义验证通过 ✅ ═══")
        print("下一步：用 all_reduce 平均梯度做多卡训练 → 02_ddp_gpt.py")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
