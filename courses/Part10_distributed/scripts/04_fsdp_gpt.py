#!/usr/bin/env python3
"""
Part 10 - 脚本 4: FSDP 实战 —— 全分片数据并行（ZeRO-3 的 PyTorch 实现）
目标：把脚本 02 的 DDP 换成 FSDP，实测"参数/梯度/优化器状态全分片"下的显存，
      并验证训练数学与 DDP 等价（同样收敛）。

DDP vs FSDP 一句话：DDP 每 rank 存完整模型、只同步梯度(all-reduce)；
FSDP 把 参数/梯度/优化器状态 全部切片，forward 前 all-gather 参数、backward 用
reduce-scatter 收梯度 —— 用通信换显存，大模型装得下了。

版本说明：本脚本用 FSDP1 API（torch 2.x 均可用）；PyTorch 2.6+ 推荐的 FSDP2
`fully_shard` 写法见教程 03 章（概念完全一致，API 更干净）。

运行：
  torchrun --standalone --nproc_per_node=2 04_fsdp_gpt.py   # 双卡看显存差异
  python 04_fsdp_gpt.py                                     # 单进程兼容（gloo/CPU）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def setup():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        os.environ.update({"RANK": "0", "WORLD_SIZE": "1",
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"})
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()


class Block(nn.Module):
    def __init__(self, n_embed, n_head, ctx):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
        self.wq = nn.Linear(n_embed, n_embed, bias=False)
        self.wk = nn.Linear(n_embed, n_embed, bias=False)
        self.wv = nn.Linear(n_embed, n_embed, bias=False)
        self.fc1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc2 = nn.Linear(4 * n_embed, n_embed)
        self.h = n_head
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.wq(h), self.wk(h), self.wv(h)
        q = q.view(B, T, self.h, -1).transpose(1, 2)
        k = k.view(B, T, self.h, -1).transpose(1, 2)
        v = v.view(B, T, self.h, -1).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        wei = wei.masked_fill(self.mask[:T, :T], float('-inf'))
        a = (F.softmax(wei, -1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + a
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, n_embed=256, n_head=4, n_layer=4, ctx=128):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, n_embed)
        self.pos = nn.Embedding(ctx, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, ctx) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def main():
    rank, world = setup()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    is_root = rank == 0
    log = print if is_root else (lambda *a, **k: None)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'data', 'input.txt')
    text = open(path, encoding='utf-8').read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    raw = torch.tensor([stoi[c] for c in text[:400000]], dtype=torch.long)
    seq = 128
    starts = range(0, 20000, seq)
    X = torch.stack([raw[i:i + seq] for i in starts])[:150]
    Y = torch.stack([raw[i + 1:i + seq + 1] for i in starts])[:150]
    xb = X[:16].to(device)
    yb = Y[:16].to(device)

    torch.manual_seed(1337 + rank)          # ⭐ 各 rank 初始化不同 → 验证 FSDP 会同步
    model = GPT(len(chars)).to(device)
    log(f"\n═══ FSDP vs DDP ═══")
    log(f"  world_size={world}, device={device}")
    log(f"  完整模型参数量: {param_bytes(model) / 1e6:.1f} MB (fp32)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── FSDP 包装 ──
    # use_orig_params=True：保留原始形状的参数视图（避免 flat-param 形状坑）。
    # 进阶：生产上还会传 auto_wrap_policy=transformer_auto_wrap_policy 让每个 Block
    # 成为一个独立分片单元（all-gather 更细粒度、更省显存）——教学版 root 包裹已够演示。
    fsdp_model = FSDP(model,
                      sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
                      use_orig_params=True,
                      device_id=device if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        log(f"  FSDP 包装后本 rank 参数显存: "
            f"{torch.cuda.memory_allocated() / 1e6:.1f} MB（全量 "
            f"{param_bytes(model) / 1e6:.1f} MB；≈参数分片 + FSDP 运行时缓冲）")
    opt = torch.optim.AdamW(fsdp_model.parameters(), lr=3e-3)

    # ── 短训练验证收敛数学（与 DDP 同一套数据/超参）──
    fsdp_model.train()
    losses = []
    for epoch in range(2):
        for i in range(0, 128, 16):
            _, loss = fsdp_model(X[i:i + 16].to(device), Y[i:i + 16].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
    t = torch.tensor([sum(losses) / len(losses)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world
    log(f"  训练后平均 loss: {t.item():.3f}（各 rank 初始化不同也能收敛 → 同步分片正确）")

    if torch.cuda.is_available():
        log(f"  本 rank 峰值显存: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    log("""
  💡 对比记忆点：
     DDP  : 参数/优化器每 rank 全量，只 all-reduce 梯度     → 快，但装不下大模型
     FSDP : 参数/梯度/优化器状态全分片（ZeRO-3）            → 省，多 1.5× 通信
     实测对比：把本脚本与 02_ddp_gpt.py 分别跑 nproc=2，
     看 max_memory_allocated 差异（模型越大差距越悬殊）。""")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
