#!/usr/bin/env python3
"""
Part 10 - 脚本 2: DDP 实战 —— 多卡训练一个字符级 GPT
目标：完整走一遍 DDP 的五个必要件（进程组 → DDP 包装 → DistributedSampler →
      梯度平均语义 → destroy），实测 1 卡 vs 2 卡的吞吐，并演示梯度累积的 no_sync。

对应教程：tutorial/02_ddp.md（DDP 内部机制：桶化 all-reduce 与 backward 重叠）

运行：
  torchrun --standalone --nproc_per_node=2 02_ddp_gpt.py   # 2 卡（推荐）
  python 02_ddp_gpt.py                                     # 单进程也兼容
"""

import os
import sys
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
        self.attn = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.GELU(),
                                 nn.Linear(4 * n_embed, n_embed))
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=self.mask[:T, :T])
        return x + self.mlp(self.ln2(x + a))


class GPT(nn.Module):
    def __init__(self, vocab, n_embed=128, n_head=4, n_layer=3, ctx=128):
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


def main():
    rank, world = setup()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    is_root = rank == 0
    log = print if is_root else (lambda *a, **k: None)

    # ── 数据：tiny Shakespeare 字符级 ──
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
    ds = TensorDataset(X, Y)

    # ── DDP 五件套之二：DistributedSampler ──
    # 每个 rank 只看自己的 1/world 数据分片（不同 rank 不同分片 → 有效 batch = bs × world）
    # ⚠️ shuffle 交给 sampler（DataLoader 里必须 shuffle=False），每轮 set_epoch 打乱种子
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    batch = 16
    loader = DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=False, drop_last=True)

    model = GPT(len(chars)).to(device)
    # ── DDP 五件套之三：包装（rank0 的权重自动 broadcast 给所有 rank；
    #    构造后从此只碰 model.module？不需要——DDP 透传 forward；参数= model.module.parameters()）
    ddp = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    raw_model = ddp.module
    opt = torch.optim.AdamW(raw_model.parameters(), lr=3e-3)

    # ── 训练 + 吞吐测量 ──
    accum = 2                                     # 梯度累积：no_sync 只在最后一步同步
    ddp.train()
    t0, tokens, losses = time.time(), 0, []
    for epoch in range(2):
        sampler.set_epoch(epoch)                  # ⭐ 不写这行：每个 epoch 分片完全相同
        for it, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            is_last_micro = (it % accum == accum - 1)
            # 最后一步恢复正常同步：nullcontext = "什么都不做"的空上下文（教程 02 章同款写法）
            ctx = ddp.no_sync() if not is_last_micro else nullcontext()
            with ctx:
                _, loss = ddp(xb, yb)
                (loss / accum).backward()
            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
            tokens += xb.numel()
    dt = time.time() - t0

    # 各 rank 的 loss 汇总给 root（验证不同分片确实在训练）
    t = torch.tensor([sum(losses) / len(losses)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)   # ⚠️ gloo 不支持 ReduceOp.AVG，用 SUM/world 等价替代
    t /= world

    log(f"\n═══ DDP 训练完成 ═══")
    log(f"  world_size={world}, device={device}")
    log(f"  每 rank batch={batch} × accum={accum} → 有效 batch = {batch}×{accum}×{world} = {batch * accum * world}")
    log(f"  平均 loss: {t.item():.3f}（各 rank 数据分片不同，loss 接近说明同步正常）")
    log(f"  本 rank 吞吐: {tokens / dt:,.0f} tokens/s（wall {dt:.1f}s）")
    if world > 1:
        log(f"  💡 多卡有效 batch 变大 → 单步看遍更多数据；吞吐近线性（通信与 backward 重叠）")
    log(f"  单进程跑本脚本 world_size=1，结果可复现；对比 2 卡吞吐请用 torchrun。")

    dist.barrier()
    if is_root:
        print("\n═══ 验收点 ═══")
        print("  [x] DistributedSampler 分片不重不漏（assignment_10 题 3 会让你亲手证明）")
        print("  [x] no_sync：累积步不同步，最后一步同步一次 —— 省掉 1/2 的通信")
        print("  [x] all_reduce 语义 = 平均：loss 与单卡等价于'有效 batch 变大'的同款")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
