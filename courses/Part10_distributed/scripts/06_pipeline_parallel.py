#!/usr/bin/env python3
"""
Part 10 - 脚本 6（进阶）: 流水线并行 —— GPipe 式微批流水线
目标：把 4 层 GPT 切成前后两个 stage 放到 2 个进程上（**同一份权重按层切开**），
      用 m=4 个 micro-batch 填流水线，验证流水线 loss 与单进程整模型一致（~1e-6），
      并对照 bubble 公式 (p-1)/(m+p-1)。

跨 stage 的自动微分：两个自定义 autograd.Function（Megatron/DeepSpeed 流水线的最小内核）
  SendActivation：forward 发激活给下一 stage，backward 从下一 stage 收梯度
  RecvActivation：forward 从上一 stage 收激活，backward 把梯度发回去

运行：
  torchrun --standalone --nproc_per_node=2 06_pipeline_parallel.py   # 2 stage 流水线
  python 06_pipeline_parallel.py                                     # 单进程参照
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


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
        q = self.wq(h).view(B, T, self.h, -1).transpose(1, 2)
        k = self.wk(h).view(B, T, self.h, -1).transpose(1, 2)
        v = self.wv(h).view(B, T, self.h, -1).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        wei = wei.masked_fill(self.mask[:T, :T], float('-inf'))
        a = (F.softmax(wei, -1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + a
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def build_full_model(vocab, n_embed=256, n_head=4, ctx=128, n_layer=4):
    model = nn.Module()
    model.tok = nn.Embedding(vocab, n_embed)
    model.pos = nn.Embedding(ctx, n_embed)
    model.blocks = nn.ModuleList([Block(n_embed, n_head, ctx) for _ in range(n_layer)])
    model.ln = nn.LayerNorm(n_embed)
    model.head = nn.Linear(n_embed, vocab)
    model.ctx = ctx
    return model


def full_forward(model, x):
    x = model.tok(x) + model.pos(torch.arange(x.shape[1], device=x.device))
    for b in model.blocks:
        x = b(x)
    return model.head(model.ln(x))


# ⚠️ 实测坑（4090 + 4090D 混合机器上）：NCCL 的 send/recv 会互相等待卡死。
# 工程解法：点对点单独建一个 gloo 进程组，张量过 CPU 中转；集合通信仍走 NCCL。
P2P_GROUP = None


class SendActivation(torch.autograd.Function):
    """forward: 发给 dst（经 gloo/CPU）；backward: 从 dst 收梯度。"""

    @staticmethod
    def forward(ctx, x, dst, tag):
        ctx.dst, ctx.tag = dst, tag
        ctx.was_cuda = x.is_cuda
        dist.send(x.detach().cpu().contiguous(), dst=dst, tag=tag, group=P2P_GROUP)
        return x

    @staticmethod
    def backward(ctx, grad):
        buf = torch.empty(grad.shape, dtype=grad.dtype)      # gloo 用 CPU 张量
        dist.recv(buf, src=ctx.dst, tag=ctx.tag + 1000, group=P2P_GROUP)
        return (buf.to(grad.device) if ctx.was_cuda else buf), None, None


class RecvActivation(torch.autograd.Function):
    """forward: 从 src 收激活（经 gloo/CPU）；backward: 把梯度发回 src。"""

    @staticmethod
    def forward(ctx, src, tag, shape, dtype, device):
        ctx.src, ctx.tag = src, tag
        ctx.device = device
        buf = torch.empty(shape, dtype=dtype)                 # CPU 收
        dist.recv(buf, src=src, tag=tag, group=P2P_GROUP)
        return buf.to(device)

    @staticmethod
    def backward(ctx, grad):
        dist.send(grad.detach().cpu().contiguous(), dst=ctx.src,
                  tag=ctx.tag + 1000, group=P2P_GROUP)
        return grad, None, None, None, None


def setup():
    global P2P_GROUP
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        os.environ.update({"RANK": "0", "WORLD_SIZE": "1",
                           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"})
        dist.init_process_group(backend="gloo")
    if dist.get_backend() == "nccl":
        P2P_GROUP = dist.new_group(backend="gloo")   # 点对点专用 gloo 组
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


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
    seq, n_micro = 128, 4
    starts = range(0, 20000, seq)
    X = torch.stack([raw[i:i + seq] for i in starts])[:16]
    Y = torch.stack([raw[i + 1:i + seq + 1] for i in starts])[:16]

    torch.manual_seed(1337)          # 所有 rank 构造同一份权重（"按层切"的前提）
    vocab, n_embed, n_layer = len(chars), 256, 4
    model = build_full_model(vocab).to(device)
    split = n_layer // 2

    p = min(world, 2)
    log(f"\n═══ 流水线并行（p={p} stage × m={n_micro} micro-batch, 4 层按层切两半）═══")

    if p == 2:
        # 每个 rank 只优化自己 stage 的层（真实流水线里另一个 stage 的层根本不在本机）
        if rank == 0:
            my_params = ([p_ for p_ in model.tok.parameters()] +
                         [p_ for p_ in model.pos.parameters()] +
                         [p_ for b in model.blocks[:split] for p_ in b.parameters()])
        else:
            my_params = ([p_ for b in model.blocks[split:] for p_ in b.parameters()] +
                         [p_ for p_ in model.ln.parameters()] +
                         [p_ for p_ in model.head.parameters()])
        opt = torch.optim.AdamW(my_params, lr=3e-3)

        micro_X = list(X.chunk(n_micro))
        micro_Y = list(Y.chunk(n_micro))
        mb_b = micro_X[0].shape[0]
        losses = []
        for mb in range(n_micro):
            tag = 100 + mb
            if rank == 0:                                   # stage 0
                act = model.tok(micro_X[mb].to(device)) + \
                    model.pos(torch.arange(seq, device=device))
                for b in model.blocks[:split]:
                    act = b(act)
                SendActivation.apply(act, 1, tag)
            else:                                           # stage 1
                act = RecvActivation.apply(0, tag, (mb_b, seq, n_embed),
                                           torch.float32, device)
                for b in model.blocks[split:]:
                    act = b(act)
                logits = model.head(model.ln(act))
                losses.append(F.cross_entropy(
                    logits.reshape(-1, vocab), micro_Y[mb].to(device).reshape(-1)))
        if rank == 1:
            for mb in reversed(range(n_micro)):             # GPipe：backward 与 forward 反序
                (losses[mb] / n_micro).backward()
        # loss 汇总：只有最后 stage 有 loss，直接透传（⚠️ 不能 /world —— rank0 贡献是 0，
        # 除以 world 会把 loss 恰好减半，数字看起来"合理"实则错了 —— 我们真踩过）
        t = torch.tensor([sum(l.item() for l in losses) / max(len(losses), 1)],
                         device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        bubble = (p - 1) / (n_micro + p - 1)
        log(f"  流水线 loss = {t.item():.6f}")
        log(f"  bubble 公式 (p-1)/(m+p-1) = ({p}-1)/({n_micro}+{p}-1) = {bubble:.0%}"
            f"（m 越大气泡越小）")
        log(f"  对照：用 world=1 单进程跑本脚本，整模型 loss 应与此一致到 ~1e-6")
    else:
        loss = F.cross_entropy(full_forward(model, X.to(device)).reshape(-1, vocab),
                               Y.to(device).reshape(-1))
        loss.backward()
        log(f"  单进程整模型 loss = {loss.item():.6f}")
        log(f"  bubble 公式 (p-1)/(m+p-1)：p=2, m=4 → 20%；m=8 → 11%（微批越多气泡越小）")
        log(f"  对照：torchrun 2 卡跑本脚本，流水线 loss 应与此一致到 ~1e-6")

    dist.barrier()
    if is_root:
        print("""
  💡 观察点：
   - "同一份权重按层切"：两个 stage 共享刚才的 build_full_model —— 所以 loss 必须一致
   - GPipe 先做完全部 forward 再 backward；1F1B 交错执行，激活驻留从 m 个降到 p 个
   - stage 间梯度沿 Send/Recv 的 tag 配对回传 —— 点对点通信，与集合通信互补""")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
