#!/usr/bin/env python3
"""
04_stability_speed.py —— 04 章「稳定与提速」配套脚本

= 03_experiment_lab.py 的全部代码，叠加本章零件：
  ① AMP 三格式：fp32 / fp16+GradScaler / bf16（amp_ctx / make_scaler）
  ② 梯度裁剪 clip_grad_norm_、梯度累积（先除 accum）
  ③ 余弦调度 get_lr（无 warmup，1.0×→0.1×）
  ④ 训练循环升级为可复用引擎 train_model()——04/06 章的实验全部复用它
  ⑤ 双卡 DDP 演示（DistributedSampler 三个关键参数）

用法：
  python 04_stability_speed.py                                # AMP 三连对比，约 1.5 分钟
  torchrun --nproc_per_node=2 04_stability_speed.py --ddp     # 双卡 DistributedSampler 演示
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if not torch.cuda.is_available():
    torch.set_num_threads(8)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'dataset'))
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'temp', 'out'))
LOG_DIR = os.path.join(OUT_DIR, 'logs')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

STEPS = int(os.environ.get('P7_STEPS', 120))   # 对比实验用短训
BATCH = 16
ACCUM = 2
LR = 5e-4
SEQ = 340
MAX_SAMPLES = 20000
SEED = 1337
# CPU toy 档：无 GPU 只保流程可跑通（数字与 GPU 档不可比）
HIDDEN, N_LAYERS, N_HEADS, MAX_POS = (64, 2, 4, 1024) if DEVICE == 'cpu' else (512, 8, 8, 1024)
if DEVICE == 'cpu':
    MAX_SAMPLES = min(MAX_SAMPLES, 3000)


# ══════════════════════════════════════════════════════════════════
# ↓↓↓ 01/03 章代码（与 03_experiment_lab.py 相同）↓↓↓
# ══════════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))


class Attention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_heads, self.hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        scores = scores + torch.triu(torch.full((T, T), float('-inf'),
                                            device=x.device), 1)
        out = F.softmax(scores.float(), dim=-1).type_as(q) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class FFN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden, bias=False)
        self.down = nn.Linear(4 * hidden, hidden, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class Block(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = Attention(hidden, n_heads)
        self.mlp = FFN(hidden)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMindForCausalLM(nn.Module):
    def __init__(self, vocab, hidden, n_layers, n_heads, max_pos):
        super().__init__()
        self.max_pos = max_pos
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(max_pos, hidden)
        self.layers = nn.ModuleList(Block(hidden, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        T = input_ids.size(1)
        x = x + self.pos_emb(torch.arange(T, device=input_ids.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            x_, y_ = logits[:, :-1, :], labels[:, 1:]
            loss = F.cross_entropy(x_.reshape(-1, x_.size(-1)),
                                   y_.reshape(-1), ignore_index=-100)
        return logits, loss


def build_dataset(tok, seq, max_samples):
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    data, truncated = [], 0
    path = os.path.join(DATA_DIR, 'pretrain_t2t_mini_sample.jsonl')
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, 'pretrain_t2t_mini.jsonl')
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                text = json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tok(text, add_special_tokens=False, max_length=seq - 2, truncation=True).input_ids
            truncated += (len(ids) == seq - 2)
            ids = [bos] + ids + [eos]
            ids += [pad] * (seq - len(ids))
            labels = [t if t != pad else -100 for t in ids]
            data.append((torch.tensor(ids), torch.tensor(labels)))
    print(f"  数据: {len(data):,} 条 × seq={seq}（截断 {truncated:,} 条 = {truncated/len(data)*100:.0f}%）")
    return data


def split_train_val(data, val_ratio=0.05):
    n_val = max(1, int(len(data) * val_ratio))
    return data[:-n_val], data[-n_val:]


@torch.no_grad()
def evaluate(model, data, batch=32):
    was_training = model.training
    model.eval()
    losses = []
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        xb = torch.stack([x for x, _ in chunk]).to(DEVICE)
        yb = torch.stack([y for _, y in chunk]).to(DEVICE)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    if was_training:
        model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(loss)


class MetricLogger:
    FIELDS = ['step', 'train_loss', 'lr', 'tok_s', 'grad_scale', 'val_loss', 'val_ppl']

    def __init__(self, run_name):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.csv_path = os.path.join(LOG_DIR, f'{run_name}.csv')
        self._csv = open(self.csv_path, 'w', encoding='utf-8')
        self._csv.write(','.join(self.FIELDS) + '\n')
        self._tb = None
        if os.environ.get('P7_NO_TB') != '1':
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(os.path.join(LOG_DIR, 'tb', run_name))
            except ImportError:
                pass

    def log(self, step, **tags):
        row = {'step': step, **tags}
        self._csv.write(','.join(
            f'{row[k]:.6g}' if isinstance(row.get(k), float) else str(row.get(k, ''))
            for k in self.FIELDS) + '\n')
        if self._tb is not None:
            for k, v in tags.items():
                self._tb.add_scalar(k, v, step)

    def close(self):
        self._csv.close()
        if self._tb is not None:
            self._tb.close()


# ══════════════════════════════════════════════════════════════════
# ↑↑↑ 03 章代码结束 ｜ 本章新零件 ↓↑↑
# ══════════════════════════════════════════════════════════════════

def amp_ctx(amp):
    """AMP 上下文：fp16 需配 GradScaler；bf16 指数位同 fp32 不需要；'off' 即 fp32。"""
    if DEVICE != 'cuda' or amp == 'off':
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if amp == 'fp16' else torch.bfloat16)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_lr(current_step, total_steps, lr):
    """余弦调度（无 warmup）：t/T=0 → 1.0×；0.5 → 0.55×；1 → 0.1×（收在 0.1×lr，不是 0）。"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def train_model(model, run_name, train_data, steps, amp='bf16',
                val_data=None, eval_every=0, sched=True, seed=SEED, quiet=False):
    """训练引擎：03 章的循环 + 本章三件（AMP/裁剪/累积）+ 调度。04/06 章实验全部复用。"""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = make_scaler(enabled=(amp == 'fp16' and DEVICE == 'cuda'))
    logger = MetricLogger(run_name)
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def batch():
        idx = torch.randint(0, len(train_data), (BATCH,))
        xb = torch.stack([train_data[i][0] for i in idx]).to(DEVICE)
        yb = torch.stack([train_data[i][1] for i in idx]).to(DEVICE)
        return xb, yb

    t0, losses = time.time(), []
    if not quiet:
        tag = {'off': 'fp32', 'fp16': 'fp16+GradScaler', 'bf16': 'bf16'}[amp if DEVICE == 'cuda' else 'off']
        print(f"  训练: steps={steps} lr={LR} batch={BATCH}×accum{ACCUM} amp={tag} clip=1.0 "
              f"{'余弦→0.1×' if sched else '恒定 lr'}")
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(ACCUM):                                       # 累积：穷人的大 batch
            xb, yb = batch()
            with amp_ctx(amp):
                _, loss = model(xb, yb)
                loss = loss / ACCUM                                  # 先除：累积的才是均值
            scaler.scale(loss).backward()                            # fp16: 放大护梯度
            accum_loss += loss.item()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)                               # 裁剪前先还原真实梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # 全局范数拉回 1 以内
        lr_now = get_lr(step, steps, LR) if sched else LR
        for g in optimizer.param_groups:
            g['lr'] = lr_now
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)
        tok_s = BATCH * SEQ * ACCUM / max((time.time() - t0) / (step + 1), 1e-9)
        tags = dict(train_loss=accum_loss, lr=lr_now, tok_s=tok_s)
        if scaler.is_enabled() and step % 20 == 0:
            tags['grad_scale'] = scaler.get_scale()                  # 保险丝读数，CSV 可查
        logger.log(step, **tags)
        if not quiet and (step % 20 == 0 or step == steps - 1):
            print(f"  step {step:4d}/{steps}: loss {accum_loss:.4f} lr {lr_now:.2e} tok/s {tok_s:.0f}")
        if eval_every and val_data and (step % eval_every == 0 or step == steps - 1):
            vl, vp = evaluate(model, val_data)
            logger.log(step, val_loss=vl, val_ppl=vp)
            if not quiet:
                print(f"    ↳ 验证集: loss {vl:.4f}  ppl {vp:.2f}")
    logger.close()
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == 'cuda' else 0
    final = sum(losses[-5:]) / 5
    if not quiet:
        print(f"  📉 loss {losses[0]:.4f} → {final:.4f} ｜ 峰值显存 {peak:.1f} GB")
    return final, peak


def amp_shootout(tok):
    """同模型、同数据、同 seed，只换数值格式——精度换速度的对照实验。"""
    data = build_dataset(tok, SEQ, MAX_SAMPLES)
    train_data, val_data = split_train_val(data)
    results = {}
    for amp in ('off', 'fp16', 'bf16'):
        tag = {'off': 'fp32', 'fp16': 'fp16+scaler', 'bf16': 'bf16'}[amp]
        model = MiniMindForCausalLM(vocab=len(tok), hidden=HIDDEN, n_layers=N_LAYERS, n_heads=N_HEADS, max_pos=MAX_POS).to(DEVICE)
        print(f"  ── {tag} ──")
        loss, peak = train_model(model, f's3_amp_{tag}', train_data, STEPS,
                                 amp=amp, val_data=val_data, sched=False, quiet=True)
        results[tag] = (loss, peak)
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print("  ────────────────────────────────")
    print("  格式            末段 loss   峰值显存")
    for tag, (loss, peak) in results.items():
        print(f"  {tag:<14}  {loss:.4f}    {peak:.1f} GB")
    print("  ↳ 稳态 tok/s 见 logs/s3_amp_*.csv 的 tok_s 列（fp32≈126k，fp16/bf16≈228k，1.8×）")
    print("  ↳ loss 三者同水平；fp16 的 GradScaler 是保险丝——本实验未触发溢出（scale 恒 65536）")


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def ddp_demo(tok):
    """双卡 DDP：DistributedSampler 为什么是标配——shuffle/drop_last/set_epoch 各防什么。"""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    dev = f'cuda:{local}'

    data = build_dataset(tok, SEQ, 4000)
    dataset = ListDataset(data)
    model = MiniMindForCausalLM(vocab=len(tok), hidden=HIDDEN, n_layers=N_LAYERS, n_heads=N_HEADS, max_pos=MAX_POS).to(dev)
    ddp_model = DDP(model, device_ids=[local])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LR)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True)     # 三个关键参数，04 章正文逐个讲
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)
    print(f"  [rank {rank}] epoch0 前 3 个样本下标: {list(iter(sampler))[:3]}", flush=True)
    if rank == 0 and world > 1:
        other = list(iter(DistributedSampler(dataset, num_replicas=world, rank=1,
                                             shuffle=True, drop_last=True)))[:3]
        print(f"  [rank 0 视角] rank1 的前 3 个: {other} → 两张卡看到不同数据，各训一半", flush=True)
    sampler.set_epoch(1)
    print(f"  [rank {rank}] set_epoch(1) 后前 3 个: {list(iter(sampler))[:3]}", flush=True)

    steps, t0, seen = 0, time.time(), 0
    ddp_model.train()
    done = False
    while not done:
        sampler.set_epoch(steps)
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = ddp_model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            optimizer.step()
            seen += xb.size(0)
            steps += 1
            if steps >= 120:
                done = True
                break
    dt = time.time() - t0
    print(f"  [rank {rank}] {steps} 步 × batch {BATCH} ｜ {dt:.1f}s ｜ 本卡 tok/s {seen*SEQ/dt:.0f}",
          flush=True)
    dist.barrier()
    if rank == 0:
        print("  ↳ 注意：双卡合计 tok/s 并没有 ×2——26M 小模型的梯度 all-reduce 通信开销"
              "占比很大，模型越大 DDP 越划算（04 章正文展开这笔账）")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ddp', action='store_true', help='双卡 DDP 演示（用 torchrun 启动）')
    args = ap.parse_args()
    torch.manual_seed(SEED)

    if args.ddp:
        world = int(os.environ.get('WORLD_SIZE', 1))
        if world < 2:
            raise SystemExit("单进程没有 DDP 可演示。请用：torchrun --nproc_per_node=2 "
                             "04_stability_speed.py --ddp")
        print(f"═══ 04 · DDP 演示 ═══  world={world}")
        ddp_demo(load_tokenizer())
        return

    print(f"═══ 04 · 稳定与提速 ═══  device={DEVICE}")
    tok = load_tokenizer()
    amp_shootout(tok)
    print("  ── 梯度累积的算术 ──")
    print(f"  本章 batch {BATCH} × accum {ACCUM} = 有效 {BATCH*ACCUM}；官方 batch 32 × accum 8 = 有效 256")
    print("  显存只付小 batch 激活的钱，梯度噪声却按大 batch 压——等效大 batch 是「穷人法宝」")
    print("  ── 双卡 DistributedSampler 演示 ──")
    print("  torchrun --nproc_per_node=2 04_stability_speed.py --ddp")


if __name__ == '__main__':
    main()
