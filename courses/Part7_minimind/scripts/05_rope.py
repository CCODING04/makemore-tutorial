#!/usr/bin/env python3
"""
05_rope.py —— 05 章「位置编码进化」配套脚本

= 04_stability_speed.py 的全部代码，叠加本章零件：
  ⓪ 数学性质三连检：旋转保范数、内积只依赖相对位置（平移不变）、θ 频率分工
  ① Attention 增加 pos 开关：'learned'（02 章的位置表）| 'rope'（旋转位置编码）
  ② RoPE 实现：precompute_cos_sin 角度表 + rotate_half 实数拼接技巧
  ③ 外推评估集构造 build_extrap_eval（专挑比训练长度长的文本）
  ④ 实验①：learned PE vs RoPE——同 seed 各训一次，比 ppl@340 与外推 ppl@512

注意：本文件里的 Attention 还没有 GQA 和 QK-Norm——那是 06 章的零件。

用法：
  python 05_rope.py                    # 两组训练 + 外推评估，约 4 分钟（RTX 4090）
  P7_STEPS=30 python 05_rope.py # 冒烟模式（数字无意义，只验流程）
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

STEPS = int(os.environ.get('P7_STEPS', 300))
BATCH = 16
ACCUM = 2
LR = 5e-4
SEQ = 340
MAX_SAMPLES = 20000
SEED = 1337
HIDDEN, N_LAYERS, N_HEADS, MAX_POS, THETA = 512, 8, 8, 1024, 1e6
# CPU toy 档：无 GPU 只保流程可跑通（数字与 GPU 档不可比）
if DEVICE == 'cpu':
    HIDDEN, N_LAYERS, N_HEADS = 64, 2, 4
    MAX_SAMPLES = min(MAX_SAMPLES, 3000)


# ══════════════════════════════════════════════════════════════════
# ↓↓↓ 02-04 章代码（与 04_stability_speed.py 相同，一处例外：模型增加 pos 开关）↓↓↓
# ══════════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))


def precompute_cos_sin(head_dim, max_seq_len, theta):
    """【05 章新增】RoPE 角度表。第 i 对频率 θ^(-2i/d)：低维高频管局部，高维低频管远距离。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), inv_freq)   # (max_pos, d/2)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)     # 拼两份 → head_dim 宽
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return cos, sin


def rotate_half(x):
    """【05 章新增】(x1, x2) → (−x2, x1)：32 个二维旋转的"sin 部分"，实数拼接写法。"""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    """【05 章修改】pos='learned'：什么都不做（位置在模型层加表）；pos='rope'：旋转 q/k。"""
    def __init__(self, hidden, n_heads, pos='learned', max_pos=MAX_POS, theta=THETA):
        super().__init__()
        self.pos = pos
        self.n_heads, self.hd = n_heads, hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        if pos == 'rope':
            cos, sin = precompute_cos_sin(self.hd, max_pos, theta)
            self.register_buffer('rope_cos', cos, persistent=False)   # 现算的表，不进权重文件
            self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_heads, self.hd)
        if self.pos == 'rope':                                    # 05 章：位置旋进 q/k
            cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(2)     # (1,T,1,hd) 广播
            sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
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
    def __init__(self, hidden, n_heads, pos='learned'):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = Attention(hidden, n_heads, pos=pos)
        self.mlp = FFN(hidden)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMindForCausalLM(nn.Module):
    """【05 章修改】pos='learned'：+位置表（02 章原样）；pos='rope'：表退役，旋转接管。"""
    def __init__(self, vocab, hidden, n_layers, n_heads, max_pos, pos='learned'):
        super().__init__()
        self.max_pos = max_pos
        self.pos = pos
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.pos_emb = (nn.Embedding(max_pos, hidden) if pos == 'learned' else None)
        self.layers = nn.ModuleList(Block(hidden, n_heads, pos=pos) for _ in range(n_layers))
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
        if self.pos_emb is not None:
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


def amp_ctx(amp):
    if DEVICE != 'cuda' or amp == 'off':
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if amp == 'fp16' else torch.bfloat16)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def train_model(model, run_name, train_data, steps, amp='bf16',
                val_data=None, eval_every=0, sched=True, seed=SEED, quiet=False):
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
        for _ in range(ACCUM):
            xb, yb = batch()
            with amp_ctx(amp):
                _, loss = model(xb, yb)
                loss = loss / ACCUM
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_now = get_lr(step, steps, LR) if sched else LR
        for g in optimizer.param_groups:
            g['lr'] = lr_now
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)
        tok_s = BATCH * SEQ * ACCUM / max((time.time() - t0) / (step + 1), 1e-9)
        logger.log(step, train_loss=accum_loss, lr=lr_now, tok_s=tok_s)
        if not quiet and (step % 100 == 0 or step == steps - 1):
            print(f"  step {step:4d}/{steps}: loss {accum_loss:.4f} lr {lr_now:.2e} tok/s {tok_s:.0f}")
        if eval_every and val_data and (step % eval_every == 0 or step == steps - 1):
            vl, vp = evaluate(model, val_data)
            logger.log(step, val_loss=vl, val_ppl=vp)
            if not quiet:
                print(f"    ↳ 验证集: loss {vl:.4f}  ppl {vp:.2f}")
    logger.close()
    final = sum(losses[-5:]) / 5
    if not quiet:
        print(f"  📉 loss {losses[0]:.4f} → {final:.4f}")
    return final


# ══════════════════════════════════════════════════════════════════
# ↑↑↑ 04 章代码结束 ｜ 本章新零件 ↓↑↑
# ══════════════════════════════════════════════════════════════════

def build_extrap_eval(tok, seq, n=150):
    """【05 章新增】外推评估集：只收 341 ≤ token 数 ≤ 510 的文本（必须"越界"才测得到外推；
    短文本 pad 到 512 会因 pad=-100 不进 loss 而测量静默失效）。"""
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    data = []
    path = os.path.join(DATA_DIR, 'pretrain_t2t_mini_sample.jsonl')
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, 'pretrain_t2t_mini.jsonl')
    with open(path, encoding='utf-8') as f:
        for line in f:
            if len(data) >= n:
                break
            try:
                text = json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tok(text, add_special_tokens=False).input_ids
            if not (SEQ + 1 <= len(ids) <= seq - 2):
                continue
            ids = [bos] + ids[:seq - 2] + [eos]
            ids += [pad] * (seq - len(ids))
            labels = [t if t != pad else -100 for t in ids]
            data.append((torch.tensor(ids), torch.tensor(labels)))
    print(f"  外推评估集: {len(data)} 条 × seq={seq}（全部长于训练长度 {SEQ}）")
    return data


def rope_math_checks():
    """【05 章·数学检查】旋转不改长度（保范数）；打分只看相对距离（平移不变）；θ 分工局部/全局。"""
    torch.manual_seed(0)
    T, H, hd = 8, 2, 16
    cos, sin = precompute_cos_sin(hd, T + 4, THETA)
    c, s = cos[:T].unsqueeze(0).unsqueeze(2), sin[:T].unsqueeze(0).unsqueeze(2)

    # ① 保范数：旋转是正交变换
    q = torch.randn(1, T, H, hd)
    qr = q * c + rotate_half(q) * s
    d_norm = (q.norm(dim=-1) - qr.norm(dim=-1)).abs().max().item()

    # ② 平移不变：同内容序列的内积矩阵 A[i,j] 只依赖 i-j
    xq = torch.randn(H, hd)
    qq = xq.unsqueeze(0).expand(1, T, H, hd)
    qqr = qq * c + rotate_half(qq) * s
    A = torch.einsum('bihd,bjhd->bijh', qqr, qqr)[0]      # (T,T,H)
    shift = 2
    d_shift = max(abs(A[i, j, h].item() - A[i-shift, j-shift, h].item())
                  for i in range(shift, T) for j in range(shift, T) for h in range(H))

    # ③ 频率分工：第 i 维对波长 2π·θ^(2i/d)——短波长管相邻，长波长管全局
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2).float() / hd))
    lam_hi = 2 * math.pi / inv[0].item()
    lam_lo = 2 * math.pi / inv[-1].item()

    print(f"  ① 保范数：旋转前后 |q| 最大变化 {d_norm:.2e}")
    print(f"  ② 相对性：内积矩阵平移 {shift} 位，最大偏差 {d_shift:.2e}（A[i,j] 只依赖 i−j）")
    print(f"  ③ 频率分工：最短波长 {lam_hi:.1f} token（管相邻顺序）｜最长 {lam_lo:,.0f} token（管全局定位）")


def rope_vs_learned(tok):
    """【05 章实验①】同 seed、同数据、同步数，只差 pos 一个开关；训练长内外各测一次 ppl。"""
    data = build_dataset(tok, SEQ, MAX_SAMPLES)
    train_data, val_data = split_train_val(data)
    ext_data = build_extrap_eval(tok, seq=512)
    results = {}
    for pos in ('learned', 'rope'):
        model = MiniMindForCausalLM(vocab=len(tok), hidden=HIDDEN, n_layers=N_LAYERS,
                                    n_heads=N_HEADS, max_pos=MAX_POS, pos=pos).to(DEVICE)
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ── pos={pos}（{n_param:.2f}M）──")
        train_model(model, f's4_{pos}', train_data, STEPS, amp='bf16', quiet=True)
        _, ppl340 = evaluate(model, val_data)
        _, ppl512 = evaluate(model, ext_data)
        results[pos] = (ppl340, ppl512)
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print("  ────────────────────────────────────────")
    print("  位置编码       ppl@340(训练长内)   ppl@512(外推)")
    for pos, (a, b) in results.items():
        print(f"  {pos:<12}    {a:8.2f}          {b:8.2f}")
    la, lb = results['learned']
    ra, rb = results['rope']
    print(f"  ↳ 外推变化: learned {(lb/la-1)*100:+.1f}%、rope {(rb/ra-1)*100:+.1f}%")
    print("  ↳ learned 的未训练位置行≈初始化噪声(0.02)：远处 token 丢位置信息但不被投毒——")
    print("    「钝痛不猝死」；真正的硬上限是表长本身，max_pos 外连算都不能算")
    print("  ↳ 读数规则：跨模型比必须同评估集、跨长度比必须同模型（ppl 绝对值无意义）")


def main():
    torch.manual_seed(SEED)
    print(f"═══ 05 · 位置编码进化 ═══  device={DEVICE}")
    print("══ 数学检查：保范数 ｜ 平移不变 ｜ 频率分工 ══")
    rope_math_checks()
    print()
    print("══ 实验①：learned PE vs RoPE（同 seed 各训一次）══")
    rope_vs_learned(load_tokenizer())


if __name__ == '__main__':
    main()
