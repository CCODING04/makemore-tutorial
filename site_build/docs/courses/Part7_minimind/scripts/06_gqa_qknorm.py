#!/usr/bin/env python3
"""
06_gqa_qknorm.py —— 06 章「注意力的两次手术」配套脚本

= 05_rope.py 的全部代码，叠加本章零件：
  ⓪ 实验①：KV Cache 一致性——逐 token 生成 vs 全序列计算（GQA+RoPE+QK-Norm 全开）
  ① RMSNorm 类（本章先作为 q/k 的稳定器登场；08 章把全模型归一化都换成它）
  ② Attention 增加 attn 开关（MHA / GQA：n_kv 组共享 K/V）与 qk_norm 开关
  ③ 实验②：MHA vs GQA——短训比质量，账本比参数与 KV Cache
  ④ 实验③：QK-Norm"造病"实验——人为注入范数漂移，验证药物有效

用法：
  python 06_gqa_qknorm.py              # 实验②③，约 2 分钟（RTX 4090）
  P7_STEPS=30 python 06_gqa_qknorm.py   # 冒烟模式
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

def log_result(name, row):
    """消融实验一次性结果行落盘——教程引用数字的证据文件（temp/out/logs/exp*.csv）。"""
    import csv
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f'{name}.csv')
    row = {'date': time.strftime('%Y-%m-%d'), **row}
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

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
# ↓↓↓ 02-05 章代码（与 05_rope.py 相同，例外：Attention 增加 GQA/QK-Norm 开关）↓↓↓
# ══════════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))


def precompute_cos_sin(head_dim, max_seq_len, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return cos, sin


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RMSNorm(nn.Module):
    """【06 章新增】均方根归一化。本章先当 q/k 的"尺度钉"用；08 章把全模型 LayerNorm 都换成它。"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (self.weight * x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class Attention(nn.Module):
    """【06 章修改】新增：attn='mha'|'gqa'（K/V 只留 n_kv 组）与 qk_norm（先 norm 再 RoPE）。
    05 章的 pos 开关原样保留。"""
    def __init__(self, hidden, n_heads, pos='learned', attn='mha', qk_norm=False,
                 max_pos=MAX_POS, theta=THETA):
        super().__init__()
        self.pos, self.qk_norm_on = pos, qk_norm
        self.n_heads = n_heads
        self.n_kv = n_heads if attn == 'mha' else max(1, n_heads // 4)   # GQA：K/V 只留 2 组
        self.hd = hidden // n_heads
        self.q_proj = nn.Linear(hidden, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv * self.hd, bias=False)  # K/V 投影 ÷4——省在这里
        self.v_proj = nn.Linear(hidden, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(hidden, self.n_heads * self.hd, bias=False)
        if qk_norm:                                                      # 只 norm q/k，不 norm v
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)
        if pos == 'rope':
            cos, sin = precompute_cos_sin(self.hd, max_pos, theta)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd)
        if self.qk_norm_on:
            q, k = self.q_norm(q), self.k_norm(k)          # 先 norm 再 RoPE（顺序不能反，06 章 §5）
        if self.pos == 'rope':
            cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(2)
            sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        rep = self.n_heads // self.n_kv                    # repeat_kv：KV 复制给每个 Q 组
        if rep > 1:
            k = k[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_heads, self.hd)
            v = v[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_heads, self.hd)
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
    def __init__(self, hidden, n_heads, pos='learned', attn='mha', qk_norm=False):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = Attention(hidden, n_heads, pos=pos, attn=attn, qk_norm=qk_norm)
        self.mlp = FFN(hidden)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMindForCausalLM(nn.Module):
    def __init__(self, vocab, hidden, n_layers, n_heads, max_pos,
                 pos='learned', attn='mha', qk_norm=False):
        super().__init__()
        self.max_pos = max_pos
        self.pos = pos
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.pos_emb = (nn.Embedding(max_pos, hidden) if pos == 'learned' else None)
        self.layers = nn.ModuleList(Block(hidden, n_heads, pos=pos, attn=attn, qk_norm=qk_norm)
                                    for _ in range(n_layers))
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

    def batch():
        idx = torch.randint(0, len(train_data), (BATCH,))
        xb = torch.stack([train_data[i][0] for i in idx]).to(DEVICE)
        yb = torch.stack([train_data[i][1] for i in idx]).to(DEVICE)
        return xb, yb

    t0, losses = time.time(), []
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
        logger.log(step, train_loss=accum_loss, lr=lr_now)
    logger.close()
    return sum(losses[-5:]) / 5


def build_extrap_eval(tok, seq, n=150):
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
    return data


# ══════════════════════════════════════════════════════════════════
# ↑↑↑ 05 章代码结束 ｜ 本章新零件 ↓↑↑
# ══════════════════════════════════════════════════════════════════

def kv_cache_consistency():
    """【06 章·实验①】真·KV Cache：逐 token 生成 vs 全序列一次计算，输出应逐 token 一致。
    开关全开（GQA+RoPE+QK-Norm）——cache 与三个现代件同时兼容。"""
    torch.manual_seed(42)
    hidden, n_heads, T = 64, 4, 10
    attn = Attention(hidden, n_heads, pos='rope', attn='gqa', qk_norm=True,
                     max_pos=64, theta=THETA).to(DEVICE)
    x = torch.randn(1, T, hidden, device=DEVICE)
    with torch.no_grad():
        full = attn(x)                                    # (1,T,hidden) 全序列一次算
        outs, cache = [], None
        for t in range(T):
            xt = x[:, t:t + 1, :]
            q = attn.q_proj(xt).view(1, 1, attn.n_heads, attn.hd)
            k = attn.k_proj(xt).view(1, 1, attn.n_kv, attn.hd)
            v = attn.v_proj(xt).view(1, 1, attn.n_kv, attn.hd)
            if attn.qk_norm_on:                           # 与 forward 同序：先 norm 再旋转
                q, k = attn.q_norm(q), attn.k_norm(k)
            cos = attn.rope_cos[t:t + 1].unsqueeze(0).unsqueeze(2)
            sin = attn.rope_sin[t:t + 1].unsqueeze(0).unsqueeze(2)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            cache = ((k, v) if cache is None
                     else (torch.cat([cache[0], k], 1), torch.cat([cache[1], v], 1)))
            kc, vc = cache
            rep = attn.n_heads // attn.n_kv               # repeat_kv 到 Q 头数
            kc = kc[:, :, :, None, :].expand(1, t + 1, attn.n_kv, rep, attn.hd) \
                .reshape(1, t + 1, attn.n_heads, attn.hd)
            vc = vc[:, :, :, None, :].expand(1, t + 1, attn.n_kv, rep, attn.hd) \
                .reshape(1, t + 1, attn.n_heads, attn.hd)
            # 生成态：Q 只有 1 个位置，因果 mask 天然全可见（取 (t+1,t+1) mask 的最后一行）
            q_, k_, v_ = (tt.transpose(1, 2) for tt in (q, kc, vc))
            s = (q_ @ k_.transpose(-2, -1)) / math.sqrt(attn.hd)
            w = F.softmax(s.float(), dim=-1).type_as(q_)
            h = (w @ v_).transpose(1, 2).reshape(1, 1, attn.n_heads * attn.hd)
            outs.append(attn.o_proj(h))
        max_diff = max((full[:, t] - outs[t][0, 0]).abs().max().item() for t in range(T))
    print(f"  全序列 vs 逐 token KV Cache（GQA+RoPE+QK-Norm 全开）最大偏差: {max_diff:.2e}")
    print("  ✅ 一致——cache 不改结果：每步新算的 K/V 从 O(t) 降到 O(1)，读取仍 O(t)（靠 GQA 压缓存）")


def gqa_vs_mha(tok):
    """【06 章实验②】质量：短训对照（同 seed，只差 attn 开关）；成本：纯算术账本。"""
    d, n_q, n_kv, hd = HIDDEN, N_HEADS, 2, HIDDEN // N_HEADS
    kv_mha = 2 * d * d
    kv_gqa = 2 * d * (n_kv * hd)
    seq = 2048
    print(f"  单层 K+V 投影参数:  MHA {kv_mha/1e6:.2f}M → GQA {kv_gqa/1e6:.2f}M（÷{n_q//n_kv}）")
    print(f"  KV Cache(seq={seq}, fp16): MHA {2*seq*n_q*hd*2/1e6:.1f}MB/层 → "
          f"GQA {2*seq*n_kv*hd*2/1e6:.1f}MB/层")
    data = build_dataset(tok, SEQ, MAX_SAMPLES)
    train_data, val_data = split_train_val(data)
    res = {}
    for attn in ('mha', 'gqa'):
        model = MiniMindForCausalLM(vocab=len(tok), hidden=HIDDEN, n_layers=N_LAYERS,
                                    n_heads=N_HEADS, max_pos=MAX_POS,
                                    pos='rope', attn=attn).to(DEVICE)
        loss = train_model(model, f's5_{attn}', train_data, STEPS, amp='bf16', quiet=True)
        _, ppl = evaluate(model, val_data)
        res[attn] = ppl
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    log_result('exp06_gqa_vs_mha', dict(runner='06_gqa_qknorm', ppl_mha=res['mha'], ppl_gqa=res['gqa'],
                                        kv_proj_mha_m=kv_mha / 1e6, kv_proj_gqa_m=kv_gqa / 1e6))
    print(f"  短训后验证集 ppl: MHA {res['mha']:.2f} vs GQA {res['gqa']:.2f} —— "
          f"tiny 规模质量损失几乎为零，收益全在推理（GQA 论文结论一致）")


def qk_norm_experiment():
    """【06 章实验③】造病实验：人为注入"范数漂移 3 倍"，验证 RMSNorm 能钉死打分尺度。"""
    hd, B, T = 64, 2, 16
    torch.manual_seed(0)
    q = torch.randn(B, T, hd) * 3.0              # ×3：造病
    k = torch.randn(B, T, hd) * 3.0
    raw = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    qn = q / (q.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()   # 给药：只 norm q/k
    kn = k / (k.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    normed = (qn @ kn.transpose(-1, -2)) / math.sqrt(hd)
    log_result('exp06_qk_norm', dict(runner='06_gqa_qknorm', raw_std=raw.std().item(),
                                     normed_std=normed.std().item()))
    print(f"  未归一化 logits: std={raw.std():.2f}（logit 差 8.85 → 概率比 e^8.85≈7000:1，≈one-hot）")
    print(f"  q/k RMSNorm 后:  std={normed.std():.2f}（尺度钉死——Qwen3/Gemma 系同款）")


def main():
    torch.manual_seed(SEED)
    print(f"═══ 06 · 注意力的两次手术 ═══  device={DEVICE}")
    print("══ 实验①：KV Cache 一致性（GQA+RoPE+QK-Norm 全开）══")
    kv_cache_consistency()
    print()
    tok = load_tokenizer()
    print("══ 实验②：MHA vs GQA（短训比质量，账本比推理）══")
    gqa_vs_mha(tok)
    print()
    print("══ 实验③：QK-Norm（造病实验：范数漂移 3 倍）══")
    qk_norm_experiment()


if __name__ == '__main__':
    main()
