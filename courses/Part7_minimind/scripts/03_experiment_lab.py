#!/usr/bin/env python3
"""
03_experiment_lab.py —— 03 章「实验仪器」配套脚本

= 02_baseline.py 的全部代码，叠加本章新仪器：
  ① 验证集切分（训练前切好）+ 验证 ppl 评估
  ② MetricLogger：CSV 恒定记录 + tensorboard 检测启用（+P7_SWANLAB=1 可选上云）
训练循环仍然是最朴素形态：fp32、无裁剪、无累积、无调度——那些是 04 章的零件。

用法：
  python 03_experiment_lab.py                  # 约 1 分钟（RTX 4090）
  tensorboard --logdir ../temp/out/logs --port 6006   # 看曲线
  P7_STEPS=10 python 03_experiment_lab.py             # 冒烟模式
"""

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
LR = 5e-4
SEQ = 340
MAX_SAMPLES = 20000
EVAL_EVERY = 100
SEED = 1337
# CPU toy 档：无 GPU 只保流程可跑通（数字与 GPU 档不可比）
HIDDEN, N_LAYERS, N_HEADS, MAX_POS = (64, 2, 4, 1024) if DEVICE == 'cpu' else (512, 8, 8, 1024)
if DEVICE == 'cpu':
    MAX_SAMPLES = min(MAX_SAMPLES, 3000)


# ══════════════════════════════════════════════════════════════════
# ↓↓↓ 以下与 02_baseline.py 相同（叠加生长：第 N 章脚本包含 1..N 的全部代码）↓↓↓
# ══════════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))


class Attention(nn.Module):
    """MHA：8 个 Q 头，每个头有自己独立的 K/V 头。"""
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
    """①截断到 seq-2 ②手工包 bos/eos ③右补 pad ④labels 仅 pad 置 -100（09 章拆解）。"""
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


# ══════════════════════════════════════════════════════════════════
# ↑↑↑ 02 章代码结束 ｜ 本章新仪器 ↓↑↑
# ══════════════════════════════════════════════════════════════════

def split_train_val(data, val_ratio=0.05):
    """训练前先切出验证集：取尾部 5%。"训练开始之前"四个字是纪律，不是细节。"""
    n_val = max(1, int(len(data) * val_ratio))
    return data[:-n_val], data[-n_val:]


@torch.no_grad()
def evaluate(model, data, batch=32):
    """验证集 loss / ppl：model.eval() + no_grad，与训练严格隔离。"""
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
    """CSV 兜底（零依赖、永远可用），检测到 tensorboard 再并行写事件文件。
    列结构显式声明，train 行与 eval 行各填各的列。"""
    FIELDS = ['step', 'train_loss', 'lr', 'val_loss', 'val_ppl']

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
        self._swan = None
        if os.environ.get('P7_SWANLAB') == '1':
            try:
                import swanlab
                swanlab.init(project='my-minimind', experiment_name=run_name)
                self._swan = swanlab
            except ImportError:
                print("  ⚠️ P7_SWANLAB=1 但未安装 swanlab——跳过云端记录")

    def log(self, step, **tags):
        row = {'step': step, **tags}
        self._csv.write(','.join(
            f'{row[k]:.6g}' if isinstance(row.get(k), float) else str(row.get(k, ''))
            for k in self.FIELDS) + '\n')
        if self._tb is not None:
            for k, v in tags.items():
                self._tb.add_scalar(k, v, step)
        if self._swan is not None:
            self._swan.log(tags, step=step)

    def close(self):
        self._csv.close()
        if self._tb is not None:
            self._tb.close()
        if self._swan is not None:
            self._swan.finish()

    @staticmethod
    def view_hint():
        print(f"  📈 曲线怎么看：tensorboard --logdir "
              f"{os.path.relpath(LOG_DIR, SCRIPT_DIR)} --port 6006（或直接打开 logs/*.csv）")


def main():
    torch.manual_seed(SEED)
    print(f"═══ 03 · 实验仪器 ═══  device={DEVICE}")

    tok = load_tokenizer()
    model = MiniMindForCausalLM(vocab=len(tok), hidden=HIDDEN, n_layers=N_LAYERS, n_heads=N_HEADS, max_pos=MAX_POS)
    model = model.to(DEVICE)

    data = build_dataset(tok, SEQ, MAX_SAMPLES)
    train_data, val_data = split_train_val(data)
    print(f"  切分: 训练 {len(train_data):,} 条 ｜ 验证 {len(val_data):,} 条（尾部 5%，训练前切好）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    logger = MetricLogger('s2_baseline_instrumented')
    t0 = time.time()
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, len(train_data), (BATCH,))
        xb = torch.stack([train_data[i][0] for i in idx]).to(DEVICE)
        yb = torch.stack([train_data[i][1] for i in idx]).to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        logger.log(step, train_loss=loss.item(), lr=LR)      # ← 仪器：每步入 CSV/tensorboard
        if step % 20 == 0 or step == STEPS - 1:
            print(f"  step {step:3d}/{STEPS}: loss {loss.item():.4f} lr {LR:.2e} "
                  f"eta {(time.time()-t0)/(step+1)*(STEPS-step-1):.0f}s")
        if step % EVAL_EVERY == 0 or step == STEPS - 1:      # ← 仪器：定期验证 ppl
            vl, vp = evaluate(model, val_data)
            logger.log(step, val_loss=vl, val_ppl=vp)
            print(f"    ↳ 验证集: loss {vl:.4f}  ppl {vp:.2f}")
    logger.close()
    print(f"  📉 训练 loss 首末读数见上；日志 {os.path.relpath(logger.csv_path, SCRIPT_DIR)}")
    MetricLogger.view_hint()
    print("""  ── 问题清单（后面每一章修一个）──
  ① fp32 训练慢、显存高            → 04 章 AMP（fp16+GradScaler / bf16）
  ② 训练 340 长、推理 512 会怎样？ → 05 章 learned PE 的外推退化 → RoPE
  ③ 同参数还能更 low loss 吗？     → 07 章 ReLU→SwiGLU（论文结论复现）
  ④ 想要官方 25.83M 的身材         → 06 章 GQA / 07 章 SwiGLU / 08 章 RMSNorm 减肥账本""")


if __name__ == '__main__':
    main()
