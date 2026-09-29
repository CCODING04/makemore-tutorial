#!/usr/bin/env python3
"""
02_baseline.py —— 02 章「基线诞生」配套脚本

本章的全部知识：BPE tokenizer（用官方的）+ 权重绑定 + 最小可用 minimind
（MHA + learned 位置表 + ReLU FFN + LayerNorm）+ 第一次训练（60 步，fp32）。

训练循环就是 Part 6 的"三行循环"：零梯度 → 前向 → 反向 → step。
裁剪、梯度累积、混合精度、调度器……都还没有——它们分别在 04 章登场。
这世上还不存在 RoPE / GQA / QK-Norm / SwiGLU：后面每章的脚本会在本章代码上"叠加生长"。

用法：
  python 02_baseline.py            # 约 30 秒（RTX 4090）
  P7_STEPS=10 python 02_baseline.py   # 冒烟模式
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 本章超参（全章可见、可直接改）─────────────────────────────
STEPS = int(os.environ.get('P7_STEPS', 60))
BATCH = 16
LR = 5e-4
SEQ = 340
MAX_SAMPLES = 8000
SEED = 1337
# CPU toy 档：无 GPU 只保流程可跑通（数字与 GPU 档不可比）
HIDDEN, N_LAYERS, N_HEADS, MAX_POS = (64, 2, 4, 1024) if DEVICE == 'cpu' else (512, 8, 8, 1024)
if DEVICE == 'cpu':
    MAX_SAMPLES = min(MAX_SAMPLES, 3000)


# ══════════════════════════════════════════════════════════════════
# 字典：官方 tokenizer（02 章 §1：BPE 与特殊 token）
# ══════════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))


def vocab_report(tok):
    print(f"  词表大小: {len(tok)}")
    print(f"  三个角色 token: bos={tok.bos_token!r}(id={tok.bos_token_id})  "
          f"eos={tok.eos_token!r}(id={tok.eos_token_id})  pad={tok.pad_token!r}(id={tok.pad_token_id})")
    print(f"  保留的特殊 token: {len(tok.added_tokens_decoder)} 个（im/vision/audio/tool_call/think…）")
    for s in ["人工智能是计算机科学的一个分支，它企图了解智能的实质。",
              "Large language models are trained on vast amounts of text data."]:
        n = len(tok(s).input_ids)
        print(f"  压缩率样例: {len(s)} 字 → {n} token（{len(s)/n:.1f} 字/token）")


# ══════════════════════════════════════════════════════════════════
# 基线模型：MHA + learned PE + ReLU FFN + LayerNorm（02 章 §3 完整版）
# 注意：本文件里没有任何 RoPE/GQA/QK-Norm/SwiGLU 的代码——它们还没有被发明
# ══════════════════════════════════════════════════════════════════

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
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)     # (B,T,8,64) 先切头
        k = self.k_proj(x).view(B, T, self.n_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_heads, self.hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))         # (B,8,T,64)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)  # (B,8,T,T)
        scores = scores + torch.triu(torch.full((T, T), float('-inf'),
                                            device=x.device), 1)  # 因果 mask：看不到未来
        out = F.softmax(scores.float(), dim=-1).type_as(q) @ v     # float() 保 softmax 稳定
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class FFN(nn.Module):
    """ReLU 版：Linear(d→4d) → ReLU → Linear(4d→d)。"""
    def __init__(self, hidden):
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden, bias=False)
        self.down = nn.Linear(4 * hidden, hidden, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class Block(nn.Module):
    """pre-norm 残差块：x + attn(norm(x)); x + ffn(norm(x))——Part 6 的骨架。"""
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
        self.pos_emb = nn.Embedding(max_pos, hidden)          # learned PE：一张可训练位置表
        self.layers = nn.ModuleList(Block(hidden, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight        # tie：输入输出共用一张表，省 12.6% 参数
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)     # HF 基类默认

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        T = input_ids.size(1)
        x = x + self.pos_emb(torch.arange(T, device=input_ids.device))   # 位置 i 查第 i 行加上
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            x_, y_ = logits[:, :-1, :], labels[:, 1:]          # shift 只在这里做一次！
            loss = F.cross_entropy(x_.reshape(-1, x_.size(-1)),
                                   y_.reshape(-1), ignore_index=-100)
        return logits, loss

    def param_report(self):
        groups = {'embedding': [], 'pos_emb': [], 'attention': [], 'ffn': [], 'norm': []}
        for name, p in self.named_parameters():
            key = ('embedding' if 'embed_tokens' in name
                   else 'pos_emb' if 'pos_emb' in name
                   else 'attention' if 'self_attn' in name
                   else 'ffn' if 'mlp' in name else 'norm')
            groups[key].append(p.numel())
        total = sum(sum(v) for v in groups.values())
        lines = [f"    {k:<10} {sum(v)/1e6:7.3f} M  ({sum(v)/total*100:5.1f}%)"
                 for k, v in groups.items() if v]
        return total, '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════
# 数据：官方 pretrain_t2t_mini.jsonl 的四步加工（为什么这么设计 → 09 章拆解）
# ══════════════════════════════════════════════════════════════════

def build_dataset(tok, seq, max_samples):
    """①截断到 seq-2 ②手工包 bos/eos ③右补 pad ④labels 仅 pad 置 -100。"""
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


def main():
    torch.manual_seed(SEED)
    print(f"═══ 02 · 基线诞生 ═══  device={DEVICE}")

    tok = load_tokenizer()
    vocab_report(tok)

    model = MiniMindForCausalLM(vocab=len(tok), hidden=HIDDEN, n_layers=N_LAYERS, n_heads=N_HEADS, max_pos=MAX_POS)
    model = model.to(DEVICE)
    total, rep = model.param_report()
    print("  基线模型（MHA+learned PE+ReLU+LayerNorm）参数账本:")
    print(rep)
    print(f"  总参数: {total/1e6:.2f} M（对照官方现代版 25.83M——升级之路先记账）")

    data = build_dataset(tok, SEQ, MAX_SAMPLES)
    print(f"  ── 第一次训练：{STEPS} 步 fp32，看 loss 从「均匀猜测」往下走 ──")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time()
    model.train()
    for step in range(STEPS):
        idx = torch.randint(0, len(data), (BATCH,))
        xb = torch.stack([data[i][0] for i in idx]).to(DEVICE)
        yb = torch.stack([data[i][1] for i in idx]).to(DEVICE)
        optimizer.zero_grad(set_to_none=True)      # ┐
        _, loss = model(xb, yb)                    # ├ Part 6 的三行循环：清梯度→前向→反向→step
        loss.backward()                            # │ （裁剪/累积/AMP/调度器？还没有——04 章见）
        optimizer.step()                           # ┘
        if step % 20 == 0 or step == STEPS - 1:
            print(f"  step {step:3d}/{STEPS}: loss {loss.item():.4f} lr {LR:.2e}")
    print(f"  📉 初始 loss 应≈ln(6400)≈8.76；用时 {time.time()-t0:.0f}s ｜ "
          f"峰值显存 {torch.cuda.max_memory_allocated()/1e9 if DEVICE=='cuda' else 0:.1f} GB")
    print("  ↳ v1 达成：最小可用 minimind——能建、能训、loss 会降。先跑通，再变强（03 章起）。")


if __name__ == '__main__':
    main()
