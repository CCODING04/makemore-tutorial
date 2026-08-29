#!/usr/bin/env python3
"""
Part 8 - 脚本 2: 预训练 GPT-2
目标：在文本数据上预训练脚本 1 的 GPT-2，学会"续写下一个 token"。
演示现代 LLM 训练的标准技巧：AdamW + cosine LR + warmup + gradient clipping +
bf16 混合精度 + gradient accumulation + checkpoint 保存/恢复。

覆盖知识点：
  - 数据加载：字符级（CPU）/ tiktoken BPE（GPU，需安装 tokenizers 库）
  - AdamW 优化器：beta=(0.9, 0.95)，weight_decay=0.1
  - cosine LR schedule + linear warmup
  - bf16 混合精度：torch.autocast（bf16 不需要 GradScaler，与 fp16 不同）
  - gradient accumulation：小 batch 多次 forward 后再 step
  - gradient clipping：clip_grad_norm_(1.0) 防止梯度爆炸
  - checkpoint：保存 model + optimizer + step + losses

torch API 速查：
  torch.autocast(device_type='cuda', dtype=torch.bfloat16) — bf16 自动混合精度
  torch.optim.AdamW — 解耦权重衰减的 Adam（比 Adam + L2 更正确）
  torch.optim.lr_scheduler.LambdaLR — 自定义 LR 调度（用 lambda 函数）
  torch.nn.utils.clip_grad_norm_ — 梯度裁剪（按范数）
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.set_num_threads(1)

# ─── 模式选择 ──────────────────────────────────────────────
CPU_MODE = not torch.cuda.is_available()
if CPU_MODE:
    vocab_size = 256
    n_embed = 64
    n_head = 4
    n_blocks = 2
    context_length = 64
    batch_size = 4
    max_steps = 50
    grad_accum = 4
    lr = 1e-3
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 4
    max_steps = 200
    grad_accum = 8
    lr = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


# ─── 模型定义（内嵌精简版，与 01_gpt_model.py 等价）────────
class Head(nn.Module):
    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        hs = n_embed // n_head
        self.heads = nn.ModuleList([Head(hs, n_embed, context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_head, n_embed, context_length, vocab_size, n_blocks):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(context_length, n_embed)
        self.blocks = nn.ModuleList([Block(n_head, n_embed, context_length) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos', torch.arange(context_length))

    def forward_hidden(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos[:T])
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)

    def forward(self, idx, targets=None):
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.context_length:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# ─── 数据加载 ──────────────────────────────────────────────
def load_data():
    """加载文本数据并编码为 token id。

    CPU 模式：用 data/input.txt（tiny Shakespeare），字符级编码
    GPU 模式：优先用 data/input.txt + 字符级（确保可运行），后续可替换为 tiktoken
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    if not os.path.exists(data_path):
        print(f"⚠️  数据文件不存在: {data_path}，使用合成数据")
        # 合成数据：随机字符
        text = ''.join(chr(i % 127) for i in range(10000))
        chars = sorted(list(set(text)))
        stoi = {c: i for i, c in enumerate(chars)}
        data = [stoi[c] for c in text]
        return torch.tensor(data, dtype=torch.long), len(chars), f"合成(vocab={len(chars)})"

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if CPU_MODE:
        # 字符级编码（CPU 模式，vocab 小，训练快）
        chars = sorted(list(set(text)))
        stoi = {c: i for i, c in enumerate(chars)}
        data = [stoi[c] for c in text]
        return torch.tensor(data, dtype=torch.long), len(chars), f"char(vocab={len(chars)})"
    else:
        # GPU 模式：尝试 tiktoken BPE，回退字符级
        try:
            import tiktoken
            enc = tiktoken.get_encoding('r50k_base')
            data = enc.encode_ordinary(text)
            return torch.tensor(data, dtype=torch.long), enc.n_vocab, f"tiktoken r50k_base(vocab={enc.n_vocab})"
        except ImportError:
            chars = sorted(list(set(text)))
            stoi = {c: i for i, c in enumerate(chars)}
            data = [stoi[c] for c in text]
            return torch.tensor(data, dtype=torch.long), len(chars), f"char(vocab={len(chars)})"


def get_batch(data, batch_size, block_size, device):
    """随机采样一个 batch。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def demo_mixed_precision():
    """演示 bf16 混合精度。"""
    print("\n═══ bf16 混合精度演示 ═══")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        c = a @ b
    print(f"  输入 fp32，autocast(bf16) 下矩阵乘法 → 输出 dtype: {c.dtype}")
    print(f"  💡 bf16 vs fp16：bf16 动态范围与 fp32 相同（8 bit 指数），不需要 GradScaler；")
    print(f"     fp16 动态范围小（5 bit 指数），容易溢出，需要 GradScaler 做 loss scaling。")
    print(f"     现代 GPU（A100/4090）推荐 bf16，旧卡（V100）只能用 fp16。")


def main():
    data, model_vocab, tok_name = load_data()

    print("═══ 预训练 GPT-2 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  分词器: {tok_name}, 数据 tokens: {len(data):,}")
    print(f"  模型: embed={n_embed}, heads={n_head}, blocks={n_blocks}, vocab={model_vocab}")
    print(f"  训练: batch={batch_size}, grad_accum={grad_accum}, "
          f"effective_batch={batch_size * grad_accum}, steps={max_steps}, lr={lr}")

    # 数据切分
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # 模型
    model = GPT(n_head, n_embed, context_length, model_vocab, n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,} = {n_params / 1e6:.2f}M")

    # 优化器：AdamW（解耦权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # LR schedule: linear warmup + cosine decay
    warmup = max(3, max_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    demo_mixed_precision()

    # ── 训练循环 ──
    print(f"\n═══ 训练（{max_steps} 步，每步 {grad_accum} 个 micro-batch 梯度累积） ═══")
    print(f"  技巧: AdamW(β=(0.9,0.95), wd=0.1) + cosine(warmup={warmup}) + "
          f"grad_clip=1.0 + autocast(bf16)")

    losses = []
    for step in range(max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mb_loss = 0.0
        for _ in range(grad_accum):
            xb, yb = get_batch(train_data, batch_size, context_length, device)
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
            else:
                _, loss = model(xb, yb)
            (loss / grad_accum).backward()
            mb_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(mb_loss / grad_accum)
        if step % 10 == 0 or step == max_steps - 1:
            print(f"  step {step:4d}: loss {losses[-1]:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")

    print(f"  📉 loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}（{max_steps} 步）")

    # ── checkpoint 保存/恢复 ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    ckpt_path = os.path.join(temp_dir, 'ckpt_pretrain.pt')

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': max_steps,
        'losses': losses,
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': model_vocab, 'context_length': context_length,
        }
    }, ckpt_path)
    print(f"\n═══ Checkpoint 保存/恢复 ═══")
    print(f"  ✅ 已保存 → {ckpt_path}（{os.path.getsize(ckpt_path) / 1024:.0f} KB）")

    # 恢复验证
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2 = GPT(n_head, n_embed, context_length, model_vocab, n_blocks).to(device)
    model2.load_state_dict(ckpt['model'])
    ok = all(torch.allclose(p1, p2) for p1, p2 in
             zip(model.parameters(), model2.parameters()))
    print(f"  ✅ 恢复后参数一致: {ok}")

    # ── 生成演示 ──
    print(f"\n═══ 预训练后生成 ═══")
    if CPU_MODE:
        prompt = torch.randint(0, model_vocab, (1, 4), device=device)
        gen = model.generate(prompt, max_new_tokens=64, temperature=0.8, top_k=40)
        print(f"  prompt: {prompt[0].tolist()}")
        print(f"  生成: {gen[0].tolist()[:20]}...")
    else:
        # GPU 模式用字符级解码演示
        prompt_text = "First Citizen:\n"
        script_dir2 = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir2, '..', '..', '..', 'data', 'input.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        itos = {i: c for i, c in enumerate(chars)}
        stoi = {c: i for i, c in enumerate(chars)}
        prompt_ids = [stoi.get(c, 0) for c in prompt_text]
        gen = model.generate(torch.tensor([prompt_ids], device=device), max_new_tokens=120,
                             temperature=0.8, top_k=40)
        gen_text = ''.join(itos.get(i, '?') for i in gen[0].tolist())
        print(f"  prompt: {prompt_text!r}")
        print(f"  生成: {gen_text[:200]!r}")

    print(f"""
═══ 总结 ═══

预训练 = 让模型学会"补全下一个 token"。本脚本串起现代训练技巧：
  AdamW + cosine(warmup) → 稳定收敛
  grad_clip(1.0) → 防梯度爆炸
  grad_accum → 等效大 batch
  autocast(bf16) → 加速省显存
  checkpoint → 断点续训

训练后 loss 明显下降、能生成像样的文本，就是预训练的效果。
下一个脚本：SFT 监督微调（Chat Template + Prompt Masking）。""")

if __name__ == '__main__':
    main()
