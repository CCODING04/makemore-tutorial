#!/usr/bin/env python3
"""
Part 7 - 脚本 6: 预训练流水线（Pretrain Pipeline）
目标：用 input.txt 预训练脚本 5 的 MiniMind 模型，演示现代 LLM 训练的
标准技巧：mixed precision、gradient accumulation、gradient clipping、
AdamW、cosine LR schedule，以及 checkpoint 保存/恢复。

覆盖知识点：
  - 模型加载：优先从 05_full_model.py 导入完整模型（需要先运行 01 生成
    bpe_tokenizer.json）；如果导入失败（如缺少依赖），自动回退到内嵌精简版
  - 数据集：优先加载脚本 1 的 BPE tokenizer（字节级），回退字符级
  - mixed precision：torch.autocast（GPU bf16 加速；CPU 上单独演示）
  - gradient accumulation：小 batch 多次 forward/backward 后再 step，
    等效于大 batch，省显存
  - gradient clipping：torch.nn.utils.clip_grad_norm_ 防止梯度爆炸
  - cosine LR schedule：warmup 后按余弦衰减到接近 0
  - checkpoint：保存 model + optimizer + step，可断点续训
"""

import os
import sys
import math
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使训练更快更稳定
torch.set_num_threads(1)

# ─── 模式选择 ──────────────────────────────────────────────
# CPU 模式: 小模型，<30s 跑完，用于学习验证
# GPU 模式: 完整规模，匹配 minimind 架构，需 GPU
CPU_MODE = not torch.cuda.is_available()
if CPU_MODE:
    vocab_size = 256
    hidden_size = 64
    n_layers = 2
    n_heads = 4
    n_kv_heads = 2
else:
    vocab_size = 6400
    hidden_size = 768
    n_layers = 8
    n_heads = 8
    n_kv_heads = 4
# ─── ───────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── 训练超参 ─────────────────────────────────────────────
if CPU_MODE:
    batch_size, block_size, max_steps, grad_accum, lr = 16, 64, 50, 4, 1e-3
else:
    batch_size, block_size, max_steps, grad_accum, lr = 32, 128, 500, 8, 3e-4
max_seq = block_size
torch.manual_seed(1337)


# ─── 模型加载：优先导入脚本 05，失败则内嵌精简版 ──────────
def load_model_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, '05_full_model.py')
    try:
        spec = importlib.util.spec_from_file_location('mm_full', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"  ⚠️ 导入 05_full_model.py 失败（{e}），改用内嵌精简模型")
        return None


def build_fallback():
    """内嵌精简版 MiniMind（与 05 等价，供 05 缺失时使用）。"""
    class MiniMindConfig:
        def __init__(self, hidden_size=64, num_hidden_layers=2, vocab_size=65,
                     num_attention_heads=4, num_key_value_heads=2,
                     intermediate_size=None, rms_norm_eps=1e-5, rope_theta=10000.0,
                     max_position_embeddings=256, tie_word_embeddings=True,
                     flash_attn=False):
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.vocab_size = vocab_size
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.head_dim = hidden_size // num_attention_heads
            self.intermediate_size = intermediate_size or \
                int((math.pi * hidden_size / 64) + 0.5) * 64
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.max_position_embeddings = max_position_embeddings
            self.tie_word_embeddings = tie_word_embeddings
            self.flash_attn = flash_attn

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class Attention(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            n, hd, nk = cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads
            self.wq = nn.Linear(cfg.hidden_size, n * hd, bias=False)
            self.wk = nn.Linear(cfg.hidden_size, nk * hd, bias=False)
            self.wv = nn.Linear(cfg.hidden_size, nk * hd, bias=False)
            self.wo = nn.Linear(n * hd, cfg.hidden_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(
                cfg.max_position_embeddings, cfg.max_position_embeddings)))

        def forward(self, x):
            B, T, C = x.shape
            n, hd, nk = self.cfg.num_attention_heads, self.cfg.head_dim, self.cfg.num_key_value_heads
            q = self.wq(x).view(B, T, n, hd)
            k = self.wk(x).view(B, T, nk, hd)
            v = self.wv(x).view(B, T, nk, hd)
            k = k.repeat_interleave(n // nk, dim=2)
            v = v.repeat_interleave(n // nk, dim=2)
            wei = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / (hd ** 0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            out = (wei @ v.transpose(1, 2)).transpose(1, 2).reshape(B, T, n * hd)
            return self.wo(out)

    class SwiGLU(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            g = cfg.intermediate_size
            self.gate = nn.Linear(cfg.hidden_size, g, bias=False)
            self.up = nn.Linear(cfg.hidden_size, g, bias=False)
            self.down = nn.Linear(g, cfg.hidden_size, bias=False)

        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class Block(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.attn = Attention(cfg)
            self.ffn = SwiGLU(cfg)
            self.ln1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.ln2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class MiniMindForCausalLM(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            if cfg.tie_word_embeddings:
                self.lm_head.weight = self.tok.weight
            self.apply(self._init)

        def _init(self, m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            x = self.tok(idx)
            for b in self.blocks:
                x = b(x)
            x = self.norm(x)
            logits = self.lm_head(x)
            if targets is None:
                return logits
            return logits, F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))

        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
            for _ in range(max_new_tokens):
                logits = self(idx[:, -self.cfg.max_position_embeddings:])[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
            return idx

    return MiniMindConfig, MiniMindForCausalLM


_MOD = load_model_module()
if _MOD is not None:
    MiniMindConfig = _MOD.MiniMindConfig
    MiniMindForCausalLM = _MOD.MiniMindForCausalLM
else:
    MiniMindConfig, MiniMindForCausalLM = build_fallback()


# ─── 分词器加载：优先 BPE，回退字符级 ─────────────────────
def load_tokenizer(temp_dir):
    path = os.path.join(temp_dir, 'bpe_tokenizer.json')
    if os.path.exists(path):
        try:
            from tokenizers import Tokenizer as BPE_Tokenizer
            tok = BPE_Tokenizer.from_file(path)
            enc = lambda s: tok.encode(s).ids
            dec = lambda ids: tok.decode(ids)
            return enc, dec, tok.get_vocab_size(), f"BPE(vocab={tok.get_vocab_size()})"
        except Exception as e:
            print(f"  ⚠️  加载 BPE tokenizer 失败（{e}），回退字符级")
    with open(os.path.join(temp_dir, '..', '..', '..', 'data', 'input.txt'),
              'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return (lambda s: [stoi[c] for c in s]), (lambda ids: ''.join(itos[i] for i in ids)), \
        len(chars), f"char(vocab={len(chars)})"


def demo_mixed_precision():
    print("\n═══ mixed precision（autocast）演示 ═══")
    a, b = torch.randn(256, 256), torch.randn(256, 256)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        c = a @ b
    print(f"  输入 fp32 (256,256)，autocast 下计算 → 输出 dtype: {c.dtype}")
    print(f"  ✅ torch.autocast(device_type='{device}') 在 CPU 也能跑（bf16 仅演示）")
    print(f"  💡 GPU 上训练用它可省一半显存并加速；CPU bf16 常更慢，故 CPU 训练走 fp32")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    enc, dec, model_vocab, tok_name = load_tokenizer(temp_dir)
    data = torch.tensor(enc(text), dtype=torch.long)

    print("═══ 预训练流水线 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  分词器: {tok_name}, 数据 tokens: {len(data):,}")
    print(f"  超参: batch={batch_size}, block={block_size}, steps={max_steps}, "
          f"grad_accum={grad_accum}, lr={lr}")

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=n_layers,
                         vocab_size=model_vocab, num_attention_heads=n_heads,
                         num_key_value_heads=n_kv_heads, rms_norm_eps=1e-5,
                         max_position_embeddings=max_seq, flash_attn=not CPU_MODE)
    model = MiniMindForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,} = {n_params / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    # warmup + cosine LR schedule
    warmup = max(3, max_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    demo_mixed_precision()

    print(f"\n═══ 训练（{max_steps} 步，每步 {grad_accum} 个 micro-batch 梯度累积） ═══")
    print(f"  技巧: AdamW + cosine(warmup={warmup}) + grad clip 1.0 + autocast(GPU)")
    losses = []
    for step in range(max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mb_loss = 0.0
        for _ in range(grad_accum):
            xb, yb = get_batch('train')
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
            else:
                _, loss = model(xb, yb)
            (loss / grad_accum).backward()
            mb_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        scheduler.step()
        losses.append(mb_loss / grad_accum)
        if step % 10 == 0 or step == max_steps - 1:
            print(f"  step {step:4d}: train loss {losses[-1]:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")
    # 前 5 步 vs 最后 5 步对比
    print(f"  📉 loss 下降: {losses[:5][0]:.4f} → {losses[-5:][0]:.4f} "
          f"（{len(losses)} 步，下降了 {losses[:5][0] - losses[-5:][0]:.4f}）")

    # ── checkpoint 保存 / 恢复 ──
    ckpt_path = os.path.join(temp_dir, 'ckpt_pretrain.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': max_steps, 'config': vars(cfg)}, ckpt_path)
    print(f"\n═══ checkpoint 保存/恢复 ═══")
    print(f"  ✅ 已保存 → {ckpt_path}（{os.path.getsize(ckpt_path) / 1024:.0f} KB）")
    print(f"     内容: model + optimizer + step + config")

    model2 = MiniMindForCausalLM(cfg).to(device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=lr)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt['model'])
    opt2.load_state_dict(ckpt['optimizer'])
    ok = all(torch.allclose(p1.detach(), p2.detach()) for p1, p2 in
             zip(model.parameters(), model2.parameters()))
    print(f"  ✅ 恢复后参数与保存时一致: {ok}（断点续训不丢进度）")

    # 续训 5 步验证
    model2.train()
    for _ in range(5):
        opt2.zero_grad(set_to_none=True)
        xb, yb = get_batch('train')
        _, loss = model2(xb, yb)
        loss.backward()
        opt2.step()
    print(f"  ✅ 续训 5 步成功，loss={loss.item():.4f}（继续下降，未报错）")

    # ── 生成演示 ──
    print("\n═══ 预训练后生成 ═══")
    prompt = "First Citizen:\n"
    gen = model2.generate(torch.tensor([enc(prompt)], dtype=torch.long, device=device),
                          max_new_tokens=120, temperature=0.8, top_k=40)
    print(f"  prompt: {prompt!r}")
    print(f"  生成  : {dec(gen[0].tolist())!r}")

    print("""
═══ 总结 ═══

预训练 = 让模型学会"补全下一个 token"。本脚本把现代训练技巧串起来：
  AdamW + cosine(warmup) → 稳；grad clip → 防爆炸；
  grad accumulation → 等效大 batch；autocast(GPU) → 加速省显存；
  checkpoint 保存/恢复 → 可断点续训。
训练 50 步后 loss 明显下降、能生成像样的英文片段，就是这套流水线的效果。

下一个脚本：SFT 监督微调（对话格式 + loss masking）。""")


if __name__ == '__main__':
    main()
