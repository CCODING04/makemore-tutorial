#!/usr/bin/env python3
"""
Part 7 - 脚本 5: 组装完整 MiniMind 模型
目标：把前面脚本的组件（RMSNorm + RoPE + GQA + SwiGLU + 权重绑定）组装成
与 minimind 架构一致的 decoder-only LLM，支持 top_k/top_p/temperature/
repetition_penalty 生成，并验证参数量与前向 shape。

架构（等价于 minimind）：
  tok_embeddings → N×MiniMindBlock → RMSNorm → lm_head（与 embedding 权重绑定）
  MiniMindBlock = pre-norm 残差：
      x = x + GQA(RMSNorm1(x))          # 通信：分组查询注意力 + RoPE
      x = x + SwiGLU(RMSNorm2(x))       # 计算：SwiGLU 前馈
  - 无 position embedding（位置信息由 RoPE 编码在 q/k 里）
  - 无 bias 的 Linear（现代 LLM 惯例）
  - tie_word_embeddings=True：lm_head 共享 embedding 权重（省一大块参数）
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

torch.manual_seed(1337)


# ─── 配置 ─────────────────────────────────────────────────
class MiniMindConfig:
    """minimind 的超参容器（字段名对齐 HuggingFace 惯例）。"""

    def __init__(self, hidden_size=768, num_hidden_layers=8, vocab_size=6400,
                 num_attention_heads=8, num_key_value_heads=4,
                 intermediate_size=None, rms_norm_eps=1e-5, rope_theta=10000.0,
                 max_position_embeddings=512, tie_word_embeddings=True,
                 flash_attn=True):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        # minimind 公式：int((pi*hidden/64)+0.5)*64（保证是 64 的倍数且≥1）
        if intermediate_size is None:
            intermediate_size = int((math.pi * hidden_size / 64) + 0.5) * 64
        self.intermediate_size = max(1, intermediate_size)
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.flash_attn = flash_attn


# ─── 基础组件（复用脚本 2/3/4）────────────────────────────
class MiniMindRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    assert dim % 2 == 0, "RoPE 需要 dim 为偶数"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_pos_emb(q, k, freqs_cis):
    B, T, nq_heads, head_dim = q.shape
    nk_heads = k.shape[2]
    half = head_dim // 2
    freqs_cis = freqs_cis[:T].view(T, 1, half)
    q_ = torch.view_as_complex(q.reshape(B, T, nq_heads, half, 2))
    k_ = torch.view_as_complex(k.reshape(B, T, nk_heads, half, 2))
    q_out = torch.view_as_real(q_ * freqs_cis).reshape(B, T, nq_heads, head_dim)
    k_out = torch.view_as_real(k_ * freqs_cis).reshape(B, T, nk_heads, head_dim)
    return q_out.type_as(q), k_out.type_as(k)


def repeat_kv(x, n_rep):
    B, T, n_kv, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :].expand(B, T, n_kv, n_rep, head_dim)
    return x.reshape(B, T, n_kv * n_rep, head_dim)


class MiniMindAttention(nn.Module):
    """GQA + RoPE，手动/Flash 双路径。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.register_buffer('freqs_cis',
                             precompute_freqs_cis(self.head_dim, config.max_position_embeddings,
                                                  config.rope_theta))
        self.register_buffer('causal', torch.tril(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings,
                       dtype=torch.bool)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, self.freqs_cis)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        q_h, k_h, v_h = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = ~self.causal[:T, :T]
        if self.config.flash_attn:
            attn = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=mask)
        else:
            wei = (q_h @ k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
            wei = wei.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            wei = F.softmax(wei, dim=-1)
            attn = wei @ v_h
        out = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMindBlock(nn.Module):
    """pre-norm 残差块：x = x + attn(rmsnorm1(x));  x = x + ffn(rmsnorm2(x))"""

    def __init__(self, config):
        super().__init__()
        self.attention = MiniMindAttention(config)
        self.feed_forward = SwiGLUFFN(config.hidden_size, config.intermediate_size)
        self.input_layernorm = MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x):
        x = x + self.attention(self.input_layernorm(x))
        x = x + self.feed_forward(self.post_attention_layernorm(x))
        return x


class MiniMindForCausalLM(nn.Module):
    """完整 decoder-only LLM（等价于 minimind）。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMindBlock(config)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight  # 权重绑定
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, repetition_penalty=1.0):
        """自回归生成，支持 temperature / top_k / top_p / repetition_penalty。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            logits = self(idx_cond)[:, -1, :] / temperature   # (B, V)
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    for tok in set(idx[b].tolist()):
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = -float('Inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ─── 分词器加载：优先 BPE，回退字符级 ─────────────────────
def load_tokenizer(temp_dir):
    """返回 (encode, decode, vocab_size, name)；encode/decode 接收/返回 Python 列表。"""
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
    # 字符级 fallback（Part 6 同款）
    with open(os.path.join(temp_dir, '..', '..', '..', 'data', 'input.txt'),
              'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    enc = lambda s: [stoi[c] for c in s]
    dec = lambda ids: ''.join(itos[i] for i in ids)
    return enc, dec, len(chars), f"char(vocab={len(chars)})"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    enc, dec, model_vocab, tok_name = load_tokenizer(temp_dir)

    print("═══ 组装 MiniMind 模型 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}")
    print(f"  分词器: {tok_name}")
    print(f"  最终 vocab_size: {model_vocab}（按实际分词器）")

    # 用实际 vocab 覆盖模板里的配置值
    cfg = MiniMindConfig(
        hidden_size=hidden_size, num_hidden_layers=n_layers, vocab_size=model_vocab,
        num_attention_heads=n_heads, num_key_value_heads=n_kv_heads,
        rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=256 if CPU_MODE else 1024,
        tie_word_embeddings=True, flash_attn=not CPU_MODE)
    print(f"  超参: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}, "
          f"intermediate={cfg.intermediate_size}")

    model = MiniMindForCausalLM(cfg)
    n_params = count_params(model)
    print(f"\n═══ 参数量 ═══")
    print(f"  MiniMindForCausalLM: {n_params:,} 参数 = {n_params / 1e6:.3f} M")
    print(f"  （lm_head 与 embedding 权重绑定 → 省了 vocab×hidden = "
          f"{model_vocab * cfg.hidden_size:,} 参数）")

    # ── 前向 shape 验证 ──
    print("\n═══ 前向 shape 验证 ═══")
    B, T = 2, 16
    idx = torch.randint(0, model_vocab, (B, T))
    logits = model(idx)
    print(f"  输入 idx: {tuple(idx.shape)} → 输出 logits: {tuple(logits.shape)} ✅")
    targets = torch.randint(0, model_vocab, (B, T))
    logits2, loss = model(idx, targets)
    print(f"  带 targets 的 loss: {loss.item():.4f}（随机初始化 ≈ ln(vocab) = {math.log(model_vocab):.4f}）")

    # ── 生成 ──
    print("\n═══ 生成（未训练，仅验证 pipeline） ═══")
    prompt = "First Citizen:\n"
    prompt_ids = enc(prompt)
    gen = model.generate(torch.tensor([prompt_ids], dtype=torch.long),
                         max_new_tokens=80, temperature=0.8, top_k=40,
                         top_p=0.9, repetition_penalty=1.1)
    print(f"  prompt: {prompt!r}")
    print(f"  生成  : {dec(gen[0].tolist())!r}")

    # ── 缩放表：CPU 小模型 → 中规模 → 官方 minimind-small ──
    print("\n═══ 参数量随规模增长 ═══")
    rows = [
        ("CPU 缩小版（本脚本）", cfg),
        ("~1M 中规模（演示）", MiniMindConfig(hidden_size=128, num_hidden_layers=4,
                                             vocab_size=6400, num_attention_heads=8,
                                             num_key_value_heads=4,
                                             max_position_embeddings=256,
                                             flash_attn=False)),
        ("minimind-small（~26M）", MiniMindConfig(hidden_size=512, num_hidden_layers=8,
                                                 vocab_size=6400, num_attention_heads=8,
                                                 num_key_value_heads=2,
                                                 max_position_embeddings=1024,
                                                 flash_attn=True)),
        ("模板 GPU 配置（v1 规模）", MiniMindConfig(hidden_size=768, num_hidden_layers=8,
                                                  vocab_size=6400, num_attention_heads=8,
                                                  num_key_value_heads=4,
                                                  max_position_embeddings=1024,
                                                  flash_attn=True)),
    ]
    for name, c in rows:
        n = count_params(MiniMindForCausalLM(c))
        print(f"  {name:<22} {n / 1e6:>8.3f} M 参数")
    print(f"  💡 minimind 官方 ~26M 用 hidden=512 / kv_heads=2；模板 GPU 配置 "
          f"(768/4) 是 v1 规模（~64M）")
    print(f"     本脚本 CPU 缩小版 {n_params / 1e6:.2f}M（模板强制 hidden=64，"
          f"~1M 对应 hidden=128）")

    print("""
═══ 总结 ═══

现代 LLM 相对 Part 6 的三处升级已全部就位：
  1. 位置编码：可学习 position embedding → RoPE（旋转进 q/k）
  2. 归一化：LayerNorm → RMSNorm（更省）
  3. FFN / 注意力：ReLU FFN → SwiGLU；MHA → GQA + KV Cache
再加上权重绑定（tie_word_embeddings）与 bias 全去掉，就是 minimind 的架构。

下一个脚本：预训练流水线（数据集 + AdamW + cosine schedule + 混合精度 + checkpoint）。""")


if __name__ == '__main__':
    main()
