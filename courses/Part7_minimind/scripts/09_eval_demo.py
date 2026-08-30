#!/usr/bin/env python3
"""
Part 7 - 脚本 9: 三阶段验收 —— Base(pretrain) vs SFT vs DPO
目标：课程的"最终交付物"。加载 06/07/08 保存的三个 checkpoint，在同一批 prompt 上
      对比三个阶段的生成行为，并在 held-out 文本上算困惑度（ppl）——
      把"训练完成"变成"可验收的证据"。

对应教程：tutorial/05_reproduce_minimind.md 的「验收」一节。

用法：
    python 09_eval_demo.py
  - 若 temp/ 下有 ckpt_pretrain.pt / ckpt_sft.pt / ckpt_dpo.pt（跑过 06/07/08），
    直接加载对比（~10s）。
  - 若没有，自动用 60 秒级迷你训练现场造三个阶段（流程演示用，行为对比弱于全量版）。

预期行为（判断训练是否成功的对照表）：
  pretrain : 输出流利、有语言统计规律，但"答非所问"（只会续写）
  sft      : 更贴合 prompt 的"回答感"（只对回答段算 loss 的效果）
  dpo      : 风格偏向偏好数据中的 chosen（幅度小是正常的，DPO 的 lr 极小）
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CPU_MODE = not torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# ─── 模型定义：与 05_full_model.py 完全一致（保证能加载 06/07/08 的权重）───
class MiniMindConfig:
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
        if intermediate_size is None:
            intermediate_size = int((math.pi * hidden_size / 64) + 0.5) * 64
        self.intermediate_size = max(1, intermediate_size)
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.flash_attn = flash_attn


class MiniMindRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.float().type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_pos_emb(q, k, freqs_cis):
    B, T, nq, hd = q.shape
    nk = k.shape[2]
    half = hd // 2
    fc = freqs_cis[:T].view(T, 1, half)
    q_ = torch.view_as_complex(q.float().reshape(B, T, nq, half, 2))
    k_ = torch.view_as_complex(k.float().reshape(B, T, nk, half, 2))
    q_out = torch.view_as_real(q_ * fc).reshape(B, T, nq, hd)
    k_out = torch.view_as_real(k_ * fc).reshape(B, T, nk, hd)
    return q_out.type_as(q), k_out.type_as(k)


def repeat_kv(x, n_rep):
    B, T, n_kv, hd = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(B, T, n_kv, n_rep, hd).reshape(B, T, n_kv * n_rep, hd)


class MiniMindAttention(nn.Module):
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
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = ~self.causal[:T, :T]
        if self.config.flash_attn and DEVICE == 'cuda':
            attn = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        else:
            wei = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
            wei = wei.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(wei, dim=-1) @ vh
        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMindBlock(config)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, self.config.vocab_size),
                                       targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.config.max_position_embeddings:])[:, -1, :]
            logits = logits / temperature
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -float('Inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, -1), 1)), dim=1)
        return idx


# ─── 工具：分词 / 数据 / 加载或快速训练 ───────────────────
def load_tokenizer(temp_dir, script_dir):
    """与 05_full_model.py 相同：优先 BPE，回退字符级。"""
    path = os.path.join(temp_dir, 'bpe_tokenizer.json')
    if os.path.exists(path):
        try:
            from tokenizers import Tokenizer as BPETok
            tok = BPETok.from_file(path)
            return (lambda s: tok.encode(s).ids), (lambda ids: tok.decode(ids)), \
                tok.get_vocab_size(), f"BPE(vocab={tok.get_vocab_size()})"
        except Exception as e:
            print(f"  ⚠️ BPE 加载失败({e})，回退字符级")
    # 注意：数据路径从 script_dir 解析（scripts→Part7→courses→根），不要穿过 temp 拼接——
    # temp 目录不存在时路径中间断链，open 会直接 FileNotFoundError（真踩过的坑）
    with open(os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt'),
              encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return (lambda s: [stoi[c] for c in s]), (lambda ids: ''.join(itos[i] for i in ids)), \
        len(chars), f"char(vocab={len(chars)})"


def build_model(vocab, hidden=64, layers=2, heads=4, kv=2, max_pos=256):
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, vocab_size=vocab,
                         num_attention_heads=heads, num_key_value_heads=kv,
                         max_position_embeddings=max_pos, flash_attn=False)
    return MiniMindForCausalLM(cfg).to(DEVICE)


def load_stage_ckpt(path, vocab):
    """尝试加载某个阶段的 ckpt；config 词表不匹配则视为无效。"""
    if not os.path.exists(path):
        return None, None
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        cfgd = ckpt.get('config', {})
        if cfgd.get('vocab_size') != vocab:
            return None, cfgd
        model = build_model(vocab,
                            hidden=cfgd.get('hidden_size', 64),
                            layers=cfgd.get('num_hidden_layers', 2),
                            heads=cfgd.get('num_attention_heads', 4),
                            kv=cfgd.get('num_key_value_heads', 2),
                            max_pos=cfgd.get('max_position_embeddings', 256))
        model.load_state_dict(ckpt['model'])
        return model, cfgd
    except Exception as e:
        print(f"    ⚠️ 加载失败({e})")
        return None, None


def quick_pretrain(model, ids, steps=250, bs=8, seq=64, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        x = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids[i + 1:i + 1 + seq]) for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def quick_sft(model, ids, steps=200, bs=8, seq=64, lr=1e-3, prompt_len=32):
    """带 prompt-mask 的迷你 SFT：prompt=文本随机窗口前半，response=后半，只对 response 算 loss。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        x = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        y = x.clone()
        y[:, :prompt_len] = -100                     # mask 掉 prompt，只学"回答"
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def quick_dpo(model, ids, steps=100, bs=4, seq=64, lr=2e-4, beta=0.5, half=32):
    """迷你 DPO：chosen=上下文的真实后续，rejected=【错配上下文】的续写（取自
    文本另一处的片段——语法同样通顺但与 prompt 无关，这正是偏好数据"chosen/rejected"
    的本质：不是好词 vs 坏词，而是贴题 vs 不贴题。参考模型=冻结副本）。"""
    ref = build_model(model.config.vocab_size, model.config.hidden_size,
                      model.config.num_hidden_layers, model.config.num_attention_heads,
                      model.config.num_key_value_heads, model.config.max_position_embeddings)
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def seq_logp(m, x):
        logits = m(x[:, :-1])
        logp = F.log_softmax(logits, -1)
        return logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)

    for _ in range(steps):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        chosen = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        jx = torch.randint(0, len(ids) - half - 1, (bs,))   # 另一处随机起点
        wrong = torch.stack([torch.tensor(ids[j:j + half]) for j in jx]).to(DEVICE)
        rejected = torch.cat([chosen[:, :half], wrong], dim=1)   # 同 prompt + 错配续写
        with torch.no_grad():
            ref_c, ref_r = seq_logp(ref, chosen), seq_logp(ref, rejected)
        pi_c, pi_r = seq_logp(model, chosen), seq_logp(model, rejected)
        loss = -F.logsigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r))).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def heldout_ppl(model, ids, seq=128, n_batches=8, bs=4):
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_batches):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        x = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids[i + 1:i + 1 + seq]) for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        total += loss.item() * bs
        count += bs
    return math.exp(total / count)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    enc, dec, vocab, tok_name = load_tokenizer(temp_dir, script_dir)
    with open(os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt'),
              encoding='utf-8') as f:
        text = f.read()
    ids = enc(text[:400000] if CPU_MODE else text)

    print("═══ Part 7 三阶段验收 ═══")
    print(f"  device={DEVICE}  tokenizer={tok_name}")

    stages = [
        ('pretrain', 'ckpt_pretrain.pt', '续写器：只见过"下一个 token"任务'),
        ('sft', 'ckpt_sft.pt', '对话模型：只对回答段算 loss'),
        ('dpo', 'ckpt_dpo.pt', '对齐后：风格偏向偏好数据的 chosen'),
    ]
    prompts = ["First Citizen:\n", "We are accounted poor citizens,\n", "What say you?\n"]

    results = {}
    for name, fname, desc in stages:
        model, _ = load_stage_ckpt(os.path.join(temp_dir, fname), vocab)
        loaded = model is not None
        if model is None:
            print(f"\n  ⚠️ 未找到 {fname}，现场 60 秒迷你训练出 {name} 阶段（完整对比请先跑 06→07→08）")
            model = build_model(vocab)
            if name == 'pretrain':
                quick_pretrain(model, ids)
            elif name == 'sft':
                quick_pretrain(model, ids, steps=100)
                quick_sft(model, ids)
            else:
                quick_pretrain(model, ids, steps=100)
                quick_sft(model, ids, steps=80)
                quick_dpo(model, ids)
        model.eval()
        ppl = heldout_ppl(model, ids)
        results[name] = (model, ppl)
        print(f"\n── {name.upper():<9} {'(加载自 ckpt)' if loaded else '(迷你训练)'} "
              f"held-out ppl = {ppl:.2f}   [{desc}]")
        for p in prompts:
            gen = model.generate(torch.tensor([enc(p)], dtype=torch.long).to(DEVICE),
                                 max_new_tokens=60)[0].tolist()
            print(f"  P: {p.strip()[:36]!r}")
            print(f"  → {dec(gen)[len(p):].strip()[:90]!r}")

    print(f"""
═══ 预期行为对照（判断训练是否成功）═══
  pretrain : 文本流畅、有莎士比亚"剧本腔"，但面对提问只会继续"演"（续写不是回答）
  sft      : 输出对 prompt 的"回应感"更强（loss 只在回答段训练的直接效果）
  dpo      : 风格向偏好数据的 chosen 靠拢；幅度小是正常的（DPO 的 lr 刻意极小）
  ppl      : 一般 pretrain 最低（它就是干这个的）；SFT/DPO 的 held-out ppl 略升不代表变差——
             它们把容量让给了"回答行为/偏好"，这不是语言建模比赛。看行为，别只看 ppl。

═══ 下一步 ═══
  想在真实中文数据上复现完整 minimind？看 tutorial/05_reproduce_minimind.md（毕业指南）。""")


if __name__ == '__main__':
    main()
