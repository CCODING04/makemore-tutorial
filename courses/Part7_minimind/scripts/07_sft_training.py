#!/usr/bin/env python3
"""
Part 7 - 脚本 7: SFT 监督微调（Supervised Fine-Tuning）
目标：用合成 Q&A 数据把预训练模型微调成"会对话"的助手，核心是
Chat Template 与 Loss Masking（只对 assistant 回答算 loss）。

覆盖知识点：
  - Chat Template：用 <|im_start|>/<|im_end|> 标记对话轮次（minimind 同款），
    把 user 提问 + assistant 回答拼成一条序列
  - Loss Masking：只对 assistant 部分的 token 计算 loss，user 部分与
    特殊 token 一律置 -100（CrossEntropyLoss 自动忽略）。这样模型学会
    "回答用户"，而不是去预测用户的提问。
  - 继续训练：优先加载脚本 6 的预训练 checkpoint（词表匹配时），否则随机初始化
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
torch.manual_seed(1337)

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


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
        def __init__(self, hidden_size=64, num_hidden_layers=2, vocab_size=67,
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

        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
            for _ in range(max_new_tokens):
                logits = self(idx[:, -self.cfg.max_position_embeddings:])[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                if top_p is not None:
                    sp, si = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sp, dim=-1), dim=-1)
                    sp[cum - F.softmax(sp, dim=-1) > top_p] = -float('Inf')
                    logits = sp.scatter(1, si, sp)
                idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
            return idx

    return MiniMindConfig, MiniMindForCausalLM


_MOD = load_model_module()
if _MOD is not None:
    MiniMindConfig = _MOD.MiniMindConfig
    MiniMindForCausalLM = _MOD.MiniMindForCausalLM
else:
    MiniMindConfig, MiniMindForCausalLM = build_fallback()


# ─── 对话分词器：优先 BPE，回退字符级（含特殊 token）─────
def build_chat_tokenizer(temp_dir, data_path):
    bpe_path = os.path.join(temp_dir, 'bpe_tokenizer.json')
    if os.path.exists(bpe_path):
        try:
            from tokenizers import Tokenizer as BPE_Tokenizer
            tok = BPE_Tokenizer.from_file(bpe_path)
            return (lambda s: tok.encode(s).ids), (lambda ids: tok.decode(ids)), \
                tok.get_vocab_size(), f"BPE(vocab={tok.get_vocab_size()})"
        except Exception as e:
            print(f"  ⚠️  加载 BPE tokenizer 失败（{e}），回退字符级+特殊token")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab = [IM_START, IM_END] + list(chars)          # 特殊 token 占低 id
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    specials = [IM_START, IM_END]

    def enc(s):
        ids, i = [], 0
        while i < len(s):
            hit = False
            for sp in specials:
                if s.startswith(sp, i):
                    ids.append(stoi[sp]); i += len(sp); hit = True; break
            if not hit:
                ids.append(stoi[s[i]]); i += 1
        return ids

    def dec(ids):
        return ''.join(itos[i] for i in ids)
    return enc, dec, len(vocab), f"char+special(vocab={len(vocab)})"


# ─── 合成 SFT 数据（莎士比亚 Q&A）─────────────────────────
SFT_DATA = [
    ("Who is the king of Denmark in Hamlet?",
     "Claudius is the king of Denmark. He is the uncle of Hamlet and married Gertrude."),
    ("What is the famous question Hamlet asks in his soliloquy?",
     "To be, or not to be, that is the question."),
    ("Who kills Hamlet at the end of the play?",
     "Laertes wounds Hamlet with a poisoned sword, and Hamlet dies from the poison."),
    ("What does Romeo do when he sees Juliet at the balcony?",
     "Romeo declares his love to Juliet and they agree to marry in secret."),
    ("Why do Romeo and Juliet die in the tragedy?",
     "A tragic misunderstanding and poison lead both to take their own lives in the tomb."),
    ("Who is the tragic hero in Macbeth?",
     "Macbeth is the tragic hero, a brave general undone by ambition and the witches' prophecy."),
    ("What do the three witches prophesy to Macbeth?",
     "They greet him as Thane of Cawdor and predict he will become king of Scotland."),
    ("Who is the main character in Othello?",
     "Othello, a Moorish general in Venice, whose jealousy destroys his wife Desdemona."),
    ("Who deceives Othello in the play?",
     "Iago deceives Othello, poisoning his mind with lies about Desdemona's fidelity."),
    ("Who is the merchant of Venice in Shakespeare's play?",
     "Antonio is the merchant of Venice, who borrows money from Shylock."),
    ("Why does Shylock demand a pound of flesh?",
     "Because Antonio defaulted on the loan, and Shylock's bond demands the forfeit."),
    ("What are the three witches known as in Macbeth?",
     "They are the Weird Sisters, who set the tragedy in motion with their prophecy."),
    ("Who is King Lear's faithful daughter?",
     "Cordelia is the faithful daughter who speaks plainly and is banished by Lear."),
    ("Why does Lear divide his kingdom?",
     "He wishes to divide it among his daughters according to their declared love."),
    ("Who is the prince of Denmark in the tragedy Hamlet?",
     "Hamlet is the prince of Denmark, son of the murdered king and Gertrude."),
    ("What does the ghost of Hamlet's father ask of Hamlet?",
     "He asks Hamlet to avenge his murder by Claudius."),
    ("Who helps Macbeth become king?",
     "Lady Macbeth encourages him and helps him murder King Duncan."),
    ("What is the setting of Much Ado About Nothing?",
     "The comedy is set in Messina, where Beatrice and Benedick trade witty words."),
    ("Who finally defeats Macbeth in battle?",
     "Macduff, who was not born of woman in the usual way, kills Macbeth."),
    ("What lesson do we learn from Romeo and Juliet?",
     "Their tragedy warns how hatred between families destroys innocent love."),
]


def make_chat_tokens(enc, user_text, assistant_text):
    """拼接一条 chat 序列，返回 (tokens, is_assistant_mask)。"""
    segs = [
        (f"{IM_START}user\n", False),
        (user_text + "\n", False),
        (f"{IM_END}\n", False),
        (f"{IM_START}assistant\n", False),
        (assistant_text + "\n", True),          # 只有 assistant 回答计入 loss
        (IM_END, False),
    ]
    tokens, mask = [], []
    for s, is_asst in segs:
        ids = enc(s)
        tokens.extend(ids)
        mask.extend([is_asst] * len(ids))
    return tokens, mask


def collate_batch(samples, max_len, pad_id):
    B = len(samples)
    T = min(max_len, max(len(t) for t, _ in samples))
    toks = torch.full((B, T), pad_id, dtype=torch.long)
    masks = torch.zeros((B, T), dtype=torch.bool)
    for i, (t, m) in enumerate(samples):
        L = min(len(t), T)
        toks[i, :L] = torch.tensor(t[:L])
        masks[i, :L] = torch.tensor(m[:L], dtype=torch.bool)
    return toks, masks


def masked_loss(model, toks, masks):
    """只对 assistant 位置的 next-token 计算交叉熵（shift 一个位置）。"""
    logits = model(toks)                                     # (B, T, V)
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = toks[:, 1:].reshape(-1)
    shift_mask = masks[:, 1:].reshape(-1)                    # 目标是 assistant 内容才算
    labels = shift_labels.clone()
    labels[~shift_mask] = -100                                # 忽略 user/特殊/padding
    return F.cross_entropy(shift_logits, labels), shift_mask.float().mean()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    enc, dec, model_vocab, tok_name = build_chat_tokenizer(temp_dir, data_path)
    pad_id = 0  # <|im_start|> 作为 padding（loss mask 会忽略）

    # 构造数据集
    dataset = [make_chat_tokens(enc, q, a) for q, a in SFT_DATA]
    max_len = max(len(t) for t, _ in dataset)
    print("═══ SFT 监督微调 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  分词器: {tok_name}, 模型 vocab={model_vocab}")
    print(f"  合成数据: {len(SFT_DATA)} 条莎士比亚 Q&A, 最长 {max_len} token")

    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=n_layers,
                         vocab_size=model_vocab, num_attention_heads=n_heads,
                         num_key_value_heads=n_kv_heads, rms_norm_eps=1e-5,
                         max_position_embeddings=max_len + 16, flash_attn=not CPU_MODE)
    model = MiniMindForCausalLM(cfg).to(device)

    # 加载预训练 checkpoint（词表匹配时）
    ckpt_path = os.path.join(temp_dir, 'ckpt_pretrain.pt')
    loaded_ckpt = False
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt['config'].get('vocab_size') == model_vocab:
                # 预训练模型 max_position_embeddings(64) 比本脚本序列长度小，
                # freqs_cis/causal 这类 buffer 形状不一致，只加载形状匹配的键
                model_sd = model.state_dict()
                filtered = {k: v for k, v in ckpt['model'].items()
                            if k in model_sd and model_sd[k].shape == v.shape}
                n_skip = len(ckpt['model']) - len(filtered)
                model.load_state_dict(filtered, strict=False)
                loaded_ckpt = True
                print(f"  ✅ 已加载预训练 checkpoint 继续训练（{os.path.basename(ckpt_path)}）")
                if n_skip:
                    print(f"     跳过 {n_skip} 个 buffer（长度不匹配，已按本序列长度重新初始化）")
            else:
                print(f"  ⚠️ checkpoint 词表({ckpt['config'].get('vocab_size')}) ≠ "
                      f"当前({model_vocab})，改用随机初始化")
        except Exception as e:
            print(f"  ⚠️ 加载 checkpoint 失败（{e}），随机初始化")
    if not loaded_ckpt:
        print(f"  ℹ️ 随机初始化（未找到匹配的预训练 checkpoint）")

    # ── 训练前生成 ──
    def demo(prompt, max_new=40):
        pid = enc(prompt)
        gen = model.generate(torch.tensor([pid], dtype=torch.long, device=device),
                             max_new_tokens=max_new, temperature=0.3, top_k=30, top_p=0.9)
        # 只解码"新生成"的 token。不能按字符数切片：BPE 解码器不会把
        # <|im_start|>/<|im_end|> 还原成文本，解码出的 prompt 比原文短。
        return dec(gen[0].tolist()[len(pid):])

    prompt = (f"{IM_START}user\nWho is the tragic hero in Macbeth?\n{IM_END}\n"
              f"{IM_START}assistant\n")
    print("\n═══ 训练前生成 ═══")
    print(f"  prompt: {prompt!r}")
    print(f"  回答  : {demo(prompt)!r}")

    # ── 训练 ──
    lr = 3e-3 if CPU_MODE else 3e-4
    steps = 300 if CPU_MODE else 500
    batch_size = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\n═══ SFT 训练（{steps} 步, batch={batch_size}, lr={lr}）═══")
    print(f"  Loss Masking: 只有 assistant 回答的 token 计入 loss，"
          f"user/特殊 token 置 -100")
    for step in range(steps):
        model.train()
        idx = torch.randint(0, len(dataset), (batch_size,))
        batch = [dataset[i] for i in idx.tolist()]
        toks, masks = collate_batch(batch, max_len, pad_id)
        toks, masks = toks.to(device), masks.to(device)
        loss, asst_ratio = masked_loss(model, toks, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}: sft loss {loss.item():.4f}  "
                  f"(assistant 占比 {asst_ratio.item() * 100:.0f}%)")

    # ── 训练后生成 ──
    model.eval()
    print("\n═══ 训练后生成（同一条 prompt） ═══")
    for q in ["Who is the tragic hero in Macbeth?",
              "Why do Romeo and Juliet die in the tragedy?",
              "Who is King Lear's faithful daughter?"]:
        prompt = f"{IM_START}user\n{q}\n{IM_END}\n{IM_START}assistant\n"
        print(f"  Q: {q}")
        print(f"  A: {demo(prompt)!r}")

    # 保存 SFT 模型（供脚本 8 DPO 使用）
    out = os.path.join(temp_dir, 'ckpt_sft.pt')
    torch.save({'model': model.state_dict(), 'config': cfg.__dict__}, out)
    print(f"\n  ✅ 已保存 SFT 模型 → {out}")

    print("""
═══ 总结 ═══

SFT 把"补全器"变成"对话助手"：
  - Chat Template 标记角色（<|im_start|>user / assistant）
  - Loss Masking 只对 assistant 回答计 loss（user 部分置 -100）
  - 训练前模型只会乱写；训练后能"复述"出数据里对应的标准回答
    （本脚本 20 条合成 Q&A = 记忆型微调，演示流程；规模大了才是真对齐）

下一个脚本：DPO 直接偏好优化（让回答更贴合"好回答"）。""")


if __name__ == '__main__':
    main()
