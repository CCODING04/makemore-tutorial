#!/usr/bin/env python3
"""
Part 7 - 脚本 8: DPO 直接偏好优化（Direct Preference Optimization）
目标：用 (prompt, chosen, rejected) 偏好数据对对齐模型，让回答更贴合
"人类更喜欢的回答"，而不需要训练奖励模型 + PPO。

覆盖知识点：
  - 从 RLHF 到 DPO：Bradley-Terry 模型可以把奖励函数消掉，直接用
    "当前策略 vs 冻结的参考策略" 的 log-prob 之差构造损失
  - DPO loss: -log sigmoid(beta * (log_pi(chosen) - log_pi(rejected)))
  - 参考策略冻结：pi 模型更新梯度，ref 模型 requires_grad_(False)
  - 效果验证：训练后 chosen 的 log-prob 应上升、rejected 应下降
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


# ─── 合成 DPO 偏好数据（prompt, chosen, rejected）──────────
# chosen 是"更受欢迎"的回答，rejected 是"较差"的回答
DPO_DATA = [
    ("Who is the king of Denmark in Hamlet?",
     "Claudius is the king of Denmark, uncle to Hamlet and husband of Gertrude.",
     "king. Denmark. Hamlet. uncle. Gertrude."),
    ("Why do Romeo and Juliet die?",
     "A tragic misunderstanding and poison lead both lovers to die in the tomb.",
     "they die because of bad luck."),
    ("Who is the tragic hero in Macbeth?",
     "Macbeth is the tragic hero, a brave general undone by ambition.",
     "Macbeth is the hero."),
    ("What does the ghost of Hamlet's father ask?",
     "He asks Hamlet to avenge his murder by Claudius.",
     "the ghost asks him stuff."),
    ("Who is King Lear's faithful daughter?",
     "Cordelia is the faithful daughter who speaks plainly and is banished by Lear.",
     "the youngest one."),
    ("Who finally defeats Macbeth?",
     "Macduff, who was not born of woman in the usual way, kills Macbeth.",
     "some soldier does."),
    ("Why does Shylock demand a pound of flesh?",
     "Because Antonio defaulted on the loan and Shylock's bond demands the forfeit.",
     "because he is greedy."),
    ("What is the setting of Much Ado About Nothing?",
     "The comedy is set in Messina, where Beatrice and Benedick trade witty words.",
     "a city."),
    ("Who deceives Othello?",
     "Iago deceives Othello, poisoning his mind with lies about Desdemona's fidelity.",
     "someone deceives him."),
    ("What lesson do we learn from Romeo and Juliet?",
     "Their tragedy warns how hatred between families destroys innocent love.",
     "love is hard."),
]


def make_chat_tokens(enc, answer):
    """把一条回答包进 assistant 角色，返回 token 序列（前缀含完整 prompt 部分）。"""
    segs = [
        (f"{IM_START}user\n", False),
        ("<placeholder>\n", False),
        (f"{IM_END}\n", False),
        (f"{IM_START}assistant\n", False),
        (answer + "\n", True),
        (IM_END, False),
    ]
    tokens = []
    for s, _ in segs:
        if s == "<placeholder>\n":
            continue
        tokens.extend(enc(s))
    return tokens


def sequence_logps(model, tokens, pad_id=0):
    """计算一条 token 序列的平均 log-prob（不含最后一个 token 的预测目标）。
    返回标量：对每个真实 token 位置，用模型预测的分布取实际 next-token 的 log 概率，再平均。
    pad 位置（id 为 pad_id）不计入分母。"""
    with torch.no_grad():
        logits = model(tokens)                                   # (1, T, V)
    logits = logits[:, :-1, :]                                   # (1, T-1, V)
    target = tokens[:, 1:]                                       # (1, T-1)
    logp = F.log_softmax(logits, dim=-1)
    gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    valid = (target != pad_id).float()                            # pad 不计
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return (gathered * valid).sum() / valid.sum()                 # 标量


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    enc, dec, model_vocab, tok_name = build_chat_tokenizer(temp_dir, data_path)

    # 先把 prompt 记下来（用第一条数据的 prompt 做生成演示）
    prompt_text = DPO_DATA[0][0]
    print("═══ DPO 直接偏好优化 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  分词器: {tok_name}, 模型 vocab={model_vocab}")
    print(f"  合成偏好数据: {len(DPO_DATA)} 条 (prompt, chosen, rejected)")

    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=n_layers,
                         vocab_size=model_vocab, num_attention_heads=n_heads,
                         num_key_value_heads=n_kv_heads, rms_norm_eps=1e-5,
                         max_position_embeddings=256, flash_attn=not CPU_MODE)

    # 参考模型 ref（冻结）与策略模型 pi（可训练）——两者共享同一份预训练/随机权重起点
    ref_model = MiniMindForCausalLM(cfg).to(device)
    pi_model = MiniMindForCausalLM(cfg).to(device)
    for p in ref_model.parameters():
        p.requires_grad_(False)          # ref 冻结，只当"锚点"
    ref_model.eval()

    # 尝试加载脚本 7 的 SFT checkpoint 作为起点（词表匹配时），其次是脚本 6 预训练
    ckpt_candidates = [('ckpt_sft.pt', 'SFT'), ('ckpt_pretrain.pt', '预训练')]
    loaded_ckpt = False
    for ckpt_name, ckpt_kind in ckpt_candidates:
        ckpt_path = os.path.join(temp_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt['config'].get('vocab_size') == model_vocab:
                model_sd = pi_model.state_dict()
                filtered = {k: v for k, v in ckpt['model'].items()
                            if k in model_sd and model_sd[k].shape == v.shape}
                n_skip = len(ckpt['model']) - len(filtered)
                pi_model.load_state_dict(filtered, strict=False)
                ref_model.load_state_dict(filtered, strict=False)
                loaded_ckpt = True
                print(f"  ✅ 已加载 {ckpt_kind} checkpoint 作为 DPO 起点（{ckpt_name}）")
                if n_skip:
                    print(f"     跳过 {n_skip} 个 buffer（形状不匹配）")
            else:
                print(f"  ⚠️ {ckpt_name} 词表不匹配，尝试下一个")
        except Exception as e:
            print(f"  ⚠️ 加载 {ckpt_name} 失败（{e}），尝试下一个")
        if loaded_ckpt:
            break
    if not loaded_ckpt:
        print(f"  ℹ️ 随机初始化（未找到匹配的 checkpoint）")

    # ── DPO loss 定义（本脚本核心）──────────────────────────
    def dpo_loss(pi_logps_chosen, pi_logps_rejected,
                 ref_logps_chosen, ref_logps_rejected, beta=0.1):
        """DPO loss：最大化 chosen 相对 rejected 的策略优势（以 ref 为锚点）。"""
        log_pi_chosen = pi_logps_chosen - ref_logps_chosen
        log_pi_rejected = pi_logps_rejected - ref_logps_rejected
        logits = log_pi_chosen - log_pi_rejected
        return -F.logsigmoid(beta * logits).mean()

    # 预计算所有样本的 token 序列
    chosen_tokens = [torch.tensor([make_chat_tokens(enc, c)], dtype=torch.long, device=device)
                     for _, c, _ in DPO_DATA]
    rejected_tokens = [torch.tensor([make_chat_tokens(enc, r)], dtype=torch.long, device=device)
                       for _, _, r in DPO_DATA]

    # ── DPO 前：计算初始 log-prob ──
    with torch.no_grad():
        init_c = [sequence_logps(pi_model, t).item() for t in chosen_tokens]
        init_r = [sequence_logps(pi_model, t).item() for t in rejected_tokens]
    print("\n═══ DPO 前 ═══")
    print(f"  chosen   平均 log-prob: {sum(init_c)/len(init_c):.4f}")
    print(f"  rejected 平均 log-prob: {sum(init_r)/len(init_r):.4f}")

    # ── DPO 训练 ──
    lr = 1e-3 if CPU_MODE else 1e-4
    steps = 60 if CPU_MODE else 300
    beta = 1.0
    optimizer = torch.optim.AdamW(pi_model.parameters(), lr=lr)
    print(f"\n═══ DPO 训练（{steps} 步, lr={lr}, beta={beta}）═══")
    print(f"  参考模型 ref 冻结，仅策略模型 pi 更新梯度")

    # 构造 batch：样本长度不同，用 pad_id 右填充到组内最长（-100 之外的 pad 会被 loss 的 gather 忽略）
    pad_id = 0
    def pad_batch(tokens, pad_id=pad_id):
        max_len = max(t.shape[1] for t in tokens)
        batched = torch.full((len(tokens), max_len), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(tokens):
            L = t.shape[1]
            batched[i, :L] = t[0, :L]
        return batched
    chosen_batch = pad_batch(chosen_tokens)      # (N, T_chosen)
    rejected_batch = pad_batch(rejected_tokens)  # (N, T_rejected)
    N = len(DPO_DATA)

    def batch_logps(model, batched, pad_id=pad_id):
        """对 batch 里的每个样本，计算真实 token 位置的平均 log-prob（返回 (N,)）。"""
        logits = model(batched)[:, :-1, :]                        # (N, T-1, V)
        target = batched[:, 1:]                                   # (N, T-1)
        logp = F.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (N, T-1)
        valid = (target != pad_id).float()                        # (N, T-1)
        denom = valid.sum(-1).clamp(min=1)
        return (gathered * valid).sum(-1) / denom                 # (N,)

    def batch_sft_loss(model, batched, pad_id=pad_id):
        """对 chosen 回答的监督交叉熵（SFT 正则），防止纯 DPO 让语言能力崩溃。"""
        logits = model(batched)[:, :-1, :]                        # (N, T-1, V)
        target = batched[:, 1:]                                   # (N, T-1)
        valid = (target != pad_id).view(-1)                       # 只对真实 token 计 loss
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               target.reshape(-1), reduction='none')
        return (loss * valid.float()).sum() / valid.sum().clamp(min=1)

    sft_coef = 0.1   # SFT 正则权重：轻微监督 chosen，防止纯 DPO 语言能力崩坏
    for step in range(steps):
        # 策略模型 pi 的前向（算梯度）
        logp_c = batch_logps(pi_model, chosen_batch)    # (N,)
        logp_r = batch_logps(pi_model, rejected_batch)  # (N,)
        # 参考模型 ref 的前向（不计算梯度）
        with torch.no_grad():
            ref_c = batch_logps(ref_model, chosen_batch)
            ref_r = batch_logps(ref_model, rejected_batch)
        loss = dpo_loss(logp_c, logp_r, ref_c, ref_r, beta)
        sft_loss = batch_sft_loss(pi_model, chosen_batch)   # 监督 chosen，稳住语言能力
        loss = loss + sft_coef * sft_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi_model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:4d}: dpo loss {loss.item():.4f}  "
                  f"(sft 正则 {sft_loss.item():.4f})")

    # ── DPO 后：重新计算 log-prob，对比效果 ──
    pi_model.eval()
    with torch.no_grad():
        final_c = [sequence_logps(pi_model, t).item() for t in chosen_tokens]
        final_r = [sequence_logps(pi_model, t).item() for t in rejected_tokens]
    print("\n═══ DPO 效果对比 ═══")
    avg_ic, avg_ir = sum(init_c)/len(init_c), sum(init_r)/len(init_r)
    avg_fc, avg_fr = sum(final_c)/len(final_c), sum(final_r)/len(final_r)
    margin_i, margin_f = avg_ic - avg_ir, avg_fc - avg_fr
    print(f"  chosen    log-prob: {avg_ic:.4f} → {avg_fc:.4f}  "
          f"({'📈 上升' if avg_fc > avg_ic else '❌ 未上升'})")
    print(f"  rejected  log-prob: {avg_ir:.4f} → {avg_fr:.4f}  "
          f"({'📉 下降' if avg_fr < avg_ir else '➖ 保持低位'})")
    print(f"  两者差距(chosen−rejected): {margin_i:.4f} → {margin_f:.4f}  "
          f"({'📐 拉大' if margin_f > margin_i else '❌ 未拉开'})")
    print(f"  💡 DPO 的核心不是绝对压低 rejected，而是拉开 chosen 与 rejected 的差距，"
          f"让模型明确更偏好 chosen")

    # ── 生成演示：训练后模型对 prompt 的回答 ──
    def demo(prompt, max_new=40):
        pid = enc(prompt)
        gen = pi_model.generate(torch.tensor([pid], dtype=torch.long, device=device),
                                max_new_tokens=max_new, temperature=0.3, top_k=30, top_p=0.9)
        return dec(gen[0].tolist()[len(pid):])

    prompt = (f"{IM_START}user\n{prompt_text}\n{IM_END}\n{IM_START}assistant\n")
    print(f"\n═══ DPO 后生成（prompt: {prompt_text}）═══")
    print(f"  回答: {demo(prompt)!r}")

    print("""
═══ 总结 ═══

DPO 用 (chosen, rejected) 偏好对直接优化策略，不需要奖励模型和 PPO：
  - Bradley-Terry：把"偏好"建模成对 chosen 更优的概率，从而消掉奖励函数
  - 损失 = -log sigmoid(beta * (log_pi(chosen) - log_pi(rejected)))
  - ref 模型冻结当"锚点"，防止策略跑偏太远、失去语言能力
  - 效果：chosen 的回答 log-prob 上升，rejected 的回答 log-prob 下降

至此 Part 7 的完整流水线跑通：
  tokenizer(01) → 组件(02/03/04) → 模型(05) → 预训练(06) → SFT(07) → DPO(08)
  —— 等价于 minimind 的 train_tokenizer → train_pretrain → train_full_sft → train_dpo""")

    # 保存 DPO 后模型
    out = os.path.join(temp_dir, 'ckpt_dpo.pt')
    torch.save({'model': pi_model.state_dict(), 'config': cfg.__dict__}, out)
    print(f"\n  ✅ 已保存 DPO 模型 → {out}")


if __name__ == '__main__':
    main()
