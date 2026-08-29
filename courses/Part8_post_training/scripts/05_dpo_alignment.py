#!/usr/bin/env python3
"""
Part 8 - 脚本 5: DPO + ORPO + KTO 对齐算法
目标：从零实现三种主流偏好对齐算法（DPO、ORPO、KTO），比较它们的异同。
演示如何用偏好数据直接优化策略模型，无需显式奖励模型。

覆盖知识点：
  - DPO（Direct Preference Optimization）：
      需要参考模型 + 成对数据，隐式奖励 = beta * (log_pi - log_ref)
      核心思想：把 RLHF 的 reward maximization 转化为分类问题
      L = -log sigmoid(beta * [(log_pi_chosen - log_pi_rejected) - (log_ref_chosen - log_ref_rejected)])
  - ORPO（Odds Ratio Preference Optimization）：
      无需参考模型 + 成对数据，用 odds ratio 代替 KL 约束
      L = NLL_chosen + lambda * (-log sigmoid(log_odds))
  - KTO（Kahneman-Tversky Optimization）：
      需要参考模型 + 无需成对数据（unpaired），基于前景理论
      L = desirable * (1 - sigmoid(beta * (log_ratio_chosen - kl)))
        + undesirable * (1 - sigmoid(beta * (kl - log_ratio_rejected)))
  - sequence_logprobs：计算序列的 log-prob（response 部分求和）
  - 隐式奖励提取：从 DPO 的 log-ratio 中提取隐式奖励

torch API 速查：
  F.logsigmoid(x) — 数值稳定的 log sigmoid
  torch.expm1(x) — exp(x) - 1（比 exp(x)-1 更精确，尤其 x≈0 时）
  torch.clamp(min=1) — 防止除零
"""

import os
import sys
import math
import copy
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
    batch_size = 2
    dpo_steps = 30
    lr = 1e-3
    n_pref_pairs = 20
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 4
    dpo_steps = 200
    lr = 5e-5
    n_pref_pairs = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


# ─── GPT 模型（内嵌，与 01~04 等价）───────────────────────
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


# ─── 序列 Log-Prob 计算 ───────────────────────────────────
def sequence_logprobs(model, sequences, response_mask):
    """计算序列在 response 区域的 log-prob 之和。

    核心步骤：
      1. model(sequences[:, :-1]) → logits (B, T-1, V)
      2. log_softmax(logits.float()) — 始终用 fp32 计算 log-prob（数值稳定性）
      3. gather 取实际 token 的 log-prob: logp[token_t+1]
      4. 乘以 response_mask 并求和 → 每条序列的总 log-prob

    为什么用 fp32？
      bf16 的 log_softmax 精度不够（尾数只有 7 bit），
      小概率 token 的 log-prob 可能被截断为 -inf，
      导致 DPO/KTO 的 log-ratio 出现 NaN。
      所以即使模型用 bf16 训练，log-prob 必须用 fp32。

    返回: (sum_logps, n_tokens)
      sum_logps (B,) — response 区域的 log-prob 之和
      n_tokens  (B,) — response 区域的 token 数（用于 ORPO 的均值归一化）
    """
    logits, _ = model(sequences[:, :-1])                # (B, T-1, V)
    logp = F.log_softmax(logits.float(), dim=-1)       # fp32!
    tokens = sequences[:, 1:]                           # (B, T-1)
    token_logp = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    resp_mask = response_mask[:, 1:]                    # (B, T-1)
    return (token_logp * resp_mask).sum(dim=1), resp_mask.sum(dim=1)


# ─── DPO Loss ─────────────────────────────────────────────
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO（Direct Preference Optimization）损失。

    公式推导：
      RLHF 的 KL-constrained reward maximization 的闭式解：
        r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))
      代入 Bradley-Terry P(y_w > y_l) = sigmoid(r(y_w) - r(y_l)):
        L = -log sigmoid(beta * [log(pi_chosen/pi_ref_chosen) - log(pi_rejected/pi_ref_rejected)])

    核心直觉：
      - policy_chosen_logps >> policy_rejected_logps：模型更偏好 chosen（好）
      - ref_chosen_logps ≈ ref_rejected_logps：参考模型无偏好（baseline）
      - logits = (policy差 - ref差)：减去 baseline 后的"纯信号"
      - beta 控制偏离参考模型的程度（越大越保守）

    参数：
      beta=0.1 — 默认值，较小意味着更激进的优化
      实际中 beta 在 0.1~0.5 之间调优

    返回: (loss, chosen_reward, rejected_reward)
      chosen_reward, rejected_reward 是隐式奖励，用于监控
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    # 隐式奖励 = beta * (log_pi - log_ref)
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    return loss, chosen_reward.detach(), rejected_reward.detach()


# ─── ORPO Loss ─────────────────────────────────────────────
def _log1mexp(x):
    """数值稳定的 log(1 - exp(x))，其中 x < 0。

    当 x > -0.693（即 exp(x) > 0.5）时：
      用 log(-expm1(x)) = log(1 - exp(x))，expm1 更精确
    当 x <= -0.693（即 exp(x) <= 0.5）时：
      用 log1p(-exp(x))，log1p 在参数接近 0 时更精确

    这个技巧来自 tfp.math.log1mexp，避免 log(0) 和精度损失。
    """
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def orpo_loss(policy_chosen_logps, policy_rejected_logps,
              chosen_n_tokens, rejected_n_tokens, orpo_lambda=1.0):
    """ORPO（Odds Ratio Preference Optimization）损失。

    ORPO 的核心创新：无需参考模型！
    用 odds ratio 代替 KL 约束，一步完成 SFT + 偏好对齐。

    公式推导：
      1. 归一化 log-prob（除以 token 数）→ per-token 平均
      2. 计算 odds: odds = p / (1-p)，log_odds = log_p - log(1-exp(log_p))
      3. Odds ratio: log_odds_chosen - log_odds_rejected
      4. OR loss = -log sigmoid(log_odds_ratio)
      5. 总 loss = NLL_chosen + lambda * OR_loss

    为什么叫 "odds ratio"？
      odds = P / (1-P) — 比概率比更直观
      odds_ratio = odds_chosen / odds_rejected
      让 chosen 的 odds 远大于 rejected 的 odds

    参数：
      orpo_lambda=1.0 — OR loss 的权重
      较大的 lambda 更强调偏好分离，较小的更强调 SFT 质量

    返回: (loss, chosen_mean_logp, rejected_mean_logp)
    """
    # Step 1: 归一化 — per-token 平均 log-prob
    chosen_mean = policy_chosen_logps / chosen_n_tokens.clamp(min=1)
    rejected_mean = policy_rejected_logps / rejected_n_tokens.clamp(min=1)

    # Step 2: 计算 log odds = log(p/(1-p)) = log_p - log(1-exp(log_p))
    # _log1mexp(log_p) = log(1 - exp(log_p)) = log(1-p)
    log_odds = (chosen_mean - _log1mexp(chosen_mean)) - \
               (rejected_mean - _log1mexp(rejected_mean))

    # Step 3: OR loss = -log sigmoid(log_odds)
    or_loss = -F.logsigmoid(log_odds).mean()

    # Step 4: NLL loss — 在 chosen 上做 SFT（参考模型不需要！）
    nll = -chosen_mean.mean()

    # Step 5: 总 loss = SFT + 偏好
    loss = nll + orpo_lambda * or_loss
    return loss, chosen_mean.detach(), rejected_mean.detach()


# ─── KTO Loss ──────────────────────────────────────────────
def kto_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    """KTO（Kahneman-Tversky Optimization）损失。

    KTO 的核心创新：无需成对数据！
    基于 Kahneman & Tversky 的前景理论（Prospect Theory）：
      - 人对"损失"比"收益"更敏感（损失厌恶）
      - desirable/undesirable 可以独立标注，不需要配对

    公式推导：
      1. log_ratio = log_pi - log_ref（每个样本独立计算）
      2. kl = mean(all log_ratios).clamp(min=0)（作为 baseline）
      3. chosen_loss = 1 - sigmoid(beta * (log_ratio_chosen - kl))
      4. rejected_loss = 1 - sigmoid(beta * (kl - log_ratio_rejected))
      5. loss = desirable_weight * chosen_loss + undesirable_weight * rejected_loss

    前景理论的核心思想：
      - chosen：让 log_ratio > kl（超过 baseline 才有正收益）
      - rejected：让 log_ratio < kl（低于 baseline 才能惩罚）
      - 不对称权重体现"损失厌恶"（undesirable_weight 可以更大）

    参数：
      beta=0.1 — 温度参数
      desirable_weight=1.0 — desirable 样本的权重
      undesirable_weight=1.0 — undesirable 样本的权重
      实际中 undesirable_weight 可设为 1.5~2.0（损失厌恶）

    返回: (loss, chosen_reward, rejected_reward)
    """
    # 每个样本的 log-ratio（policy vs ref）
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps

    # KL baseline：所有 log-ratio 的均值（作为参考点）
    kl = torch.cat([chosen_logratio, rejected_logratio]).mean().clamp(min=0).detach()

    # 损失：前景理论风格的非对称损失
    # chosen: 超过 baseline 才有正收益
    chosen_losses = 1.0 - torch.sigmoid(beta * (chosen_logratio - kl))
    # rejected: 低于 baseline 才能惩罚
    rejected_losses = 1.0 - torch.sigmoid(beta * (kl - rejected_logratio))

    loss = (desirable_weight * chosen_losses).mean() + \
           (undesirable_weight * rejected_losses).mean()

    return loss, (beta * chosen_logratio).detach(), (beta * rejected_logratio).detach()


# ─── 数据加载与模型初始化 ─────────────────────────────────
def load_text_data():
    """加载文本数据。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    if not os.path.exists(data_path):
        text = "Hello world. How are you? I am fine. " * 500
        return text, "合成数据"
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text, "input.txt"


def load_or_create_policy():
    """加载 SFT checkpoint 作为 policy，或快速训练一个。

    返回: (policy_model, actual_vocab, stoi)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    text, _ = load_text_data()

    # 优先加载 SFT checkpoint
    for ckpt_name in ['ckpt_sft.pt', 'ckpt_pretrain.pt']:
        ckpt_path = os.path.join(script_dir, '..', 'temp', ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"  加载 checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = ckpt['config']
            model = GPT(config['n_head'], config['n_embed'], config['context_length'],
                        config['vocab_size'], config['n_blocks']).to(device)
            model.load_state_dict(ckpt['model'])
            chars = sorted(list(set(text)))
            stoi = {c: i for i, c in enumerate(chars)}
            print(f"  加载成功，vocab={config['vocab_size']}")
            return model, config['vocab_size'], stoi

    # 快速训练
    print(f"  checkpoint 不存在，快速预训练...")
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    actual_vocab = len(chars)

    model = GPT(n_head, n_embed, context_length, actual_vocab, n_blocks).to(device)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pretrain_steps = 20 if CPU_MODE else 50
    for step in range(pretrain_steps):
        ix = torch.randint(len(data) - context_length, (batch_size,))
        xb = torch.stack([data[i:i + context_length] for i in ix]).to(device)
        yb = torch.stack([data[i + 1:i + context_length + 1] for i in ix]).to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"  快速预训练完成")
    return model, actual_vocab, stoi


def generate_preference_pairs(stoi, n_pairs):
    """生成合成偏好对。

    构造 chosen（好回答）和 rejected（差回答）：
      - chosen：更长、更详细、信息量更大
      - rejected：更短、更模糊、信息量少
    """
    pairs = []
    if CPU_MODE:
        templates = [
            {"prompt": "Explain the sun.",
             "chosen": "The sun is a star at the center of our solar system.",
             "rejected": "Hot ball."},
            {"prompt": "What is 2+2?",
             "chosen": "2+2 equals 4, basic arithmetic.",
             "rejected": "4."},
            {"prompt": "Describe water.",
             "chosen": "Water is H2O, essential for all life on earth.",
             "rejected": "Wet stuff."},
            {"prompt": "What is Python?",
             "chosen": "Python is a popular programming language for many uses.",
             "rejected": "A snake."},
            {"prompt": "Tell me about AI.",
             "chosen": "AI is a field of CS creating intelligent computer systems.",
             "rejected": "Robots."},
        ]
        for i in range(n_pairs):
            pairs.append(templates[i % len(templates)])
    else:
        # GPU 模式：更多样的模板
        good_responses = [
            "This is a comprehensive and detailed explanation of the topic.",
            "Let me provide a thorough answer with multiple perspectives.",
            "The answer involves several key concepts that are important to understand.",
        ]
        bad_responses = [
            "idk",
            "no",
            "sure whatever",
        ]
        for i in range(n_pairs):
            pairs.append({
                "prompt": f"Question number {i}: explain this topic.",
                "chosen": good_responses[i % len(good_responses)],
                "rejected": bad_responses[i % len(bad_responses)],
            })
    return pairs


def encode_pair(pair, stoi, system_prompt="You are helpful."):
    """将偏好对编码为 token 序列 + response mask。

    格式：{system}\n{prompt}\n{response}
    response_mask: prompt 部分=0, response 部分=1
    """
    prefix = f"{system_prompt}\n{pair['prompt']}\n"

    chosen_ids = [stoi.get(c, 0) for c in prefix + pair['chosen']]
    rejected_ids = [stoi.get(c, 0) for c in prefix + pair['rejected']]

    # 截断到 context_length
    chosen_ids = chosen_ids[:context_length]
    rejected_ids = rejected_ids[:context_length]

    prefix_len = len([stoi.get(c, 0) for c in prefix])

    # 构造 response mask
    chosen_mask = [0.0] * min(prefix_len, len(chosen_ids)) + \
                  [1.0] * max(0, len(chosen_ids) - prefix_len)
    rejected_mask = [0.0] * min(prefix_len, len(rejected_ids)) + \
                    [1.0] * max(0, len(rejected_ids) - prefix_len)

    # Padding 到相同长度（取较长者）
    max_len = max(len(chosen_ids), len(rejected_ids))
    chosen_ids += [0] * (max_len - len(chosen_ids))
    rejected_ids += [0] * (max_len - len(rejected_ids))
    chosen_mask += [0.0] * (max_len - len(chosen_mask))
    rejected_mask += [0.0] * (max_len - len(rejected_mask))

    return (torch.tensor([chosen_ids], device=device),
            torch.tensor([rejected_ids], device=device),
            torch.tensor([chosen_mask], device=device),
            torch.tensor([rejected_mask], device=device))


# ─── Main ──────────────────────────────────────────────────
def main():
    print("═══ DPO + ORPO + KTO 对齐算法 ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  架构: embed={n_embed}, heads={n_head}, blocks={n_blocks}, ctx={context_length}")

    # ── 1. 加载策略模型 ──
    print(f"\n── Step 1: 加载策略模型（policy）──")
    policy, actual_vocab, stoi = load_or_create_policy()
    itos = {i: c for c, i in stoi.items()}
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  参数量: {n_params:,} = {n_params / 1e6:.2f}M")

    # ── 2. 创建参考模型 ──
    print(f"\n── Step 2: 创建参考模型（ref）──")
    ref_model = copy.deepcopy(policy)
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  参考模型已冻结（不参与梯度更新）")
    print(f"  DPO/KTO 需要 ref 模型作为 baseline")
    print(f"  ORPO 不需要 ref 模型（reference-free）")

    # ── 3. 生成偏好数据 ──
    print(f"\n── Step 3: 生成偏好对 ──")
    pairs = generate_preference_pairs(stoi, n_pref_pairs)
    print(f"  生成 {len(pairs)} 个偏好对")
    for i in range(min(3, len(pairs))):
        print(f"    [{i}] prompt:   {pairs[i]['prompt']!r}")
        print(f"         chosen:   {pairs[i]['chosen']!r}")
        print(f"         rejected: {pairs[i]['rejected']!r}")

    # ── 4. 计算初始 log-probs ──
    print(f"\n── Step 4: 计算初始 log-probs ──")
    print(f"  sequence_logprobs: 对 response tokens 求 log-prob 之和")
    print(f"  使用 fp32 计算 log_softmax（bf16 精度不够）")

    policy.eval()
    ref_model.eval()
    with torch.no_grad():
        pair = pairs[0]
        c_seq, r_seq, c_mask, r_mask = encode_pair(pair, stoi)
        c_logp, c_n = sequence_logprobs(policy, c_seq, c_mask)
        r_logp, r_n = sequence_logprobs(policy, r_seq, r_mask)
        c_ref_logp, _ = sequence_logprobs(ref_model, c_seq, c_mask)
        r_ref_logp, _ = sequence_logprobs(ref_model, r_seq, r_mask)

    print(f"\n  单个偏好对的 log-probs:")
    print(f"    policy: chosen={c_logp.item():.3f}, rejected={r_logp.item():.3f}")
    print(f"    ref:    chosen={c_ref_logp.item():.3f}, rejected={r_ref_logp.item():.3f}")
    print(f"    n_tokens: chosen={c_n.item():.0f}, rejected={r_n.item():.0f}")

    # ── 5. DPO / ORPO / KTO 损失计算 ──
    print(f"\n── Step 5: 三种对齐算法的损失计算 ──")

    beta = 0.1
    orpo_lambda = 1.0

    dpo_l, dpo_c_r, dpo_r_r = dpo_loss(
        c_logp, r_logp, c_ref_logp, r_ref_logp, beta=beta)
    orpo_l, orpo_c_m, orpo_r_m = orpo_loss(
        c_logp, r_logp, c_n, r_n, orpo_lambda=orpo_lambda)
    kto_l, kto_c_r, kto_r_r = kto_loss(
        c_logp, r_logp, c_ref_logp, r_ref_logp, beta=beta)

    print(f"\n  {'算法':<8} {'损失':<10} {'chosen 信号':<14} {'rejected 信号':<14}")
    print(f"  {'─'*50}")
    print(f"  {'DPO':<8} {dpo_l.item():<10.4f} {dpo_c_r.item():<+14.4f} {dpo_r_r.item():<+14.4f}")
    print(f"  {'ORPO':<8} {orpo_l.item():<10.4f} {orpo_c_m.item():<+14.4f} {orpo_r_m.item():<+14.4f}")
    print(f"  {'KTO':<8} {kto_l.item():<10.4f} {kto_c_r.item():<+14.4f} {kto_r_r.item():<+14.4f}")

    # ── 6. 算法对比表 ──
    print(f"\n── Step 6: 算法对比 ──")
    print(f"""
  | 算法 | 参考模型 | 成对数据 | 复杂度 | 代表 |
  | DPO  | 需要     | 需要     | 中     | Zephyr, Tulu-2 |
  | ORPO | 不需要   | 需要     | 低     | Llama-3, Qwen2 |
  | KTO  | 需要     | 不需要   | 低     | 稀疏标注场景 |

  核心区别：
    DPO:  L = -log sigmoid(beta * [(log_pi_c - log_pi_r) - (log_ref_c - log_ref_r)])
          -- 需要 ref 模型作为 baseline，用 log-ratio 差值作为偏好信号
    ORPO: L = NLL + lambda * (-log sigmoid(log_odds_ratio))
          -- 不需要 ref 模型！一步完成 SFT + 偏好对齐
    KTO:  L = (1 - sigmoid(beta * (lr_c - kl))) + (1 - sigmoid(beta * (kl - lr_r)))
          -- 需要 ref 模型，但不需要成对数据（前景理论的损失厌恶）

  隐式奖励（DPO）:
    r_chosen  = beta * (log_pi_chosen  - log_ref_chosen)
    r_rejected = beta * (log_pi_rejected - log_ref_rejected)
    训练目标：增大 r_chosen - r_rejected""")

    # ── 7. DPO 训练 ──
    print(f"\n── Step 7: DPO 训练（{dpo_steps} 步）──")
    print(f"  beta={beta}, lr={lr}, batch_size={batch_size}")

    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    ref_model.eval()

    dpo_losses = []
    reward_gaps = []

    for step in range(dpo_steps):
        # 采样 batch
        batch_c_logp = []
        batch_r_logp = []
        batch_c_ref_logp = []
        batch_r_ref_logp = []
        batch_c_n = []
        batch_r_n = []

        for _ in range(batch_size):
            idx = torch.randint(0, len(pairs), (1,)).item()
            c_seq, r_seq, c_mask, r_mask = encode_pair(pairs[idx], stoi)
            c_lp, c_nt = sequence_logprobs(policy, c_seq, c_mask)
            r_lp, r_nt = sequence_logprobs(policy, r_seq, r_mask)
            with torch.no_grad():
                c_ref_lp, _ = sequence_logprobs(ref_model, c_seq, c_mask)
                r_ref_lp, _ = sequence_logprobs(ref_model, r_seq, r_mask)
            batch_c_logp.append(c_lp)
            batch_r_logp.append(r_lp)
            batch_c_ref_logp.append(c_ref_lp)
            batch_r_ref_logp.append(r_ref_lp)
            batch_c_n.append(c_nt)
            batch_r_n.append(r_nt)

        policy_chosen = torch.stack(batch_c_logp)
        policy_rejected = torch.stack(batch_r_logp)
        ref_chosen = torch.stack(batch_c_ref_logp)
        ref_rejected = torch.stack(batch_r_ref_logp)

        loss, c_rew, r_rew = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        dpo_losses.append(loss.item())
        reward_gap = (c_rew.mean() - r_rew.mean()).item()
        reward_gaps.append(reward_gap)

        if step % 10 == 0 or step == dpo_steps - 1:
            print(f"  step {step:4d}: loss={dpo_losses[-1]:.4f}  "
                  f"r_chosen={c_rew.mean():+.3f}  "
                  f"r_rejected={r_rew.mean():+.3f}  "
                  f"gap={reward_gap:+.3f}")

    print(f"  loss: {dpo_losses[0]:.4f} -> {dpo_losses[-1]:.4f}")
    print(f"  reward gap: {reward_gaps[0]:+.4f} -> {reward_gaps[-1]:+.4f}")
    if reward_gaps[-1] > reward_gaps[0]:
        print(f"  reward gap 增大 — 模型学会了区分 chosen 和 rejected！")
    else:
        print(f"  reward gap 未增大 — 数据量或训练步数可能不足")

    # ── 8. 对比训练后三种算法的隐式奖励 ──
    print(f"\n── Step 8: 训练后对比三种算法 ──")
    policy.eval()
    with torch.no_grad():
        c_logp, c_n = sequence_logprobs(policy, c_seq, c_mask)
        r_logp, r_n = sequence_logprobs(policy, r_seq, r_mask)
        c_ref_lp, _ = sequence_logprobs(ref_model, c_seq, c_mask)
        r_ref_lp, _ = sequence_logprobs(ref_model, r_seq, r_mask)

    dpo_l2, dpo_c2, dpo_r2 = dpo_loss(
        c_logp, r_logp, c_ref_lp, r_ref_lp, beta=beta)
    orpo_l2, orpo_c2, orpo_r2 = orpo_loss(
        c_logp, r_logp, c_n, r_n, orpo_lambda=orpo_lambda)
    kto_l2, kto_c2, kto_r2 = kto_loss(
        c_logp, r_logp, c_ref_lp, r_ref_lp, beta=beta)

    print(f"\n  {'算法':<8} {'loss':<10} {'chosen':<12} {'rejected':<12} {'gap':<10}")
    print(f"  {'─'*54}")
    print(f"  {'DPO':<8} {dpo_l2.item():<10.4f} {dpo_c2.item():<+12.4f} "
          f"{dpo_r2.item():<+12.4f} {(dpo_c2-dpo_r2).item():<+10.4f}")
    print(f"  {'ORPO':<8} {orpo_l2.item():<10.4f} {orpo_c2.item():<+12.4f} "
          f"{orpo_r2.item():<+12.4f} {(orpo_c2-orpo_r2).item():<+10.4f}")
    print(f"  {'KTO':<8} {kto_l2.item():<10.4f} {kto_c2.item():<+12.4f} "
          f"{kto_r2.item():<+12.4f} {(kto_c2-kto_r2).item():<+10.4f}")

    # ── 9. 生成对比 ──
    print(f"\n── Step 9: DPO 训练后生成对比 ──")
    policy.eval()
    test_prompts = ["Hello", "What is"] if CPU_MODE else ["Explain AI.", "What is Python?"]
    for prompt in test_prompts:
        prompt_ids = [stoi.get(c, 0) for c in prompt]
        with torch.no_grad():
            gen = policy.generate(
                torch.tensor([prompt_ids], device=device),
                max_new_tokens=20, temperature=0.8, top_k=10)
        gen_text = ''.join(itos.get(i, '?') for i in gen[0].tolist())
        print(f"  '{prompt}' -> '{gen_text[:60]}'")

    # ── 10. 保存 checkpoint ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    dpo_ckpt_path = os.path.join(temp_dir, 'ckpt_dpo.pt')

    torch.save({
        'model': policy.state_dict(),
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': actual_vocab, 'context_length': context_length,
        },
        'dpo_losses': dpo_losses,
        'reward_gaps': reward_gaps,
    }, dpo_ckpt_path)
    print(f"\n  DPO checkpoint 已保存 -> {dpo_ckpt_path}")

    # ── 总结 ──
    print(f"""
═══ 总结 ═══

本脚本从零实现了三种偏好对齐算法：

1. DPO（Direct Preference Optimization）
   - 核心：把 RLHF 的 reward maximization 转化为分类问题
   - 公式：L = -log sigmoid(beta * [(log_pi_c - log_pi_r) - (log_ref_c - log_ref_r)])
   - 隐式奖励：r(x,y) = beta * (log_pi - log_ref)
   - 优点：训练稳定，不需要 online sampling
   - 代表：Zephyr, Tulu-2

2. ORPO（Odds Ratio Preference Optimization）
   - 核心：无需参考模型，用 odds ratio 代替 KL 约束
   - 公式：L = NLL + lambda * (-log sigmoid(log_odds_ratio))
   - 优点：一步完成 SFT + 偏好对齐，节省显存
   - 代表：Llama-3, Qwen2

3. KTO（Kahneman-Tversky Optimization）
   - 核心：基于前景理论，无需成对数据
   - 公式：L = desirable_loss + undesirable_loss（非对称）
   - 优点：适合稀疏标注场景（只有好/坏标签，没有配对）
   - 损失厌恶：undesirable_weight 可以设更大

关键实现细节：
  sequence_logprobs: fp32 log_softmax（bf16 精度不够）
  _log1mexp: 数值稳定的 log(1-exp(x))（防止 log(0)）
  零初始化 + 深拷贝：policy 和 ref 初始相同

下一个脚本：PPO 强化学习训练（GAE + Clipped Surrogate + Value Head）。""")


if __name__ == '__main__':
    main()
