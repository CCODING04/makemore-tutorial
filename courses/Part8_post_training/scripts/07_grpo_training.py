#!/usr/bin/env python3
"""
Part 8 - 脚本 7: GRPO 强化学习训练（DeepSeek-R1 风格）
目标：从零实现 GRPO（Group Relative Policy Optimization），与 PPO 对比。
演示 Group Advantage 估计、k3 KL 估计器、Token-level Clipped Surrogate。

覆盖知识点：
  - GRPO（Group Relative Policy Optimization）：
      不需要 Value Network！用同一 prompt 下 G 个采样的平均奖励作基线。
      A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
      DeepSeek-R1 的选择：简单、省显存、效果好。
  - k3 KL 估计器（Schulman's unbiased estimator）：
      KL(π || π_ref) = exp(log_ref - log_new) - (log_ref - log_new) - 1
      非负、无偏、per-token 计算。
  - Token-level Clipped Surrogate：
      与 PPO 相同的 clip 机制，但 advantage 来自 group-relative 而非 GAE。
  - RLVR（RL with Verifiable Rewards）：
      奖励来自可验证的规则（如数学题答案正确=1，错误=0）。
      无需训练 Reward Model，直接用 verifier 判定。

GRPO vs PPO：
  PPO:  需要 Value Network 估计基线 V(s) → 额外显存和计算
  GRPO: 用组内多个采样的平均奖励作基线 → 不需要 Value Network
  DeepSeek-R1 的选择：简单、省显存、效果好

torch API 速查：
  torch.exp(x) — 指数（用于 ratio = exp(new_logp - old_logp)）
  torch.clamp(x, lo, hi) — 截断（PPO clip 的核心）
  torch.std(x) — 标准差（用于 group advantage 归一化）
"""

import os
import sys
import re
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
    grpo_steps = 20
    lr = 1e-3
    n_prompts = 10
    group_size = 4
    generate_len = 16
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 4
    grpo_steps = 100
    lr = 1e-5
    n_prompts = 50
    group_size = 8
    generate_len = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# GRPO 超参数
clip_eps = 0.2         # policy clip 范围
kl_coef = 0.04         # KL 惩罚系数
max_grad_norm = 1.0    # 梯度裁剪


# ─── GPT 模型（内嵌，与 01~06 等价）───────────────────────
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


# ─── GRPO 核心函数 ─────────────────────────────────────────
def group_advantages(rewards, group_size, eps=1e-4):
    """Group-relative advantage: (r - group_mean) / (group_std + eps)。

    GRPO 的核心创新 —— 不需要 Value Network！
    同一个 prompt 下采样 G 个回答，用组内平均奖励作基线。

    公式：
      A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

    rewards: (num_prompts * group_size,) laid out group-contiguously.
    即 [p0_g0, p0_g1, ..., p0_gG-1, p1_g0, p1_g1, ...]

    直觉：
      A_i > 0: 这个回答比同组平均好 → 增大概率
      A_i < 0: 这个回答比同组平均差 → 减小概率
      A_i = 0: 和平均一样 → 不更新

    与 PPO 的区别：
      PPO:  A = r + γV(s') - V(s)  — 需要 Value Network 估计 V(s)
      GRPO: A = (r - group_mean) / group_std  — 直接用采样统计量
    """
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1)


def k3_kl(new_logp, ref_logp):
    """Per-token unbiased, non-negative KL estimator (Schulman's k3)。

    公式：KL(π || π_ref) = exp(log_ref - log_new) - (log_ref - log_new) - 1

    为什么用 k3 而不是简单的 log_new - log_ref？
      - k3 是无偏估计（unbiased）
      - k3 保证非负（KL 散度 >= 0）
      - 数值稳定（不需要 clamp）

    数学推导：
      KL(p||q) = E_p[log(p/q)] = E_p[exp(log q - log p) * (log p - log q)]
               = E_p[exp(d) * (-d)]  where d = log q - log p
               = E_p[-d * exp(d)]
      但 k3 = exp(d) - d - 1 >= 0 (by convexity of exp)

    返回: per-token KL (B, L)
    """
    diff = ref_logp - new_logp
    return torch.exp(diff) - diff - 1.0


def grpo_loss(new_logp, old_logp, ref_logp, advantages, resp_mask, clip=0.2, kl_coef=0.04):
    """Token-level clipped surrogate + KL penalty。

    与 PPO 的 clipped surrogate 几乎相同，但 advantage 来自 group-relative。

    公式：
      ratio = exp(new_logp - old_logp)        — 新旧策略比
      surr1 = ratio * A                        — 无截断
      surr2 = clip(ratio, 1-ε, 1+ε) * A       — 截断后
      surrogate = min(surr1, surr2)            — 保守更新
      kl = k3_kl(new_logp, ref_logp)           — KL 惩罚
      L = -(surrogate - kl_coef * kl)          — 最终 loss

    参数：
      new_logp/old_logp/ref_logp: (B, L) per-token log-probs
      advantages: (B,) one scalar per completion（来自 group_advantages）
      resp_mask: (B, L) bool over response tokens
      clip=0.2: PPO-style clipping
      kl_coef=0.04: KL penalty coefficient

    返回: (loss, stats_dict)
    """
    adv = advantages[:, None]  # broadcast over tokens: (B, 1) -> (B, L)
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    surrogate = torch.min(surr1, surr2)
    kl = k3_kl(new_logp, ref_logp)
    per_token = surrogate - kl_coef * kl
    mask_f = resp_mask.float()
    loss = -(per_token * mask_f).sum() / mask_f.sum().clamp(min=1)
    stats = {
        "kl": (kl * mask_f).sum().item() / mask_f.sum().clamp(min=1).item(),
        "clipfrac": (((ratio - 1.0).abs() > clip).float() * mask_f).sum().item() / mask_f.sum().clamp(min=1).item(),
    }
    return loss, stats


def compute_logprobs(model, sequences, response_mask, temperature=1.0):
    """Compute per-token log-probs for sequences. Shape (B, T-1)。

    Teacher-forced log-prob 计算：
      1. model(sequences[:, :-1]) → logits (B, T-1, V)
      2. log_softmax(logits / temperature) — 温度缩放
      3. gather 取实际 token 的 log-prob
      4. 乘以 response_mask（只计算 response 区域）

    返回: (B, T-1) per-token log-probs
    """
    logits, _ = model(sequences[:, :-1])
    logp = F.log_softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    tokens = sequences[:, 1:]
    token_logp = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return token_logp * response_mask[:, 1:]


# ─── RLVR Verifier ─────────────────────────────────────────
def verify_answer(predicted, expected):
    """Check if predicted answer matches expected (simple numeric comparison)。

    RLVR（RL with Verifiable Rewards）的核心思想：
      - 奖励来自可验证的规则，不需要训练 Reward Model
      - 数学题：答案正确 = 1，错误 = 0
      - 代码题：通过测试用例 = 1，失败 = 0

    这是 DeepSeek-R1 的关键创新之一：
      用规则验证器替代人类标注的奖励模型 → 更便宜、更可靠

    返回: bool
    """
    numbers = re.findall(r'-?\d+\.?\d*', str(predicted))
    if not numbers:
        return False
    try:
        return abs(float(numbers[-1]) - float(expected)) < 0.01
    except Exception:
        return False


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
    """加载 SFT/DPO checkpoint 作为 policy，或快速训练一个。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    text, _ = load_text_data()

    for ckpt_name in ['ckpt_dpo.pt', 'ckpt_sft.pt', 'ckpt_pretrain.pt']:
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


# ─── 数学题生成 ─────────────────────────────────────────────
def generate_math_prompts(n_prompts, stoi):
    """生成合成数学题 prompt + 期望答案。

    CPU: 简单加减法（a+b=?, a-b=?）
    GPU: 更多样的算术题

    RLVR 场景：答案是可验证的 → reward = 1 if correct, 0 if wrong
    """
    import random
    random.seed(1337)
    prompts = []
    for i in range(n_prompts):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(['+', '-'])
        if op == '+':
            answer = a + b
            prompt_text = f"{a}+{b}="
        else:
            if b > a:
                a, b = b, a
            answer = a - b
            prompt_text = f"{a}-{b}="
        prompt_ids = [stoi.get(c, 0) for c in prompt_text]
        prompts.append({
            'text': prompt_text,
            'ids': prompt_ids,
            'answer': answer,
        })
    return prompts


# ─── Main ──────────────────────────────────────────────────
def main():
    print("═══ GRPO 强化学习训练（DeepSeek-R1 风格）═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  架构: embed={n_embed}, heads={n_head}, blocks={n_blocks}, ctx={context_length}")
    print(f"  GRPO 超参数: group_size={group_size}, clip={clip_eps}, kl_coef={kl_coef}")

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
    print(f"  参考模型已冻结（用于 KL 散度计算）")
    print(f"  KL(π || π_ref) — 防止策略偏离太远")

    # ── 3. GRPO 核心思想 ──
    print(f"\n── Step 3: GRPO 核心思想 ──")
    print(f"""
  GRPO vs PPO：

  PPO 需要 Value Network 估计基线 V(s):
    A_t = r_t + γV(s_{{t+1}}) - V(s_t)
    → 需要额外的 value_head（~50% 额外参数）
    → 需要 GAE 递推计算
    → 训练更复杂

  GRPO 用组内多个采样的平均奖励作基线:
    同一个 prompt 采样 G={group_size} 个回答
    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
    → 不需要 Value Network！
    → 不需要 GAE！
    → 更简单，更省显存

  DeepSeek-R1 的选择：GRPO
    原因：简单、省显存、在推理任务上效果好""")

    # ── 4. 数值演示：Group Advantage ──
    print(f"\n── Step 4: Group Advantage 数值演示 ──")
    print(f"  假设 prompt='3+5=' 采样 G={group_size} 个回答:")
    demo_rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])  # 2 个正确，2 个错误
    demo_adv = group_advantages(demo_rewards, group_size=4)
    print(f"  rewards:    {demo_rewards.tolist()}")
    print(f"  group_mean: {demo_rewards.mean():.3f}")
    print(f"  group_std:  {demo_rewards.std():.3f}")
    print(f"  advantages: {[f'{a:+.3f}' for a in demo_adv.tolist()]}")
    print(f"")
    print(f"  正确答案 advantage > 0 → 增大概率")
    print(f"  错误答案 advantage < 0 → 减小概率")

    # ── 5. k3 KL 估计器演示 ──
    print(f"\n── Step 5: k3 KL 估计器 ──")
    print(f"  公式: KL = exp(log_ref - log_new) - (log_ref - log_new) - 1")
    print(f"  特性: 非负、无偏、per-token 计算")
    demo_new_logp = torch.tensor([-1.0, -2.0, -0.5])
    demo_ref_logp = torch.tensor([-1.0, -1.0, -2.0])
    demo_kl = k3_kl(demo_new_logp, demo_ref_logp)
    print(f"  new_logp:  {demo_new_logp.tolist()}")
    print(f"  ref_logp:  {demo_ref_logp.tolist()}")
    print(f"  k3_kl:     {[f'{k:.4f}' for k in demo_kl.tolist()]}")
    print(f"  注意: new_logp == ref_logp 时 KL=0（策略没有偏离）")

    # ── 6. 数学题生成 ──
    print(f"\n── Step 6: 生成数学题 prompt（RLVR 场景）──")
    math_prompts = generate_math_prompts(n_prompts, stoi)
    print(f"  生成 {len(math_prompts)} 个数学题")
    for i in range(min(5, len(math_prompts))):
        print(f"    [{i}] '{math_prompts[i]['text']}'  期望答案: {math_prompts[i]['answer']}")

    # ── 7. GRPO 训练循环 ──
    print(f"\n── Step 7: GRPO 训练（{grpo_steps} 步，每步 G={group_size} 采样）──")
    print(f"  每步: 对每个 prompt 采样 G 个回答 → verify → group advantage → update")

    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    all_metrics = []

    for step in range(grpo_steps):
        # 选择一批 prompt
        batch_prompts = []
        for i in range(batch_size):
            idx = (step * batch_size + i) % len(math_prompts)
            batch_prompts.append(math_prompts[idx])

        # ── 对每个 prompt 采样 G 个回答 ──
        all_sequences = []
        all_resp_masks = []
        all_rewards = []

        for prompt_info in batch_prompts:
            prompt_ids = prompt_info['ids']
            prompt_len = len(prompt_ids)
            prompt_tensor = torch.tensor([prompt_ids] * group_size, device=device)

            # 采样 G 个回答
            with torch.no_grad():
                generated = policy.generate(
                    prompt_tensor,
                    max_new_tokens=generate_len,
                    temperature=0.8, top_k=10,
                )

            # 验证每个回答 → binary reward
            for g in range(group_size):
                gen_ids = generated[g].tolist()
                # 提取 response 部分（跳过 prompt）
                resp_ids = gen_ids[prompt_len:]
                resp_text = ''.join(itos.get(tid, '?') for tid in resp_ids)

                # RLVR: 用 verifier 检查答案
                is_correct = verify_answer(resp_text, prompt_info['answer'])
                reward = 1.0 if is_correct else 0.0
                all_rewards.append(reward)

                # 构造序列 + response mask
                seq = torch.tensor(gen_ids, device=device)
                all_sequences.append(seq)

        # Pad sequences 到相同长度
        max_len = max(len(s) for s in all_sequences)
        sequences = torch.zeros(len(all_sequences), max_len, dtype=torch.long, device=device)
        resp_masks = torch.zeros(len(all_sequences), max_len, device=device)
        for i, seq in enumerate(all_sequences):
            sequences[i, :len(seq)] = seq
            prompt_len_i = len(batch_prompts[i // group_size]['ids'])
            resp_masks[i, prompt_len_i:] = 1.0

        rewards = torch.tensor(all_rewards, device=device)

        # ── 计算 Group Advantage ──
        advantages = group_advantages(rewards, group_size=group_size)

        # ── 计算 log-probs (new, old, ref) ──
        # new: 当前 policy 的 log-probs（需要梯度）
        new_logp = compute_logprobs(policy, sequences, resp_masks)
        # old: 生成时的 log-probs（不需要梯度）
        with torch.no_grad():
            old_logp = compute_logprobs(policy, sequences, resp_masks)
        # ref: 参考模型的 log-probs（不需要梯度）
        with torch.no_grad():
            ref_logp = compute_logprobs(ref_model, sequences, resp_masks)

        # ── GRPO Loss ──
        loss, stats = grpo_loss(
            new_logp, old_logp, ref_logp,
            advantages, resp_masks[:, 1:],
            clip=clip_eps, kl_coef=kl_coef,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        # 记录指标
        mean_reward = rewards.mean().item()
        accuracy = (rewards > 0.5).float().mean().item()
        all_metrics.append({
            'reward': mean_reward,
            'accuracy': accuracy,
            'kl': stats['kl'],
            'clipfrac': stats['clipfrac'],
            'loss': loss.item(),
        })

        if step % 5 == 0 or step == grpo_steps - 1:
            print(f"  step {step:3d}: reward={mean_reward:.3f}  acc={accuracy:.3f}  "
                  f"kl={stats['kl']:.4f}  clip={stats['clipfrac']:.3f}  "
                  f"loss={loss.item():.4f}")

    # ── 8. 训练统计 ──
    print(f"\n── Step 8: 训练统计 ──")
    rewards_list = [m['reward'] for m in all_metrics]
    acc_list = [m['accuracy'] for m in all_metrics]
    kl_list = [m['kl'] for m in all_metrics]
    print(f"  reward:   {rewards_list[0]:.3f} -> {rewards_list[-1]:.3f}")
    print(f"  accuracy: {acc_list[0]:.3f} -> {acc_list[-1]:.3f}")
    print(f"  KL:       {kl_list[0]:.4f} -> {kl_list[-1]:.4f}")
    print(f"  clip fraction (最后): {all_metrics[-1]['clipfrac']:.3f}")

    if acc_list[-1] > acc_list[0]:
        print(f"  准确率提升 -- GRPO 有效！模型学会了做数学题")
    else:
        print(f"  准确率未提升 -- 数据量或训练步数可能不足")

    # ── 9. 训练后生成对比 ──
    print(f"\n── Step 9: GRPO 训练后生成对比 ──")
    policy.eval()
    test_prompts_info = math_prompts[:3]
    for prompt_info in test_prompts_info:
        prompt_ids = prompt_info['ids']
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            gen = policy.generate(prompt_tensor, max_new_tokens=10, temperature=0.8, top_k=10)
        resp_ids = gen[0].tolist()[len(prompt_ids):]
        resp_text = ''.join(itos.get(tid, '?') for tid in resp_ids)
        is_correct = verify_answer(resp_text, prompt_info['answer'])
        mark = "OK" if is_correct else "WRONG"
        print(f"  '{prompt_info['text']}' -> '{resp_text}' [{mark}]")

    # ── 10. 保存 checkpoint ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    grpo_ckpt_path = os.path.join(temp_dir, 'ckpt_grpo.pt')

    torch.save({
        'model': policy.state_dict(),
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': actual_vocab, 'context_length': context_length,
        },
        'metrics': all_metrics,
    }, grpo_ckpt_path)
    print(f"\n  GRPO checkpoint 已保存 -> {grpo_ckpt_path}")

    # ── 11. PPO vs GRPO 对比表 ──
    print(f"\n── Step 11: PPO vs GRPO 对比 ──")
    print(f"""
  | 维度       | PPO                          | GRPO                         |
  |------------|------------------------------|------------------------------|
  | 基线估计   | Value Network V(s)           | 组内平均奖励                 |
  | 额外网络   | 需要 value_head              | 不需要                       |
  | 显存       | 更多（policy + value + ref） | 更少（policy + ref）         |
  | 优势计算   | GAE (γ, λ)                   | Group normalization          |
  | KL 惩罚    | per-token KL penalty         | k3 KL estimator              |
  | 采样方式   | 每步生成一次                 | 每个 prompt 生成 G 次        |
  | 代表       | InstructGPT, ChatGPT         | DeepSeek-R1                  |

  GRPO 的公式总结：
    1. 对每个 prompt 采样 G 个回答: y_1, ..., y_G ~ π(·|x)
    2. 用 verifier 打分: r_i = verify(y_i, answer)
    3. 计算 group advantage: A_i = (r_i - mean) / std
    4. 计算 log-probs: new, old, ref
    5. Clipped surrogate + KL penalty:
       L = -min(ratio*A, clip(ratio)*A) + kl_coef * KL(π||π_ref)
    6. 梯度更新 policy""")

    # ── 总结 ──
    print(f"""
═══ 总结 ═══

本脚本从零实现了 GRPO（Group Relative Policy Optimization）。

关键组件：

1. Group Advantage（不需要 Value Network）
   同一个 prompt 采样 G 个回答，用组内统计量作基线
   A_i = (r_i - group_mean) / (group_std + eps)
   省掉了 PPO 的 value_head（~50% 额外参数）

2. k3 KL 估计器（Schulman's unbiased estimator）
   KL = exp(log_ref - log_new) - (log_ref - log_new) - 1
   非负、无偏、数值稳定

3. RLVR（RL with Verifiable Rewards）
   奖励来自可验证的规则（数学题答案正确=1，错误=0）
   不需要训练 Reward Model → 更便宜、更可靠
   DeepSeek-R1 用此方法训练推理能力

4. Token-level Clipped Surrogate
   与 PPO 相同的 clip 机制，但 advantage 来自 group-relative
   L = -min(ratio*A, clip(ratio)*A) + kl_coef * KL

为什么 DeepSeek-R1 选择 GRPO？
  1. 简单：不需要 Value Network，不需要 GAE
  2. 省显存：少一个 value_head 网络
  3. 效果好：在推理任务上表现优异
  4. RLVR 兼容：数学/代码题天然可验证

完整后训练流程：
  Pretrain (02) → SFT (03) → Reward Model (04) → PPO (06)
  Pretrain (02) → SFT (03) → DPO (05)
  Pretrain (02) → SFT (03) → GRPO (07) ← 本脚本（最简单）

本教程 Part 8 的完整脚本序列：
  01_gpt_model.py      — 构建 GPT 架构
  02_pretrain.py        — 预训练
  03_sft.py             — 监督微调
  04_reward_model.py    — 奖励模型
  05_dpo_alignment.py   — DPO/ORPO/KTO 对齐
  06_ppo_training.py    — PPO 强化学习
  07_grpo_training.py   — GRPO 强化学习 ← 本脚本""")


if __name__ == '__main__':
    main()
