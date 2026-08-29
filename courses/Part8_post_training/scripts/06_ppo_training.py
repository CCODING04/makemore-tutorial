#!/usr/bin/env python3
"""
Part 8 - 脚本 6: PPO 强化学习训练（GAE + Clipped Surrogate）
目标：从零实现 PPO（Proximal Policy Optimization）训练 LLM。
演示 Value Head、GAE 优势估计、Clipped Surrogate Loss、熵正则化。

覆盖知识点：
  - TransformerWithValueHead：GPT backbone + value_head（标量价值估计）
  - GAE（Generalized Advantage Estimation）：
      A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
      δ_t = r_t + γV(s_{t+1}) - V(s_t)
  - PPO Clipped Surrogate Loss：
      L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
  - Value Loss：MSE with clipping（防止价值函数剧变）
  - Entropy Bonus：鼓励探索，防止模式坍缩
  - KL 惩罚：policy vs reference 的 KL 散度，防止过度偏离

PPO 完整公式：
  L = L_policy + c1 * L_value - c2 * L_entropy
  L_policy = -E[min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)]
  L_value  = 0.5 * max((V-R)^2, (V_clip-R)^2)
  L_entropy = -Σ π(a|s) * log π(a|s)

torch API 速查：
  torch.exp(x) — 指数（用于 ratio = exp(new_logp - old_logp)）
  torch.clamp(x, lo, hi) — 截断（PPO clip 的核心）
  torch.min(a, b) — 逐元素最小值（取 surr1 和 surr2 的较小者）
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
    ppo_epochs = 3
    ppo_steps = 15
    lr = 1e-3
    rollout_len = 32
    generate_len = 16
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 4
    ppo_epochs = 4
    ppo_steps = 100
    lr = 1e-5
    rollout_len = 256
    generate_len = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# PPO 超参数
clip_eps = 0.2         # policy clip 范围
vf_clip_eps = 0.2      # value clip 范围
gamma = 1.0            # 折扣因子（LLM 通常用 1.0）
lam = 0.95             # GAE lambda
entropy_coeff = 0.01   # 熵正则化系数
kl_coeff = 0.1         # KL 惩罚系数
max_grad_norm = 1.0    # 梯度裁剪


# ─── GPT 模型（内嵌，与 01~05 等价）───────────────────────
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


# ─── TransformerWithValueHead ──────────────────────────────
class TransformerWithValueHead(nn.Module):
    """GPT backbone + Value Head — PPO 的 Actor-Critic 架构。

    架构：
      GPT.forward_hidden(idx) → hidden (B, T, n_embed)
      lm_head(hidden) → logits (B, T, vocab_size)  — Actor（策略头）
      value_head(hidden) → values (B, T)             — Critic（价值头）

    价值头的作用：
      V(s_t) = 从状态 s_t 开始，未来累积奖励的期望值
      用于 GAE 计算 advantage: A_t = r_t + γV(s_{t+1}) - V(s_t)
      A_t > 0：这个动作比平均好 → 增大其概率
      A_t < 0：这个动作比平均差 → 减小其概率

    价值头零初始化：
      训练初期 V(s) ≈ 0，避免 advantage 的初始偏差。
      与 RewardModel 的 reward_head 零初始化同理。
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        n_embed = transformer.lm_head.in_features
        self.value_head = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, 1),
        )
        # 零初始化最后一层
        last = self.value_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @property
    def context_length(self):
        return self.transformer.context_length

    def forward(self, idx):
        """返回 (logits, values)。

        logits (B, T, V) — Actor 的策略输出
        values (B, T)    — Critic 的价值估计
        """
        hidden = self.transformer.forward_hidden(idx)   # (B, T, n_embed)
        logits = self.transformer.lm_head(hidden)       # (B, T, vocab_size)
        values = self.value_head(hidden).squeeze(-1)     # (B, T)
        return logits, values


# ─── GAE ──────────────────────────────────────────────────
def compute_gae(rewards, values, values_next, resp_mask, gamma=1.0, lam=0.95):
    """GAE（Generalized Advantage Estimation）。

    公式：
      δ_t = r_t + γ * V(s_{t+1}) - V(s_t)        — TD error
      A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}       — GAE advantage
      R_t = A_t + V(s_t)                            — returns（用于 value loss）

    GAE 的权衡（gamma * lambda）：
      gamma * lambda = 0：只看单步 TD error（高 bias，低 variance）
      gamma * lambda = 1：看完整轨迹（低 bias，高 variance）
      lambda=0.95, gamma=1.0：实际中效果最好（平衡 bias 和 variance）

    实现细节：
      从后往前递推：lastgae = delta + gamma*lambda*nonterminal*lastgae
      nonterminal：response 区域内 = 1，超出 = 0（避免跨边界传播）

    返回: (advantages, returns)
      advantages (B, L) — 标准化后的优势估计（乘以 resp_mask）
      returns    (B, L) — value 目标 = advantages + values（乘以 resp_mask）
    """
    B, L = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, device=rewards.device)
    m = resp_mask.float()

    for t in reversed(range(L)):
        # nonterminal：下一个时间步是否还在 response 区域内
        nonterminal = m[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        # TD error: δ_t = r_t + γ*V(s_{t+1})*mask - V(s_t)
        delta = rewards[:, t] + gamma * values_next[:, t] * nonterminal - values[:, t]
        # 递推: A_t = δ_t + γλ * nonterminal * A_{t+1}
        lastgae = delta + gamma * lam * nonterminal * lastgae
        adv[:, t] = lastgae

    returns = adv + values
    return adv * m, returns * m


# ─── PPO Losses ───────────────────────────────────────────
def ppo_policy_loss(new_logp, old_logp, advantages, mask, clip=0.2):
    """PPO Clipped Surrogate Policy Loss。

    公式：
      ratio = exp(new_logp - old_logp)     — 新旧策略的概率比
      surr1 = ratio * A                      — 无截断
      surr2 = clip(ratio, 1-ε, 1+ε) * A     — 截断后
      L = -min(surr1, surr2)                 — 取较小值（保守更新）

    为什么 clip？
      - ratio ≈ 1：策略变化小，信任域内，直接用 ratio*A
      - ratio >> 1：策略变化太大，用 (1+ε)*A 截断，防止剧变
      - ratio << 1：策略变化太大反方向，用 (1-ε)*A 截断
      - ε=0.2：实际中效果最好（论文推荐）

    clip_fraction：被截断的比例（监控用，正常应 < 0.2）

    返回: (loss, clip_fraction)
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum())

    # 监控：被截断的比例
    clipped = ((ratio - 1.0).abs() > clip).float()
    return loss, (clipped * mask).sum() / mask.sum()


def ppo_value_loss(new_values, old_values, returns, mask, vf_clip=0.2):
    """PPO Value Loss with Clipping。

    公式：
      V_clipped = old_V + clip(new_V - old_V, -ε, ε)
      L = 0.5 * max((new_V - R)^2, (V_clipped - R)^2)

    为什么 clip value？
      - 防止价值函数剧变（类似 policy clip 的思路）
      - 如果 new_V 变化太大，用 clipped_V 计算 loss
      - 取两者最大值 = 更保守的更新

    返回: scalar loss
    """
    v_clipped = old_values + torch.clamp(new_values - old_values, -vf_clip, vf_clip)
    loss_unclipped = (new_values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * ((torch.max(loss_unclipped, loss_clipped) * mask).sum() / mask.sum())


def compute_entropy(logits, mask):
    """计算策略的熵（用于 entropy bonus）。

    公式：H(π) = -Σ_a π(a|s) * log π(a|s)

    熵越大 → 策略越"随机" → 探索越多
    熵越小 → 策略越"确定" → 可能过早收敛

    entropy_bonus = coeff * H(π) — 加到 loss 里（取负号因为要最大化熵）
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
    return (entropy * mask).sum() / mask.sum()


# ─── 奖励函数 ──────────────────────────────────────────────
def compute_reward(sequences, prompt_len):
    """简单的奖励函数（教学用）。

    实际 RLHF 中，奖励来自 Reward Model。
    这里用启发式规则模拟：
      - 基础奖励：response 长度的对数（鼓励详细回答）
      - 重复惩罚：重复 token 比例越高，惩罚越大
      - 多样性奖励：unique token 比例越高，奖励越大

    返回: per-token rewards (B, T-1)
    """
    B, T = sequences.shape
    rewards = torch.zeros(B, T - 1, device=sequences.device)

    for i in range(B):
        response = sequences[i, prompt_len:]
        resp_len = len(response)

        if resp_len == 0:
            continue

        # 基础奖励：长度（鼓励更长的回答）
        length_reward = math.log(max(resp_len, 1)) * 0.1

        # 重复惩罚（重复越多，惩罚越大）
        tokens = response.tolist()
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        diversity_bonus = unique_ratio * 0.5

        # 总奖励（分配到最后一个 response token）
        total_reward = length_reward + diversity_bonus
        last_resp_pos = min(prompt_len + resp_len - 1, T - 2)
        rewards[i, last_resp_pos] = total_reward

    return rewards


# ─── 数据加载 ──────────────────────────────────────────────
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


def load_or_create_model():
    """加载 SFT checkpoint 或快速训练一个。"""
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


# ─── Per-Token Log Probs ──────────────────────────────────
def per_token_log_probs(logits, tokens):
    """计算每个 token 的 log π(a_t | s_t)。

    logits (B, T, V) → log_softmax → gather 取实际 token 的 log-prob

    返回: (B, T-1) — 对齐 shift
    """
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    return log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)


# ─── Main ──────────────────────────────────────────────────
def main():
    print("═══ PPO 强化学习训练（GAE + Clipped Surrogate）═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  架构: embed={n_embed}, heads={n_head}, blocks={n_blocks}, ctx={context_length}")
    print(f"  PPO 超参数: clip={clip_eps}, gamma={gamma}, lambda={lam}")
    print(f"              entropy_coeff={entropy_coeff}, kl_coeff={kl_coeff}")

    # ── 1. 加载 SFT 模型 ──
    print(f"\n── Step 1: 加载 SFT 模型 ──")
    sft_model, actual_vocab, stoi = load_or_create_model()
    itos = {i: c for c, i in stoi.items()}
    n_sft_params = sum(p.numel() for p in sft_model.parameters())
    print(f"  SFT 参数量: {n_sft_params:,} = {n_sft_params / 1e6:.2f}M")

    # ── 2. 创建 TransformerWithValueHead ──
    print(f"\n── Step 2: 创建 TransformerWithValueHead ──")
    ppo_model = TransformerWithValueHead(sft_model).to(device)
    n_total = sum(p.numel() for p in ppo_model.parameters())
    n_value = sum(p.numel() for p in ppo_model.value_head.parameters())
    print(f"  Actor（lm_head）: 继承 SFT 权重")
    print(f"  Critic（value_head）: {n_value:,} 参数（新初始化）")
    print(f"  总参数: {n_total:,} = {n_total / 1e6:.2f}M")
    print(f"")
    print(f"  架构图:")
    print(f"    idx (B, T)")
    print(f"     |")
    print(f"    GPT.forward_hidden() -> hidden (B, T, n_embed)")
    print(f"     |          |")
    print(f"    lm_head    value_head")
    print(f"     |          |")
    print(f"    logits     values")
    print(f"   (Actor)    (Critic)")

    # ── 3. 创建参考模型 ──
    print(f"\n── Step 3: 创建参考模型（KL 惩罚用）──")
    ref_model = copy.deepcopy(sft_model)
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  参考模型已冻结（用于 KL 散度计算）")
    print(f"  KL(π || π_ref) = E[log π - log π_ref] — 防止策略偏离太远")

    # ── 4. GAE 公式演示 ──
    print(f"\n── Step 4: GAE（Generalized Advantage Estimation）──")
    print(f"""
  GAE 核心公式：
    δ_t = r_t + γ * V(s_{{t+1}}) - V(s_t)           -- TD error
    A_t = Σ_{{l=0}}^{{T-t}} (γλ)^l * δ_{{t+l}}       -- GAE advantage
    R_t = A_t + V(s_t)                               -- returns

  直觉：
    δ_t > 0: 这个动作比预期好 → 增大概率
    δ_t < 0: 这个动作比预期差 → 减小概率
    γλ 权衡: γλ 小 → 只看近处（高 bias），γλ 大 → 看全程（高 variance）

  gamma={gamma}, lambda={lam} → γλ={gamma*lam}""")

    # 数值演示
    print(f"\n  数值演示（5 步序列）:")
    demo_rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    demo_values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    demo_v_next = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.0]])
    demo_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    demo_adv, demo_ret = compute_gae(demo_rewards, demo_values, demo_v_next,
                                     demo_mask, gamma=gamma, lam=lam)
    print(f"    rewards:  {demo_rewards[0].tolist()}")
    print(f"    values:   {demo_values[0].tolist()}")
    print(f"    advantages: {[f'{a:.3f}' for a in demo_adv[0].tolist()]}")
    print(f"    returns:    {[f'{r:.3f}' for r in demo_ret[0].tolist()]}")

    # ── 5. PPO Loss 公式演示 ──
    print(f"\n── Step 5: PPO Loss 公式 ──")
    print(f"""
  PPO 的三个损失组件：

  1. Policy Loss（Clipped Surrogate）:
     ratio = exp(new_logp - old_logp)
     L_policy = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
     clip_eps = {clip_eps}

  2. Value Loss（with Clipping）:
     L_value = 0.5 * max((V-R)^2, (V_clip-R)^2)
     vf_clip_eps = {vf_clip_eps}

  3. Entropy Bonus:
     H(π) = -Σ π(a|s) * log π(a|s)
     鼓励探索，防止模式坍缩

  总 Loss: L = L_policy + {entropy_coeff} * (-H) + value_coeff * L_value""")

    # clip 演示
    print(f"\n  Clip 行为演示:")
    ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    for r in ratios:
        clipped = max(1 - clip_eps, min(r, 1 + clip_eps))
        bar = '#' * int(clipped * 10)
        print(f"    ratio={r:.1f} -> clipped={clipped:.1f}  {bar}")

    # ── 6. PPO 训练循环 ──
    print(f"\n── Step 6: PPO 训练（{ppo_steps} 步，每步 {ppo_epochs} epochs）──")
    print(f"  每步: generate -> reward -> GAE -> {ppo_epochs} epochs clipped update")

    optimizer = torch.optim.AdamW(ppo_model.parameters(), lr=lr)

    # 准备 prompt
    prompts = ["Hello", "The", "What", "How"] if CPU_MODE else \
              ["Explain", "What is", "Tell me", "Describe"]

    all_metrics = []

    for step in range(ppo_steps):
        ppo_model.train()

        # ── Rollout: 生成 response ──
        prompt_idx = step % len(prompts)
        prompt_text = prompts[prompt_idx]
        prompt_ids = [stoi.get(c, 0) for c in prompt_text]
        prompt_len = len(prompt_ids)

        prompt_tensor = torch.tensor([prompt_ids] * batch_size, device=device)

        with torch.no_grad():
            # 生成 response
            responses = []
            for b in range(batch_size):
                gen = sft_model.generate(
                    prompt_tensor[b:b+1],
                    max_new_tokens=generate_len,
                    temperature=0.8, top_k=10)
                responses.append(gen[0])

            # Pad 到相同长度
            max_len = max(len(r) for r in responses)
            sequences = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            for b, r in enumerate(responses):
                sequences[b, :len(r)] = r

        seq_len = sequences.shape[1]

        # ── 计算 old log probs ──
        with torch.no_grad():
            logits_old, values_old = ppo_model(sequences)
            old_log_probs = per_token_log_probs(logits_old, sequences)

            # Ref log probs（KL 惩罚）
            ref_logits, _ = ref_model(sequences[:, :-1])
            ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
            ref_token_logps = ref_log_probs.gather(
                -1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)

        # ── 奖励 ──
        rewards = compute_reward(sequences, prompt_len)

        # KL 惩罚（per-token）
        kl = old_log_probs - ref_token_logps
        kl_penalty = kl_coeff * kl
        rewards = rewards - kl_penalty * (sequences[:, 1:] != 0).float()

        # ── Response mask ──
        resp_mask = torch.zeros_like(rewards)
        for b in range(batch_size):
            resp_mask[b, prompt_len - 1:] = 1.0

        # ── GAE ──
        values_old_t = values_old[:, :-1]      # (B, T-1)
        values_next = values_old[:, 1:]         # (B, T-1)
        advantages, returns = compute_gae(
            rewards, values_old_t, values_next, resp_mask, gamma=gamma, lam=lam)

        # 标准化 advantages
        valid_adv = advantages[resp_mask > 0]
        if len(valid_adv) > 0:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std().clamp(min=1e-8)
            advantages = (advantages - adv_mean) / adv_std

        # ── PPO 多 epoch 更新 ──
        epoch_metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'clip_frac': 0}

        for epoch in range(ppo_epochs):
            # 前向
            logits_new, values_new = ppo_model(sequences)
            new_log_probs = per_token_log_probs(logits_new, sequences)

            # Policy loss
            p_loss, clip_frac = ppo_policy_loss(
                new_log_probs, old_log_probs.detach(), advantages.detach(),
                resp_mask, clip=clip_eps)

            # Value loss
            v_loss = ppo_value_loss(
                values_new[:, :-1], values_old_t.detach(), returns.detach(),
                resp_mask, vf_clip=vf_clip_eps)

            # Entropy
            entropy = compute_entropy(logits_new[:, :-1, :], resp_mask)

            # 总 loss
            total_loss = p_loss - entropy_coeff * entropy + 0.5 * v_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(ppo_model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_metrics['policy_loss'] += p_loss.item()
            epoch_metrics['value_loss'] += v_loss.item()
            epoch_metrics['entropy'] += entropy.item()
            epoch_metrics['clip_frac'] += clip_frac.item()

        # 平均 epoch 指标
        for k in epoch_metrics:
            epoch_metrics[k] /= ppo_epochs

        # 记录
        with torch.no_grad():
            avg_reward = rewards[resp_mask > 0].mean().item() if resp_mask.sum() > 0 else 0
            avg_kl = kl[resp_mask > 0].mean().item() if resp_mask.sum() > 0 else 0

        all_metrics.append({
            'reward': avg_reward, 'kl': avg_kl, **epoch_metrics
        })

        if step % 3 == 0 or step == ppo_steps - 1:
            print(f"  step {step:3d}: reward={avg_reward:+.3f}  kl={avg_kl:.4f}  "
                  f"p_loss={epoch_metrics['policy_loss']:.4f}  "
                  f"v_loss={epoch_metrics['value_loss']:.4f}  "
                  f"entropy={epoch_metrics['entropy']:.3f}  "
                  f"clip={epoch_metrics['clip_frac']:.3f}")

    # ── 7. 训练统计 ──
    print(f"\n── Step 7: 训练统计 ──")
    rewards_list = [m['reward'] for m in all_metrics]
    kl_list = [m['kl'] for m in all_metrics]
    print(f"  reward: {rewards_list[0]:+.3f} -> {rewards_list[-1]:+.3f}")
    print(f"  KL:     {kl_list[0]:.4f} -> {kl_list[-1]:.4f}")
    print(f"  clip fraction (最后): {all_metrics[-1]['clip_frac']:.3f}")
    print(f"  entropy (最后): {all_metrics[-1]['entropy']:.3f}")

    # ── 8. PPO 后生成对比 ──
    print(f"\n── Step 8: PPO 训练后生成对比 ──")
    ppo_model.eval()
    test_prompts = ["Hello", "What is"] if CPU_MODE else ["Explain AI.", "What is Python?"]
    for prompt in test_prompts:
        prompt_ids = [stoi.get(c, 0) for c in prompt]
        prompt_tensor = torch.tensor([prompt_ids], device=device)

        with torch.no_grad():
            gen_sft = sft_model.generate(prompt_tensor.clone(), max_new_tokens=20,
                                         temperature=0.8, top_k=10)
            gen_ppo = ppo_model.transformer.generate(prompt_tensor.clone(), max_new_tokens=20,
                                                     temperature=0.8, top_k=10)

        sft_text = ''.join(itos.get(i, '?') for i in gen_sft[0].tolist())
        ppo_text = ''.join(itos.get(i, '?') for i in gen_ppo[0].tolist())
        print(f"  '{prompt}':")
        print(f"    SFT: '{sft_text[:60]}'")
        print(f"    PPO: '{ppo_text[:60]}'")

    # ── 9. 保存 checkpoint ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    ppo_ckpt_path = os.path.join(temp_dir, 'ckpt_ppo.pt')

    torch.save({
        'model': ppo_model.transformer.state_dict(),
        'value_head': ppo_model.value_head.state_dict(),
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': actual_vocab, 'context_length': context_length,
        },
        'metrics': all_metrics,
    }, ppo_ckpt_path)
    print(f"\n  PPO checkpoint 已保存 -> {ppo_ckpt_path}")

    # ── 总结 ──
    print(f"""
═══ 总结 ═══

本脚本从零实现了 PPO（Proximal Policy Optimization）训练 LLM。

关键组件：

1. TransformerWithValueHead（Actor-Critic 架构）
   Actor（lm_head）：输出策略 π(a|s) → logits
   Critic（value_head）：估计状态价值 V(s) → scalar
   价值头零初始化：V(s) 初始 ≈ 0，避免 advantage 偏差

2. GAE（Generalized Advantage Estimation）
   δ_t = r_t + γV(s_{{t+1}}) - V(s_t)         -- TD error
   A_t = Σ(γλ)^l * δ_{{t+l}}                  -- 优势估计
   γλ=0.95：平衡 bias 和 variance 的最佳点

3. PPO Clipped Surrogate
   ratio = exp(new_logp - old_logp)
   L = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)
   ε=0.2：防止策略剧变的"信任域"

4. 训练流程
   generate → reward → GAE → multi-epoch clipped update
   每步生成新 response（on-policy），多 epoch 复用数据

PPO vs DPO 对比：
  PPO: 需要 reward model + on-policy sampling + value head
       优点：能优化任意 reward，理论上更好
       缺点：训练复杂，需要大量 compute
  DPO: 只需要偏好对 + off-policy
       优点：训练简单稳定
       缺点：受限于偏好数据质量

完整后训练流程：
  Pretrain (02) → SFT (03) → Reward Model (04) → PPO (06)
  或
  Pretrain (02) → SFT (03) → DPO (05)（更简单）

本教程 Part 8 的完整脚本序列：
  01_gpt_model.py      — 构建 GPT 架构
  02_pretrain.py        — 预训练
  03_sft.py             — 监督微调
  04_reward_model.py    — 奖励模型
  05_dpo_alignment.py   — DPO/ORPO/KTO 对齐
  06_ppo_training.py    — PPO 强化学习 ← 本脚本""")


if __name__ == '__main__':
    main()
