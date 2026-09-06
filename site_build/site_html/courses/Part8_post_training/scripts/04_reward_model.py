#!/usr/bin/env python3
"""
Part 8 - 脚本 4: 奖励模型（Bradley-Terry）
目标：训练一个奖励模型（Reward Model），学习人类偏好排序。
演示 Bradley-Terry 偏好模型、RewardModel 架构、偏好数据生成与训练。

覆盖知识点：
  - Bradley-Terry 模型：P(A > B) = sigmoid(r(A) - r(B))
  - RewardModel：GPT backbone + reward_head（最后 token 的标量奖励）
  - reward_head 零初始化：训练初期奖励接近 0，避免初始偏差
  - 偏好对训练：L = -log sigmoid(r_chosen - r_rejected)
  - 奖励分布：chosen 和 rejected 的奖励应该逐渐分离

torch API 速查：
  nn.Linear(n_embed, 1, bias=False) — 标量奖励头（无 bias，简化）
  nn.init.zeros_() — 零初始化（训练初期奖励 ≈ 0，避免偏差）
  F.logsigmoid(x) — 数值稳定的 log sigmoid(x)（比 log(sigmoid(x)) 更稳定）
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
    max_steps = 40
    lr = 1e-3
    n_preference_pairs = 30
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 8
    max_steps = 200
    lr = 1e-4
    n_preference_pairs = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


# ─── GPT 模型（内嵌，与 01/02/03 等价）────────────────────
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
        """backbone 前向：返回 final LN 之后的 hidden state (B, T, n_embed)。
        奖励头的接入点。"""
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


# ─── RewardModel ───────────────────────────────────────────
class RewardModel(nn.Module):
    """奖励模型：GPT backbone + reward_head。

    架构：
      GPT.forward_hidden(idx) → hidden (B, T, n_embed)
      reward_head(hidden) → scalar reward per token (B, T, 1)
      取最后一个 token 的奖励作为序列的总奖励 r(x)

    为什么取最后一个 token？
      - Transformer 是因果模型，最后一个 token 的 hidden state
        包含了整个序列的信息（通过注意力机制汇总）
      - 这是 InstructGPT / ChatGPT 的标准做法

    为什么零初始化？
      - 训练初期所有奖励 ≈ 0，避免初始偏差
      - Bradley-Terry 只关心 r(A) - r(B)，零初始化意味着
        初始时 P(A > B) ≈ 0.5（无偏好），符合直觉
    """

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        n_embed = gpt.lm_head.in_features
        self.reward_head = nn.Linear(n_embed, 1, bias=False)
        # 零初始化：训练初期奖励接近 0
        nn.init.zeros_(self.reward_head.weight)

    def forward(self, idx):
        """返回最后 token 的标量奖励 r(x)。

        输入: idx (B, T)
        输出: reward (B,) — 每个序列一个标量
        """
        hidden = self.gpt.forward_hidden(idx)   # (B, T, n_embed)
        reward = self.reward_head(hidden)         # (B, T, 1)
        return reward[:, -1, 0]                   # (B,) — 取最后 token

    def token_rewards(self, idx):
        """返回每个 token 的奖励（用于诊断/可视化）。

        输入: idx (B, T)
        输出: rewards (B, T) — 每个 token 一个标量
        """
        hidden = self.gpt.forward_hidden(idx)     # (B, T, n_embed)
        return self.reward_head(hidden).squeeze(-1)  # (B, T)


# ─── Bradley-Terry Loss ────────────────────────────────────
def bradley_terry_loss(r_chosen, r_rejected):
    """Bradley-Terry 偏好损失。

    公式：L = -log sigmoid(r_chosen - r_rejected)

    直觉：
      - r_chosen > r_rejected 时：sigmoid → 1, loss → 0（正确排序）
      - r_chosen < r_rejected 时：sigmoid → 0, loss → 很大（错误排序）
      - 训练目标：让 chosen 的奖励高于 rejected

    Bradley-Terry 模型：
      P(A > B) = sigmoid(r(A) - r(B))
      这是 Luce's choice axiom 的特例，广泛用于 Elo rating、RLHF 等。

    为什么用 F.logsigmoid 而不是 log(sigmoid())？
      - logsigmoid 数值更稳定（内部用 softplus 实现，避免 log(0)）
      - torch API: F.logsigmoid(x) = -softplus(-x)
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


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


# ─── 偏好数据生成 ──────────────────────────────────────────
def generate_preference_pairs(text, stoi, n_pairs):
    """生成合成偏好对。

    用简单规则给两个 response 打分：
      - 更长的 response = chosen（信息量更大）
      - 更短的 response = rejected（信息量少）

    实际 RLHF 中，偏好对来自人类标注员。
    这里用启发式规则模拟：长度和信息量作为质量代理。
    """
    pairs = []

    if CPU_MODE:
        # 合成偏好对：好回答 vs 差回答
        templates = [
            {
                'prompt': "What is the capital of France?",
                'chosen': "The capital of France is Paris, a beautiful city.",
                'rejected': "Paris.",
            },
            {
                'prompt': "Explain gravity.",
                'chosen': "Gravity is a fundamental force that attracts objects with mass.",
                'rejected': "Things fall down.",
            },
            {
                'prompt': "What is Python?",
                'chosen': "Python is a popular programming language used for many applications.",
                'rejected': "A snake.",
            },
            {
                'prompt': "Tell me about the sun.",
                'chosen': "The sun is a star at the center of our solar system.",
                'rejected': "Hot.",
            },
            {
                'prompt': "How does photosynthesis work?",
                'chosen': "Plants use sunlight to convert CO2 and water into glucose and oxygen.",
                'rejected': "Plants eat sun.",
            },
        ]
        for i in range(n_pairs):
            t = templates[i % len(templates)]
            pairs.append(t)
    else:
        # 从文本生成偏好对
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.')
                     if len(s.strip()) > 20]
        for i in range(min(n_pairs, len(sentences) - 2)):
            prompt = f"Explain: {sentences[i][:40]}"
            # chosen: 更长、更详细
            chosen = sentences[i + 1][:100]
            # rejected: 更短、更简略
            rejected = sentences[i + 1][:30]
            pairs.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })

    return pairs


# ─── 加载 backbone 模型 ───────────────────────────────────
def load_backbone():
    """加载 SFT 或预训练模型作为 reward model 的 backbone。

    优先级：SFT checkpoint > 预训练 checkpoint > 快速预训练
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    text, _ = load_text_data()

    # 1. 尝试 SFT checkpoint
    sft_path = os.path.join(script_dir, '..', 'temp', 'ckpt_sft.pt')
    if os.path.exists(sft_path):
        print(f"  加载 SFT checkpoint: {sft_path}")
        ckpt = torch.load(sft_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = GPT(config['n_head'], config['n_embed'], config['context_length'],
                    config['vocab_size'], config['n_blocks']).to(device)
        model.load_state_dict(ckpt['model'])
        chars = sorted(list(set(text)))
        stoi = {c: i for i, c in enumerate(chars)}
        print(f"  SFT 模型加载成功，vocab={config['vocab_size']}")
        return model, config['vocab_size'], stoi

    # 2. 尝试预训练 checkpoint
    pt_path = os.path.join(script_dir, '..', 'temp', 'ckpt_pretrain.pt')
    if os.path.exists(pt_path):
        print(f"  加载预训练 checkpoint: {pt_path}")
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = GPT(config['n_head'], config['n_embed'], config['context_length'],
                    config['vocab_size'], config['n_blocks']).to(device)
        model.load_state_dict(ckpt['model'])
        chars = sorted(list(set(text)))
        stoi = {c: i for i, c in enumerate(chars)}
        print(f"  预训练模型加载成功，vocab={config['vocab_size']}")
        return model, config['vocab_size'], stoi

    # 3. 快速预训练
    print(f"  checkpoint 不存在，快速预训练...")
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    actual_vocab = len(chars)

    model = GPT(n_head, n_embed, context_length, actual_vocab, n_blocks).to(device)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(20):
        ix = torch.randint(len(data) - context_length, (batch_size,))
        xb = torch.stack([data[i:i + context_length] for i in ix]).to(device)
        yb = torch.stack([data[i + 1:i + context_length + 1] for i in ix]).to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"  快速预训练完成")
    return model, actual_vocab, stoi


# ─── Main ──────────────────────────────────────────────────
def main():
    print("═══ 奖励模型（Bradley-Terry 偏好模型）═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")

    # ── 1. 加载 backbone ──
    print(f"\n── Step 1: 加载 backbone 模型 ──")
    backbone, actual_vocab, stoi = load_backbone()
    itos = {i: c for c, i in stoi.items()}

    # ── 2. 创建 RewardModel ──
    print(f"\n── Step 2: RewardModel 架构 ──")
    reward_model = RewardModel(backbone).to(device)
    n_total = sum(p.numel() for p in reward_model.parameters())
    n_backbone = sum(p.numel() for p in backbone.parameters())
    n_head_params = sum(p.numel() for p in reward_model.reward_head.parameters())
    print(f"  Backbone 参数: {n_backbone:,}")
    print(f"  reward_head 参数: {n_head_params:,} (n_embed -> 1)")
    print(f"  总参数: {n_total:,}")
    print(f"  reward_head 初始化: nn.init.zeros_（零初始化）")
    print(f"")
    print(f"  架构图:")
    print(f"    idx (B, T)")
    print(f"     |")
    print(f"    GPT.forward_hidden() -> hidden (B, T, n_embed)")
    print(f"     |")
    print(f"    reward_head(hidden) -> (B, T, 1)")
    print(f"     |")
    print(f"    取最后 token -> r(x) (B,)")

    # ── 3. Bradley-Terry 模型演示 ──
    print(f"\n── Step 3: Bradley-Terry 偏好模型 ──")
    print(f"  公式:")
    print(f"    P(A > B) = sigmoid(r(A) - r(B))")
    print(f"    L = -log sigmoid(r_chosen - r_rejected)")
    print(f"")
    print(f"  直觉:")
    print(f"    r_chosen >> r_rejected → P → 1, loss → 0 (正确排序)")
    print(f"    r_chosen << r_rejected → P → 0, loss → 很大 (错误排序)")
    print(f"    r_chosen = r_rejected  → P = 0.5, loss = ln(2) (无法区分)")

    # 数值演示
    print(f"\n  数值示例:")
    diffs = [3.0, 1.0, 0.0, -1.0, -3.0]
    for d in diffs:
        p = torch.sigmoid(torch.tensor(d)).item()
        loss = -math.log(max(p, 1e-8))
        bar = '#' * int(p * 30)
        print(f"    r_diff={d:+.1f}  P(chosen>rejected)={p:.3f}  loss={loss:.3f}  {bar}")

    # ── 4. 生成偏好数据 ──
    print(f"\n── Step 4: 生成偏好对 ──")
    text, data_name = load_text_data()
    pairs = generate_preference_pairs(text, stoi, n_preference_pairs)
    print(f"  数据源: {data_name}")
    print(f"  生成 {len(pairs)} 个偏好对")
    print(f"  示例:")
    for i in range(min(3, len(pairs))):
        print(f"    [{i}] prompt:   {pairs[i]['prompt']!r}")
        print(f"         chosen:   {pairs[i]['chosen']!r}")
        print(f"         rejected: {pairs[i]['rejected']!r}")

    # ── 5. 训练前奖励分布 ──
    print(f"\n── Step 5: 训练前奖励分布 ──")
    reward_model.eval()
    pre_chosen, pre_rejected = [], []
    with torch.no_grad():
        for pair in pairs[:10]:
            prompt = f"<|system|>\nRate this.\n<|user|>\n{pair['prompt']}\n<|assistant|>\n"
            c_ids = [stoi.get(c, 0) for c in prompt + pair['chosen']][:context_length]
            r_ids = [stoi.get(c, 0) for c in prompt + pair['rejected']][:context_length]
            rc = reward_model(torch.tensor([c_ids], device=device)).item()
            rr = reward_model(torch.tensor([r_ids], device=device)).item()
            pre_chosen.append(rc)
            pre_rejected.append(rr)

    avg_c = sum(pre_chosen) / len(pre_chosen)
    avg_r = sum(pre_rejected) / len(pre_rejected)
    print(f"  训练前 chosen 平均奖励: {avg_c:.4f}")
    print(f"  训练前 rejected 平均奖励: {avg_r:.4f}")
    print(f"  差值: {avg_c - avg_r:.4f}")
    print(f"  零初始化 → 初始奖励接近 0，无偏好")

    # ── 6. 训练奖励模型 ──
    print(f"\n── Step 6: 训练奖励模型（{max_steps} 步）──")
    print(f"  Bradley-Terry: L = -log sigmoid(r_chosen - r_rejected)")
    print(f"  每步采样 batch_size={batch_size} 个偏好对，分别计算奖励后求 loss")
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr)
    losses = []
    chosen_rewards_log = []
    rejected_rewards_log = []

    for step in range(max_steps):
        reward_model.train()

        # 采样 batch_size 个偏好对
        batch_chosen = []
        batch_rejected = []
        for _ in range(batch_size):
            idx = torch.randint(0, len(pairs), (1,)).item()
            pair = pairs[idx]
            prompt = f"<|system|>\nRate this.\n<|user|>\n{pair['prompt']}\n<|assistant|>\n"
            c_ids = [stoi.get(c, 0) for c in prompt + pair['chosen']][:context_length]
            r_ids = [stoi.get(c, 0) for c in prompt + pair['rejected']][:context_length]
            batch_chosen.append(c_ids)
            batch_rejected.append(r_ids)

        # 逐条计算奖励（不同长度的序列，避免 padding 导致 last-token 相同）
        r_chosen_list = []
        r_rejected_list = []
        for c_ids, r_ids in zip(batch_chosen, batch_rejected):
            rc = reward_model(torch.tensor([c_ids], device=device))
            rr = reward_model(torch.tensor([r_ids], device=device))
            r_chosen_list.append(rc)
            r_rejected_list.append(rr)

        r_chosen = torch.cat(r_chosen_list, dim=0)    # (batch_size,)
        r_rejected = torch.cat(r_rejected_list, dim=0) # (batch_size,)

        loss = bradley_terry_loss(r_chosen, r_rejected)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        chosen_rewards_log.append(r_chosen.mean().item())
        rejected_rewards_log.append(r_rejected.mean().item())

        if step % 10 == 0 or step == max_steps - 1:
            print(f"  step {step:4d}: loss={losses[-1]:.4f}  "
                  f"r_chosen={chosen_rewards_log[-1]:+.3f}  "
                  f"r_rejected={rejected_rewards_log[-1]:+.3f}  "
                  f"diff={chosen_rewards_log[-1] - rejected_rewards_log[-1]:+.3f}")

    print(f"  loss 下降: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ── 7. 训练后奖励分布 ──
    print(f"\n── Step 7: 训练后奖励分布 ──")
    reward_model.eval()
    post_chosen, post_rejected = [], []
    with torch.no_grad():
        for pair in pairs[:10]:
            prompt = f"<|system|>\nRate this.\n<|user|>\n{pair['prompt']}\n<|assistant|>\n"
            c_ids = [stoi.get(c, 0) for c in prompt + pair['chosen']][:context_length]
            r_ids = [stoi.get(c, 0) for c in prompt + pair['rejected']][:context_length]
            rc = reward_model(torch.tensor([c_ids], device=device)).item()
            rr = reward_model(torch.tensor([r_ids], device=device)).item()
            post_chosen.append(rc)
            post_rejected.append(rr)

    avg_c_post = sum(post_chosen) / len(post_chosen)
    avg_r_post = sum(post_rejected) / len(post_rejected)

    print(f"  训练前: r_chosen={avg_c:.4f}, r_rejected={avg_r:.4f}, diff={avg_c - avg_r:.4f}")
    print(f"  训练后: r_chosen={avg_c_post:.4f}, r_rejected={avg_r_post:.4f}, diff={avg_c_post - avg_r_post:.4f}")
    print(f"  理想: chosen 的奖励逐渐高于 rejected，差值增大")

    # reward 分布直方图（文本版）
    print(f"\n  奖励分布直方图（训练后）:")
    all_rewards = post_chosen + post_rejected
    r_min, r_max = min(all_rewards), max(all_rewards)
    r_range = r_max - r_min if r_max > r_min else 1.0
    n_bins = 10

    print(f"  chosen:  ", end='')
    for b in range(n_bins):
        lo = r_min + b * r_range / n_bins
        hi = r_min + (b + 1) * r_range / n_bins
        count = sum(1 for r in post_chosen if lo <= r < hi)
        bar = '#' * count
        print(bar, end='')
    print()

    print(f"  rejected:", end='')
    for b in range(n_bins):
        lo = r_min + b * r_range / n_bins
        hi = r_min + (b + 1) * r_range / n_bins
        count = sum(1 for r in post_rejected if lo <= r < hi)
        bar = '#' * count
        print(bar, end='')
    print()
    print(f"  (理想: chosen 分布右移，rejected 分布左移)")

    # ── 8. Token Rewards 诊断 ──
    print(f"\n── Step 8: Token Rewards 诊断 ──")
    example_pair = pairs[0]
    prompt = f"<|system|>\nRate this.\n<|user|>\n{example_pair['prompt']}\n<|assistant|>\n"
    chosen_text = prompt + example_pair['chosen']
    c_ids = [stoi.get(c, 0) for c in chosen_text][:context_length]

    with torch.no_grad():
        token_rews = reward_model.token_rewards(torch.tensor([c_ids], device=device))

    print(f"  序列长度: {len(c_ids)}")
    print(f"  前 5 个 token 的奖励: {token_rews[0, :5].tolist()}")
    print(f"  后 5 个 token 的奖励: {token_rews[0, -5:].tolist()}")
    print(f"  最后 token 的奖励 = Bradley-Terry 用的 r(x)")
    print(f"  前面 token 的奖励是中间信号（可用于 PPO 的 token-level KL 惩罚）")

    # ── 9. 保存 checkpoint ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    rm_ckpt_path = os.path.join(temp_dir, 'ckpt_reward.pt')

    torch.save({
        'model': reward_model.state_dict(),
        'backbone': backbone.state_dict(),
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': actual_vocab, 'context_length': context_length,
        }
    }, rm_ckpt_path)
    print(f"\n  奖励模型 checkpoint 已保存 -> {rm_ckpt_path}")

    # ── 10. 用奖励模型评分 ──
    print(f"\n── Step 9: 用奖励模型评分 ──")
    reward_model.eval()
    test_prompts = [
        "What is 2+2?",
        "Tell me about AI.",
    ]
    good_responses = [
        "2+2 equals 4, a basic arithmetic fact.",
        "AI is a field of computer science that creates intelligent systems.",
    ]
    bad_responses = [
        "idk",
        "robots",
    ]

    with torch.no_grad():
        for prompt, good, bad in zip(test_prompts, good_responses, bad_responses):
            prefix = f"<|system|>\nRate.\n<|user|>\n{prompt}\n<|assistant|>\n"
            g_ids = [stoi.get(c, 0) for c in prefix + good][:context_length]
            b_ids = [stoi.get(c, 0) for c in prefix + bad][:context_length]
            r_good = reward_model(torch.tensor([g_ids], device=device)).item()
            r_bad = reward_model(torch.tensor([b_ids], device=device)).item()
            prob = torch.sigmoid(torch.tensor(r_good - r_bad)).item()
            print(f"  Q: {prompt}")
            print(f"    good answer: r={r_good:+.3f}  '{good[:40]}'")
            print(f"    bad  answer: r={r_bad:+.3f}  '{bad[:40]}'")
            print(f"    P(good > bad) = {prob:.3f}")
            print()

    print(f"""
═══ 总结 ═══

奖励模型 = GPT backbone + reward_head（标量输出）。
训练目标：Bradley-Terry 偏好模型。

关键公式：
  r(x) = reward_head(hidden_last_token)   -- 序列的标量奖励
  P(A > B) = sigmoid(r(A) - r(B))          -- Bradley-Terry 概率
  L = -log sigmoid(r_chosen - r_rejected)  -- 训练损失

RewardModel 架构：
  GPT.forward_hidden() → hidden (B, T, n_embed)
  reward_head(hidden) → (B, T, 1)
  取最后 token → r(x) (B,)

关键设计选择：
  reward_head = nn.Linear(n_embed, 1, bias=False) — 简化
  nn.init.zeros_() — 零初始化，避免初始偏差
  取最后 token — 因果 attention 汇总了整个序列信息

训练过程：
  chosen 的奖励逐渐高于 rejected
  奖励分布逐渐分离（chosen 右移，rejected 左移）

下一步（RLHF/DPO）：
  奖励模型提供偏好信号 r(x)
  PPO: 用 r(x) 作为 reward，优化策略模型
  DPO: 直接用偏好对优化策略，不需要显式奖励模型

下一个脚本：PPO / DPO 强化学习对齐。""")


if __name__ == '__main__':
    main()
