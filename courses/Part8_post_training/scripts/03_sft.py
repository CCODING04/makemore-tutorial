#!/usr/bin/env python3
"""
Part 8 - 脚本 3: SFT 监督微调（Chat Template + Prompt Masking）
目标：在预训练模型上做 SFT（Supervised Fine-Tuning），教模型"按指令回答"。
演示 Chat Template、Prompt Masking（只在 response 上算 loss）。

覆盖知识点：
  - Chat Template: <|system|>, <|user|>, <|assistant|> 结构化对话格式
  - Prompt Masking: 只在 response tokens 上计算 loss，prompt 不贡献梯度
  - SFT vs 预训练 loss 区别：
      预训练: L = -Σ_all log P(token_i | context)
      SFT:    L = -Σ_response log P(token_i | context)
  - sft_loss() 实现：shift logits + mask + cross_entropy(reduction='none')

torch API 速查：
  F.cross_entropy(reduction='none') — 逐 token CE loss（不自动求均值）
  torch.clamp(min=1.0) — 防止除零（mask 全 0 时）
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
    batch_size = 1
    max_steps = 30
    lr = 1e-3
    n_sft_samples = 20
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 8
    max_steps = 200
    lr = 1e-4
    n_sft_samples = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


# ─── GPT 模型（内嵌，与 01/02 等价）───────────────────────
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


# ─── Chat Template ─────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."


def format_chat(system, user, assistant=""):
    """格式化 Chat Template。

    GPU 模式（标准 ChatML）:
      <|system|>\\n{system}\\n<|user|>\\n{user}\\n<|assistant|>\\n{assistant}
      这是多数开源模型（ChatGLM/Qwen/Mistral）使用的格式。

    CPU 模式（简化版，适配 context_length=64）:
      Q: {user}\\nA: {assistant}
      用最简格式演示 prompt masking 的核心思想。

    特殊 token（<|system|> 等）在字符级编码下就是普通字符序列。
    """
    if CPU_MODE:
        # 简化格式：Q: {question}\nA: {response}
        # prompt 部分 = "Q: {user}\nA: "（mask=0）
        # response 部分 = "{assistant}"（mask=1）
        return f"Q: {user}\nA: {assistant}"
    else:
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"


# ─── SFT Loss（Prompt Masking）─────────────────────────────
def sft_loss(logits, tokens, loss_mask):
    """SFT loss with prompt masking — 只在 response tokens 上计算 loss。

    步骤：
      1. Shift logits by 1: 用位置 t 的 logits 预测位置 t+1 的 token
      2. 计算逐 token CE（reduction='none'）
      3. 乘以 mask（prompt 区域 = 0，response 区域 = 1）
      4. 求和 / mask.sum()（只对 response tokens 求均值）

    公式：
      标准 CE:  L = -Σ_all log P(token_i | context)
      SFT mask: L = -Σ_response log P(token_i | context)

    为什么不直接用标准 CE？
      如果不 mask，模型会浪费 capacity 去"预测 prompt"——
      但 prompt 是已知的输入，预测它没有意义。
      Mask 让梯度只流过 response tokens，聚焦在"学会回答"上。
    """
    # Step 1: Shift — 用位置 t 的 logits 预测 t+1 的 token
    logits = logits[:, :-1, :]    # (B, T-1, V)
    targets = tokens[:, 1:]       # (B, T-1)
    mask = loss_mask[:, 1:]       # (B, T-1) — mask 也要 shift 对齐

    # Step 2: 逐 token cross entropy
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="none"
    )
    ce = ce.view(B, T)  # (B, T-1)

    # Step 3: 乘以 mask（prompt 区域 loss 归零）
    ce = ce * mask

    # Step 4: 求和归一化（只除以 response token 数量）
    loss = ce.sum() / mask.sum().clamp(min=1.0)
    return loss


# ─── 数据加载 ──────────────────────────────────────────────
def load_text_data():
    """加载文本数据。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    if not os.path.exists(data_path):
        print(f"  数据文件不存在: {data_path}，使用合成数据")
        text = "Hello world. How are you? I am fine. " * 500
        return text, "合成数据"
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text, "input.txt"


def create_sft_pairs(text, n_pairs):
    """从文本创建 SFT (instruction, response) 对。

    CPU 模式：合成简单问答对（快速验证）
    GPU 模式：从文本提取句子构造 instruction-response 对（真实数据）
    """
    pairs = []

    if CPU_MODE:
        # 合成简单问答对（覆盖不同模式，让模型学到"按指令回答"）
        import random
        rng = random.Random(1337)
        static_pairs = [
            ("Say hello.", "Hello! How can I help you?"),
            ("What color is the sky?", "The sky is blue."),
            ("What is Python?", "Python is a programming language."),
            ("Tell me a fact.", "The earth orbits the sun."),
            ("Say goodbye.", "Goodbye! Have a nice day."),
        ]
        for i in range(n_pairs):
            if i % 3 == 0:
                # 加法
                x, y = rng.randint(1, 9), rng.randint(1, 9)
                pairs.append((f"What is {x}+{y}?", f"The answer is {x + y}."))
            elif i % 3 == 1:
                # 乘法
                x, y = rng.randint(1, 9), rng.randint(1, 9)
                pairs.append((f"What is {x}*{y}?", f"The result is {x * y}."))
            else:
                # 静态模板
                pairs.append(static_pairs[i % len(static_pairs)])
    else:
        # 从文本中提取句子对
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.')
                     if len(s.strip()) > 20]
        for i in range(min(n_pairs, len(sentences) - 1)):
            user = f"Summarize this: {sentences[i][:60]}"
            assistant = sentences[i + 1][:100]
            pairs.append((user, assistant))

    return pairs


# ─── 加载预训练模型 ────────────────────────────────────────
def load_or_create_model():
    """加载预训练 checkpoint，如果不存在则快速预训练。

    返回: (model, actual_vocab_size, stoi_dict)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, '..', 'temp', 'ckpt_pretrain.pt')

    text, _ = load_text_data()

    if os.path.exists(ckpt_path):
        print(f"  加载预训练 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = GPT(config['n_head'], config['n_embed'], config['context_length'],
                    config['vocab_size'], config['n_blocks']).to(device)
        model.load_state_dict(ckpt['model'])
        print(f"  加载成功，vocab={config['vocab_size']}")

        # 重建 stoi（与 checkpoint 一致）
        if config['vocab_size'] <= 256:
            chars = sorted(list(set(text)))
            stoi = {c: i for i, c in enumerate(chars)}
        else:
            # GPU 模式 tiktoken
            try:
                import tiktoken
                enc = tiktoken.get_encoding('r50k_base')
                stoi = enc  # encode 对象
            except ImportError:
                chars = sorted(list(set(text)))
                stoi = {c: i for i, c in enumerate(chars)}

        return model, config['vocab_size'], stoi

    # 快速预训练
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
        if step % 10 == 0:
            print(f"    pretrain step {step}: loss {loss.item():.4f}")

    # 保存 checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': actual_vocab, 'context_length': context_length,
        }
    }, ckpt_path)
    print(f"  快速预训练完成，已保存 checkpoint")

    return model, actual_vocab, stoi


# ─── Main ──────────────────────────────────────────────────
def main():
    print("═══ SFT 监督微调（Chat Template + Prompt Masking）═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  架构: embed={n_embed}, heads={n_head}, blocks={n_blocks}, ctx={context_length}")

    # ── 1. 加载预训练模型 ──
    print(f"\n── Step 1: 加载预训练模型 ──")
    model, actual_vocab, stoi = load_or_create_model()
    itos = {i: c for c, i in stoi.items()} if isinstance(stoi, dict) else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,} = {n_params / 1e6:.2f}M")

    # ── 2. Chat Template 演示 ──
    print(f"\n── Step 2: Chat Template ──")
    if CPU_MODE:
        print(f"  CPU 模式使用简化格式: Q: {{question}}\\nA: {{response}}")
        print(f"  （context_length=64，标准 ChatML 太长放不下）")
        print(f"  GPU 模式使用标准 ChatML:")
        print(f"    <|system|>\\n{{system}}\\n<|user|>\\n{{prompt}}\\n<|assistant|>\\n{{response}}")
    else:
        print(f"  格式: <|system|>\\n{{system}}\\n<|user|>\\n{{prompt}}\\n<|assistant|>\\n{{response}}")
        print(f"  这是简化版 ChatML，多数开源模型（ChatGLM/Qwen/Mistral）使用类似格式。")

    example = format_chat(SYSTEM_PROMPT, "What is 1+1?", "The answer is 2.")
    print(f"\n  完整示例:")
    print(f"  {example!r}")

    prompt_part = format_chat(SYSTEM_PROMPT, "What is 1+1?", "")
    print(f"\n  Prompt 部分（mask=0，不计算 loss）:")
    print(f"  {prompt_part!r}")
    print(f"  Response 部分（mask=1，计算 loss）: 'The answer is 2.'")
    print(f"  特殊 token 在字符级编码下就是普通字符序列。")

    # ── 3. 创建 SFT 数据 ──
    print(f"\n── Step 3: 创建 SFT 数据 ──")
    text, data_name = load_text_data()
    pairs = create_sft_pairs(text, n_sft_samples)
    print(f"  数据源: {data_name}")
    print(f"  生成 {len(pairs)} 个 (instruction, response) 对")
    print(f"  示例:")
    for i in range(min(3, len(pairs))):
        print(f"    [{i}] Q: {pairs[i][0]!r}")
        print(f"         A: {pairs[i][1]!r}")

    # ── 4. Prompt Masking 演示 ──
    print(f"\n── Step 4: Prompt Masking 演示 ──")
    user, assistant = pairs[0]
    full_text = format_chat(SYSTEM_PROMPT, user, assistant)
    prompt_text = format_chat(SYSTEM_PROMPT, user, "")

    # 编码
    if isinstance(stoi, dict):
        full_tokens = [stoi.get(c, 0) for c in full_text]
        prompt_tokens = [stoi.get(c, 0) for c in prompt_text]
    else:
        full_tokens = stoi.encode_ordinary(full_text)
        prompt_tokens = stoi.encode_ordinary(prompt_text)

    prompt_len = len(prompt_tokens)
    total_len = min(len(full_tokens), context_length)
    response_len = total_len - prompt_len

    print(f"  完整序列长度: {total_len}")
    print(f"  Prompt 长度: {prompt_len} (mask=0，不计算 loss)")
    print(f"  Response 长度: {response_len} (mask=1，计算 loss)")
    print(f"  Response 占比: {response_len / total_len * 100:.1f}%")

    # ── 5. Masked vs Unmasked Loss 对比 ──
    print(f"\n── Step 5: Masked vs Unmasked Loss 对比 ──")
    tokens = torch.tensor([full_tokens[:context_length]], device=device)
    mask = torch.zeros(1, total_len, device=device)
    mask[0, prompt_len:] = 1.0

    model.eval()
    with torch.no_grad():
        logits, _ = model(tokens)

        # Unmasked loss（标准 CE，所有 token 都参与）
        unmasked_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, actual_vocab),
            tokens[:, 1:].reshape(-1)
        )

        # Masked loss（SFT，只在 response 上算）
        masked_loss = sft_loss(logits, tokens, mask)

    print(f"  Unmasked loss（所有 token）: {unmasked_loss.item():.4f}")
    print(f"  Masked loss（仅 response）: {masked_loss.item():.4f}")
    print(f"  差值: {abs(unmasked_loss.item() - masked_loss.item()):.4f}")
    print(f"")
    print(f"  标准 CE:  L = -Σ_all log P(token_i | context)")
    print(f"  SFT mask: L = -Σ_response log P(token_i | context)")
    print(f"")
    print(f"  为什么需要 Prompt Masking？")
    print(f"    - 不 mask：模型浪费 capacity 去'预测 prompt'（它已经知道了）")
    print(f"    - mask 后：梯度只流过 response tokens，聚焦在'学会回答'")

    # ── 6. SFT 训练 ──
    print(f"\n── Step 6: SFT 训练（{max_steps} 步）──")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(max_steps):
        model.train()

        # 随机选一个 pair
        idx = torch.randint(0, len(pairs), (1,)).item()
        user, assistant = pairs[idx]
        full_text = format_chat(SYSTEM_PROMPT, user, assistant)
        prompt_text = format_chat(SYSTEM_PROMPT, user, "")

        # 编码
        if isinstance(stoi, dict):
            full_tokens = [stoi.get(c, 0) for c in full_text]
            prompt_len = len([stoi.get(c, 0) for c in prompt_text])
        else:
            full_tokens = stoi.encode_ordinary(full_text)
            prompt_len = len(stoi.encode_ordinary(prompt_text))

        # 截断到 context_length
        if len(full_tokens) > context_length:
            full_tokens = full_tokens[:context_length]
            prompt_len = min(prompt_len, context_length)

        tokens = torch.tensor([full_tokens], device=device)
        mask = torch.zeros(1, len(full_tokens), device=device)
        mask[0, prompt_len:] = 1.0

        logits, _ = model(tokens)
        loss = sft_loss(logits, tokens, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 10 == 0 or step == max_steps - 1:
            print(f"  step {step:4d}: loss {losses[-1]:.4f}")

    print(f"  loss 下降: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ── 7. 保存 SFT checkpoint ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    sft_ckpt_path = os.path.join(temp_dir, 'ckpt_sft.pt')

    torch.save({
        'model': model.state_dict(),
        'config': {
            'n_head': n_head, 'n_embed': n_embed, 'n_blocks': n_blocks,
            'vocab_size': actual_vocab, 'context_length': context_length,
        }
    }, sft_ckpt_path)
    print(f"  SFT checkpoint 已保存 -> {sft_ckpt_path}")

    # ── 8. SFT 后生成演示 ──
    print(f"\n── Step 7: SFT 后生成 ──")
    model.eval()

    test_pairs = [
        ("What is 3+4?", ""),
        ("Say hello.", ""),
        ("What color is the sky?", ""),
    ]

    for user, _ in test_pairs:
        prompt = format_chat(SYSTEM_PROMPT, user, "")
        if isinstance(stoi, dict):
            prompt_ids = [stoi.get(c, 0) for c in prompt]
        else:
            prompt_ids = stoi.encode_ordinary(prompt)

        with torch.no_grad():
            gen = model.generate(
                torch.tensor([prompt_ids], device=device),
                max_new_tokens=30, temperature=0.8, top_k=10
            )

        if itos:
            gen_text = ''.join(itos.get(i, '?') for i in gen[0].tolist())
        else:
            gen_text = f"token ids: {gen[0].tolist()[-10:]}"

        print(f"  Q: {user}")
        print(f"  A: {gen_text[:80]!r}")
        print()

    print(f"""
═══ 总结 ═══

SFT（监督微调）= 在"指令-回答"对上微调预训练模型。

关键技巧：
  Chat Template: <|system|>, <|user|>, <|assistant|> 结构化输入
  Prompt Masking: 只在 response 上算 loss，prompt 不贡献梯度

公式对比：
  预训练: L = -Σ_all log P(token_i | context)     -- 所有 token
  SFT:    L = -Σ_response log P(token_i | context) -- 仅 response

为什么需要 Prompt Masking？
  - 不 mask：模型浪费 capacity 去"预测 prompt"（已知输入，无意义）
  - mask 后：梯度只流过 response，聚焦在"学会按指令回答"
  - 实际效果：masked loss 通常 > unmasked loss（只看难的部分）

本脚本的 sft_loss() 实现：
  1. logits[:, :-1, :] — shift: 用位置 t 预测 t+1
  2. F.cross_entropy(reduction='none') — 逐 token CE
  3. ce * mask — prompt 区域 loss 归零
  4. ce.sum() / mask.sum() — 只对 response 求均值

下一个脚本：奖励模型（Bradley-Terry + preference pairs）。""")


if __name__ == '__main__':
    main()
