#!/usr/bin/env python3
"""
Part 12 - 脚本 01: 手写"LoRA SFT 微型管线"——LLaMA-Factory 自动化的到底是什么
目标：在一个玩具模型上，把 LLaMA-Factory 一个 yaml 背后的完整流水线手写一遍：
      chat template 构造 → prompt masking → LoRA 注入 → SFT 训练循环 → 合并（merge）。
      跑通后再看 02 章的 yaml，每个字段你都能指出"对应我手写的哪几行"。

对应教程：tutorial/01_handwritten_sft_lora.md
运行（~40 秒，CPU/GPU 均可；无任何外部依赖）：
    python 01_handwritten_sft_lora.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── 0. 玩具世界：词表里有 4 个"特殊 token"+ 若干普通词 ───
SPECIALS = ["<|im_start|>", "<|im_end|>", "user:", "assistant:"]
WORDS = [f"w{i}" for i in range(20)]
VOCAB = SPECIALS + WORDS
STOI = {t: i for i, t in enumerate(VOCAB)}
IM_START, IM_END = STOI["<|im_start|>"], STOI["<|im_end|>"]


def encode(s): return [STOI[t] for t in s.split()]


def decode(ids): return " ".join(VOCAB[i] for i in ids)


# ─── 1. chat template（LLaMA-Factory 的 template 机制做的事）────────
def build_sample(instruction, response):
    """构造 (input_ids, labels)，labels 只对 response 段生效（prompt masking）。
    LLaMA-Factory 的 template/  目录管的就是这段逻辑。"""
    prompt = f"<|im_start|> user: {instruction} <|im_end|>"
    full = prompt + f" assistant: {response} <|im_end|>"
    ids = torch.tensor(encode(full), dtype=torch.long)
    n_prompt = len(encode(prompt))
    labels = ids.clone()
    labels[:n_prompt] = -100               # prompt 段不算 loss（Part 8 02 章）
    return ids, labels


# ─── 2. 玩具模型（Part 8 10 章同款结构）────────────────────
class Block(nn.Module):
    def __init__(self, n_embed, n_head, ctx):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 3 * n_embed), nn.GELU(),
                                 nn.Linear(3 * n_embed, n_embed))
        self.register_buffer('mask', torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=self.mask[:T, :T])
        return x + self.mlp(self.ln2(x + a))


class ToyGPT(nn.Module):
    def __init__(self, vocab, n_embed=96, n_head=4, n_layer=2, ctx=32):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, n_embed)
        self.pos = nn.Embedding(ctx, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, ctx) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))


# ─── 3. LoRA（Part 8 10 章同款：W 冻结 + BA 旁路，B 零初始化）───
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        out_f, in_f = linear.weight.shape
        self.A = nn.Parameter(torch.randn(r, in_f) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.alpha, self.r = alpha, r

    def forward(self, x):
        return self.linear(x) + (self.alpha / self.r) * (x @ self.A.T) @ self.B.T


def apply_lora(model, r=4, alpha=8.0):
    """对应 yaml 的 lora_target / lora_rank / lora_alpha 三个字段。
    注入 MLP 的两个 Linear（真实 LlamaFactory 的 lora_target 常填 q_proj,v_proj 或 all）。"""
    n = 0
    for block in model.blocks:
        block.mlp[0] = LoRALinear(block.mlp[0], r=r, alpha=alpha)
        block.mlp[2] = LoRALinear(block.mlp[2], r=r, alpha=alpha)
        n += 2
    return n


# ─── 4. SFT 数据（20 条"身份+算术"指令，呼应 LLaMA-Factory 的 identity 数据集）───
def make_sft_data(n=64):
    """玩具任务"回声指令"：instruction = 两个随机词，response = 复述第一个词。
    SFT 要学的是两件事：① 严格按 chat 格式在 assistant 段作答；② 任务映射本身。"""
    import random
    rng = random.Random(1337)
    data = []
    word_ids = [STOI[w] for w in WORDS]
    for _ in range(n):
        a, b = rng.choice(word_ids), rng.choice(word_ids)
        data.append(build_sample(f"{WORDS[a - len(SPECIALS)]} {WORDS[b - len(SPECIALS)]}",
                                 WORDS[a - len(SPECIALS)]))
    return data


def pad_batch(samples):
    """对应 trainer 的 padding + labels 对齐（-100 填充）。"""
    maxlen = max(len(ids) for ids, _ in samples)
    X, Y = [], []
    for ids, labels in samples:
        pad = maxlen - len(ids)
        X.append(F.pad(ids, (0, pad)))
        Y.append(F.pad(labels, (0, pad), value=-100))
    return torch.stack(X).to(DEVICE), torch.stack(Y).to(DEVICE)


# ─── 5. SFT 训练循环（LoRA 模式：只训 BA + lm_head？——本课严格冻结 lm_head 演示纯 LoRA）───
def sft_train(model, data, steps=400, bs=8, lr=3e-3):
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    for _ in range(steps):
        batch = [data[i] for i in torch.randint(0, len(data), (bs,))]
        X, Y = pad_batch(batch)
        logits = model(X[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               Y[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, n_train


@torch.no_grad()
def chat(model, instruction, max_new=12):
    """推理演示：prompt 前缀 → 逐 token 生成到 <|im_end|>。"""
    model.eval()
    ids = torch.tensor([encode(f"<|im_start|> user: {instruction} <|im_end|>")],
                       device=DEVICE)
    for _ in range(max_new):
        logits = model(ids[:, -model.ctx:])[:, -1, :]
        nxt = logits.argmax(-1).item()          # 贪心
        if nxt == IM_END:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return decode(ids[0].tolist())


def merge_lora(model):
    """对应 llamafactory-cli export：把 BA 合并回 W（推理零开销）。"""
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.linear.weight += (module.alpha / module.r) * module.B @ module.A
            merged += 1
    return merged


def main():
    print("═══ 手写 LoRA SFT 微型管线 ═══")
    print(f"  device={DEVICE}\n")

    # 基座预热（"预训练过的"玩具基座）
    model = ToyGPT(len(VOCAB)).to(DEVICE)
    g = torch.Generator().manual_seed(7)
    corpus = torch.randint(0, len(VOCAB), (256, 24), generator=g).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(200):
        ix = corpus[torch.randint(0, 256, (16,))]
        _, loss = model(ix[:, :-1], ix[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"[0] 基座预热 loss: {loss.item():.3f}")

    # LoRA 注入（严格协议：先全冻结）
    for p_ in model.parameters():
        p_.requires_grad_(False)
    n_injected = apply_lora(model, r=4, alpha=8.0)   # 4 层 MLP Linear
    model.to(DEVICE)  # ⚠️ 全课程第二次踩到：注入新建的 A/B 在 CPU，注入后必须再 .to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[1] LoRA 注入 {n_injected} 层（r=4, alpha=8）→ 可训练 {n_train:,}/{n_total:,} "
          f"参数（{n_train / n_total:.1%}）   ← yaml: lora_target/rank/alpha")

    # SFT
    data = make_sft_data()
    losses, _ = sft_train(model, data)
    print(f"[2] SFT 400 步: loss {losses[0]:.3f} → {sum(losses[-50:]) / 50:.3f}"
          f"   ← yaml: dataset/learning_rate/num_train_epochs")

    # 推理验证
    import random as _r
    rng = _r.Random(9)
    word_ids = [STOI[w] for w in WORDS]
    print(f"[3] 推理（合并前，任务=回声指令：回应应复述指令的第一个词）:")
    for _ in range(3):
        a, b = rng.choice(word_ids), rng.choice(word_ids)
        q = f"{WORDS[a - len(SPECIALS)]} {WORDS[b - len(SPECIALS)]}"
        print(f"    {q!r} → {chat(model, q)!r}")

    # 合并
    n_merged = merge_lora(model)
    a, b = word_ids[0], word_ids[5]
    q = f"{WORDS[a - len(SPECIALS)]} {WORDS[b - len(SPECIALS)]}"
    print(f"[4] 合并 {n_merged} 个 LoRA 层回 W（llamafactory-cli export 的作用）")
    print(f"    合并后 {q!r} → {chat(model, q)!r}  ← 行为不变，但零额外开销")

    print("""
═══ 与 LLaMA-Factory yaml 的字段对照 ═══
  build_sample() 的 prompt/masking      ← template: <template名> + train_on_prompt: false
  apply_lora(target/r/alpha)            ← lora_target / lora_rank / lora_alpha
  make_sft_data() 的 (instruction,response) ← dataset_info.json + dataset 字段
  sft_train() 的循环/优化器/lr           ← learning_rate / num_train_epochs / per_device_batch
  pad_batch()                           ← cutoff_len + padding（或 packing）
  merge_lora()                          ← llamafactory-cli export
  → 下一步：02 章用真实 yaml + 真实 7B 模型走同一流程（QLoRA 6GB 就能跑）。""")


if __name__ == '__main__':
    main()
