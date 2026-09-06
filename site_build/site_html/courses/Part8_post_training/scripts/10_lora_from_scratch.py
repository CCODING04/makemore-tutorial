#!/usr/bin/env python3
"""
Part 8 - 脚本 10: LoRA 从零实现 + 分类微调（对照 rasbt LLMs-from-scratch 附录 E / ch06）
目标：① 手写 LoRALinear（冻结 W + 低秩 BA 旁路），注入玩具 GPT；
      ② 跑"全参微调 vs LoRA 微调"对比：可训练参数量、显存估算、收敛速度；
      ③ 走一遍分类微调的标准范式（换分类头 + 冻结主干）——rasbt ch06 的核心流程。
对应教程：tutorial/08_lora_and_classification.md

运行（~30 秒，CPU/GPU 均可）：
    python 10_lora_from_scratch.py
预期：LoRA 可训练参数 ≈ 全参的 3-5%，收敛速度与全参相当（小任务上），
      这是"参数高效微调"的最小可验证证据。
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


# ─── 玩具模型：2 层小 GPT（结构同 Part 8 01 章）────────────
class Block(nn.Module):
    def __init__(self, n_embed, n_head, ctx):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.GELU(),
                                 nn.Linear(4 * n_embed, n_embed))
        self.register_buffer('mask', torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=self.mask[:T, :T])
        return x + self.mlp(self.ln2(x + a))


class ToyGPT(nn.Module):
    def __init__(self, vocab, n_embed=96, n_head=4, n_layer=2, ctx=32, n_classes=None):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, n_embed)
        self.pos = nn.Embedding(ctx, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, ctx) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)
        if n_classes is not None:                     # 分类头（rasbt ch06 范式：换头）
            self.cls_head = nn.Linear(n_embed, n_classes)

    def features(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln(x)                             # (B, T, C)

    def forward_lm(self, idx, targets=None):
        logits = self.head(self.features(idx))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))

    def forward_cls(self, idx):
        """分类：取最后一个位置的隐向量过分类头（rasbt ch06 的做法）。"""
        return self.cls_head(self.features(idx)[:, -1, :])


# ─── LoRA 从零： rasbt 附录 E 的最小实现 ────────────────────
class LoRALinear(nn.Module):
    """冻结原权重 W，旁路 BA（B: out×r 初始化 0，A: r×in 高斯初始化）。
    前向：y = W x + (α/r)·B(A x)。B 初始化 0 → 训练起点 = 原模型。
    A 用 1/√r 缩放的高斯（Kaiming 风格）——与原论文一致。"""

    def __init__(self, linear: nn.Linear, r=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)                   # 冻结 W
        out_f, in_f = linear.weight.shape
        self.r, self.alpha = r, alpha
        self.A = nn.Parameter(torch.randn(r, in_f) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(out_f, r))  # 零初始化：起点无损

    def forward(self, x):
        return self.linear(x) + (self.alpha / self.r) * (x @ self.A.T) @ self.B.T


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


# ─── 分类任务（玩具版"垃圾邮件识别"）：序列含 3 个及以上 '7' → 正类 ───
def make_cls_data(vocab=16, n=4000, seq=16):
    g = torch.Generator().manual_seed(42)
    X = torch.randint(1, vocab, (n, seq), generator=g)
    y = ((X == 7).sum(dim=1) >= 3).long()   # cross_entropy 需要 long 标签
    return (X[:3200].to(DEVICE), y[:3200].to(DEVICE),
            X[3200:].to(DEVICE), y[3200:].to(DEVICE))


def train_cls(model, data, steps=300, bs=32, lr=1e-3):
    """只训练 requires_grad 的参数（LoRA 模式下天然只有 BA 和分类头）。"""
    Xtr, ytr = data[0], data[1]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    for _ in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (bs,))
        logits = model.forward_cls(Xtr[ix])
        loss = F.cross_entropy(logits, ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def eval_acc(model, X, y, bs=256):
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], bs):
        pred = model.forward_cls(X[i:i + bs]).argmax(-1)
        correct += (pred == y[i:i + bs]).sum().item()
    model.train()
    return correct / X.shape[0]


def main():
    print("═══ LoRA 从零 + 分类微调对比 ═══")
    print(f"  device={DEVICE}\n")

    # ── 阶段 1：预训练一个"会写这段语言"的基座（让主干有可微调的表征）──
    vocab = 16
    base = ToyGPT(vocab).to(DEVICE)
    g = torch.Generator().manual_seed(7)
    lm_x = torch.randint(1, vocab, (256, 16), generator=g).to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
    for _ in range(200):
        ix = lm_x[torch.randint(0, 256, (16,))]
        _, loss = base.forward_lm(ix[:, :-1], ix[:, 1:])   # 标准的下一 token 预测
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"[阶段1] 基座语言建模预热: loss {loss.item():.3f}（结构就绪即可）\n")

    # ── 阶段 2：全参微调 vs LoRA 微调 ──
    data = make_cls_data()

    # 2a. 全参微调（换分类头，全部参数可训练）
    full = ToyGPT(vocab, n_classes=2).to(DEVICE)
    full.load_state_dict({k: v for k, v in base.state_dict().items()
                          if not k.startswith('cls_head')}, strict=False)
    n_full = count_trainable(full)
    losses_full = train_cls(full, data)
    acc_full = eval_acc(full, data[2], data[3])

    # 2b. LoRA 微调（MLP 层注入 LoRA，主干其余冻结；分类头可训练）
    lora_model = ToyGPT(vocab, n_classes=2).to(DEVICE)
    lora_model.load_state_dict({k: v for k, v in base.state_dict().items()
                                if not k.startswith('cls_head')}, strict=False)
    for p_ in lora_model.parameters():                # 严格 LoRA 协议：先冻结一切
        p_.requires_grad_(False)
    n_injected = 0
    for blk in lora_model.blocks:                     # 注入：MLP 的两个 Linear
        blk.mlp[0] = LoRALinear(blk.mlp[0], r=4, alpha=8)   # 新模块的 A/B 默认可训练
        blk.mlp[2] = LoRALinear(blk.mlp[2], r=4, alpha=8)
        n_injected += 2
    lora_model.cls_head.weight.requires_grad_(True)   # 分类头可训练（rasbt ch06 同款）
    lora_model.to(DEVICE)   # ⚠️ 注入新建的 A/B 默认在 CPU，必须再搬一次（真实踩过的坑）
    n_lora = count_trainable(lora_model)
    losses_lora = train_cls(lora_model, data)
    acc_lora = eval_acc(lora_model, data[2], data[3])

    print(f"[阶段2] 分类微调对比（任务: 序列含≥3个'7' → 正类）")
    print(f"  全参微调: 可训练 {n_full:>8,} 参数（100%）| 验证 acc = {acc_full:.3f} | "
          f"末 50 步平均 loss = {sum(losses_full[-50:]) / 50:.4f}")
    print(f"  LoRA 微调: 可训练 {n_lora:>8,} 参数（{n_lora / n_full:.1%}，注入 {n_injected} 层 r=4）"
          f"| 验证 acc = {acc_lora:.3f} | 末 50 步平均 loss = {sum(losses_lora[-50:]) / 50:.4f}")

    # ── 显存估算（16 bytes/参数 的账本，呼应 Part 10 03 章）──
    def mem_mb(n_train, n_total):
        # 可训练: fp32 参数+梯度+AdamW(2×4B)=12B/个；冻结: bf16 参数 2B/个
        return (n_train * 12 + (n_total - n_train) * 2) / 1e6
    print(f"\n[阶段3] 训练显存估算（本课账本口径: 可训练 12B/参数 + 冻结 2B/参数）")
    print(f"  全参: {mem_mb(n_full, n_full):8.1f} MB   LoRA: {mem_mb(n_lora, n_full):6.1f} MB"
          f"   （真实 7B 模型上差距是 GB 级 vs MB 级）")

    print("""
═══ 预期与解读 ═══
  - LoRA 可训练参数 ≈ 全参的百分之几，但小任务上 acc 与全参相当 —— "低秩足够"的最小证据
  - LoRA 的 B 初始化为 0 → 训练起点 = 原模型（不会一开始就破坏预训练表征）
  - acc 若两者都接近 1.0：任务太简单，把 make_cls_data 的阈值 3 调成 4 增加难度再看差距
  💡 面试问法："LoRA 为什么 B 初始化为 0？α/r 缩放是干什么的？LoRA 省的是参数还是显存？"
     （答：B=0 保证起点无损；α/r 让 r 变化时学习率尺度稳定；省的是优化器状态+梯度显存，
       权重本体还是全量存着——推理时 BA 可合并回 W 变成零开销。）""")


if __name__ == '__main__':
    main()
