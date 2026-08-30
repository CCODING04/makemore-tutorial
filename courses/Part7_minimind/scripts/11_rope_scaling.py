#!/usr/bin/env python3
"""
Part 7 - 脚本 11: RoPE 长上下文外推实验 —— naive vs PI vs NTK
目标：把"训练 128、推理 256"的外推问题做成可测量的实验。同一模型、同一数据，
      只改位置编码方案，看困惑度（ppl）如何变化：
        ① naive     ：RoPE 角度表直接外推到 256（训练时只见过位置 0..127）
        ② PI        ：位置插值（Position Interpolation），m → m/s（s=2，把 256 压回训练范围）
        ③ NTK-aware ：改 base：θ' = θ·s^(dim/(dim-2))，高频维度几乎不动
对应教程：tutorial/05_reproduce_minimind.md「进阶实验」（minimind 的
inference_rope_scaling 选项就是这些方案）。

运行（~1 分钟；CPU 也能跑）：
    python 11_rope_scaling.py
预期（与文献一致的方向）：训练长度内三种方案等价（sanity check）；
外推区域 naive 明显变差，PI/NTK 零样本外推显著更稳。
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
TRAIN_CTX, EVAL_CTX = 128, 256          # 训练 128、评测 256 —— 2 倍外推
EXTEND_S = EVAL_CTX // TRAIN_CTX        # 外推倍数 s = 2
BASE = 10000.0


def rope_angles(head_dim, max_pos, base=BASE):
    """角度表 (max_pos, head_dim/2)：第 p 行 = 位置 p 各维对的旋转角。"""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2)[:head_dim // 2].float() / head_dim))
    return torch.outer(torch.arange(max_pos).float(), freqs)


def apply_rope(x, angles, pos_scale=1.0):
    """x:(B,T,H,D) 按位置旋转。pos_scale≠1 即 Position Interpolation（查表位置 m/s）。"""
    B, T, H, D = x.shape
    pos = (torch.arange(T, device=x.device).float() * pos_scale).long()
    ang = angles.to(x.device)[pos].view(1, T, 1, D // 2)
    xc = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
    out = torch.view_as_real(xc * torch.polar(torch.ones_like(ang), ang))
    return out.reshape(B, T, H, D).type_as(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.h = n_head
        self.hd = n_embed // n_head
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
        self.wq = nn.Linear(n_embed, n_embed, bias=False)
        self.wk = nn.Linear(n_embed, n_embed, bias=False)
        self.wv = nn.Linear(n_embed, n_embed, bias=False)
        self.wo = nn.Linear(n_embed, n_embed, bias=False)
        self.fc1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc2 = nn.Linear(4 * n_embed, n_embed)
        # naive/PI 共用 BASE 表（建到 EVAL_CTX）；NTK 用新 base 重建
        self.angles_base = rope_angles(self.hd, EVAL_CTX)
        self.angles_ntk = rope_angles(self.hd, EVAL_CTX,
                                      base=BASE * (EXTEND_S ** (self.hd / (self.hd - 2))))

    def forward(self, x, scheme='naive'):
        B, T, C = x.shape
        if scheme == 'ntk':
            angles, pos_scale = self.angles_ntk, 1.0
        else:
            angles, pos_scale = self.angles_base, (1.0 / EXTEND_S if scheme == 'pi' else 1.0)
        h = self.ln1(x)
        q = self.wq(h).view(B, T, self.h, self.hd)
        k = self.wk(h).view(B, T, self.h, self.hd)
        v = self.wv(h).view(B, T, self.h, self.hd)
        q = apply_rope(q, angles, pos_scale)
        k = apply_rope(k, angles, pos_scale)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        wei = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        wei = wei.masked_fill(mask, float('-inf'))
        a = (F.softmax(wei, -1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.wo(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class RoPEGPT(nn.Module):
    def __init__(self, vocab, n_embed=128, n_head=4, n_layer=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)

    def forward(self, idx, targets=None, scheme='naive'):
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, scheme)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1))


@torch.no_grad()
def evaluate(model, ids, seq, scheme, n_batches=10, bs=4):
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        x = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids[i + 1:i + seq + 1]) for i in ix]).to(DEVICE)
        total += model(x, y, scheme)[1].item() * bs
    return math.exp(total / (n_batches * bs))


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'data', 'input.txt')
    text = open(path, encoding='utf-8').read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids_all = [stoi[c] for c in text[:400000]]

    print("═══ RoPE 长上下文外推实验（训练 128 → 推理 256，s=2）═══")
    print(f"  device={DEVICE}")

    # 训练：只在 TRAIN_CTX 长度上训（模拟"模型只见过位置 0..127"）
    model = RoPEGPT(len(chars)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    steps = 300
    for _ in range(steps):
        ix = torch.randint(0, len(ids_all) - TRAIN_CTX - 1, (8,))
        x = torch.stack([torch.tensor(ids_all[i:i + TRAIN_CTX]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids_all[i + 1:i + TRAIN_CTX + 1]) for i in ix]).to(DEVICE)
        _, loss = model(x, y, scheme='naive')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"  训练完成（ctx={TRAIN_CTX}, {steps} 步）\n")

    print(f"{'方案':<28}{'ppl @ctx=128（训练内）':>24}{'ppl @ctx=256（外推）':>22}")
    for name, kind in (('① naive（直接外推）', 'naive'),
                       ('② PI（位置 ÷2）', 'pi'),
                       ('③ NTK（base×s^(d/(d-2))）', 'ntk')):
        ppl_in = evaluate(model, ids_all, TRAIN_CTX, kind)
        ppl_out = evaluate(model, ids_all, EVAL_CTX, kind)
        print(f"{name:<28}{ppl_in:>24.2f}{ppl_out:>22.2f}")

    print("""
═══ 实测解读（以你跑出的数字为准，下面是典型形态）═══
  - 训练长度内 ctx=128：naive 与 NTK 几乎一致（sanity check：换方案没破坏模型）；
    【PI 偏高是"正确的结果"】—— PI 把训练时见过的位置 0..127 也压缩成了 0..63，
    未经微调就等于换了一套位置分布。这正是"PI 必须配微调"的原因，实测直接展示了
  - 外推 ctx=256：NTK 最稳（保高频、只改低频的 base）；naive 劣化（未见过的旋转角
    → 分布外的 q·k 组合）；PI 零样本最差，但文献中配 ~1000 步微调即可完全恢复
    （Llama-2 4k→32k），且恢复后上限比 NTK 高
  - 三者的调和版就是 YaRN：逐维 ramp 混合 PI/NTK + 注意力温度（400 步到 128k）
  💡 面试问法："RoPE 为什么不能直接外推？PI 和 NTK 分别动了什么？哪个能零样本用？"
     —— 用上面三行实测数字回答，比背论文摘要有力得多。""")


if __name__ == "__main__":
    main()
