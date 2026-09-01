#!/usr/bin/env python3
"""
Part 7 - 脚本 11: RoPE 长上下文外推实验 —— naive vs PI vs NTK vs YaRN
目标：把"训练 128、推理 256"的外推问题做成可测量的实验。同一模型、同一数据，
      只改位置编码方案，看困惑度（ppl）如何变化：
        ① naive     ：RoPE 角度表直接外推到 256（训练时只见过位置 0..127）
        ② PI        ：位置插值（Position Interpolation），m → m/s（s=2，把 256 压回训练范围）
        ③ NTK-aware ：改 base：θ' = θ·s^(dim/(dim-2))，高频维度几乎不动
        ④ YaRN      ：NTK-by-parts —— 逐维 ramp 混合"外推/插值" + 注意力温度
                      √(1/t) = 0.1·ln(s)+1（论文 Eq.15）
对应教程：tutorial/05_reproduce_minimind.md「进阶实验」（minimind 的
inference_rope_scaling 选项就是这些方案）。
参考：YaRN 论文 arXiv 2309.00071（Eq.14/15）；实现对照 HF transformers
      modeling_rope_utils.py::_compute_yarn_parameters（4.57.6 源码核对）。

运行（~1 分钟；CPU 也能跑）：
    python 11_rope_scaling.py
预期（与文献一致的方向）：训练长度内各方案等价或接近（sanity check）；
外推区域 naive 明显变差，PI/NTK 零样本外推显著更稳，YaRN 在 NTK 之上再加
逐维 ramp + 温度微调，是四者中工业界最常用的"零样本+少量微调"方案。
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


def rope_angles(head_dim, max_pos, base=BASE, inv_freq=None):
    """角度表 (max_pos, head_dim/2)：第 p 行 = 位置 p 各维对的旋转角。
    inv_freq 传入时直接用（YaRN 的逐维混合频率），否则按 base 生成。"""
    if inv_freq is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2)[:head_dim // 2].float() / head_dim))
    return torch.outer(torch.arange(max_pos).float(), inv_freq)


def yarn_params(head_dim, base, s, train_ctx, alpha=1.0, beta=32.0):
    """YaRN（arXiv 2309.00071）三部件：修正维边界 + 逐维 ramp 混合频率 + 注意力温度。
    实现对照 HF transformers modeling_rope_utils.py::_compute_yarn_parameters（4.57.6 核对）。

    ⚠️ 命名陷阱（面试加分点）：HF 的 beta_fast=32 / beta_slow=1 与论文的希腊字母**正好相反**——
      论文里 β_slow=32 圈对应 ramp 的【下界】low（高频维、波长 < 上下文、外推不动它），
      论文里 β_fast=1 圈对应【上界】high（低频维、波长 > 上下文、全插值）。
      即 HF 的 beta_fast 反而标记"慢"的边界。本函数参数沿用论文记号：beta=32（下界）、alpha=1（上界）。

    返回 (inv_freq, attn_factor)：
      inv_freq    : (head_dim/2,) 各维对混合后的频率 1/f → 介于 1/f（外推）与 1/(s·f)（插值）之间
      attn_factor : 注意力温度 √(1/t) = 0.1·ln(s)+1（论文 Eq.15），softmax 前乘在 q 上
    """
    def find_correction_dim(num_rot, dim, base, max_pos):
        # 反解"在 max_pos 内恰好转 num_rot 圈"的维度编号 i：θ^(2i/dim) = max_pos/(num_rot·2π)
        return dim * math.log(max_pos / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    half = head_dim // 2
    # low：边界内的维在训练长度内转 ≥32 圈（高频/短波长，局部顺序信息最熟 → 保留外推）
    low = max(math.floor(find_correction_dim(beta, head_dim, base, train_ctx)), 0)
    # high：边界外的维转 ≤1 圈（低频/长波长，全局粗定位 → 全插值压回训练范围）
    high = min(math.ceil(find_correction_dim(alpha, head_dim, base, train_ctx)), half - 1)

    # 逐维 ramp：0（高频维，全外推）→ 1（低频维，全插值）。
    # ⚠️ 方向别写反：ramp 要随维度编号 i 递增（i 大 = 频率低 = 波长长 = 该插值）。
    i = torch.arange(half, dtype=torch.float32)
    ramp = ((i - low) / (high - low)).clamp(0, 1)      # (head_dim/2,)
    f = base ** (2 * i / head_dim)                     # 分母 θ^(2i/dim)：i 越大转得越慢（波长越长）
    inv_freq = (1.0 / f) * (1 - ramp) + (1.0 / (s * f)) * ramp
    # 注意力温度：论文 Eq.14 把 softmax(q·k/(t·√D)) 的温度 t 折成 Eq.15 的 √(1/t)=0.1·ln(s)+1。
    # 官方实现把 √(1/t) 同时乘在 q、k 上（HF 干脆折进 cos/sin，logit 效果 ×1/t）；
    # 本教程选最小改动的版本：只乘 q（logit 效果 ×√(1/t)），s=2 时 1.069，两者差异可忽略。
    attn_factor = 0.1 * math.log(s) + 1.0
    return inv_freq, attn_factor


def apply_rope(x, angles, pos_scale=1.0, attn_scale=1.0):
    """x:(B,T,H,D) 按位置旋转。pos_scale≠1 即 Position Interpolation（查表位置 m/s）。
    attn_scale：YaRN 注意力温度 √(1/t)（只传给 q——旋转与标量乘可交换，效果即 q 整体放大）。"""
    B, T, H, D = x.shape
    pos = (torch.arange(T, device=x.device).float() * pos_scale).long()
    ang = angles.to(x.device)[pos].view(1, T, 1, D // 2)
    xc = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
    out = torch.view_as_real(xc * torch.polar(torch.ones_like(ang), ang))
    return out.reshape(B, T, H, D).type_as(x) * attn_scale


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
        # naive/PI 共用 BASE 表（建到 EVAL_CTX）；NTK 用新 base 重建；YaRN 用逐维混合频率重建
        self.angles_base = rope_angles(self.hd, EVAL_CTX)
        self.angles_ntk = rope_angles(self.hd, EVAL_CTX,
                                      base=BASE * (EXTEND_S ** (self.hd / (self.hd - 2))))
        yarn_inv_freq, self.yarn_attn = yarn_params(self.hd, BASE, EXTEND_S, TRAIN_CTX)
        self.angles_yarn = rope_angles(self.hd, EVAL_CTX, inv_freq=yarn_inv_freq)

    def forward(self, x, scheme='naive'):
        B, T, C = x.shape
        if scheme == 'ntk':
            angles, pos_scale, attn = self.angles_ntk, 1.0, 1.0
        elif scheme == 'yarn':
            angles, pos_scale, attn = self.angles_yarn, 1.0, self.yarn_attn
        else:
            angles, pos_scale, attn = (self.angles_base,
                                       1.0 / EXTEND_S if scheme == 'pi' else 1.0, 1.0)
        h = self.ln1(x)
        q = self.wq(h).view(B, T, self.h, self.hd)
        k = self.wk(h).view(B, T, self.h, self.hd)
        v = self.wv(h).view(B, T, self.h, self.hd)
        q = apply_rope(q, angles, pos_scale, attn_scale=attn)   # YaRN 温度只乘 q
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
    """固定窗口评测：用 torch.Generator(seed=seq) 抽窗口，四种方案看到**同一批**文本，
    ppl 差异就只来自方案本身（否则各方案各抽各的窗口，方案间比较会混入采样噪声）。"""
    g = torch.Generator().manual_seed(seq)
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,), generator=g)
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

    # 训练：只在 TRAIN_CTX 长度上训（模拟"模型只见过位置 0..127"）。
    # 1000 步（比旧版 300 步更充分）：欠训模型对慢频率维的利用很噪，
    # 会掩盖 naive/NTK/YaRN 的真实差距（实测 300 步时三者差距落在评测噪声内）
    model = RoPEGPT(len(chars)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    steps = 1000
    for _ in range(steps):
        ix = torch.randint(0, len(ids_all) - TRAIN_CTX - 1, (8,))
        x = torch.stack([torch.tensor(ids_all[i:i + TRAIN_CTX]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids_all[i + 1:i + TRAIN_CTX + 1]) for i in ix]).to(DEVICE)
        _, loss = model(x, y, scheme='naive')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"  训练完成（ctx={TRAIN_CTX}, {steps} 步）\n")

    # ── 单元自检：s=1 时 YaRN 必须逐元素退化为 naive（频率一致、温度=1）──
    hd = model.blocks[0].hd                                   # head_dim = n_embed / n_head
    inv_naive = BASE ** (-2 * torch.arange(hd // 2).float() / hd)
    inv_yarn_s1, attn_s1 = yarn_params(hd, BASE, 1.0, TRAIN_CTX)
    ok = torch.allclose(inv_yarn_s1, inv_naive, atol=1e-6) and attn_s1 == 1.0
    print(f"  自检 1  s=1: inv_freq 与 naive 逐元素 allclose={torch.allclose(inv_yarn_s1, inv_naive, atol=1e-6)}, "
          f"温度={attn_s1:.4f}  {'✅ 通过（YaRN 在训练长度内即 naive）' if ok else '❌ 失败，检查 ramp 方向'}")

    # ── 自检 2：打印 ramp 边界与温度，确认"哪些维被插值"符合预期 ──
    def _fcd(num_rot):
        return hd * math.log(TRAIN_CTX / (num_rot * 2 * math.pi)) / (2 * math.log(BASE))
    low = max(math.floor(_fcd(32)), 0)
    high = min(math.ceil(_fcd(1)), hd // 2 - 1)
    inv_yarn, attn_yarn = yarn_params(hd, BASE, EXTEND_S, TRAIN_CTX)
    n_interp = int((inv_yarn < inv_naive * (1 - 1e-9)).sum())
    print(f"  自检 2  ramp 区间 [low={low}, high={high}]：维 {low}..{high - 1} 线性过渡，"
          f"{n_interp}/{hd // 2} 个维对被（部分）插值；温度 √(1/t)=0.1·ln({EXTEND_S})+1={attn_yarn:.4f}\n")

    print(f"{'方案':<28}{'ppl @ctx=128（训练内）':>24}{'ppl @ctx=256（外推）':>22}")
    ppl = {}
    for name, kind in (('① naive（直接外推）', 'naive'),
                       ('② PI（位置 ÷2）', 'pi'),
                       ('③ NTK（base×s^(d/(d-2))）', 'ntk'),
                       (f'④ YaRN（ramp+温度{attn_yarn:.3f}）', 'yarn')):
        ppl_in = evaluate(model, ids_all, TRAIN_CTX, kind)
        ppl_out = evaluate(model, ids_all, EVAL_CTX, kind)
        ppl[kind] = (ppl_in, ppl_out)
        print(f"{name:<28}{ppl_in:>24.2f}{ppl_out:>22.2f}")

    # ── 验收：预期排序（外推区）ppl@256: yarn ≤ ntk < naive；训练内 yarn ≈ naive ──
    c1 = ppl['yarn'][1] <= ppl['ntk'][1] < ppl['naive'][1]
    c2 = ppl['yarn'][0] - ppl['naive'][0] < 0.25 * (ppl['pi'][0] - ppl['naive'][0])
    print(f"\n  验收 1  ppl@256: yarn({ppl['yarn'][1]:.2f}) ≤ ntk({ppl['ntk'][1]:.2f}) < naive({ppl['naive'][1]:.2f})"
          f"  {'✅' if c1 else '❌ 检查 ramp 方向 / 温度是否乘在 q 上'}")
    print(f"  验收 2  ppl@128: yarn({ppl['yarn'][0]:.2f}) ≈ naive({ppl['naive'][0]:.2f})"
          f"（偏离远小于 PI 的 {ppl['pi'][0]:.2f}，温度 {attn_yarn:.3f} 影响极小）"
          f"  {'✅' if c2 else '❌'}")

    print("""
═══ 实测解读（以你跑出的数字为准，下面是典型形态）═══
  - 训练长度内 ctx=128：naive / NTK / YaRN 三者接近（sanity check：换方案没破坏模型）；
    【PI 偏高是"正确的结果"】—— PI 把训练时见过的位置 0..127 也压缩成了 0..63，
    未经微调就等于换了一套位置分布。这正是"PI 必须配微调"的原因，实测直接展示了
  - 外推 ctx=256：NTK 最稳（保高频、只改低频的 base）；naive 劣化（未见过的旋转角
    → 分布外的 q·k 组合）；PI 零样本最差，但文献中配 ~1000 步微调即可完全恢复
    （Llama-2 4k→32k），且恢复后上限比 NTK 高
  - YaRN = "NTK-by-parts + 温度"的调和版：低频维全插值、高频维原样外推、
    softmax 前给 q 乘 √(1/t)=0.1·ln(s)+1 微微锐化注意力。论文里 Llama-2 7B 用它
    **400 步**微调到 128k，比 PI 省 ~10× token —— 工业界长上下文扩容的默认选择
  💡 面试问法："RoPE 为什么不能直接外推？PI 和 NTK 分别动了什么？YaRN 的温度因子
     √(1/t)=0.1·ln(s)+1 是干什么的？" —— 用上面四行实测数字回答，比背论文摘要有力得多。""")


if __name__ == "__main__":
    main()
