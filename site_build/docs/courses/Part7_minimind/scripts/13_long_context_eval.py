#!/usr/bin/env python3
"""
Part 7 - 脚本 13: 迷你 RULER —— 四种 RoPE 方案的"needle 检索"长上下文评测
目标：脚本 11 用"困惑度"证明了 yarn/ntk 外推更稳，但 ppl 只测"读得顺不顺"，
      不测"记得住记不住"。本脚本合成 RULER 风格的 KV 检索任务（arXiv 2404.06654 的
      迷你版），直接量"在 2×/4× 外推长度上还能不能从上下文里精确取回一条信息"：

        上下文 = "a3,b7,c1,...,a3,b7,...(每对出现两次)... ?b"  → 模型须答出 b 的值
        （每个 key-value 对出现两次：第二次出现是训练信号，末尾 query 是考题）

      同一模型（训练长度 128），四套推理方案 × 三档上下文：
        ctx=128（训练内，s=1，四方案应等价 —— sanity check）
        ctx=256 / 512（外推区，s=2 / 4，naive 应崩、yarn/ntk 应守住）
      输出 needle 准确率表 + 曲线图 output_long_context.png。

对应教程：tutorial/05_reproduce_minimind.md「进阶实验」。
参考：RULER (2404.06654) · YaRN (2309.00071)（yarn_params 与脚本 11 一致，
      实现对照 HF transformers modeling_rope_utils.py 4.57.6）。

运行：GPU ~15 秒；CPU ~40-60 秒（检索能力需 ~6.5 万条样本才会"顿悟"式形成，
      见下方 train() 注释——步数不能再砍，砍了模型根本学不会检索）：
    python 13_long_context_eval.py
"""

import os
import sys
import math
import time
import random

import matplotlib
matplotlib.use('Agg')                # 无显示环境也能存图
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)              # 训练可复现（评测另用独立种子，见 needle_accuracy）
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_CTX, BASE = 128, 10000.0       # 训练长度 128；RoPE θ=1e4（与脚本 11 一致）
CTXS = (128, 256, 512)               # 评测三档：1×/2×/4×
SCHEMES = ('naive', 'pi', 'ntk', 'yarn')
N_VARS = 21                          # 字典固定 21 个 key：长度变长只拉远检索距离（单变量）

# ── 词表：52 个字母做 key、10 个数字做 value、',/ ?/.' 做结构符 ──
KEYS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHARS = sorted(KEYS + '0123456789' + ',?.')
STOI = {c: i for i, c in enumerate(CHARS)}


# ═══════════════ 第一部分：合成任务（迷你 RULER 的 KV 检索）═══════════════
def make_kv_task(n_vars, ctx_len, rng=None):
    """合成一条 KV 检索样本（RULER kv-retrieval 的字符级迷你版）。

    Args:
        n_vars: 字典里 key-value 对数（每对在上下文中出现两次）
        ctx_len: 上下文长度（query 前正好 ctx_len 个字符）
        rng: random.Random 实例（不传则临时新建，保证可复现）
    Returns:
        ids: 长度 ctx_len+1 的 token 列表 —— 前 ctx_len 个是输入（prompt），
             最后 1 个是标准答案（key 对应的数字）
    示例（n_vars=3, ctx_len=24）: "b7,c1,a3,a3,b7,c1,......?b" → 答案 "7"

    设计要点：
      - 每对出现两次：第二次出现 + 末尾 query 都要求"查字典"，训练信号 ×(n_vars+1)
      - key 全部唯一（rng.sample 不放回）→ query 无歧义，随机猜对概率 = 1/10
      - "K3," 三字符一对：数字紧跟 key（+1 偏移），2 层模型即可形成归纳头电路
        （第 1 层"看前一个 token"、第 2 层"内容匹配"——Olsson et al. 2022）
    """
    if rng is None:
        rng = random.Random(0)
    keys = rng.sample(KEYS, n_vars)                    # 不放回采样：key 唯一
    vals = [rng.choice('0123456789') for _ in range(n_vars)]
    tokens = [f'{k}{v},' for k, v in zip(keys, vals)] * 2
    rng.shuffle(tokens)                                # 两次出现的位置随机
    qi = rng.randrange(n_vars)                         # 随机抽一个 key 来考
    prompt = ''.join(tokens).ljust(ctx_len - 2, '.') + '?' + keys[qi]
    return [STOI[c] for c in prompt] + [STOI[vals[qi]]]


def batch_tasks(n, ctx_len, seed):
    """一次性生成 n 条评测样本：x (n, ctx_len) 输入，y 只在最后一位放答案（其余 -100）。"""
    rng = random.Random(seed)                          # 独立种子 → 四方案看到同一批题
    xs, ys = [], []
    for _ in range(n):
        ids = make_kv_task(N_VARS, ctx_len, rng)
        xs.append(torch.tensor(ids[:-1]))              # (ctx_len,)
        y = torch.full((ctx_len,), -100, dtype=torch.long)
        y[-1] = ids[-1]                                # 只考最后一位：key 后面的数字
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)            # (n, ctx_len) 各一


# ═══════════════ 第二部分：四方案的 RoPE 旋转表（与脚本 11 同一套数学）═══════════════
def yarn_params(head_dim, base, s, train_ctx, alpha=1.0, beta=32.0):
    """YaRN 逐维 ramp 频率 + 注意力温度 √(1/t)=0.1·ln(s)+1（同脚本 11，含 HF 命名陷阱注释）。"""
    def find_correction_dim(num_rot, dim, base, max_pos):
        return dim * math.log(max_pos / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    half = head_dim // 2
    low = max(math.floor(find_correction_dim(beta, head_dim, base, train_ctx)), 0)   # 高频边界
    high = min(math.ceil(find_correction_dim(alpha, head_dim, base, train_ctx)), half - 1)
    i = torch.arange(half, dtype=torch.float32)
    ramp = ((i - low) / (high - low)).clamp(0, 1)      # 0=高频维(外推) → 1=低频维(全插值)
    f = base ** (2 * i / head_dim)                     # 分母 θ^(2i/dim)：i 大转得慢
    inv_freq = (1.0 / f) * (1 - ramp) + (1.0 / (s * f)) * ramp
    attn_factor = 0.1 * math.log(s) + 1.0              # 温度（s=1 时恰为 1）
    return inv_freq, attn_factor


def scheme_rope(head_dim, scheme, s, max_pos):
    """按方案构建 (cos, sin, attn_factor)。cos/sin: (max_pos, head_dim/2)。

      naive: 原表直接外推                pi: 位置 m → m/s（角度表整体压缩）
      ntk:   base ← base·s^(d/(d-2))     yarn: 逐维 ramp 混合 + 温度乘 q
    """
    def freqs(base):
        return 1.0 / (base ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))

    if scheme == 'naive':
        inv, pos_scale, attn = freqs(BASE), 1.0, 1.0
    elif scheme == 'pi':
        inv, pos_scale, attn = freqs(BASE), 1.0 / s, 1.0
    elif scheme == 'ntk':
        inv, pos_scale, attn = freqs(BASE * s ** (head_dim / (head_dim - 2))), 1.0, 1.0
    else:  # 'yarn'
        inv, attn = yarn_params(head_dim, BASE, s, TRAIN_CTX)
        pos_scale = 1.0
    pos = torch.arange(max_pos).float() * pos_scale    # (max_pos,)
    ang = torch.outer(pos, inv)                        # (max_pos, head_dim/2)
    return torch.cos(ang), torch.sin(ang), attn


class Block(nn.Module):
    """pre-norm Transformer 块（结构与脚本 11 的 Block 同构；RoPE 用 cos/sin 表实现，
    等价于脚本 11 的复数版，但省去逐 batch 构造复数旋转因子，CPU 快 ~20%）。"""

    def __init__(self, n_embed, n_head, max_pos):
        super().__init__()
        self.h, self.hd = n_head, n_embed // n_head
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
        self.wq = nn.Linear(n_embed, n_embed, bias=False)
        self.wk = nn.Linear(n_embed, n_embed, bias=False)
        self.wv = nn.Linear(n_embed, n_embed, bias=False)
        self.wo = nn.Linear(n_embed, n_embed, bias=False)
        self.fc1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc2 = nn.Linear(4 * n_embed, n_embed)
        self._rope_cache = {}                          # (scheme, s) → (cos, sin, attn)

    def rope(self, scheme, s, T, device):
        key = (scheme, s)
        if key not in self._rope_cache:                # 建到 max_pos，任意 T 直接切片
            cos, sin, attn = scheme_rope(self.hd, scheme, s, 512)
            self._rope_cache[key] = (cos.to(device), sin.to(device), attn)
        return self._rope_cache[key]

    def forward(self, x, scheme='naive', s=1.0):
        B, T, C = x.shape                              # x: (B, T, n_embed)
        h = self.ln1(x)
        q = self.wq(h).view(B, T, self.h, self.hd)     # (B,T,H,hd)
        k = self.wk(h).view(B, T, self.h, self.hd)
        v = self.wv(h).view(B, T, self.h, self.hd)
        cos, sin, attn = self.rope(scheme, s, T, x.device)
        cos, sin = cos[:T].view(1, T, 1, -1), sin[:T].view(1, T, 1, -1)   # (1,T,1,hd/2)

        def rot(t):                                     # (B,T,H,hd) 实数版旋转（等效复数×e^{iθ}）
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

        q = rot(q) * attn                               # YaRN 温度只乘 q（与脚本 11 一致）
        k = rot(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                   # (B,H,T,hd)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)        # 含 1/√hd 与因果 mask
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + self.wo(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class RetrievalGPT(nn.Module):
    def __init__(self, vocab, n_embed=96, n_head=4, n_layer=2, max_pos=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, max_pos) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)

    def forward(self, idx, targets=None, scheme='naive', s=1.0):
        x = self.tok(idx)                              # (B, T, n_embed)
        for blk in self.blocks:
            x = blk(x, scheme, s)
        logits = self.head(self.ln(x))                 # (B, T, vocab)
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1), ignore_index=-100)


# ═══════════════ 第三部分：训练与评测 ═══════════════
def train(model, steps=2100, bs=32, lr=1e-3):
    """在 ctx=128、naive 方案下训练检索（模拟"模型只见过 0..127 的位置"")。

    ⚠️ 步数不能再省：检索电路（归纳头）不是慢慢变好，而是在 ~1900-2000 步
    "顿悟"式出现（loss 平台期后 accuracy 0.3→1.0 跳变）。我们实测过：
    砍 batch/减步数/只对答案位算 loss/课程学习，都推迟甚至阻止这一跳变——
    这是 grokking 式现象，需要 ~6.5 万条样本才触发。2100 步 = 过渡点 + 10% 余量。
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(steps):
        rng = random.Random(10_000 + step)             # 每步新题（旧题不复用，防死记）
        xs, ys = [], []
        for _ in range(bs):
            ids = make_kv_task(N_VARS, TRAIN_CTX, rng)
            xs.append(torch.tensor(ids[:-1]))
            ys.append(torch.tensor(ids[1:]))           # 全序列 CE：结构位 + 检索位一起学
        x, y = torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)
        _, loss = model(x, targets=y, scheme='naive', s=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"  step {step:>4}: loss = {loss.item():.3f}")
    print(f"  训练完成：{steps} 步 / {time.time() - t0:.0f}s（device={DEVICE}）")


@torch.no_grad()
def needle_accuracy(model, scheme, ctx_len, n_samples=64):
    """在 ctx_len 长度上用 scheme 方案测 needle 检索准确率。

    固定 seed=ctx_len 生成考题 → 同一 ctx 下四种方案做的是同一张卷子，
    分差只来自位置编码方案本身（与脚本 11 的"固定窗口评测"同一原则）。
    """
    s = max(1, ctx_len // TRAIN_CTX)                   # 128→1, 256→2, 512→4
    x, y = batch_tasks(n_samples, ctx_len, seed=ctx_len)
    logits = model(x.to(DEVICE), scheme=scheme, s=s)   # (n, ctx, vocab)
    pred = logits[:, -1].argmax(-1).cpu()              # 最后一位 = query 后的答案
    return (pred == y[:, -1]).float().mean().item()


def main():
    print("═══ 迷你 RULER：KV 检索 × 四种 RoPE 方案（训练 ctx=128）═══")
    print(f"  device={DEVICE} | 字典 {N_VARS} 对（每对出现两次）| 随机基线 = 1/10 = 0.100")

    # ── 自检：解码一条样本给读者看 + 验证任务无歧义 ──
    ids = make_kv_task(N_VARS, TRAIN_CTX, random.Random(0))
    s = ''.join(CHARS[i] for i in ids)
    qk, ans = s[-2], CHARS[ids[-1]]                    # prompt 末字符 = 被考的 key
    occ = [i for i in range(len(s) - 1) if s[i] == qk]
    vals = {s[i + 1] for i in occ if s[i + 1].isdigit()}
    print(f"  样例 prompt 头部: {s[:24]!r} … 尾部: {s[-12:]!r}")
    print(f"  query key={qk!r} 在上下文出现 {len(occ)} 次，对应值 {vals}，答案={ans!r}"
          f"  {'✅ 无歧义' if vals == {ans} else '❌ 任务生成有 bug'}")

    # ── 训练 ──
    print(f"\n── 训练（ctx={TRAIN_CTX}, naive, 检索电路需 ~2000 步'顿悟'）──")
    model = RetrievalGPT(len(CHARS)).to(DEVICE)
    train(model)

    # ── 评测：4 方案 × 3 档上下文 ──
    print(f"\n── needle 准确率（64 题/格；s = ctx/{TRAIN_CTX}）──")
    acc = {}
    print(f"{'ctx (s)':<10}{'naive':>8}{'pi':>8}{'ntk':>8}{'yarn':>8}")
    for ctx in CTXS:
        acc[ctx] = {k: needle_accuracy(model, k, ctx) for k in SCHEMES}
        print(f"{ctx} (s={ctx // TRAIN_CTX})".ljust(10) + ''.join(f"{acc[ctx][k]:>8.3f}" for k in SCHEMES))

    same = max(acc[128].values()) - min(acc[128].values())
    print(f"\n  ✅ ctx=128（s=1）四方案最大差距 {same:.3f}（应≈0：换方案不破坏模型）" if same < 0.05
          else f"  ⚠️ ctx=128 四方案差距 {same:.3f} 偏大（s=1 时四者数学上应完全一致）")

    # ── 画图 ──
    plt.figure(figsize=(7, 4.5))
    styles = {'naive': 'o-', 'pi': 's--', 'ntk': '^-', 'yarn': '*-'}
    for k in SCHEMES:
        plt.plot(CTXS, [acc[c][k] for c in CTXS], styles[k], label=k, markersize=9)
    plt.axhline(0.1, color='gray', linestyle=':', label='random guess (0.1)')
    plt.axvline(TRAIN_CTX, color='red', linestyle=':', alpha=0.6, label='train ctx = 128')
    plt.xscale('log', base=2)
    plt.xticks(CTXS, [str(c) for c in CTXS])
    plt.xlabel('context length (tokens)')
    plt.ylabel('needle accuracy')
    plt.title('Mini-RULER KV retrieval: 4 RoPE schemes beyond train length 128')
    plt.ylim(-0.03, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc='center left', fontsize=9)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_long_context.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 图已保存: {out}")

    print("""
═══ 实测解读（以你跑出的数字为准）═══
  - ctx=128（训练内）：四方案等价 —— s=1 时 PI/NTK/YaRN 的数学定义都退化为 naive
  - 外推区：naive 崩（训练只见过 ≤126 的相对距离，长距离 q·k 全是分布外）；
    pi 零样本不稳（把所有距离压一半 = 局部分辨率受损）；ntk/yarn 保住大部分准确率
  - 与脚本 11 互补：ppl 度量"读得顺"，needle 准确率度量"取得回"——
    长上下文的实用能力是后者，这也是 RULER 论文的核心主张（光刷上下文长度不够，
    得看有效上下文）
  💡 面试问法：'"号称 128K 的模型，长下文能力怎么测？"——先讲 NIAH 不够
    （RULER：17 个声称 ≥32K 的模型 NIAH 全近满分，但只有一半在 32K 上撑得住），
    再讲 needle/多跳/聚合这几类合成任务 + 真实长文榜单怎么配着用。""")


if __name__ == '__main__':
    main()
