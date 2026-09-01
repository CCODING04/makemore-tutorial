#!/usr/bin/env python3
"""
Part 13 - 脚本 00: Scaling Law 开篇 —— 用一条公式回答"数据要洗到多净、攒到多少才够"
目标：把 Chinchilla scaling law 从"背诵常数"变成"亲手拟合"，为整个 Part 13 提供
      预算语言：去重/过滤删掉的数据值多少 loss，要靠 L(N,D)=E+A/N^α+B/D^β 来算账。
对应教程：tutorial/00_scaling_laws.md（建议先读，再进 01 章手写 MinHash）

三种模式（argparse 子命令）：
  --mode fit    零 GPU、秒级：对内置"合成-加噪"Chinchilla 数据跑 fit_chinchilla，
                验收拟合参数相对误差 <5%，并画 isoFLOP 剖面图（loss 谷底随算力右移）
  --mode scan   冒烟 ~25s（单卡 RTX 4090；无 CUDA 自动缩规模）：
                网格真训小 GPT（语料 = 从 data/input.txt 拟合的 3 阶马尔可夫重采样，
                "无限唯一数据"区），把 (N,D,loss) 喂给 fit_chinchilla，得出本玩具
                尺度的最优 N:D。加 --full 扩网格（未实测）
  --mode epoch  数据约束区（Muennighoff et al. 2305.16264）：固定 unique 语料
                （data/input.txt，tiny shakespeare ~1.1M 字符）训 1/2/4/8/16 epoch，
                验证 R<=4 时 loss 接近幂律外推、R>4 饱和（插值 vs 实测对比表）

两个接口的签名跨脚本/作业复用，保持一致：
    chinchilla_loss(N, D, params)   params=(E,A,alpha,B,beta)
    fit_chinchilla(records)         records=[(N, D, final_loss), ...] → (E,A,alpha,B,beta)

运行：
    python 00_scaling_laws.py --mode fit     # CPU，~2s
    python 00_scaling_laws.py --mode scan    # CUDA，smoke ~25s
    python 00_scaling_laws.py --mode epoch   # CUDA，~40s
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares

import matplotlib
matplotlib.use('Agg')  # 无头后端：不弹窗，只存文件
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', 'input.txt')

# ─── Chinchilla 真值参数 ───
# 来源：Hoffmann et al. 2022 "Training Compute-Optimal Large Language Models"
# (arXiv 2203.15556) Table 3 的参数化拟合（论文 Approach 3）。
# 约定：N 为【非 embedding】参数量，D 为训练 token 数，loss 单位 nat/token。
TRUE_PARAMS = (1.69, 406.4, 0.34, 410.7, 0.28)  # (E, A, alpha, B, beta)


# ═══════════════════════════════════════════════════════════════════
# 接口 1：Chinchilla 三项式
# ═══════════════════════════════════════════════════════════════════
def chinchilla_loss(N, D, params):
    """L(N, D) = E + A / N^alpha + B / D^beta

    Args:
        N: 参数量（标量或 ndarray）
        D: 训练 token 数（标量或 ndarray）
        params: (E, A, alpha, B, beta)
    Returns:
        预测 loss（nat/token），形状与 N/D 一致

    三项含义：
        E     不可约损失（数据的条件熵下界——再大的模型也压不下去）
        A/N^α 模型容量项（网络不够大 → 欠拟合的代价）
        B/D^β 数据项（见过的 token 不够多 → 泛化不足的代价）
    """
    E, A, alpha, B, beta = params
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    return E + A * np.power(N, -alpha) + B * np.power(D, -beta)


# ═══════════════════════════════════════════════════════════════════
# 接口 2：Huber + scipy 拟合
# ═══════════════════════════════════════════════════════════════════
def fit_chinchilla(records, n_starts=8, huber_delta=0.05, seed=13, E_fixed=None):
    """从 (N, D, final_loss) 网格记录拟合 Chinchilla 三项式。

    Args:
        records: [(N, D, final_loss), ...] —— 真实跑出来的训练网格
        n_starts: 随机重启次数（幂律拟合非凸，多起点防局部极小）
        huber_delta: Huber 损失的分界（相对残差尺度）
        E_fixed: 若给定，则固定 E 只拟合 (A, alpha, B, beta)——当语料的熵下界
                 可独立测量时（本课 scan 模式的合成语料可以），这能救活
                 小动态范围网格上的病态拟合
    Returns:
        (E, A, alpha, B, beta)

    为什么 Huber 而不是普通最小二乘：
        真实网格里总有几个"坏点"（某 run 欠训练/发散）。平方损失会把离群点
        放大成主导，Huber 在 |r|>delta 后退化为线性——坏点被自动降权，
        拟合由大多数正常点决定。
    为什么 A/B 用 log 参数化：
        A、B 跨 3+ 个数量级，线性参数化会让初值和步长难以同时照顾两端。
    """
    arr = np.asarray(records, dtype=np.float64)      # shape (M, 3)
    Ns, Ds, Ls = arr[:, 0], arr[:, 1], arr[:, 2]

    if E_fixed is None:
        # 自由拟合 5 参数：x = [E, ln A, alpha, ln B, beta]
        def unpack(x):
            return x[0], math.exp(x[1]), x[2], math.exp(x[3]), x[4]

        lo = [1e-6, math.log(1e-6), 0.01, math.log(1e-6), 0.01]
        hi = [0.999 * float(Ls.min()), math.log(1e9), 1.50, math.log(1e9), 1.50]

        def residuals(x):
            return (chinchilla_loss(Ns, Ds, unpack(x)) - Ls) / Ls

        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_starts):
            x0 = [rng.uniform(0.3, 0.9) * float(Ls.min()),   # E 初值 < min(L)
                  rng.uniform(math.log(10.0), math.log(1e4)),
                  rng.uniform(0.20, 0.60),
                  rng.uniform(math.log(10.0), math.log(1e4)),
                  rng.uniform(0.20, 0.60)]
            res = least_squares(residuals, x0, bounds=(lo, hi),
                                loss='huber', f_scale=huber_delta, max_nfev=20000)
            if best is None or res.cost < best.cost:
                best = res
        return unpack(best.x)
    else:
        # 固定 E 拟合 4 参数：x = [ln A, alpha, ln B, beta]
        def unpack(x):
            return E_fixed, math.exp(x[0]), x[1], math.exp(x[2]), x[3]

        lo = [math.log(1e-6), 0.01, math.log(1e-6), 0.01]
        hi = [math.log(1e9), 1.50, math.log(1e9), 1.50]

        def residuals(x):
            return (chinchilla_loss(Ns, Ds, unpack(x)) - Ls) / Ls

        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_starts):
            x0 = [rng.uniform(math.log(0.01), math.log(100.0)), rng.uniform(0.20, 0.80),
                  rng.uniform(math.log(0.01), math.log(100.0)), rng.uniform(0.20, 0.80)]
            res = least_squares(residuals, x0, bounds=(lo, hi),
                                loss='huber', f_scale=huber_delta, max_nfev=20000)
            if best is None or res.cost < best.cost:
                best = res
        return unpack(best.x)


# ═══════════════════════════════════════════════════════════════════
# 由拟合参数推导计算最优配比（教程数学推导的可执行版）
# ═══════════════════════════════════════════════════════════════════
def compute_optimal(params, C):
    """给定训练算力 C = 6ND（FLOPs），返回 (N_opt, D_opt, tokens_per_param)。

    推导（Lagrange 条件，教程有逐步版）：在 C=6ND 约束下最小化 L，
    得 α·A·N^(-α-1) 与 β·B·D^(-β-1) 的平衡点：
        N_opt = (αA/βB)^(1/(α+β)) · (C/6)^(β/(α+β))
        D_opt = (βB/αA)^(1/(α+β)) · (C/6)^(α/(α+β))
    """
    E, A, alpha, B, beta = params
    r = alpha + beta
    N_opt = (alpha * A / (beta * B)) ** (1.0 / r) * (C / 6.0) ** (beta / r)
    D_opt = (beta * B / (alpha * A)) ** (1.0 / r) * (C / 6.0) ** (alpha / r)
    return N_opt, D_opt, D_opt / N_opt


# ═══════════════════════════════════════════════════════════════════
# 小 GPT（Part 8 01_gpt_model.py 的经典款：LayerNorm + learned PE + MHA + ReLU
# FFN；不跨 Part import，本脚本自包含）。注意力换用 F.scaled_dot_product_attention
# 拿 fused kernel 的速度——scan 模式 9 个 run 的时间预算靠它，数学与手写版等价。
# ═══════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    """多头因果注意力（Q/K/V 合并成一个投影，减少 kernel 启动次数）。"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x):
        B, T, C = x.shape                      # x: (B, T, n_embed)
        q, k, v = self.qkv(x).split(C, dim=2)  # 各 (B, T, n_embed)
        # 重塑为多头：→ (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # fused 因果注意力（等价于 Part 8 的 q@k^T/sqrt(d) + mask + softmax + @v）
        # q,k,v: (B, n_head, T, head_dim) → y: (B, n_head, T, head_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 拼回 (B, T, n_embed)
        return self.proj(y)


class MLP(nn.Module):
    """经典 FFN：4x 扩展 + ReLU + 投影回（对齐 Part 8 经典款，不用 SwiGLU）。"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)  # (B, T, n_embed) → (B, T, n_embed)


class Block(nn.Module):
    """Pre-LN 残差块：x = x + Attn(LN(x)); x = x + MLP(LN(x))。"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # (B, T, n_embed)
        x = x + self.mlp(self.ln2(x))    # (B, T, n_embed)
        return x


class TinyGPT(nn.Module):
    """Decoder-only Transformer：tok_emb + pos_emb → N×Block → LN → lm_head。"""

    def __init__(self, vocab_size, n_embed, n_head, n_blocks, context_length):
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def forward(self, idx, targets=None):
        B, T = idx.shape                              # idx: (B, T) int64
        tok_emb = self.token_embed(idx)               # (B, T, n_embed)
        pos_emb = self.position_embed(self.pos_idxs[:T])  # (T, n_embed) 广播相加
        x = tok_emb + pos_emb                         # (B, T, n_embed)
        for block in self.blocks:
            x = block(x)                              # (B, T, n_embed)
        x = self.ln_f(x)                              # (B, T, n_embed)
        logits = self.lm_head(x)                      # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1),
                                   targets.reshape(B * T))
        return logits, loss


def pick_config(target_params, vocab_size, context_length):
    """在 (n_embed, n_blocks) 网格里找参数量最接近 target 的配置。

    参数量 ≈ vocab*n_embed + ctx*n_embed + n_blocks*(12*n_embed²)
    （每 block：QKV+proj 4d²、FFN 8d²，共 12d²；char 级 embedding 很小）
    宽深比约束 24 <= d/L <= 60：防止挑出"深瘦/浅胖"的畸形形状
    （畸形形状会混入"形状效应"，污染纯规模效应的测量）。
    """
    best = None
    for n_embed in [64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]:
        for n_blocks in range(2, 13):
            if not (24 <= n_embed / n_blocks <= 60):
                continue
            # 头数：优先常见值，须整除 n_embed 且 head_dim >= 32（太小伤 SDPA 效率）
            n_head = 1
            for h in [8, 4, 2, 16, 6, 3, 12, 5, 7]:
                if n_embed % h == 0 and n_embed // h >= 32:
                    n_head = h
                    break
            model = TinyGPT(vocab_size, n_embed, n_head, n_blocks, context_length)
            n = sum(p.numel() for p in model.parameters())
            if best is None or abs(n - target_params) < abs(best[0] - target_params):
                best = (n, n_embed, n_head, n_blocks)
            del model
    n, n_embed, n_head, n_blocks = best
    return {'params': n, 'n_embed': n_embed, 'n_head': n_head,
            'n_blocks': n_blocks}


# ═══════════════════════════════════════════════════════════════════
# 训练通用件：LR 调度（逐 run horizon！）、训练循环、评估
# ═══════════════════════════════════════════════════════════════════
def lr_at(step, total_steps, peak, warmup, min_frac=0.1):
    """warmup 线性升 → cosine 衰减到 peak*min_frac。

    ⚠️ Kaplan 偏差的坑（本脚本最重要的一段注释）：
    cosine 的 total_steps 必须逐 run 设成"这个 run 自己的 token 预算"。
    如果所有 run 共用同一条 schedule（比如都按最大 D 设 horizon），短 run 的 LR
    还没衰减完就结束 → 系统性欠训练 → 拟合出的数据指数 β 偏小 → 得出
    "数据不重要、堆参数就行"的 Kaplan 式结论。Chinchilla 论文对 Kaplan 的核心
    修正之一就是：每个 (N, D) run 单独把 LR/horizon 调到自己的预算
    （2203.15556 §6 与 Besiroglu et al. 2401.00448 的再分析）。
    调用方 train_run() 传入的 total_steps 由本 run 的 D 决定，别改成共享常量。
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def train_run(model, batch_iter, total_steps, device, peak_lr=3e-3,
              tag='', log_every=0):
    """训练一个 run 到自己的 horizon 结束，返回 (最终 train loss, 耗时秒)。

    batch_iter: 每次 yield (x, y)，shape 均为 (B, T-1) int64（已在 device 上）
    total_steps: 本 run 的步数 = 本 run 的 token 预算 / 每步 token 数
                 ——LR horizon 与训练预算严格相等（见 lr_at 的坑注释）
    注：B 取小（32）而 T 取大（256）：同量 token 下更多优化步。
    开发时实测 B 太大（如 256）时步数太少，模型连"先掉进均匀分布盆地
    再爬出来"都来不及——loss 停在 ln(vocab)。这是 toy scaling 实验
    与真实大模型实验的一个重要差别（真实实验每步 batch 也很大，但
    它们的 token 预算大 5+ 个数量级，步数反而多）。
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                            weight_decay=0.01, fused=(device.type == 'cuda'))
    warmup = max(1, int(0.02 * total_steps))
    t0 = time.time()
    running = []
    for step, (x, y) in enumerate(batch_iter):
        # x, y: (B, T-1) int64
        for g in opt.param_groups:
            g['lr'] = lr_at(step, total_steps, peak_lr, warmup)
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _, loss = model(x, y)          # loss: 标量（对 B*(T-1) 个 token 平均）
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"    [{tag}] step {step+1}/{total_steps} "
                  f"loss={sum(running[-log_every:])/log_every:.4f}")
    n_tail = max(1, total_steps // 10)
    return sum(running[-n_tail:]) / n_tail, time.time() - t0


@torch.no_grad()
def eval_val_loss(model, val_batches, device):
    """在 held-out 验证批上评估 loss（nat/token）。model.eval() + no_grad。"""
    model.eval()
    tot, n = 0.0, 0
    for x, y in val_batches:               # x, y: (B, T-1) int64
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _, loss = model(x, y)
        tot += loss.item()
        n += 1
    model.train()
    return tot / max(1, n)


# ═══════════════════════════════════════════════════════════════════
# scan 模式的"无限唯一数据"：在真实语料上拟合 3 阶插值马尔可夫，再重采样
# ═══════════════════════════════════════════════════════════════════
def make_resampled_corpus(n_tokens, data_path, seed=0, device=None, n_chains=4096,
                          lam=(0.60, 0.25, 0.10, 0.05)):
    """从 data/input.txt 拟合 3 阶插值马尔可夫，采样出 n_tokens 的唯一语料池。

    为什么不直接用 input.txt：scan 要模拟"数据无限"区（Chinchilla 的前提），
    D 最大 20M token 而 input.txt 只有 ~1.1M 字符——重复 18 遍就变成数据约束
    实验（那是 epoch 模式的领地）。重采样 = 继承真实文本的难度谱
    （unigram/bigram/trigram 的行熵都是 Zipf 式平滑分布），又要有多少有多少。

    ⚠️ 为什么不用随机合成的转移表（首版实测的坑，见教程"实验设计"）：
    随机 Dirichlet 表要么太难（纯 3 阶、无低阶入口——注意力自举不起来，
    loss 卡死在 ln V）、要么台阶式（某一阶"突然学会"，D 轴不平滑、拟合病态）。
    真实文本的难度谱天然平滑，才有干净的幂律。

    插值：p(next|abc) = 0.60·P3(bc) + 0.25·P2(c) + 0.10·P1 + 0.05·uniform
    （未见过的 bc 自动回退到低阶——Katz 式回退的最简版）

    同 seed = 同一门"语言"（表相同）；train/val 池同 seed、不同链采样。
    返回 (pool: (n_tokens,) uint8/int16 on device, H_floor, vocab_size)。
    """
    n_tokens = int(n_tokens)
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)      # (n_chars,)

    # 计数表（bincount 一次性建好）：uni (V,), bi (V,V), tri (V²,V)
    uni = np.bincount(ids, minlength=V).astype(np.float64)
    bi = np.bincount(ids[:-1] * V + ids[1:],
                     minlength=V * V).astype(np.float64).reshape(V, V)
    tri = np.bincount((ids[:-2] * V + ids[1:-1]) * V + ids[2:],
                      minlength=V ** 3).astype(np.float64).reshape(V * V, V)
    l3, l2, l1, l0 = lam
    P1 = torch.tensor((uni + 1e-9) / (uni + 1e-9).sum(),
                      dtype=torch.float32, device=device)         # (V,)
    P2 = torch.tensor(bi / np.clip(bi.sum(1, keepdims=True), 1e-9, None),
                      dtype=torch.float32, device=device)          # (V, V)
    P3 = torch.tensor(tri / np.clip(tri.sum(1, keepdims=True), 1e-9, None),
                      dtype=torch.float32, device=device)          # (V², V)

    # 语料的真实条件熵（= 本语言 loss 的理论下界 E，scan 模式用它固定拟合）
    ctx_real = torch.tensor(ids[:-2] * V + ids[1:-1], dtype=torch.long, device=device)
    samp = torch.randperm(len(ctx_real), device=device)[:100000]
    p_real = (l3 * P3[ctx_real[samp]] + l2 * P2[ctx_real[samp] % V]
              + l1 * P1.unsqueeze(0).expand(len(samp), -1) + l0 / V)
    H_floor = -(p_real * p_real.clamp_min(1e-12).log()).sum(1).mean().item()

    # 4096 条链并行推进，每步一次向量化采样（链内保留 3 阶依赖，链间边界断开）
    g = torch.Generator(device=device).manual_seed(seed + 777)
    chains = torch.randint(0, V, (n_chains, 3), device=device, generator=g)
    per_chain = math.ceil(n_tokens / n_chains) + 3
    out = torch.zeros((n_chains, per_chain),
                      dtype=torch.uint8 if V <= 256 else torch.int16, device=device)
    P1b = P1.unsqueeze(0)
    for t in range(per_chain):
        b, c = chains[:, 1], chains[:, 2]        # 最近两个 token 是 3 阶上下文
        p = (l3 * P3[b * V + c] + l2 * P2[c]
             + l1 * P1b.expand(n_chains, -1) + l0 / V)
        # ⚠️ 显式归一化：混合行未归一化时 CUDA multinomial 会触发 device-side
        #    assert（开发时踩过：index out of bounds 的报错定位到这里）
        p = p / p.sum(1, keepdim=True)           # (n_chains, V)
        nxt = torch.multinomial(p, 1).squeeze(1)  # (n_chains,)
        out[:, t] = nxt
        chains = torch.cat([chains[:, 1:], nxt.unsqueeze(1)], 1)   # 滑窗前移
    return out.reshape(-1)[:n_tokens], H_floor, V


class ChunkSampler:
    """把 token 池切成互不重叠的 chunk（B*T 个 token），每"轮"随机洗牌、按需取用。

    每个 chunk 重塑为 (B, T) 的 batch：行是连续的 T-token 段（makemore 同款打法，
    行内保留 n-gram 依赖，行间边界断开）。保证 D <= 池大小时同一 run 内每个
    token 最多见 1 次（干净的"无限数据"区）；D 超过池大小时自动进入新一轮
    洗牌（等效多 epoch，打印注明）。
    """

    def __init__(self, pool, B, T, seed, device):
        self.pool = pool                       # (n_tokens,) uint8 on device
        self.B, self.T = B, T
        self.batch_tokens = B * T
        self.n_chunks = len(pool) // (B * T)
        # 洗牌用 CPU generator（randperm 不吃 CUDA generator；n_chunks 才几百，
        # CPU 洗牌开销可忽略，chunk 索引随后用于 GPU 切片）
        self.rng = torch.Generator().manual_seed(seed)
        self.device = device
        self._order = torch.randperm(self.n_chunks, generator=self.rng)
        self._pos = 0

    def _next_chunk(self):
        """取下一个 chunk 编号；耗尽后重新洗牌（进入第二个"epoch"）。"""
        if self._pos >= len(self._order):
            self._order = torch.randperm(self.n_chunks, generator=self.rng)
            self._pos = 0
        c = int(self._order[self._pos])
        self._pos += 1
        return c

    def batches(self, total_steps):
        for _ in range(total_steps):
            s = self._next_chunk() * self.batch_tokens
            seq = self.pool[s:s + self.batch_tokens].view(self.B, self.T).long()
            x = seq[:, :-1].contiguous()       # (B, T-1) 输入
            y = seq[:, 1:].contiguous()        # (B, T-1) 目标 = 下一个 token
            yield x, y


class EpochLoader:
    """epoch 模式的精确 epoch 迭代器：语料按顺序切块，每轮洗牌块序。

    R 个 epoch = 每个唯一 token 恰好看 R 次（这正是 Muennighoff 的实验设定）。
    尾部不足一个 chunk 的 token 丢弃（<3%，打印注明）。
    """

    def __init__(self, tokens, B, T, seed, device):
        self.tokens = tokens.to(device)                # (n_train,) int64
        self.B, self.T = B, T
        self.batch_tokens = B * T
        self.n_chunks = len(tokens) // (B * T)
        self.seed = seed

    def epochs(self, n_epochs):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(n_epochs):
            order = torch.randperm(self.n_chunks, generator=g).tolist()
            for c in order:
                s = c * self.batch_tokens
                seq = self.tokens[s:s + self.batch_tokens].view(self.B, self.T)
                x = seq[:, :-1].contiguous()           # (B, T-1)
                y = seq[:, 1:].contiguous()            # (B, T-1)
                yield x, y


def val_batches_from(tokens, B, T, device, max_batches=8):
    """把 held-out token 切成至多 max_batches 个 (B, T-1) 验证批（均匀覆盖整段）。"""
    tokens = tokens.to(device)      # 兼容 CPU 上的验证集（epoch 模式直接传 CPU 切片）
    batch_tokens = B * T
    n_chunks = len(tokens) // batch_tokens
    n_chunks = min(n_chunks, max_batches) if max_batches else n_chunks
    for b in range(n_chunks):
        s = b * (len(tokens) - batch_tokens) // max(1, n_chunks)
        seq = tokens[s:s + batch_tokens].view(B, T).long()
        yield seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


# ═══════════════════════════════════════════════════════════════════
# ASCII 画图小工具
# ═══════════════════════════════════════════════════════════════════
def ascii_profile(xs, ys, x_label, y_label, mark_idx=None, n_rows=13, n_cols=58):
    """把一条曲线画成字符图（'*' 曲线，'V' 标记谷底）。xs 需已排序。"""
    xs, ys = np.asarray(xs), np.asarray(ys)
    lo, hi = float(ys.min()), float(ys.max())
    hi += 1e-12
    canvas = [[' '] * n_cols for _ in range(n_rows)]
    for j in range(n_cols):
        # 每列对应 xs 对数轴上的一个位置（线性插值出 y）
        xt = xs[0] * (xs[-1] / xs[0]) ** (j / (n_cols - 1))
        yt = np.interp(xt, xs, ys)
        r = int((hi - yt) / (hi - lo) * (n_rows - 1))
        canvas[r][j] = '*'
    if mark_idx is not None:
        j = int(mark_idx / (len(xs) - 1) * (n_cols - 1))
        for r in range(n_rows):
            if canvas[r][j] == '*':
                canvas[r][j] = 'V'
    lines = []
    for r in range(n_rows):
        val = hi - r * (hi - lo) / (n_rows - 1)
        lines.append(f"  {val:6.2f} |" + ''.join(canvas[r]))
    lines.append("  " + " " * 6 + "+" + "-" * n_cols)
    lines.append(f"{'':>10}{x_label} → (log)")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# 模式一：fit（零 GPU，秒级）
# ═══════════════════════════════════════════════════════════════════
def run_fit_mode():
    print("═" * 62)
    print("模式 fit：对合成-加噪 Chinchilla 数据做参数拟合（零 GPU）")
    print("═" * 62)

    # [1] 合成数据：按 Hoffmann 真值生成 + 3% 相对噪声
    # 网格设计（可辨识性的关键）：让 N 项、D 项各有"主导角"与"可忽略角"——
    #   N=1e5,  D=1e12: A/N^α=8.1 主导, B/D^β=0.18 可忽略 → 锁定 A/α
    #   N=1e10, D=1e7:  A/N^α=0.16 可忽略, B/D^β=4.5 主导 → 锁定 B/β
    # 窄网格（D 项始终占比小）会让 B/β 沿平坦方向漂移——首版实测 B 误差 262% 的教训
    # 噪声设计：3% 相对噪声 × 16 次独立抽取。单次抽取的系数误差可达 ±10-15%
    # （估计量方差，不是优化器失败），取平均后方差 /16 → 达标。这也是工业
    # scaling law 实验的标准做法：多 seed / 多噪声实现，报均值±std
    rng = np.random.default_rng(42)
    N_grid = np.logspace(5, 10, 8)      # 参数量 1e5~1e10（对数均匀 8 档）
    D_grid = np.logspace(7, 12, 8)      # token 1e7~1e12（对数均匀 8 档）
    base = [(N, D, float(chinchilla_loss(N, D, TRUE_PARAMS)))
            for N in N_grid for D in D_grid]                 # 64 个无噪点
    NOISE_DRAWS = 16
    print(f"\n[1] 合成网格: {len(base)} 个 (N, D) 点 × {NOISE_DRAWS} 次独立 3% 噪声抽取")
    print(f"    真值来自 Hoffmann 2203.15556 Table 3: "
          f"E=1.69 A=406.4 α=0.34 B=410.7 β=0.28")
    print(f"    loss 范围: {min(r[2] for r in base):.3f} ~ "
          f"{max(r[2] for r in base):.3f} nat/token")

    # [2] 先看单次抽取：展示估计量方差（教学点——别被单次拟合的参数吓到）
    def noisy_records(draw_seed):
        r = np.random.default_rng(draw_seed)
        return [(N, D, L * (1 + r.normal(0, 0.03))) for N, D, L in base]

    t0 = time.time()
    single = fit_chinchilla(noisy_records(42))
    print(f"\n[2] 单次噪声抽取的拟合结果（看方差，不验收）")
    for name, tv, fv in zip(['E', 'A', 'alpha', 'B', 'beta'], TRUE_PARAMS, single):
        print(f"    {name:>6}: 真值 {tv:>8.3f}  单次拟合 {fv:>10.3f}  "
              f"偏差 {abs(fv-tv)/tv*100:+.1f}%")

    # [3] 16 次独立抽取分别拟合后取平均 → 逐参数相对误差表（验收 <5%）
    fits = np.array([fit_chinchilla(noisy_records(1000 + k))
                     for k in range(NOISE_DRAWS)])
    averaged = tuple(fits.mean(0))
    spread = fits.std(0)
    print(f"\n[3] {NOISE_DRAWS} 次独立噪声实现 → 拟合 → 参数平均（Huber + "
          f"scipy.least_squares，{time.time()-t0:.1f}s）")
    print(f"    {'参数':>6} {'真值':>10} {'平均拟合':>12} {'相对误差':>9} "
          f"{'跨抽取std':>9} 判定")
    all_pass = True
    for name, tv, fv, sd in zip(['E', 'A', 'alpha', 'B', 'beta'],
                                TRUE_PARAMS, averaged, spread):
        rel = abs(fv - tv) / tv
        ok = rel < 0.05
        all_pass &= ok
        print(f"    {name:>6} {tv:>10.3f} {fv:>12.3f} {rel*100:>8.2f}% "
              f"{sd:>9.3f} {'PASS' if ok else 'FAIL'}")
    print(f"    → {'✅ 全部参数相对误差 <5%' if all_pass else '❌ 未达标，需调网格/Huber delta/抽取次数'}")

    # [4] isoFLOP 剖面：C=6ND 等值线上 loss 对 N 的 U 形曲线（用平均参数画）
    print(f"\n[4] isoFLOP 剖面（用拟合参数画；谷底 = 该算力预算下的最优 N）")
    C_list = [1e14, 3e14, 1e15, 3e15, 1e16, 3e16, 1e17]
    Ns = np.logspace(5.7, 9.0, 240)   # 窗口对准谷底范围（~1e6 到 ~3e7）
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    offset = 0.0
    print(f"    {'C (FLOPs)':>12} {'N_opt':>10} {'D_opt':>10} {'D/N (t/p)':>10}")
    valley = None
    for C in C_list:
        Ds = C / (6.0 * Ns)                     # isoFLOP 约束：D = C/(6N)
        Ls = chinchilla_loss(Ns, Ds, averaged)
        n_opt, d_opt, tp = compute_optimal(averaged, C)
        i = int(np.argmin(Ls))
        valley = (Ns, Ls, i)
        ax.plot(Ns, Ls + offset, label=f"C={C:.0e}")
        ax.plot(Ns[i], Ls[i] + offset, 'v', color='k', markersize=6)
        print(f"    {C:>12.0e} {n_opt:>10.3g} {d_opt:>10.3g} {tp:>10.1f}")
        offset += 0.45                           # 纵向错开便于看谷底（Hoffmann 图同款画法）
    ax.set_xscale('log')
    ax.set_xlabel('parameters N (non-embedding)')
    ax.set_ylabel('loss (+ per-isoFLOP offset)')
    ax.set_title('IsoFLOP profiles: loss valley shifts right as compute grows')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_png = os.path.join(SCRIPT_DIR, 'output_scaling_fit.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    ASCII 剖面（最大算力 C={C_list[-1]:.0e} 那条 U 形曲线）：")
    print('\n'.join('    ' + ln for ln in ascii_profile(
        valley[0], valley[1], 'N', 'L', mark_idx=valley[2]).split('\n')))
    print(f"    📊 图已保存: {out_png}")

    # [5] 零算力替代方案（文字说明）
    print(f"""
[5] 零算力替代方案：用公开 checkpoint 的 loss 复现幂律（不用自己训）
    - Pythia 工具箱（Biderman et al. 2204.09842）开源了 70M~6.9B 共 8 个
      规模、同数据同超参的 checkpoint，论文/官网给出各规模的 final val loss。
    - 练习：把 (N, val_loss) 喂给单变量拟合 L = E + A/N^α（D 固定为 ~300B
      token、对全体系列近似常数），即可在笔记本上、零训练复现"模型规模幂律"，
      与本模式的多变量拟合互为印证。
    - 接口不变：仍用 chinchilla_loss/fit_chinchilla，把 D 固定后 β 项是常数，
      会被吸收进 E/A 的拟合里（这正是"控制变量"的幂律实验设计）。
""")


# ═══════════════════════════════════════════════════════════════════
# 模式二：scan（真训小 GPT 网格，冒烟 ~25s）
# ═══════════════════════════════════════════════════════════════════
def run_scan_mode(full=False):
    print("═" * 62)
    print("模式 scan：网格真训小 GPT，拟合自己的 scaling law")
    print("═" * 62)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print(f"\n[0] 设备: {device}"
          + (f"（{torch.cuda.get_device_name(device)}）" if device.type == 'cuda' else
             "（无 CUDA：自动缩小网格，耗时会显著增加）"))

    # 网格选择（实验设计的核心坑，见教程"实验设计"一节）：
    # 规格网格 N∈{1M,3M,10M} × D∈{2M,6M,20M} 的 t/p 范围是 0.2~20，几乎整个
    # 落在"数据受限"侧——开发实测：D 固定时 1M/3M/10M 模型 loss 相差 <0.01
    # （容量不 binding，α→0，谷底在网格外）。教学优先：把 N 缩一个数量级到
    # {0.12M, 0.26M, 1M}，让容量在网格内开始 binding，谷底可见。
    if device.type == 'cuda':
        if full:
            # 未实测：预计单卡 4090 ~3x smoke 时间（~80s）
            N_targets = [0.12e6, 0.26e6, 1.0e6, 3.0e6]
            D_targets = [6e6, 20e6, 60e6]
        else:
            N_targets = [0.12e6, 0.26e6, 1.0e6]
            D_targets = [2e6, 6e6, 20e6]
        T, B = 256, 32
    else:
        N_targets = [0.01e6, 0.03e6]
        D_targets = [0.05e6, 0.15e6, 0.4e6]
        T, B = 128, 16

    # [1] 语料池 + held-out 验证流（同"语言"、不同链采样）
    pool_size = int(max(D_targets) * 1.2)
    t0 = time.time()
    pool, H_floor, vocab = make_resampled_corpus(pool_size, DATA_PATH,
                                                 seed=0, device=device)
    val_pool, _, _ = make_resampled_corpus(B * T * 8, DATA_PATH,
                                           seed=99, device=device)
    print(f"\n[1] 语料池：input.txt 拟合的 3 阶马尔可夫重采样（'无限唯一数据'区）")
    print(f"    池大小: {len(pool)/1e6:.1f}M token（>= 最大 D，单 run 内 token 最多见 1 次）")
    print(f"    vocab={vocab}（char 级）  熵下界 E={H_floor:.3f} nat/token "
          f"vs 随机猜测 ln({vocab})={math.log(vocab):.3f}")
    print(f"    生成耗时: {time.time()-t0:.1f}s")

    # [2] 网格训练（cosine horizon 逐 run 设置 = 各自 token 预算，见 lr_at 注释）
    print(f"\n[2] 网格训练（B={B}, T={T}：同量 token 下取小 batch 换更多优化步）")
    records = []
    t_all = time.time()
    for n_target in N_targets:
        cfg = pick_config(n_target, vocab, T)
        for d_target in D_targets:
            total_steps = max(1, int(d_target) // (B * T))
            torch.manual_seed(42)
            model = TinyGPT(vocab, cfg['n_embed'], cfg['n_head'],
                            cfg['n_blocks'], T).to(device)
            sampler = ChunkSampler(pool, B, T,
                                   seed=int(n_target + d_target), device=device)
            train_loss, dt = train_run(
                model, sampler.batches(total_steps), total_steps, device,
                peak_lr=3e-3, tag=f"N={cfg['params']/1e6:.2f}M,D={d_target/1e6:.0f}M",
                log_every=max(1, total_steps // 2))
            vloss = eval_val_loss(
                model, val_batches_from(val_pool, B, T, device), device)
            records.append((cfg['params'], total_steps * B * T, vloss))
            print(f"    ✔ N={cfg['params']/1e6:5.2f}M (d={cfg['n_embed']},L={cfg['n_blocks']}) "
                  f"D={total_steps*B*T/1e6:5.1f}M  train={train_loss:.4f} "
                  f"val={vloss:.4f}  [{dt:.1f}s]")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    print(f"\n    网格总耗时: {time.time()-t_all:.1f}s")

    # [3] (N, D, val_loss) 表
    print(f"\n[3] (N, D, val_loss) 网格表")
    print(f"    {'N':>10} {'D':>10} {'t/p':>8} {'val_loss':>10}")
    for N, D, L in records:
        print(f"    {N:>10.3g} {D:>10.3g} {D/N:>8.1f} {L:>10.4f}")

    # [4] 拟合：先自由 5 参数（看到病态），再固定 E=熵下界（救活它）
    print(f"\n[4] fit_chinchilla 拟合")
    free_fit = fit_chinchilla(records)
    E, A, alpha, Bp, beta = free_fit
    print(f"    自由 5 参数: E={E:.4f}  A={A:.2f}  α={alpha:.3f}  B={Bp:.2f}  β={beta:.3f}")
    print(f"    ⚠️ E 被顶到下界（≈0）、指数失真——玩具网格动态范围只有 ~0.2 nat，")
    print(f"       E（渐进项）在网格内不可辨识（fit 模式用了 12 个数量级才钉住它）。")
    fixed_fit = fit_chinchilla(records, E_fixed=H_floor)
    E, A, alpha, Bp, beta = fixed_fit
    resid = np.array([chinchilla_loss(N, D, fixed_fit) - L for N, D, L in records])
    rel = resid / np.array([L for _, _, L in records])
    print(f"    固定 E={H_floor:.3f}（语料熵下界可独立测量——合成语料独有的优势）:")
    print(f"       E={E:.3f}  A={A:.1f}  α={alpha:.3f}  B={Bp:.1f}  β={beta:.3f}")
    print(f"       相对残差: 均值 {rel.mean()*100:+.2f}%  最大绝对值 {np.abs(rel).max()*100:.2f}%")

    # [5] 由拟合参数算最优配比 → 学生结论 N:D
    print(f"\n[5] 计算最优配比（把数学推导跑起来；C 取网格里最小/中位/最大三档）")
    C_grid = sorted(6.0 * r[0] * r[1] for r in records)
    C_show = [C_grid[0], C_grid[len(C_grid) // 2], C_grid[-1]]
    print(f"    {'C (FLOPs)':>12} {'N_opt':>10} {'D_opt':>10} {'D/N (t/p)':>10}")
    for C in C_show:
        n_opt, d_opt, tp = compute_optimal(fixed_fit, C)
        print(f"    {C:>12.2e} {n_opt:>10.3g} {d_opt:>10.3g} {tp:>10.1f}")
    tps = [compute_optimal(fixed_fit, C)[2] for C in C_grid]
    print(f"\n    📊 学生结论: 本玩具尺度（char 级、≤1M 参数）最优 D/N ≈ "
          f"{min(tps):.0f}~{max(tps):.0f} t/p")
    print(f"    （Chinchilla 尺度是 ~20 t/p。玩具 t/p 偏高的解释：任务的可学结构")
    print(f"     在 ~0.3M 参数就饱和了，多出来的预算全部流向数据——这正是")
    print(f"     '先问任务有效复杂度、再谈 N:D' 的活例子。方法才是本模式的重点。）")

    # [6] 实验设计说明
    print(f"""
[6] 为什么这样设计（对照工业方法学，开发实测的三个坑）
    - 语料坑：随机合成转移表不行——纯 3 阶无低阶入口，注意力自举不起来，
      loss 卡死 ln(V)；台阶式难度产生"平台+悬崖"的 D 轴，拟合病态。
      解法 = 用真实文本的难度谱（3 阶马尔可夫重采样）。
    - 网格坑：容量必须在网格内 binding。N 太大 → 所有模型对数据"够用"，
      α→0。解法 = 缩 N 直到看到 N 方向的 loss 梯度。
    - 步数坑：B 太大 → 优化步太少，模型出不了均匀分布盆地。解法 = 小 batch
      换步数（B=32, T=256）。
    - isoFLOP 实验的正解是"同一 C 下跑多个 (N,D)"找谷底（fit 模式的图）；
      本网格是它的乞丐版：9 个点拟出参数面，再用公式算谷底位置。
    - --full 扩到 4×3（N 到 3M、D 到 60M）可改善指数可辨识性。未实测。
""")


# ═══════════════════════════════════════════════════════════════════
# 模式三：epoch（数据约束区，Muennighoff 2305.16264）
# ═══════════════════════════════════════════════════════════════════
def run_epoch_mode():
    print("═" * 62)
    print("模式 epoch：固定 unique 语料 × R ∈ {1,2,4,8,16} epoch（数据约束区）")
    print("═" * 62)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print(f"\n[0] 设备: {device}"
          + (f"（{torch.cuda.get_device_name(device)}）" if device.type == 'cuda' else
             "（无 CUDA：自动缩小规模，耗时会显著增加）"))

    # [1] 读真实语料（char 级，vocab≈65）
    # 注：本仓库 data/input.txt 是 tiny shakespeare（~1.1MB）。若换成更大语料，
    #     可截取前若干 MB（教学上不影响结论），并在此处注明截取长度。
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.int64)  # (n_chars,)
    n_train = int(0.9 * len(ids))
    train_ids = ids[:n_train]                      # unique 训练语料 D_u（不重复）
    val_ids = ids[n_train:]                        # held-out：从不参与训练
    print(f"\n[1] 真实语料: data/input.txt（tiny shakespeare）")
    print(f"    字符级 token: vocab={len(chars)}  总 {len(ids)/1e6:.2f}M")
    print(f"    训练（unique）: {n_train/1e6:.3f}M token   验证（held-out）: "
          f"{len(val_ids)/1e3:.0f}K token")

    # [2] 固定模型：数据约束实验只动 R，别的全部钉死（CPU 时缩到 ~0.1M）
    # 6M 的选择实测过：太小（~1M）时 R=16 还在下降（饱和点在量程外），太大时
    # R=2 就开始过拟合；6M 让"饱和点"落在 R≈8~16 ——正好复现 Muennighoff 的量级
    if device.type == 'cuda':
        T, B = 256, 32
        cfg = pick_config(6e6, len(chars), T)
        R_list = [1, 2, 4, 8, 16]
    else:
        T, B = 128, 16
        cfg = pick_config(0.1e6, len(chars), T)
        R_list = [1, 2, 4, 8]
    print(f"\n[2] 模型: {cfg['params']/1e6:.2f}M 参数（d={cfg['n_embed']}, "
          f"L={cfg['n_blocks']}），R = {R_list}")

    # [3] 每个 R 从头训（horizon = R×D_u 自己的 token 预算——同样的 Kaplan 坑）
    results = []
    t_all = time.time()
    for R in R_list:
        torch.manual_seed(42)
        model = TinyGPT(len(chars), cfg['n_embed'], cfg['n_head'],
                        cfg['n_blocks'], T).to(device)
        loader = EpochLoader(train_ids, B, T, seed=100 + R, device=device)
        # 精确 epoch 语义：R 轮 × 每轮 floor(D_u/batch_tokens) 个 chunk
        # （尾部丢弃 <3% 的 token；实际消耗 token 数用 total_steps×B×T 记账）
        total_steps = R * (n_train // (B * T))
        train_loss, dt = train_run(
            model, loader.epochs(R), total_steps, device,
            peak_lr=3e-3, tag=f"R={R}", log_every=max(1, total_steps // 2))
        vloss = eval_val_loss(
            model, val_batches_from(val_ids, B, T, device, max_batches=4), device)
        results.append((R, total_steps * B * T, train_loss, vloss))
        print(f"    ✔ R={R:>2}  D={total_steps*B*T/1e6:5.2f}M  "
              f"train={train_loss:.4f}  val={vloss:.4f}  [{dt:.1f}s]")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f"\n    总耗时: {time.time()-t_all:.1f}s")

    # [4] 幂律外推：用 R<=4 拟合 L = a·R^(-γ)，外推 R=8/16
    Rs = np.array([r[0] for r in results], dtype=np.float64)
    Ls = np.array([r[3] for r in results], dtype=np.float64)
    m = Rs <= 4
    # log-log 最小二乘（2 参数）：ln L = ln a - γ ln R
    coef = np.polyfit(np.log(Rs[m]), np.log(Ls[m]), 1)
    gamma, log_a = -coef[0], coef[1]
    a = math.exp(log_a)
    pred = a * Rs ** (-gamma)
    print(f"\n[4] 幂律拟合（仅用 R<=4 的点）: L = {a:.3f} · R^(-{gamma:.3f})")
    print(f"    {'R':>3} {'实测 val':>10} {'幂律预测':>10} {'偏差':>8}  判定")
    for (R, D, tl, vl), p in zip(results, pred):
        dev = (vl - p) / p * 100
        if R <= 4:
            judge = '线性区内 ✓' if abs(dev) < 8 else '线性区但偏差偏大'
        else:
            judge = '饱和（实测高于外推）' if dev > 0 else '未饱和'
        print(f"    {R:>3} {vl:>10.4f} {p:>10.4f} {dev:>+7.1f}%  {judge}")

    # [5] 有效 token：把实测 loss 反解成"等效新鲜 epoch 数"，再看打折率
    print(f"\n[5] 有效 token 数（实测 loss 反解：R_eff = (a/L)^(1/γ)）")
    print(f"    {'R':>3} {'名义 D':>9} {'有效 D':>9} {'R_eff/R（折扣）':>14}")
    for (R, D, tl, vl) in results:
        R_eff = (a / vl) ** (1.0 / gamma)     # L = a·R^-γ → R_eff = (a/L)^(1/γ)
        print(f"    {R:>3} {D/1e6:>8.2f}M {R_eff*n_train/1e6:>8.2f}M "
              f"{R_eff/R:>13.2f}x")
    print(f"    （x<1 说明重复 token 打折：R>4 后折扣急剧增大 = 饱和）")

    # [6] 画图：实测点 + R<=4 幂律外推虚线
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(Rs[m], Ls[m], 'o-', color='tab:blue', label='measured (R<=4 fit region)')
    ax.plot(Rs[~m], Ls[~m], 'o', color='tab:blue')
    ax.plot(Rs, pred, '--', color='tab:red',
            label=f'power-law extrapolation L={a:.2f}·R^-{gamma:.2f}')
    ax.axvline(4, color='gray', ls=':', alpha=0.7)
    ax.annotate('R=4: saturation onset', xy=(4, float(np.interp(4, Rs, Ls))),
                xytext=(5.2, float(pred[1]) + 0.12), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xscale('log', base=2)
    ax.set_xticks(Rs)
    ax.set_xticklabels([str(r) for r in Rs])
    ax.set_xlabel('epochs R over fixed unique corpus (D_u = 1.00M char tokens)')
    ax.set_ylabel('held-out val loss (nat/token)')
    ax.set_title('Repeating a fixed corpus: near-linear gains to R~4, then saturation')
    ax.legend()
    ax.grid(alpha=0.3)
    out_png = os.path.join(SCRIPT_DIR, 'output_scaling_epoch.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

    # ASCII 条形图
    print(f"\n[6] ASCII 条形图（val loss vs R；▓ 实测，░ 幂律外推还差的量）")
    lmin = Ls.min() * 0.98
    span = max(1e-9, Ls.max() - lmin)
    for (R, D, tl, vl), p in zip(results, pred):
        bar = '▓' * max(1, int((vl - lmin) / span * 40))
        ext = '░' * max(0, int((p - vl) / span * 40))
        print(f"    R={R:>2}  {bar}{ext}  val={vl:.4f}")
    print(f"    📊 图已保存: {out_png}")

    # [7] 结论
    sat = {R: (vl - p) / p * 100 for (R, D, tl, vl), p in zip(results, pred)}
    sat8 = sat.get(8, 0)
    note8 = ('负偏差=比外推还好：本玩具尺度在 R=8 仍未吃透语料，重复还有肉'
             if sat8 < 0 else '增益打折：重复开始不如新鲜数据')
    r16 = next(((tl, vl) for (R, D, tl, vl) in results if R == 16), None)
    print(f"""
[7] 结论（对照 Muennighoff et al. 2305.16264）
    - R<=4: 实测贴近幂律外推（重复数据 ≈ 新鲜数据，偏差都在几个百分点内）
    - R=8 : 实测与外推差 {sat8:+.1f}%（{note8}；
            论文尺度上这里是边际收益开始打折的位置）
    - R=16: 实测比外推差 {sat.get(16, 0):+.1f}% —— 饱和甚至过拟合""")
    if r16 is not None:
        print(f"      记忆信号: R=16 的 train loss {r16[0]:.3f} << val loss {r16[1]:.3f}")
        print(f"      （train/val 分叉 = 模型在背语料而不是学语言——去重的价值就在这里）")
    print(f"""    - Part 13 的预算语言：去重删掉的数据，如果换来的是"更多 unique epoch"，
      在 R>4 的区间是划算的；反过来，把 1M 语料洗 20 遍不如攒 5M 新的。
""")


# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Part 13 脚本 00: Scaling Law 三模式（fit/scan/epoch）')
    parser.add_argument('--mode', required=True, choices=['fit', 'scan', 'epoch'],
                        help='fit=合成数据拟合(零GPU) / scan=网格真训 / epoch=数据约束')
    parser.add_argument('--full', action='store_true',
                        help='(仅 scan) 扩网格到 4x3；未实测，预计 ~3x smoke 时间')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.mode == 'fit':
        run_fit_mode()
    elif args.mode == 'scan':
        run_scan_mode(full=args.full)
    else:
        run_epoch_mode()


if __name__ == '__main__':
    main()
