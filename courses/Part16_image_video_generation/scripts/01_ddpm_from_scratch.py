#!/usr/bin/env python3
"""
Part 16 - 脚本 01: 从零手写 DDPM（2D 玩具分布上的扩散模型全流程）
目标：把扩散模型的三段核心数学全部手写并验证——
  ① 前向加噪闭式：q(x_t|x_0) = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε   （DDPM 论文式 4）
  ② 训练：网络从 (x_t, t) 预测噪声 ε，损失 = ‖ε − ε̂‖²（等价于变分下界的简化）
  ③ 采样：从 x_T~N(0,I) 反向逐步去噪 p_θ(x_{t−1}|x_t)（论文式 11）
对应教程：tutorial/01_ddpm_from_scratch.md

运行（CPU <30 秒）：python 01_ddpm_from_scratch.py
任务：学会生成 2D "双月环"分布（机制与 512×512 图像生成完全同构，只是维度小）。
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 400                                   # 扩散步数


# ═══ ① 噪声 schedule 与前向闭式（手写 DDPM §4）═══
def make_betas(T, beta_start=1e-4, beta_end=0.02):
    """线性 β schedule（论文 §4；后续有 cosine 等改进，原理相同）。"""
    return torch.linspace(beta_start, beta_end, T)


betas = make_betas(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)          # ᾱ_t：t 步累积保留的信号比例


def q_sample(x0, t, noise):
    """前向闭式：x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε —— 任意 t 一步到位，无需迭代！
    这条闭式是扩散模型能高效训练的关键（训练时随机采 t 直接算）。"""
    s = alphas_cumprod[t].view(-1, 1).to(x0.device)
    return s.sqrt() * x0 + (1 - s).sqrt() * noise


# ═══ ② 去噪网络：给定 (x_t, t) 预测 ε ═══
class Denoiser(nn.Module):
    """2D toy 版"U-Net"：位置编码 t 进入网络（真实模型用 sinusoidal/AdaLN）。"""

    def __init__(self, hidden=128, t_dim=32):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(1, t_dim), nn.GELU(), nn.Linear(t_dim, t_dim))
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2))

    def forward(self, x_t, t):                     # x_t: (B, 2), t: (B,)
        temb = self.t_mlp(t.view(-1, 1).float() / T)
        return self.net(torch.cat([x_t, temb], dim=-1))


# ═══ ③ 采样循环（论文式 11 的祖传写法）═══
@torch.no_grad()
def sample(model, n=2000):
    x = torch.randn(n, 2, device=DEVICE)           # x_T ~ N(0, I)
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        eps_pred = model(x, torch.full((n,), t, device=DEVICE))
        mean = (1 / alphas[t].sqrt().to(DEVICE)) * \
               (x - betas[t].to(DEVICE) / (1 - alphas_cumprod[t].to(DEVICE)).sqrt() * eps_pred)
        var = betas[t].to(DEVICE)
        x = mean + var.sqrt() * z
    return x


def make_data(n=4096):
    """两个"月环"（玩具版分布——对应真实世界的高维图像流形）。"""
    inner = torch.tensor([[math.cos(a) * 0.6, math.sin(a) * 0.6 + 0.5]
                          for a in [i * 0.08 for i in range(40)]])
    outer = torch.tensor([[math.cos(a), math.sin(a)] for a in [0.4 + i * 0.06 for i in range(60)]])
    idx = torch.randint(0, len(inner) + len(outer), (n,))
    all_pts = torch.cat([inner, outer], dim=0)
    return all_pts[idx] + 0.03 * torch.randn(n, 2)


def main():
    print("═══ 手写 DDPM（2D 玩具分布）═══")
    print(f"  device={DEVICE}, T={T}\n")

    global betas, alphas, alphas_cumprod
    betas, alphas, alphas_cumprod = betas.to(DEVICE), alphas.to(DEVICE), alphas_cumprod.to(DEVICE)

    data = make_data().to(DEVICE)

    # ── 训练：ε 预测目标（注意随机采 t——前向闭式让这一步 O(1)）──
    model = Denoiser().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(3000):
        x0 = data[torch.randint(0, len(data), (256,))]
        t = torch.randint(0, T, (256,), device=DEVICE)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, noise)               # ← 前向闭式：一步加噪
        eps_pred = model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)         # 简化后的训练目标：预测噪声
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 600 == 0:
            print(f"  step {step:4d}  denoising loss = {loss.item():.4f}")

    # ── 采样 ──
    samples = sample(model, n=2000)
    mean_dist = samples.mean(0) - data.mean(0)
    std_ratio = samples.std(0) / data.std(0)
    print(f"\n  采样 {len(samples)} 点 vs 真实数据:")
    print(f"    均值偏移: [{mean_dist[0]:.3f}, {mean_dist[1]:.3f}]（≈0 = 分布学到了中心）")
    print(f"    方差比  : [{std_ratio[0]:.3f}, {std_ratio[1]:.3f}]（≈1 = 学到了形状）")
    ok = mean_dist.abs().max() < 0.1 and (std_ratio - 1).abs().max() < 0.15
    print(f"    {'✅ 分布匹配' if ok else '❌ 分布偏差过大'}（增加步数/训练量可改善）")

    print("""
═══ 扩散三段数学的"账本"（面试向）═══
  前向闭式 q(x_t|x_0)：√ᾱ_t·x₀ + √(1−ᾱ_t)·ε —— ᾱ_t 就是"信号保留比例"，
    t 越大越接近纯噪声；训练时随机采 t 一步到位（不必迭代 T 次）。
  训练目标 ‖ε − ε̂‖²：从 VLB 推导简化后的等价形式（噪声加权谱），DDPM 论文 §3.2。
  采样 p(x_{t−1}|x_t)：均值由 ε̂ 反推 x₀ 的方向，方差由 β_t 控制；
    t=0 时 z=0（最后一步不加噪）。
  💡 面试："为什么扩散训练能一步加噪？"→ 前向是固定的高斯马尔可夫链，
     任意时刻的边际分布有闭式解（ᾱ 的累积乘积）——这是 vs GAN/flow 的核心差异之一。
  下一步（02 章）：Latent Diffusion 把这套数学搬到 VAE 潜空间 + cross-attention
  条件注入 → 就是我们熟知的 Stable Diffusion 文生图。""")


if __name__ == '__main__':
    main()
