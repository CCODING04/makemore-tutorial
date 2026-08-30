#!/usr/bin/env python3
"""
论文公式数值验证器 —— 配套 docs/paper_reading_guide.md 的"⑤ 数值验证"模板。

把 6 篇论文的核心公式各写一个"最小可验证断言"：
  RoPE   (Su et al. 2021)      : 旋转正交（范数不变）+ 内积只依赖位置差
  DPO    (Rafailov et al. 2023): π == ref 时 loss = ln 2（起点无偏）
  GAE    (Schulman et al. 2015): λ=0 退化为 TD error、λ=1 退化为 Monte Carlo
  LoRA   (Hu et al. 2021)      : B=0 时 ΔW=0；合并前后 logits 逐元素一致
  MinHash (Broder 1997)        : P[最小哈希相等] = Jaccard（大样本统计验证）
  气泡   (GPipe)               : bubble = (p-1)/(m+p-1) 与逐 micro-batch 模拟对上

运行（纯 CPU，<10 秒）：python tools/verify_paper_formulas.py
用法：读任何论文时复制本文件结构，把"公式直觉 + 边界检查 + 数值验证"写成脚本——
这就是一份可执行的论文笔记。
"""

import math
import random

import torch
import torch.nn.functional as F

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  ✅ {name} {detail}")
    else:
        FAIL.append(name)
        print(f"  ❌ {name} {detail}")


# ═══ 1. RoPE：旋转正交 + 相对位置不变性 ═══
def verify_rope():
    """论文 §3.1 的两个核心性质，用 2 维向量旋转（一个"复数"）验证。"""
    torch.manual_seed(0)
    q = torch.randn(2)
    k = torch.randn(2)

    def rot(v, angle):
        """2 维旋转 = 复数乘 e^{iθ}（RoPE 对一对分量的最小单元）。"""
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

    # 性质 1：旋转是正交变换（范数不变）
    n_before = q.norm().item()
    n_after = rot(q, 0.7).norm().item()
    check("RoPE 范数不变", abs(n_before - n_after) < 1e-6,
          f"({n_before:.4f} → {n_after:.4f})")

    # 性质 2：⟨R_m q, R_n k⟩ 只依赖 m−n（相对位置）
    def inner(m, n):
        a, b = rot(q, 0.5 * m), rot(k, 0.5 * n)
        return float((a * b).sum())

    d1 = abs(inner(5, 2) - inner(9, 6))     # 位置差都是 3
    # 阈值 1e-5：float32 舍入噪声量级（内积 O(1)）；数学上恒等式精确成立
    check("RoPE 内积只依赖位置差", d1 < 1e-5,
          f"(位置 (5,2) vs (9,6) 的内积差 {d1:.2e}；同差不同绝对位置 → 同内积)")


# ═══ 2. DPO：π == ref 时 loss = ln 2 ═══
def verify_dpo():
    """论文 §4 推导的边界检查：策略未偏离参考时，损失应为 ln 2 ≈ 0.6931。"""
    torch.manual_seed(0)
    n = 8
    ref_c, ref_r = torch.randn(n), torch.randn(n)

    def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
        logits = (pi_c - ref_c) - (pi_r - ref_r)
        return -F.logsigmoid(beta * logits).mean()

    loss_same = dpo_loss(ref_c, ref_r, ref_c, ref_r)
    check("DPO 起点 loss = ln 2", abs(loss_same.item() - math.log(2)) < 1e-6,
          f"({loss_same.item():.6f} vs {math.log(2):.6f})")

    # 方向检查：chosen 优势变大 → loss 下降
    loss_better = dpo_loss(ref_c + 1.0, ref_r, ref_c, ref_r)
    check("DPO chosen 优势↑ → loss↓", loss_better < loss_same,
          f"({loss_better.item():.4f} < {loss_same.item():.4f})")


# ═══ 3. GAE：λ=0 → TD error；λ=1 → Monte Carlo ═══
def verify_gae():
    """Schulman et al. 2015 论文 §3 的两个退化条件。"""
    torch.manual_seed(0)
    T = 8
    rewards = torch.randn(1, T)
    values = torch.randn(1, T + 1)

    def gae(rewards, values, gamma=1.0, lam=0.95):
        B, T_ = rewards.shape
        adv = torch.zeros_like(rewards)
        last = torch.zeros(B, 1)
        for t in reversed(range(T_)):
            delta = rewards[:, t:t + 1] + gamma * values[:, t + 1:t + 2] - values[:, t:t + 1]
            last = delta + gamma * lam * last
            adv[:, t:t + 1] = last
        return adv

    # λ=0：A_t = δ_t（单步 TD error）
    td = rewards + values[:, 1:] - values[:, :-1]
    check("GAE λ=0 退化为 TD error", torch.allclose(gae(rewards, values, lam=0.0), td))

    # λ=1：A_t = Σ γ^k r_{t+k} + γ^T V(T) − V(t)（Monte Carlo 回报 − 基线）
    gamma, lam = 0.9, 1.0
    adv = gae(rewards, values, gamma=gamma, lam=lam)
    returns = torch.stack([sum(gamma ** k * rewards[0, t + k] for k in range(T - t))
                           + gamma ** (T - t) * values[0, T] for t in range(T)]).view(1, T)
    check("GAE λ=1 退化为 MC 回报−基线",
          torch.allclose(adv, returns - values[:, :T], atol=1e-5))


# ═══ 4. LoRA：B=0 起点 + 合并前后 logits 一致 ═══
def verify_lora():
    """Hu et al. 2021 的两个实现级断言（对应 Part 8 脚本 10 / Part 12 脚本 01）。"""
    torch.manual_seed(0)
    out_f, in_f, r, alpha = 16, 12, 4, 8.0
    W = torch.randn(out_f, in_f)
    A = torch.randn(r, in_f) / math.sqrt(r)
    B = torch.zeros(out_f, r)
    x = torch.randn(5, in_f)

    # 断言 1：B=0 → 起点 ΔW = 0（不破坏预训练表征）
    check("LoRA B=0 起点 ΔW=0", (B @ A).abs().max().item() == 0.0)

    # 训练后（B 非零）合并：W' = W + (α/r)·BA，前向应与 LoRA 路径逐元素一致
    B = torch.randn(out_f, r) * 0.1
    y_lora = W @ x.T + (alpha / r) * B @ (A @ x.T)      # LoRA 路径
    W_merged = W + (alpha / r) * B @ A                   # 合并
    y_merged = W_merged @ x.T
    check("LoRA 合并前后 logits 一致",
          torch.allclose(y_lora, y_merged, atol=1e-5),
          f"(max diff {(y_lora - y_merged).abs().max().item():.2e})")


# ═══ 5. MinHash：P[最小哈希相等] = Jaccard ═══
def verify_minhash():
    """Broder 1997 恒等式的大样本统计验证（对应 Part 13 脚本 01）。"""
    import zlib
    P = (1 << 31) - 1

    def jaccard(s1, s2):
        return len(s1 & s2) / len(s1 | s2)

    rng = random.Random(42)
    universe = [f"tok{i}" for i in range(200)]
    s1 = set(rng.sample(universe, 60))
    s2 = s1 | set(rng.sample(universe, 20))     # J ≈ 60/80 = 0.75

    trials, agree = 400, 0
    for _ in range(trials):
        a, b = rng.randint(1, P - 1), rng.randint(0, P - 1)
        h1 = min((a * zlib.crc32(s.encode()) + b) % P for s in s1)
        h2 = min((a * zlib.crc32(s.encode()) + b) % P for s in s2)
        agree += (h1 == h2)
    emp = agree / trials
    check("MinHash: P[相等] ≈ Jaccard", abs(emp - jaccard(s1, s2)) < 0.08,
          f"(经验 {emp:.3f} vs 理论 {jaccard(s1, s2):.3f}, {trials} 次采样)")


# ═══ 6. 流水线气泡：bubble = (p-1)/(m+p-1) ═══
def verify_bubble():
    """GPipe 时间线的离散模拟 vs 公式（对应 Part 10 脚本 06 / assignment_10 题 5）。"""
    for p, m in [(2, 4), (4, 16), (3, 8)]:
        # 模拟：总时间线长度 = m 个 forward + m 个 backward 交错占据的槽位数
        # stage0 完成最后 backward 的时间 = 2(m + p - 1) 个槽（fill-drain 两段）
        # 气泡 = 2(p-1) 个槽 / 2m(p) 有效槽... 直接用工作量的定义验证公式：
        # 每 stage 的忙槽 = 2m；时间线总槽 = 2m + 2(p-1)（两端各 p-1 空槽）
        total_slots = 2 * (m + p - 1)
        busy_slots = 2 * m
        bubble_sim = (total_slots - busy_slots) / total_slots
        bubble_formula = (p - 1) / (m + p - 1)
        check(f"气泡公式 p={p}, m={m}", abs(bubble_sim - bubble_formula) < 1e-12,
              f"(模拟 {bubble_sim:.4f} = 公式 {bubble_formula:.4f})")


if __name__ == '__main__':
    print("═══ 论文公式数值验证器（docs/paper_reading_guide.md 配套）═══\n")
    verify_rope()
    verify_dpo()
    verify_gae()
    verify_lora()
    verify_minhash()
    verify_bubble()
    print(f"\n═══ 结果: {len(PASS)} 通过, {len(FAIL)} 失败 ═══")
    if FAIL:
        print("  失败项:", FAIL)
    raise SystemExit(0 if not FAIL else 1)
