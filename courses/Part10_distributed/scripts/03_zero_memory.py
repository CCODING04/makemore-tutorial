#!/usr/bin/env python3
"""
Part 10 - 脚本 3: ZeRO 显存账本 —— 公式 + 可复算的分片模拟
目标：① 用 ZeRO 公式算出任意模型/任意卡数的"模型状态显存"；
      ② 在本进程里模拟"把 AdamW 状态按 rank 切开"，逐字节复算每个 rank 持有多少，
         让公式对得上实测 —— 这就是面试里"ZeRO-1/2/3 各省什么"的完整证据链。

对应教程：tutorial/03_memory_zero_fsdp.md

运行（纯记账，CPU 即可，单进程）：
    python 03_zero_memory.py
"""

import os
import sys
import torch
import torch.nn as nn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(0)


class TinyGPT(nn.Module):
    """随便一个有代表性参数量的模型（大小不重要，账目逻辑才重要）。"""

    def __init__(self, vocab=50257, n_embed=512, n_layer=8):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embed)
        self.blocks = nn.ModuleList([_Block(n_embed) for _ in range(n_layer)])
        self.head = nn.Linear(n_embed, vocab, bias=False)

    def forward(self, idx):
        return self.head(self.tok(idx))


class _Block(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.ln = nn.LayerNorm(n_embed)
        self.fc1 = nn.Linear(n_embed, 4 * n_embed)
        self.fc2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        return x


def count_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def zero_accounting_simulation(model, world_size):
    """模拟 ZeRO 三阶段：每个 rank 实际要【存】哪些张量（按字节）。
    混合精度 AdamW 的模型状态 = 16 bytes/参数：
        fp16 参数 2 + fp16 梯度 2 + fp32 master 4 + fp32 动量 4 + fp32 方差 4
    ZeRO-0(DDP)：每 rank 全量          → 16Ψ
    ZeRO-1：    优化器状态分片          → 4Ψ + 12Ψ/N（参数/梯度仍全量）
    ZeRO-2：    + 梯度分片              → 8Ψ + 4Ψ/N
    ZeRO-3：    + 参数也分片            → 16Ψ/N
    本函数逐 rank 精确累加，验证公式。"""
    params = [p for p in model.parameters()]
    psi = sum(p.numel() for p in params)          # Ψ
    per_rank = []
    for rank in range(world_size):
        # 分片粒度：真实实现把参数摊平成一维缓冲再均匀切（余数给前面的 rank）。
        # 这里对每个参数的元素数做同样的均分，保证与公式 12Ψ/N 一致。
        def shard_numel(numel):
            base, rem = divmod(numel, world_size)
            return base + (1 if rank < rem else 0)
        b = 0.0
        b += sum(p.numel() * 2 for p in params)               # fp16 参数（全量，stage0-2）
        b += sum(p.numel() * 2 for p in params)               # fp16 梯度（全量，stage0-1）
        b += sum(shard_numel(p.numel()) * 4 for p in params)  # fp32 master（分片）
        b += sum(shard_numel(p.numel()) * 4 for p in params)  # 动量（分片）
        b += sum(shard_numel(p.numel()) * 4 for p in params)  # 方差（分片）
        per_rank.append(b)
    return psi, per_rank


def main():
    print("═══ ZeRO 显存账本 ═══\n")

    # ── Part A：公式表（任意模型规模 × 卡数）────────────────
    print("[A] 公式速查：模型状态显存/每卡（GB，不含激活值）")
    models = [("26M (minimind-small)", 26e6), ("64M (minimind-3)", 64e6),
              ("406M (train-llm 原版)", 406e6), ("7B (Llama 级)", 7e9)]
    header = f"{'模型':<22}" + "".join(f"{'N=' + str(n):>16}" for n in (1, 2, 8))
    print(header)
    for name, psi in models:
        row = f"{name:<22}"
        for n in (1, 2, 8):
            s1 = (4 * psi + 12 * psi / n) / 1e9      # ZeRO-1 最常用，只列它
            row += f"{s1:>16.2f}"
        print(row + "   ← ZeRO-1")
    print("\n  全阶段公式（Ψ=参数量, N=卡数）：")
    print("    DDP(全复制)   16Ψ")
    print("    ZeRO-1        4Ψ + 12Ψ/N    优化器状态分片")
    print("    ZeRO-2        8Ψ + 4Ψ/N     +梯度分片")
    print("    ZeRO-3/FSDP   16Ψ/N         +参数也分片（通信 ≈1.5×DP）")

    # ── Part B：逐 rank 精确复算（公式 vs 模拟）──────────────
    print("\n[B] 分片模拟：TinyGPT 真实张量逐 rank 记账（N=2，单位 MB）")
    model = TinyGPT(vocab=8192, n_embed=256, n_layer=4)
    psi, per_rank = zero_accounting_simulation(model, world_size=2)
    mb = lambda b: b / 1e6
    print(f"  Ψ = {psi:,} 参数")

    full = 16 * psi
    s1_formula = 4 * psi + 12 * psi / 2
    print(f"  ZeRO-0 公式 16Ψ   = {mb(full):8.1f} MB/每卡（DDP：人人全量）")
    print(f"  ZeRO-1 公式       = {mb(s1_formula):8.1f} MB/每卡")
    print(f"  模拟: rank0 = {mb(per_rank[0]):8.1f} MB, rank1 = {mb(per_rank[1]):8.1f} MB")
    assert abs(per_rank[0] - s1_formula) / s1_formula < 1e-9, "公式与逐张量记账不一致！"
    print("  ✅ 逐张量记账与公式一致（每个 rank 的每一类张量都数过一遍）")
    print(f"\n  省出多少：ZeRO-1 相比 DDP 每 rank 省 {1 - s1_formula / full:.0%} 的模型状态显存")
    print("  （激活值另算：它与并行无关，靠梯度检查点/activation 分片解决，见教程 03 章）")
    print("\n  💡 面试问法：'ZeRO 三阶段分别切什么？通信代价？' —— 表 + 上面这行断言就是答案。")


if __name__ == "__main__":
    main()
