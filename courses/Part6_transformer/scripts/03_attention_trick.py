#!/usr/bin/env python3
"""
Part 6 - 脚本 3: Attention 的数学技巧
目标：理解 Transformer 中自注意力高效实现背后的数学技巧——用矩阵乘法做
"过去 token 的加权聚合"（带下三角遮罩）。独立玩具脚本，不训练任何网络。

三种等价版本（用 torch.allclose 验证）：
  v1: for 循环 bag-of-words 平均（最弱聚合，但直观）
  v2: torch.tril 下三角矩阵 × 矩阵乘法（加权求和）
  v3: masked_fill(-inf) + softmax（亲和力 + 归一化）

关键洞察：
  - 下三角遮罩保证"未来不能看向过去"（自回归）
  - wei 矩阵 = token 间亲和力（affinity），softmax 把它归一化成行和为 1
  - 亲和力将来是"数据依赖"的（Self-Attention 的预告）
"""

import sys
import torch
import torch.nn.functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使运行更快更稳定
torch.set_num_threads(1)

torch.manual_seed(1337)


def main():
    # ─── 玩具数据：B 个序列、T 个 token、C 维信息 ─────────────────
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    print("═══ 玩具数据 ═══")
    print(f"  x shape: {x.shape} (B={B} 个序列, T={T} 个 token, C={C} 维信息)")
    print(f"  目标：每个 token 聚合它自己及之前所有 token 的信息")

    # ─── v1: for 循环 bag-of-words 平均 ────────────────────────────
    # 最弱的通信：把过去所有 token 的信息简单平均成一条特征向量
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t + 1]        # (t+1, C)：当前及之前的 token
            xbow[b, t] = xprev.mean(0)  # 对时间维求平均 → bag of words
    print("\n═══ v1: for 循环平均 (bag of words) ═══")
    print(f"  xbow shape: {xbow.shape}")
    print(f"  第 0 个序列: token 5 聚合的是前 5 个 token 的平均")

    # ─── v2: 用矩阵乘法做加权聚合 ──────────────────────────────────
    # 下三角全 1 矩阵：第 t 行只对 ≤t 的 token 求和（未来不参与）
    tril = torch.tril(torch.ones(T, T))
    print("\n═══ v2: 矩阵乘法 (torch.tril) ═══")
    print("  下三角权重矩阵（1=聚合该 token，0=忽略）：")
    print(tril)

    wei = tril
    wei = wei / wei.sum(1, keepdim=True)  # 每行归一化成和为 1 → 变成"平均"
    print("\n  归一化后的权重（每行和为 1）:")
    print(wei)
    xbow2 = wei @ x                        # (T,T) @ (B,T,C) → 批矩阵乘法 (B,T,C)
    print(f"  xbow2 shape: {xbow2.shape}")
    # 放宽容差：mean()（树状求和）与矩阵乘法（点积求和）存在 ~1e-8 的浮点舍入差
    print(f"  v1 与 v2 等价 (torch.allclose): "
          f"{torch.allclose(xbow, xbow2, atol=1e-5, rtol=1e-5)}")

    # ─── v3: softmax 版本（亲和力 + 遮罩）──────────────────────────
    # 把"权重"看成亲和力 wei：初始为 0 → 通过 softmax 归一化
    wei = torch.zeros((T, T))                  # 亲和力矩阵，初始全 0（无差异）
    wei = wei.masked_fill(tril == 0, float('-inf'))  # 未来禁连 → -inf
    wei = F.softmax(wei, dim=-1)               # 每行 softmax → 行和为 1
    xbow3 = wei @ x
    print("\n═══ v3: softmax 版本 (masked_fill + softmax) ═══")
    print("  softmax 归一化后的权重（与 v2 相同）:")
    print(wei)
    print(f"  xbow3 shape: {xbow3.shape}")
    print(f"  v2 与 v3 等价 (torch.allclose): "
          f"{torch.allclose(xbow2, xbow3, atol=1e-5, rtol=1e-5)}")
    print(f"  v1 与 v3 等价 (torch.allclose): "
          f"{torch.allclose(xbow, xbow3, atol=1e-5, rtol=1e-5)}")

    print("""
═══ 总结 ═══

三个版本数学等价，但 v3（softmax）最灵活、最值得记住：
  - 下三角遮罩 → 自回归：未来不能看向过去
  - wei = 亲和力（affinity）矩阵，softmax 把每一行归一化成概率
  - wei @ x → 按亲和力对过去的信息做加权聚合

预告：真实 Self-Attention 里，wei 不再全是 0！
  它由每个 token 发出的 query 与 key 的内积算出 → 数据依赖。
  token 会主动寻找自己感兴趣的其它 token，按兴趣程度聚合信息。
""")
    print(f"聚合结果第 0 序列:\n  {xbow3[0].tolist()}")


if __name__ == '__main__':
    main()
