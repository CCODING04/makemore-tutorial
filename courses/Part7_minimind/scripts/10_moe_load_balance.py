#!/usr/bin/env python3
"""
Part 7 - 脚本 10: MoE 负载均衡实验 —— aux loss 到底救了什么
目标：亲眼看到"路由器贫富分化"：没有负载均衡损失时，路由会塌缩到少数专家；
      加上 Switch Transformer 的辅助损失 L_aux = α·N·Σ(f_i · P_i) 后负载被拉平。
      对比 α = 0 / 0.01(Switch 推荐) / 5e-4(minimind 默认) 三档。

对应教程：tutorial/05_reproduce_minimind.md「进阶实验」；概念见 03_gqa_and_ffn.md 的 MoE 一节。

运行（CPU/GPU 均可，~30s）：
    python 10_moe_load_balance.py

指标定义（本脚本打印的每一列都能手工复算）：
  f_i = 被路由到专家 i 的 token 比例（top-1 路由；均匀时 = 1/N）
  P_i = 路由器分给专家 i 的平均 softmax 概率
  gini = Σ_i Σ_j |f_i - f_j| / (2·N²·mean(f))   —— 0=绝对均匀，越大越分化
  max/mean = max(f_i) / (1/N)                   —— 直觉版"最忙专家是平均的几倍"
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_EXPERTS = 8          # 专家数
HIDDEN = 128           # token 向量维度
STEPS = 400            # 训练步数（每步都测负载）
BATCH = 256            # 每步 token 数


class Expert(nn.Module):
    """迷你专家 = 2 层 MLP（真实 minimind 里是 SwiGLU FFN）。"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """top-1 路由 MoE：y = Expert_{argmax p}(x)。可选 aux loss（Switch 公式）。"""

    def __init__(self, dim, n_experts, aux_coef=0.0):
        super().__init__()
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])
        self.n = n_experts
        self.aux_coef = aux_coef
        self.last_f = None      # 最近一次路由的 f_i（供外部观测）

    def forward(self, x, target=None):
        logits = self.router(x)                       # (B*T, N)
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)                    # top-1 路由
        onehot = F.one_hot(idx, self.n).float()
        f = onehot.mean(dim=0)                        # f_i：实际路由比例
        self.last_f = f.detach()

        # 每个专家只算分给自己的 token（真实实现用容量/分组，这里教学版逐专家 mask）
        y = torch.zeros_like(x)
        for e in range(self.n):
            mask = idx == e
            if mask.any():
                y[mask] = self.experts[e](x[mask])

        loss = F.mse_loss(y, target)
        if self.aux_coef > 0:
            # L_aux = α · N · Σ_i (f_i · P_i)：P 用可微的 softmax 概率，
            # f 用不可微的实际路由比例（onehot）。均匀路由时 = α。
            p_mean = probs.mean(dim=0)
            aux = self.n * (f * p_mean).sum()
            loss = loss + self.aux_coef * aux
        return loss


def gini(f):
    """基尼系数：0=绝对均匀。Σ|f_i-f_j| / (2·N²·mean(f))"""
    n = f.numel()
    diff = (f.view(1, -1) - f.view(-1, 1)).abs().sum()
    return (diff / (2 * n * n * f.mean())).item()


def make_cluster_data():
    """构造"有结构的"输入：token 来自 4 个簇 → 天然存在'某些专家更好'的诱惑，
    路由塌缩才有发生的土壤（真实语料同理：词频长尾）。"""
    centers = torch.randn(4, HIDDEN) * 3
    assign = torch.randint(0, 4, (BATCH,))
    x = centers[assign] + 0.3 * torch.randn(BATCH, HIDDEN)
    # 目标 = 可学习的簇专属线性映射（保证专家真的"各管一摊"是最优解）
    w = torch.randn(4, HIDDEN, HIDDEN) / (HIDDEN ** 0.5)
    target = torch.einsum('bd,bdh->bh', x - centers[assign], w[assign]) + centers[assign] * 0.1
    return x, target


def run(alpha):
    torch.manual_seed(0)                          # 三档 α 用同一初始化，公平对比
    moe = MoELayer(HIDDEN, N_EXPERTS, aux_coef=alpha).to(DEVICE)
    opt = torch.optim.AdamW(moe.parameters(), lr=3e-3)
    x, target = make_cluster_data()
    x, target = x.to(DEVICE), target.to(DEVICE)

    history = []
    for step in range(STEPS):
        loss = moe(x, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % (STEPS // 4) == 0 or step == STEPS - 1:
            f = moe.last_f
            history.append((step, f))

    f_final = history[-1][1]
    return {
        'alpha': alpha,
        'gini': gini(f_final),
        'max_over_mean': (f_final.max() * N_EXPERTS).item(),
        'task_loss': loss.item(),
        'loads': [f'{v:.2f}' for v in f_final.tolist()],
        'trace': history,
    }


def main():
    print("═══ MoE 负载均衡实验 ═══")
    print(f"  device={DEVICE}, experts={N_EXPERTS}, tokens/step={BATCH}, steps={STEPS}")
    print(f"  均匀路由基线: f_i = 1/{N_EXPERTS} = {1 / N_EXPERTS:.3f}, gini = 0\n")

    rows = []
    for alpha in (0.0, 0.01, 5e-4):
        rows.append(run(alpha))

    print(f"{'α (aux系数)':<12}{'gini ↓':<10}{'max/mean ↓':<12}{'任务 loss':<12}专家负载 f_i")
    print("─" * 88)
    for r in rows:
        print(f"{r['alpha']:<12.4g}{r['gini']:<10.3f}{r['max_over_mean']:<12.2f}"
              f"{r['task_loss']:<12.4f}{r['loads']}")

    print(f"""
═══ 观察点 ═══
  1. α=0（无均衡）：gini 和 max/mean 明显最大 —— 早期"占优"的专家拿到更多 token、
     学得更好、又吸引更多 token（rich-get-richer），其他专家逐渐"失业"。
  2. α>0：负载被拉向均匀（gini→0，max/mean→1），代价是任务 loss 略升 ——
     均衡是用"一点点任务性能"换"所有专家都活着"（否则大模型训着训着就退化成 dense）。
  3. α 的量级：Switch 推荐 0.01，minimind 用 5e-4，DeepSeek-V3 干脆改用
     "无 aux loss"的 per-expert bias 调节（arXiv 2408.15664）—— 系数是超参，不是定律。

  💡 面试问法："MoE 为什么会负载不均？aux loss 公式每一项是什么？为什么 P_i 可微而 f_i 不可微？"
     （答案就在上面的公式注释里：P 来自 softmax 可传梯度；f 是 argmax 的 onehot，不可导，
       所以公式里两个都要有——一个负责"可优化"，一个负责"反映真实"。）""")


if __name__ == '__main__':
    main()
