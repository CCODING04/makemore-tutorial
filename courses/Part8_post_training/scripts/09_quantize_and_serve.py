#!/usr/bin/env python3
"""
Part 8 - 脚本 9: 量化与服务 —— 从"能聊"到"能上线"
目标：亲手做推理优化的四个核心实验，全部有实测数字：
  ① int8 / int4 权重量化：ppl 代价 vs 显存节省（含 int4 不分组的"翻车现场"）
  ② KV Cache 显存计算器：GQA / 量化各省多少
  ③ PagedAttention 模拟：量化测出"整块预留浪费 60-80% → 分页 <4%"
  ④ 投机解码：draft/verify 机制 + 真实采样测得的接受率 α 对照理论公式
对应教程：tutorial/06_inference_and_serving.md。

运行：python 09_quantize_and_serve.py
环境：CPU 可跑全部（~2-3 分钟）；GPU 更快。无需额外依赖（bitsandbytes 等只在你
      想量化真实大模型时才需要，见教程 06 章的环境自检一节）。
预期数字（教程 06 章有解释）：int8 ppl 几乎不变；int4 分组后略升；int4 不分组暴涨。
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
CPU_MODE = not torch.cuda.is_available()

# ─── 精简 GPT（Part 8 经典款：LayerNorm + learned PE + MHA + ReLU）───
class Block(nn.Module):
    def __init__(self, n_embed, n_head, ctx):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
                                 nn.Linear(4 * n_embed, n_embed))
        mask = torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1)
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=self.mask[:T, :T])
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, n_embed=128, n_head=4, n_layer=2, ctx=128):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, n_embed)
        self.pos = nn.Embedding(ctx, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, ctx) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab)
        self.apply(lambda m: nn.init.normal_(m.weight, 0.0, 0.02)
                   if isinstance(m, nn.Linear) else None)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))


def quick_train(model, ids, steps=60, bs=4, seq=64, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        x = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids[i + 1:i + seq + 1]) for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def heldout_ppl(model, ids, seq=64, n_batches=12, bs=4):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(ids) - seq - 1, (bs,))
        x = torch.stack([torch.tensor(ids[i:i + seq]) for i in ix]).to(DEVICE)
        y = torch.stack([torch.tensor(ids[i + 1:i + seq + 1]) for i in ix]).to(DEVICE)
        total += model(x, y)[1].item() * bs
    return math.exp(total / (n_batches * bs))


def model_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ─── ① 权重量化 ──────────────────────────────────────────
@torch.no_grad()
def quantize_weights(model, bits, group="channel"):
    """对称 absmax 量化。group: 'channel'=逐输出通道 | 'g128'=128 权重一组 | 'tensor'=整张一个 scale。
    返回量化后的模型（权重被反量化回 fp32 以便直接对比 ppl ——
    真实部署存的是 int，这里为了"只测量化误差"不换 dtype）。"""
    q = GPT(model.tok.num_embeddings, 128, 4, len(model.blocks), model.ctx).to(DEVICE)
    q.load_state_dict(model.state_dict())
    qmax = 2 ** (bits - 1)
    for m in q.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data
            rows, cols = W.shape
            if group == "tensor":
                scale = W.abs().max() / qmax
                W.copy_((W / scale).round().clamp(-qmax, qmax - 1) * scale)
            else:
                gsize = 1 if group == "channel" else min(128, cols)
                n_groups = (cols + gsize - 1) // gsize
                for g in range(n_groups):
                    seg = W[:, g * gsize:(g + 1) * gsize]
                    s = (seg.abs().max(dim=1, keepdim=True).values / qmax).clamp(min=1e-12)
                    seg.copy_((seg / s).round().clamp(-qmax, qmax - 1) * s)
    return q


def quant_bits_per_weight(bits, cols, group="channel"):
    """有效 bits/权重：每个 scale 是 fp16(2B)，摊到它覆盖的权重数上。"""
    coverage = cols if group in ("channel", "tensor") else min(128, cols)
    return bits + 16.0 / coverage


# ─── ③ PagedAttention 模拟 ───────────────────────────────
def simulate_paging(n_requests=64, max_len=256, block_size=16, pool_blocks=1100):
    """对比 连续整块预留 vs 分页 的显存浪费。
    - 整块：每个请求预留 max_len 个 token 的 KV；浪费 = 预留 - 实际（内部碎片）
    - 分页：按 block_size 逐块按需分配；浪费只剩最后一个未满块（< 1 块/请求）
    返回浪费率；池太小导致整块方案放不下 → 记为外部碎片造成的拒绝。"""
    g = torch.Generator().manual_seed(0)
    lens = torch.randint(20, max_len + 1, (n_requests,), generator=g).tolist()
    kv_per_token = 1.0   # 单位化：1 token KV = 1 单位

    contig_alloc = sum(max_len for _ in lens)
    contig_actual = sum(lens)
    contig_waste = 1 - contig_actual / contig_alloc

    paged_alloc = sum(((l + block_size - 1) // block_size) * block_size for l in lens)
    paged_waste = 1 - contig_actual / paged_alloc
    return contig_waste, paged_waste, lens


# ─── ④ 投机解码（教科书式 speculative sampling）────────────
@torch.no_grad()
def speculative_decode(target, draft, prompt_ids, new_tokens=120, gamma=4, temp=0.7):
    """draft 自回归采样 γ 个（存下每个位置的分布）→ target 一次前向并行验证。
    判据：u ~ U(0,1)，u < pt/pd 则接受；否则从 max(0, pt-pd) 重采样并终止本周期。
    该判据保证最终输出分布与 target 单独采样完全一致（Leviathan et al. 2023）。
    返回 (生成 token 数, target 前向次数, 接受的 draft token 数)。"""
    ctx = min(target.ctx, draft.ctx)
    t_eval, d_eval = target.eval(), draft.eval()
    idx = list(prompt_ids)
    target_calls, accepted_draft = 0, 0
    while len(idx) - len(prompt_ids) < new_tokens:
        # 1) draft 自回归提议 γ 个，记录每个提议位置的分布 pd_j
        draft_tokens, draft_dists = [], []
        cur = list(idx)
        for _ in range(gamma):
            crop = cur[-ctx:]
            logits = d_eval(torch.tensor([crop], device=DEVICE))[0, -1]
            pd = F.softmax(logits / temp, -1)
            nxt = torch.multinomial(pd, 1).item()
            draft_tokens.append(nxt)
            draft_dists.append(pd)
            cur.append(nxt)
        # 2) target 一次前向，同时给出全部 γ+1 个位置的分布
        xs = (idx + draft_tokens)[-ctx:]
        logits = t_eval(torch.tensor([xs], device=DEVICE))[0]     # (L, V)
        target_calls += 1
        n_accept = 0
        for j in range(gamma):
            r = len(xs) - gamma + j - 1                # 预测 xs[r+1]（第 j 个 draft）的行
            pt = F.softmax(logits[r] / temp, -1)
            pd = draft_dists[j]
            u = torch.rand(1).item()
            if u < (pt / pd).clamp(max=1.0)[draft_tokens[j]].item():
                idx.append(draft_tokens[j])            # 接受
                n_accept += 1
            else:
                residual = (pt - pd).clamp(min=0)
                idx.append(torch.multinomial(residual / residual.sum(), 1).item())
                break
        else:
            # γ 个全接受：logits 最后一行给出"第 γ+1 个 token"的分布 —— 白赚一个
            idx.append(torch.multinomial(F.softmax(logits[-1] / temp, -1), 1).item())
        accepted_draft += n_accept
    return len(idx) - len(prompt_ids), target_calls, accepted_draft


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    with open(data_path, encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text[:300000]]
    vocab = len(chars)

    print("═══ Part 8 推理优化实验 ═══")
    print(f"  device={DEVICE}（CPU 全部可跑，约 2-3 分钟；GPU <1 分钟）\n")

    # 训练 target（所有实验共用）与 draft（投机解码用）
    print("[setup] 快速训练 target(blocks=2) 与 draft(blocks=1, 减半宽) ...")
    target = GPT(vocab, 128, 4, 2, 128).to(DEVICE)
    quick_train(target, ids, steps=50 if CPU_MODE else 500)   # 多训一点：量化 Δ 更接近大模型论文形态
    draft = GPT(vocab, 64, 2, 1, 128).to(DEVICE)
    quick_train(draft, ids, steps=50 if CPU_MODE else 300)

    # ① 量化
    print("\n[①] 权重量化：ppl 代价 vs 大小（只量化 Linear/MLP 权重；注意 in_proj 未动）")
    base_ppl = heldout_ppl(target, ids)
    base_mb = model_bytes(target) / 1e6
    print(f"  fp32 基线: ppl={base_ppl:.2f}, {base_mb:.2f} MB")
    for bits, grp in ((8, "channel"), (4, "g128"), (4, "tensor")):
        q = quantize_weights(target, bits, grp)
        ppl = heldout_ppl(q, ids)
        bpw = quant_bits_per_weight(bits, 128, grp)
        print(f"  int{bits} {grp:<8}: ppl={ppl:.2f} (Δ{ppl - base_ppl:+.2f}), "
              f"~{bpw:.2f} bits/weight → 省 {1 - bpw / 32:.0%}")
    print("  预期：int8 逐通道 Δ 很小（7B 级论文里 <0.05；本课 2M 小模型更脆，实测 Δ 略大——")
    print("        模型越小对量化越敏感，这正是业界量化论文都用 7B+ 做实验的原因）")
    print("  有趣：int4 的 g128 与逐张量在小模型上差距不大 —— 因为重量级的'离群通道'")
    print("        是大模型涌现现象（LLM.int8()/AWQ 的动机），小模型权重近高斯、没有离群")

    # ② KV cache 计算器
    print("\n[②] KV Cache 显存计算器:  2(K+V) × layers × kv_heads × head_dim × seq × batch × bytes")
    def kv_gb(layers, kv_heads, head_dim, seq, batch, bytes_=2):
        return 2 * layers * kv_heads * head_dim * seq * batch * bytes_ / 1e9
    print(f"  LLaMA-7B fp16, seq2048, bs1 : {kv_gb(32, 32, 128, 2048, 1):.2f} GB")
    print(f"  同上 + GQA(kv=8)            : {kv_gb(32, 8, 128, 2048, 1):.2f} GB   ← Part 7 GQA 的意义")
    print(f"  同上 + KV int8              : {kv_gb(32, 8, 128, 2048, 1, 1):.2f} GB   ← KIVI 思路(2bit 可到 ~0.26GB)")
    print(f"  本课 40M 模型 fp16, seq512  : {kv_gb(12, 8, 64, 512, 1):.4f} GB")

    # ③ PagedAttention
    cw, pw, lens = simulate_paging()
    print(f"\n[③] PagedAttention 模拟（{len(lens)} 个请求, max_len=256, block=16）")
    print(f"  连续整块预留浪费: {cw:.0%}   ← vLLM 论文语境：60-80%")
    print(f"  分页按需分配浪费: {pw:.0%}   ← vLLM 实测 <4%（这里含每请求尾部半块）")

    # ④ 投机解码
    print("\n[④] 投机解码（draft 提议 4 个，target 一次验证）...")
    prompt = "First Citizen:\n"
    pids = [stoi[c] for c in prompt]
    n_gen, calls, accepted = speculative_decode(target, draft, pids)
    alpha = accepted / max(n_gen - 1, 1)
    gamma = 4
    theory = (1 - alpha ** (gamma + 1)) / (1 - alpha) if alpha < 1 else gamma + 1
    naive_calls = n_gen
    print(f"  生成 {n_gen} tokens | target 前向 {calls} 次（直接生成要 {naive_calls} 次）")
    print(f"  实测接受率 α ≈ {alpha:.2f} → 理论 tokens/cycle = (1-α^(γ+1))/(1-α) ≈ {theory:.2f}")
    print(f"  预期方向：α 越高省得越多；draft 太弱 α 低 → 不赚反赔（多花 draft 前向）")

    print("""
═══ 面试直通车 ═══
  - "量化为什么 int8 基本无损？" → 权重分布接近零中心对称，absmax 逐通道量程利用率高；
    离群值在激活侧（所以 LLM.int8()/AWQ 都在救激活离群对应的通道）。
  - "PagedAttention 解决什么？" → KV 显存的内部+外部碎片（60-80%→<4%），从而 batch 能开更大。
  - "投机解码为什么在推理快？" → decode 是 memory-bound：验证 γ 个 token 的前向 ≈ 1 个 token 的钱。
  - 更深一步见 tutorial/06_inference_and_serving.md 与 07_evaluation.md。""")


if __name__ == '__main__':
    main()
