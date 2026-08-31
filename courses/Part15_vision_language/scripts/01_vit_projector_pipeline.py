#!/usr/bin/env python3
"""
Part 15 - 脚本 01: 手写"拼接式 VLM"微型管线（LLaVA 架构的最小闭环）
目标：亲手实现拼接式 VLM 的四件套——
  ① patch embedding（图像 → 视觉 token 序列）
  ② ViT 注意力块（视觉 token 之间的自交流）
  ③ 投影器 projector（mlp2x_gelu：视觉维度 → LLM 维度的"翻译器"）
  ④ token 拼接（图像向量 + 文本 embedding 拼成同一条序列喂给 LLM）
并跑 LLaVA 两阶段训练的玩具版：
  Stage 1 冻结 ViT+LLM、只训投影器（对齐）；Stage 2 全部解冻端到端微调。
对应教程：tutorial/01_handwritten_projection_vlm.md

运行（CPU <10 秒）：python 01_vit_projector_pipeline.py
全部 shape 断言内联——这是理解多模态的"形状账本"。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══ ① Patch Embedding：图像 → 视觉 token 序列 ═══
class PatchEmbed(nn.Module):
    """玩具图像 8×8×3（真实 224×224×3）；patch 2×2 → (8/2)²=16 个视觉 token。
    Conv(kernel=stride=patch) = "切 patch + 线性投影"一步完成。"""

    def __init__(self, img_size=8, patch_size=2, in_ch=3, embed_dim=24):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, img):                        # (B, 3, 8, 8)
        x = self.proj(img)                         # (B, embed_dim, 4, 4)
        return x.flatten(2).transpose(1, 2)        # (B, 16, embed_dim)


# ═══ ② ViT 注意力块（pre-norm，与 Part 6 Block 同款思想）═══
class ViTBlock(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))

    def forward(self, x, attn_mask=None):          # (B, T, dim)；attn_mask 仅 LLM 用（因果）
        h = self.ln(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask)   # 视觉段无遮罩=双向；文本段因果
        return x + a + self.mlp(self.ln(x + a))    # 残差


# ═══ ③ 投影器：LLaVA 的 mlp2x_gelu（视觉维度 → LLM 维度的翻译器）═══
class Projector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vision_dim, llm_dim), nn.GELU(),
                                 nn.Linear(llm_dim, llm_dim))

    def forward(self, vision_tokens):              # (B, 16, vision_dim)
        return self.net(vision_tokens)             # (B, 16, llm_dim) ← 已是"LLM 的话"


# ═══ ④ 玩具 LLM（Part 6 风格）——注意：forward 接受【现成 embedding】═══
class ToyLLM(nn.Module):
    def __init__(self, vocab, llm_dim=32, n_layer=2, n_head=4, ctx=32):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, llm_dim)    # 只给文本位置用
        self.blocks = nn.ModuleList([ViTBlock(llm_dim, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(llm_dim)
        self.head = nn.Linear(llm_dim, vocab)
        self.register_buffer('mask', torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x):                          # x: (B, T, llm_dim) 现成 embedding 序列
        # ⭐ 拼接式的关键：x 在输入侧由"图像位置的投影向量"与"文本位置的查表向量"
        #   拼成——LLM 本体不感知模态差异（Fuyu/early-fusion 把这个性质推到极致）
        T = x.shape[1]
        causal = self.mask[:T, :T]                 # 文本段用因果遮罩（自回归）
        for b in self.blocks:
            h = b.ln(x)
            a, _ = b.attn(h, h, h, attn_mask=causal)
            x = x + a + b.mlp(b.ln(x + a))
        return self.head(self.ln(x))


def build_sample(vis_vec, instruction_ids, answer_ids, pad_to, tok_weight):
    """token 拼接：[视觉投影向量（直接当输入 embedding）]
       + [instruction 查表] + [answer 查表]。
    labels：answer 段 = 真 id，其余 -100（Part 8 02 章 prompt masking 的多模态版）。

    Args:
        vis_vec: (n_vis, llm_dim) 已投影的视觉向量
        tok_weight: llm.tok.weight（查表用；梯度仍流向 embedding 参数）
    """
    dev = tok_weight.device
    instruction = torch.tensor(instruction_ids, dtype=torch.long, device=dev)
    answer = torch.tensor(answer_ids, dtype=torch.long, device=dev)
    seq = torch.cat([
        vis_vec,                                   # 视觉段：直接用投影向量当输入
        F.embedding(instruction, tok_weight),      # instruction 段：查表
        F.embedding(answer, tok_weight),           # answer 段：查表（labels 监督）
    ], dim=0)
    T = seq.shape[0]
    labels = torch.full((T,), -100, dtype=torch.long)
    v0 = vis_vec.shape[0]
    labels[v0 + len(instruction_ids): v0 + len(instruction_ids) + len(answer)] = answer
    pad = pad_to - T
    if pad > 0:
        seq = F.pad(seq, (0, 0, 0, pad))
        labels = F.pad(labels, (0, pad), value=-100)
    return seq, labels


def main():
    print("═══ 手写拼接式 VLM 微型管线 ═══")
    print(f"  device={DEVICE}\n")

    V, LLM_D = 12, 32
    patch_embed = PatchEmbed(embed_dim=24).to(DEVICE)
    vit = nn.Sequential(*[ViTBlock(24, 4) for _ in range(2)]).to(DEVICE)
    projector = Projector(24, LLM_D).to(DEVICE)
    llm = ToyLLM(V, LLM_D).to(DEVICE)

    img = torch.randn(2, 3, 8, 8, device=DEVICE)   # 2 张玩具"图"
    vis = patch_embed(img)                         # (2, 16, 24)
    vis = vit(vis)                                 # (2, 16, 24) 视觉自交流后
    vis_proj = projector(vis)                      # (2, 16, 32) ← 已翻译成 LLM 维度

    print(f"[shapes] image {tuple(img.shape)} → patches {tuple(vis.shape)}"
          f" → projector {tuple(vis_proj.shape)}")
    assert vis_proj.shape == (2, 16, LLM_D)

    # ── Stage 1（对齐）：冻结 ViT + LLM，只训投影器 ──
    for p in vit.parameters():
        p.requires_grad_(False)
    for p in llm.parameters():
        p.requires_grad_(False)

    # ⚠️ 数据每步重建：X 的计算图含可训练的投影器，静态缓存会导致
    #    "backward through the graph a second time"（真实踩到的经典坑）
    def build_batch():
        vis = vit(patch_embed(img))    # 切 patch → ViT 自交流（与 main 开头的 shape 演示一致）
        vis_proj = projector(vis)
        samples = [build_sample(vis_proj[i], [3, 4, 5], [6, 7], pad_to=24,
                                tok_weight=llm.tok.weight)
                   for i in range(2)]
        X = torch.stack([d[0] for d in samples]).to(DEVICE)   # (2, 24, 32) float
        Y = torch.stack([d[1] for d in samples]).to(DEVICE)   # (2, 24) long
        return X, Y

    def loss_fn(X, Y):
        logits = llm(X[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               Y[:, 1:].reshape(-1), ignore_index=-100)

    trainable = [p for p in projector.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-3)
    l0 = None
    for step in range(200):
        X, Y = build_batch()                    # ← 每步重建：图新鲜，梯度直达投影器
        loss = loss_fn(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        l0 = l0 or loss.item()
    print(f"[Stage 1] 只训投影器（{sum(p.numel() for p in trainable):,} 参数）: "
          f"loss {l0:.3f} → {loss.item():.3f}   ← LLaVA Stage 1：只做'翻译器'对齐")

    # ── Stage 2（指令微调）：全部解冻端到端 ──
    for p in vit.parameters():
        p.requires_grad_(True)
    for p in llm.parameters():
        p.requires_grad_(True)
    trainable2 = [p for p in list(vit.parameters()) + list(llm.parameters())
                  + list(projector.parameters()) if p.requires_grad]
    opt2 = torch.optim.AdamW(trainable2, lr=1e-3)
    for step in range(100):
        X, Y = build_batch()
        loss = loss_fn(X, Y)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
    print(f"[Stage 2] 端到端微调（{sum(p.numel() for p in trainable2):,} 参数）: "
          f"loss → {loss.item():.3f}   ← LLaVA Stage 2：视觉指令微调")

    print("""
═══ 与工业实现的对照 ═══
  PatchEmbed(Conv)   ← LLaVA 的 CLIP ViT patch embedding（224²/14² = 576 token）
  Projector(mlp2x)   ← LLaVA 的 mlp2x_gelu（一字不差的同名结构）
  两阶段训练         ← LLaVA 论文 Stage 1 (558K 对齐) / Stage 2 (665K 指令微调)
  进阶差异（02 章讲）：Qwen-VL 动态分辨率、InternVL 像素洗牌、Flamingo 门控、Fuyu 直入
  💡 面试："LLaVA 为什么 Stage 1 冻结两端只训投影器？"——视觉特征与 LLM 空间
     本来就"语言不通"，先把翻译器训好；两端同时乱动会让脆弱的对齐被随机梯度冲垮。""")


if __name__ == '__main__':
    main()
