#!/usr/bin/env python3
"""
Part 16 - 脚本 02: 多模态"特征对齐"机制手写 —— 生成侧的三种注入方式
目标：用可验证的最小实现讲清三个机制（文生图/图生图/参考图 的共同地基）：
  ① Cross-Attention 条件注入：文本 K/V ↔ 图像潜变量 Q（Latent Diffusion 的条件方式）
  ② 解耦交叉注意力（IP-Adapter 的核心）：参考图特征作为"类文本 token"，
     用**独立的** K/V 投影注入（scale 控制强度）——多图参考的对齐基础
  ③ CFG（Classifier-Free Guidance）：cond/uncond 两路预测外推，放大条件影响
对应教程：tutorial/03_alignment_and_video.md
运行（CPU <5 秒）：python 02_alignment_mechanisms.py
全部 shape 断言内联；CFG 部分给出"外推方向"的数值证据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)

B, T_img, T_txt = 2, 16, 8       # 图像潜变量 16 个 patch/token；文本 8 个 token
D = 32                            # 模型宽度（真实 SD1.5 是 320，FLUX 是 3072）


class CrossAttention(nn.Module):
    """Latent Diffusion 的条件注入：Q 来自图像潜变量，K/V 来自文本嵌入。
    shape 流：Q (B,T_img,D) · K^T (B,D,T_txt) → 权重 (B,T_img,T_txt) · V → (B,T_img,D)
    —— 图像的每个位置"按相似度挑选"文本信息。"""

    def __init__(self, d, d_txt):
        super().__init__()
        self.q_proj = nn.Linear(d, d)          # Q 投影：图像空间
        self.k_proj = nn.Linear(d_txt, d)      # K/V 投影：文本空间 → 对齐到图像维度
        self.v_proj = nn.Linear(d_txt, d)

    def forward(self, img_latent, txt_emb):    # (B,T_img,D), (B,T_txt,D_txt)
        Q = self.q_proj(img_latent)
        K, V = self.k_proj(txt_emb), self.v_proj(txt_emb)
        wei = F.softmax(Q @ K.transpose(-2, -1) / D ** 0.5, dim=-1)   # (B,T_img,T_txt)
        return wei @ V, wei


# ═══ ② 解耦交叉注意力（IP-Adapter 核心，arXiv 2308.06721）═══
class DecoupledCrossAttention(nn.Module):
    """在文本 cross-attention 旁边，为参考图特征加一套【独立的】K/V 投影：
        out = attn(Q, K_txt, V_txt) + scale · attn(Q, K_ref, V_ref)
    与直接把参考 token 拼进文本序列相比，独立 KV 保留了原模型的文本能力，
    只用 scale 调节参考强度——这就是 IP-Adapter 用 22M 参数做到"图生图/多图参考"的
    全部秘密，也是多模态特征对齐的教科书案例。"""

    def __init__(self, d, d_txt, d_ref):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_txt = nn.Linear(d_txt, d)
        self.v_txt = nn.Linear(d_txt, d)
        self.k_ref = nn.Linear(d_ref, d)       # 参考图专用的独立 K/V（原模型不动）
        self.v_ref = nn.Linear(d_ref, d)

    def forward(self, img_latent, txt_emb, ref_emb, scale=1.0):
        Q = self.q_proj(img_latent)
        out_txt = F.softmax(Q @ self.k_txt(txt_emb).transpose(-2, -1) / D ** 0.5, -1) \
            @ self.v_txt(txt_emb)
        out_ref = F.softmax(Q @ self.k_ref(ref_emb).transpose(-2, -1) / D ** 0.5, -1) \
            @ self.v_ref(ref_emb)
        return out_txt + scale * out_ref       # ← 解耦注入：文本能力保留，参考强度可调


def main():
    print("═══ 生成侧特征对齐机制 ═══\n")

    img_latent = torch.randn(B, T_img, D)          # VAE 潜变量（图像的"压缩表示"）
    txt_emb = torch.randn(B, T_txt, 16)            # CLIP/T5 文本嵌入
    ref_emb = torch.randn(B, 4, 24)                # 参考图特征（IP-Adapter 用 CLIP 图像嵌入）

    # ── ① cross-attention：文本条件注入的 shape 流 ──
    ca = CrossAttention(D, 16)
    out, wei = ca(img_latent, txt_emb)
    assert out.shape == (B, T_img, D) and wei.shape == (B, T_img, T_txt)
    print(f"[1] Cross-Attention: Q{tuple(img_latent.shape)} × K/V{tuple(txt_emb.shape)}"
          f" → {tuple(out.shape)}，注意力权重 {tuple(wei.shape)}（每行和 1）")
    print(f"    每行和 = {wei.sum(-1).mean().item():.4f} ← softmax 归一化"
          f"；图像 token 按『相似度』从文本挑选信息")

    # ── ② 解耦交叉注意力：参考图注入（IP-Adapter）──
    dca = DecoupledCrossAttention(D, 16, 24)
    out_s0 = dca(img_latent, txt_emb, ref_emb, scale=0.0)
    out_s1 = dca(img_latent, txt_emb, ref_emb, scale=1.0)
    out_s3 = dca(img_latent, txt_emb, ref_emb, scale=3.0)
    base = dca.k_txt(txt_emb).abs().mean().item()  # sanity: ref 分支确实参与
    assert (out_s1 - out_s0).abs().max() > 1e-4, "scale=1 应与 scale=0 不同"
    print(f"\n[2] 解耦交叉注意力（IP-Adapter）:")
    print(f"    scale 0→1→3 的输出变化 max: {(out_s1 - out_s0).abs().max():.3f} / "
          f"{(out_s3 - out_s1).abs().max():.3f}")
    print(f"    scale=0 时输出 = 纯文本条件（原模型行为不变）；scale 越大参考图影响越强")
    print(f"    参数量对比：解耦 KV 只新增 "
          f"{sum(p.numel() for p in dca.k_ref.parameters()) + sum(p.numel() for p in dca.v_ref.parameters()):,} 参数"
          f"（IP-Adapter 全套仅 22M——不动基座模型）")

    # ── ③ CFG：条件引导外推 ──
    eps_cond, eps_uncond = torch.randn(B, T_img, D), torch.randn(B, T_img, D)
    guidance = 7.5                                  # SD 默认
    eps_cfg = eps_uncond + guidance * (eps_cond - eps_uncond)
    align = F.cosine_similarity(eps_cfg.reshape(B, -1),
                                (eps_cond - eps_uncond).reshape(B, -1), dim=-1)
    print(f"\n[3] CFG（无分类器引导）: eps = uncond + w·(cond − uncond)")
    print(f"    w={guidance} 时输出与'条件方向'的余弦 = {align.mean().item():.3f}"
          f"（w 越大越贴条件，但过大会过饱和/失真）")
    check_align = align.mean().item() > 0.99
    print(f"    {'✅ 外推方向正确' if check_align else '❌ 方向异常'}")

    print("""
═══ 特征对齐主线（Part 15 + 16 的统一视角）═══
  理解侧（Part 15）：图像特征 → projector 翻译 → LLM token 空间（拼接式）
  生成侧（本脚本）：文本/参考图特征 → K/V 投影 → 扩散网络的条件空间
  共通原理：多模态 = 把一个模态的表征"翻译"成另一个模态注意力能消费的 token；
    翻译器（projector / adapter KV）+ 翻译训练（对齐阶段/adapter 训练）= 全部秘密。
  视频生成（04 章）：把图像潜变量换成 3D VAE 的时空潜变量，
    在空间注意力块之间插入 temporal attention——图文对齐的机制原封不动沿用。
  💡 面试："IP-Adapter 为什么不微调整个模型？"→ 解耦 KV 只训 22M 参数、
     保留基座能力、强度可调（scale）、可与 ControlNet 正交组合。""")


def check_align_unused():
    pass


if __name__ == '__main__':
    main()
