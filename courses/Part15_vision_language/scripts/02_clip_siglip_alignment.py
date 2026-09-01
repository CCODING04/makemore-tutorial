#!/usr/bin/env python3
"""
Part 15 - 脚本 02: CLIP 对比学习 —— InfoNCE vs SigLIP（多模态对齐的"理解侧"损失）
目标：在玩具数据上实现两种主流图文对齐损失并对比训练行为：
  ① InfoNCE（CLIP）：相似度矩阵按行/列做 softmax 交叉熵（"从 N 个候选里挑出配对"）
  ② SigLIP（sigmoid 成对）：每个 (i,j) 对独立二分类，不做全局 softmax
并验证温度 τ 的作用与对齐几何（配对样本在共享空间中互相靠近）。
对应教程：tutorial/02_alignment_losses_and_schemes.md

运行（CPU <10 秒）：python 02_clip_siglip_alignment.py
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)


# ─── 玩具任务：4 个"概念"，图像/文本各用一层投影映射到共享空间 ───
class Towers(nn.Module):
    def __init__(self, n_concepts=4, dim=8):
        super().__init__()
        self.img = nn.Linear(n_concepts, dim)      # 玩具"图像编码器"（真实=ViT）
        self.txt = nn.Linear(n_concepts, dim)      # 玩具"文本编码器"（真实=Transformer）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))  # 可学习温度（CLIP 同款）

    def forward(self, img_onehot, txt_onehot):
        f_img = F.normalize(self.img(img_onehot), dim=-1)   # 单位球面上（余弦相似度）
        f_txt = F.normalize(self.txt(txt_onehot), dim=-1)
        return f_img, f_txt, self.logit_scale.exp().clamp(max=100.0)


def infonce_loss(f_img, f_txt, scale):
    """CLIP InfoNCE：对称双方向。logits[i,j] = scale * ⟨f_img[i], f_txt[j]⟩，
    标签 = 对角线（第 i 张图配第 i 条文本）。"""
    logits = scale * f_img @ f_txt.T                       # (N, N)
    labels = torch.arange(len(f_img), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def siglip_loss(f_img, f_txt, scale):
    """SigLIP：逐对 sigmoid（+1 配对 / -1 非配对），无全局 softmax。
    论文核心主张：去掉对整个 batch 的依赖，更大 batch/更稳。"""
    logits = scale * f_img @ f_txt.T                       # (N, N)
    n = logits.shape[0]
    targets = 2 * torch.eye(n, device=logits.device) - 1   # 对角 +1，其余 -1
    return -F.logsigmoid(targets * logits).mean()


def main():
    print("═══ CLIP InfoNCE vs SigLIP 对齐损失对比 ═══\n")

    # 玩具数据：每概念 8 个样本（图像/文本是同一概念的两个"视角"的噪声版）
    n_concepts, per = 4, 8
    N = n_concepts * per
    img_x = torch.zeros(N, n_concepts)
    txt_x = torch.zeros(N, n_concepts)
    for i in range(N):
        c = i // per
        img_x[i, c] = 1.0
        txt_x[i, c] = 1.0
    img_x += 0.3 * torch.randn(N, n_concepts) * torch.eye(n_concepts)[torch.arange(N) % n_concepts]
    txt_x += 0.3 * torch.randn(N, n_concepts) * torch.eye(n_concepts)[torch.arange(N) % n_concepts]

    for name, loss_fn in (("CLIP InfoNCE", infonce_loss), ("SigLIP sigmoid", siglip_loss)):
        torch.manual_seed(1)
        model = Towers(n_concepts)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        for step in range(300):
            f_img, f_txt, scale = model(img_x, txt_x)
            loss = loss_fn(f_img, f_txt, scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            f_img, f_txt, scale = model(img_x, txt_x)
            sim = f_img @ f_txt.T
            # 对齐质量：每个样本最相似的文本是否是同概念
            correct = 0
            for i in range(N):
                j = sim[i].argmax().item()
                correct += (j // per) == (i // per)
            # 检索准确率（i2t top-1）
        print(f"[{name}] loss {losses[0]:.3f} → {sum(losses[-20:]) / 20:.3f} | "
              f"学习到的温度 τ = {scale.item():.2f} | 图→文检索 top-1 acc = {correct / N:.2%}")

    print("""
═══ 两种损失的本质差异 ═══
  InfoNCE（CLIP）：每一行做 softmax——"N 个候选里谁是对的"，隐含全 batch 归一化，
    要求大 batch（batch 越小，负例越少，对比信号越弱）。
  SigLIP：每一对独立 sigmoid——把 N×N 的 softmax 分解成 N² 个二分类，
    论文实测 batch 1/4 即可持平 CLIP 且更稳；SmolVLM/InternVL/PaliGemma 都改用它。
  共同点：两个塔（图像/文本）在**共享空间**里对齐——这正是 Part 16 生成模型里
    "文本嵌入控制图像生成"的对齐地基（cross-attention 消费的就是这个空间）。
  💡 面试："CLIP 和 SigLIP 的区别？"→ softmax 对比 vs 逐对 sigmoid；
     batch 依赖性；温度 τ 的作用（调锐度，可学习）。""")


if __name__ == '__main__':
    main()
