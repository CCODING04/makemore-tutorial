# 01 — 手写拼接式 VLM 四件套

> 🧭 拼接式（projector）方案占了 2026 年开源 VLM 的绝大多数（LLaVA/SmolVLM/Qwen-VL/
> InternVL 全是这个家族）。它的全部秘密只有一句话：**把图像变成一串"LLM 能消费的
> token"，拼进文本序列**。本章手写这四件套（跑
> [scripts/01_vit_projector_pipeline.py](../scripts/01_vit_projector_pipeline.py)，CPU 10 秒），
> 并跑 LLaVA 两阶段训练的玩具版。

## 📖 前置知识

- **Part 6**：Transformer block（本章 ViTBlock 同款）；**Part 8 02 章**：prompt masking

## 1. 四件套的"形状账本"

```
图像 (B,3,8,8)
  ① PatchEmbed(Conv k=s=2)  → (B, 16, 24)     # (8/2)²=16 个视觉 token，每个 24 维
  ② ViTBlock ×2             → (B, 16, 24)     # 视觉 token 之间先"自交流"（双向注意力）
  ③ Projector mlp2x_gelu    → (B, 16, 32)     # 翻译成 LLM 维度 ← "翻译器"
  ④ token 拼接              → (B, 16+文本, 32) # 图像向量直接当 LLM 输入 embedding
```

- 🔑 **③ 是 LLaVA 的全部增量**：视觉塔（CLIP ViT）和 LLM 都是现成的、冻结的，
  唯一训练的新东西是一个 2 层 MLP（`Linear→GELU→Linear`，LLaVA 源码里就叫
  `mlp2x_gelu`）。多模态的门槛不在"造模型"，在"训对齐"。
- ⭐ **拼接式的本质**（脚本注释里最重要的一行）：LLM 的 forward 接受**现成 embedding**
  ——图像位置的输入不是查表，而是投影向量；文本位置照常查表。**LLM 本体不感知模态
  差异**。理解了这一点，Flamingo/Fuyu 的差异只是"注入方式不同"（02 章）。

## 2. 两阶段训练（LLaVA 论文的玩具版）

**实测（脚本输出，4090）**：

```
[Stage 1] 只训投影器（1,856 参数）: loss 2.907 → 1.825  ← 冻结 ViT+LLM
[Stage 2] 端到端微调（29,308 参数）: loss → 0.034       ← 全部解冻
```

- **Stage 1（特征对齐，LLaVA 558K 数据）**：视觉特征与 LLM 空间"语言不通"，先用
  现成的图文对只训投影器——两端冻结，防止脆弱的对齐被随机梯度冲垮。
- **Stage 2（视觉指令微调，665K）**：投影器就位后端到端微调（LLaVA-1.5 是全参，
  资源紧张可用 LoRA——Part 8 08 章的技术）。
- ⚠️ 实现坑实录（调试本脚本时真实踩到的三个）：① X 的计算图含可训练投影器，
  **必须每步重建**（静态缓存会 "backward through the graph a second time"）；
  ② 图像位置**不能查 embedding 表**（输入就是投影向量）；③ 文本 id 张量的 device
  要与 tok_weight 对齐。

## 3. 与工业实现的对照

| 本脚本（玩具） | LLaVA-1.5（真实） | Qwen2.5-VL（进阶） |
|---|---|---|
| 8×8 玩具图 | CLIP ViT-L/14-336px → 576 token | 原生动态分辨率（patch 打包，token 随图像大小变） |
| mlp2x_gelu | mlp2x_gelu（同名！） | MLP projector + M-RoPE |
| 200+100 步合成数据 | 558K 对齐 + 665K 指令 | 数十亿多模态 token |

- 💡 **动态分辨率是 2024-2026 最重要的演进**：LLaVA 把图像压到固定 336²（小图糊、
  大图丢细节、OCR 类任务受伤）；Qwen-VL 系按原图切块打包，token 数随内容自适应。
  token 数估算：`ceil(H/patch)×ceil(W/patch) ÷ 压缩率`（作业题 4 练习）。

## 学完本部分你能...

- ✅ 画出拼接式 VLM 的数据流并标注每步 shape（形状账本）
- ✅ 说清 projector 的角色（模态翻译器）与两阶段训练的理由
- ✅ 指出 LLaVA→Qwen-VL 的关键演进（固定分辨率 → 原生动态分辨率）
- ✅ 绕开"静态图缓存 backward"等三个真实实现坑

**课后练习**

<details>
<summary>Q1: Stage 1 如果不冻结 ViT 和 LLM 会怎样？</summary>
A: 投影器随机初始化时输出是"噪声 token"，两端未对齐——同时更新三个模块，
LLM 会被噪声 token 冲得偏离预训练分布（灾难性遗忘），ViT 也偏离其 CLIP 对齐。
先训投影器 = 在固定的两端之间"搭桥"，桥稳了再动两端。
</details>

<details>
<summary>Q2: 16 个视觉 token 拼进序列后，LLM 的注意力为什么用因果遮罩也合理？</summary>
A: 图像 token 在序列最前，文本 token 在后——因果遮罩下文本能看到全部图像 token
（这正是"看图"），图像 token 之间互相可见（位置 0-15 互在前缀内）。若把图像放中间
则需要分段遮罩（图像段双向、文本段因果）——部分实现确实这么做。
</details>

## 📝 课后作业

👉 [Assignment 15](../../../assignments/assignment_15/)

## 下一步

会"用"图像了。但视觉特征和文本特征**在同一个空间里对齐**是更底层的能力（检索、
打分、引导生成全靠它）——以及三大方案的全景对比。

👉 [02 — 三大方案与对齐损失](02_alignment_losses_and_schemes.md)
