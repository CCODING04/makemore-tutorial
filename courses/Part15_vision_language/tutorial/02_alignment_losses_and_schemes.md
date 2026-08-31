# 02 — 三大方案与对齐损失（CLIP vs SigLIP）

> 🧭 01 章手写了"拼接式"这一主流方案。本章把视野拉开：① 三大架构方案的对照；
> ② 对齐的底层损失——CLIP 的 InfoNCE 与 SigLIP 的 sigmoid 成对损失（跑
> [scripts/02_clip_siglip_alignment.py](../scripts/02_clip_siglip_alignment.py)，CPU 10 秒，
> 两种损失都收敛且检索 100%）。

## 📖 前置知识

- **01 章**：拼接式四件套；**Part 8 07 章**：对比"规则评估 vs 学习评估"的思维
- 概率论：softmax 与 sigmoid 的关系（前者是后者的全局归一化版）

## 1. 三大方案全景（2026 开源格局）

| 方案 | 代表 | 注入机制 | 优/劣 | 现状 |
|---|---|---|---|---|
| **(a) 拼接式 projector** | LLaVA、SmolVLM、Qwen-VL、InternVL、nanoVLM、minimind-v | 视觉 token **拼进序列**，LLM 无改动 | 简单、复用全部 LLM 生态 | **绝对主流** |
| (b) 交叉注意力门控 | Flamingo (2204.14198) | Perceiver Resampler 压缩视觉 → **gated xattn** 注入冻结 LM | 可保留纯文本能力；结构复杂 | 现代开源几乎弃用，作对比 |
| (c) early-fusion/native | Fuyu-8B（patch 线性投影直入 LLM，无视觉编码器）、Chameleon（VQ 图像 token） | 图像就是"另一种 token" | 训练贵但上限高；any-to-any 的路线 | 工业兑现中（原生多模态旗舰） |

- 🔑 **一条演进主线**：拼接式（外挂翻译器）→ 原生多模态（预训练时就混模态）。
  GLM-5.3-Flash（GLM 系首个原生多模态）、DeepSeek-V4 的混合模态注意力，都是
  (c) 路线的工业兑现。理解 (a) 是理解 (c) 的前提——注入点从"输入侧"移到"预训练数据侧"。
- 📌 细节差异点（面试常问）：**Qwen-VL 系的原生动态分辨率**——不把图压到固定 336²，
  而是按原始尺寸切 patch 打包（token 数随内容变），OCR/图表类任务大幅受益；
  **InternVL 的像素洗牌**——把相邻 2×2 的视觉通道重排进特征维，视觉 token 数直接 ÷4。

## 2. 对齐损失：CLIP InfoNCE vs SigLIP（跑脚本 02）

玩具实验：4 个概念，图像/文本各有一个塔投影到共享空间，两种损失训练后
**图→文检索 top-1 全部 100%**——两种方法都能对齐，但行为不同：

```python
# InfoNCE（CLIP）：N×N 相似度矩阵按行/列 softmax，标签=对角线
logits = scale * f_img @ f_txt.T                      # scale = 可学习温度 τ
loss = 0.5 * (CE(logits, labels) + CE(logits.T, labels))   # 对称双方向

# SigLIP：逐对 sigmoid（对角 +1，非对角 -1），无全局归一化
targets = 2 * eye(N) - 1
loss = -F.logsigmoid(targets * logits).mean()
```

| | InfoNCE（CLIP, 2103.00020） | SigLIP（2303.15343） |
|---|---|---|
| 归一化 | 行/列 softmax（全 batch 参与） | 逐对独立 sigmoid |
| batch 依赖 | **强**（负例来自 batch，小 batch 信号弱） | 弱（论文实测 batch 1/4 持平） |
| 使用者 | CLIP、LLaVA 的视觉塔 | SigLIP、SmolVLM、InternVL、PaliGemma |

- 🔑 **温度 τ 的作用**：`scale = exp(log_scale)` 可学习——控制 softmax 锐度。
  τ 太小对比信号弱，太大早期训练不稳。脚本实测学习到的 τ：CLIP 路径 ≈16，SigLIP ≈8.5。
- 💡 **和生成侧的连接**（Part 16 的地基）：对比对齐学到的共享空间，正是生成模型
  cross-attention 消费的空间——文本嵌入能"指挥"图像生成，前提是两个模态在这个
  空间里已经对齐。理解侧（本章）与生成侧（Part 16）共享同一个对齐世界观。

## 3. 两阶段与对齐损失的关系

LLaVA Stage 1 用的是**生成式对齐**（图文对上的 next-token loss 只训投影器），
而不是 CLIP 式对比对齐——为什么？因为投影器的目标不是"检索"，而是"让 LLM 读得懂"。
两条对齐路线：

```
对比式（CLIP/SigLIP）：拉近配对、推远非配对 → 适合检索/打分/视觉塔预训练
生成式（LLaVA Stage1）：图文对上的 CE → 适合"让 LLM 消费视觉 token"
现代实践：视觉塔用 CLIP/SigLIP 预训练好，Stage 1 再做生成式投影对齐——两条都用
```

## 学完本部分你能...

- ✅ 画出三大方案的注入位置图，说出各自代表模型与现状
- ✅ 实现 InfoNCE 与 SigLIP，说清 batch 依赖性与温度 τ
- ✅ 解释 LLaVA Stage 1 为什么用生成式对齐而视觉塔用对比式预训练
- ✅ 估算动态分辨率下图像 token 数（作业题 4）

**课后练习**

<details>
<summary>Q1: 为什么 Flamingo 的 gated xattn 要加一个可学习的门控（tanh 前乘 0 初始化）？</summary>
A: 视觉信息对预训练 LM 是"外语"——门控初始为 0 让视觉分支的扰动从零开始，
LM 行为完全不受影响，训练中模型自己决定"开多大门"。这与 LoRA 的 B=0、
ResNet 的零初始化残差是同一个设计模式：**新分支从恒等/零出发**。
</details>

<details>
<summary>Q2: 一张 1024×768 的图，patch 14、压缩率 4（pixel shuffle），大约多少视觉 token？</summary>
A: ceil(1024/14)×ceil(768/14) = 74×55 = 4070 个 patch token，pixel shuffle ÷4 → ~1018 个。
作业题 4 会算：这就是为什么动态分辨率模型要做 token 预算控制（否则长图吃掉整个上下文）。
</details>

## 📝 课后作业

👉 [Assignment 15](../../../assignments/assignment_15/)

## 下一步

理解侧会"看"了，生成侧呢？Part 16 从 DDPM 的手写开始，走进扩散模型、文生图、
图生图与视频生成——并且继续沿"特征对齐"主线深入。

👉 [Part 16 图像/视频生成](../../Part16_image_video_generation/tutorial/README.md)
