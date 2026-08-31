# 03 — 特征对齐与视频生成：从 IP-Adapter 到 Wan2.1

> 🧭 收官章。三件事：① 手写**解耦交叉注意力**（IP-Adapter 的核心）——参考图特征
> 注入的教科书案例（跑 [scripts/02_alignment_mechanisms.py](../scripts/02_alignment_mechanisms.py)）；
> ② CFG 的外推数学；③ 视频生成——图像模型加"时间维度"的最小增量。

## 📖 前置知识

- **02 章**：cross-attention 条件注入；**Part 15 02 章**：对齐损失（本章的"生成侧"呼应）

## 1. 解耦交叉注意力：参考图作为"类文本 token"

**问题**：想让生成严格遵循一张参考图（人物/风格/物体），微调整个模型太贵且会
遗忘；直接把参考 token 拼进文本序列会干扰原模型的文本能力。

**IP-Adapter（2308.06721，仅 22M 参数）的解法**：参考图经 CLIP 图像编码器提特征，
为一套**全新的独立 K/V 投影**（原模型权重冻结不动）：

```
out = attn(Q, K_txt, V_txt) + scale · attn(Q, K_ref, V_ref)
                                    ↑ 独立新增的投影（参考图专用）
```

脚本 02 的实测：scale=0 时输出与纯文本条件完全一致（原行为不变）、scale 增大
参考影响线性增强——**"解耦"= 保留基座能力 + 强度可调 + 可与 ControlNet 正交组合**。

- 🔑 这就是**跨模态特征对齐**的生成侧形态：把参考图的嵌入投影进文本条件所在的
  token 空间（"类文本 token"），让扩散网络的 cross-attention 像消费文本一样消费它。
  变体谱系：IP-Adapter Plus（细粒度）、FaceID（ArcFace 人脸嵌入）、InstantID
  （IdentityNet+人脸嵌入，单照片免调）、PuLID（对比对齐，保护可编辑性，有 FLUX 版）。

## 2. CFG：条件引导的外推数学

```
ε = ε_uncond + w · (ε_cond − ε_uncond)     # w = guidance scale（SD 默认 7.5）
```

- (ε_cond − ε_uncond) 是"条件方向"——w 放大这个方向的步长。
- 脚本 02 实测：w=7.5 时输出与条件方向的余弦 ≈0.998（外推方向正确）。
- ⚠️ w 过大 → 过饱和/失真（外推出训练分布）；这就是 Part 8 07 章"goodput 思维"
  的生成版：**不是越引导越好，是指令遵循与自然度的权衡**。
- 训练侧配套：训练时以 ~10% 概率把条件置空（uncond）→ 让模型两种模式都会——
  这是"分类器引导"进化为"无分类器引导"的关键。

## 3. 视频生成：图像模型 + 时间维度

| 组件 | 图像模型（SD 系） | 视频模型（Latte/CogVideoX/Wan） |
|---|---|---|
| 压缩 | 2D VAE（空间） | **3D Causal VAE**（空间+时间一起压） |
| 去噪骨干 | 2D U-Net / DiT | 同款 + **temporal attention**（空间块间插入时间轴注意力） |
| 条件 | 文本 cross-attention | 文本 + 可选首帧/尾帧（图生视频） |

- 🔑 **最小增量视角**：视频 = 把图像的 (B, T_frame, C, H, W) 潜变量 reshaping 成
  (B×T_frame, C, H, W) 做空间注意力，再 reshape 回 (B, T_frame, C×H×W) 做**时间轴
  注意力**——空间块之间插入一层"帧间交流"。CogVideoX 的 expert adaptive LayerNorm、
  Wan2.1 的 flow matching + 文本编码器升级（UMT5），都是在此骨架上的强化。
- **24GB 实测路径**（都有官方/社区 diffusers 支持）：
  - **CogVideoX-2B**（Apache-2.0）：fp16 ~4GB、int8 3.6GB——文生视频/图生视频的
    教学首选，连 1080Ti 都能跑
  - **Wan2.1-1.3B**（Apache-2.0）：8.2GB，4090 上 ~4 分钟出 5 秒 480p——质量最强的
    24GB 选项
  - HunyuanVideo（13B）：720p 需 ~60GB，社区量化可到 24GB——引述不实操

## 4. 跨模态对齐主线（Part 15+16 收官总图）

```
理解侧（Part 15）              生成侧（Part 16）
图像 → ViT → projector ──┐      文本 → CLIP/T5 → K/V ──┐
                         ▼                             ▼
                    LLM token 空间                扩散条件空间
                         ▲                             ▲
参考图 → CLIP → IP-Adapter KV ──────────────────────────┘（本脚本 ②）
对齐三件套：翻译器（projector/adapter KV）+ 对齐训练（Stage1/adapter 训练）
+ 可控强度（scale/CFG）——理解与生成共享同一套设计模式
```

## 学完本部分你能...

- ✅ 手写解耦交叉注意力，说清 IP-Adapter"22M 参数不动基座"的原理
- ✅ 写出 CFG 公式并解释 w 的权衡与训练侧配套（条件置空）
- ✅ 用"图像模型 + temporal attention"的最小增量视角理解视频生成
- ✅ 在 24GB 上选型：CogVideoX-2B / Wan2.1-1.3B / HunyuanVideo 量化

**课后练习**

<details>
<summary>Q1: IP-Adapter 的 scale 设很大（如 10）会怎样？为什么？</summary>
A: 参考分支的注意力权重压过文本分支——生成"抄死"参考图、文本指令失效，
且可能跑出分布外伪影。与 CFG 的 w 过大同理：引导强度与自然度是权衡，
实践上 0.5-1.5 起步按效果调。
</details>

<details>
<summary>Q2: 视频模型的 temporal attention 为什么通常"跳过第一帧"或用因果化设计？</summary>
A: 与文本因果遮罩同源：自回归/可控生成的场景下，未来帧不应影响已确定的帧；
另外非因果的全帧注意力训练成本高（帧数平方）。CogVideoX 用 3D 因果 VAE +
分层策略平衡质量与成本。
</details>

## 📝 课后作业

👉 [Assignment 16](../../../assignments/assignment_16/)

## 🎓 课程毕业

Part 1-16：从手写 bigram 到多模态与生成——理解侧（15）、生成侧（16）双线收拢。
下一步：面试备战（docs/llm_interview_guide.md）、论文训练（docs/paper_reading_guide.md），
或深入 GPUMODE/Ultra-Scale Playbook。

---

[← 上一章：文生图与图生图](02_t2i_i2i_pipelines.md) | [Part 16 README](README.md)
