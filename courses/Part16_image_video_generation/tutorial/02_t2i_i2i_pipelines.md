# 02 — 文生图与图生图工具链：Latent Diffusion → SD → img2img

> 🧭 从 2D 玩具到真实文生图只差两个工程跃迁：**① 搬进 VAE 潜空间**（8× 空间压缩，
> 512² 图像 → 64×64 潜变量）、**② 用 cross-attention 注入文本条件**（CLIP/T5 嵌入做
> K/V，Part 15 对齐空间的直接消费）。工具锚点：diffusers（34.4k，全谱系支持）。

## 📖 前置知识

- **01 章**：DDPM 三段数学；**Part 15 02 章**：文本对齐空间（cross-attention 消费它）

## 1. Latent Diffusion（2112.10752）：为什么搬进潜空间

```
像素空间扩散 512²×3 = 786K 维/图   →  太贵
VAE 编码到 64×64×4 = 16K 维        →  8× 空间压缩（论文：感知无损）
扩散在潜空间进行；生成后 VAE 解码回像素
```

条件注入：**cross-attention**——图像潜变量的 Q，文本嵌入的 K/V（Part 16 脚本 02
的 ① 就是它的最小版）。SD1.5 用 CLIP 文本塔；SDXL 双塔；SD3/FLUX 用 T5。

## 2. 模型谱系与 4090 跑法

| 模型 | 机制 | 4090 24GB | 备注 |
|---|---|---|---|
| DDPM 玩具（01 章） | 像素空间 | CPU ✅ | 教学数学 |
| **SD1.5** | LDM + U-Net + CLIP | fp16 ~2GB ✅ | ⚠️ 用 `stable-diffusion-v1-5/stable-diffusion-v1-5`（runwayml 原 ID 已删） |
| SDXL | 双文本塔 + 更大 U-Net | fp16 ~7GB ✅ | |
| SD3 / **FLUX.1** | **Rectified Flow + MMDiT**（文本/图像分块独立权重、双向 joint attention） | fp8 ≈12GB ✅ | FLUX 12B DiT；dev 非商用/schnell Apache |

```bash
# 工具实操（独立 venv：pip install diffusers transformers accelerate）
vllm 无关——这里就是 diffusers 两行：
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
image = pipe("a photo of an astronaut riding a horse", guidance_scale=7.5).images[0]
```

- 🔑 **Rectified Flow（SD3/FLUX）**：把"数据↔噪声"的路径拉直成直线，训练目标改为
  预测速度场 v = ε − x₀，采样路径更直 → 步数更少。MMDiT = 文本与图像 patch 各自
  独立权重 + 双向 joint attention（不再是单向 cross-attention）——条件与生成的
  对齐从"单向消费"升级为"双向融合"。

## 3. 图生图：strength 参数的数学（对 01 章闭式的直接复用）

img2img 不是"重画"，而是**从参考图的部分噪声起步**：

```
t₀ = ⌊num_inference_steps × strength⌋          # strength ∈ (0,1]
x_{t₀} = √ᾱ_{t₀}·encode(参考图) + √(1−ᾱ_{t₀})·ε   # ← 01 章 q_sample 的直接调用！
从 t₀ 反向去噪到 0（跳过前段，保留参考图内容结构）
```

- strength=1 → 从纯噪声起步（= 文生图）；strength→0 → 几乎照抄参考图。
- 💡 这就是"01 章闭式公式的第二次消费"：训练用它高效加噪，img2img 用它构造起点。

## 4. ControlNet：空间结构控制（旁路注入）

ControlNet（2302.05543，34.1k）：把冻结的 SD 编码块复制一份作为可训练副本，
条件（canny/深度/姿态/涂鸦）经副本处理后注入原网络——注入点是**零卷积**
（1×1、零初始化：训练起点不扰动原模型——LoRA B=0 的同款设计模式！）。
用途：让生成严格服从边缘图/骨架图——"prompt 控制语义，ControlNet 控制结构"。

## 学完本部分你能...

- ✅ 说清 Latent Diffusion 的两个跃迁（潜空间 + cross-attention 条件）
- ✅ 在 4090 上跑通 SD1.5 文生图（含正确模型 ID）
- ✅ 推导 img2img 的 strength→t₀ 映射，解释 strength 的两端行为
- ✅ 说出 ControlNet 零卷积的设计意图（与 LoRA B=0 归入同一模式）

**课后练习**

<details>
<summary>Q1: img2img 的 strength 太小（如 0.1）会发生什么？太大呢？</summary>
A: t₀ = ⌊steps×0.1⌋ → 几乎从干净潜变量起步，只去掉最后几步噪声——输出≈参考图
（变化微小）；strength→1 则结构信息全丢（=纯文生图）。工程上 0.4-0.7 是"保留构图、
改变内容"的常用档位。
</details>

<details>
<summary>Q2: Rectified Flow 相比 DDPM 的"直路"优势在采样上怎么体现？</summary>
A: DDPM 的概率流轨迹弯曲，需要几十步数值积分；RF 的直线路径让 ODE 求解步数大幅
减少（SD3/FLUX 常用 <30 步甚至蒸馏到 4 步），且理论分析更简单。
</details>

## 进阶与缺口（面试向：本课未深挖的高频考点）

- **采样器**：DDIM = 把 DDPM 的随机反向改成确定性 ODE 步进（可跳步：从 t 直接跳到
  t−20，eta=0 时同一 x_T 结果可复现）；DPM-Solver 系 = 高阶 ODE solver，10-15 步
  达到 DDPM 25 步质量。面试一句话："DDIM 解决确定性/跳步，DPM-Solver 解决步数效率"。
- **生成评估**：FID（真实/生成样本在 Inception 特征空间的高斯距离，越低越好，但
  对模式坍塌不敏感）；CLIP-score（图文一致性）；HPS/人评（美学）。生成侧没有单一
  指标——**组报数 + 人评抽样**是行业实践（呼应 Part 8 07 章）。
- **扩散模型微调**：SD LoRA（与 Part 8 08 章同机制，target 注入 U-Net 的
  attention 层）、Textual Inversion（学一个新 token 的 embedding）、DreamBooth
  （少量图全参微调 + class-specific prior preservation）——LLaMA-Factory/diffusers
  均可训练，4090 可跑。
- **蒸馏与步数压缩**：LCM/Turbo/DMD 把几十步蒸馏到 1-4 步——生产部署的标配方向
  （呼应 Part 14 的吞吐优化：生成侧同样有"步数 vs 质量"的 goodput 权衡）。
- **inpainting / 外扩 / 潜空间插值（slerp）**：工程日常三件套，diffusers 均有
  现成 pipeline，机制都在 01 章 q_sample 的框架内。

## 📝 课后作业

👉 [Assignment 16](../../../assignments/assignment_16/)

## 下一步

文生图会了，**参考图怎么"指挥"生成**（多图参考/人物一致性）？以及视频怎么在图像
模型上加"时间维度"？——这些全是**特征对齐**的不同形态。

👉 [03 — 特征对齐与视频生成](03_alignment_and_video.md)
