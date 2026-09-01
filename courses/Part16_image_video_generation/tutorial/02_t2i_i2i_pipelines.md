# 02 — 文生图与图生图工具链：Latent Diffusion → SD → img2img

> 🧭 从 2D 玩具到真实文生图只差两个工程跃迁：**① 搬进 VAE 潜空间**（8× 空间压缩，
> 512² 图像 → 64×64 潜变量）、**② 用 cross-attention 注入文本条件**（CLIP/T5 嵌入做
> K/V，Part 15 对齐空间的直接消费）。工具锚点：diffusers（34.4k，全谱系支持）。

## 学习目标

完成本章后，你将能够：

- ✅ **解释** Latent Diffusion 的两个工程跃迁（潜空间压缩 + cross-attention 条件）及其收益
- ✅ **推导** img2img 的 strength→t₀ 映射，并用 ᾱ_{t₀} 解释 strength 两端的行为
- ✅ **运行** diffusers 完成 SD1.5 文生图与图生图（正确处理模型 ID、dtype、seed）
- ✅ **区分** prompt（语义）/ ControlNet（结构）/ strength（保留度）三条控制通道
- ✅ **识别** strength 取值极端、ControlNet 权重过高等陷阱并给出参数修正

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
# 工具实操（独立 venv，与训练环境隔离）
pip install diffusers transformers accelerate
```

```python
# vllm 在这里帮不上忙——文生图就是 diffusers 两行：
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

## 工程实践

### 常见陷阱

#### 陷阱 1：strength=1.0 时输出与参考图毫无关系（"噪声把参考图全破坏"）

**症状：** img2img 结果不含参考图的任何构图/物体，看起来就是一张普通文生图。

**原因：** t₀ = ⌊steps × 1.0⌋ = 全部步数，起点 x_{t₀} = √ᾱ_{t₀}·encode(参考图) + √(1−ᾱ_{t₀})·ε
中的 ᾱ_{t₀}→0（§3 的公式）——参考图信号几乎被噪声淹没。

**解法：** "保留构图、改变内容"用 0.4-0.7；要彻底重画不如直接用文生图管线（还省一次
参考图的 VAE 编码）。动手实践练习 1 的 √ᾱ 表可以提前算出每个 strength 保留多少信号。

#### 陷阱 2：加载 SD1.5 报 401/404（repo not found）

**症状：**
```
OSError: runwayml/stable-diffusion-v1-5 is not a local folder and is not a valid huggingface.co repository
```

**原因：** runwayml 原模型 ID 已因版权问题从 Hub 删除（社区迁移到镜像 ID）。

**解法：** 换社区镜像 `stable-diffusion-v1-5/stable-diffusion-v1-5`（§2 表格已标注）。

#### 陷阱 3：ControlNet 权重过高——输出贴死条件图、prompt 失效

**症状：** 生成严格"描"出 canny 边缘，边缘处发黑、纹理过曝（"烧焦"），改 prompt
几乎无反应。

**原因：** `controlnet_conditioning_scale` 过大 → 旁路残差压过主干输出，模型被结构
条件"锁死"（零卷积只保证训练起点不扰动，不限制推理时手动调高权重）。

**解法：** 0.5-0.8 起步逐步上调；结构约束过强时先降 ControlNet 权重，而不是加大
guidance。记住分工："prompt 控制语义，ControlNet 控制结构"（§4）。

### 最佳实践：SD 推理配置推荐（24GB 单卡起步值）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| dtype | fp16 | SD1.5 ~2GB / SDXL ~7GB，无感质量损失 |
| 采样步数 | 25-30 | 配 DPM-Solver++ / DDIM，再加步收益递减 |
| guidance_scale | 7-8 | SD 默认 7.5；w 过大的代价见 03 章 CFG |
| img2img strength | 0.4-0.7 | "保留构图、改变内容"的工程常用档（陷阱 1） |
| ControlNet scale | 0.5-0.8 起步 | 过高被条件"锁死"（陷阱 3） |
| 对比实验 | 固定 generator seed | 一切可复现对比的前提 |

## 学完本部分你能...

- ✅ 说清 Latent Diffusion 的两个跃迁（潜空间 + cross-attention 条件）
- ✅ 在 4090 上跑通 SD1.5 文生图（含正确模型 ID）
- ✅ 推导 img2img 的 strength→t₀ 映射，解释 strength 的两端行为
- ✅ 说出 ControlNet 零卷积的设计意图（与 LoRA B=0 归入同一模式）

## 🤔 概念检验

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

<details>
<summary>Q3: 为什么在 VAE 潜空间而不是像素空间做扩散？</summary>
A: 像素空间 512²×3 = 786K 维/图，U-Net 每一步的计算与显存成本爆炸；VAE 编码到
64×64×4 = 16K 维（8× 空间压缩，论文实测感知近无损）后，训练/采样成本低一个量级
（§1 的对比）。代价：细节上限受 VAE 重建误差约束——手部、文字这类高频细节的失真
多来自 VAE 而非扩散过程本身。
</details>

<details>
<summary>Q4: ControlNet 的零卷积为什么初始化为 0？</summary>
A: 旁路输出从"恒为 0（完全不扰动原模型）"的状态起步微调，避免随机初始化的副本
一开始就破坏预训练能力——与 LoRA 的 B=0 是同一设计模式：**新增模块从零映射起步，
把保护基座的责任放进初始化里**（§4）。
</details>

## 🔧 动手实践

### 练习 1：strength→t₀→信号保留比例表（CPU 纯数学，无需 GPU）

**任务：** 用 01 章的线性 β schedule 和 ᾱ 闭式，把 §3 的 strength→t₀ 映射量化成
"信号保留表"——把陷阱 1 的"噪声全破坏"变成数字。

**验收标准：**
- [ ] T=400、num_inference_steps=30，输出 strength ∈ {0.2, 0.5, 0.8, 1.0} 四行：
      strength / t₀ / √ᾱ_{t₀}
- [ ] √ᾱ 随 strength 单调递减；strength=1.0 时 ≈0.13（信号只剩 ~13%）
- [ ] 用一句话解释：为什么 strength=1.0 等价于纯文生图

**步骤提示：**
```python
import torch
betas = torch.linspace(1e-4, 0.02, 400)        # 01 章的线性 schedule
alpha_bar = torch.cumprod(1 - betas, dim=0)
for s in [0.2, 0.5, 0.8, 1.0]:
    t0 = int(30 * s)                            # §3：t₀ = ⌊steps × strength⌋
    idx = min(int(t0 / 30 * 399), 399)          # 推理步 → 400 步扩散时间线
    print(f"strength={s:.1f}  t0={t0:2d}  sqrt_abar={alpha_bar[idx].sqrt():.4f}")
```

> 参考数值（本课开发机 CPU 实算）：0.9204 / 0.6017 / 0.2737 / 0.1322。

### 练习 2（操作型，需 GPU + 独立 venv）：SD1.5 img2img strength 扫参

**任务：** 同一参考图 + 同一 prompt + 同一 seed，strength ∈ {0.2, 0.5, 0.8, 1.0}
各生成一张，做"结构保留 vs 语义改变"的定性记录表。

**验收标准：**
- [ ] 产出 4 张图 + 一张记录表：strength / t₀=⌊30·s⌋ / 构图保留度 / prompt 遵循度
- [ ] 固定 seed（`torch.Generator("cuda").manual_seed(42)`），重跑得到相同 4 张图
- [ ] 记录表的趋势与练习 1 的 √ᾱ 数值一致：strength 越大，构图保留越低
- [ ] 模型 ID 使用镜像 `stable-diffusion-v1-5/stable-diffusion-v1-5`（陷阱 2）

**步骤提示：**
```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
for s in [0.2, 0.5, 0.8, 1.0]:
    g = torch.Generator("cuda").manual_seed(42)
    img = pipe("a fantasy landscape, cinematic lighting", image=init_image,
               strength=s, num_inference_steps=30, generator=g).images[0]
    img.save(f"out_strength_{s}.png")     # 逐张回填记录表
```

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
