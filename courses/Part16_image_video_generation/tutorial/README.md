# Part 16: 图像生成 与 视频生成 — 扩散模型与跨模态对齐

> 🧭 理解侧（Part 15）会"看"之后，本部分走生成侧：**文生图、图生图、参考图条件、
> 文生视频**——并沿一条主线贯穿：**跨模态特征对齐**（文本/参考图特征如何"指挥"
> 扩散网络生成）。锚点工具：[huggingface/diffusers](https://github.com/huggingface/diffusers)（34.4k）
> · 量化/小模型路径见各章。

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 图像/视频生成在 LLM 链路中的位置和价值
- ✅ **手写** DDPM 的完整数学（前向闭式 + ε 预测训练 + 采样循环）并解释其工程权衡
- ✅ **配置** diffusers 的推理服务并理解每个参数的含义
- ✅ **完成** 文生图、图生图、文生视频任务
- ✅ **识别** 图像/视频生成中的常见陷阱并设计防范策略

## 理论背景（导览）

**为什么需要生成侧？** 多模态理解（Part 15）只能"看图说话"，不能"按话画图"——
生成侧补上"创造"的能力：扩散模型把"从噪声逐步去噪出数据"变成可训练、可控的目标。

DDPM 三段数学速览（记号约定：**α_t = 1−β_t 为单步信号保留率；ᾱ_t = ∏ᵢ αᵢ 为累积保留率**）：

| 机器 | 公式 | 一句话 |
|---|---|---|
| 前向加噪 | x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε | 边际闭式一步到位——训练高效的全部秘密 |
| 反向去噪 | x_{t−1} = 1/√α_t·(x_t − β_t/√(1−ᾱ_t)·ε̂) + √β_t·z | t=0 时 z=0（最后一步不加噪） |
| 条件注入 | 文本/参考图嵌入 → cross-attention 的 K/V | 一个模态的表征"指挥"另一个模态生成 |

> 📖 完整的问题引入（vs GAN/VAE）、逐步推导、直觉解释与 2D 玩具验证，
> 见 [01 章 · 手写 DDPM](01_ddpm_from_scratch.md)——本 README 不重复展开。

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 DDPM](01_ddpm_from_scratch.md) | 前向闭式 / ε 预测训练 / 采样循环——2D 玩具分布上全流程 | `01` |
| 02 | [文生图与图生图](02_t2i_i2i_pipelines.md) | Latent Diffusion → SD/SDXL/SD3/FLUX 工具链 → img2img strength → ControlNet | —（diffusers 实操） |
| 03 | [特征对齐与视频生成](03_alignment_and_video.md) | IP-Adapter 解耦 KV / CFG / InstantID、PuLID → CogVideoX-2B 与 Wan2.1-1.3B | `02` |

## 🧰 前置知识

**必须掌握：**
- 高斯分布的加法（两个高斯之和仍是高斯，方差相加）——01 章前向闭式的全部数学基础

**建议掌握：**
- **Part 6**：cross-attention（02/03 章生成条件注入的地基）；**Part 8 06 章**：VAE/量化概念（02 章 SD 的 fp16 显存策略与它同源）

**可选：**
- 无扩散基础也可直接开始：01 章 from scratch，三段数学全部现推

## 🔗 在 LLM 链路中的位置

```
Part 15（理解侧：图→文 对齐）⇄ 【本部分: 生成侧（文/图 → 图/视频）】
共通主线：跨模态特征对齐——把一个模态的表征"翻译"成另一个模态能消费的 token
```

## 📦 环境与版本策略

| 层 | 环境 | 说明 |
|---|---|---|
| 脚本 01/02（手写） | **CPU 可跑、零新依赖** | 数学机制，玩具规模 |
| 工具实操（diffusers） | 独立 venv：`pip install diffusers transformers accelerate`（latest） | SD1.5 fp16 ~2GB（⚠️ 用镜像 `stable-diffusion-v1-5/stable-diffusion-v1-5`，runwayml 原 ID 已删） |
| 视频生成 | CogVideoX-2B（fp16 ~4GB）/ Wan2.1-1.3B（8.2GB） | 24GB 全兼容；量化路径更低 |

## 📈 学习地图

```
手写 DDPM（01：前向闭式→训练→采样，2D 玩具）    ← 点（扩散数学）
   ↓ "搬到 VAE 潜空间 + cross-attention 条件"
文生图工具链（02：SD→SDXL→FLUX）+ img2img        ← 线
   ↓ 参考条件注入
解耦交叉注意力（03：对齐机制手写）/ ControlNet（02 §4）← 面
   ↓ 图像潜变量 → 时空潜变量
视频生成（03：CogVideoX / Wan2.1）                → 面试/工作就绪
```

## 📝 课后作业

👉 [Assignment 16](../../../assignments/assignment_16/)

## 🔗 相关资源

- 🐙 [diffusers](https://github.com/huggingface/diffusers)（docs/ 概念指南是扩散最好教程）
- 🐙 [ControlNet](https://github.com/lllyasviel/ControlNet)（34.1k）· [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) · [InstantID](https://github.com/instantX-research/InstantID)
- 🐙 [CogVideo](https://github.com/zai-org/CogVideo) · [Wan2.1](https://github.com/Wan-Video/Wan2.1) · [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- 📄 DDPM (2006.11239) · LDM (2112.10752) · SD3/Rectified Flow (2403.03206) · IP-Adapter (2308.06721) · CogVideoX (2408.06072) · Wan2.1 (2503.20314)

---

[← 上一章：Part 15 VLM](../../Part15_vision_language/tutorial/README.md) | [返回课程总览](../../../README.md) | [下一站：Part 17 Agentic RL →](../../Part17_agentic_rl/tutorial/README.md)
