# Part 16: 图像生成 与 视频生成 — 扩散模型与跨模态对齐

> 🧭 理解侧（Part 15）会"看"之后，本章走生成侧：**文生图、图生图、参考图条件**到

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 图像/视频生成在 LLM 链路中的位置和价值
- ✅ **手写** DDPM 的完整数学（前向闭式 + ε 预测训练 + 采样循环）
- ✅ **解释** 扩散模型的数学原理和工程权衡
- ✅ **配置** diffusers 的推理服务并理解每个参数的含义
- ✅ **完成** 文生图、图生图、文生视频任务
- ✅ **识别** 图像/视频生成中的常见陷阱并设计防范策略

## 理论背景

### 问题引入：为什么需要图像/视频生成？

多模态理解虽然强大，但只能"理解"不能"创造"：

1. **理解限制**：只能描述图像，不能生成图像
2. **创造限制**：不能根据文本描述生成图像/视频

图像/视频生成通过**扩散模型**来弥补：

```
多模态理解:  "可以描述图像"
图像生成:    "可以根据文本生成图像"
视频生成:    "可以根据文本生成视频"
```

> 💡 **类比**：多模态理解像是会看图的人，图像生成像是会画画的人。画画的能力让创造力更丰富。

### 数学推导：DDPM 的扩散过程

**问题设定：**
- 原始图像：x_0
- 噪声图像：x_t
- 扩散过程：逐步添加噪声

**推导过程：**

```
Step 1: 前向扩散
  x_t = √(α_t) * x_0 + √(1-α_t) * ε
  其中 ε ~ N(0, I)

Step 2: 反向去噪
  x_{t-1} = (x_t - √(1-α_t) * ε_θ(x_t, t)) / √(α_t)
  其中 ε_θ 是去噪网络

Step 3: 采样
  从 x_T ~ N(0, I) 开始
  逐步去噪：x_T → x_{T-1} → ... → x_0
```

**关键洞察：**
- 前向扩散是固定的，不需要学习
- 反向去噪需要学习，是训练的重点
- 采样过程是从噪声逐步恢复图像
> **文生视频**——并沿一条主线贯穿：**跨模态特征对齐**（文本/参考图特征如何"指挥"
> 扩散网络生成）。锚点工具：[huggingface/diffusers](https://github.com/huggingface/diffusers)（34.4k）
> · 量化/小模型路径见各章。

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 DDPM](01_ddpm_from_scratch.md) | 前向闭式 / ε 预测训练 / 采样循环——2D 玩具分布上全流程 | `01` |
| 02 | [文生图与图生图](02_t2i_i2i_pipelines.md) | Latent Diffusion → SD/SDXL/SD3/FLUX 工具链 → img2img strength → ControlNet | —（diffusers 实操） |
| 03 | [特征对齐与视频生成](03_alignment_and_video.md) | IP-Adapter 解耦 KV / CFG / InstantID、PuLID → CogVideoX-2B 与 Wan2.1-1.3B | `02` |

## 🧰 前置知识

- **Part 6**：cross-attention（生成条件注入的地基）；**Part 8 06 章**：VAE/量化概念
- 无扩散基础也可：01 章 from scratch，数学全部现推

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
解耦交叉注意力 / ControlNet（03：对齐机制手写）   ← 面
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

[← 上一章：Part 15 VLM](../../Part15_vision_language/tutorial/README.md) | [返回课程总览](../../../README.md)
