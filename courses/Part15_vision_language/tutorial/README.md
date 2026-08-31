# Part 15: 多模态理解（VLM）— 视觉如何"说"语言

> 🧭 让模型"看懂"图片并对话——这是 2026 年所有旗舰模型的标配（GLM-5.3-Flash、Qwen3-VL

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 多模态理解在 LLM 链路中的位置和价值
- ✅ **手写** 拼接式 VLM 的四件套（Patch Embedding + ViT + Projector + Token 拼接）
- ✅ **解释** CLIP/SigLIP 的数学原理和对齐损失
- ✅ **配置** VLM 的推理服务并理解每个参数的含义
- ✅ **完成** 图像理解任务（如 OCR、VQA）
- ✅ **识别** 多模态理解中的常见陷阱并设计防范策略

## 理论背景

### 问题引入：为什么需要多模态理解？

预训练模型虽然强大，但只能处理文本：

1. **模态限制**：无法理解图像、视频、音频等非文本信息
2. **任务限制**：无法完成图像描述、视觉问答等多模态任务

多模态理解通过**跨模态对齐**来弥补：

```
纯文本模型:  "只能处理文本"
多模态模型:  "可以理解图像、视频、音频"
```

> 💡 **类比**：纯文本模型像是只会听的人，多模态模型像是会听又会看的人。看的能力让理解更全面。

### 数学推导：CLIP 的对比学习

**问题设定：**
- 图像集合：{I_1, I_2, ..., I_n}
- 文本集合：{T_1, T_2, ..., T_n}
- 对比学习目标：匹配的图文对相似度高，不匹配的相似度低

**推导过程：**

```
Step 1: 图像编码
  I_feat = ImageEncoder(I)  # (batch, d_model)

Step 2: 文本编码
  T_feat = TextEncoder(T)   # (batch, d_model)

Step 3: 相似度计算
  sim(I_i, T_j) = I_feat[i] @ T_feat[j].T / temperature

Step 4: 对比损失
  L = -log(exp(sim(I_i, T_i)) / Σ_j exp(sim(I_i, T_j)))
```

**关键洞察：**
- CLIP 通过对比学习对齐图像和文本的表征空间
- temperature 控制相似度的分布
- batch_size 越大，负样本越多，学习效果越好
> 均为原生多模态）。本章手写拼接式 VLM 的四件套（LLaVA 架构的最小闭环），再横向对比
> 三大主流方案与对齐损失。学完你能独立设计/调试一个多模态系统。
> 锚点仓库：[huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)（5.0k，"VLM 版 nanoGPT"）
> · [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v)（8.5k，中文 65M）
> · 工业对照 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)（19.9k）

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写拼接式 VLM 四件套](01_handwritten_projection_vlm.md) | patch embedding → ViT → mlp2x 投影器 → token 拼接；LLaVA 两阶段训练玩具版 | `01` |
| 02 | [三大方案与对齐损失](02_alignment_losses_and_schemes.md) | 拼接式/门控/early-fusion；CLIP InfoNCE vs SigLIP；动态分辨率与 token 压缩 | `02` |

## 🧰 前置知识

- **Part 6**：Transformer block（本章 ViT 块同款）；**Part 8 02 章**：prompt masking（多模态版）
- 零视觉基础也可：patch embedding 是"切块+线性"，ViT 块就是 Part 6 的 Block

## 🔗 在 LLM 链路中的位置

```
【本部分: 视觉→语言 对齐（理解侧）】→ 与 Part 16 生成侧互为镜像
图像 ──ViT──▶ 视觉 token ──projector──▶ LLM token 空间 ──▶ 对话
```

## 📦 环境

脚本 01/02 均 **CPU 可跑、零新依赖**（torch 即可）。02 章为"方案对照 + 对齐损失"；
工业模型推理实操（SmolVLM-500M <1.5GB、Qwen2-VL-2B ~5GB，4090 全兼容）为进阶自练，
权重获取见 [docs/datasets.md §5](../../../docs/datasets.md)。

## 📈 学习地图

```
手写四件套（脚本01：形状账本）      ← 点
   ↓ "我的 mlp2x_gelu 就是 LLaVA 的同名结构"
两阶段训练（Stage1 对齐/Stage2 指令）
   ↓
三大方案对照 + CLIP vs SigLIP       ← 面 → 面试/选型就绪
```

## 📝 课后作业

👉 [Assignment 15](../../../assignments/assignment_15/)

## 🔗 相关资源

- 🐙 [nanoVLM](https://github.com/huggingface/nanoVLM)（~750 行纯 PyTorch VLM，SigLIP+SmolLM2=222M）
- 🐙 [minimind-v](https://github.com/jingyaogong/minimind-v)（中文 65M，3090 数小时可训，两阶段与 LLaVA 论文一一对应）
- 🐙 [LLaVA](https://github.com/haotian-liu/LLaVA)（25k，注意仓库名；续作 LLaVA-NeXT）
- 📄 CLIP (2103.00020) · SigLIP (2303.15343) · LLaVA (2304.08485) · Qwen2-VL (2409.12191) · Flamingo (2204.14198)

---

[← 上一章：Part 14 vLLM](../../Part14_inference_vllm/tutorial/README.md) | [下一章：Part 16 图像/视频生成 →](../../Part16_image_video_generation/tutorial/README.md)
