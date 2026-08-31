# Part 15: 多模态理解（VLM）— 视觉如何"说"语言

> 🧭 让模型"看懂"图片并对话——这是 2026 年所有旗舰模型的标配（GLM-5.3-Flash、Qwen3-VL
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

脚本 01/02 均 **CPU 可跑、零新依赖**（torch 即可）。工具实操（SmolVLM-500M <1.5GB、
Qwen2-VL-2B ~5GB，4090 全兼容）见 02 章。

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
