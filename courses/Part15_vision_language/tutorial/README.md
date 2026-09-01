# Part 15: 多模态理解（VLM）— 视觉如何"说"语言

> 🧭 让模型"看懂"图片并对话——这是 2026 年所有旗舰模型的标配（GLM-5.3-Flash、
> Qwen3-VL 均为原生多模态）。本章手写拼接式 VLM 的四件套（LLaVA 架构的最小闭环），
> 再横向对比三大主流方案与对齐损失。学完你能独立设计/调试一个多模态系统。
> 锚点仓库：[huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)（5.0k，"VLM 版 nanoGPT"）
> · [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v)（8.5k，中文 65M）
> · 工业对照 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)（19.9k）

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 多模态理解在 LLM 链路中的位置和价值
- ✅ **手写** 拼接式 VLM 的四件套（Patch Embedding + ViT + Projector + Token 拼接）
- ✅ **解释** CLIP/SigLIP 的数学原理、对齐损失与 batch 依赖性差异
- ✅ **画出** 三大方案（拼接式/门控/early-fusion）的注入位置图并完成选型
- ✅ **识别** 多模态理解中的常见陷阱（静态图缓存、温度 τ、batch 依赖）并设计防范策略

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写拼接式 VLM 四件套](01_handwritten_projection_vlm.md) | patch embedding → ViT → mlp2x 投影器 → token 拼接；LLaVA 两阶段训练玩具版 | `01` |
| 02 | [三大方案与对齐损失](02_alignment_losses_and_schemes.md) | 拼接式/门控/early-fusion；CLIP InfoNCE vs SigLIP；动态分辨率与 token 压缩 | `02` |

## 🔑 理论背景（导览）

- **为什么需要多模态**：纯文本模型"只能听"，跨模态对齐让它"又能看"——完整的痛点引入与类比在 [01 章](01_handwritten_projection_vlm.md)。
- **两条数学主线**：拼接式数据流（Patch → ViT → Projector → 拼接，含逐步形状账本）在 [01 章](01_handwritten_projection_vlm.md)；对齐损失推导（CLIP InfoNCE softmax vs SigLIP 逐对 sigmoid）在 [02 章](02_alignment_losses_and_schemes.md)。
- 💡 **一句话洞察**：多模态的门槛不在"造模型"，在"训对齐"——LLaVA 的全部增量只是一个 2 层 MLP 投影器（`mlp2x_gelu`）+ 两阶段训练；对比学习中 temperature 控制相似度分布的锐度，batch 越大负样本越多、对比信号越强。

## 🧰 前置知识

**必须掌握：**
- [Part 6 03 章 Transformer Block](../../Part6_transformer/tutorial/03_transformer_block.md)：本章的 ViTBlock 与玩具 LLM 都是它的同款（残差 + pre-norm + 注意力），看懂它 = 看懂半个 ViT
- [Part 8 02 章 SFT 与对话](../../Part8_post_training/tutorial/02_sft_and_chat.md)：prompt masking（labels 置 -100）——两阶段训练只监督 answer 段，全靠它

**建议掌握：**
- [Part 8 07 章评估](../../Part8_post_training/tutorial/07_evaluation.md)："规则评估 vs 学习评估"的对比思维，直接迁移到 CLIP/SigLIP 的对比损失（02 章）
- [Part 8 08 章 LoRA](../../Part8_post_training/tutorial/08_lora_and_classification.md)：资源紧张时 Stage 2 用 LoRA 代替全参微调（对照投影器"小参数撬动大模型"的账）

**可选：**
- 零视觉基础完全可学：patch embedding 就是"切块 + 线性投影"，ViT 块就是 Part 6 的 Block——本部分不要求任何 CV 前置

## 🔗 在 LLM 链路中的位置

```
【本部分: 视觉→语言 对齐（理解侧）】→ 与 Part 16 生成侧互为镜像
图像 ──ViT──▶ 视觉 token ──projector──▶ LLM token 空间 ──▶ 对话
```

## 📦 环境

脚本 01/02 均 **CPU 可跑、零新依赖**（torch 即可）；教程中引用的实测数字为 **RTX 4090**
复跑结果（已逐项核对一致），CPU 结果同量级。02 章为"方案对照 + 对齐损失"；
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
