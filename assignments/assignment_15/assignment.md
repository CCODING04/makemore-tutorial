# Assignment 15：多模态理解（VLM）

> 对应 Part 15 教程（[01 手写四件套](../../courses/Part15_vision_language/tutorial/01_handwritten_projection_vlm.md) / [02 三大方案与对齐损失](../../courses/Part15_vision_language/tutorial/02_alignment_losses_and_schemes.md)）。
> 四题纯 CPU 可完成。

## 题目（实现 `vlm_exercises.py` 后 `python test_vlm_exercises.py`）

1. **patch 数量与形状**（25 分）：给定 (H, W, patch)，算图像 token 数与 ViT 输出形状
2. **InfoNCE 对比损失**（30 分）：对称双方向 softmax CE（CLIP 核心）
3. **投影器参数量**（25 分）：mlp2x_gelu 的参数账（对照 Part 8 08 章 LoRA 的 3.4%）
4. **动态分辨率 token 估算**（20 分，🌟）：给定原图与 tile 策略，估算 Qwen 式打包 token 数

## 实验题（观测型）

- 跑脚本 01，把 Stage 1 的 lr 从 3e-3 调到 3e-2，观察"对齐被冲垮"的现象（呼应面试题）
- 跑脚本 02，把 SigLIP 的 batch 从 32 减到 8，对比 InfoNCE 在同 batch 下的表现

## 🎯 面试直通车

- "三大多模态方案？"——拼接式（主流）/ Flamingo 门控（历史）/ early-fusion（原生，下一代）
- "LLaVA 两阶段为什么这么设计？"——先训翻译器防冲垮、再端到端（1,856 参数→全参的实证）
- "CLIP vs SigLIP？"——softmax 对比 vs 逐对 sigmoid；batch 依赖性；τ 可学习
- "Qwen-VL 为什么 OCR 强？"——原生动态分辨率，token 数随内容自适应（非固定 336²）
