# Assignment 16：图像/视频生成（扩散数学与对齐机制）

> 对应 Part 16 教程（[01 手写 DDPM](../../courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.md) / [02 文生图与图生图](../../courses/Part16_image_video_generation/tutorial/02_t2i_i2i_pipelines.md) / [03 特征对齐与视频生成](../../courses/Part16_image_video_generation/tutorial/03_alignment_and_video.md)）。
> 四题纯 CPU 可完成。

## 题目（实现 `generation_exercises.py` 后 `python test_generation_exercises.py`）

1. **DDPM 前向闭式**（30 分）：`x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`（按 t 索引逐行验证）+
   信号保留比例 √ᾱ_t 的单调性检查（t=0 时 ≈1）
2. **CFG 公式**（25 分）：`eps = uncond + w·(cond − uncond)`；w=0 退化为无条件
3. **img2img strength**（25 分）：`t₀ = floor(steps × strength)`；两端行为（1=纯文生图，
   →0 几乎照抄参考图）
4. **🌟 IP-Adapter 解耦交叉注意力**（20 分）：`attn(Q,K_txt,V_txt) + scale·attn(Q,K_ref,V_ref)`；
   scale=0 时必须等于纯文本注意力（解耦性的可验证定义）

## 实验题（观测型）

- 跑脚本 01：把 T 从 400 降到 100，观察采样分布匹配度变化（噪声 schedule 与步数的权衡）
- 跑脚本 02：把 CFG 的 w 从 7.5 调到 15，观察与条件方向余弦的变化（过引导的失真起点）
- （进阶，需 GPU）按 02 章 diffusers 两行代码跑 SD1.5 文生图，换 5 个 prompt 记录
  guidance_scale 7.5 vs 15 的视觉差异

## 🎯 面试直通车

- "DDPM 为什么训练能一步加噪？"——前向是固定高斯链，ᾱ 闭式（01 章 Q1）
- "img2img 的 strength 是什么？"——t₀ = ⌊steps×strength⌋，从 q(x_{t₀}|x₀) 起步（题 3）
- "IP-Adapter 为什么不动基座？"——解耦 KV 只训 22M，scale 可调，可与 ControlNet 正交组合
- "CFG 的 w 过大会怎样？"——过饱和/失真（外推出训练分布）；训练侧 10% 条件置空是配套
- "DDIM 和 DDPM 的区别？"——确定性采样/可跳步（教程 02 章"采样器"小节，进阶阅读）
