# Assignment 16：图像/视频生成（扩散数学与对齐机制）

> 对应 Part 16 教程（[01 手写 DDPM](../../courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.md) / [02 文生图与图生图](../../courses/Part16_image_video_generation/tutorial/02_t2i_i2i_pipelines.md) / [03 特征对齐与视频生成](../../courses/Part16_image_video_generation/tutorial/03_alignment_and_video.md)）。
> 题 1-3 必做、题 4 选做（🌟 Stretch），全部纯 CPU 可完成。

## 题目（实现 `generation_exercises.py` 后 `python test_generation_exercises.py`）

### 题 1 · DDPM 前向闭式（30 分）

实现 `q_sample(x0, alphas_cumprod, t, noise)`：`x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`（按 t 索引
逐行验证），以及 `signal_ratio(t, betas)` 返回信号保留比例 √ᾱ_t。

**验收标准：**
- [ ] 逐行满足闭式：`x_t[i] = √ᾱ_{t[i]}·x0[i] + √(1−ᾱ_{t[i]})·noise[i]`（atol 1e-6）
- [ ] 输出形状与 x0 一致；broadcast 正确（ᾱ 按 t 索引后 reshape 成 `(−1, 1)`）
- [ ] `signal_ratio(0, betas) ≈ 1`（t=0 几乎无损），且随 t 单调递减

### 题 2 · CFG 公式（25 分）

实现 `cfg(eps_uncond, eps_cond, w)`：`eps = uncond + w·(cond − uncond)`。

**验收标准：**
- [ ] `uncond=0, cond=1, w=7.5` → 输出全 7.5
- [ ] `w=0` 退化为无条件（= uncond）；`w=1` 退化为 cond
- [ ] 输出形状与输入一致（一行实现即可）

### 题 3 · img2img strength→起始步（25 分）

实现 `img2img_start_step(strength, num_inference_steps)`：`t₀ = floor(steps × strength)`。

**验收标准：**
- [ ] `strength=1.0, steps=50 → 50`（纯文生图）；`0.5 → 25`；`0.02 → 1`
- [ ] 返回 `int`（floor 语义，不是 round）
- [ ] 结果 ∈ `[0, steps]`，无越界

### 题 4 · 🌟 IP-Adapter 解耦交叉注意力（20 分，Stretch 选做）

实现 `decoupled_cross_attn(Q, K_txt, V_txt, K_ref, V_ref, scale)`：
`attn(Q,K_txt,V_txt) + scale·attn(Q,K_ref,V_ref)`，其中 `attn(X,K,V) = softmax(X@K.T/√d)@V`。

**验收标准：**
- [ ] `scale=0` 时输出与纯文本注意力逐元素一致（atol 1e-6）——解耦性的可验证定义
- [ ] `scale=1` 时 = `txt_only + ref_branch`（两分支线性可加）
- [ ] 未实现保持 `return None` → 测试优雅 SKIP ⏭️ 不判 FAIL（实现后自动生效）

## 🤔 思考题

**Q1：** DDPM 训练时为什么随机采一个 t 用闭式一步加噪，而不是迭代 t 次？又为什么必须"随机"？

<details>
<summary>💡 提示</summary>

前向 q 是固定的高斯马尔可夫链（没有可学习参数），任意 t 的边际分布有闭式解
q(x_t|x₀) = N(√ᾱ_t·x₀, (1−ᾱ_t)I)——所以"跳到任意 t"零成本。随机采 t 是为了让每个
训练 step 覆盖所有噪声级别（对 t 求期望的损失），逐个轮询 T=1000 步太慢且无必要。
需要逐步迭代的只有反向链——那才是有可学习参数（ε̂ 网络）的部分，只在采样时发生。

</details>

**Q2：** img2img 取 strength=0.5、num_inference_steps=50 时，起点潜变量里参考图的信号还剩多少？strength=1.0 呢？

<details>
<summary>💡 提示</summary>

t₀ = ⌊50×0.5⌋ = 25，位于扩散时间线中点。按线性 schedule（T=400）实算 √ᾱ_{t₀} ≈ 0.60
（见 02 章动手实践练习 1）——信号约占六成、噪声约四成，正是"保留构图、改变内容"的
工程档位。strength=1.0 时 √ᾱ ≈ 0.13，信号几乎被噪声淹没，等价于纯文生图
（02 章"陷阱 1"的定量版本）。

</details>

**Q3：** CFG 为什么要求训练时以 ~10% 概率把条件置空？不做会怎样？

<details>
<summary>💡 提示</summary>

CFG 采样公式 ε = ε_uncond + w·(ε_cond − ε_uncond) 需要同一个网络给出两路预测做外推，
所以模型必须"有条件/无条件两种模式都会"。不做置空，模型从未见过空条件 → uncond
分支输出失真，外推方向 (cond − uncond) 被污染，w 越大伪影越大。这 10% 是 CFG 的
训练侧配套，不是数据增强（03 章概念检验 Q3）。

</details>

**Q4：** IP-Adapter 为什么不把参考图 token 直接拼进文本序列，而要新开一套独立的 K/V 投影？

<details>
<summary>💡 提示</summary>

拼接会改变原文本 K/V 上的注意力分布，破坏基座已学好的文本对齐能力。独立 K/V +
线性叠加（out_txt + scale·out_ref）保证：scale=0 时行为与原模型完全一致（可验证的
解耦性）、参考强度可调、只训 22M 新参数不动基座、还能与 ControlNet 正交组合
（03 章 §1，脚本 02 的 [2] 号实验）。

</details>

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
