# 测验 · Part 16（图像与视频生成：扩散模型与跨模态对齐）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** DDPM 训练时为什么能"随机采一个 t、一步加噪"，而不必迭代加噪 t 次？写出前向闭式并解释 ᾱ_t 的物理含义。

- **答案**：前向过程是固定的高斯马尔可夫链、没有可学习参数，任意时刻 t 的边际分布有闭式解 $q(x_t|x_0)=\mathcal{N}(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\,I)$，其中 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s\le t}\alpha_s$。所以训练时对每个样本随机采一个 t 即可 O(1) 覆盖所有噪声级别（对 t 求期望的损失），T=1000 也零额外成本；需要迭代的是有可学习参数（ε 网络）的反向链，且只在采样时发生。ᾱ_t 是"信号保留比例"：t 小信号多，t→T 信号趋零。实测：2D 双月环玩具 3000 步，denoising loss 从 1.11 降到 0.21 后稳定。
- **锚点**：教程 01_ddpm_from_scratch.md §数学推导：前向扩散的闭式解 + §2 训练：预测噪声 + §概念检验 Q1

**Q2（数字）** img2img 的 strength 如何映射成起始步 t₀？在 T=400、num_inference_steps=30 的实算中，strength ∈ {0.2, 0.5, 0.8, 1.0} 对应的信号保留 √ᾱ_{t₀} 分别是多少？由此解释 strength 两端的行为。

- **答案**：t₀ = ⌊num_inference_steps × strength⌋，起点 x_{t₀} = √ᾱ_{t₀}·encode(参考图) + √(1−ᾱ_{t₀})·ε（01 章 q_sample 的直接复用）。参考数值（本课 CPU 实算）：0.9204 / 0.6017 / 0.2737 / 0.1322，随 strength 单调递减。strength→0 几乎照抄参考图；0.4-0.7 是"保留构图、改变内容"常用档（0.5≈0.60，信号约六成）；strength=1.0 时 √ᾱ≈0.13，参考图信号几乎被噪声淹没——等价于纯文生图。
- **锚点**：教程 02_t2i_i2i_pipelines.md §3 图生图：strength 参数的数学 + §动手实践练习 1

**Q3（对比）** 从图像模型到视频模型，"最小增量"视角下三个组件各换了什么？24GB 单卡怎么选视频模型？

- **答案**：① 压缩：2D VAE（只压空间）→ 3D causal VAE（空间+时间一起压）；② 去噪骨干：2D U-Net/DiT → 同款 + temporal attention（潜变量 reshape 成 (B×T, C, H, W) 做空间注意力，再 reshape 回 (B, T, C×H×W) 做时间轴注意力——空间块之间插入"帧间交流"）；③ 条件：文本 cross-attention + 可选首/尾帧（图生视频）。选型：CogVideoX-2B（Apache-2.0，fp16 ~4GB，教学首选，1080Ti 都能跑）、Wan2.1-1.3B（8.2GB，24GB 上质量最强、~4 分钟出 5 秒 480p）、HunyuanVideo 13B（720p 需 ~60GB，量化才到 24GB）。
- **锚点**：教程 03_alignment_and_video.md §3 视频生成：图像模型 + 时间维度

**Q4（诊断）** 两个故障：① img2img 设 strength=1.0，输出与参考图毫无关系；② 生成结果严格"描"出 canny 边缘、边缘发黑过曝、改 prompt 几乎无反应。各诊断原因并给参数修正。

- **答案**：① t₀ 取到全部步数，起点公式中 ᾱ_{t₀}→0，参考图信号被噪声淹没（√ᾱ≈0.13）。修正："保留构图、改变内容"用 0.4-0.7；要彻底重画不如直接用文生图管线（还省一次 VAE 编码）。② `controlnet_conditioning_scale` 过大：零卷积只保证训练起点不扰动原模型，不限制推理时手动调高权重——旁路残差压过主干，模型被结构条件"锁死"。修正：0.5-0.8 起步逐步上调；结构约束过强时先降 ControlNet 权重而不是加大 guidance。记住分工："prompt 控制语义，ControlNet 控制结构"。
- **锚点**：教程 02_t2i_i2i_pipelines.md §工程实践 · 陷阱 1 + 陷阱 3

**Q5（对比）** IP-Adapter 为什么不把参考图 token 直接拼进文本序列，而要新开一套独立的 K/V 投影？写出解耦公式，并解释"仅 22M 参数、基座冻结"何以可能。

- **答案**：直接拼接会改变原文本 K/V 上的注意力分布，破坏基座已学好的文本对齐能力。解耦公式：$\mathrm{out}=\mathrm{attn}(Q,\ K_{txt},\ V_{txt})+s\cdot\mathrm{attn}(Q,\ K_{ref},\ V_{ref})$，参考图经 CLIP 图像编码器提特征后走一套全新独立投影。只训练 22M 新参数、原模型冻结；s=0 时输出与纯文本条件完全一致（可验证的解耦性，脚本 02 实测），强度可调、可与 ControlNet 正交组合——这就是跨模态特征对齐的生成侧形态：把参考图嵌入变成扩散 cross-attention 能消费的"类文本 token"。
- **锚点**：教程 03_alignment_and_video.md §1 解耦交叉注意力 + Assignment 16 思考题 Q4

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 理解图像/视频生成在 LLM 链路中的位置和价值 | Q5 | 跨模态特征对齐的生成侧形态（理解侧 Part 15 ⇄ 生成侧本 Part 的共通主线） |
| 手写 DDPM 的完整数学并解释工程权衡 | Q1 | 前向闭式 + 随机采 t 的权衡；训练目标/采样循环细节见闪卡 |
| 配置 diffusers 的推理服务并理解参数含义 | Q4、Q5 | strength / ControlNet scale / IP-Adapter scale 的推荐档位与过头症状（guidance 见闪卡 CFG 卡） |
| 完成文生图、图生图、文生视频任务 | Q2、Q3 | 图生图 strength→t₀ 映射与四档实算；文生视频三组件对比与 24GB 选型（文生图管线见闪卡） |
| 识别常见陷阱并设计防范策略 | Q4 | strength=1.0 与 ControlNet 过高两个陷阱的症状/原因/解法（CFG 过大、帧间闪烁见闪卡） |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| DDPM 前向闭式 | $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$，ᾱ_t=∏α_s（α=1−β）为信号保留比例 |
| DDPM 训练目标 | 从 (x_t, t) 预测加进去的噪声 ε（VLB 简化成 MSE）；等价于预测 x₀ 但方差更小、训练更稳 |
| 采样循环的关键细节 | 从 x_T~N(0,I) 反向到 0；每步减掉 ε̂ 方向再加 √β·z；**t=0 时 z=0（最后一步不加噪）** |
| 训练/采样实测数字 | denoising loss 1.11→0.21（3000 步，双月环）；采样 2000 点均值偏移 ≈0（[0.069, −0.011]）、方差比 ≈1（[1.039, 1.065]） |
| t 未归一化直接进网络的症状 | t 量级（0~T−1=399）淹没 x_t（±1）→ 时间条件没学到、loss 居高不下甚至 NaN；先除以 T 或用 sinusoidal/AdaLN 编码 |
| β schedule 选择 | 线性末端噪声过猛；cosine（Nichol & Dhariwal）让 ᾱ 更平滑衰减、训练更稳；推荐 T=1000 |
| Latent Diffusion 两个工程跃迁 | 潜空间：512²×3=786K 维 → 64×64×4=16K 维（8× 压缩，感知近无损）；条件：cross-attention（图像 Q、文本 K/V） |
| SD 谱系与 4090 跑法 | SD1.5 fp16 ~2GB（镜像 ID stable-diffusion-v1-5/stable-diffusion-v1-5，runwayml 原 ID 已删）、SDXL ~7GB、SD3/FLUX fp8 ≈12GB（Rectified Flow 预测速度场 $v=\epsilon-x_0$，采样路径更直步数更少） |
| img2img strength 数字 | t₀=⌊steps×strength⌋；√ᾱ 四档 0.9204/0.6017/0.2737/0.1322；工程常用 0.4-0.7 |
| ControlNet 零卷积 | 1×1、零初始化：训练起点完全不扰动原模型；与 LoRA 的 B=0 是同一设计模式（新增模块从零映射起步） |
| CFG 公式与权衡 | $\epsilon=\epsilon_{uncond}+w\,(\epsilon_{cond}-\epsilon_{uncond})$；SD 默认 w=7.5；训练侧以 ~10% 概率把条件置空是配套；w≥15 过饱和/失真；推荐图像 7-8、视频 5-7 |
| CFG 外推方向实测 | 随机两路 ε，w ∈ {1, 3, 7.5, 15} 与条件方向余弦 = 0.7004 / 0.9803 / 0.9974 / 0.9994——方向收敛但步长过头发散出训练分布 |
| IP-Adapter 要点 | 独立 K/V + scale 线性叠加；仅 22M 参数基座冻结；scale 0.5-1.5 起步；过大"抄死"参考图、文本指令失效 |
| 视频生成最小增量 | 3D causal VAE + temporal attention（帧间交流）；文本 cross-attention + 可选首尾帧；去噪步数 50（质量）/30（快），步数不足 → 帧间闪烁 |
| 三条对齐通道正交 | CFG / IP-Adapter / ControlNet 可组合；逐个从起点值调起，一次只动一个参数 |
