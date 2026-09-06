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

## 🎯 面试直通车（话术卡：结论 → 原理 → 边界）

> 每张卡按"总分总"组织：先一句话结论压场，再两三句原理支撑，最后一句边界/代价收尾——面试答题的固定骨架。

**Q1："DDPM 为什么训练能一步加噪？"**

- **结论**：前向是一条固定的噪声规划，没有任何可学习参数，任意 t 的边际分布有闭式解——跳到任意噪声级零成本。
- **原理**：$q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I)$，其中 $\bar\alpha_t=\prod_{s<t}(1-\beta_s)$（本课脚本 T=400、线性 schedule）；训练时随机采 t 用闭式一步合成 $x_t$，"随机"是为了让每个 step 覆盖所有噪声级——损失本来就是对 t 求期望。
- **边界**：需要逐步迭代的只有反向链——$\hat\varepsilon$ 网络是可学习部分，只在采样时发生；闭式只对高斯前向成立。

**Q2："img2img 的 strength 是什么？"**

- **结论**：strength 决定起始时间步 $t_0=\lfloor \mathrm{steps}\times\mathrm{strength}\rfloor$——即把参考图加噪到第几步再开始反向去噪。
- **原理**：起点潜变量按前向闭式构造 $x_{t_0}=\sqrt{\bar\alpha_{t_0}}\,z_{ref}+\sqrt{1-\bar\alpha_{t_0}}\,\varepsilon$（题 3 的实现）；T=400 线性 schedule 实测 strength 0.2/0.5/0.8/1.0 对应 $\sqrt{\bar\alpha_{t_0}}$ = 0.9204/0.6017/0.2737/0.1322（02 章练习 1 的参考数值）——0.5 保留约六成信号，正是"保构图、改内容"的工程档位。
- **边界**：strength=1.0 信号只剩 ~13%，等价纯文生图（参考图白给）；strength 太小则改不动内容，只能算轻微修图。

**Q3："IP-Adapter 为什么不动基座？"**

- **结论**：参考图走一套独立新增的 K/V 投影，与文本注意力分支线性叠加——基座权重一个不动，只训 22M 新参数。
- **原理**：解耦交叉注意力 $\mathrm{out}=\mathrm{attn}(Q,K_{txt},V_{txt})+s\cdot\mathrm{attn}(Q,K_{ref},V_{ref})$；独立 K/V 保证不改写基座已学好的文本注意力分布，scale=0 时输出与原模型逐元素一致（可验证的解耦性，题 4 的验收），参考强度还连续可调。
- **边界**：它只做"条件注入"，改不动基座本身的能力；与 ControlNet 是正交组合而非互替（03 章 §1、脚本 02 的 [2] 号实验）。

**Q4："CFG 的 w 过大会怎样？"**

- **结论**：w 是沿条件方向的外推倍率，过大把预测推出训练分布——过饱和、失真；训练侧 ~10% 条件置空是它的必配。
- **原理**：$\varepsilon=\varepsilon_{uncond}+w\,(\varepsilon_{cond}-\varepsilon_{uncond})$；脚本 02 实测 w=7.5（SD 默认）时输出与条件方向的余弦 ≈0.998，外推方向正确，而随机两路下 w=1 余弦只有 ≈0.7；训练时不以 ~10% 概率置空条件，模型没学过无条件模式，外推方向被污染，w 越大伪影越大。
- **边界**：w 从 7.5 拉到 15 就开始过引导失真（实验题的观测点）；10% 置空是 CFG 的训练侧配套，不是数据增强。

**Q5："DDIM 和 DDPM 的区别？"**

- **结论**：DDIM 把随机反向改成确定性 ODE 步进，换来可跳步与可复现。
- **原理**：DDPM 每步注入随机噪声、必须逐步走 T 步；DDIM 在 eta=0 时同一 $x_T$ 结果可复现，且可以从 t 直接跳步（教程 02 章"采样器"小节）；同族分工的一句话版本：DDIM 解决确定性/跳步，DPM-Solver 系解决步数效率（10-15 步达到 DDPM 25 步质量）。
- **边界**：确定性采样牺牲了随机性带来的多样性；跳步太狠质量下降——进阶阅读，本课未实现。
