# 03 — 特征对齐与视频生成：从 IP-Adapter 到 Wan2.1

> 🧭 收官章。三件事：① 手写**解耦交叉注意力**（IP-Adapter 的核心）——参考图特征
> 注入的教科书案例（跑 [scripts/02_alignment_mechanisms.py](../scripts/02_alignment_mechanisms.py)）；
> ② CFG 的外推数学；③ 视频生成——图像模型加"时间维度"的最小增量。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** 解耦交叉注意力（IP-Adapter 核心），解释"仅 22M 参数、基座冻结"何以可能
- ✅ **写出** CFG 外推公式，解释 w 的权衡与训练侧配套（~10% 条件置空）
- ✅ **应用**"图像模型 + temporal attention"的最小增量视角拆解视频生成管线
- ✅ **选型** 24GB 单卡上的视频模型（CogVideoX-2B / Wan2.1-1.3B / HunyuanVideo 量化）
- ✅ **识别** CFG 过强、IP-Adapter scale 过大、视频帧间闪烁等陷阱并给出修正

## 📖 前置知识

- **02 章**：cross-attention 条件注入；**Part 15 02 章**：对齐损失（本章的"生成侧"呼应）

## 1. 解耦交叉注意力：参考图作为"类文本 token"

**问题**：想让生成严格遵循一张参考图（人物/风格/物体），微调整个模型太贵且会
遗忘；直接把参考 token 拼进文本序列会干扰原模型的文本能力。

**IP-Adapter（2308.06721，仅 22M 参数）的解法**：参考图经 CLIP 图像编码器提特征，
为一套**全新的独立 K/V 投影**（原模型权重冻结不动）：

```
out = attn(Q, K_txt, V_txt) + scale · attn(Q, K_ref, V_ref)
                                    ↑ 独立新增的投影（参考图专用）
```

脚本 02 的实测：scale=0 时输出与纯文本条件完全一致（原行为不变）、scale 增大
参考影响线性增强——**"解耦"= 保留基座能力 + 强度可调 + 可与 ControlNet 正交组合**。

- 🔑 这就是**跨模态特征对齐**的生成侧形态：把参考图的嵌入投影进文本条件所在的
  token 空间（"类文本 token"），让扩散网络的 cross-attention 像消费文本一样消费它。
  变体谱系：IP-Adapter Plus（细粒度）、FaceID（ArcFace 人脸嵌入）、InstantID
  （IdentityNet+人脸嵌入，单照片免调）、PuLID（对比对齐，保护可编辑性，有 FLUX 版）。

## 2. CFG：条件引导的外推数学

```
ε = ε_uncond + w · (ε_cond − ε_uncond)     # w = guidance scale（SD 默认 7.5）
```

- (ε_cond − ε_uncond) 是"条件方向"——w 放大这个方向的步长。
- 脚本 02 实测：w=7.5 时输出与条件方向的余弦 ≈0.998（外推方向正确）。
- ⚠️ w 过大 → 过饱和/失真（外推出训练分布）；这就是 Part 8 07 章"goodput 思维"
  的生成版：**不是越引导越好，是指令遵循与自然度的权衡**。
- 训练侧配套：训练时以 ~10% 概率把条件置空（uncond）→ 让模型两种模式都会——
  这是"分类器引导"进化为"无分类器引导"的关键。

## 3. 视频生成：图像模型 + 时间维度

| 组件 | 图像模型（SD 系） | 视频模型（Latte/CogVideoX/Wan） |
|---|---|---|
| 压缩 | 2D VAE（空间） | **3D Causal VAE**（空间+时间一起压） |
| 去噪骨干 | 2D U-Net / DiT | 同款 + **temporal attention**（空间块间插入时间轴注意力） |
| 条件 | 文本 cross-attention | 文本 + 可选首帧/尾帧（图生视频） |

- 🔑 **最小增量视角**：视频 = 把图像的 (B, T_frame, C, H, W) 潜变量 reshaping 成
  (B×T_frame, C, H, W) 做空间注意力，再 reshape 回 (B, T_frame, C×H×W) 做**时间轴
  注意力**——空间块之间插入一层"帧间交流"。CogVideoX 的 expert adaptive LayerNorm、
  Wan2.1 的 flow matching + 文本编码器升级（UMT5），都是在此骨架上的强化。
- **24GB 实测路径**（都有官方/社区 diffusers 支持）：
  - **CogVideoX-2B**（Apache-2.0）：fp16 ~4GB、int8 3.6GB——文生视频/图生视频的
    教学首选，连 1080Ti 都能跑
  - **Wan2.1-1.3B**（Apache-2.0）：8.2GB，4090 上 ~4 分钟出 5 秒 480p——质量最强的
    24GB 选项
  - HunyuanVideo（13B）：720p 需 ~60GB，社区量化可到 24GB——引述不实操

## 4. 跨模态对齐主线（Part 15+16 收官总图）

```
理解侧（Part 15）              生成侧（Part 16）
图像 → ViT → projector ──┐      文本 → CLIP/T5 → K/V ──┐
                         ▼                             ▼
                    LLM token 空间                扩散条件空间
                         ▲                             ▲
参考图 → CLIP → IP-Adapter KV ──────────────────────────┘（本脚本 ②）
对齐三件套：翻译器（projector/adapter KV）+ 对齐训练（Stage1/adapter 训练）
+ 可控强度（scale/CFG）——理解与生成共享同一套设计模式
```

## 工程实践

### 常见陷阱

#### 陷阱 1：CFG 的 w 过大——过饱和/失真

**症状：** 颜色过饱和、对比度过高、细节出现"塑料感"伪影；w≥15 时肉眼可见失真。

**原因：** ε 沿 (cond − uncond) 方向外推，w 越大离训练分布越远（§2 的 ⚠️）——
方向对了，但步长过头。

**解法：** 回落到 6-8（视频 5-7）。想要更强的指令遵循，优先改 prompt /
negative prompt，而不是拉满 w。

#### 陷阱 2：IP-Adapter 的 scale 过大——"抄死"参考图

**症状：** 输出几乎复刻参考图，文本指令失效，还可能出现分布外伪影。

**原因：** §1 的解耦公式是线性叠加 `attn_txt + scale·attn_ref`，scale 过大时
参考分支注意力压过文本分支。

**解法：** 0.5-1.5 起步按效果调（与 CFG 的 w 同理：引导强度与可控性的权衡）。

#### 陷阱 3：视频 temporal 一致性差——帧间闪烁/物体跳变

**症状：** 帧间物体身份、颜色跳变，背景周期性闪烁，动作不连贯。

**原因：** 帧与帧之间缺少信息交流——逐帧独立解码的 VAE、没有 temporal attention
的图像模型直连视频，或去噪步数不足导致高频时序噪声残留。

**解法：** 用带 3D causal VAE + temporal attention 的模型（CogVideoX / Wan2.1，
§3 的管线）；提高去噪步数（50 起步）、必要时降帧数/分辨率；可控性优先的场景
用图生视频锚定首帧。

### 最佳实践：对齐机制参数推荐（起点值）

| 机制 | 推荐起点 | 过头的症状 |
|---|---|---|
| CFG w | 图像 7-8 / 视频 5-7 | 过饱和（陷阱 1） |
| IP-Adapter scale | 0.5-1.5 | 抄死参考图（陷阱 2） |
| ControlNet scale | 0.5-0.8（02 章陷阱 3） | 被条件图"锁死" |
| 视频去噪步数 | 50（质量）/ 30（快） | 步数不足 → 闪烁（陷阱 3） |
| 24GB 视频选型 | CogVideoX-2B（教学）/ Wan2.1-1.3B（质量） | HunyuanVideo 需量化 |

> 三条对齐通道（CFG / IP-Adapter / ControlNet）彼此**正交**，可以组合使用——
> 逐个从起点值调起，一次只动一个参数。

## 学完本部分你能...

- ✅ 手写解耦交叉注意力，说清 IP-Adapter"22M 参数不动基座"的原理
- ✅ 写出 CFG 公式并解释 w 的权衡与训练侧配套（条件置空）
- ✅ 用"图像模型 + temporal attention"的最小增量视角理解视频生成
- ✅ 在 24GB 上选型：CogVideoX-2B / Wan2.1-1.3B / HunyuanVideo 量化

## 🤔 概念检验

<details>
<summary>Q1: IP-Adapter 的 scale 设很大（如 10）会怎样？为什么？</summary>
A: 参考分支的注意力权重压过文本分支——生成"抄死"参考图、文本指令失效，
且可能跑出分布外伪影。与 CFG 的 w 过大同理：引导强度与自然度是权衡，
实践上 0.5-1.5 起步按效果调。
</details>

<details>
<summary>Q2: 视频模型的 temporal attention 为什么通常"跳过第一帧"或用因果化设计？</summary>
A: 与文本因果遮罩同源：自回归/可控生成的场景下，未来帧不应影响已确定的帧；
另外非因果的全帧注意力训练成本高（帧数平方）。CogVideoX 用 3D 因果 VAE +
分层策略平衡质量与成本。
</details>

<details>
<summary>Q3: 训练扩散模型时为什么以 ~10% 概率把条件置空？不做会怎样？</summary>
A: CFG 采样要同时算 ε_cond 和 ε_uncond 做外推（§2 公式），模型必须"两种模式
都会"。不做置空，模型从未见过空条件 → uncond 分支输出失真，外推方向
(cond − uncond) 被污染，引导越强伪影越大。这 10% 是 CFG 的**训练侧配套**，
不是普通的数据增强。
</details>

## 🔧 动手实践

### 练习 1：CFG 外推方向的数值验证（CPU 纯数学，无需 GPU）

**任务：** 复现脚本 02 的 [3] 号实验——随机两路 ε，扫 w ∈ {1, 3, 7.5, 15}，
计算外推结果与条件方向 (cond − uncond) 的余弦，找到"方向稳定"的 w 区间，
并对照陷阱 1 理解"w 大 ≠ 更好"。

**验收标准：**
- [ ] 输出 w / 余弦 两列的表，4 行
- [ ] w=1 时余弦明显偏低（随机两路下 ≈0.7），w≥7.5 时 >0.99（外推方向收敛）
- [ ] 一句话回答：w 越大越贴条件方向，为什么实践中不把 w 拉满？

**步骤提示：**
```python
import torch, torch.nn.functional as F
torch.manual_seed(1337)
u, c = torch.randn(1, 512), torch.randn(1, 512)
d = c - u                                   # 条件方向
for w in [1, 3, 7.5, 15]:
    eps = u + w * d                         # CFG 外推（§2 公式）
    cos = F.cosine_similarity(eps, d, dim=-1)
    print(f"w={w:5.1f}  cos={cos.item():.4f}")
```

> 参考数值（seed=1337，本课开发机 CPU 实算）：0.7004 / 0.9803 / 0.9974 / 0.9994。

### 练习 2（操作型，需 GPU）：视频生成最小闭环 + 一致性观察

**任务：** 用 CogVideoX-2B（教学首选）生成两段 ~5 秒视频：默认参数一段；只改一个
参数（去噪步数 50→25 或 guidance 6→9）再一段，对照观察 temporal 一致性差异。

**验收标准：**
- [ ] 产出 2 段视频 + 记录表：参数 / 生成时长 / 帧间一致性（物体身份、背景闪烁）/ 首帧是否漂移
- [ ] 能指出管线的三个组件位置：3D causal VAE（压缩）、temporal attention（帧间交流）、
      文本 cross-attention（条件）——对照 §3 表格
- [ ] 记录"参数改动 → 一致性变化"的对应关系（如步数减半 → 闪烁增多，对应陷阱 3）

**步骤提示：**
```python
from diffusers import CogVideoXPipeline
import torch
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
video = pipe("a panda dancing in a bamboo forest", num_frames=49,
             num_inference_steps=50, guidance_scale=6.0).frames[0]
# 导出：from diffusers.utils import export_to_video; export_to_video(video, "out.mp4")
# 第二段只改 num_inference_steps=25，其余不动（一次只动一个变量）
```

## 📝 课后作业

👉 [Assignment 16](../../../assignments/assignment_16/)

## 🎓 生成线毕业（Part 1-16）——但故事没完

Part 1-16：从手写 bigram 到多模态与生成——理解侧（15）、生成侧（16）双线收拢。
到这一步，语言、理解、生成三块基石都已就位。**生成模型已经能"画"，下一步让它学会
"动手"——调用工具、多轮决策**：Part 17 用 RL（GRPO 的 agent 版）训练模型自己拆解
任务、调用工具、从环境反馈中改进。

👉 [Part 17 — Agentic RL：从单轮对话到会调工具的智能体](../../Part17_agentic_rl/tutorial/README.md)

> 面试备战（[docs/llm_interview_guide.md](../../../docs/llm_interview_guide.md)）、
> 论文训练（[docs/paper_reading_guide.md](../../../docs/paper_reading_guide.md)）
> 随时可取；想继续深挖工程侧，GPUMODE / Ultra-Scale Playbook 是下一层。

---

[← 上一章：文生图与图生图](02_t2i_i2i_pipelines.md) | [Part 16 README](README.md) | [下一站：Part 17 Agentic RL →](../../Part17_agentic_rl/tutorial/README.md)
