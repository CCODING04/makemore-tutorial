# Part 15/16 设计方案 — 多模态理解 与 图像/视频生成

> **调研基础**（2026-08-31 GitHub API + README 实查）：nanoVLM 5.0k、minimind-v 8.5k、
> SmolVLM（smollm 3.9k）、Qwen3-VL 19.9k、InternVL 10.1k、LLaVA 25.0k（haotian-liu）；
> diffusers 34.4k、ControlNet 34.1k、FLUX 25.9k、CogVideo 13.0k、Wan2.1 16.9k、
> HunyuanVideo 12.5k、IP-Adapter 6.7k、InstantID 12.0k。
> **结论**：两章都采用"手写核心机制 → 工业工具对照"双轨（与 Part 9-14 一致）。

## Part 15：多模态理解（VLM）— 锚点与章设

**锚点**：拼接式 projector 架构（绝对主流）。教学组合：
- **手写**：patch embedding → ViT 块 → 投影器（mlp2x_gelu）→ token 拼接（玩具 CPU 可跑）
- **对照仓库**：huggingface/nanoVLM（"Karpathy 式 VLM"，~750 行纯 PyTorch）
  + jingyaogong/minimind-v（中文、65M、4090 数小时可训、两阶段对应 LLaVA 论文）
- **工业对照**：Qwen3-VL（动态分辨率 patch packing + M-RoPE）、InternVL（pixel shuffle
  4× 视觉 token 压缩 + 动态 tiling）、SmolVLM（256M-2.2B，工业 mini 级）
- **LLaVA 两阶段训练**（Stage 1 投影器对齐：冻结 ViT+LLM 只训 projector，558K；
  Stage 2 视觉指令微调：端到端，150K+515K）——与课程 Part 7/8 的 SFT 叙事无缝衔接

**三大方案对照**（02 章主体）：
| 方案 | 代表 | 机制 | 状态 |
|---|---|---|---|
| (a) 拼接式 projector | LLaVA/SmolVLM/Qwen-VL/InternVL/nanoVLM | 冻结 ViT → MLP → tokens 拼进 LLM | 绝对主流 |
| (b) 交叉注意力门控 | Flamingo (2204.14198) | Perceiver Resampler + gated xattn 注入冻结 LM | 现代开源几乎弃用，对比讲 |
| (c) early-fusion/native | Fuyu-8B（patch 直入 LLM，无视觉编码器）、Chameleon（VQ token） | 图像 patch 当文本 token | 概念教学（Fuyu 仓库已删，HF 卡留档） |

**对齐损失**（特征对齐的"理解侧"起点）：CLIP InfoNCE（softmax 对比）vs SigLIP（sigmoid
成对）——脚本 02 两个都实现并对比。

**章节**：README + 01（手写拼接式 VLM 四件套：patch/ViT/投影器/拼接）+ 02（三大方案 +
对齐损失 CLIP vs SigLIP + 动态分辨率/像素洗牌/token 压缩 + 两阶段训练）。脚本：
01_vit_projector_pipeline.py、02_clip_siglip_alignment.py（均 CPU 可跑）。
**作业**：patch 数量公式、InfoNCE 实现、投影器参数量、动态分辨率 token 估算、🌟 检索增强
tokenizer 之外的对齐方案比较。

## Part 16：图像生成 + 视频生成 — 锚点与章设

**锚点**：huggingface/diffusers（DDPM→SD→FLUX→视频 全谱系；DDPM 快速入门 CPU 可跑；
CogVideoX/Wan 官方 diffusers 权重）。教学路线：
- **01 扩散数学 + 手写 DDPM**（β schedule → α̅ 闭式前向 q(x_t|x_0)=√ᾱx₀+√(1−ᾱ)ε →
  去噪网络训练 → 采样循环；2D toy 分布 CPU 可跑）
- **02 文生图工具链**：LDM（VAE 潜空间 8× 压缩 + cross-attention 条件）→ SD1.5 fp16
  （4090 ~2GB；⚠️ runwayml 原 ID 已删，用 `stable-diffusion-v1-5/stable-diffusion-v1-5`
  镜像）→ SDXL → SD3/FLUX（rectified flow + MMDiT；fp8 ≈12GB 可在 4090 推理）
- **03 图生图与参考条件**：img2img（strength → t₀ = ⌊steps·s⌋，噪声从 q(x_t|x₀) 采样）
  → ControlNet（零卷积注入）→ **IP-Adapter/InstantID/PuLID**（参考图特征经投影作为
  "类文本 token"注入解耦交叉注意力——**跨模态特征对齐的生成侧实践**）
- **04 视频生成**：CogVideoX-2B（fp16 ~4GB / int8 3.6GB，Apache-2.0，24GB 乃至 1080Ti
  可跑——教程友好首选）/ Wan2.1-1.3B（8.19GB，4090 ~4min/5s 480p）/ HunyuanVideo 13B
  （60GB 级，量化可 24GB，引述不实操）；视频 = 3D VAE 潜空间 + **temporal attention**
  （Latte/CogVideoX/Wan 的空间块内插入时间注意力——与图像模型的 diff 就是它）
- **论文**：DDPM 2006.11239 · LDM 2112.10752 · SD3/Rectified Flow 2403.03206 ·
  ControlNet 2302.05543 · IP-Adapter 2308.06721 · PuLID 2404.16022 · CogVideoX 2408.06072 ·
  Wan2.1 2503.20314

**用户强调的"特征对齐"主线**（贯穿 15/16 两章的统一视角）：
```
理解侧(P15)：图像特征 → 对齐到 LLM token 空间（projector 对齐 = LLaVA Stage 1）
生成侧(P16)：文本/参考图特征 → 对齐到扩散网络的条件空间（cross-attn K/V / IP-Adapter KV）
共通原理：把一个模态的表征"翻译"成另一个模态注意力机制能消费的 token——
          翻译器（projector/adapter）+ 翻译训练（对齐阶段）= 多模态的全部秘密
```

**SOTA 挂接（2026-08）**：GLM-5.3-Flash（GLM 系首个原生多模态 + 稀疏/线性混合注意力）、
Qwen3-VL、DeepSeek-V4 原生多模态——"原生多模态/any-to-any"是拼接式之后的下一代方向
（early-fusion 路线的工业兑现），Part 15 02 章"方案演进"收尾于此。

**章节**：README + 01（手写 DDPM）+ 02（文生图工具链与 img2img）+ 03（对齐机制与视频）。
脚本：01_ddpm_from_scratch.py（2D toy，CPU <60s）、02_alignment_mechanisms.py
（cross-attention/IP-Adapter 解耦 KV/CFG 手写，CPU 可跑）。
**作业**：DDPM 闭式前向、SNR 与 β schedule、CFG 公式、img2img strength→t₀、
动态分辨率 token 估算（联动 a15）。

## 环境与版本

- Part 15 脚本：CPU 可跑，零新依赖（torch 即可）
- Part 16 脚本 01/02：CPU 可跑，零新依赖；工具链实操（diffusers）独立 venv：
  `pip install diffusers transformers accelerate`（版本策略：latest；模型权重另下，
  见 docs/datasets.md 增补节）
- 4090 推理验证路径：SmolVLM-500M（<1.5GB）/ Qwen2-VL-2B（4-9GB）/ SD1.5 fp16（~2GB）/
  CogVideoX-2B（~4GB）/ Wan2.1-1.3B（8.2GB）——全部 24GB 内
