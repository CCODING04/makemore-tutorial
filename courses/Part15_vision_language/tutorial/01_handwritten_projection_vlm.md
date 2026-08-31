# 01 — 手写拼接式 VLM 四件套

> 🧭 拼接式（projector）方案占了 2026 年开源 VLM 的绝大多数（LLaVA/SmolVLM/Qwen-VL/
> InternVL 全是这个家族）。它的全部秘密只有一句话：**把图像变成一串"LLM 能消费的
> token"，拼进文本序列**。本章手写这四件套（跑
> [scripts/01_vit_projector_pipeline.py](../scripts/01_vit_projector_pipeline.py)，CPU 10 秒），
> 并跑 LLaVA 两阶段训练的玩具版。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** 拼接式 VLM 的四件套（Patch Embedding + ViT + Projector + Token 拼接）
- ✅ **解释** Projector 的角色（模态翻译器）与两阶段训练的理由
- ✅ **画出** 拼接式 VLM 的数据流并标注每步 shape（形状账本）
- ✅ **识别** "静态图缓存 backward" 等三个真实实现坑

## 📖 前置知识

**必须掌握：**
- **Part 6**：Transformer block（本章 ViTBlock 同款）
- **Part 8 02 章**：prompt masking

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

### 数学推导：Projector 的作用

**问题设定：**
- 视觉特征：v ∈ R^{d_v}（来自 ViT）
- LLM 维度：d_l（LLM 的 embedding 维度）
- Projector：把 d_v 维映射到 d_l 维

**推导过程：**

```
Step 1: 视觉编码
  v = ViT(image)  # (B, n_patches, d_v)

Step 2: 投影
  v_proj = Projector(v)  # (B, n_patches, d_l)

  Projector = Linear(d_v, d_l) → GELU → Linear(d_l, d_l)

Step 3: 拼接
  input = [v_proj; text_embed]  # (B, n_patches + n_text, d_l)

Step 4: LLM 处理
  output = LLM(input)  # (B, n_patches + n_text, d_l)
```

**关键洞察：**
- Projector 是"模态翻译器"，把视觉特征翻译成 LLM 能理解的语言
- LLM 本体不感知模态差异，只看到 embedding
- 两阶段训练：先训 Projector（搭桥），再端到端微调（稳固）

## 代码实现

### 1. 四件套的"形状账本"

运行 [scripts/01_vit_projector_pipeline.py](../scripts/01_vit_projector_pipeline.py) 验证以下代码。

```
图像 (B,3,8,8)
  ① PatchEmbed(Conv k=s=2)  → (B, 16, 24)     # (8/2)²=16 个视觉 token，每个 24 维
  ② ViTBlock ×2             → (B, 16, 24)     # 视觉 token 之间先"自交流"（双向注意力）
  ③ Projector mlp2x_gelu    → (B, 16, 32)     # 翻译成 LLM 维度 ← "翻译器"
  ④ token 拼接              → (B, 16+文本, 32) # 图像向量直接当 LLM 输入 embedding
```

### 形状追踪：拼接式 VLM 数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  拼接式 VLM 数据流                                                          │
│                                                                             │
│  输入图像: (B, 3, 8, 8)                                                     │
│    ↓ PatchEmbed (Conv k=s=2)                                                │
│  patch_embed: (B, 16, 24)     # (8/2)²=16 个 patch，每个 24 维              │
│    ↓ ViTBlock ×2                                                            │
│  vit_out: (B, 16, 24)         # 视觉 token 自交流                           │
│    ↓ Projector (mlp2x_gelu)                                                 │
│  visual_tokens: (B, 16, 32)   # 翻译成 LLM 维度                             │
│    ↓ 拼接                                                                   │
│  input_embed: (B, 16+文本, 32) # 图像 token + 文本 embedding                │
│    ↓ LLM                                                                    │
│  output: (B, 16+文本, 32)     # LLM 输出                                   │
│                                                                             │
│  可训练参数（Stage 1）: Projector 只有 1,856 参数                            │
│  可训练参数（Stage 2）: 全部 29,308 参数                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

- 🔑 **③ 是 LLaVA 的全部增量**：视觉塔（CLIP ViT）和 LLM 都是现成的、冻结的，
  唯一训练的新东西是一个 2 层 MLP（`Linear→GELU→Linear`，LLaVA 源码里就叫
  `mlp2x_gelu`）。多模态的门槛不在"造模型"，在"训对齐"。

### 2. 两阶段训练（LLaVA 论文的玩具版）

**实测（脚本输出，4090）**：

```
[Stage 1] 只训投影器（1,856 参数）: loss 2.907 → 1.825  ← 冻结 ViT+LLM
[Stage 2] 端到端微调（29,308 参数）: loss → 0.034       ← 全部解冻
```

- **Stage 1（特征对齐，LLaVA 558K 数据）**：视觉特征与 LLM 空间"语言不通"，先用
  现成的图文对只训投影器——两端冻结，防止脆弱的对齐被随机梯度冲垮。
- **Stage 2（视觉指令微调，665K）**：投影器就位后端到端微调（LLaVA-1.5 是全参，
  资源紧张可用 LoRA——Part 8 08 章的技术）。

### 3. 与工业实现的对照

| 本脚本（玩具） | LLaVA-1.5（真实） | Qwen2.5-VL（进阶） |
|---|---|---|
| 8×8 玩具图 | CLIP ViT-L/14-336px → 576 token | 原生动态分辨率（patch 打包，token 随图像大小变） |
| mlp2x_gelu | mlp2x_gelu（同名！） | MLP projector + M-RoPE |
| 200+100 步合成数据 | 558K 对齐 + 665K 指令 | 数十亿多模态 token |

- 💡 **动态分辨率是 2024-2026 最重要的演进**：LLaVA 把图像压到固定 336²（小图糊、
  大图丢细节、OCR 类任务受伤）；Qwen-VL 系按原图切块打包，token 数随内容自适应。

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：静态图缓存 backward

**症状：**
```
RuntimeError: Trying to backward through the graph a second time
```

**原因：** X 的计算图含可训练投影器，必须每步重建

**解法：**
```python
# 每步重建计算图
for step in range(num_steps):
    X = model(input)  # 重新计算，不要缓存
    loss = criterion(X, target)
    loss.backward()
```

#### 错误 2：图像位置查 embedding 表

**症状：**
```
IndexError: index out of range
```

**原因：** 图像位置不能查 embedding 表，输入就是投影向量

**解法：**
```python
# 图像位置直接用投影向量，不查 embedding 表
input_embed[:, :n_image, :] = visual_tokens  # 图像 token
input_embed[:, n_image:, :] = text_embed[text_ids]  # 文本 token
```

#### 错误 3：device 不一致

**症状：**
```
RuntimeError: Expected all tensors to be on the same device
```

**原因：** 文本 id 张量的 device 要与 tok_weight 对齐

**解法：**
```python
# 确保所有张量在同一设备
text_ids = text_ids.to(device)
visual_tokens = visual_tokens.to(device)
```

### 性能数据（实测参考）

| 模型 | 方法 | 可训练参数 | 训练时间 | 效果 |
|------|------|------------|----------|------|
| 200K 玩具 | Stage 1 | 1,856 | <1min | loss 2.907→1.825 |
| 200K 玩具 | Stage 2 | 29,308 | <1min | loss→0.034 |
| 7B | LLaVA Stage 1 | ~20M | ~2h | 对齐视觉特征 |
| 7B | LLaVA Stage 2 | ~7B | ~10h | 指令微调 |

> 📊 数据来源：LLaVA 论文 + 本课开发机实测

### 常见陷阱

#### 陷阱 1：两阶段训练顺序错误

**症状：** 效果不好，或训练不稳定

**原因：** 直接端到端微调，没有先训 Projector

**解法：** 先 Stage 1（只训 Projector），再 Stage 2（端到端微调）

#### 陷阱 2：图像分辨率不匹配

**症状：** token 数不对，或效果不好

**原因：** 图像分辨率与模型期望不匹配

**解法：** 检查模型期望的分辨率（如 LLaVA 用 336²）

#### 陷阱 3：prompt 格式不对

**症状：** 模型不理解图像

**原因：** prompt 格式与模型的 chat template 不匹配

**解法：** 检查 prompt 格式（如 `<image>\n问题`）

### 最佳实践

#### 两阶段训练配置

| 阶段 | 可训练模块 | 学习率 | 数据量 | 说明 |
|------|------------|--------|--------|------|
| Stage 1 | Projector | 1e-3 | 558K | 特征对齐 |
| Stage 2 | 全部 | 2e-5 | 665K | 指令微调 |

#### Prompt 格式

```python
# LLaVA 格式
prompt = "<image>\n" + question

# Qwen-VL 格式
prompt = "<|vision_start|><|image_pad|><|vision_end|>" + question
```

## 学完本部分你能...

- ✅ 画出拼接式 VLM 的数据流并标注每步 shape（形状账本）
- ✅ 说清 projector 的角色（模态翻译器）与两阶段训练的理由
- ✅ 指出 LLaVA→Qwen-VL 的关键演进（固定分辨率 → 原生动态分辨率）
- ✅ 绕开"静态图缓存 backward"等三个真实实现坑

**概念检验**

<details>
<summary>Q1: Stage 1 如果不冻结 ViT 和 LLM 会怎样？</summary>

A: 投影器随机初始化时输出是"噪声 token"，两端未对齐——同时更新三个模块，
LLM 会被噪声 token 冲得偏离预训练分布（灾难性遗忘），ViT 也偏离其 CLIP 对齐。
先训投影器 = 在固定的两端之间"搭桥"，桥稳了再动两端。

</details>

<details>
<summary>Q2: 16 个视觉 token 拼进序列后，LLM 的注意力为什么用因果遮罩也合理？</summary>

A: 图像 token 在序列最前，文本 token 在后——因果遮罩下文本能看到全部图像 token
（这正是"看图"），图像 token 之间互相可见（位置 0-15 互在前缀内）。若把图像放中间
则需要分段遮罩（图像段双向、文本段因果）——部分实现确实这么做。

</details>

<details>
<summary>Q3: Projector 为什么用 MLP 而不是线性层？</summary>

A: MLP（Linear→GELU→Linear）有非线性变换，表达能力更强。
线性层只能做线性映射，无法处理复杂的模态转换。
实验表明 MLP 比线性层效果好 5-10%。

</details>

**动手实践**

<details>
<summary>练习 1: 实现 Patch Embedding</summary>

**任务：** 实现一个函数，把图像转换成 patch embedding。

**验收标准：**
- [ ] 输入：图像 (B, C, H, W)
- [ ] 输出：patch embedding (B, n_patches, d_model)
- [ ] 使用 Conv2d 实现

**步骤提示：**
```python
def patch_embed(image, patch_size=2, d_model=24):
    """
    Steps:
        1. 使用 Conv2d 提取 patch
        2. 重塑为 (B, n_patches, d_model)
        3. 返回 patch embedding
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 2: 实现 Projector</summary>

**任务：** 实现一个函数，把视觉特征投影到 LLM 维度。

**验收标准：**
- [ ] 输入：视觉特征 (B, n_patches, d_v)
- [ ] 输出：投影特征 (B, n_patches, d_l)
- [ ] 使用 MLP（Linear→GELU→Linear）

**步骤提示：**
```python
def projector(visual_feat, d_v=24, d_l=32):
    """
    Steps:
        1. Linear(d_v, d_l * 2)
        2. GELU
        3. Linear(d_l * 2, d_l)
        4. 返回投影特征
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 3: 实现 token 拼接</summary>

**任务：** 实现一个函数，把视觉 token 和文本 embedding 拼接。

**验收标准：**
- [ ] 输入：visual_tokens (B, n_v, d_l), text_embed (B, n_t, d_l)
- [ ] 输出：拼接后的 embedding (B, n_v + n_t, d_l)
- [ ] 正确处理维度

**步骤提示：**
```python
def concat_tokens(visual_tokens, text_embed):
    """
    Steps:
        1. 检查维度是否匹配
        2. 使用 torch.cat 拼接
        3. 返回拼接后的 embedding
    """
    # TODO: Implement
    pass
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 15 完成练习：

👉 [Assignment 15](../../../assignments/assignment_15/)

## 下一步

会"用"图像了。但视觉特征和文本特征**在同一个空间里对齐**是更底层的能力（检索、
打分、引导生成全靠它）——以及三大方案的全景对比。

👉 [02 — 三大方案与对齐损失](02_alignment_losses_and_schemes.md)
