# 测验 · Part 15（多模态理解：拼接式 VLM 与对齐损失）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** 拼接式 VLM 的"四件套"是什么？为什么说 Projector 是 LLaVA 的全部增量？"多模态的门槛不在造模型，在训对齐"怎么理解？

- **答案**：四件套：① Patch Embedding（Conv k=s=2 把图切块）→ ② ViT（视觉 token 先双向自交流）→ ③ Projector（mlp2x_gelu，把视觉特征"翻译"进 LLM 维度）→ ④ Token 拼接（图像向量直接当 LLM 输入 embedding 拼进序列）。LLaVA 的视觉塔（CLIP ViT）和 LLM 都是现成冻结的，唯一训练的新东西就是一个 2 层 MLP 投影器（Stage 1 只训它，玩具实测仅 1,856 参数）——所以门槛不在"造模型"，在把两个模态"训对齐"。
- **锚点**：教程 01_handwritten_projection_vlm.md §代码实现（形状账本 + 两阶段训练）

**Q2（数字）** 写出玩具 VLM 的形状账本：(B,3,8,8) 的图经过四件套后每步 shape 是多少？两阶段训练各训多少参数、loss 怎么走？

- **答案**：(B,3,8,8) → PatchEmbed(Conv k=s=2) → (B,16,24)，(8/2)²=16 个视觉 token、每个 24 维 → ViTBlock×2 → (B,16,24) → Projector mlp2x_gelu → (B,16,32)，翻译成 LLM 维度 → 拼接 → (B,16+文本,32)。Stage 1 只训投影器 1,856 参数（冻结 ViT+LLM），loss 2.907→1.825；Stage 2 端到端解冻全部 29,620 参数，loss →0.034。
- **锚点**：教程 01_handwritten_projection_vlm.md §代码实现（形状账本 + 实测输出）

**Q3（对比）** InfoNCE（CLIP）与 SigLIP 的损失形式、batch 依赖性有何差异？课程实测学到的温度 τ 各是多少？对比式对齐与生成式对齐各适合什么场景？

- **答案**：InfoNCE 对 N×N 相似度矩阵按行/列做 softmax 交叉熵（对称双方向，标签=对角线），负样本来自 batch——batch 依赖强，小 batch 信号弱；SigLIP 用逐对 sigmoid（对角 +1、非对角 -1），无全局归一化，batch 依赖弱（论文实测 batch 1/4 持平）。可学习温度 $\tau$ 控制 softmax 锐度，实测 CLIP 路径学到 16.21、SigLIP 8.53；玩具实验两种损失都收敛且图→文检索 top-1 100%（InfoNCE loss 4.155→1.968，SigLIP 0.912→0.106）。适用场景：对比式（CLIP/SigLIP）适合检索/打分/视觉塔预训练；生成式（LLaVA Stage 1 的 next-token CE）适合"让 LLM 消费视觉 token"——现代实践两条都用。
- **锚点**：教程 02_alignment_losses_and_schemes.md §2（对齐损失）+ §3（两阶段与对齐损失的关系）

**Q4（对比）** 三大 VLM 架构方案的注入位置、代表模型与现状是什么？Qwen-VL 的动态分辨率与 InternVL 的 pixel shuffle 各解决什么问题？224² 的图在 patch=14 下产生多少视觉 token？

- **答案**：(a) 拼接式 projector——视觉 token 拼进序列、LLM 无改动（LLaVA/SmolVLM/Qwen-VL/InternVL/nanoVLM），绝对主流；(b) 交叉注意力门控——Perceiver Resampler 压缩后 gated xattn 注入冻结 LM（Flamingo），可保留纯文本能力但结构复杂，现代开源几乎弃用；(c) early-fusion/native——图像就是另一种 token（Fuyu-8B、Chameleon），训练贵上限高，是原生多模态旗舰的工业兑现路线。固定 336² 把小图压糊、大图丢细节（OCR 受伤），Qwen-VL 按原图尺寸切 patch 打包（token 随内容自适应），InternVL 的 pixel shuffle 把 2×2 通道重排进特征维、token 数 ÷4。token 估算：$\lceil 224/14 \rceil^2 = 256$ 个；1024×768 → ceil(1024/14)×ceil(768/14)=4070 个，pixel shuffle ÷4 → 1017 个，超预算（如 2560）还要按比例缩分辨率重算。
- **锚点**：教程 02_alignment_losses_and_schemes.md §1（三大方案全景）+ 概念检验 Q2 + 作业 15 题 4

**Q5（诊断）** Stage 1 如果不冻结 ViT 和 LLM 会怎样？训练中报 `Trying to backward through the graph a second time` 是什么坑？对比学习 loss 不降、检索效果差又该查什么？

- **答案**：不冻结则投影器随机初始化输出的是"噪声 token"，同时更新三个模块会让 LLM 被噪声 token 冲得偏离预训练分布（灾难性遗忘）、ViT 偏离 CLIP 对齐——应先训投影器"搭桥"，桥稳了再动两端。backward 报错是"静态图缓存"坑：X 的计算图含可训练投影器，若把视觉 token 缓存起来跨步复用，第二次 backward 会穿过同一张图——必须每步重建计算图。对比学习 loss 不降先查两点：温度 τ 初始化不当（太小信号弱、太大早期不稳，用可学习 log_scale，CLIP 默认 1/0.07）和 batch 太小（InfoNCE 负样本全来自 batch，加大 batch 或改用 SigLIP）。
- **锚点**：教程 01_handwritten_projection_vlm.md 概念检验 Q1 + §调试展示（错误 1）+ 02_alignment_losses_and_schemes.md §工程实践（错误 1/2、陷阱 1/2）

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 理解多模态理解在 LLM 链路中的位置和价值 | Q1 | 跨模态对齐定位；与 Part 16 生成侧互为镜像并入 Q3 |
| 手写拼接式 VLM 的四件套 | Q1、Q2 | 四件套 + 形状账本 + 两阶段参数账 |
| 解释 CLIP/SigLIP 的数学原理、对齐损失与 batch 依赖性差异 | Q3 | 两种损失 + τ 实测值 + 对比式 vs 生成式 |
| 画出三大方案的注入位置图并完成选型 | Q4 | 拼接式/门控/early-fusion + 动态分辨率选型 |
| 识别常见陷阱（静态图缓存、温度 τ、batch 依赖）并设计防范策略 | Q5 | 三类陷阱的症状—原因—解法 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| 拼接式 VLM 四件套？ | Patch Embedding → ViT → Projector（mlp2x_gelu）→ Token 拼接 |
| 玩具形状账本？ | (B,3,8,8) → (B,16,24) → (B,16,24) → (B,16,32) → (B,16+文本,32)；(8/2)²=16 个 patch |
| LLaVA 两阶段各训什么？ | Stage 1 只训投影器（1,856 参数，loss 2.907→1.825）；Stage 2 端到端（29,620 参数，loss→0.034） |
| Stage 1 为什么冻结两端？ | 随机投影器输出是噪声 token，会冲垮 LLM 预训练分布（灾难性遗忘）与 CLIP 对齐——先搭桥再动两端 |
| InfoNCE 公式要点？ | $0.5 \times (\text{CE}(s f_i f_t^T, y) + \text{CE}((s f_i f_t^T)^T, y))$，对称双方向 softmax，标签=对角线 |
| SigLIP 公式要点？ | targets = 2·eye−1，loss = $-\text{logsigmoid}(\text{targets} \cdot \text{logits}).\text{mean}()$，逐对 sigmoid 无全局归一化 |
| InfoNCE vs SigLIP 的 batch 依赖？ | InfoNCE 负样本来自 batch、依赖强；SigLIP 逐对独立、batch 1/4 持平 |
| 实测学到的温度 τ？ | CLIP 路径 16.21、SigLIP 8.53；τ 控 softmax 锐度，可学习，CLIP 默认 1/0.07 |
| 三大方案与代表？ | (a) 拼接式 LLaVA/Qwen-VL（主流）；(b) 门控 xattn Flamingo（近乎弃用）；(c) early-fusion Fuyu/Chameleon（原生旗舰） |
| 224² 图的视觉 token 数？ | $\lceil 224/14 \rceil^2 = 256$；LLaVA-1.5 336² → 576 token |
| 动态分辨率 token 估算？ | $\lceil h/14 \rceil \times \lceil w/14 \rceil$，pixel shuffle ÷4：1024×768 → 4070 → 1017，超预算 2560 缩分辨率重算 |
| InternVL 的像素洗牌？ | 相邻 2×2 视觉通道重排进特征维，token 数 ÷4（无参压缩；对照 Q-Former 可学习压缩） |
| "静态图缓存 backward"坑？ | 视觉 token 计算图含可训练投影器，必须每步重建，不能跨步缓存 |
| LLaVA Stage 1 用哪种对齐？ | 生成式（图文对 next-token CE），不是 CLIP 对比对齐——目标是让 LLM 读得懂，不是检索 |
