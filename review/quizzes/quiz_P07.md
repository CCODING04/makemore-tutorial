# 测验 · Part 07（现代 LLM / Minimind：RoPE、GQA、SwiGLU、DPO）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（数字）** 同一句 "To be or not to be"，字符级和 BPE（6400 词表）各编成多少 token？整库压缩率是多少？为什么 Part 7 预训练的初始 loss 比 Part 6 高得多？

- **答案**：字符级 19 个整数 → BPE 只要 **6 个**；整库（111 万字符）压缩率 ≈ **3.5×**（约 32 万 token）。minimind 词表 6400（GPT-2 约 5 万、Llama 2 3.2 万）。初始 loss 变高是因为均匀分布的熵随词表变大：$\ln 65 \approx 4.17$ 升到 $\ln 6400 \approx 8.76$——两套 tokenizer 的 loss 不可直接比较，可比的是趋势：收敛后 val loss ≈ 2.0、ppl ≈ 7~12（字符级 ≈2.23）。
- **锚点**：教程 01_bpe_tokenizer.md §"运行结果（预期输出）"、§"对比 minimind 的 6400 词表"；04_training_pipeline.md §"② 预训练（Pretrain）：文档补全器"

**Q2（对比）** RMSNorm 相比 LayerNorm 砍掉了什么、保留了什么？为什么这样砍几乎不掉点？顺带：权重绑定（tie_word_embeddings）省的是哪块参数、占多大比例？

- **答案**：砍掉①均值中心化（不减均值）②可学习平移 β ③bias；保留均方根归一化与可学习缩放 γ：$x / \sqrt{\mathrm{mean}(x^2)+\epsilon} \cdot \gamma$。手算 [1,2,3]：RMSNorm 得 [0.463, 0.926, 1.389]（RMS=1，保持正负形状），LayerNorm 得 [-1.225, 0, 1.225]（以 0 为中心）。安全的原因：残差堆叠后激活均值信息量很低，实验性能相当，还省一次减均值/算均值、更稳。权重绑定让 embed_tokens 与 lm_head 共享同一份权重，省 vocab×hidden = 6400×512 ≈ 3.3M，约 26M 模型的 **12%**（GPT-2/Llama 默认开启）。
- **锚点**：教程 02_modern_components.md §"RMSNorm：只算均方根，砍掉均值和平移"、§"数值例子：RMSNorm vs LayerNorm（手算）"、§"权重绑定（tie_word_embeddings）"

**Q3（概念）** 为什么 GQA 敢让 KV 头比 Q 头少（8 Q 头 / 4 KV 头）而质量损失很小？KV Cache 为什么对生成有效、对训练没用？

- **答案**：Q 头必须每头独立（各代表一种"注意力视角"），而 K/V 是"被查询的内容"、各头需求高度重合——分组共享只去冗余。`repeat_kv` 以 $n_{rep} = 8/4 = 2$ 把 K/V 逻辑复制回 8 份。hidden=512/8 层例：KV 线性层参数 MHA ≈524K → GQA ≈262K（减半）、MQA ≈65K；4096 token fp16 缓存 ≈67MB → 33MB → 8MB。KV Cache 有效的根源：因果遮罩下前缀 token 的 K/V 不变，生成时每步只算最后一个 token 的注意力，每步成本从 $O(T^2)$ 降到 $O(T)$；训练时所有位置一起前向+反传，缓存毫无意义。
- **锚点**：教程 03_gqa_and_ffn.md §"GQA：分组共享，折中"、§"GQA 到底省了多少"、§"KV Cache：生成时只算最后一个 token"

**Q4（诊断）** SFT 训练后发现模型"背问题"：会复述或续写 user 的问题，而不是好好回答。最可能漏了什么？怎么修？

- **答案**：漏了 **loss masking**。修法：labels 全置 -100，只把 assistant 区间填成对应目标 token，用 `F.cross_entropy(..., ignore_index=-100)` 跳过非回答位置。注意偏移：labels[t] 存"位置 t 应预测的下一个 token"（input_ids[t+1]），从 `<|im_start|>assistant\n` 的最后一个 token 开始监督回答首词，并在 a_end-1 处监督 `<|im_end|>`（让模型"说完闭嘴"）。延伸：DPO 的参考模型必须冻结（no_grad 只当基准），否则隐式奖励失去"锚点"，模型会在优化偏好时丢掉语言能力。
- **锚点**：教程 04_training_pipeline.md §"关键：Loss Masking（只对 assistant 算 loss）"、§"代码：DPO 训练"

**Q5（数字）** RoPE 长上下文外推四方案（naive/PI/NTK/YaRN）实测："训练 128 → 推理 256"的 ppl、ctx=512 的 needle 检索准确率各是什么排序？YaRN 的温度因子公式是什么？

- **答案**：ppl@外推 256：naive 6.37 / PI 13.35 / NTK 5.20 / **YaRN 5.05**（最优，甚至低于它训练内的 5.27）；PI 训练内就崩到 14.06（零样本换位置分布，"PI 必配微调"）。needle 检索@ctx=512（随机猜 0.100）：naive 0.422 / PI 0.500 / NTK 0.891 / **YaRN 1.000**——ppl 排序与检索排序互相印证，"读得顺 ≠ 记得住"。YaRN 温度因子 $\sqrt{1/t} = 0.1\ln(s)+1$（s=2 时 ≈1.069，微锐化注意力）。
- **锚点**：教程 05_reproduce_minimind.md §"实验 1：RoPE 长上下文四件套（ppl 版）"、§"实验 2：迷你 RULER（needle 检索版）"

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 手写/训练 BPE（65 → 6400 子词），压缩率与 OOV 取舍 | Q1 | 含三难问题（词级/字符级/subword）背景 |
| 从零实现 RMSNorm，讲清为何抛弃均值中心化和 bias | Q2 | 含手算对比 |
| 从零实现 RoPE，理解旋转正交/相对位置/可外推 | Q5 | 可外推与四方案已考；"旋转 → 内积只依赖位置差"（cos(位置差×θ)，数值例 0.878 与绝对位置无关）由闪卡补充 |
| 权重绑定为什么让 embedding"不花参数" | Q2 | 延伸考点（省 vocab×hidden ≈ 3.3M ≈ 12%） |
| 从零实现 GQA 与 repeat_kv，讲清 KV 头为何可少 | Q3 | 含 MHA/MQA/GQA 参数与缓存对照数字 |
| 实现 KV Cache，理解只算最后一个 token、复用什么 | Q3 | 含 $O(T^2) \to O(T)$ 与 start_pos 坑（见闪卡） |
| 从零实现 SwiGLU FFN，对比 ReLU FFN | ⚠️ 正文覆盖不足 | 题量所限未出题：gate/up/down 三投影、中间维度 4× → ~3.2×（$\lceil \pi \cdot h / 64 \rceil \cdot 64$）、silu 负半轴梯度不为 0——正文 03 章 §SwiGLU 已充分展开，收录为闪卡 |
| 理解 MoE 概念、路由器与负载均衡损失 | ⚠️ 正文覆盖不足 | 题量所限未出题：4 专家 / top-1、aux loss 惩罚"被选过多 × 得分高"的重叠、`use_moe` 可选——正文 03 章 §MoE 已展开，收录为闪卡 |
| 跑通 Pretrain → SFT → DPO，讲清 masking、Bradley-Terry、参考模型冻结 | Q4 | Bradley-Terry 与 DPO loss 结构由闪卡补充 |
| 用 PI/NTK/YaRN 做外推并以迷你 RULER 验证 | Q5 | 含 YaRN 三部件与"三层长上下文验证法"背景 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| BPE 算法一句话？ | 从字符集出发，反复合并出现频率最高的相邻 token 对，直到词表达到目标大小（贪心、统计驱动、不管语义） |
| BPE 训练时特殊 token 为什么要用 special_tokens 预留？ | 否则会被当普通文本吃掉/合并掉，词表里腾不出固定位置；预留后它们占据词表最前面（endoftext=0、im_start=1、im_end=2） |
| tokenizer 好坏的三道体检？ | 往返一致性（decode(encode(s))==s）、覆盖性（无 unk）、压缩率（英文字符/token 通常 3~4） |
| learned PE 的两个短板 vs RoPE？ | ①要花 block_size×hidden 的参数表（RoPE 零参数）②不能外推（RoPE 的 cos/sin 表公式生成，改 end 即可重算） |
| RoPE"相对位置"的数值证据？ | 单位向量 q=k=[1,0]、θ=0.5：位置 2/3 与 5/6 的内积都是 $\cos(0.5) \approx 0.878$；位置差 6 时 $\cos(3.0) \approx -0.99$——内积只依赖位置差 |
| 带 KV Cache 时 RoPE 的经典坑？ | cos/sin 角度必须从 start_pos 接着算（`cos[start_pos:start_pos+seq_len]`），从 0 重数会让新 token 位置全错 |
| GQA 与 MHA 在代码上最直接的差别？ | K/V 投影输出头数是 n_kv_heads（4）而非 n_heads（8），使用前多一次 `repeat_kv`；Q 投影完全不动 |
| SwiGLU 的结构与激活？ | `down_proj(silu(gate_proj(x)) * up_proj(x))` 三投影；silu(z)=z·σ(z) 平滑、负区间有小梯度；中间维度从 4× 缩到 ~3.2× 控制总参数量 |
| MoE 为什么参数量 ≠ 计算量？ | 所有专家都装进模型（参数多），但每个 token 只路由到 top-k 个专家（如 4 选 1，计算少）；路由器 = gate 线性层打分 + top-k 加权 |
| MoE 负载均衡损失防什么？ | 防"路由器把 token 全堆给某几个专家"：惩罚 load（被选次数）与 scores.mean（得分）同时大的情形，逼路由器摊开 token |
| Bradley-Terry 模型？ | $P(y_w \succ y_l) = \sigma(r_w - r_l)$；训练损失 $-\log\sigma(r_{chosen}-r_{rejected})$（用 logsigmoid 保数值稳定） |
| DPO 的隐式奖励与 loss？ | 隐式奖励 $= \beta(\log\pi_\theta - \log\pi_{ref})$；loss = $-\log\sigma(\beta(\Delta_w - \Delta_l))$，β 控制离参考模型多远（课程 0.1，minimind 官方 0.15 配 lr 4e-8） |
| 现代 LLM 预训练的四个工程件？ | 混合精度（bf16）、梯度累积（等效大 batch）、梯度裁剪（clip_grad_norm 1.0）、cosine 学习率衰减（min_lr ≈ 0.1·max_lr） |
| MLA 与 NSA 分别压什么？ | MLA 低秩压缩 KV 缓存（实测 MHA 1.07GB → GQA 0.27GB → MLA 0.08GB，为 MHA 的 7.0%，需解耦 RoPE 携带位置专用 key）；NSA 三分支（压缩/选择/滑窗）稀疏化注意力计算，减 FLOPs 不减缓存 |
