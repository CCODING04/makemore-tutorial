# 测验 · Part 12（微调实战：LLaMA-Factory，LoRA/QLoRA/DPO 全流程）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** 手写 LoRA SFT 流水线有哪五个环节？它们分别对应 LLaMA-Factory 的哪些 yaml 字段？

- **答案**：五步为：[0] 基座预热 → [1] LoRA 注入（apply_lora，注入 MLP Linear，冻结底座）→ [2] SFT 训练（chat 格式 + labels[:n_prompt]=-100 的 prompt masking）→ [3] 推理验证 → [4] 合并 merge（$W' = W + (\alpha/r) \cdot BA$）。字段映射：prompt 拼接与 masking ↔ `template:` + `train_on_prompt: false`；注入 ↔ `lora_target:/lora_rank:/lora_alpha:`；数据 ↔ `dataset:` + `dataset_info.json`；训练超参 ↔ `learning_rate:/num_train_epochs:` 等；合并 ↔ `llamafactory-cli export`。
- **锚点**：教程 01_handwritten_sft_lora.md §代码实现（微型管线五步 + 五步与 yaml 字段的对照）

**Q2（对比）** LoRA 与 QLoRA 各省的是哪部分显存？"QLoRA 7B 官方数字 6GB"里，4bit 量化的到底是什么？LoRA 的 A/B 被量化了吗？

- **答案**：LoRA 省的是"动量账"——可训练参数的梯度与优化器状态（AdamW 约 12 字节/参数，全参 7B 这部分 84GB+，LoRA r=8 只训约 20M 参数）；但底座 W 必须常驻（bf16 约 14GB）。QLoRA 再省"底座账"：把冻结底座量化为 NF4 4bit（0.5 字节/参数，7B 约 3.5GB，双重量化再省约 0.37GB），官方口径 4bit 7B 约 6GB 可跑。A/B 不被量化，保持 bf16/fp32 高精度训练——否则低精度梯度更新学不动。代价上 QLoRA 约 93% 全参效果，LoRA bf16 约 95%。
- **锚点**：教程 02_llamafactory_workflow.md §理论背景（为什么需要 QLoRA）+ 作业 12 思考题 Q1/Q4

**Q3（数字）** 课程实测的玩具 LoRA：可训练参数是多少、占全参比例多少？d=4096、k=4096、r=8 时单层压缩多少倍？LoRA 的 A、B 分别如何初始化、为什么？

- **答案**：玩具模型（200K 参数 GPT，r=4，注入 4 层 MLP Linear）可训练 6,144 / 200,664，即 3.1%；训练 loss 从 3.572 降到 0.076。单层压缩比：全参 $d \times k = 16{,}777{,}216$，LoRA $r \times (d+k) = 65{,}536$，压缩 256 倍。初始化：A 高斯初始化、B 零初始化——训练起点 $\Delta W = BA = 0$ 不破坏预训练权重（起点无损，Frobenius 范数为 0）；若 A 也为 0，则 $\partial L/\partial A$ 与 $\partial L/\partial B$ 同时为 0，参数永远不动（死鞍点）。
- **锚点**：教程 01_handwritten_sft_lora.md §理论背景 + §代码实现（形状追踪）+ 作业 12 题 3 与思考题 Q2

**Q4（诊断）** 一份 yaml 跑起来 loss 不下降、输出格式混乱；另一份报 `CUDA out of memory`。各最可能是什么原因？如何排查？为什么说跑挂时 90% 的问题在数据格式与显存估算？

- **答案**：格式混乱/loss 不降 → 数据格式与模型 chat template 不匹配（如 Qwen 需要 `<|im_start|>`）或 template 名不存在，排查手段是核对 template 与 dataset_info.json 的字段映射；OOM → cutoff_len 太大或 batch_size 太大，解法是减小 cutoff_len（如 512）、batch 置 1、开 gradient checkpointing。这两个 debug 能力恰恰来自手写对照：template/masking 对应 build_sample 的 labels[:n_prompt]=-100，显存账对应 cutoff_len × batch 的激活开销。
- **锚点**：教程 02_llamafactory_workflow.md §工程实践（错误 1/2）+ §代码实现第 6 节（手写 vs 工具：一张总账）

**Q5（概念）** DPO-LoRA 训练后，rewards/margins 曲线怎么读才算收敛？DPO 的学习率为什么比 SFT 更小？合并 adapter 有没有"不该合并"的场景？

- **答案**：DPO 后 `rewards/chosen` 应上升、`rewards/rejected` 应下降、`rewards/margins` 应变正且扩大；收敛标志是 margins 稳定在 0.5-2.0 之间不再明显波动。DPO 的 lr 用 5e-6 量级（比 SFT 的 1e-4~5e-5 更小）——"越靠后阶段 lr 越小"规律的又一实证。不该合并的场景：一份底座要服务多任务/多租户、动态挂不同 adapter（如 vLLM multi-LoRA）——合并就失去"共享底座 + 热插拔"的意义；合并本身是精确加法（实测合并前后 max|Δlogits| ≤ 2.4e-06），推理零额外开销。
- **锚点**：教程 02_llamafactory_workflow.md §代码实现（第 5 步 DPO-LoRA + 概念检验 Q2/Q3）+ 作业 12 思考题 Q3

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 理解微调在 LLM 链路中的位置和价值 | Q2 | 全参 vs LoRA vs QLoRA 的成本/效果权衡（7B 全参 ~120GB vs QLoRA ~6GB） |
| 手写 LoRA SFT 的完整流水线 | Q1 | 五步管线与 yaml 字段映射 |
| 解释 LoRA/QLoRA/DPO 的数学原理和工程权衡 | Q2、Q3 | 低秩分解/初始化/量化对象；DPO 曲线并入 Q5 |
| 配置 LLaMA-Factory 的 yaml 并完成生产链路 | Q4 | 字段含义与 debug；export→chat/api 链路并入 Q1/Q5 |
| 识别微调中的常见陷阱并设计防范策略 | Q4 | 数据格式不匹配、OOM、注入后忘搬 device |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| LoRA 参数量公式？ | 每层 $r \times (d+k)$；示例 d=k=4096、r=8 时 65,536，压缩 256 倍 |
| LoRA 前向与合并公式？ | 前向 $h = Wx + (\alpha/r) \cdot BAx$；合并 $W' = W + (\alpha/r) \cdot BA$（精确加法，推理零开销） |
| A、B 的初始化与原因？ | A 高斯、B 零；起点 $\Delta W = BA = 0$ 起点无损；A 若也为 0 则梯度全零死鞍点 |
| 玩具脚本 [1] 的可训练参数与占比？ | 6,144 / 200,664 = 3.1%（r=4，注入 4 层 MLP Linear） |
| LoRA vs QLoRA 各省哪笔账？ | LoRA 省"动量账"（梯度+优化器，约 12B/可训练参数）；QLoRA 再省"底座账"（NF4 4bit，7B 约 3.5GB） |
| QLoRA 的 Q 量化什么？ | 只量化冻结底座（NF4 + 双重量化）；A/B 保持 bf16/fp32 高精度 |
| prompt masking 的实现与字段？ | labels[:n_prompt] = -100；对应 `train_on_prompt: false` |
| LLaMA-Factory 跑挂 90% 的两类问题？ | 数据格式（template 不匹配）与显存估算（cutoff_len × batch） |
| DPO 收敛怎么看？ | rewards/chosen 上升、rejected 下降、margins 变正扩大且稳定在 0.5-2.0 |
| 什么时候不该合并 LoRA？ | 多租户/多任务动态挂 adapter（vLLM multi-LoRA），保留共享底座 + 热插拔 |
| lora_rank 推荐值与加大时机？ | 8-64，从 8 起步；任务分布差距大/数据量大时值得，参数随 r 线性增长 |
| QLoRA 显存粗估公式？ | 底座 params × bits/8 + 可训练部分 × 12 字节；`qlora_vram_gb(7.0, 4, 20M) ≈ 3.74GB` |
