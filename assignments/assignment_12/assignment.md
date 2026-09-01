# Assignment 12：微调实战（LLaMA-Factory）

> 对应 Part 12 教程（[01 手写管线](../../courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.md) / [02 工作流](../../courses/Part12_finetune_llamafactory/tutorial/02_llamafactory_workflow.md)）。
> 核心四道题全部纸笔/纯 Python 可完成；🌟 题 5 为 stretch 加分（torch 实现，可选，未实现返回 None 时测试自动 SKIP ⏭️）；实验题跑 02 章的工具链。

## 题目（实现 `finetune_exercises.py` 后 `python test_finetune_exercises.py`）

| 题 | 内容 | 分值 |
|---|---|---|
| 1 | LoRA 参数账：算可训练参数 Σ r·(out+in) 及占全参比例 | 25 分 |
| 2 | 合并数学：`W' = W + (α/r)·B@A`，并验证"合并前后前向一致" | 25 分 |
| 3 | B 零初始化：证明 B=0 时起点 ΔW=0（Frobenius 范数）——"起点无损"的数学表述 | 25 分 |
| 4 | QLoRA 显存账：底座量化存储（params × bits/8）+ 可训练部分（×12B/参数） | 25 分 |
| 5 🌟 | 多 rank 对比实验（stretch，torch 实现） | 加分 20 分，不计入基础分 |

> **计分口径**：核心 4 题 25×4 = **100 分**；🌟 题 5 为 **stretch 加分项**（+20 分，不计入
> 100 分基础分，单独记录）。题 5 未实现返回 `None` 时测试优雅 SKIP ⏭️，不影响任何得分。

### 题 1：LoRA 参数账（25 分）

给定被注入层 dims，算可训练参数 Σ r·(out+in) 及占全参比例——02 章 yaml 里
`lora_rank` 一个数字到底买了多少参数，这题就是它的公式。

**验收标准：**
- [ ] `lora_params([(288, 96), (96, 288)], 4) == 3072`（精确整数，非近似）
- [ ] `lora_ratio(...)` = 参数量 / base_params（返回浮点比例）
- [ ] 纯 Python 完成（不需要 torch）

### 题 2：合并数学（25 分）

`W' = W + (α/r)·B@A`，并验证"合并前后前向输出一致"（merge 的正确性）——
这是 01 章 `merge_lora()` / `llamafactory-cli export` 背后的全部数学。

**验收标准：**
- [ ] `merged_weight` 返回 (out, in) 张量，**不原地修改** W
- [ ] 与 `W + (α/r)·B@A` 逐元素 allclose（atol=1e-6）
- [ ] `merge_changes_output` 对随机 W/A/B/x 返回 True（合并前后 `Wx+(α/r)·BAx` 与 `W'x` 一致）

### 题 3：B 零初始化的意义（25 分）

证明 B=0 时起点 ΔW=0（Frobenius 范数）——"起点无损"的数学表述。

**验收标准：**
- [ ] `initial_delta_norm(A, torch.zeros(8, 4)) < 1e-9`
- [ ] 返回 `float`（Frobenius 范数 = 元素平方和开根）
- [ ] 对非零 B 返回正确的范数（不是恒返回 0）

### 题 4：QLoRA 显存账（25 分）

底座量化存储（params × bits/8）+ 可训练部分（×12B/参数：fp32 参数+梯度+AdamW 两个动量）。
这是"QLoRA 7B 为什么 6GB 能跑"的粗估公式。

**验收标准：**
- [ ] `qlora_vram_gb(7.0, 4, 20_000_000) ≈ 3.74`（相对误差 < 1e-9）
- [ ] 单位换算正确：参数量 × 字节数 / 1e9 得 GB
- [ ] 可训练部分用 12 字节/参数口径（呼应 Part 10 的优化器账本）

### 题 5：🌟 多 rank 对比实验（stretch，加分 20 分，不计入 100）

冻结底座 W，真实权重更新是**秩 4** 的固定矩阵 ΔW\*=B\*@A\*；对 r ∈ {1,2,4,8} 各注入一个
A(高斯)/B(零) 旁路，Adam 只训 A/B，对比起点/终点 loss——r ≥ 4 能完整表达 ΔW\*（loss → 0），
r < 4 只能学到最优低秩近似（loss 卡在更高的平台）。这题用实验回答"lora_rank 该开多大"。

**验收标准：**
- [ ] 返回 dict，keys 恰为 `ranks=(1, 2, 4, 8)`
- [ ] `params[r] == r·(out_f+in_f)`（精确整数，呼应题 1 公式）
- [ ] 各 r 的 `loss_start` 相同（B=0 ⇒ 起点输出即纯底座，与 r 无关）
- [ ] 每个 r 的 `loss_end < loss_start`（任意 rank ≥ 1 都能学到一点）
- [ ] `r=8` 的 `loss_end ≤ r=1` 的（rank 越大表达力越强）
- [ ] `r=4`（≥ 目标秩）的 `loss_end < 0.1 × loss_start`（足够大的 rank 学到近零）

**步骤提示**（骨架 docstring 里有完整 Steps）：前向 `Yh = X@W.T + (α/r)·(X@A.T)@B.T`，
`F.mse_loss(Yh, Y)`，Adam 300 步、lr=5e-2 即可（参考实现 ~0.5 秒跑完 4 个 rank）。
未实现返回 `None` → 测试 SKIP ⏭️，不扣分。

## 实验题（观测型，写 3-5 行结论进面经）

- 跑 02 章工具链：QLoRA 7B × identity 数据集，记录显存曲线（对照官方 6GB 数字）
- `lora_rank` 8 → 64：记录可训练参数比例与显存变化（呼应题 1 的公式）

## 🤔 思考题

**Q1：** LoRA 省的到底是什么显存？底座权重还要不要常驻？

<details>
<summary>💡 参考答案</summary>

省的是**可训练参数的梯度和优化器状态**：AdamW 为每个可训练参数维护 fp32 的参数副本、
梯度、一阶/二阶动量（约 12 字节/参数）。全参微调 7B 时这部分 ≈ 84GB+；LoRA r=8 只训
约 20M 参数，优化器侧 < 1GB。但底座 W 必须全程常驻（bf16 ≈ 14GB，QLoRA 4bit ≈ 3.5GB）——
LoRA 省的是"动量账"，不是"底座账"（正是题 4 公式的两项来源）。

</details>

**Q2：** 为什么 B 初始化为 0，而 A 不能也是 0？

<details>
<summary>💡 参考答案</summary>

B=0 保证训练起点 ΔW = BA = 0，不破坏预训练权重（题 3 的 Frobenius 范数为 0 即其数学表述）。
但 ∂L/∂B 依赖于 A、∂L/∂A 依赖于 B——若 A 也为 0，两个梯度同时为 0，参数永远不动
（一个死鞍点）。"高斯 A + 零 B"是同时满足"起点无损"和"梯度非零"的最简单组合。

</details>

**Q3：** 合并后为什么推理零开销？有没有**不该**合并的场景？

<details>
<summary>💡 参考答案</summary>

`W' = W + (α/r)·BA` 是精确加法：合并后前向从 `Wx + (α/r)·BAx`（两次矩阵乘）变成
`W'x`（一次），数学上逐元素相等——01 章脚本实测合并前后 max|Δlogits| ≤ 2.4e-06
（纯浮点舍入，脚本断言阈值 1e-4）。不该合并的场景：一份底座要服务多个任务/租户、
动态挂不同 adapter（如 vLLM multi-LoRA）——合并就失去了"共享底座 + 热插拔"的意义。

</details>

**Q4：** QLoRA 的 "Q" 量化的是哪部分？LoRA 的 A/B 也被量化吗？

<details>
<summary>💡 参考答案</summary>

只量化**冻结底座**（NF4 + 双重量化，0.5 字节/参数）；LoRA 的 A/B 以及 lm_head 等
可训练部分保持 bf16/fp32 高精度，反传时对量化权重做反量化计算。若把 A/B 也量化，
低精度的梯度更新基本学不动——这正是题 4 公式"底座 × bits/8 + 可训练 × 12B"两口径不同的原因。

</details>

**Q5：** `lora_rank` 从 4 提到 64，什么时候值得？代价是什么？

<details>
<summary>💡 参考答案</summary>

每层参数量随 r **线性**增长（r·(out+in)）。值得加大的场景：任务与预训练分布差距大、
数据量大（几十万条以上）、或要学"新知识"而不只是"新格式"。代价：优化器状态与梯度线性
上涨、小数据上过拟合风险上升。题 5 的实验给出判据——若目标更新本身是低秩的（秩 4 的
目标用 r=4 就把 loss 打到近零），再加 r 只加参数不加效果。实践：从小 r 起步做基线，
效果卡住再加倍。

</details>
