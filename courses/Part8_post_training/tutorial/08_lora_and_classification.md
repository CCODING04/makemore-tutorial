# 08 — LoRA 与分类微调：参数高效微调从零写

> 🧭 SFT 要更新**全部**参数，但真实世界里最常见的诉求是"在一张卡上、不破坏预训练能力、
> 快速把模型调到我的任务上"——这就是 **参数高效微调（PEFT）**，其中事实标准是 **LoRA**。
> 本章从零实现它（对照 rasbt/LLMs-from-scratch 附录 E），顺带走一遍**分类微调**的标准范式
> （rasbt ch06）——这是很多从零课程漏掉、但工作中天天用的技能。

## 📖 前置知识

- **02 章**：SFT 与 prompt masking（LoRA 就是在它之上做减法）
- **10 章（Part 10）**：显存账本 16 字节/参数——本章算"LoRA 省多少"直接用

## 1. 问题：全参微调贵在哪

用 Part 10 的账本算 7B 模型全参微调：**可训练参数 12 字节/个**（fp32 参数+梯度+AdamW 两个状态）
→ 7B × 12 = 84GB 起步，再加激活。这就是"全参微调 7B 要多卡"的全部原因。

LoRA（Low-Rank Adaptation, Hu et al. 2021）的洞察：

```
微调引起的权重变化 ΔW 是低秩的（任务适配不需要全部 7B 个自由度）
  → 不学 ΔW 本身，学它的低秩分解 ΔW = B·A（B: d×r, A: r×k，r≪d,k）
  → 冻结 W，只训 A、B：可训练参数从 d×k 降到 r×(d+k)
```

- 🔑 **三个实现细节决定成败**：① `A` 高斯初始化（1/√r 缩放）、`B` 初始化为 **0**
  → 训练起点 ΔW=0，不破坏预训练表征；② 前向 `y = Wx + (α/r)·B(Ax)`，α/r 缩放让
  调 r 时学习率尺度稳定；③ 推理时可把 BA **合并回 W**（`W += (α/r)·BA`），零额外延迟。
- ⚠️ LoRA 省的是**优化器状态+梯度的显存**（可训练参数从 100% → 百分之几），
  权重本体（bf16 推理副本）仍然全量在显存里——"LoRA 微调 70B 单卡可行"靠的是
  4-bit 量化底座（QLoRA，见 Part 12）。

## 2. 从零实现（跑 `scripts/10_lora_from_scratch.py`）

```python
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)                    # 冻结 W
        out_f, in_f = linear.weight.shape
        self.A = nn.Parameter(torch.randn(r, in_f) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(out_f, r))   # 零初始化 → 起点无损

    def forward(self, x):
        return self.linear(x) + (self.alpha / self.r) * (x @ self.A.T) @ self.B.T
```

注入后**只让 BA + 分类头可训练**（其余全部 `requires_grad_(False)`），然后按
rasbt ch06 的分类微调范式跑对比——任务：序列含 ≥3 个 '7' → 正类（玩具版垃圾邮件识别）：

```
全参微调: 可训练 230,226 参数（100%）  | 验证 acc = 0.955
LoRA 微调: 可训练   7,872 参数（ 3.4%）| 验证 acc = 0.924   ← 注入 4 层，r=4
训练显存估算: 全参 2.8 MB vs LoRA 0.5 MB（真实 7B 上是 GB 级 vs MB 级）
```

- 🔑 **3.4% 的参数达到接近的精度**——"低秩足够"假设的最小可验证证据。
  把数据阈值从 3 调成 4 增加任务难度，会看到 LoRA 与全参的差距开始拉开（r 太小 → 提高r）。
- ⚠️ 实现坑（我们真实踩到）：注入新建的 A/B 默认在 CPU，**注入后要再 `.to(device)` 一次**；
  分类标签要 `.long()` 才能进 cross_entropy。

## 3. 分类微调范式（rasbt ch06 的标准流程，4 步）

```
① 换头：lm_head → cls_head（Linear(hidden, n_classes)），隐藏向量取【最后一个位置】
② 数据：(sequence, label) 对；不需要 prompt/masking —— 这是它与 SFT 的本质区别
③ 冻结策略任选：全参 / 只训头 / +LoRA —— 三档本脚本都给了骨架
④ 评测：accuracy（不是 ppl）—— 语言模型的评估指标在分类任务上不适用
```

- 💡 什么时候用分类微调而不是 SFT？——**输出是离散类别**（情感/风控/路由/质检）时。
  工作里"用 LLM 做分类器"比"用 LLM 生成"更常见于生产管线，这是 rasbt ch06 值得单独
  一章的原因。

## 4. 与 Part 12 的关系

本章 = **概念与从零实现**（几十行，看得见每一行）；Part 12 = **工业工具**
（LLaMA-Factory 的 LoRA/QLoRA/DPO 全家桶 + 7B 模型实战）。学完本章再去 Part 12，
工具的每个 yaml 字段（`lora_rank`/`lora_alpha`/`lora_target`）你都知道对应哪行代码。

## 学完本部分你能...

- ✅ 手写 LoRALinear，说清 A/B 初始化约定与 α/r 缩放的作用
- ✅ 算清 LoRA 省的是哪部分显存、不省哪部分
- ✅ 走通分类微调 4 步范式，说出它与 SFT 的本质区别
- ✅ 用"可训练参数比例 + acc 对比"验证参数高效微调的有效性

**课后练习**

<details>
<summary>Q1: 为什么 B 初始化为 0 而 A 不为 0？两个都为 0 行不行？</summary>
A: B=0 使初始 ΔW=BA=0，训练起点等价原模型。若 A 也为 0，则 ∂L/∂A ∝ B=0、∂L/∂B ∝ A=0，
两个矩阵的梯度都恒为 0——永远学不动。所以必须"一零一非零"打破对称。
</details>

<details>
<summary>Q2: LoRA 应该注入哪些层？注入 attention 还是 MLP？</summary>
A: 原论文在 attention 的 Wq/Wv 上实验；实践共识（和 QLoRA 论文）是"全部 Linear 都注入
效果最好"，预算有限时优先 attention 的 q/v。注入层越多可训练参数越多、越接近全参效果。
本脚本注入 MLP 两个 Linear 是为了 toy 上快速演示，把 target 换成 attention 可自行实验。
</details>

<details>
<summary>Q3: 推理时 LoRA 有额外开销吗？多租户（一个基座 + N 个适配器）怎么办？</summary>
A: 合并回 W 后零开销；不合并则有 BA 的额外矩阵乘。多租户场景（vLLM 的 multi-LoRA）：
基座共享一份，N 个适配器按请求动态切换——这正是 Part 12/14 工具链的卖点之一。
</details>

## 📝 课后作业

跑通 [scripts/10_lora_from_scratch.py](../scripts/10_lora_from_scratch.py) 并完成三个实验：
r=2/4/16 对比 acc、注入 attention vs MLP 对比、任务难度（阈值 3→4）对 LoRA-全参差距的影响。

## 下一步

参数高效微调的"从零"到此为止。工具级实战（QLoRA 7B、DPO-LoRA、WebUI）在 Part 12
（LLaMA-Factory）；再往后是工业 RL 框架（Part 11 verl）。

👉 [Part 12 LLaMA-Factory 微调实战（拟开）](../../Part12_finetune_llamafactory/tutorial/README.md)

---

[← 上一章：评估学](07_evaluation.md) | [Part 8 README](README.md)
