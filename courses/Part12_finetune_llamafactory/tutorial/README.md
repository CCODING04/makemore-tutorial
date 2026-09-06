# Part 12: 微调实战 — LLaMA-Factory（LoRA / QLoRA / DPO 全流程）

> 🧭 Part 8 用几十行手写了 LoRA 的原理；本部分把同样的技能放大到**工业工具**：
> 用 LLaMA-Factory 在真实 7B 模型上走完 LoRA SFT → QLoRA → 合并导出 → DPO 的完整工作流。
> 学完你能独立承担"把一个开源基座调到业务任务上"的工程任务。
> 主源：[hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)（74.4k★，截至 2026-09，Apache-2.0）

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** 微调在 LLM 链路中的位置和价值
- ✅ **手写** LoRA SFT 的完整流水线（chat template + masking + LoRA 注入 + 训练）
- ✅ **解释** LoRA/QLoRA/DPO 的数学原理和工程权衡
- ✅ **配置** LLaMA-Factory 的 yaml 并完成 LoRA SFT → QLoRA 7B → export → DPO 的生产链路
- ✅ **识别** 微调中的常见陷阱并设计防范策略

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 LoRA SFT：工具自动化的到底是什么](01_handwritten_sft_lora.md) | chat template+masking+LoRA 注入+训练循环 手写一遍（工具的"内部透视"） | `01` |
| 02 | [LLaMA-Factory 工作流](02_llamafactory_workflow.md) | identity LoRA SFT → WebUI → QLoRA 7B → export 合并 → DPO-LoRA | —（CLI 实操） |

## 🧰 前置知识

**必须掌握：**
- **[Part 8 08 章](../../Part8_post_training/tutorial/08_lora_and_classification.md)**：从零 LoRA（A/B 初始化、α/r、注入位置）——本章工具的每个 yaml 字段都对应它
- **[Part 8 02 章](../../Part8_post_training/tutorial/02_sft_and_chat.md)**：SFT 与 prompt masking
  （P8 用乘法 mask 实现，本 Part 脚本改用 `labels=-100`——两种等价实现）
- **[Part 8 03 章](../../Part8_post_training/tutorial/03_reward_and_dpo.md)**：DPO

**建议掌握：**
- **[Part 10](../../Part10_distributed/tutorial/README.md)**：FSDP 与多卡基础（多卡微调用到）

**可选：**
- **[Part 11](../../Part11_alignment_verl/tutorial/README.md)**：verl 对齐实战（DPO 与 GRPO 的对比）

## 🔗 在 LLM 链路中的位置

```
预训练(Part 7/13) → [本部分: 微调 SFT/LoRA/QLoRA/DPO] → 对齐 RL(Part 11) → 部署(Part 14)
                        ↑
                        你在这里
```

**为什么微调是"业务可用"的第一手段：**

| 证据 | 说明 | 来源与口径 |
|------|------|------------|
| 成本 | 全参微调 7B 需要 ~120GB 显存，QLoRA 只需 ~6GB | LLaMA-Factory 官方 benchmark（量级参考，未逐行本机复现；账本拆解见 01 章性能表 / 02 章 §3） |
| 效果 | LoRA 在大部分任务上能达到全参微调 90%+ 的效果 | 趋势参考：QLoRA 论文（arXiv 2305.14314）口径为 65B 达 ChatGPT 99.3%；"90%+"是课程量级表述 |
| 速度 | QLoRA 7B 在 4090 上 1-2 小时可完成 | 官方 benchmark 量级参考（10K 条数据档） |
| 生态 | LLaMA-Factory 支持 100+ 模型、多种微调方法 | LLaMA-Factory README（截至 2026-09） |

**微调是把基座变成"业务可用"的第一手段**；LLaMA-Factory 是这条路上最流行的统一工具
（一个 yaml 覆盖 100+ 模型 / 全参+LoRA+QLoRA+DoRA+GaLore / SFT+RM+DPO+KTO+ORPO）。

## 理论背景

### 问题引入：为什么需要微调？

预训练模型虽然强大，但有两个根本限制：

1. **知识截止**：预训练数据有截止日期，不知道最新信息
2. **任务适配**：预训练目标是"预测下一个 token"，不是"回答问题"或"遵循指令"

微调（Fine-Tuning）通过**在特定任务数据上继续训练**来弥补：

```
预训练:  "学习语言和知识"           → 通用能力
微调:    "学习特定任务和格式"       → 任务适配
对齐:    "学习人类偏好"             → 安全和有用
```

> 💡 **类比**：预训练像是读完大学，微调像是入职培训。大学教你通用知识，
> 但入职培训教你如何在这家公司工作。

### 数学推导：LoRA 的低秩分解

LoRA（Low-Rank Adaptation）的核心思想是：**用低秩矩阵近似权重更新**。

**问题设定：**
- 预训练权重：$W \in \mathbb{R}^{d \times k}$
- 全参微调更新：$\Delta W \in \mathbb{R}^{d \times k}$
- LoRA 更新：$\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

**为什么低秩就够（动机）：** 微调是"小改动"——多项实证（Aghajanyan et al. 2020，
LoRA 论文继承该假设）表明权重更新 $\Delta W$ 的**内在秩很低**：适配一个下游任务只需要
在少数几个"方向"上修改权重。LoRA 于是把"学一个 $d \times k$ 的大更新"换成
"学两个瘦矩阵的乘积"。

**推导过程：**

Step 1 全参微调的参数量：$W' = W + \Delta W$，$\Delta W$ 有 $d \times k$ 个参数。

Step 2 LoRA 的参数量：$\Delta W = BA$，$B$ 贡献 $d \times r$、$A$ 贡献 $r \times k$，共 $r \times (d + k)$。

Step 3 压缩比：$\dfrac{d \times k}{r \times (d + k)}$。示例 $d = k = 4096$，$r = 8$：

$$\frac{4096 \times 4096}{8 \times (4096 + 4096)} = \frac{16{,}777{,}216}{65{,}536} = 256$$

**性质：**
- LoRA 不增加推理延迟（合并后 $W' = W + \frac{\alpha}{r}BA$）
- LoRA 的 A 用高斯初始化（实现常取 $\mathcal{N}(0,\ 1/r)$，即 `randn/√r`，本课脚本同口径），
  B 用零初始化（训练开始时 $\Delta W = BA = 0$）
- $\alpha/r$ 是缩放因子，控制 LoRA 的"学习强度"——除以 $r$ 是为了让有效强度与 $r$
  解耦（$BA$ 的尺度会随 $r$ 漂移），调 $r$ 时不必重调 $\alpha$ 和学习率

### 历史脉络：微调方法演进

```
2018: 全参微调（Full Fine-Tuning）
  ↓ 显存开销大
2019: Adapter（Houlsby et al.）
  ↓ 增加推理延迟
2021: LoRA（Microsoft）
  ↓ 不增加推理延迟
2023: QLoRA（Dettmers et al.）
  ↓ 4bit 量化 + LoRA
2024: DoRA/GaLore/...
  ↓ 更多优化方法
```

**关键论文：**
- LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- QLoRA: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- DPO: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

## 📦 环境与版本策略

```bash
# 独立 venv（不污染课程主环境）；策略：跟随 latest（耦合轻）
uv venv .venv-lf && source .venv-lf/bin/activate
git clone https://github.com/hiyouga/LlamaFactory && cd LlamaFactory
pip install -e ".[torch,metrics]"      # python ≥3.11；flash-attn 可选（装不上可跳过）
llamafactory-cli version               # 验证
```

### 模型 / 数据从哪来（下载依赖档位）

本 Part 只有脚本 01 零下载；02 章的 CLI 链路需要联网拉权重与数据集
（首次运行自动下载到 `~/.cache/huggingface`）：

| 步骤 | 首次要下载什么 | 大致磁盘占用 |
|---|---|---|
| 脚本 01（手写） | 无（玩具数据内存生成） | 0 |
| 02 章 §1-§2 identity / WebUI | `Qwen/Qwen2.5-0.5B-Instruct` + `identity`、`alpaca_gpt4_zh` 数据集 | ~1GB + MB 级 |
| 02 章 §3-§5 QLoRA / export / DPO | `Qwen/Qwen2.5-7B-Instruct` | ~15GB |

- 国内网络建议先设 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- **离线降级边界**：无网时只能完成脚本 01（零下载）；02 章各步骤在对应权重/数据
  下载完成前不可跑，缺模型会卡在 HF 拉取报错（设好 `HF_ENDPOINT` 再重试）。

| 硬件 | 可做什么（官方文档数字） |
|---|---|
| CPU | identity 小模型 LoRA 演示（脚本 01 的手写版无需任何安装） |
| 1×24GB（4090） | **QLoRA 7B（官方 4bit=6GB）**、LoRA bf16 7B（16GB）、DPO-LoRA |
| 多卡 | `FORCE_TORCHRUN=1` 起 DDP / DeepSpeed ZeRO-3（见 Part 10 知识） |

## 📈 学习地图（由点到面）

```
手写 LoRA SFT（脚本01：看得见每一行）         ← 点
   ↓ "这几百行被 yaml 的哪几个字段自动化了？"
LLaMA-Factory 最小闭环（identity 0.5B）      ← 线
   ↓ 真实模型 + 真实数据
QLoRA 7B → export 合并 → chat/api           ← 面
   ↓ 偏好对齐
DPO-LoRA（呼应 Part 8 03 章）                →  面试/工作就绪
```

## 📝 课后作业

每章末尾有思考题（`<details>` 折叠答案）。全部学完后：

👉 [Assignment 12](../../../assignments/assignment_12/)

## 🔗 相关资源

- 🐙 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)（官方 docs 与 examples/yaml 是最好的教程）
- 🐙 [unsloth](https://github.com/unslothai/unsloth)（75.2k★，截至 2026-09；单卡加速微调，免费 Colab notebook 丰富——作业对照用）
- 🐙 [huggingface/peft](https://github.com/huggingface/peft)（LoRA 底层库）
- 📄 [LoRA 论文](https://arxiv.org/abs/2106.09685) · [QLoRA 论文](https://arxiv.org/abs/2305.14314)

---

[← 上一章：Part 11 verl 对齐实战](../../Part11_alignment_verl/tutorial/README.md) | [下一章：Part 13 数据工程 →](../../Part13_data_engineering/tutorial/README.md)
