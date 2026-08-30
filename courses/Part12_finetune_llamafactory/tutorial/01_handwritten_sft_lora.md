# 01 — 手写 LoRA SFT：LLaMA-Factory 自动化的到底是什么

> 🧭 工具的价值只有在你**知道它替你做了什么**时才能兑现。本章用 ~250 行把
> LLaMA-Factory 一个 yaml 背后的完整流水线手写一遍（跑 [scripts/01_handwritten_sft_lora.py](../scripts/01_handwritten_sft_lora.py)），
> 然后给出**逐字段对照表**——之后看任何微调 yaml，你都能指出"每个字段对应哪几行代码"。

## 📖 前置知识

- **Part 8 08 章**：LoRALinear 的 A/B 初始化与 α/r（本章直接复用）
- **Part 8 02 章**：prompt masking（labels=-100）

## 1. 微型管线六步（玩具任务："回声指令"——回应复述指令的第一个词）

```
[0] 基座预热          —— "预训练过的"玩具基座（结构就绪即可）
[1] LoRA 注入         —— 4 层 MLP Linear，可训练 6,144/200,664（3.1%）
[2] SFT 训练          —— loss 3.572 → 0.076（chat 格式 + 任务映射都学会了）
[3] 推理验证          —— chat 格式 3/3 正确；回声任务 2/3（400 步玩具训练的
                         正常欠拟合——真任务上这个位置由更多数据/步数兜底）
[4] 合并（merge）     —— BA 并回 W（精确加法），同一批 prompt 前后行为一致，零额外开销
```

五步与 yaml 字段的对照（[2] 内含 padding 环节）（**本章的核心产出**）：

| 手写函数 | LLaMA-Factory yaml / CLI 字段 |
|---|---|
| `build_sample()` 的 prompt 拼接 + `labels[:n_prompt]=-100` | `template:` + `train_on_prompt: false` |
| `apply_lora(r, alpha)` 注入 MLP Linear | `lora_target:` / `lora_rank:` / `lora_alpha:` |
| `make_sft_data()` 的 (instruction, response) | `dataset:` + `dataset_info.json` 的列映射 |
| `sft_train()` 的 AdamW/lr/步数 | `learning_rate:` / `num_train_epochs:` / `per_device_train_batch_size:` |
| `pad_batch()` 的右侧 padding + -100 填充 | `cutoff_len:`（packing=多条样本拼进定长序列免 padding 浪费，与 padding 二选一） |
| `merge_lora()` 的 `W += (α/r)·BA` | `llamafactory-cli export` |

- 🔑 **读 yaml 的新能力**：`lora_target: all` = "所有 Linear 都注入"；`lora_dropout` 是
  BA 旁路上的 dropout；`train_on_prompt: true` = 把 masking 撤掉（我们 02 章
  讲过为什么不这么做）。
- ⚠️ **全课程第二次踩到的坑**（Part 8 10 章第一次）：注入 LoRA 新建的 A/B 参数默认在 CPU，
  **注入之后必须再 `.to(device)`**——报错形态是"Expected all tensors on same device"。

## 2. 手写版 vs 工具版的真实差距

| 维度 | 手写（本脚本） | LLaMA-Factory |
|---|---|---|
| 数据 | 20 条玩具指令，内存 list | 100+ 数据集目录、json/jsonl/sharegpt 格式自动识别 |
| 模型 | 200K 玩具 GPT | 100+ HF 模型开箱即用（Qwen/Llama/GLM/DeepSeek…） |
| 训练技巧 | 固定 lr、无 warmup/梯度累积/断点续训 | 全部内置（对应附录 D 的"bells & whistles"） |
| 量化 | 无 | QLoRA 4bit（NF4）一行开关 |
| 多卡 | 无 | DDP/ZeRO-3/FSDP 一键（`FORCE_TORCHRUN=1`） |
| 产物 | 内存里的权重 | checkpoint + `export` 合并 + `chat`/`api` 部署 |

- 🔑 结论：**手写教会你"字段↔代码"的映射，工具给你工程完备性**。两者都过一遍，
  你就同时具备"改得动工具"（debug、自定义）和"用得对工具"（选字段、估显存）的能力。

## 学完本部分你能...

- ✅ 说出 LoRA SFT 流水线的六个环节及各自对应的 yaml 字段
- ✅ 解释 merge 之后为什么推理零开销
- ✅ 警觉"注入后忘搬 device"这类扩展模块的经典 bug

**课后练习**

<details>
<summary>Q1: 为什么本脚本严格冻结 lm_head？放开它（全参 + LoRA 混合）会有什么变化？</summary>
A: 放开 lm_head 是常见的"LoRA + 部分全参"混合档位（embed/lm_head 可训练），任务适配
通常更快，但可训练参数与优化器显存随之上升。LlamaFactory 有 `additional_target: trainables`
字段控制这一档。玩具实验：放开 lm_head 后 400 步内 acc 应更稳。
</details>

<details>
<summary>Q2: 如果把 lora_rank 从 4 提到 64，可训练参数变成多少？什么时候值得？</summary>
A: 每层从 r·(d+k)=4·(96+288)×2 变为 64·384×2 的量级——参数比例从 3.1% 升到两位数。
值得的场景：任务与预训练分布差距大、或数据量大（几十万条以上）；小任务 r=8/16 通常够。
</details>

## 📝 课后作业

👉 [Assignment 12](../../../assignments/assignment_12/)

## 下一步

同样的流程交给工具：identity 数据集 → WebUI → QLoRA 7B → export → DPO-LoRA。

👉 [02 — LLaMA-Factory 工作流](02_llamafactory_workflow.md)
