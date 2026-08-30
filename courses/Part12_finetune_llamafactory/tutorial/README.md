# Part 12: 微调实战 — LLaMA-Factory（LoRA / QLoRA / DPO 全流程）

> 🧭 Part 8 用几十行手写了 LoRA 的原理；本部分把同样的技能放大到**工业工具**：
> 用 LLaMA-Factory 在真实 7B 模型上走完 LoRA SFT → QLoRA → 合并导出 → DPO 的完整工作流。
> 学完你能独立承担"把一个开源基座调到业务任务上"的工程任务。
> 主源：[hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)（74.4k，Apache-2.0）

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [手写 LoRA SFT：工具自动化的到底是什么](01_handwritten_sft_lora.md) | chat template+masking+LoRA 注入+训练循环 手写一遍（工具的"内部透视"） | `01` |
| 02 | [LLaMA-Factory 工作流](02_llamafactory_workflow.md) | identity LoRA SFT → WebUI → QLoRA 7B → export 合并 → DPO-LoRA | `02` |

## 🧰 前置知识

- **Part 8 08 章**：从零 LoRA（A/B 初始化、α/r、注入位置）——本章工具的每个 yaml 字段都对应它
- **Part 8 02 章**：SFT 与 prompt masking；**Part 8 03 章**：DPO

## 🔗 在 LLM 链路中的位置

```
预训练(Part 7/13) → [本部分: 微调 SFT/LoRA/QLoRA/DPO] → 对齐 RL(Part 11) → 部署(Part 14)
```

微调是把基座变成"业务可用"的第一手段；LLaMA-Factory 是这条路上最流行的统一工具
（一个 yaml 覆盖 100+ 模型 / 全参+LoRA+QLoRA+DoRA+GaLore / SFT+RM+DPO+KTO+ORPO）。

## 📦 环境与版本策略

```bash
# 独立 venv（不污染课程主环境）；策略：跟随 latest（耦合轻）
uv venv .venv-lf && source .venv-lf/bin/activate
git clone https://github.com/hiyouga/LlamaFactory && cd LlamaFactory
pip install -e ".[torch,metrics]"      # python ≥3.11；flash-attn 可选（装不上可跳过）
llamafactory-cli version               # 验证
```

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

👉 [Assignment 12](../../../assignments/assignment_12/)

## 🔗 相关资源

- 🐙 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)（官方 docs 与 examples/yaml 是最好的教程）
- 🐙 [unsloth](https://github.com/unslothai/unsloth)（75.2k，单卡加速微调，免费 Colab notebook 丰富——作业对照用）
- 🐙 [huggingface/peft](https://github.com/huggingface/peft)（LoRA 底层库）
- 📄 [LoRA 论文](https://arxiv.org/abs/2106.09685) · [QLoRA 论文](https://arxiv.org/abs/2305.14314)

---

[← 上一章：Part 11 verl 对齐实战](../../Part11_alignment_verl/tutorial/README.md) | [下一章：Part 13 数据工程 →](../../Part13_data_engineering/tutorial/README.md)
