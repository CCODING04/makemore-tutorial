# Assignment 12：微调实战（LLaMA-Factory）

> 对应 Part 12 教程（[01 手写管线](../../courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.md) / [02 工作流](../../courses/Part12_finetune_llamafactory/tutorial/02_llamafactory_workflow.md)）。
> 四道题全部纸笔/纯 Python 可完成；实验题跑 02 章的工具链。

## 题目（实现 `finetune_exercises.py` 后 `python test_finetune_exercises.py`）

1. **LoRA 参数账**（30 分）：给定被注入层 dims，算可训练参数 Σ r·(out+in) 及占全参比例
2. **合并数学**（30 分）：`W' = W + (α/r)·B@A`，并验证"合并前后前向输出一致"（merge 的正确性）
3. **B 零初始化**（20 分）：证明 B=0 时起点 ΔW=0（Frobenius 范数）——"起点无损"的数学表述
4. **QLoRA 显存账**（20 分）：底座量化存储（params × bits/8）+ 可训练部分（×12B/参数）

## 实验题（观测型，写 3-5 行结论进面经）

- 跑 02 章工具链：QLoRA 7B × identity 数据集，记录显存曲线（对照官方 6GB 数字）
- `lora_rank` 8 → 64：记录可训练参数比例与显存变化（呼应题 1 的公式）

## 🎯 面试直通车

- "LoRA 省的是什么显存？"——优化器状态+梯度；底座本体仍需 bf16/4bit 常驻
- "为什么 B 初始化 0？"——起点无损（ΔW=0）；"两个都零会怎样？"——梯度全零学不动
- "合并后为什么零开销？"——BA 并回 W；不合并的场景=多租户动态挂 adapter（vLLM multi-LoRA）
- "QLoRA 的 Q 量化谁？"——冻结底座；A/B 保持高精度训练
