# 02 — LLaMA-Factory 工作流：identity → QLoRA 7B → export → DPO-LoRA

> 🧭 手写完成（01 章），现在把同一流程交给工具，规模放大到 **7B 真实模型**。
> 本章是一条**可直接照抄的命令流水线**（每步标注预期产物与耗时，4090 实测量级），
> 环境按 README 的版本策略：独立 venv 跟随 LLaMA-Factory latest。

## 📖 前置知识

- **01 章**：六步管线与 yaml 字段映射（本章每个 yaml 字段都引用它）
- **Part 8 03 章**：DPO（本章用工具跑一遍）

## 0. 环境（一次性）

```bash
uv venv .venv-lf && source .venv-lf/bin/activate
git clone https://github.com/hiyouga/LlamaFactory && cd LlamaFactory
pip install -e ".[torch,metrics]"
llamafactory-cli version   # 能打印版本即 OK
```

## 1. 最小闭环：identity LoRA SFT（小模型，小时级内出结果）

LLaMA-Factory 自带 `identity` 数据集（教模型"我是谁"），最适合第一次跑通：

```bash
# 官方示例 yaml：examples/train_lora/qwen_lora_sft.yaml 改两行即可
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset identity,alpaca_gpt4_zh \
  --template qwen --finetuning_type lora \
  --lora_target all --lora_rank 8 --lora_alpha 16 \
  --output_dir saves/qwen05-identity --per_device_train_batch_size 4 \
  --learning_rate 5e-5 --num_train_epochs 3.0 --plot_loss true
# ⚠️ yaml 字段以你安装版本的 examples/ 实际文件名为准（仓库迭代快，
#    例如 qwen_lora_sft.yaml 在新版已更名 qwen3_lora_sft.yaml）
```

**对照 01 章六步**：`--template` = build_sample；`--lora_target all` = 注入所有 Linear；
`--train_on_prompt` 默认 false = prompt masking。产物：`saves/qwen05-identity/`（adapter
权重 + loss 图）。

## 2. WebUI：LLaMA Board（建立配置直觉）

```bash
llamafactory-cli webui    # 浏览器打开，零代码配置并启动训练
```

用途不是生产训练，而是**把字段玩一遍**：改 `lora_rank`/`cutoff_len`/`learning_rate` 时
页面会实时估算显存——把 01 章的手写账本和 GUI 的估算互相印证。

## 3. QLoRA 7B（4090 主菜，官方数字：4bit 7B ≈ 6GB）

```bash
llamafactory-cli train examples/train_qlora/qwen3_lora_sft_otfq.yaml
# （文件名以安装版本 examples/ 为准；关键这 4 个字段——对照手写版"缺的量化"）：
#   quantization_bit: 4          ← NF4 底座（QLoRA 的 Q；NF4=4-bit NormalFloat 网格量化格式）
#   finetuning_type: lora        ← 只训 BA
#   double_quantization: true    ← 双重量化：把每组的量化常数 scale 再量化一遍，省常数开销
#   ⚠️ 记得加 --output_dir saves/qwen7b-qlora（§4 export 要用这个路径）
```

预期：7B 模型 + batch 1-2，显存 6-10GB（4090 余量充足），10K 条数据 1-2 小时量级。
**观察点**：`nvidia-smi` 里权重本体常驻 ~4GB（4bit），训练波动部分来自梯度/优化器——
**只有 BA 有梯度**，这正是 Part 8 08 章"LoRA 省的是优化器+梯度"的实证。

## 4. 合并与部署

```bash
llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen7b-qlora --export_dir models/qwen7b-merged \
  --export_size 4 --export_legacy_format false
# 合并 = 手写版的 W += (α/r)·BA（01 章 merge_lora）；之后是普通模型：
llamafactory-cli chat --model_name_or_path models/qwen7b-merged
llamafactory-cli api --model_name_or_path models/qwen7b-merged   # OpenAI 兼容服务
```

## 5. DPO-LoRA（偏好对齐，呼应 Part 8 03 章）

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml   # 文件名以版本为准
# 数据: UltraFeedback 的 (prompt, chosen, rejected) 三元组（Part 8 03 章同款语义）
# 关键字段: pref_beta: 0.1（= DPO 的 β）、pref_loss: sigmoid（标准 DPO）
```

预期现象（记录进面经）：DPO 后 `rewards/chosen` 上升、`rewards/margins` 变正且扩大；
lr 用 5e-6 量级（比 SFT 更小——Part 7 05 章"越靠后 lr 越小"规律的又一实证）。

## 6. 手写 vs 工具：一张总账

| 能力 | 01 章手写 | LLaMA-Factory |
|---|---|---|
| 模型规模 | 200K 玩具 | 7B（QLoRA 6GB）/ 100+ 模型 |
| 数据 | 20 条内存 list | 100+ 数据集 + 自定义 json/sharegpt |
| 量化 | 无 | 4bit NF4（QLoRA）一行 |
| 对齐 | — | DPO/KTO/ORPO/RM 全家桶 |
| 多卡 | 无 | DDP/ZeRO-3（FORCE_TORCHRUN） |
| 部署 | 内存权重 | export 合并 + chat/api |

> ⚠️ 工具不是魔法：跑挂时 90% 的问题在**数据格式**（template 不匹配、字段名不对）与
> **显存估算**（cutoff_len × batch）。这两个 debug 能力恰恰来自 01 章的手写对照。

## 学完本部分你能...

- ✅ 独立完成 LoRA SFT → QLoRA 7B → export → chat/api 的生产链路
- ✅ 用"官方显存数字 + Part 10 账本"预估自己的训练能不能跑
- ✅ 用工具跑 DPO-LoRA 并读懂 rewards/margins 曲线
- ✅ 把任何微调 yaml 翻译成"六步管线"来 debug

**课后练习**

<details>
<summary>Q1: QLoRA 里"4bit"量化的到底是什么？LoRA 的 A/B 也被量化了吗？</summary>
A: 只量化冻结的底座权重（NF4 存储）；LoRA 的 A/B 保持 bf16/fp16 训练——
"4bit 底座 + 高精度小适配器"正是 QLoRA 的名字含义。这也是它省显存的来源：
7B×0.5B≈3.5GB 的底座 + MB 级的可训练部分。
</details>

<details>
<summary>Q2: export 合并时如果忘了先 CPU 化或 dtype 不一致会怎样？生产上为什么不合并的场景也存在？</summary>
A: dtype 不一致会静默精度损失或报错（fp16 底座 + bf16 BA 要先统一）。不合并的场景：
多租户动态切换适配器（vLLM multi-LoRA）——保留 adapter、按请求挂载更省显存。
</details>

## 📝 课后作业

👉 [Assignment 12](../../../assignments/assignment_12/)

## 下一步

数据从哪来、怎么清洗？Part 13 用手写 MinHash + Data-Juicer 回答（RL 基建见 Part 11）。

👉 [Part 13 数据工程](../../Part13_data_engineering/tutorial/README.md)
