# 02 — LLaMA-Factory 工作流：identity → QLoRA 7B → export → DPO-LoRA

> 🧭 手写完成（01 章），现在把同一流程交给工具，规模放大到 **7B 真实模型**。
> 本章是一条**可直接照抄的命令流水线**（每步标注预期产物与耗时，4090 实测量级），
> 环境按 README 的版本策略：独立 venv 跟随 LLaMA-Factory latest。

## 学习目标

完成本章后，你将能够：

- ✅ **配置** LLaMA-Factory 的 yaml 文件并理解每个字段的含义
- ✅ **完成** LoRA SFT → QLoRA 7B → export → chat/api 的生产链路
- ✅ **用** "官方显存数字 + Part 10 账本"预估自己的训练能不能跑
- ✅ **用** 工具跑 DPO-LoRA 并读懂 rewards/margins 曲线
- ✅ **把** 任何微调 yaml 翻译成"六步管线"来 debug

## 前置知识

**必须掌握：**
- **01 章**：六步管线与 yaml 字段映射（本章每个 yaml 字段都引用它）
- **Part 8 03 章**：DPO（本章用工具跑一遍）

## 理论背景

### 问题引入：为什么需要 QLoRA？

LoRA 虽然已经很省显存，但 7B 模型的底座权重仍然需要 ~14GB（fp16）。
QLoRA 通过**量化底座权重**来进一步节省显存：

```
LoRA:   底座 fp16 (14GB) + LoRA bf16 (~20MB) = ~14GB
QLoRA:  底座 4bit (3.5GB) + LoRA bf16 (~20MB) = ~3.5GB
```

> 💡 **类比**：LoRA 像是只换几件家具，QLoRA 像是把家具换成折叠的。
> 平时折叠起来省空间，用的时候展开。

### 数学推导：QLoRA 的量化过程

**问题设定：**
- 底座权重：W ∈ R^{d×k}（fp16，每个参数 2 字节）
- 量化后：W_q ∈ R^{d×k}（4bit，每个参数 0.5 字节）

**推导过程：**

```
Step 1: fp16 存储
  每个参数 2 字节
  7B 模型 = 7×10^9 × 2 = 14GB

Step 2: 4bit 量化
  每个参数 0.5 字节
  7B 模型 = 7×10^9 × 0.5 = 3.5GB

Step 3: 双重量化
  量化常数也量化（每 64 个参数共享一个量化常数）
  额外节省 ~0.37GB
  总计: ~3.5GB + 0.37GB ≈ 3.87GB
```

**性质：**
- NF4（NormalFloat4）是专门为正态分布设计的 4bit 格式
- 双重量化（Double Quantization）把量化常数也量化，进一步节省显存
- LoRA 的 A/B 保持 bf16 训练，不被量化

## 代码实现

### 0. 环境（一次性）

```bash
uv venv .venv-lf && source .venv-lf/bin/activate
git clone https://github.com/hiyouga/LlamaFactory && cd LlamaFactory
pip install -e ".[torch,metrics]"
llamafactory-cli version   # 能打印版本即 OK
```

### 1. 最小闭环：identity LoRA SFT（小模型，小时级内出结果）

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

### 2. WebUI：LLaMA Board（建立配置直觉）

```bash
llamafactory-cli webui    # 浏览器打开，零代码配置并启动训练
```

用途不是生产训练，而是**把字段玩一遍**：改 `lora_rank`/`cutoff_len`/`learning_rate` 时
页面会实时估算显存——把 01 章的手写账本和 GUI 的估算互相印证。

### 3. QLoRA 7B（4090 主菜，官方数字：4bit 7B ≈ 6GB）

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

### 4. 合并与部署

```bash
llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen7b-qlora --export_dir models/qwen7b-merged \
  --export_size 4 --export_legacy_format false
# 合并 = 手写版的 W += (α/r)·BA（01 章 merge_lora）；之后是普通模型：
llamafactory-cli chat --model_name_or_path models/qwen7b-merged
llamafactory-cli api --model_name_or_path models/qwen7b-merged   # OpenAI 兼容服务
```

### 5. DPO-LoRA（偏好对齐，呼应 Part 8 03 章）

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml   # 文件名以版本为准
# 数据: UltraFeedback 的 (prompt, chosen, rejected) 三元组（Part 8 03 章同款语义）
# 关键字段: pref_beta: 0.1（= DPO 的 β）、pref_loss: sigmoid（标准 DPO）
```

预期现象（记录进面经）：DPO 后 `rewards/chosen` 上升、`rewards/margins` 变正且扩大；
lr 用 5e-6 量级（比 SFT 更小——Part 7 05 章"越靠后 lr 越小"规律的又一实证）。

### 6. 手写 vs 工具：一张总账

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

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：数据格式不匹配

**症状：**
```
ValueError: Template qwen does not exist
```

**原因：** template 名称不对，或模型不支持该 template

**解法：**
```bash
# 检查支持的 template
llamafactory-cli template list

# 使用正确的 template
--template default  # 或 auto
```

#### 错误 2：显存不足 (OOM)

**症状：**
```
CUDA out of memory. Tried to allocate 2.00 MiB
```

**原因：** cutoff_len 太大，或 batch_size 太大

**解法：**
```bash
# 减小 cutoff_len
--cutoff_len 512

# 减小 batch_size
--per_device_train_batch_size 1

# 启用 gradient checkpointing
--gradient_checkpointing true
```

#### 错误 3：数据集不存在

**症状：**
```
ValueError: Dataset xxx does not exist
```

**原因：** 数据集名称不对，或未注册

**解法：**
```bash
# 检查支持的数据集
llamafactory-cli dataset list

# 注册自定义数据集
# 在 dataset_info.json 中添加数据集定义
```

#### 错误 4：export 失败

**症状：**
```
Error: adapter_name_or_path does not exist
```

**原因：** adapter 路径不对，或训练未完成

**解法：**
```bash
# 检查 adapter 是否存在
ls saves/qwen7b-qlora/

# 确保训练完成后再 export
```

### 性能数据（实测参考）

| 模型 | 方法 | 显存占用 | 训练时间 | 效果 |
|------|------|----------|----------|------|
| 0.5B | LoRA r=8 | ~4GB | ~10min | identity 学会 |
| 7B | LoRA bf16 r=8 | ~16GB | ~2h | ~95% 全参效果 |
| 7B | QLoRA 4bit r=8 | ~6GB | ~2h | ~93% 全参效果 |
| 7B | DPO-LoRA | ~8GB | ~3h | rewards/margins 改善 |

> 📊 数据来源：LLaMA-Factory 官方 benchmark + 本课开发机实测（RTX 4090，torch 2.5.1）

### 常见陷阱

#### 陷阱 1：数据格式不匹配

**症状：** 训练时 loss 不下降，或输出格式混乱

**原因：** 数据格式与模型的 chat template 不匹配

**解法：** 检查数据格式是否符合模型的 chat template（如 Qwen 用 `<|im_start|>`）

#### 陷阱 2：显存估算不准

**症状：** 训练时 OOM，但估算应该够

**原因：** 没有考虑激活值的显存开销

**解法：** 使用 WebUI 的显存估算功能，或手动计算

#### 陷阱 3：LoRA rank 选择不当

**症状：** 效果不好，或显存不足

**原因：** rank 太小（表达能力不足）或太大（显存开销大）

**解法：** 从 r=8 开始，根据效果和显存调整

### 最佳实践

#### 配置推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lora_rank` | 8-64 | 从 8 开始，根据效果调整 |
| `lora_alpha` | 2 × lora_rank | 常见配置 |
| `learning_rate` | 1e-4 ~ 5e-5 | LoRA 学习率通常比全参大 |
| `cutoff_len` | 512-2048 | 根据任务调整 |
| `batch_size` | 1-4 | 根据显存调整 |
| `num_train_epochs` | 3-5 | 根据数据量调整 |

#### 调试流程

1. 先用小模型（0.5B）跑通流程
2. 检查数据格式是否正确
3. 检查显存是否足够
4. 逐步增大模型和数据量

## 学完本部分你能...

- ✅ 独立完成 LoRA SFT → QLoRA 7B → export → chat/api 的生产链路
- ✅ 用"官方显存数字 + Part 10 账本"预估自己的训练能不能跑
- ✅ 用工具跑 DPO-LoRA 并读懂 rewards/margins 曲线
- ✅ 把任何微调 yaml 翻译成"六步管线"来 debug
- ✅ 识别数据格式不匹配、显存不足等常见陷阱

**概念检验**

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

<details>
<summary>Q3: DPO 的 rewards/margins 曲线怎么读？什么时候算"收敛"？</summary>

A: rewards/chosen 应上升，rewards/rejected 应下降，margins 应变正且扩大。
收敛标志：margins 稳定在 0.5-2.0 之间，不再明显波动。

</details>

**动手实践**

<details>
<summary>练习 1: 估算显存占用</summary>

**任务：** 估算 7B 模型 QLoRA 微调的显存占用。

**验收标准：**
- [ ] 考虑参数、梯度、优化器状态
- [ ] 考虑 4bit 量化
- [ ] 结果与官方数字（~6GB）接近

**步骤提示：**
```python
def estimate_qlora_memory(model_params_B=7, lora_rank=8, batch_size=1, seq_len=512):
    """
    Steps:
        1. 计算底座权重显存（4bit）
        2. 计算 LoRA 参数显存（bf16）
        3. 计算梯度显存
        4. 计算优化器显存
        5. 计算激活显存
        6. 汇总
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 2: 转换 yaml 字段</summary>

**任务：** 将以下命令转换为 yaml 配置文件。

**命令：**
```bash
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset alpaca_gpt4_zh \
  --template qwen --finetuning_type lora \
  --lora_target all --lora_rank 16 --lora_alpha 32 \
  --output_dir saves/qwen7b-lora \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-4 --num_train_epochs 3.0
```

**验收标准：**
- [ ] 所有字段都正确转换
- [ ] 格式符合 LLaMA-Factory 规范
- [ ] 可以直接使用

</details>

## 📝 课后作业

完成本章后，去 Assignment 12 完成练习：

👉 [Assignment 12](../../../assignments/assignment_12/)

## 下一步

数据从哪来、怎么清洗？Part 13 用手写 MinHash + Data-Juicer 回答（RL 基建见 Part 11）。

---

[← 上一章](01_handwritten_sft_lora.md) | [Part 12 README](README.md)
