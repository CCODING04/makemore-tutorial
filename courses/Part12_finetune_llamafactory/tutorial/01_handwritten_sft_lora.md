# 01 — 手写 LoRA SFT：LLaMA-Factory 自动化的到底是什么

> 🧭 工具的价值只有在你**知道它替你做了什么**时才能兑现。本章用 ~250 行把
> LLaMA-Factory 一个 yaml 背后的完整流水线手写一遍（跑 [scripts/01_handwritten_sft_lora.py](../scripts/01_handwritten_sft_lora.py)），
> 然后给出**逐字段对照表**——之后看任何微调 yaml，你都能指出"每个字段对应哪几行代码"。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** LoRA SFT 的完整流水线（5 个步骤）
- ✅ **解释** 每个步骤对应的 LLaMA-Factory yaml 字段
- ✅ **识别** "注入后忘搬 device" 等常见陷阱
- ✅ **说出** 手写版与工具版的真实差距

## 前置知识

**必须掌握：**
- **Part 8 08 章**：LoRALinear 的 A/B 初始化与 α/r（本章直接复用）
- **Part 8 02 章**：prompt masking（labels=-100）

## 理论背景

### 问题引入：为什么需要 LoRA 而不是全参微调？

全参微调（Full Fine-Tuning）虽然效果好，但有两个根本限制：

1. **显存开销大**：7B 模型全参微调需要 ~120GB 显存（参数 + 梯度 + 优化器）
2. **容易过拟合**：小数据集上全参微调容易"忘记"预训练知识

LoRA（Low-Rank Adaptation）通过**只训练低秩矩阵**来弥补：

```
全参微调:  W' = W + ΔW          # ΔW ∈ R^{d×k}，参数量 = d×k
LoRA:      W' = W + (α/r)·BA    # B ∈ R^{d×r}, A ∈ R^{r×k}，参数量 = r×(d+k)

示例：d=4096, k=4096, r=8
全参: 16,777,216 参数
LoRA: 65,536 参数（压缩 256 倍）
```

> 💡 **类比**：全参微调像是重新装修整栋房子，LoRA 像是只换几件家具。
> 效果差不多，但成本低很多。

### 数学推导：LoRA 的初始化和缩放

**问题设定：**
- 预训练权重：W ∈ R^{d×k}
- LoRA 矩阵：B ∈ R^{d×r}, A ∈ R^{r×k}
- 缩放因子：α（学习强度）、r（秩）

**推导过程：**

```
Step 1: 初始化
  A ~ N(0, σ²)  # 高斯初始化
  B = 0          # 零初始化

  性质：训练开始时 ΔW = BA = 0，不改变预训练权重

Step 2: 前向传播
  h = Wx + (α/r)·BAx

  其中：
  - Wx 是预训练的输出
  - (α/r)·BAx 是 LoRA 的增量

Step 3: 合并（推理时）
  W' = W + (α/r)·BA

  性质：合并后推理零额外开销
```

**关键洞察：**
- α/r 控制 LoRA 的"学习强度"：α 越大，LoRA 影响越大
- r 越大，LoRA 的表达能力越强，但参数也越多
- 实践中 r=8-64，α=2r 是常见配置

## 代码实现

### 微型管线五步（玩具任务："回声指令"——回应复述指令的第一个词）

运行 [scripts/01_handwritten_sft_lora.py](../scripts/01_handwritten_sft_lora.py) 验证以下步骤。

```
[0] 基座预热          —— "预训练过的"玩具基座（结构就绪即可）
[1] LoRA 注入         —— 4 层 MLP Linear，可训练 6,144/200,664（3.1%）
[2] SFT 训练          —— loss 3.572 → 0.076（chat 格式 + 任务映射都学会了）
[3] 推理验证          —— chat 格式 3/3 正确；回声任务 2/3（400 步玩具训练的
                         正常欠拟合——真任务上这个位置由更多数据/步数兜底）
[4] 合并（merge）     —— BA 并回 W（精确加法），同一批 prompt 前后行为一致，零额外开销
```

> 📝 以上为脚本真实输出（RTX 4090 / CPU 均可复现，~3 秒）。

### 形状追踪：LoRA 注入过程

口径说明：下图与脚本同口径（`apply_lora(model, r=4, alpha=8.0)`），数字可直接对上
脚本 `[1]` 的输出——每层 1,536 × 注入 4 层（2 个 Block × MLP 两个 Linear）= **6,144**。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LoRA 注入过程（以 MLP 的第一个 Linear 为例，r=4, α=8）                     │
│                                                                             │
│  原始层: Linear(in_features=96, out_features=288)                          │
│  权重 W: (288, 96)                                                          │
│                                                                             │
│  注入后:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LoRALinear(                                                         │   │
│  │    base_layer: Linear(96, 288)  # 冻结                               │   │
│  │    lora_A: (4, 96)              # 可训练，A ~ N(0,1)/√r              │   │
│  │    lora_B: (288, 4)             # 可训练，B 零初始化                 │   │
│  │    scaling: α/r = 8/4 = 2.0     # 缩放因子                           │   │
│  │  )                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  前向传播:                                                                   │
│  x: (batch, seq_len, 96)                                                    │
│    ↓ base_layer                                                             │
│  base_out: (batch, seq_len, 288)                                            │
│    ↓ lora_A（x @ A.T: (…,96) @ (96,4)）                                     │
│  lora_out: (batch, seq_len, 4)                                              │
│    ↓ lora_B（… @ B.T: (…,4) @ (4,288)）                                     │
│  lora_out: (batch, seq_len, 288)                                            │
│    ↓ scaling (α/r)                                                          │
│  lora_out: (batch, seq_len, 288) * 2.0                                      │
│    ↓ addition                                                               │
│  output: base_out + lora_out                                                │
│                                                                             │
│  可训练参数: 4×96 + 288×4 = 384 + 1152 = 1,536（本层）                      │
│  原始参数: 288×96 = 27,648                                                  │
│  压缩比: 27,648 / 1,536 = 18 倍                                             │
│  全模型: 1,536 + (MLP 第二个 Linear 同为 1,536) = 3,072/Block               │
│          × 2 个 Block = 6,144 —— 正是脚本 [1] 打印的可训练参数               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 五步与 yaml 字段的对照（本章核心产出）

| 手写函数 | LLaMA-Factory yaml / CLI 字段 |
|---|---|
| `build_sample()` 的 prompt 拼接 + `labels[:n_prompt]=-100` | `template:` + `train_on_prompt: false` |
| `apply_lora(r, alpha)` 注入 MLP Linear | `lora_target:` / `lora_rank:` / `lora_alpha:` |
| `make_sft_data()` 的 (instruction, response) | `dataset:` + `dataset_info.json` 的列映射 |
| `sft_train()` 的 AdamW/lr/步数 | `learning_rate:` / `num_train_epochs:` / `per_device_train_batch_size:` |
| `pad_batch()` 的右侧 padding + -100 填充 | `cutoff_len:`（packing=多条样本拼进定长序列免 padding 浪费，与 padding 二选一） |
| `merge_lora()` 的 `W += (α/r)·BA` | `llamafactory-cli export` |

- 🔑 **读 yaml 的新能力**：`lora_target: all` = "所有 Linear 都注入"；`lora_dropout` 是
  BA 旁路上的 dropout；`train_on_prompt: true` = 把 masking 撤掉（Part 8 02 章
  讲过为什么不这么做）。

### 手写版 vs 工具版的真实差距

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

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：注入后忘搬 device

**症状：**
```
RuntimeError: Expected all tensors to be on the same device
```

**原因：** LoRA 新建的 A/B 参数默认在 CPU，注入之后必须再 `.to(device)`

**解法：**
```python
# 注入 LoRA
apply_lora(model, r=8, alpha=16)

# 必须搬回 GPU！
model = model.to(device)
```

#### 错误 2：dtype 不一致

**症状：**
```
RuntimeError: expected scalar type Half but found Float
```

**原因：** 模型是 fp16，但 LoRA 参数是 fp32

**解法：**
```python
# 注入 LoRA 后统一 dtype
model = model.to(device, dtype=torch.float16)
```

#### 错误 3：梯度未正确 mask

**症状：** loss 不下降，或 loss 为 0

**原因：** labels 没有正确设置 -100（prompt 部分不应该计算 loss）

**解法：**
```python
# 确保 prompt 部分的 labels 是 -100
labels[:n_prompt] = -100
```

### 性能数据（实测参考）

| 模型 | 方法 | 可训练参数 | 显存占用 | 训练时间 | 效果 |
|------|------|------------|----------|----------|------|
| 200K 玩具 | LoRA r=4 | 6,144 (3.1%) | ~1GB | <1min | 2/3 正确 |
| 7B | 全参微调 | 7B (100%) | ~120GB | ~10h | 基准 |
| 7B | LoRA r=8 | ~20M (0.3%) | ~16GB | ~2h | ~95% 全参效果 |
| 7B | QLoRA r=8 | ~20M (0.3%) | ~6GB | ~2h | ~93% 全参效果 |

> 📊 数据来源：LLaMA-Factory 官方 benchmark + 本课开发机实测
> （RTX 4090，torch 2.6.0+cu124；玩具行 = 脚本 01 复跑，7B 行 = 官方量级参考）

### 常见陷阱

#### 陷阱 1：数据格式不匹配

**症状：** 训练时 loss 不下降，或输出格式混乱

**原因：** 数据格式与模型的 chat template 不匹配

**解法：** 检查数据格式是否符合模型的 chat template（如 Qwen 用 `<|im_start|>`）

#### 陷阱 2：显存不足 (OOM)

**症状：** `CUDA out of memory`

**原因：** cutoff_len 太大，或 batch_size 太大

**解法：** 减小 cutoff_len（如 512）或 batch_size（如 1）

#### 陷阱 3：LoRA rank 选择不当

**症状：** 效果不好，或显存不足

**原因：** rank 太小（表达能力不足）或太大（显存开销大）

**解法：** 从 r=8 开始，根据效果和显存调整

### 最佳实践

#### LoRA 配置推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lora_rank` | 8-64 | 从 8 开始，根据效果调整 |
| `lora_alpha` | 2 × lora_rank | 常见配置 |
| `lora_target` | all | 注入所有 Linear |
| `lora_dropout` | 0.05-0.1 | 防止过拟合 |
| `learning_rate` | 1e-4 ~ 5e-5 | LoRA 学习率通常比全参大 |

#### QLoRA 配置推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `quantization_bit` | 4 | NF4 量化 |
| `double_quantization` | true | 双重量化，省常数开销 |
| `quantization_type` | nf4 | NormalFloat4 格式 |

## 学完本章你能...

- ✅ 说出 LoRA SFT 流水线的五个环节及各自对应的 yaml 字段
- ✅ 解释 merge 之后为什么推理零开销
- ✅ 警觉"注入后忘搬 device"这类扩展模块的经典 bug
- ✅ 画出 LoRA 注入的形状追踪图
- ✅ 识别数据格式不匹配、显存不足等常见陷阱

**概念检验**

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

<details>
<summary>Q3: LoRA 的 A 和 B 分别用什么初始化？为什么这样设计？</summary>

A: A 用高斯初始化，B 用零初始化。这样训练开始时 ΔW = BA = 0，不改变预训练权重。
随着训练进行，B 逐渐非零，LoRA 开始"学习"任务特定的权重更新。

</details>

**动手实践**

<details>
<summary>练习 1: 实现 LoRA 注入函数</summary>

**任务：** 实现一个函数，给模型的 Linear 层注入 LoRA。

**验收标准：**
- [ ] 正确注入 lora_A 和 lora_B
- [ ] 正确设置 scaling = α/r
- [ ] 冻结原始权重
- [ ] 返回可训练参数数量

**步骤提示：**
```python
def apply_lora(model, r=8, alpha=16):
    """
    Steps:
        1. 遍历模型的所有模块
        2. 找到 Linear 层
        3. 替换为 LoRALinear
        4. 冻结原始权重
        5. 返回可训练参数数量
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 2: 实现 LoRA 合并函数</summary>

**任务：** 实现一个函数，将 LoRA 权重合并回原始权重。

**验收标准：**
- [ ] 正确计算 W' = W + (α/r)·BA
- [ ] 合并后删除 LoRA 参数
- [ ] 合并后推理结果不变

**步骤提示：**
```python
def merge_lora(model):
    """
    Steps:
        1. 遍历模型的所有模块
        2. 找到 LoRALinear 层
        3. 计算 W' = W + (α/r)·BA
        4. 替换原始权重
        5. 删除 LoRA 参数
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 3: 实现显存估算函数</summary>

**任务：** 实现一个函数，估算 LoRA 微调的显存占用。

**验收标准：**
- [ ] 输入：模型参数量、LoRA rank、batch_size、seq_len
- [ ] 输出：预估显存占用（GB）
- [ ] 考虑参数、梯度、优化器状态

**步骤提示：**
```python
def estimate_lora_memory(
    model_params_B: float,  # 模型参数量（单位：B）
    lora_rank: int,
    batch_size: int,
    seq_len: int,
) -> float:
    """
    估算 LoRA 微调的显存占用

    经验公式：
    - 参数: model_params_B * 2 bytes (fp16)
    - 梯度: model_params_B * 2 bytes (fp16)
    - 优化器: model_params_B * 8 bytes (Adam)
    - 激活: batch_size * seq_len * d_model * 4 bytes

    Steps:
        1. 计算参数显存
        2. 计算梯度显存
        3. 计算优化器显存
        4. 计算激活显存
        5. 汇总
    """
    # TODO: Implement
    pass
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 12 完成练习：

👉 [Assignment 12](../../../assignments/assignment_12/)

## 下一步

同样的流程交给工具：identity 数据集 → WebUI → QLoRA 7B → export → DPO-LoRA。

👉 [02 — LLaMA-Factory 工作流](02_llamafactory_workflow.md)
