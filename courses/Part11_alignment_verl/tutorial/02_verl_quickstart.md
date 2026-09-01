# 02 — verl 快速上手：0.5B GRPO 实战

> 🧭 本章在 Docker 里跑通 verl 官方 quickstart（Qwen2.5-0.5B 在 GSM8K 上的 PPO/GRPO，
> 官方文档明确**单卡 ≥24GB**——4090 直接可跑），然后换成 GRPO、写自己的奖励函数、
> 扩到双卡。所有步骤的"手写对应物"都标注 01 章的编号。

## 学习目标

完成本章后，你将能够：

- ✅ **配置** Docker 环境并启动 verl 容器
- ✅ **跑通** 0.5B 模型的 PPO/GRPO 训练
- ✅ **读懂** verl 的关键日志（rollout/权重同步/优势计算）
- ✅ **编写** 自定义奖励函数并集成到 verl
- ✅ **扩展** 到双卡训练并理解 FSDP 分片

## 前置知识

**必须掌握：**
- **01 章**：概念映射表（本章配置行 = 01 章的手写代码）
- **Part 10**：FSDP（第 5 步的扩展实验用）

**建议回顾：**
- **Part 8 04 章**：GRPO 的数学原理

## 理论背景

### 为什么需要 Docker？

verl 与 vllm/torch/transformers 版本锁步耦合严重：

```
verl 0.4.0 → vllm 0.8.x → torch 2.5.x → transformers 4.45.x
           ↓ 版本冲突
裸 pip install → 依赖地狱 → 无法运行
```

**解决方案：** 官方 Docker 镜像锁定所有版本

```bash
docker pull verlai/verl:latest
# 内含：verl + vllm + torch + transformers + 所有依赖
# 版本完全兼容，开箱即用
```

### verl 的三角色架构

```
┌─────────────────────────────────────────────────────────────────┐
│  verl 三角色架构                                                │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Actor     │    │   Rollout   │    │   Ref       │          │
│  │  (训练)     │    │  (生成)     │    │  (参考)     │          │
│  │  FSDP2     │    │  vLLM      │    │  冻结SFT   │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ↓                                     │
│                    权重回同步（weight sync）                      │
│                    3D-HybridEngine 优化                          │
└─────────────────────────────────────────────────────────────────┘
```

**对应 01 章的手写代码：**

| verl 角色 | 手写对应 | 功能 |
|-----------|----------|------|
| Actor | 训练循环 | 更新模型参数 |
| Rollout | `for step: 采样 G 个回答` | 生成回答 |
| Ref | ref 模型 | 计算 KL 惩罚 |

## 代码实现

### Step 1: 环境配置（Docker）

```bash
# 拉取官方镜像
docker pull verlai/verl:latest

# 启动容器
docker run --gpus all --shm-size=32g --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -it verlai/verl:latest bash

# 验证环境
python3 -c "import verl; print(verl.__version__)"
python3 -c "import vllm; print(vllm.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
```

**常见陷阱：**

| 症状 | 原因 | 解法 |
|------|------|------|
| `docker: command not found` | 未安装 Docker | 安装 Docker Desktop |
| `permission denied` | 当前用户不在 docker 组 | `sudo usermod -aG docker $USER` |
| `CUDA out of memory` | 显存不足 | 减小 micro-batch 或使用更小模型 |

### Step 2: 跑通 quickstart（PPO @ GSM8K, 0.5B）

```bash
# 官方 quickstart 的核心行（见 verl docs/start/quickstart）：
python3 -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_files=gsm8k/train \
  data.val_files=gsm8k/test \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  algorithm.adv_estimator=gae
```

**看日志的三行（示意，非本机实录；关键词与 verl 实际日志一致，每行对应 01 章的一个手写件）：**

```
[INFO] rollout.generate_sequences: Generating 256 sequences...
[INFO] sync_rollout_weights: Syncing weights from actor to rollout...
[INFO] actor.loss: Computing PPO loss with GAE advantage...
```

| 日志关键词 | 对应手写 | 它在干什么 |
|---|---|---|
| `rollout` / `generate_sequences` | 01 章"采样 G 个回答" | vLLM 批量生成 |
| `sync_rollout_weights` | 01 章"权重回同步" | 训练权重 → 推理引擎（HybridEngine） |
| `actor/...` + `critic/...` 或 `adv` | 优势计算 + clip 更新 | `adv_estimator=gae`(PPO) / `grpo` |

**验收：** `val/test_score` 随 step 上升（GSM8K 准确率从基线涨几个点即成功）。

### Step 3: PPO → GRPO（一处配置）

```bash
# 同一命令把 advantage 换成组内标准化（01 章手写的 group_advantages）：
python3 -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_files=gsm8k/train \
  data.val_files=gsm8k/test \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.n=5
```

**n=5 的含义：** 每个 prompt 采 5 个回答（= 我们的 G；论文常用 4-16）

**预期观察：**
- GRPO **显存占用比 PPO 低**（没有 critic/Value 网络——Part 8 04 章"Critic-Free"的实证）
- 每个 prompt 的采样数 n 直接乘进 rollout 成本

### Step 4: 自定义奖励函数（你唯一必写的代码）

quickstart 内置 GSM8K 规则奖励；换成你自己的（目录方式）：

```python
# my_reward.py —— 语义与 Part 11 脚本 01 的 gsm8k_reward 相同，接口按 verl 约定
def compute_score(response: str, ground_truth: str) -> float:
    """verl 约定的奖励函数接口。

    Args:
        response: 模型生成的回答
        ground_truth: 标准答案

    Returns:
        float: 奖励分数（0.0 或 1.0）
    """
    # 抽取逻辑与脚本 01 相同
    import re

    # Step 1: 尝试抽取 \boxed{} 中的内容
    m = re.findall(r"\\boxed\{(-?[\d,\.]+)\}", response)

    # Step 2: 如果没有 \boxed{}，尝试抽取 '#### 42' 格式
    if not m:
        m = re.findall(r"####\s*(-?[\d,\.]+)", response)

    # Step 3: 如果都没有，抽取最后一个数字
    if not m:
        m = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))

    # Step 4: 没有找到任何数字 → 错误
    if not m:
        return 0.0

    # Step 5: 取最后一个匹配的数字
    pred = m[-1].replace(",", "").rstrip(".")

    # Step 6: 比较预测值和真实值
    try:
        return 1.0 if abs(float(pred) - float(ground_truth)) < 1e-4 else 0.0
    except ValueError:
        return 0.0
```

**练习方向（每个都是真实的 RLVR 项目形态）：**
- 数学（数字对错）→ 代码（跑单测通过率）→ 格式遵循（JSON schema 校验）

⚠️ **奖励函数是 RLVR 的最高杠杆也是最大风险点：**
规则有洞（如"只看最后数字"）→ 模型学会钻洞（reward hacking，Part 8 07 章的污染近亲）。

### Step 5: 双卡扩展（有 2×4090 时）

```bash
python3 -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_files=gsm8k/train \
  data.val_files=gsm8k/test \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.n=5
```

**verl 自动用 FSDP2 分片训练角色（Part 10 03 章的知识直接兑现）；**
rollout 引擎也有 tensor-parallel 尺寸可配（`actor_rollout_ref.rollout.tensor_model_parallel_size`）

**观察：**
- 显存峰值下降
- step 时间未必减半（rollout 常是新瓶颈——**RL 训练是生成瓶颈**，这解释了为什么 verl/slime 都在 rollout 引擎上卷）

## 工程实践

### 性能分析

**RL 训练的时间分布：**

| 阶段 | 时间占比 | 瓶颈类型 |
|------|----------|----------|
| rollout（生成回答） | 60-80% | 生成瓶颈（memory-bound） |
| reward（打分） | 5-10% | 计算瓶颈（CPU-bound） |
| training（更新参数） | 15-30% | 计算瓶颈（compute-bound） |

**优化方向：**
- rollout：使用 vLLM/SGLang 高吞吐生成
- reward：并行打分（多进程）
- training：FSDP 分片 + 混合精度

### 常见陷阱

#### 陷阱 1: OOM（显存不足）

**症状：** `CUDA out of memory`

**原因：** micro-batch 太大，或模型太大

**解法：** 与[错误 4: 显存不足 (OOM)](#错误-4-显存不足-oom)同源——
减小 micro-batch、换更小模型、必要时 QLoRA + 梯度检查点，完整命令见错误 4。

#### 陷阱 2: 版本冲突

**症状：** `ImportError: cannot import name 'xxx' from 'yyy'`（或 Docker 容器启动失败）

**原因：** verl 与 vllm/torch/transformers 版本锁步耦合，裸 pip 装不出兼容组合

**解法：**
- 使用官方 Docker 镜像，不要裸 pip
- 使用 latest release tag 的官方镜像
- 遇到问题先检查版本兼容性

#### 陷阱 3: 训练不收敛

**症状：** loss 不下降，或准确率不上升

**原因：** 奖励函数有 bug，或学习率太大

**解法：**
- 检查奖励函数（用脚本 01 验证）
- 减小学习率（RL 阶段通常比 SFT 阶段小）
- 增加 KL 惩罚系数

### 最佳实践

#### 配置调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | `grpo` | 比 PPO 更稳定 |
| `actor_rollout_ref.rollout.n` | 4-16 | 组大小，越大越稳定但越贵 |
| `actor_rollout_ref.rollout.micro_batch_size` | 1 | 4090 上防 OOM |
| `algorithm.kl_penalty` | 0.01-0.1 | 防止策略偏离太远 |
| `actor_rollout_ref.actor.lr` | 1e-6 ~ 1e-5 | RL 阶段学习率 |

#### 日志解读

```bash
# 关键指标
rollout/generate_sequences: 生成序列数
sync_rollout_weights: 权重同步时间
actor/loss: 训练损失
val/test_score: 验证分数（最重要！）
```

### 调试展示：常见错误与修复

#### 错误 1: Docker 容器无法访问 GPU

**症状：**
```bash
docker: Error response from daemon: could not select device driver "nvidia"
```

**原因：** 未安装 NVIDIA Container Toolkit

**解法：**
```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 错误 2: verl 启动报错 "No module named 'vllm'"

**症状：**
```bash
ModuleNotFoundError: No module named 'vllm'
```

**原因：** Docker 镜像版本不对，或手动安装了错误版本

**解法：**
```bash
# 使用官方 Docker 镜像
docker pull verlai/verl:latest

# 或在容器内安装
pip install vllm==0.8.x  # 版本要与 verl 兼容
```

#### 错误 3: 训练过程中 NaN loss

**症状：**
```
[INFO] actor/loss: nan
[INFO] val/test_score: 0.0
```

**原因：** 学习率太大，或奖励函数返回异常值

**解法：**
```bash
# 减小学习率
actor_rollout_ref.actor.lr=1e-6

# 检查奖励函数
python3 -c "
from my_reward import compute_score
print(compute_score('答案是 42', '42'))  # 应该返回 1.0
print(compute_score('我不会', '42'))     # 应该返回 0.0
"
```

#### 错误 4: 显存不足 (OOM)

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 MiB
```

**原因：** micro-batch 太大，或模型太大

**解法：**
```bash
# 减小 micro-batch
actor_rollout_ref.rollout.micro_batch_size=1

# 或使用更小的模型
actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct

# 或使用 QLoRA 减少显存
actor_rollout_ref.model.enable_gradient_checkpointing=true
```

#### 错误 5: 权重同步失败

**症状：**
```
[ERROR] sync_rollout_weights: Failed to sync weights
```

**原因：** GPU 内存不足，或 NCCL 通信问题

**解法：**
```bash
# 检查 GPU 内存
nvidia-smi

# 使用 gloo 后端（更稳定）
trainer.backend=gloo

# 或减少 rollout 并行度
actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

### 性能数据（量级参考）

| 模型 | 硬件 | 算法 | 组大小 n | 每步时间 | 显存占用 | 验证分数 |
|------|------|------|----------|----------|----------|----------|
| 0.5B | 1×4090 | PPO | - | ~3s | ~12GB | GSM8K 20% → 35% |
| 0.5B | 1×4090 | GRPO | 5 | ~2s | ~8GB | GSM8K 20% → 38% |
| 0.5B | 1×4090 | GRPO | 16 | ~5s | ~12GB | GSM8K 20% → 42% |
| 0.5B | 2×4090 | GRPO | 5 | ~1.5s | ~6GB/卡 | GSM8K 20% → 38% |
| 7B | 2×4090 | GRPO | 8 | ~30s | ~20GB/卡 | GSM8K 45% → 65% |

> 📊 数据来源：官方 benchmark 与课程设计推算的量级参考（非本机实录；Docker 实操后请以自己日志为准）
> 环境口径：本课脚本环境 torch 2.6.0+cu124；Docker 内以镜像为准
>
> **观察：**
> - GRPO 比 PPO 省显存（没有 critic 网络）
> - 组大小 n 越大，效果越好但成本线性增加
> - 双卡不一定比单卡快（rollout 是瓶颈）

### 之后读什么

`recipe/` 目录是 verl 的"生产配方库"（DAPO=字节论文同款、GSPO、SPPO…），每个 recipe
= 论文方法 + 完整可跑配置。

**面试聊"你们 RL 用什么框架"时：**
能说出 **verl/slime 的角色分工 + HybridEngine 权重同步问题 + recipe 调优经验**，
就是"上手过工业 RL"的可信信号。

## 学完本章你能...

- ✅ 在 Docker 里跑通 0.5B 的 PPO/GRPO 并读懂关键日志
- ✅ 写自定义奖励函数，说出 reward hacking 的风险与防范
- ✅ 解释 RL 训练中 rollout/训练双引擎与权重同步
- ✅ 配置 GRPO 的组大小 n 并评估其对成本/效果的影响
- ✅ 扩展到双卡训练并理解 FSDP 分片

**课后练习**

<details>
<summary>Q1: 为什么 GRPO 模式下不需要 critic 角色？显存省了多少？</summary>

A: **原因：**
- 基线来自组内平均（无需学习），省掉 Value 网络

**显存节省（按 Part 10 的 16Ψ 账本）：**
- 一整份 7B 的训练状态（参数+梯度+优化器）≈112GB 不再需要
- 这就是 Part 8 04 章"Critic-Free"的工业意义

</details>

<details>
<summary>Q2: n（组大小）从 5 提到 32，成本和效果各怎么变？</summary>

A: **成本：**
- rollout 成本线性 ×6.4（生成是 RL 最贵阶段）
- 显存也线性增加（需要存储更多回答）

**效果：**
- 优势估计更准（std 估计更稳）
- 更可能有"组内有区分度"（不会全对/全错）
- 但收益递减（从 4→16 提升大，从 16→32 提升小）

**实际权衡：** 在 8-16 之间权衡，另配 prompt 难度过滤（全对的题跳过）

</details>

<details>
<summary>Q3: verl 的 HybridEngine 解决了什么问题？怎么验证它生效了？</summary>

A: **问题：**
- rollout 和 training 用不同的引擎（vLLM vs FSDP2）
- 每次更新后要把新权重搬进推理引擎（大模型上这是 GB 级拷贝）
- 这个拷贝开销很大

**解决方案：**
- HybridEngine 用重分片+原地转换
- 消除 train↔rollout 转换的显存冗余

**验证方法：**
- 看日志里的 `sync_rollout_weights` 时间
- 应该比直接拷贝快很多

</details>

## 📝 课后作业

完成本章后，去 Assignment 11 完成练习：

👉 [Assignment 11](../../../assignments/assignment_11/)

## 下一步

RL 的前提是好的 SFT 与数据——回 Part 12 补工具链；或去 Part 13 看数据本身怎么来。

---

[← 上一章](01_handwritten_to_verl.md) | [Part 11 README](README.md)
