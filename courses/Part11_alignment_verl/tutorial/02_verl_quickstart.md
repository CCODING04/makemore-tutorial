# 02 — verl 快速上手：0.5B GRPO 实战

> 🧭 本章在 Docker 里跑通 verl 官方 quickstart（Qwen2.5-0.5B 在 GSM8K 上的 PPO/GRPO，
> 官方文档明确**单卡 ≥24GB**——4090 直接可跑），然后换成 GRPO、写自己的奖励函数、
> 扩到双卡。所有步骤的"手写对应物"都标注 01 章的编号。

## 📖 前置知识

- **01 章**：概念映射表（本章配置行 = 01 章的手写代码）
- **Part 10**：FSDP（第 5 步的扩展实验用）

## 1. 环境：Docker（版本策略：latest release tag 的官方镜像）

```bash
docker pull verlai/verl:latest
docker run --gpus all --shm-size=32g --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -it verlai/verl:latest bash
# ⚠️ 为什么不用裸 pip：verl 与 vllm/torch/transformers 版本锁步耦合，
#    Docker 是官方推荐的唯一低摩擦路径（配 uv sync 的 git 方式次之）
```

## 2. 跑通 quickstart（PPO @ GSM8K, 0.5B）

```bash
# 官方 quickstart 的核心行（见 verl docs/start/quickstart）：
python3 -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=1 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  ...   # 完整命令抄官方 quickstart；micro batch 置 1 防 24GB OOM
```

**看日志的三行**（每行对应 01 章的一个手写件）：

| 日志关键词 | 对应手写 | 它在干什么 |
|---|---|---|
| `rollout` / `generate_sequences` | 01 章"采样 G 个回答" | vLLM 批量生成 |
| `sync_rollout_weights` | 01 章"权重回同步" | 训练权重 → 推理引擎（HybridEngine） |
| `actor/...` + `critic/...` 或 `adv` | 优势计算 + clip 更新 | `adv_estimator=gae`(PPO) / `grpo` |

验收：`val/test_score` 随 step 上升（GSM8K 准确率从基线涨几个点即成功）。

## 3. PPO → GRPO（一处配置）

```bash
# 同一命令把 advantage 换成组内标准化（01 章手写的 group_advantages）：
algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5
# n=5：每个 prompt 采 5 个回答（= 我们的 G；论文常用 4-16）
```

预期观察：GRPO **显存占用比 PPO 低**（没有 critic/Value 网络——Part 8 04 章
"Critic-Free"的实证）；每个 prompt 的采样数 n 直接乘进 rollout 成本。

## 4. 自定义奖励函数（你唯一必写的代码）

quickstart 内置 GSM8K 规则奖励；换成你自己的（目录方式）：

```python
# my_reward.py —— 语义与 Part 11 脚本 01 的 gsm8k_reward 相同，接口按 verl 约定
def compute_score(response: str, ground_truth: str) -> float:
    ...  # boxed/####/最后数字 → 0/1
```

练习方向（每个都是真实的 RLVR 项目形态）：数学（数字对错）→ 代码（跑单测通过率）→
格式遵循（JSON schema 校验）。⚠️ 奖励函数是 RLVR 的**最高杠杆也是最大风险点**：
规则有洞（如"只看最后数字"）→ 模型学会钻洞（reward hacking，Part 8 07 章的污染近亲）。

## 5. 双卡扩展（有 2×4090 时）

```bash
trainer.n_gpus_per_node=2 trainer.nnodes=1
# verl 自动用 FSDP2 分片训练角色（Part 10 03 章的知识直接兑现）；
# rollout 引擎也有 tensor-parallel 尺寸可配（actor_rollout_ref.rollout.tensor_model_parallel_size）
```

观察：显存峰值下降、step 时间未必减半（rollout 常是新瓶颈——**RL 训练是生成瓶颈**，
这解释了为什么 verl/slime 都在 rollout 引擎上卷）。

## 6. 之后读什么

`recipe/` 目录是 verl 的"生产配方库"（DAPO=字节论文同款、GSPO、SPPO…），每个 recipe
= 论文方法 + 完整可跑配置。面试聊"你们 RL 用什么框架"时：能说出 **verl/slime 的角色分工 +
HybridEngine 权重同步问题 + recipe 调优经验**，就是"上手过工业 RL"的可信信号。

## 学完本部分你能...

- ✅ 在 Docker 里跑通 0.5B 的 PPO/GRPO 并读懂关键日志
- ✅ 写自定义奖励函数，说出 reward hacking 的风险与防范
- ✅ 解释 RL 训练中 rollout/训练双引擎与权重同步
- ✅ 配置 GRPO 的组大小 n 并评估其对成本/效果的影响

**课后练习**

<details>
<summary>Q1: 为什么 GRPO 模式下不需要 critic 角色？显存省了多少？</summary>
A: 基线来自组内平均（无需学习），省掉 Value 网络 ≈ 省掉一整份模型规模的
参数+梯度+优化器显存（16Ψ/N 账本里的一整个 Ψ）——7B 模型上约省 7B×12B≈84GB 级别的
训练状态。这就是 Part 8 04 章"Critic-Free"的工业意义。
</details>

<details>
<summary>Q2: n（组大小）从 5 提到 32，成本和效果各怎么变？</summary>
A: rollout 成本线性×6.4（生成是 RL 最贵阶段）；效果上优势估计更准（std 估计更稳）、
更可能有"组内有区分度"。实际在 8-16 之间权衡，另配 prompt 难度过滤（全对的题跳过）。
</details>

## 📝 课后作业

👉 [Assignment 11](../../../assignments/assignment_11/)

## 下一步

RL 的前提是好的 SFT 与数据——回 Part 12 补工具链；或去 Part 13 看数据本身怎么来。

---

[← 上一章](01_handwritten_to_verl.md) | [Part 11 README](README.md)
