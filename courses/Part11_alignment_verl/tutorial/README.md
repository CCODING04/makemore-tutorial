# Part 11: 对齐实战 — verl 工业级 RL 后训练

> 🧭 Part 8 手写了 PPO/GRPO 的原理（玩具规模）；本部分把它们放上**工业级 RL 基建** verl
> （字节跳动 HybridFlow，DAPO/Seed-Thinking/Doubao 的训练系统），在真实 0.5B 模型上跑
> GRPO。学完你能承担"跑 RL 实验、改奖励函数、调 rollout 配置"这类真实的对齐岗日常。
> 主源：[verl-project/verl](https://github.com/verl-project/verl)（23.2k，Apache-2.0）

## 学习目标

完成本部分后，你将能够：

- ✅ **理解** RL 后训练在 LLM 链路中的位置和价值
- ✅ **手写** RLVR 奖励函数（\boxed / #### / 最后数字的抽取链）
- ✅ **解释** GRPO 的数学原理和"全对组优势全零"现象
- ✅ **画出** 从手写 GRPO 到 verl 三角色架构的映射图
- ✅ **配置** verl 的 PPO/GRPO 训练并读懂关键日志
- ✅ **识别** reward hacking 的风险并设计防范策略

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [从手写 GRPO 到 verl：概念桥接](01_handwritten_to_verl.md) | 逐概念映射（rollout/权重同步/优势/KL/奖励）+ 奖励函数手写 | `01` |
| 02 | [verl 快速上手：0.5B GRPO 实战](02_verl_quickstart.md) | Docker 环境 → quickstart PPO → 换 GRPO → 自定义奖励 → 双卡 | —（CLI 实操） |

## 🧰 前置知识

**必须掌握：**
- **Part 8 04 章**：手写 PPO（GAE/clip）与 GRPO（组内标准化）——本章的"手写侧"
- **Part 8 07 章**：评估学（RLVR 的"可验证奖励"就是它的应用）

**建议掌握：**
- **Part 10**：FSDP 与多卡基础（verl 的训练后端就是 FSDP2/Megatron）
- **Part 8 02 章**：SFT 训练流程（RL 是 SFT 之后的阶段）

**可选：**
- **Part 14**：vLLM 推理引擎（verl 的 rollout 引擎用 vLLM/SGLang）

## 🔗 在 LLM 链路中的位置

```
预训练(Part 7) → SFT(Part 8/12) → 【本部分: RL 后训练 (PPO/GRPO/DAPO…)】 → 部署(Part 14)
                                    ↑
                                    你在这里
```

**为什么这一步是 2026 年的主战场：**

| 证据 | 说明 |
|------|------|
| GLM-5.3 | 基座与 5.2 完全相同，全部提升来自后训练 RL Scaling |
| DeepSeek-V4 | 把 GRPO 下沉到"专家模型"层 |
| Kimi-Researcher | 平均 23 次工具调用/回答，端到端 RL on hard tasks |

**框架生态：** verl（字节）与 slime（智谱，8.3k）是两大开源 RL Scaling 框架。

## 📦 环境与版本策略（⚠️ 全课程安装摩擦最高的一章）

verl 的 vllm/torch/transformers 版本锁步耦合严重——**用官方 Docker，不要裸 pip**：

```bash
# 策略：latest release tag 的官方镜像（锁版本 + 免依赖地狱）
docker pull verlai/verl:latest
# 最小硬件：官方 quickstart = Qwen2.5-0.5B 单卡 PPO（文档明确 ≥24GB）
#           → 单张 4090 可跑（micro-batch 置 1 防 OOM）；双卡可玩 FSDP 分片
```

| 你有什么 | 能做什么 |
|---|---|
| CPU only | 01 章脚本（奖励函数/组内优势/KL 全部纯 Python 可跑）+ 通读概念 |
| 1×4090 | Docker quickstart 完整跑通（0.5B PPO/GRPO，小时级） |
| 2×4090 | + FSDP 分片实验（02 章第 5 步） |

## 📈 学习地图

```
手写过的 GRPO（Part 8）        ← 点（原理）
   ↓ 逐概念映射
奖励函数手写（脚本01：RLVR 入口）
   ↓
verl quickstart 跑通           ← 线（工业工具）
   ↓ GRPO / 自定义奖励 / 双卡
读 recipe/dapo（生产配方）      →  面试/工作就绪
```

## 📝 课后作业

每章末尾有思考题（`<details>` 折叠答案）。全部学完后：

👉 [Assignment 11](../../../assignments/assignment_11/)

## 🔗 相关资源

- 🐙 [verl](https://github.com/verl-project/verl)（docs/quickstart 是最接近教程的官方材料）
- 🐙 [slime](https://github.com/THUDM/slime)（智谱 RL Scaling 框架，GLM-4.5 起的 GLM 系列用）
- 🐙 [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)（裸 PyTorch 从零写 RLVR-GRPO——原理侧对照）
- 📄 HybridFlow（EuroSys'25）· GRPO（arXiv 2402.03300）· DAPO

---

[← 上一章：Part 10 分布式](../../Part10_distributed/tutorial/README.md) | [下一章：Part 12 LLaMA-Factory →](../../Part12_finetune_llamafactory/tutorial/README.md)
