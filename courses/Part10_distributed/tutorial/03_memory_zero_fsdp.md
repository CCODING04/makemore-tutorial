# 03 — 显存账本与 ZeRO / FSDP

> 🧭 "7B 模型要用几张卡训？"能答对的人都会先算一笔账。本章先教会**显存账本**
> （脚本 03 的公式 + 逐字节模拟可复算），再用 FSDP 实测分片效果（脚本 04）。

## 📖 前置知识

- **02 章**：DDP（每卡完整模型，只同步梯度）——本章的"对照面"
- **Part 8**：AdamW、bf16 混合精度

## 1. 模型状态显存：16 字节/参数

混合精度（bf16/fp16 + fp32 master）AdamW 下，**每个参数**要存：

| 项 | 字节 |
|---|---:|
| bf16 参数 | 2 |
| bf16 梯度 | 2 |
| fp32 master 权重 | 4 |
| fp32 动量 m | 4 |
| fp32 方差 v | 4 |
| **合计** | **16Ψ**（Ψ=参数量） |

| 模型 | DDP（每卡 16Ψ） | ZeRO-1（N=8） | ZeRO-3（N=8） |
|---|---:|---:|---:|
| 7B | 112 GB | 7.4 GB | 14 GB→2 GB/卡 |
| 70B | 1120 GB | 74 GB | 280 GB→35 GB/卡 |

- 🔑 **这就是"为什么 7B 在 24G 卡上训不动"的完整答案**：哪怕 bf16 存参数只要 14GB，
  AdamW 状态一加就到 112GB。ZeRO 的全部思想就一句话：**这 16Ψ 里大部分是优化器状态，
  没必要每个卡都存完整份**。

**ZeRO 三阶段**（Rajbhandari et al. 2019）：

| 阶段 | 切什么 | 每卡模型状态 | 通信代价 |
|---|---|---:|---|
| ZeRO-0 (DDP) | 什么都不切 | 16Ψ | 基准 |
| ZeRO-1 | 优化器状态 | 4Ψ + 12Ψ/N | 与 DDP 相同 |
| ZeRO-2 | +梯度 | 8Ψ + 4Ψ/N | 与 DDP 相同 |
| ZeRO-3 / FSDP | +参数 | 16Ψ/N | ≈1.5× DDP |

[脚本 03](../scripts/03_zero_memory.py) Part B 把每个 rank 持有的每一类张量逐字节加起来，
断言与公式一致——面试被追问"12Ψ/N 怎么来的"时，这就是你的证据链。

- ⚠️ **激活值不在这本账里**：激活 ≈ sbh×(34 + 5·a·s/h) 字节/层（s=seq, b=batch, h=hidden,
  a=heads），随并行方式无关、随 seq² 有平方项。省激活靠**梯度检查点**（只存层输入、
  反传时重算，显存从每层全存降到 2sbhL，代价 ≈ 多一次 forward，即 +33% 计算）。

## 2. FSDP：ZeRO-3 的 PyTorch 原生实现

**DDP vs FSDP 一句话**：DDP 每 rank 全量存储、只 all-reduce 梯度；FSDP 把参数/梯度/
优化器状态全分片，forward 前 **all-gather** 收齐当前层的参数、算完即释放，backward 用
**reduce-scatter** 收梯度分片 —— 用 1.5× 通信换"单卡装得下"。

[脚本 04](../scripts/04_fsdp_gpt.py)（双卡实测）：

```
完整模型参数量: 11.8 MB (fp32)
FSDP 包装后本 rank 参数显存: 6.0 MB（全量 5.9 MB；≈一半参数分片 + FSDP 运行时缓冲）
训练后平均 loss: 2.981（各 rank 初始化不同 → 广播同步正确）
```

```python
# FSDP1 API（torch 2.x 全可用；本课脚本用这个）
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
fsdp_model = FSDP(model, auto_wrap_policy=wrap_policy,     # 每个 Block 一个分片单元
                  sharding_strategy=ShardingStrategy.FULL_SHARD,  # = ZeRO-3
                  use_orig_params=True)

# FSDP2（torch 2.6+ 推荐，API 更干净；概念完全一致）
from torch.distributed.fsdp import fully_shard
for layer in model.blocks:
    fully_shard(layer)          # 先子模块
fully_shard(model)              # 后根模块（参数变 DTensor，优化器不用改）
```

- ⚠️ 版本提示：PyTorch 2.13 起 FSDP1 已弃用，新项目直接用 FSDP2 的 `fully_shard`；
  但网上大量代码（和本课脚本）是 FSDP1——读得懂两者，迁移只是换 API。
- 💡 实践分工：**显存够用就 DDP**（通信最少）；装不下再 ZeRO-1（免费午餐）；还不够一路
  ZeRO-3/FSDP。transformer_auto_wrap_policy 让每个 Block 成为独立分片单元，
  all-gather 粒度更细、峰值显存更低。

## 3. 决策树（面试可直接背）

```
模型状态 16Ψ 装得下？
├─ 是 → DDP（最优通信）
└─ 否 → 先 ZeRO-1（免费，通信不变）→ 再不够 ZeRO-2 → 还不够 FSDP/ZeRO-3
         激活也爆？→ 梯度检查点 + 减 micro-batch（配梯度累积保 effective batch）
         单层都放不下？→ 张量并行 TP（04 章）
```

## 学完本章你能...

- ✅ 背出 16Ψ 的五项构成，现场算 7B/70B 的 DDP 显存
- ✅ 说出 ZeRO 三阶段各切什么、通信代价怎么变
- ✅ 写出 FSDP1 与 FSDP2 的最小用法，解释 all-gather/reduce-scatter 在其中的角色
- ✅ 用"决策树"为任意模型/卡数选并行策略

**课后练习**

<details>
<summary>Q1: ZeRO-1 为什么"免费"（通信量与 DDP 相同）？切掉的 12Ψ/N 里有什么？</summary>
A: fp32 master/动量/方差只在 optimizer.step() 时用，且每个 rank 只 step 自己分片的参数
（参数全量在，梯度全量在，只对分片做更新再原地写回）。通信路径上没有任何新增集合操作。
切掉的是 12 字节/参数的优化器状态 —— 它们是纯粹的"存储负担"，不参与前反向计算。
</details>

<details>
<summary>Q2: FSDP 的 all-gather 为什么能"算完就释放"？什么配置下不释放？</summary>
A: 前向只需当前层的参数：FSDP 按 Block 为单元 all-gather → 算完该层即可释放，下一个层
再收下一份（reshard_after_forward=True）。若设 False（=SHARD_GRAD_OP/ZeRO-2 语义），
参数收齐后整个 forward+backward 期间保留 —— 用显存换通信，第一层之后的层不用重复收。
</details>

<details>
<summary>Q3: 7B 模型、8×24GB 卡。模型状态 112GB → ZeRO-3 每卡 14GB，能训了吗？还差什么？</summary>
A: 只能说"模型状态装得下了"。还要算：① 激活（seq/batch 大时轻松几十 GB → 梯度检查点）；
② 通信是否成为瓶颈（ZeRO-3 的 1.5×）；③ NCCL 通信缓冲与碎片余量（实际可用 <24GB）。
工业答案：8 卡训 7B 常配 ZeRO-2/3 + 梯度检查点 + bf16 + flash-attention。
</details>

## 📝 课后作业

👉 [Assignment 10](../../../assignments/assignment_10/) 题 2（显存账本计算器——公式驱动，
自动验算 ZeRO 各阶段）

## 下一步

到此为止每一层还是完整的。如果**单层**都放不下（70B 的 FFN 一层就有几个 GB），
就要把计算本身切开：张量并行与流水线并行（进阶可选章）。

👉 [04 — 张量并行、流水线并行与工业栈](04_tp_pp_and_beyond.md)
