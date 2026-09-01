# 03 — 显存账本与 ZeRO / FSDP

> 🧭 "7B 模型要用几张卡训？"能答对的人都会先算一笔账。本章先教会**显存账本**
> （脚本 03 的公式 + 逐字节模拟可复算），再用 FSDP 实测分片效果（脚本 04）。

## 🎯 学习目标

完成本章后，你将能够：

- ✅ **背出** 16Ψ 的五项构成（参数/梯度/master/动量/方差），现场心算 7B/70B 的 DDP 显存
- ✅ **推导** ZeRO 三阶段的每卡公式：4Ψ+12Ψ/N → 2Ψ+14Ψ/N → 16Ψ/N，及各自的通信代价
- ✅ **写出** FSDP1 与 FSDP2 的最小用法，解释 all-gather / reduce-scatter 在其中的角色
- ✅ **应用**决策树为任意模型/卡数选并行策略（DDP → ZeRO-1 → ZeRO-2 → FSDP/ZeRO-3）

## 📖 前置知识

**必须掌握：**

- **02 章**：DDP（每卡完整模型，只同步梯度）——本章的"对照面"

**建议掌握：**

- **Part 8**：AdamW、bf16 混合精度（16Ψ 账本会用到，本章也会当场重推）

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
| 7B | 112 GB | 38.5 GB | 14 GB/卡 |
| 70B | 1120 GB | 385 GB | 140 GB/卡 |

- 🔑 **这就是"为什么 7B 在 24G 卡上训不动"的完整答案**：哪怕 bf16 存参数只要 14GB，
  AdamW 状态一加就到 112GB。ZeRO 的全部思想就一句话：**这 16Ψ 里大部分是优化器状态，
  没必要每个卡都存完整份**。

**ZeRO 三阶段**（Rajbhandari et al. 2019）：

| 阶段 | 切什么 | 每卡模型状态 | 通信代价 |
|---|---|---:|---|
| ZeRO-0 (DDP) | 什么都不切 | 16Ψ | 基准 |
| ZeRO-1 | 优化器状态 | 4Ψ + 12Ψ/N | 与 DDP 相同 |
| ZeRO-2 | +梯度 | 2Ψ + 14Ψ/N | 与 DDP 相同 |
| ZeRO-3 / FSDP | +参数 | 16Ψ/N | ≈1.5× DDP |

[脚本 03](../scripts/03_zero_memory.py) Part B 把每个 rank 持有的每一类张量逐字节加起来，
断言 ZeRO-1 与公式 4Ψ+12Ψ/N 一致（ZeRO-2/3 的逐字节复算留作本章动手 2）——面试被追问
"12Ψ/N 怎么来的"时，这就是你的证据链。

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

> 📝 上面是工业写法（按 Block 切分分片单元）。本课 [脚本 04](../scripts/04_fsdp_gpt.py)
> 为简洁起见用**整体包裹**（不传 `auto_wrap_policy`，分片单元 = 整个模型）——数学完全一致，
> 工业上用 `auto_wrap_policy` 按 Block 切分，all-gather 粒度更细、峰值显存更低（见下方 💡）。

- ⚠️ 版本提示：FSDP1 已被官方标记弃用，新项目推荐直接用 FSDP2 的 `fully_shard`；
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
- ✅ 扩展脚本 03 的账本函数复算 ZeRO-2/3，并代入 7B/70B 填出三档对照表（动手 1/2）

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

## 🛠️ 动手实践（依托脚本 03，纯 CPU 即可）

### 动手 1：把账本代入 7B / 70B，填出三档显存对照表

**任务**：复制 `scripts/03_zero_memory.py` 为 `03_lab.py`，把 Part A 的 `models` 列表换成
7B 与 70B，并仿照 `s1` 那行的算式补打 ZeRO-2、ZeRO-3 两列（公式就印在它下面四行）。

**步骤提示：**
1. `s2 = (2 * psi + 14 * psi / n) / 1e9`、`s3 = (16 * psi / n) / 1e9`，照抄 `s1` 的
   格式各加一格；
2. 跑 `python 03_lab.py`，把 N=8 那列抄进你自己的表。

**验收标准：**
- [ ] N=8 时你的表与教程 §1 的表对得上，并补齐它没列的 ZeRO-2 一档：7B：DDP 112 /
      ZeRO-1 38.5 / ZeRO-2 **26.25** / ZeRO-3 14 GB；70B：1120 / 385 / **262.5** / 140 GB
      （GB 按 1e9 字节口径，与脚本一致）
- [ ] 任意 N>1 都有 ZeRO-2 < ZeRO-1：两式相减 (2Ψ+14Ψ/N) − (4Ψ+12Ψ/N) = (2/N−2)·Ψ < 0，
      与 assignment 题 2 的"恒等式预警"互相印证
- [ ] 用你的表回答：8 卡从 DDP 换 ZeRO-3，每卡模型状态降到几分之一？（1/8，与 N 无关的
      整数倍关系只在 ZeRO-3 成立——另外两档不是纯整除）

### 动手 2：把逐字节模拟扩展到 ZeRO-2/3（教程只替你断言了 ZeRO-1）

**任务**：给 `zero_accounting_simulation` 加 `stage` 参数：stage2 把"bf16 梯度"那一行也改走
`shard_numel`（参数仍全量）；stage3 把三类全改走 `shard_numel`。对 N=2 与 N=3 各断言一次。

**步骤提示：** 每类张量就是一行 `b += ...`——把"全量"的 `p.numel()` 换成"分片"的
`shard_numel(p.numel())` 就是升一阶。ZeRO 的"阶段"在代码里就是"多切一类张量"，
这就是三行代码讲完的 ZeRO。

**验收标准：**
- [ ] stage2：`sum(per_rank)/N` 与公式 2Ψ+14Ψ/N **严格相等**（分片不重不漏 → 平均值精确
      对上；对照公式别用 `==`，用 `abs(a-b) < 1e-6`）
- [ ] 单个 rank 与公式的偏差不为 0 的情形：N=3 时能看到百字节量级的 rank 间差
      （本机实测 stage2 @ N=3：+233 / −103 / −131 字节）——"公式是均摊值"的直观含义
- [ ] stage3：`sum(per_rank)/N == 16Ψ/N`，且 N=1 时 stage 0/1/2/3 四条全部退化为 16Ψ
      （assignment 题 2 的"自检锚点"同款）
- [ ] 用脚本里的 TinyGPT（Ψ=6,298,624）复核一个数：stage3 @ N=2 每卡 ≈ **50.4 MB**
      （16Ψ/2；对照 Part B 打印的 16Ψ = 100.8 MB）

## 📝 课后作业

👉 [Assignment 10](../../../assignments/assignment_10/) 题 2（显存账本计算器——公式驱动，
自动验算 ZeRO 各阶段）

## 下一步

到此为止每一层还是完整的。如果**单层**都放不下（70B 的 FFN 一层就有几个 GB），
就要把计算本身切开：张量并行与流水线并行（进阶可选章）。

👉 [04 — 张量并行、流水线并行与工业栈](04_tp_pp_and_beyond.md)
