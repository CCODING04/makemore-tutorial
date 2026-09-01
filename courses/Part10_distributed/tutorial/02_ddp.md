# 02 — DDP：数据并行深入

> 🧭 DDP（DistributedDataParallel）是所有分布式训练的地基，也是 90% 场景的正确起点。
> 本章走完它的五个必要件，用三步推导看清"all-reduce 平均 == 大 batch 梯度"，
> 看穿内部机制（桶化 all-reduce 与 backward 重叠），并用双卡实测吞吐。
> 跑 [scripts/02_ddp_gpt.py](../scripts/02_ddp_gpt.py)。

## 🎯 学习目标

完成本章后，你将能够：

- ✅ **写出** DDP 五件套，并避开两个高频 bug（DataLoader 双重打乱、忘 `set_epoch`）
- ✅ **推导**"all-reduce 平均后的梯度 == 大 batch 梯度"（§2），说清等价成立的三个前提
- ✅ **画出**桶化 all-reduce 与 backward 重叠的时序图，解释"多卡近线性加速"从哪来
- ✅ **组合**梯度累积与 `no_sync`：累积步不通信、最后一步只同步一次
- ✅ **解读**多卡吞吐数字——为什么 toy 模型的加速比不可信、MFU 该用真实规模测

## 📖 前置知识

**必须掌握：**

- **01 章**：SPMD 心智模型、all_reduce 平均语义、torchrun

**建议掌握：**

- [Part 9 01 章](../../Part9_cuda_kernels/tutorial/01_gpu_and_first_kernel.md)："内核异步执行"——理解桶化 all-reduce 与 backward 重叠的前提

## 1. DDP 五件套（脚本 02 的骨架）

```python
# ① 进程组（torchrun 注入 RANK/WORLD_SIZE/MASTER_*，env:// 自动读取）
dist.init_process_group(backend="nccl")          # 单机多卡；CPU 用 gloo
rank = dist.get_rank()
torch.cuda.set_device(rank)                       # 一卡一进程，各绑各的 GPU

# ② 数据分片：DistributedSampler（不是把 batch 切开，是 dataset 切开！）
sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
loader = DataLoader(dataset, batch_size=16, sampler=sampler, shuffle=False)  # ⚠️ shuffle=False

# ③ 包装模型（rank0 的权重自动 broadcast 给所有 rank → 起点一致）
ddp_model = DDP(model, device_ids=[rank])

# ④ 训练循环：每轮 set_epoch（sampler 用 epoch 做打乱种子）
for epoch in range(E):
    sampler.set_epoch(epoch)
    for xb, yb in loader:
        _, loss = ddp_model(xb, yb)               # DDP 透传 forward
        loss.backward()                           # ← all-reduce 在这里自动发生
        opt.step(); opt.zero_grad()

# ⑤ 收尾
dist.destroy_process_group()
```

- ⚠️ 两个高频 bug：DataLoader 里忘写 `shuffle=False`（与 sampler 双重打乱）；忘写
  `sampler.set_epoch`（每个 epoch 数据分片完全相同，等效于只在 1/world 的数据上训练 N 遍）。

## 2. 数学：为什么"all-reduce 平均后 == 大 batch 一步"

五件套注释里那句"all-reduce 在 backward 里自动发生"，藏着一个值得摊开的数学事实：
**DDP 走一步，严格等于"用大 batch 单卡走一步"**。设 N 个 rank、每 rank 本地 batch b，
总有效 batch B = N·b，全局损失是 B 个样本损失的平均：

```
L(θ) = (1/B) · Σ_{i=1..B} ℓ_i(θ)            # ℓ_i：第 i 个样本的交叉熵
```

**第 1 步（求导穿过求和号）**：梯度对样本损失是**线性**的——和的导数 = 导数的和：

```
∇L = (1/B) · Σ_{i=1..B} ∇ℓ_i
```

**第 2 步（按 rank 把样本分组）**：DistributedSampler 把 B 个样本不重不漏地切成
D_0, D_1, …, D_{N-1}（每份恰好 b 个）。大求和就可以"先组内、再组间"地重排：

```
∇L = (1/(N·b)) · Σ_{r=0..N-1} Σ_{i∈D_r} ∇ℓ_i
   = (1/N) · Σ_{r=0..N-1} [ (1/b) · Σ_{i∈D_r} ∇ℓ_i ]
                      └─────────┬─────────┘
                      g_r：rank r 的"本地 batch 梯度"
```

**第 3 步（对照 DDP 实际做的事）**：rank r 的 `F.cross_entropy(...)` 默认
`reduction='mean'`，本地 loss 正是 (1/b)·Σ_{i∈D_r} ℓ_i，backward 得到的就是 g_r；
DDP 对每个桶做 all-reduce(SUM) 后 ÷N，于是每个 rank 最终拿到：

```
(1/N) · Σ_{r=0..N-1} g_r   ==   ∇L          # 与大 batch 单卡一步逐参数相等
```

等价成立依赖三个前提，缺一个都不成立：

| 前提 | 谁负责 | 破坏时的后果 |
|---|---|---|
| 分片**不重不漏且等大** | DistributedSampler + DataLoader `drop_last` | 不等大 → "平均的平均 ≠ 总平均"（见下方直觉）；重复/遗漏样本 → 平均的根本不是目标 loss |
| 本地 loss 取**均值** | `reduction='mean'`（默认） | 若是求和，g_r 大 b 倍，等效学习率放大 |
| 各 rank 从**同一权重**出发 | DDP 构造时 broadcast | 各练各的，见 Q1 |

> 💡 **直觉：梯度是线性的**。整条推导只用了一件事——样本梯度可以任意分组、先组内求
> 平均再组间求平均。所以"等大分组的平均的平均"完全等于"对全体直接取平均"。
> **唯一能打破它的是组不等大**：(1/N)·Σg_r 给每个 rank 等权，而总平均按样本数加权——
> 小组里的样本会被"超权"。这正是 DistributedSampler 宁可复制/丢弃少量样本也要把每份
> 切得等大的数学原因（样本数不整除 world 时它复制少量样本补齐等大；`drop_last=True`
> 丢掉凑不满一个 micro-batch 的尾巴，同理）。动手 2（本章末）让你用 10 行代码亲眼
> 看到"等大成立、不等大失效、加权能救回来"。

## 3. 内部机制：通信是怎么"藏"进 backward 的

朴素实现是"forward → backward 全算完 → all-reduce 所有梯度 → step"，通信完全串行。
DDP 的真实做法（[设计笔记](https://docs.pytorch.org/docs/stable/notes/ddp.html)）：

```
构造时：按参数的反向传播【就绪顺序】把参数分桶（默认桶 25MB）
backward 时：某个桶的全部梯度一就绪 → 立刻异步 all-reduce 这个桶
             ↓
       通信与"剩余层的 backward 计算"重叠 → 大部分通信时间被计算盖住
全部桶发起后，backward 结束前才阻塞等最后一个桶
```

把一次 iteration 画成"计算/通信"两条流（world_size=2，桶按反向就绪顺序编号，示意 3 桶）：

```
时间 ════════════════════════════════════════════════════════════════════════════▶
计算流    ┌───────────┐ ┌──────────────────────────────┐ ┌─────────┐
（GPU）   │  forward  │ │ backward：head→L3→L2→L1→emb │ │  step   │
          │ emb→L1→L2 │ │ 参数就绪顺序 = 反向传播顺序   │ │(读.grad)│
          │ →L3→head  │ └──────────────────────────────┘ └─────────┘
          └───────────┘        │           │         │
                       桶0(head,L3)   桶1(L2)    桶2(L1,emb)
                             ▼           ▼         ▼
通信流               ┌────────────┐ ┌────────┐ ┌─────────┐
（NCCL 独立引擎）    │ all-reduce │ │ all-  │ │ all-red │──▶ wait()：最后一个桶收齐
                     │    桶0     │ │ reduce │ │   桶2   │    才放行 backward 返回
                     └────────────┘ └─ 桶1 ──┘ └─────────┘
                           ↑ 通信与"还没反向完的层"同时进行 → 大部分通信被计算盖住
```

- 🔑 这就是"多卡吞吐接近线性"的原因：通信不是没有，而是**被藏起来了**。
  顺带记住面试常问的 **MFU**（Model FLOPs Utilization）＝实测 FLOPS ÷ 卡的峰值 FLOPS——
  评估分布式训练效率的标准指标，第四部分会反复用到。
- 两个相关参数：`find_unused_parameters=True`（有未参与 loss 的参数时防挂死，有遍历开销）；
  `broadcast_buffers=True`（默认开：每次 forward 前 rank0 的 buffer 广播给所有 rank——
  BatchNorm 的 running stats 靠它同步。这也解释了 DDP+BN 的行为）。
- 📝 对照脚本 02 的真实规模：它的 GPT 只有 **628,161 个参数（fp32 ≈2.5MB）**，远小于默认
  桶上限 25MB——DDP 实际只建了 **1 个桶**（backward 后用
  `len(ddp.reducer._get_zeros_like_grad_buckets())` 可验证；实测环境 RTX 4090×2,
  torch 2.6.0+cu124, NCCL）。所以"桶间重叠"在 toy 规模上根本无从体现，§5 里双卡吞吐
  与单卡持平才是符合预期的读数；真实大模型单层参数就是 GB 级，桶化+重叠才是吞吐生命线。
- 从 no_sync 的视角再看这张图：前 K−1 个 micro-step，通信流上**一格都不发**；第 K 步的
  backward 才把上面这组 all-reduce 一次性发出（数学见 §4）。

## 4. 梯度累积：no_sync

梯度累积（小卡模拟大 batch）与 DDP 组合有个坑：默认**每次 backward 都 all-reduce**，
累积 4 步就通信 4 次，其中 3 次是浪费。正确姿势：

```python
for it, (xb, yb) in enumerate(loader):
    is_last = (it % accum == accum - 1)
    ctx = ddp_model.no_sync() if not is_last else nullcontext()
    with ctx:
        _, loss = ddp_model(xb, yb)
        (loss / accum).backward()        # no_sync 内：只累积本地梯度，不通信
    if is_last:
        opt.step(); opt.zero_grad()      # 最后一步的 backward 才 all-reduce
```

### 数学：K 步累积后一次 all-reduce == K·N·b 大 batch 的梯度

目标 batch 是 K·N·b（K 个 micro-step × N 卡 × 本地 b）。把 §2 的结论再用一次——先在
第 k 个 micro-step 内部对 N 卡平均，再对 K 步平均（还是那件事：**梯度是线性的**）：

```
∇L_big = (1/K) · Σ_{k=1..K} ĝ_k ,        ĝ_k = (1/N) · Σ_{r=0..N-1} g_{r,k}
```

脚本每个 micro-step 执行 `(loss/accum).backward()`，于是：

- **前 K−1 步**（`no_sync` 上下文内）：PyTorch 的 `.grad` 是**累加**语义，本地缓冲里
  依次叠加 (1/K)·g_{r,1}, (1/K)·g_{r,2}, …，一次通信都不发；
- **第 K 步**（恢复正常 backward）：触发 all-reduce，它同步的对象是 `.grad` 缓冲的
  **当前值**——里面已经装着前 K−1 份，本次再叠加 (1/K)·g_{r,K} 后一起求和平均：

```
all-reduce 后 = (1/N) · Σ_r Σ_{k=1..K} (1/K) · g_{r,k}
             = (1/K) · Σ_{k=1..K} ĝ_k = ∇L_big      # 正是目标大 batch 的梯度
```

**为什么能省 K−1 次通信**：all-reduce 是线性算子，"每步通信、通信完再累加"与"先本地
累加、最后一次通信"结果相同——通信的内容（各 rank 之和）不因延后而改变，所以延后
合法；延后 K−1 步，就是省掉 K−1 次。反过来，**第 K 步那一次不能省**：optimizer.step
前必须拿到全体平均，这就是 `is_last` 分支存在的全部原因（忘了切回正常 backward =
梯度永远是本地值，训练静默走歪，不报错）。

实测收益：累积 2 步时通信量减半（脚本 02 默认 accum=2）。

## 5. 实测（RTX 4090×2，torch 2.6.0+cu124，脚本 02）

```
world_size=2: 每 rank batch=16 × accum=2 → 有效 batch = 64
  本 rank 吞吐: ~77k–95k tokens/s（run-to-run 波动大，取 ~90k；平均 loss ≈3.6）
world_size=1: 有效 batch = 32
  本 rank 吞吐: ~180k–185k tokens/s（平均 loss ≈3.3）
```

（实测环境：RTX 4090×2, torch 2.6.0+cu124, NCCL；wall ≈0.2s 的 toy 规模，吞吐数字每次运行波动较大，仅供量级对照。）

- 💡 怎么读这组数字：单卡 tokens/s 是"一张卡的产能"；双卡世界总产能 ≈ 90k × 2 = 180k，
  与单卡 ~183k 基本持平（甚至略低）——因为这个 toy 模型太小（计算 0.2s 就完了，启动/通信占比高）。
  **真实大模型上多卡才能体现价值**；小模型测分布式加速比是新手常见误区。
  （nanotron 的基准也是用真实规模模型测 MFU，而不是 toy。）
- 另一个正确观察：双卡 loss（≈3.6）与单卡（≈3.3）不同——**有效 batch 变大 + 数据分片不同**，
  loss 不可直接比；要对比请固定有效 batch 与 sampler 种子做对照实验。

## 学完本章你能...

- ✅ 不看资料写出 DDP 五件套，并说出两个高频 bug（shuffle 双打乱、忘 set_epoch）
- ✅ 推导"all-reduce 平均 == 大 batch 梯度"，指出三个前提（等大分片/均值 loss/同起点）
- ✅ 画出桶化 all-reduce 与 backward 重叠的时序，解释"多卡近线性加速"从哪来
- ✅ 正确组合梯度累积与 no_sync，说清为什么省的是 K−1 次且最后一次不能省
- ✅ 对"多卡吞吐数字"保持警惕：toy 模型的加速比不可信，看 MFU 要用真实规模

**课后练习**

<details>
<summary>Q1: DDP 构造时为什么要 broadcast 参数？不广播会怎样？</summary>
A: 保证所有 rank 从同一份权重出发。不广播的话各 rank 随机初始化各练各的——虽然每步
all-reduce 平均梯度理论上会逐渐趋同，但早期"观点分裂"浪费算力，且有 BN 统计量等状态
不一致的实际问题。广播一次换来全程一致，成本可忽略。
</details>

<details>
<summary>Q2: 梯度累积 accum=4 时，loss 要不要除以 accum？为什么脚本里除的是 (loss/accum).backward()？</summary>
A: 要。PyTorch 的 backward 是把梯度【累加】到 .grad 上，累积 4 个 micro-batch 的目的是
模拟"一个 4 倍大的 batch"——大 batch 的梯度是子 batch 梯度的平均，所以每个 micro-batch
backward 前先除以 accum（或最后统一除）。不除的话等效学习率放大约 4 倍。
</details>

<details>
<summary>Q3: DDP + BatchNorm 有什么特殊行为？为什么 LayerNorm 没这个问题？</summary>
A: BN 的 running stats 是 buffer：DDP 默认每次 forward 前 broadcast rank0 的 buffer 同步它。
但注意 BN 的 batch 统计仍是各 rank 自己的 batch 的（跨卡不同步统计，除非用 SyncBN）。
LayerNorm 按样本内归一化、没有 running stats，天生无此问题——这也是现代 LLM 全用 LN/RMSNorm
的工程红利之一（呼应 Part 7 RMSNorm 一章）。
</details>

## 🛠️ 动手实践（依托脚本 02；动手 2 纯 CPU 可做）

### 动手 1：扫描 micro-batch × accum 组合——吞吐和 loss 到底哪个会变

**任务**：复制 `scripts/02_ddp_gpt.py` 为 `02_lab.py`，固定有效 batch = 64（world_size=2），
跑三组组合并填表：

| 组合 | batch_size | accum | 有效 batch | 每次 optimizer.step 的 all-reduce 轮数 |
|---|---|---|---|---|
| A | 32 | 1 | 64 | 1（每个 micro-step 都同步） |
| B（脚本默认） | 16 | 2 | 64 | 1（前 1 步 no_sync） |
| C | 8 | 4 | 64 | 1（前 3 步 no_sync） |

**步骤提示：**
1. 每组只改两行：`batch_size=16` 与 `accum = 2`；
2. 打印行已经是变量拼接（`batch × accum × world`）——改完上面两行参数它会自动跟上，
   无需再手改（顺带的小教训：日志要跟着参数走，硬编码的输出会"说谎"）；
3. `torchrun --standalone --nproc_per_node=2 02_lab.py` 每组跑一遍，抄"平均 loss"和
   "本 rank 吞吐"两行。

**验收标准：**
- [ ] 三组打印的有效 batch 都是 64（`bs × accum × world`，打印行用变量拼接、自动对上）
- [ ] 三组的平均 loss 相差 **< 0.1**——同有效 batch 下梯度数学等价（§2/§4），残差只来自
      浮点求和顺序与不同的数据分组。本机实测（RTX 4090×2, torch 2.6.0+cu124, NCCL）：
      A/B/C = 3.687 / 3.607 / 3.617
- [ ] 吞吐落在 §5 的 77k–95k tokens/s 带内（本机实测 86.9k / 90.1k / 94.0k——accum 越大
      通信越少、吞吐略升，但 toy 规模下差距与 run-to-run 波动同量级）。能说出"这个
      实验看不出通信收益、要换什么规模才看得出"，也算过关。

### 动手 2：10 行代码验证 §2——"平均的平均 == 总平均"，以及不等大时翻车

**任务**：验证等大分组的"平均的平均"等于总平均；再构造不等大分组看误差冒出来；最后
用样本数加权救回来。单进程 CPU 即可（fp32）。

**步骤提示（骨架）：**

```python
import torch, torch.nn as nn
torch.manual_seed(0)
lin = nn.Linear(4, 4)

def grad_of(x, y):                     # 返回这次 backward 后的权重梯度
    lin.zero_grad()
    nn.functional.cross_entropy(lin(x), y).backward()
    return lin.weight.grad.clone()

x, y = torch.randn(16, 4), torch.randint(0, 4, (16,))
g_full = grad_of(x, y)                 # 16 个样本一次算：总平均（基准）
# Step 1: 等大两半 8+8 → (g1+g2)/2，打印 max|Δ| 与 allclose(atol=1e-6)
# Step 2: 不等大两半 4+12 → 同样等权平均，打印 max|Δ|
# Step 3: 不等大但按样本数加权 (4·g1 + 12·g2)/16，再打印 max|Δ|
```

**验收标准：**
- [ ] 等大 8+8：`(g1+g2)/2` 与 `g_full` `allclose(atol=1e-6)` 为 True（本机实测
      max|Δ| ≈ 1.5e-8，CPU fp32）
- [ ] 不等大 4+12 等权平均：max|Δ| 跳到 **~0.1 量级**（本机实测 0.118，比等大情形大
      约 7 个数量级）——4 个样本占了 50% 的权重
- [ ] 不等大但按样本数加权：max|Δ| 回到 1e-8 量级、allclose 为 True——说清 §2 表格里
      "等大"前提平时替你兜住了什么（DistributedSampler 为什么要切得等大）

## 📝 课后作业

👉 [Assignment 10](../../../assignments/assignment_10/) 题 3（亲手证明 DistributedSampler 不重不漏）

## 下一步

DDP 解决"数据装不下"，但**模型状态**（参数+梯度+优化器）仍然是每卡一份。
7B 模型 × 16 字节/参数 = 112GB——下一章算清这本账，并用 ZeRO/FSDP 把它切开。

👉 [03 — 显存账本与 ZeRO/FSDP](03_memory_zero_fsdp.md)
