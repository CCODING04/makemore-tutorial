# 02 — DDP：数据并行深入

> 🧭 DDP（DistributedDataParallel）是所有分布式训练的地基，也是 90% 场景的正确起点。
> 本章走完它的五个必要件，看穿内部机制（桶化 all-reduce 与 backward 重叠），
> 并用双卡实测吞吐。跑 [scripts/02_ddp_gpt.py](../scripts/02_ddp_gpt.py)。

## 📖 前置知识

- **01 章**：SPMD 心智模型、all_reduce 平均语义、torchrun

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

## 2. 内部机制：通信是怎么"藏"进 backward 的

朴素实现是"forward → backward 全算完 → all-reduce 所有梯度 → step"，通信完全串行。
DDP 的真实做法（[设计笔记](https://docs.pytorch.org/docs/stable/notes/ddp.html)）：

```
构造时：按参数的反向传播【就绪顺序】把参数分桶（默认桶 25MB）
backward 时：某个桶的全部梯度一就绪 → 立刻异步 all-reduce 这个桶
             ↓
       通信与"剩余层的 backward 计算"重叠 → 大部分通信时间被计算盖住
全部桶发起后，backward 结束前才阻塞等最后一个桶
```

- 🔑 这就是"多卡吞吐接近线性"的原因：通信不是没有，而是**被藏起来了**。
- 两个相关参数：`find_unused_parameters=True`（有未参与 loss 的参数时防挂死，有遍历开销）；
  `broadcast_buffers=True`（默认开：每次 forward 前 rank0 的 buffer 广播给所有 rank——
  BatchNorm 的 running stats 靠它同步。这也解释了 DDP+BN 的行为）。

## 3. 梯度累积：no_sync

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

实测收益：累积 2 步时通信量减半（脚本 02 默认 accum=2）。

## 4. 实测（2×4090，脚本 02）

```
world_size=2: 每 rank batch=16 × accum=2 → 有效 batch = 64
  本 rank 吞吐: ~78,500 tokens/s（含双卡各自）
world_size=1: 有效 batch = 32
  本 rank 吞吐: ~199,500 tokens/s
```

- 💡 怎么读这组数字：单卡 tokens/s 是"一张卡的产能"；双卡世界总产能 = 78.5k × 2 = 157k，
  比单卡 199k 低——因为这个 toy 模型太小（计算 0.2s 就完了，启动/通信占比高）。
  **真实大模型上多卡才能体现价值**；小模型测分布式加速比是新手常见误区。
  （nanotron 的基准也是用真实规模模型测 MFU，而不是 toy。）
- 另一个正确观察：双卡 loss（3.63）与单卡（3.29）不同——**有效 batch 变大 + 数据分片不同**，
  loss 不可直接比；要对比请固定有效 batch 与 sampler 种子做对照实验。

## 学完本章你能...

- ✅ 不看资料写出 DDP 五件套，并说出两个高频 bug（shuffle 双打乱、忘 set_epoch）
- ✅ 画出桶化 all-reduce 与 backward 重叠的时序，解释"多卡近线性加速"从哪来
- ✅ 正确组合梯度累积与 no_sync
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

## 📝 课后作业

👉 [Assignment 10](../../../assignments/assignment_10/) 题 3（亲手证明 DistributedSampler 不重不漏）

## 下一步

DDP 解决"数据装不下"，但**模型状态**（参数+梯度+优化器）仍然是每卡一份。
7B 模型 × 16 字节/参数 = 112GB——下一章算清这本账，并用 ZeRO/FSDP 把它切开。

👉 [03 — 显存账本与 ZeRO/FSDP](03_memory_zero_fsdp.md)
