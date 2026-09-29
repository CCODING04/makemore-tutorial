# 04 — 稳定与提速：自动混合精度（AMP）、GradScaler 与 DistributedSampler（v3）

> 🧭 问题清单第 ① 条：fp32 训练慢、显存高。本章做一组教科书级的对照实验——**同一个模型、
> 同一份数据、同一个种子，只换数值格式**（fp32 / fp16+GradScaler / bf16），亲眼看到精度换速度
> 的账；再把梯度裁剪、梯度累积、双卡 DDP 逐件装上。这些件不改变模型"是什么"，但决定你
> **能不能便宜、稳定地把它训出来**——工业训练的稳定性半壁江山在这章。

## 🎒 前置回忆包（不翻旧章也能读）

- **浮点格式**：fp32 = 1 符号 + 8 指数 + 23 尾数；fp16 = 1+5+10；bf16 = 1+**8**+7。
  指数位决定"能表示多大/多小的数"，尾数位决定"精度多细"。
- **反向传播的梯度**常是极小的数（1e-7 量级很常见）——记住这个，GradScaler 的存在理由就是它。
- **DDP**（分布式数据并行）：每张卡持完整模型副本，各吃一份数据，反向后梯度 all-reduce 平均。

## 📍 你现在的位置

```text
v2 会做实验的基线 ──本章──▶ v3：训练工程齐备（快 1.8×、稳定、可双卡）
```

## 实验 ①：fp32 vs fp16+GradScaler vs bf16（同 seed 三连）

```bash
python 04_stability_speed.py        # 三次 120 步短训，自动出对比表
```

实测（RTX 4090，batch 16×accum 2，seq 340）：

```text
  格式            末段 loss   峰值显存
  fp32            6.4355    2.6 GB
  fp16+scaler     6.5192    2.4 GB
  bf16            6.5287    2.4 GB
```

稳态吞吐（`logs/s3_amp_*.csv` 的 `tok_s` 列，26M 模型）：

```text
  fp32          ≈ 125k tok/s
  fp16+scaler   ≈ 228k tok/s     ┐ 1.8×
  bf16          ≈ 229k tok/s     ┘
```

🔑 **loss 三者同水平、速度 1.8 倍**——这就是全行业切到混合精度的原因。三个数字背后是三种
"梯度怎么算"的策略，逐个拆开：

## 第 1 件：GradScaler——fp16 的"保险丝"

**为什么 fp16 需要 scaler**：反向传播算出的梯度经常是 1e-7 量级，而 fp16 能表示的最小正规数
约 6e-5——**大量梯度直接下溢成 0**，深网络尤其严重（越靠前的层梯度越小）。GradScaler 的
解法朴素有效：

```python
scaler = torch.amp.GradScaler('cuda', enabled=(amp == 'fp16'))   # 初始 scale = 65536 (2^16)
with torch.autocast('cuda', dtype=torch.float16):
    loss = model(xb, yb) / accum
scaler.scale(loss).backward()        # ① 放大 loss → 反向传的是放大 65536 倍的梯度（不溢出了）
if scaler.is_enabled():
    scaler.unscale_(optimizer)       # ② 裁剪/更新前把梯度除回真实值
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)              # ③ 若本步出现过 inf/nan：跳过整步（不更新）
scaler.update()                     # ④ 有 inf/nan → scale 减半；连续 2000 步干净 → scale×2
```

四个要点：① scale 初始 65536，把 1e-7 的梯度抬进 fp16 可表示区间；② **unscale 必须发生在
裁剪之前**——不然裁的是放大 6 万倍的范数，等于没裁；③ 某步出现 inf/nan，`step` 自动跳过、
`update` 把 scale 减半——**训练不会死，只会回退一小步**；④ scale 是动态的，`grad_scale`
这条曲线本身就是训练健康度信号（03 章记进 CSV 的就是它）。

本实验没触发溢出（scale 恒 65536）——**小模型+裁剪+中等 lr 下 fp16 通常安全；深网络/大 lr/
长训练才见保险丝出手**。但 scaler 是免费的上保险，官方 train_pretrain.py 的 fp16 路径一直挂着它。

## 第 2 件：bf16——为什么现代默认不需要 scaler

bf16 的尾数只有 7 位（精度粗），**但指数位 8 位与 fp32 完全相同**——能表示的范围和 fp32 一样大，
梯度再小也不会下溢。代价是精度粗，可 LM 训练对此不敏感（累加在 fp32 主权重上进行，
autocast 只把矩阵乘降精度）。于是 **bf16 = fp16 的速度 + 免 scaler 的省心**：官方
`--dtype bfloat16` 默认，本课 quick 档同样。一句记忆：**fp16 怕小（下溢），bf16 不怕小怕粗**。

## 第 3 件：梯度裁剪与梯度累积——稳定性的"一紧一松"

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 裁剪：全局范数 >1 就等比缩回
```

- **裁剪**防"一步走飞"：LM 训练偶发梯度尖峰（坏数据/数值抖动），把全局范数拉回 1 以内。
  注意在 AMP 下要先 `unscale_`（见上）。
- **累积**是"穷人的大 batch"：

```text
教学档 batch 16 × accum 2 = 有效 32；官方 batch 32 × accum 8 = 有效 256
显存只付 batch 16/32 激活的钱，梯度噪声却按 256 压——等效大 batch 是「穷人法宝」
```

  loss 要**先除 accum 再 backward**（累积的是各 micro-batch（每次实际前向的一小批）的 loss 之和，除过才是均值）；
  裁剪和 step 只在累积周期末做一次。

## 第 4 件：DistributedSampler——双卡 DDP 的数据分发官

单卡训练自己 shuffle；双卡 DDP 必须**每张卡看到不同的、合起来恰好一遍的数据**——这就是
`DistributedSampler` 的全部工作，也是它三个关键参数的由来：

```python
sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                             shuffle=True, drop_last=True)
#  shuffle=True   → 不是各卡随便切，而是"全数据先统一洗牌，再按 rank 交错分"
#  drop_last=True → 总量不整除 world×batch 时丢尾：两卡的 batch 数必须相等，否则 DDP 在
#                   all-reduce 处互等死锁——这是新手 DDP 卡死的第一原因
#  set_epoch(e)   → 每个epoch 调一次：洗牌种子=f(seed, epoch)，不调它每个 epoch 顺序都一样
```

真机演示（本机 2×4090，`torchrun --nproc_per_node=2 04_stability_speed.py --ddp`）：

```text
[rank 0] epoch0 前 3 个样本下标: [44, 3559, 407]
[rank 0 视角] rank1 的前 3 个: [961, 3372, 209] → 两张卡看到不同数据，各训一半
[rank 0] set_epoch(1) 后前 3 个: [3845, 1784, 3755]     ← 不调 set_epoch 这行永远不变
[rank 0] 120 步 × batch 16 ｜ 5.9s ｜ 本卡 tok/s 111277
[rank 1] 120 步 × batch 16 ｜ 5.9s ｜ 本卡 tok/s 111274
```

🔑 **反直觉的诚实结论**：单卡 bf16 ≈ 229k tok/s，双卡合计 ≈ 223k——**没快**。26M 模型的
计算量太小，每步反向后的梯度 all-reduce 通信开销把收益吃光了。**DDP 的收益随模型变大才
显现**——"加卡就快"是有前提的，这笔账在 [Part 10 分布式](../../Part10_distributed/tutorial/README.md)
展开成整整一章。

## 第 5 件：checkpoint——断点续训存什么

`save_model` 只存推理权重（fp16）。官方还存**续训状态**，且写盘用原子替换：

```python
# 官方做法（train_pretrain.py 节选思路）：权重 + AdamW 动量 + scaler + epoch/step 一起存
torch.save({...}, path + '.tmp'); os.replace(path + '.tmp', path)   # 防写到一半被杀留半截文件
```

🔑 没有 AdamW 动量的"续训"会让优化器失忆、loss 抖很久——**推理权重和续训状态是两种存档**。

## ▶️ 运行本章成果

```bash
python 04_stability_speed.py        # AMP 三连对比（约 1.5 分钟）
torchrun --nproc_per_node=2 04_stability_speed.py --ddp   # 双卡演示（可选）
```

## 🏭 工业界

混合精度是标配中的标配（fp16 时代配 scaler，bf16 时代裸奔）；更大规模上 bf16 + TF32 +
`torch.compile` 层层叠加。FP8 训练（DeepSeek-V3 等旗舰已上线）是下一个精度台阶——思路仍是
"哪些计算可以更粗"。多卡数据并行的下一步是 ZeRO/FSDP（切分 optimizer 状态/参数，
[Part 10](../../Part10_distributed/tutorial/README.md)），warmup-stable-decay 调度（升温-平稳-衰减三段调度，09 章展开）与 loss
尖峰体检（回滚到 checkpoint + 跳过坏数据批）是千卡训练的日常。

## 🪝 引子

- 训练又快又稳了，仪器也齐了——下一章开始修模型本身：learned 位置表在训练长度之外会发生
  什么？（问题清单 ②）

## 🎯 面试直通车

<details>
<summary>Q: GradScaler 的机制？bf16 为什么不需要？</summary>
A: fp16 指数位 5 位，最小正规数 ~6e-5，而梯度常为 1e-7 量级会下溢成 0。GradScaler 把 loss
乘 65536，反向梯度同倍放大避开下溢，optimizer.step 前再 unscale 回真实值；遇到 inf/nan 自动
跳步并把 scale 减半，连续 2000 步干净则逐步增大。bf16 指数位 8 位与 fp32 相同，范围不缩水，
只牺牲尾数精度——LM 训练不敏感，所以不需要 scaler。
</details>

## ✅ 本章验收

- [ ] `python 04_stability_speed.py` 跑通，能报出 fp32→bf16 的加速比（≈1.8×）与"loss 同水平"这个关键事实
- [ ] 能画出 GradScaler 四步流程（scale→backward→unscale→step/update），并说出"unscale 在裁剪前"的原因
- [ ] 能解释 DistributedSampler 的 drop_last 和 set_epoch 各防什么事故
- [ ] 能解释"双卡没变快"这笔通信账

---

[← 上一章：实验仪器](03_experiment_lab.md) | [课程 README](README.md) | [下一章：位置编码进化 →](05_rope.md)
