# Assignment 10：分布式训练

> 对应 Part 10 教程：[courses/Part10_distributed/tutorial/](../../courses/Part10_distributed/tutorial/README.md)
> 参考项目：[pytorch/examples minGPT-ddp](https://github.com/pytorch/examples/tree/main/distributed/minGPT-ddp) · [nanotron](https://github.com/huggingface/nanotron)

## 🎯 作业目标

分布式的核心难点**不是 API，而是四件"看不见"的事**：

1. **梯度语义**——多卡一步到底等效于什么（平均！）
2. **显存账本**——16 字节/参数的五项构成与 ZeRO 切分
3. **数据分片**——DistributedSampler 的不重不漏
4. **切分数学**——TP 为什么等价于稠密计算、流水线气泡怎么算

这份作业**全部纯 CPU / 纸上推导可完成**——这正是设计意图：分布式最难的部分是"看不见"
的逻辑，先把它变成看得见的数字。

## 📋 完成方式

```bash
cd assignments/assignment_10
python test_distributed_exercises.py     # 或 pytest test_distributed_exercises.py
```

先跑一遍脚本 01-06（单进程即可，有 GPU 用 `torchrun --standalone --nproc_per_node=2`），
对题目里的每个数字建立直觉，再回来做题。

## 📝 题目列表

### 题 1：all_reduce 平均语义（20 分）——`ddp_gradient` / `effective_batch`

- `ddp_gradient(grads_per_rank)`：4 个 rank 的本地梯度 [1,2,3,4]，DDP 同步后每个 rank 上的
  梯度是多少？（**平均不是求和**——这是本作业最想让你记住的一个数）
- `effective_batch(local_batch, accum_steps, world_size)`：有效 batch = 三者乘积

### 题 2：显存账本计算器（30 分）——`model_state_bytes` / `can_train_7b_on_24gb`

按公式实现四阶段（Ψ=参数量，N=卡数）：

```
DDP     = 16Ψ          ZeRO-1 = 4Ψ + 12Ψ/N
ZeRO-2  = 8Ψ + 4Ψ/N    ZeRO-3 = 16Ψ/N
```

- `can_train_7b_on_24gb(8)`：7B 模型 + ZeRO-3 + 6GB 激活，8 张 24GB 卡能不能训？

<details>
<summary>💡 反直觉预警（测试会查）</summary>

ZeRO-1 和 ZeRO-2 **谁更省取决于 N**：zero1=4+12/N、zero2=8+4/N，N>2 时 zero1 更小！
"ZeRO 数字越大越省"只在"每阶段切换点"上对，具体 N 下要真的算。这就是"账本"的价值。
</details>

### 题 3：DistributedSampler 不重不漏（30 分）——`sampler_indices` / `sampler_coverage_ok`

模拟 torch 的 DistributedSampler：补齐到整除 → 按种子打乱 → `padded[rank::world_size]` 等间隔切片。

`sampler_coverage_ok` 验证三条性质：各 rank 长度相等 / 并集覆盖 0..n-1 / **每个 rank 内部
无重复**（补齐样本只会落在末尾某些 rank，不会与同 rank 原有样本撞车——想想为什么）。

<details>
<summary>💡 这题和真实 bug 的关系</summary>

忘了 `sampler.set_epoch(epoch)` = 每个 epoch 用同一个 seed = 数据分片完全不变 = 每个
模型实际上只在 1/N 的数据上反复训练。这也是面试题"DDP 训练 loss 降得慢的常见原因"。
</details>

### 题 4：TP 分块数学（10 分）——`tp_mlp_max_error`

纯 CPU 模拟 Megatron 式列/行并行（对照脚本 05）：把 W1 按行切、W2 按列切，
分别算 `gelu(X@W1_r.T)@W2_r.T` 再求和——验证与稠密计算的 max 误差 < 1e-5
（脚本 05 双卡实测 ~6e-7）。

### 题 5：🌟 流水线气泡（10 分）——`pipeline_bubble_fraction` / `in_flight_activations`

- bubble = (p-1)/(m+p-1)；验证 p=2,m=4 → 20%，且 m 越大气泡越小
- `in_flight_activations('gpipe', m, p)` = m；`'1f1b'` = p（1F1B 用激活驻留换气泡）

## 🤔 思考题

**Q1：DDP、ZeRO-1、ZeRO-2、ZeRO-3、TP、PP——如果老板让你用 32 台 8 卡机训 70B，
给出一套配置并说清每一维为什么这么选。**

<details>
<summary>💡 提示</summary>

机内 TP=8（NVLink 扛 TP 的高频 all-reduce）→ 32 个"流水线单元"；模型 80 层 → PP=4 或 8
（配 1F1B 控制气泡）；剩下全给 DP（DP=8 或 4）叠 ZeRO-1；激活装不下就梯度检查点。
面试考察的是"机内 TP 优先 → 气泡与显存权衡 → 剩余给 DP"的推理链，不是唯一答案。
</details>

**Q2：DDP 训练时 loss 曲线比单卡"看起来"差一点，可能的原因有哪些（至少 3 个）？**

<details>
<summary>💡 提示</summary>

① 有效 batch 变大 → 同样 epoch 数下参数更新次数变少；② 忘了 set_epoch → 只见 1/N 数据；
③ lr 没随有效 batch 调整（线性缩放规则）；④ 打印的是本 rank 的 loss、样本不同本来就有
波动（应 all-reduce 平均后再对比）。
</details>

## ✅ 提交检查清单

- [ ] `python test_distributed_exercises.py` 5/5 ✅
- [ ] 跑过 `torchrun --standalone --nproc_per_node=2` 的脚本 02（或 CPU 双进程），看过自己的多卡吞吐
- [ ] 能背出 16Ψ 的五项构成，现场算 7B 的 DDP 显存
- [ ] 能向别人解释：为什么 ZeRO-1 和 ZeRO-2 谁更省取决于 N（费曼检验）
