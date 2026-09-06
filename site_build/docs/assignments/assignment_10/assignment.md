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

**验收标准：**
- [ ] `ddp_gradient([1,2,3,4])` == 2.5（平均不是求和）
- [ ] `effective_batch(16, 2, 2)` == 64

### 题 2：显存账本计算器（30 分）——`model_state_bytes` / `can_train_7b_on_24gb`

按公式实现四阶段（Ψ=参数量，N=卡数）：

```
DDP     = 16Ψ          ZeRO-1 = 4Ψ + 12Ψ/N
ZeRO-2  = 2Ψ + 14Ψ/N   ZeRO-3 = 16Ψ/N
```

- `can_train_7b_on_24gb(8)`：7B 模型 + ZeRO-3 + 6GB 激活，8 张 24GB 卡能不能训？

<details>
<summary>💡 恒等式预警（测试会查）</summary>

ZeRO 每升一阶**恒更省**：ZeRO-2 = 2Ψ+14Ψ/N 恒 ≤ ZeRO-1 = 4Ψ+12Ψ/N——
ZeRO-2 在切优化器状态（12Ψ）之外把梯度（2Ψ）也切走了，每卡只剩 bf16 参数 2Ψ 全量。
N>1 时 ZeRO-2 比 ZeRO-1 每卡再省 (2−2/N)·Ψ，**N 越大省得越多**。
自检锚点：N=1 时 DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 四条公式**全部 = 16Ψ**
（什么都切不出去），这是背公式时最好的校验点。
</details>

**验收标准：**
- [ ] 四条公式全部正确（注意 ZeRO-2 是 2Ψ+14Ψ/N，不是 8Ψ+4Ψ/N）
- [ ] 未知 stage 抛 `ValueError`
- [ ] 任意 N≥1 满足 `ddp ≥ zero1 ≥ zero2 ≥ zero3`，且 N=1 时四者全等（16Ψ）
- [ ] `can_train_7b_on_24gb(8)` 为 True、`(2)` 为 False

### 题 3：DistributedSampler 不重不漏（30 分）——`sampler_indices` / `sampler_coverage_ok`

模拟 torch 的 DistributedSampler：补齐到整除 → 按种子打乱 → `padded[rank::world_size]` 等间隔切片。

`sampler_coverage_ok` 验证三条性质：各 rank 长度相等 / 并集覆盖 0..n-1 / **每个 rank 内部
无重复**（补齐样本只会落在末尾某些 rank，不会与同 rank 原有样本撞车——想想为什么）。

<details>
<summary>💡 这题和真实 bug 的关系</summary>

忘了 `sampler.set_epoch(epoch)` = 每个 epoch 用同一个 seed = 数据分片完全不变 = 每个
模型实际上只在 1/N 的数据上反复训练。这也是面试题"DDP 训练 loss 降得慢的常见原因"。
</details>

**验收标准：**
- [ ] 每 rank 分片长度 = total/world（n=10, world=3 → 每片 4 个）
- [ ] 所有 rank 的并集覆盖 0..n-1，且每个 rank 内部无重复
- [ ] 同种子可复现、不同种子（epoch）分片不同

### 题 4：TP 分块数学（10 分）——`tp_mlp_max_error`

纯 CPU 模拟 Megatron 式列/行并行（对照脚本 05）：把 W1 按行切、W2 按列切，
分别算 `gelu(X@W1_r.T)@W2_r.T` 再求和——验证与稠密计算的 max 误差 < 1e-5
（脚本 05 双卡实测 ~6e-7）。

**验收标准：**
- [ ] n_shards=2 与 n_shards=4 的 max 误差都 < 1e-5
- [ ] 分片数不改变结果（"切分 = 稠密"的数学不变量）

### 题 5：🌟 流水线气泡（10 分，**可选**）——`pipeline_bubble_fraction` / `in_flight_activations`

- bubble = (p-1)/(m+p-1)；验证 p=2,m=4 → 20%，且 m 越大气泡越小
- `in_flight_activations('gpipe', m, p)` = m；`'1f1b'` = p（1F1B 用激活驻留换气泡）

🌟 本题为选做加分：未实现时测试打印 ⏭️ SKIP 并计为跳过（不算失败），
实现后冲 5/5。

**验收标准：**
- [ ] `pipeline_bubble_fraction(2, 4)` == 0.2，且 m 越大返回值越小
- [ ] `in_flight_activations('gpipe', 8, 4)` == 8、`('1f1b', 8, 4)` == 4

## 🎯 面试直通车（话术卡：结论 → 原理 → 边界）

> 每张卡按"总分总"组织：先一句话结论压场，再两三句原理支撑，最后一句边界/代价收尾——面试答题的固定骨架。

**Q1："DDP 的桶化 all-reduce 是怎么和 backward 重叠的？"**

- **结论**：按参数"反向传播就绪顺序"分桶（默认 25MB），某个桶的梯度一就绪就立刻异步 all-reduce，通信与剩余层的反传重叠——这是多卡近线性加速的来源。
- **原理**：朴素实现"backward 全算完 → 一次性 all-reduce → step"让通信完全串行（教程 02 章）；DDP 构造时就按反向就绪顺序把参数分桶，backward 过程中桶一满就发起异步通信，全部桶发起后才在 backward 结束前阻塞等最后一个。梯度语义是平均而非求和（课程验收 `ddp_gradient([1,2,3,4]) == 2.5`）。
- **边界**：gloo 后端不支持 `ReduceOp.AVG`，只能 SUM 再 /world（脚本 02 的写法）；桶太大等待久、太小开销高，粒度是个折中。

**Q2："ZeRO 三个阶段各切什么？7B 模型 8 卡每卡多少 GB？"**

- **结论**：16Ψ 五项（bf16 参数 2 + bf16 梯度 2 + fp32 master 4 + 动量 4 + 方差 4）里，ZeRO-1 切优化器状态、ZeRO-2 再切梯度、ZeRO-3 连参数也切；7B、N=8 时每卡 112 → 38.5 → 26.25 → 14 GB。
- **原理**：公式（教程 03 章）：ZeRO-1 $= 4\Psi + 12\Psi/N$、ZeRO-2 $= 2\Psi + 14\Psi/N$、ZeRO-3 $= 16\Psi/N$；7B 带入即参考答案那组数字。恒等式校验：ZeRO-2 恒比 ZeRO-1 省 $(2 - 2/N)\Psi$，且 N=1 时四条公式全部退化为 16Ψ（背公式的最好校验点）。ZeRO-1/2 通信路径与 DDP 相同，号称"免费午餐"。
- **边界**：ZeRO-3 通信 ≈ 1.5× DDP（forward 前 all-gather 收齐、backward 后 reduce-scatter 分发），还引入 prefetch 调度复杂度；且 112→14GB 只是模型状态，激活、NCCL 缓冲与碎片余量另算。

**Q3："流水线气泡公式是什么？p=8 时要多少个 micro-batch 才能把气泡压到 10% 以下？"**

- **结论**：bubble $= (p-1)/(m+p-1)$；p=8 想压到 10% 以下需 $m + p - 1 > 70$，即 $m > 63$（$m \ge 64$；$m=63$ 恰好 10.0%，off-by-one 警示）——深流水线必须配大梯度累积。
- **原理**：课程验收锚点 p=2、m=4 → 20%（作业 10 题 5）；m 增大气泡被摊薄。GPipe 与 1F1B 的 bubble 公式相同，区别在激活驻留：GPipe 先做完全部 forward、驻留 = m，1F1B 交错执行后稳态驻留 ≈ p（验收 `('gpipe',8,4)==8`、`('1f1b',8,4)==4`）。
- **边界**：1F1B 没有减小气泡，用"更早释放显存"换同样的 bubble；当 p 很深而 m 被激活显存卡住时 bubble 反成主要矛盾——所以深流水线标配 1F1B + 梯度检查点，再往上是 interleaved/AFAB 调度。

**Q4："TP 的列并行/行并行怎么切？为什么中间不需要通信？"**

- **结论**：Megatron MLP $Y = \mathrm{gelu}(X W_1^T) W_2^T$ 中 $W_1$ 按输出维列切、$W_2$ 按输入维行切，gelu 作用在分片内部所以前向无通信，最后对部分和做一次 all-reduce。
- **原理**：每个 rank 算 $\mathrm{gelu}(X W_{1,r}^T) W_{2,r}^T$ 的部分和，all-reduce 求和即完整 $Y$（教程 04 章）；f/g 共轭算子保证每层 forward/backward 各恰一次 all-reduce。Attention 同构：QKV 按头切是天然列并行、输出投影行并行。课程实测（脚本 05 双卡）：与稠密计算的 max 误差 ~6e-7——"切分 = 稠密"的数学不变量。
- **边界**：TP 的 all-reduce 落在每层正中间、前后都是依赖它的计算，无法像 DDP 那样与计算重叠，对带宽最敏感——所以 TP 只放机内 NVLink，跨机会把带宽吃光。

**Q5："DDP 的 all-reduce 是平均还是求和？这个语义会坑你什么？"**

- **结论**：梯度是平均，且这一步严格等价于"有效 batch = local × 累积步 × N"的单卡大 batch 一步；坑在手工汇总指标时"除错了对象"。
- **原理**：梯度对样本损失是线性的，all-reduce(SUM)/N 后每 rank 拿到的正是大 batch 的平均梯度（教程 02 章三步推导）；课程验收 `ddp_gradient([1,2,3,4]) == 2.5`、`effective_batch(16, 2, 2) == 64`。所以调 lr 的参照系是有效 batch（线性缩放规则）。
- **边界**：凡跨 rank 汇总的量（loss、acc）都要 SUM 后 /world；经典坑在流水线——只有最后一个 stage 持有 loss，机械地 /world 会把 loss 恰好砍半，数字"看起来合理"实则错了（脚本 06 记录过）。

---

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

**Q3：为什么 ZeRO-2（2Ψ+14Ψ/N）恒比 ZeRO-1（4Ψ+12Ψ/N）省？省多少随 N 怎么变？
既然"越大越省"，为什么不直接全用 ZeRO-3？**

<details>
<summary>💡 提示</summary>

ZeRO-2 在切走优化器状态（12Ψ）之外，把梯度（2Ψ）也切走了——每卡只剩 bf16 参数 2Ψ 全量，
所以恒有 zero2 = zero1 − (2−2/N)Ψ ≤ zero1，N 越大省得越多；N=1 时四条公式全部退化为 16Ψ。
但不无脑上 ZeRO-3 的原因在**通信**：ZeRO-1/2 不改变通信路径（与 DDP 相同量级，ZeRO-1 号称
"免费午餐"）；ZeRO-3 连参数也切，forward 前要 all-gather 收齐、backward 后 reduce-scatter
分发，通信量 ≈1.5× DDP，还引入 prefetch/调度复杂度。决策树因此是：显存够 → DDP；
不够 → ZeRO-1（免费）→ ZeRO-2 → FSDP/ZeRO-3，最后才动激活（梯度检查点）。
</details>

**Q4：DDP 的梯度 all-reduce 是"平均"不是"求和"。这个语义对手工汇总多卡指标有什么影响？
举一个"除错了对象"的坑。**

<details>
<summary>💡 提示</summary>

凡是要跨 rank 汇总的量（loss、acc），all_reduce(SUM) 之后都要 `/world` 才是平均值——
gloo 后端还不支持 ReduceOp.AVG，只能 SUM 再除（脚本 02 就是这么写的）。梯度平均还意味着
DDP 一步严格等价于"有效 batch = local×accum×N 的单步大 batch"，所以 lr 的参照系是有效 batch。
经典坑在流水线：只有最后一个 stage 持有 loss，汇总时如果机械地 `/world`，会把 loss 恰好
砍半——数字"看起来合理"实则错了（脚本 06 注释里记录过）。
</details>

**Q5：GPipe 和 1F1B 的 bubble 公式相同（都是 (p−1)/(m+p−1)），那 1F1B 到底省了什么？
什么时候 bubble 会成为主要矛盾？**

<details>
<summary>💡 提示</summary>

1F1B 省的是**激活驻留**：GPipe 先做完全部 m 个 forward，每个 micro-batch 的激活都要留到
backward，驻留 = m；1F1B 交错执行 forward/backward，稳态驻留 ≈ p 个——用"更早释放显存"
换同样的 bubble。bubble 随 p 增大而上升、随 m 增大被摊薄；当 p 很深而 m 又被激活显存
卡住不能加大时，bubble 成为主要矛盾——所以深流水线标配 1F1B + 梯度检查点，
再往上是 interleaved / AFAB 这类更激进的调度（nanotron 里都能读到）。
</details>

## ✅ 提交检查清单

- [ ] `python test_distributed_exercises.py`：题 1-4 全过 ✅（🌟 题 5 可选，未实现显示 ⏭️ SKIP 不算失败；实现了就冲 5/5）
- [ ] 跑过 `torchrun --standalone --nproc_per_node=2` 的脚本 02（或 CPU 双进程），看过自己的多卡吞吐
- [ ] 能背出 16Ψ 的五项构成，现场算 7B 的 DDP 显存
- [ ] 能向别人解释：为什么 ZeRO-2 恒比 ZeRO-1 省、省的量 (2−2/N)Ψ 随 N 怎么变、N=1 时四条公式重合在哪（费曼检验）
