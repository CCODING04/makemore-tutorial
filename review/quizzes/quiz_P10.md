# 测验 · Part 10（分布式训练：DDP、ZeRO/FSDP、TP/PP）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** 什么是 SPMD 心智模型？预测四个集合通信原语的输出，并说出 DDP 梯度同步的语义为什么是"平均不是求和"。

- **答案**：SPMD = 一份程序被 torchrun 复制成 N 个进程同时执行，唯一区别是环境变量里的 rank；进程间不传消息而做集体对齐的集合通信（少调一次会永远卡住不报错）。四原语：broadcast（rank0 → 所有人，DDP 构造时同步权重）、all_reduce（各持一份 → 规约 → 人人有结果，DDP 梯度同步）、all_gather（人人有数据 → 拼出所有人的，FSDP forward 前收齐参数分片）、reduce_scatter（规约后按 rank 切片分发，FSDP backward 分片梯度）。脚本 01 实测：rank i 持 i+1，all_reduce(SUM) 后人人得到 [3,3,3,3]，再 ÷ world 得平均 1.5。DDP 要的是**大 batch 的平均梯度**：本地 loss 已是 mean，N 个 rank 的本地梯度再平均才等于 N·b 大 batch 的梯度——若求和则等效学习率放大 N 倍；gloo 不支持 ReduceOp.AVG，用 SUM 再 ÷ world。
- **锚点**：教程 01_why_and_collectives.md §"2. 心智模型：SPMD"、§"3. 集合通信字母表"

**Q2（诊断）** 一位同学的 DDP 训练"每个 epoch 数据完全相同"，另一位的数据"被打乱了两次"。各漏了/多写了什么？再答：梯度累积配 `no_sync` 时省的是哪几次通信、哪一次绝对不能省？

- **答案**：前者忘了 `sampler.set_epoch(epoch)`——DistributedSampler 用 epoch 做打乱种子，不调就每个 epoch 分片相同，等效只在 1/world 的数据上训练 N 遍。后者 DataLoader 里没写 `shuffle=False`——sampler 已负责打乱与分片，再 shuffle 就双重打乱。`no_sync`：前 K−1 个 micro-step 在 `no_sync()` 内只本地累加 `.grad`、一次通信都不发；第 K 步恢复正常 backward 才 all-reduce，它同步的是已装着前 K−1 份的 `.grad` 当前值——因为 all-reduce 是线性算子，延后合法，省 K−1 次。**最后一次不能省**：optimizer.step 前必须拿到全体平均；忘了切回正常 backward = 梯度永远是本地值，训练静默走歪不报错。
- **锚点**：教程 02_ddp.md §"1. DDP 五件套"、§"4. 梯度累积：no_sync"

**Q3（数字）** 混合精度 AdamW 下每个参数占多少字节、由哪五项构成？7B 模型在 N=8 卡下，DDP / ZeRO-1 / ZeRO-2 / ZeRO-3 每卡模型状态各多少 GB？写出三阶段的每卡公式与通信代价。

- **答案**：**16 字节/参数**：bf16 参数 2 + bf16 梯度 2 + fp32 master 权重 4 + fp32 动量 4 + fp32 方差 4。7B × 16 = 112 GB——这就是"7B 在 24G 卡上训不动"的完整答案（bf16 参数本身只要 14 GB，AdamW 状态一加就爆）。N=8：DDP **112** / ZeRO-1 **38.5** / ZeRO-2 **26.25** / ZeRO-3 **14** GB（70B 对应 1120 / 385 / 262.5 / 140 GB，GB 按 1e9 字节）。公式：ZeRO-1 切优化器状态 $4\Psi + 12\Psi/N$（通信与 DDP 相同，"免费午餐"）→ ZeRO-2 再切梯度 $2\Psi + 14\Psi/N$（通信仍同 DDP）→ ZeRO-3/FSDP 再切参数 $16\Psi/N$（通信 ≈1.5× DDP：forward 前 all-gather、backward 用 reduce-scatter）。激活不在这本账里，靠梯度检查点（代价约 +33% 计算）。
- **锚点**：教程 03_memory_zero_fsdp.md §"1. 模型状态显存：16 字节/参数"、"动手 1"验收标准

**Q4（数字）** 流水线并行的气泡公式是什么？p=2、m=4 时气泡占比多少？若 p=8 想让 bubble < 10%，micro-batch m 至少多大、代价是什么？1F1B 相对 GPipe 省了什么？

- **答案**：$\text{bubble} = (p-1)/(m+p-1)$。p=2、m=4：1/5 = **20%**。p=8、bubble < 10%：$7/(m+7) < 0.1 \Rightarrow m+7 > 70 \Rightarrow m > 63$，按整数 micro-batch 即 $m \ge 64$（$m=63$ 恰好 10.00%，off-by-one 警示——审计已将教程答案同步为 m ≥ 64）。代价：GPipe 下这几十个 micro-batch 的激活全部驻留 → 激活显存爆炸；**1F1B** 交错执行 forward/backward，把激活驻留从 m 份降到 p 份——大模型流水线标配，再叠梯度检查点。实证"按层切开不改变数学"：脚本 06 流水线 loss 与单进程整模型完全一致（4.380254 == 4.380254）。
- **锚点**：教程 04_tp_pp_and_beyond.md §"2. 流水线并行（GPipe / 1F1B）：按层接力"、课后练习 Q2

**Q5（对比）** 一张表对比四种并行：DDP / ZeRO-FSDP / TP / PP 各"切什么"、用什么通信、解决什么、代价是什么？给出选并行策略的决策树。

- **答案**：DDP 切数据，每步 all-reduce 梯度，解决数据太大/想加速，代价每 rank 存完整模型；ZeRO/FSDP 切模型状态（参数/梯度/优化器），all-gather + reduce-scatter，解决模型状态装不下，代价通信 ≈1.5×；TP 切单层权重矩阵（Megatron：W1 列并行、W2 行并行，f/g 共轭算子使每层 forward/backward 恰各 1 次 all-reduce；脚本 05 实测前向误差 5.96e-07），解决单层放不下/大激活，代价通信最频繁且在每层正中间无法与计算重叠——只适合机内 NVLink；PP 按层分组，点对点传激活，解决层数太多装不下，代价气泡 (p−1)/(m+p−1)。决策树：模型状态 16Ψ 装得下 → DDP；不够 → ZeRO-1（免费）→ ZeRO-2 → FSDP/ZeRO-3；激活也爆 → 梯度检查点 + 减 micro-batch 配梯度累积；单层都放不下 → TP。3D 并行读配置顺序：机内 TP 优先 → PP 权衡气泡 → 剩余给 DP（TP × PP × DP = 总卡数）。
- **锚点**：教程 README.md §"一张表看懂四种并行"；03_memory_zero_fsdp.md §"3. 决策树"；04_tp_pp_and_beyond.md §"1. 张量并行（Megatron 式）"、§"3. 拼起来：3D 并行与工业栈"

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 预测 broadcast / all_reduce / all_gather / reduce_scatter 的输出，说清 DDP"平均不是求和"（01 章） | Q1 | torchrun 六类报错 FAQ 见闪卡 |
| 写出 DDP 五件套与 no_sync 梯度累积，推导"多卡一步 == 大 batch 一步"，画出桶化 all-reduce 与 backward 重叠时序（02 章） | Q2 | 等价推导的三个前提与桶化重叠、toy 吞吐读数见闪卡 |
| 背出并推导 16 字节账本与 ZeRO 1/2/3 每卡公式，用决策树选并行策略（03 章） | Q3、Q5 | FSDP1/FSDP2 最小用法与实测（11.8 MB → 6.0 MB）见闪卡 |
| 画出 Megatron TP 列/行切法与 GPipe/1F1B 时间线，推导 bubble 公式，读懂 3D 并行配置（04 章，进阶可选） | Q4、Q5 | 4090+4090D 上 NCCL p2p 卡死的工程解法见闪卡 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| 三个"装不下/不够"对应哪种并行？ | 数据喂不饱/想加速 → DDP；模型状态装不下 → ZeRO/FSDP；单层放不下 → TP；层数太多 → PP；学习顺序按通信代价从小到大 DDP → ZeRO → TP/PP |
| rank / world_size / local_rank？ | rank 进程编号（0 起）；world_size 总进程数；local_rank 本机内编号（多机用）；GPU 用 NCCL、CPU 用 gloo |
| 程序卡住不报错，第一反应查什么？ | 某个 rank 没进同一个 collective（如 `if rank == 0` 分支漏了别人的集体操作）——集合通信是握手协议，少一人其余人永远等 |
| `Cannot use ReduceOp.AVG with Gloo` 怎么办？ | gloo 不支持 AVG：用 SUM 再 ÷ world（脚本 02 踩过） |
| DDP 五件套？ | init_process_group + set_device → DistributedSampler（切 dataset 不是切 batch，DataLoader shuffle=False）→ DDP(model) 自动 broadcast → 每 epoch set_epoch → destroy_process_group |
| "all-reduce 平均 == 大 batch 梯度"成立的三个前提？ | 分片不重不漏且等大（DistributedSampler + drop_last）；本地 loss 取均值（reduction='mean'）；各 rank 从同一权重出发（构造时 broadcast）——不等大时"平均的平均 ≠ 总平均"（实测 4+12 分组 max 误差 0.118） |
| DDP 怎么把通信藏进 backward？ | 按反向就绪顺序把参数分桶（默认 25MB），某桶梯度一就绪立刻异步 all-reduce，与剩余层的 backward 重叠；toy GPT 仅 628,161 参数只有 1 个桶，所以双卡吞吐（~90k×2）与单卡（~183k tokens/s）持平——小模型测加速比是误区 |
| MFU 是什么？ | Model FLOPs Utilization = 实测 FLOPS ÷ 卡的峰值 FLOPS；评估分布式训练效率的标准指标，要用真实规模模型测 |
| 激活值显存怎么估、怎么省？ | 每层约 sbh×(34 + 5·a·s/h) 字节，随 seq 有平方项；梯度检查点只存层输入、反传重算，显存降到 2sbhL，代价约 +33% 计算 |
| ZeRO-1 为什么"免费"？ | fp32 master/动量/方差只在 optimizer.step 时用，每个 rank 只更新自己分片的参数再原地写回，通信路径上没有新增集合操作；切掉的 12 字节/参数是纯存储负担 |
| DDP vs FSDP 一句话？ | DDP 每 rank 全量存储、只 all-reduce 梯度；FSDP 全分片，forward 前 all-gather 当前层参数、算完即释放，backward 用 reduce-scatter——用 1.5× 通信换单卡装得下 |
| FSDP1 与 FSDP2 的最小用法？ | FSDP1：`FSDP(model, auto_wrap_policy=..., sharding_strategy=FULL_SHARD, use_orig_params=True)`；FSDP2：先对每个 Block `fully_shard(layer)` 再 `fully_shard(model)`（参数变 DTensor）。实测 11.8 MB 模型分片后本 rank 6.0 MB |
| TP 为什么只适合机内？ | TP 的 all-reduce 在每一层正中间、前后都是依赖它的计算，遮不住；DDP 的通信可按桶与反传重叠。机内 NVLink ~900GB/s 级，跨机 25-100GB/s 会吃掉大半 MFU |
| Megatron MLP 的 f/g 算子？ | f：forward 恒等 / backward all-reduce；g：forward all-reduce / backward 恒等；W1 按输出维列并行（gelu 在分片内部，无通信）、W2 按输入维行并行后 all-reduce |
| 4090+4090D 混合机上 NCCL p2p 卡死怎么办？ | 集合通信正常但 send/recv 互相卡死：为点对点单独建 gloo 组、张量过 CPU 中转（脚本 06 实测解法）——分布式问题不总是逻辑 bug |
| 32 卡（4 台 × 8 卡）训 70B 怎么分？ | TP=8 占满机内 NVLink → 剩 4 个流水线单元 → PP=2 × DP=2（或 PP=4 × DP=1），按激活显存与气泡权衡配 1F1B，DP 部分叠 ZeRO-1；LLaMA 2 70B 官方报告 MFU ≈ 46% |
