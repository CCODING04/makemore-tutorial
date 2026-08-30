# 01 — 为什么并行 + 分布式 Hello World

> 🧭 单卡到多卡，最难跨过的不是 API，而是**心智模型**：同一份代码被 N 个进程同时执行。
> 本章先建立这个模型，再用四个集合通信原语（脚本 01 亲手验证）补上分布式训练的"字母表"，
> 最后给一份 torchrun 常见报错 FAQ——01 章不卡人，是硬性设计目标。

## 📖 前置知识

- Part 8 的训练循环（知道 optimizer.step() 干什么即可）

## 1. 三个"装不下/不够"，三种并行

| 痛点 | 症状 | 药方 |
|---|---|---|
| 数据喂不饱 / 想要加速 | GPU 利用率高但训完要一个月 | **数据并行 DDP**（每卡不同数据，梯度同步） |
| 模型状态装不下 | 7B 模型 AdamW 要 16B/参数 → 7B×16=112GB | **ZeRO / FSDP**（把模型状态切片） |
| 单层激活/权重放不下 | 巨大 embedding、超宽 FFN | **张量并行 TP**（把一层矩阵切开算） |
| 层数太多装不下 | 406M×24 层 + 激活 | **流水线并行 PP**（按层分组接力） |

真实 LLM 训练是它们的组合（"3D 并行"）：TP × PP × DP，LLaMA 2 70B = 8 路 TP × ... 
（04 章细讲）。学习顺序按"通信代价从小到大"：DDP → ZeRO/FSDP → TP/PP。

## 2. 心智模型：SPMD

分布式训练的主流形态是 **SPMD**（Single Program, Multiple Data）：

```
你只写一份程序，torchrun 把它复制成 N 份进程同时启动：
  进程 0（rank=0）：跑 main()
  进程 1（rank=1）：跑同一个 main()
  ...
唯一区别：每个进程从环境变量里读到的 rank 不同 → 用 rank 决定"我处理哪份数据/哪片权重"
进程之间不传"消息"，而是做【集合通信】（collective）——N 个进程对齐的集体操作
```

- 🔑 `world_size` = 总进程数；`rank` = 进程编号（0 起）；`local_rank` = 本机内的编号（多机时用）。
- 初始化：`dist.init_process_group(backend)`。backend 选择：GPU 用 **NCCL**（快），
  CPU 用 **gloo**（兼容性好）。torchrun 注入 `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`，
  `env://` 方式自动读取——这就是为什么脚本第一行经常是 `os.environ["RANK"]`。
- 脚本 01 的 `setup()` 展示了"单进程也能跑"的技巧：没有 RANK 时自己伪装 world_size=1。
  本课全部脚本兼容两种启动，这也是给你自己的分布式代码调试建议：**先用单进程调通逻辑，再加卡**。

## 3. 集合通信字母表（跑 `scripts/01_distributed_basics.py`）

| 原语 | 语义 | 训练中的用途 |
|---|---|---|
| `broadcast` | rank0 → 所有人 | 开场同步权重；DDP 构造时 broadcast state_dict |
| `all_reduce` | 所有 rank 各持一份 → 规约（SUM/AVG…）→ 人人有结果 | **DDP 梯度同步**（默认等价于取平均） |
| `all_gather` | 人人有数据 → 拼出所有人的 | FSDP forward 前收齐参数分片 |
| `reduce_scatter` | 规约后按 rank 切片分发 | FSDP backward 分片梯度 |

- 🔑 **最容易错的一点**：DDP 的 all-reduce 语义是**平均不是求和**（脚本里 SUM 后 ÷N 演示）。
  所以"多卡有效 batch = 每 rank batch × N"时，学习率的参照系是有效 batch，不是单卡 batch。
- ⚠️ **集合通信是集体对齐的**：任何一个 rank 少调一次 all_reduce，其他 N-1 个进程会
  **永远等它**（表现为程序卡住不报错）。调试卡死第一反应：各 rank 的执行路径是否分叉了。

脚本 01 的实测（双卡）：

```
[2] all_reduce(SUM): rank i 持 i+1 → 人人得到 [3,3,3,3]
    all_reduce(SUM)/world = 梯度平均 [1.5, ...]
[4] reduce_scatter(SUM): 规约后按 rank 切片，rank0 得 [1, 3]
```

## 4. torchrun 报错 FAQ（01 章必读，省你一晚上）

| 症状 | 原因 | 解法 |
|---|---|---|
| `Connection reset by peer` / 卡在 init | MASTER_PORT 被占（上个进程没退干净） | `--master_port=29501` 换端口；`pkill -f torchrun` |
| `NCCL error` / invalid usage | 驱动/拓扑/容器 IPC 问题 | 先换 `backend='gloo'` 定位是不是 NCCL 的锅 |
| 程序卡住不报错 | 某 rank 没进同一个 collective | 检查 if rank==0 分支里是否漏了别人的集体操作 |
| `Cannot use ReduceOp.AVG with Gloo` | gloo 不支持 AVG | 用 `SUM` 再 `/world`（本课脚本 02 踩过） |
| 两台机器连不上 | MASTER_ADDR 写了 localhost | rank0 的真实 IP；防火墙放行 MASTER_PORT |
| 每个 epoch 数据完全相同 | 忘了 `sampler.set_epoch(epoch)` | DistributedSampler 用 epoch 做打乱种子 |

> 💡 本课机器实测还有一个彩蛋坑：这台 4090+4090D 的混合机型上 **NCCL 的 send/recv 点对点
> 会互相卡死**（脚本 06 真踩到），而集合通信正常。脚本 06 的解法是给点对点单独建 gloo 组、
> 张量过 CPU 中转——遇到类似"集合通信正常但 p2p 卡死"，这是可抄的工程解法。

## 学完本章你能...

- ✅ 说出四种并行各自解决的"装不下/不够"，以及学习顺序为什么是 DDP→ZeRO→TP/PP
- ✅ 解释 SPMD：一份代码、N 个进程、rank 区分身份、集合通信对齐
- ✅ 手写四个集合通信原语的语义（并用脚本 01 验证过）
- ✅ 背出 torchrun 六大常见报错的"症状→解法"

**课后练习**

<details>
<summary>Q1: world_size=4，每 rank local batch=8，DDP 一步等效于单卡多大的 batch？梯度是什么的梯度？</summary>
A: 等效 32。all-reduce 平均后，每个参数的梯度 = 4 个 rank 各自 batch=8 梯度的平均 = 完整
batch 32 的平均梯度。所以 DDP 不是"4 个独立训练"也不是"求和"，而是"一个大 batch 被切开算"。
</details>

<details>
<summary>Q2: 为什么集合通信"少调一次"是卡死而不是报错？</summary>
A: all_reduce 等操作是 N 个进程的握手协议：每个 rank 到齐才开始交换。少一个人，其余人
的等待永不超时（默认无超时或超长）。这也是为什么"分 rank 的 if 分支"里要格外小心——
分支里也要保证集体操作的对齐。
</details>

<details>
<summary>Q3: gloo 和 NCCL 各是什么？什么时候必须用哪个？</summary>
A: NCCL 是 NVIDIA 的 GPU 集合通信库（走 NVLink/PCIe/RDMA，GPU 张量最快）；gloo 是
PyTorch 的通用后端（CPU/GPU 都行，兼容性最好）。GPU 集合通信首选 NCCL；CPU 或排障时
用 gloo。点对点 send/recv 在个别机器拓扑上 NCCL 有坑，可建 gloo 组兜底（脚本 06 实测）。
</details>

## 📝 课后作业

👉 [Assignment 10](../../../assignments/assignment_10/) 题 1（all-reduce 语义）

## 下一步

字母表齐了。现在把最常用的原语 all-reduce 用在真正的大事上——**多卡训练一个 GPT**，
并搞懂 DDP 内部怎么把通信藏进 backward 的间隙里。

👉 [02 — DDP：数据并行深入](02_ddp.md)
