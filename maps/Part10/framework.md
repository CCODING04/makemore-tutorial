# Part 10: 分布式训练
## 01. 为什么并行 + 分布式 Hello World
- 并行分类
- rank/world_size 心智模型
- 集合通信原语
- torchrun 报错 FAQ
## 02. DDP：数据并行深入
- 梯度平均
- 桶化 all-reduce
- DistributedSampler
- no_sync
- 吞吐实测
## 03. 显存账本与 ZeRO/FSDP
- 16Ψ 公式
- ZeRO 1/2/3
- FSDP 实战与显存对比