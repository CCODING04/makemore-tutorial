# Part 9: CUDA 内核
## 01. GPU 架构与第一个内核
- CPU vs GPU
- 线程层级
- 够用的 C
- CUDA 五步曲
## 02. matmul 优化阶梯
- 算力墙/内存墙
- 合并访存
- SMEM tiling
- block tiling
- cuBLAS 对照
## 03. Profiling 与 CUDA 库
- nsys/ncu
- atomics 归约
- streams 重叠
- cuBLAS（cuDNN 概念带过）
## 04. Triton 与 PyTorch 扩展
- Triton 内核
- 自定义算子接入 PyTorch
- 通向 llm.c 的路线图
## 05. Flash Attention 毕业内核
- online softmax 推导
- causal 三阶段
- SDPA 四后端实测
- FlexAttention/SageAttention