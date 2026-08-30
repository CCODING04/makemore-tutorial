#!/usr/bin/env python3
"""
Part 10 作业：分布式训练

设计原则：全部题目纯 CPU / 纯数学可完成（分布式最难的是"看不见"——索引、显存、通信、
气泡都是可以纸上精确推导的）。实现后用 test_cuda_exercises.py 同目录的
test_distributed_exercises.py 验证。

每道题的 TODO 注释是步骤级提示；先自己推导，卡住了再对照对应脚本。
"""

import math

try:
    import torch
except ImportError:
    torch = None


# ═════════════════════════════════════════════════════════════════════
#  题 1：all_reduce 的平均语义（20 分）
# ═════════════════════════════════════════════════════════════════════

def ddp_gradient(grads_per_rank):
    """
    DDP 一步之后，每个 rank 上 param.grad 应该是什么？

    Args:
        grads_per_rank: list，每个元素是一个 float，表示该 rank 用本地 batch
                        算出的某个参数的梯度

    Returns:
        float：DDP all-reduce 之后每个 rank 上该参数的梯度
        （提示：all_reduce(SUM) 的结果再除以 world_size —— 平均不是求和）
    """
    # TODO: 一行
    return None


def effective_batch(local_batch, accum_steps, world_size):
    """
    梯度累积 + 多卡的"有效 batch"。

    Returns:
        int：local_batch * accum_steps * world_size
    """
    # TODO: 三个数相乘
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 2：显存账本计算器（30 分）
# ═════════════════════════════════════════════════════════════════════

def model_state_bytes(param_count, world_size, stage):
    """
    混合精度 AdamW 的"模型状态"显存/每卡。公式（Ψ=param_count）：
        DDP     = 16Ψ
        ZeRO-1  = 4Ψ + 12Ψ/N     （优化器状态分片）
        ZeRO-2  = 8Ψ + 4Ψ/N      （+梯度分片）
        ZeRO-3  = 16Ψ/N          （+参数也分片）

    Args:
        param_count: Ψ（参数个数，不是字节！）
        world_size:  N
        stage:       'ddp' | 'zero1' | 'zero2' | 'zero3'

    Returns:
        int/float：字节数
    """
    # TODO: 按 stage 分支返回；未知 stage 抛 ValueError
    return None


def can_train_7b_on_24gb(world_size, activation_bytes=6_000_000_000):
    """
    7B 模型（混合精度 AdamW），激活值另需 activation_bytes。
    问：给定卡数（每张 24GB = 24_000_000_000 字节），ZeRO-3 下模型状态 + 激活
        能否塞进单卡？

    Returns:
        bool
    """
    # TODO: model_state_bytes(7e9, world_size, 'zero3') + activation_bytes <= 24e9
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 3：DistributedSampler 的不重不漏证明（30 分）
# ═════════════════════════════════════════════════════════════════════

def sampler_indices(n, world_size, rank, seed=0, with_torch=None):
    """
    模拟 torch 的 DistributedSampler（drop_last=False 路径）：
      1. total = ceil(n / world_size) * world_size          # 补齐到整除
      2. 用 torch.Generator(manual_seed=seed) 生成 randperm(n) 作为基准顺序
      3. 若 total > n，从基准顺序开头循环补齐
      4. 本 rank 的索引 = padded[rank::world_size]           # 等间隔抽样

    Args:
        n: 数据集大小
        world_size, rank: 并行拓扑
        seed: 打乱种子（sampler.set_epoch 改的就是它）
        with_torch: 是否允许用 torch（None=自动）；无 torch 时用纯 Python
                    (random.Random(seed).shuffle) 模拟——测试只验证性质不验证具体排列

    Returns:
        list[int]：本 rank 的索引，长度应为 total // world_size
    """
    # TODO:
    #   1. 算 total 并构造 padded 列表（原排列 + 循环补齐）
    #   2. 切片 [rank::world_size] 返回
    return None


def sampler_coverage_ok(n, world_size, seed=0):
    """
    验证性质：
      1. 各 rank 长度相等
      2. 并集覆盖 0..n-1 全部索引（补齐的重复样本不影响）
      3. 每个 rank 内部无重复（补齐只会落在"末尾的某些 rank"，
         不会与同 rank 的原有样本撞车 —— n % world_size == 0 时根本没有补齐）

    Returns:
        bool
    """
    # TODO:
    #   1. 取每个 rank 的 sampler_indices
    #   2. 检查上面三条性质
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 4：张量并行的分块数学（10 分）
# ═════════════════════════════════════════════════════════════════════

def tp_mlp_max_error(X, W1, W2, n_shards=2):
    """
    模拟 Megatron 式 MLP 的列/行并行（对照脚本 05，纯 CPU 张量运算）：
        稠密：Y = gelu(X @ W1.T) @ W2.T            # W1:(H4,IN), W2:(OUT,H4)
        分片：H_r = gelu(X @ W1_r.T)               # W1 按行切成 n_shards 份
              Y  = Σ_r H_r @ W2_r.T                # W2 按列切（对应 H 的分块）

    Args:
        X:  (B, IN) 输入
        W1: (H4, IN) 第一层权重（torch Linear 的存法）
        W2: (OUT, H4) 第二层权重
        n_shards: 切几份（H4 必须能整除）

    Returns:
        float：分片前向 vs 稠密前向的 max abs error（应 < 1e-5）
    """
    # TODO:
    #   1. 稠密参照
    #   2. for r in range(n_shards): 切 W1 的行块、W2 的列块，累加 gelu(X@W1_r.T)@W2_r.T
    #   3. 返回两者最大绝对误差
    return None


# ═════════════════════════════════════════════════════════════════════
#  题 5：🌟 流水线气泡（10 分）
# ═════════════════════════════════════════════════════════════════════

def pipeline_bubble_fraction(p, m):
    """
    GPipe/1F1B 的气泡占比 = (p-1)/(m+p-1)。

    Args:
        p: stage 数；m: micro-batch 数
    Returns:
        float
    """
    # TODO: 一行
    return None


def in_flight_activations(schedule, m, p):
    """
    两种流水线调度下，"同时在显存里的激活份量"：
        'gpipe' → m（先做完全部 forward，每个 micro-batch 的激活都要留到 backward）
        '1f1b'  → p（交错执行，驻留量 ≈ stage 数）

    Args:
        schedule: 'gpipe' | '1f1b'
        m: micro-batch 数；p: stage 数
    Returns:
        int
    """
    # TODO: 两行（一个 if）
    return None
