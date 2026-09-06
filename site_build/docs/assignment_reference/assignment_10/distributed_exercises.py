"""Part 10 作业参考答案：分布式训练"""
import math

def ddp_gradient(grads_per_rank):
    """all_reduce(SUM)/world = 各 rank 梯度的【平均】（不是求和）"""
    return sum(grads_per_rank) / len(grads_per_rank)

def effective_batch(local_batch, accum_steps, world_size):
    return local_batch * accum_steps * world_size

def model_state_bytes(param_count, world_size, stage):
    """混合精度 AdamW 模型状态账本（Ψ=param_count，N=卡数）
    DDP=16Ψ；ZeRO-1=4Ψ+12Ψ/N；ZeRO-2=2Ψ+14Ψ/N；ZeRO-3=16Ψ/N（N=1 时全等于 16Ψ）"""
    if stage == 'ddp':   return 16 * param_count
    if stage == 'zero1': return 4 * param_count + 12 * param_count / world_size
    if stage == 'zero2': return 2 * param_count + 14 * param_count / world_size
    if stage == 'zero3': return 16 * param_count / world_size
    raise ValueError(f"未知 stage: {stage}")

def can_train_7b_on_24gb(world_size, activation_bytes=6_000_000_000):
    return model_state_bytes(7e9, world_size, 'zero3') + activation_bytes <= 24e9

def sampler_indices(n, world_size, rank, seed=0, with_torch=None):
    """DistributedSampler 语义：补齐到整除 → 按 seed 打乱 → padded[rank::world]
    优先 torch.randperm 路径（torch.Generator(seed) 保确定性，set_epoch 改的就是它）；
    无 torch 或 with_torch=False 时退回 random.Random(seed).shuffle 纯 Python 模拟。"""
    total = math.ceil(n / world_size) * world_size
    order = None
    if with_torch is not False:                      # None = 自动：有 torch 就用
        try:
            import torch
            order = torch.randperm(
                n, generator=torch.Generator().manual_seed(seed)).tolist()
        except ImportError:
            order = None
    if order is None:
        import random
        order = list(range(n))
        random.Random(seed).shuffle(order)
    padded = (order * (total // n + 1))[:total]
    return padded[rank::world_size]

def sampler_coverage_ok(n, world_size, seed=0):
    idx = [sampler_indices(n, world_size, r, seed) for r in range(world_size)]
    if len({len(x) for x in idx}) != 1:
        return False
    seen = set()
    for x in idx:
        if len(x) != len(set(x)):
            return False          # rank 内部无重复
        seen |= set(x)
    return seen == set(range(n))  # 并集全覆盖

def tp_mlp_max_error(X, W1, W2, n_shards=2):
    """Megatron 式列/行并行 vs 稠密的 max 误差（应 <1e-5）"""
    import torch, torch.nn.functional as F
    y = F.gelu(X @ W1.T) @ W2.T
    H4 = W1.shape[0]; chunk = H4 // n_shards
    y_tp = torch.zeros_like(y)
    for r in range(n_shards):
        h = F.gelu(X @ W1[r * chunk:(r + 1) * chunk, :].T)
        y_tp = y_tp + h @ W2[:, r * chunk:(r + 1) * chunk].T
    return (y - y_tp).abs().max().item()

def pipeline_bubble_fraction(p, m):
    """GPipe/1F1B 气泡 = (p-1)/(m+p-1)"""
    return (p - 1) / (m + p - 1)

def in_flight_activations(schedule, m, p):
    return m if schedule == 'gpipe' else p
