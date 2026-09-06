#!/usr/bin/env python3
"""
Assignment 14：推理部署（vLLM）。题 1-4 纯纸笔/纯 Python 可完成（4 × 25 = 100 分），
题 5 🌟 为 stretch 选做（不实现返回 None，测试会优雅 SKIP ⏭️）。
实现后运行 test_serving_exercises.py 验证（python 或 pytest 均可）。

对应教程：Part 14（01 朴素基线 / 02 vLLM 实战）+ Part 8 06 章（量化/KV/分页/投机解码）。
"""

import math


# ══════════════════════════════════════════════════════════════════════
#  题 1：serving 指标（25 分）
# ══════════════════════════════════════════════════════════════════════

def e2e_latency_ms(ttft_ms, tpot_ms, n_out):
    """端到端延迟 = TTFT + TPOT × (输出 token 数 − 1)。

    Args:
        ttft_ms: 首 token 延迟（毫秒）
        tpot_ms: 每 token 生成间隔（毫秒）
        n_out:   输出 token 总数（含首 token）

    Returns:
        float: 端到端延迟（毫秒）。n_out=1 时应恰为 ttft_ms。

    Steps:
        1. 首 token 只等 TTFT；其后每个 token 等 TPOT
        2. 后续 token 数 = n_out − 1
        3. 返回 ttft_ms + tpot_ms × (n_out − 1)

    Acceptance Criteria:
        - e2e_latency_ms(200, 50, 5) == 400.0
        - e2e_latency_ms(200, 50, 1) == 200.0（单 token 输出 = TTFT）
    """
    # TODO: 一行
    return None


def throughput_tokens_per_s(n_requests, n_out_tokens, wall_seconds):
    """吞吐 = 全部请求的 token 总数 / 墙钟时间。

    Args:
        n_requests:  请求总数
        n_out_tokens: 每个请求的输出 token 数（全部相同）
        wall_seconds: 墙钟时间（秒）

    Returns:
        float: 吞吐（tok/s）

    Steps:
        1. token 总数 = n_requests × n_out_tokens
        2. 除以 wall_seconds

    Acceptance Criteria:
        - throughput_tokens_per_s(8, 32, 2.0) == 128.0
    """
    # TODO: 一行
    return None


# ══════════════════════════════════════════════════════════════════════
#  题 2：KV 容量账（25 分）
# ══════════════════════════════════════════════════════════════════════

def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    """KV 显存 = 2(K+V) × layers × kv_heads × head_dim × seq × batch × bytes（GB）。

    对照 Part 8 06 章 §2 的公式表（fp16 → bytes_per_elem=2）。

    Args:
        n_layers:     层数
        n_kv_heads:   KV 头数（GQA 下 < 注意力头数）
        head_dim:     每头维度
        seq_len:      序列长度
        batch:        并发序列数
        bytes_per_elem: 每元素字节数（fp16=2，int8=1）

    Returns:
        float: KV cache 显存（GB，按 1e9 换算）

    Steps:
        1. 单序列单层单头：head_dim × seq_len 个元素
        2. 乘 kv_heads、n_layers、batch
        3. K 和 V 各一份（×2），再乘 bytes_per_elem
        4. 除以 1e9 得 GB

    Acceptance Criteria:
        - kv_cache_gb(32, 32, 128, 2048, 1) ≈ 1.07（LLaMA-7B fp16 seq2048）
        - GQA（kv_heads 32→8）恰好缩小 4 倍（线性缩放不变量）
    """
    # TODO: Part 8 06 章同款公式
    return None


def max_batch_for_vram(kv_gb_per_seq, vram_gb=24.0, model_gb=4.0, headroom_gb=2.0):
    """给定单序列 KV 与显存预算，求最大并发 batch。

    Args:
        kv_gb_per_seq: 每个序列的 KV 显存（GB，可用题 2 的 kv_cache_gb 算出）
        vram_gb:      总显存（GB）
        model_gb:     模型权重占用（GB）
        headroom_gb:  预留余量（激活/碎片，GB）

    Returns:
        int: 最大并发 batch（至少 1，即使预算不足也返回 1）

    Steps:
        1. 可用于 KV 的显存 = vram_gb − model_gb − headroom_gb
        2. batch = floor(可用显存 / kv_gb_per_seq)
        3. 用 max(1, ...) 保底（宁可 OOM 也不返回 0）

    Acceptance Criteria:
        - max_batch_for_vram(0.27, 24, 4, 2) == 66（可用 18GB / 0.27GB）
        - max_batch_for_vram(1.0, 24, 4, 2) == 18
    """
    # TODO: 三行
    return None


# ══════════════════════════════════════════════════════════════════════
#  题 3：静态批处理 vs 连续批处理的浪费（25 分）
# ══════════════════════════════════════════════════════════════════════

def static_batch_waste(jobs, pad_to_max=True):
    """静态批处理把整个 batch pad 到最长 job：浪费 = pad 掉的 token 份额。

    Args:
        jobs: list[int]（各请求输出 token 数）
        pad_to_max: True 按 max(jobs) pad（静态批处理）；
                    False 按"逐请求"即无浪费（连续批处理的理想）

    Returns:
        float: 浪费率（0~1）= 1 − 实际 token / 分配 token

    Steps:
        1. 实际 token = sum(jobs)
        2. pad_to_max=True: 分配 = len(jobs) × max(jobs)（每个槽位都跑最长那么久）
        3. pad_to_max=False: 分配 = sum(jobs)
        4. 返回 1 − 实际/分配

    Acceptance Criteria:
        - static_batch_waste([10,10,10,10]) == 0（等长无浪费）
        - static_batch_waste([100,10,10,10]) == 0.675（67.5%）
        - static_batch_waste([100,10,10,10], pad_to_max=False) == 0.0
    """
    # TODO: 四行
    return None


# ══════════════════════════════════════════════════════════════════════
#  题 4：投机解码——接受率与有效加速（25 分）
# ══════════════════════════════════════════════════════════════════════

def spec_tokens_per_cycle(alpha, gamma):
    """每个 verify 周期期望产出的 token 数：E = (1 − α^(γ+1)) / (1 − α)。

    对照 Part 8 06 章 §4：draft 先写 γ 个 token，target 一次前向并行验证；
    每个草稿以概率 α 被接受，全拒时 target 仍"白赚"1 个 token。
    Part 8 实测 α≈0.60、γ=4 → 理论 2.31 tokens/cycle。

    Args:
        alpha: 接受率（0~1）
        gamma: 草稿长度 γ

    Returns:
        float: 每周期期望 token 数

    Steps:
        1. α=1 时公式为 0/0，取极限值 γ+1（等比级数 1+α+…+α^γ 在 α=1 处的和）
        2. 否则返回 (1 − α^(gamma+1)) / (1 − alpha)
        3. 注意 α=0 应自然得到 1.0（全拒仍出 1 个 token）

    Acceptance Criteria:
        - spec_tokens_per_cycle(0.0, 4) == 1.0（全拒）
        - spec_tokens_per_cycle(1.0, 4) == 5.0（全收 = γ+1）
        - spec_tokens_per_cycle(0.6, 4) ≈ 2.3056（Part 8 实测口径）
        - 对 α 单调递增：tpc(0.3) < tpc(0.6) < tpc(0.9)
    """
    # TODO: 两行（先处理 alpha == 1 的极限，再套公式）
    return None


def spec_decode_speedup(alpha, gamma, draft_overhead=0.0):
    """投机解码的加速比上界 = tokens_per_cycle / (1 + draft_overhead)。

    基线：1 次 target 前向出 1 个 token。一个投机周期 = 1 次 target 验证前向
    + draft 开销（折算成 target 前向时间的倍数）。memory-bound 下验证 γ+1 个
    token ≈ 生成 1 个 token 的代价，所以这是**上界**。

    Args:
        alpha:          接受率（0~1）
        gamma:          草稿长度 γ
        draft_overhead: 每周期 draft 开销 / 1 次 target 前向（0.5 = 一半）

    Returns:
        float: 加速比上界（< 1 表示"越推越慢"——draft 太弱时的真实陷阱）

    Steps:
        1. 每周期成本 = 1 + draft_overhead（单位：target 前向）
        2. 每周期产出 = spec_tokens_per_cycle(alpha, gamma)
        3. 返回 产出 / 成本

    Acceptance Criteria:
        - spec_decode_speedup(0.6, 4, 0.0) == spec_tokens_per_cycle(0.6, 4)
        - draft_overhead 越大加速比越小（单调）
        - spec_decode_speedup(0.0, 4, 0.5) ≈ 0.667 < 1（低接受率 + 贵草稿 = 负收益）
    """
    # TODO: 两行
    return None


# ══════════════════════════════════════════════════════════════════════
#  题 5 🌟 stretch：连续批处理调度模拟器（选做，不实现返回 None）
# ══════════════════════════════════════════════════════════════════════

def simulate_batching(arrivals, gen_lens, max_batch=8, mode="continuous"):
    """🌟 Stretch：模拟静态/连续批处理调度，量化 Orca 论文的动机。

    时间离散为"步"：每步每个在跑请求生成 1 个 token（1 次 decode 前向）。

    Args:
        arrivals: list[int/float]  各请求到达时间（步，非递减）
        gen_lens: list[int]        各请求需要生成的 token 数
        max_batch: int             并发槽位上限
        mode:      "static"        逐批组队：凑满 max_batch 或等不到更多，
                                   整批跑到最慢的结束才放下一批（每槽位都
                                   占用 max(gen_lens) 步 → 题 3 的浪费）
                   "continuous"    每步可换人：请求完成立刻腾槽位，等待者
                                   立刻补位（理想连续批处理，无换入换出开销）

    Returns:
        dict: {"makespan": 全部完成的总步数,
               "slot_steps": 槽位×步 总数（分配量）,
               "tokens": 实际产出 token 数,
               "waste_rate": 1 − tokens/slot_steps}
        未实现时返回 None（测试会 SKIP ⏭️）

    Steps（continuous 分支）:
        1. 维护 waiting/running 两个集合，时钟 t 从 0 走
        2. 每步先把"已到达且有空位"的请求装入 running
        3. 全体在跑请求各出 1 token；完成的移出，记录 slot_steps
        4. GPU 全空且还有 waiting → 时钟直接跳到下一个到达时刻
    Steps（static 分支）:
        1. 按到达顺序凑批：从当前时刻已到达的请求里取最多 max_batch 个
        2. 批开始时间 = max(上一批结束, 首个成员到达)
        3. 批时长 = max(批内 gen_lens)，slot_steps += 批大小 × 批时长

    Acceptance Criteria:
        - arrivals=[0,0], gen=[10,2], max_batch=2：
          static  → makespan=10, waste=0.4；continuous → makespan=10, waste=0
        - arrivals=[0,0,2], gen=[10,2,5], max_batch=2：
          static makespan=15；continuous makespan=10（Orca 的胜利）
        - 两种模式 tokens 都 == sum(gen_lens)
        - max_batch=1 且 gen 全 1：两种模式 makespan 都 == len(gen_lens)
    """
    # TODO: 先实现 continuous（更直观），再实现 static，最后拼 dict
    return None
