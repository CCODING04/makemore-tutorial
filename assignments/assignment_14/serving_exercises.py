#!/usr/bin/env python3
"""
Part 14 作业：推理部署。纯纸笔/纯 Python 可完成（serving 指标与容量账）。
实现后运行 test_serving_exercises.py 验证。
"""

import math


# ── 题 1：serving 指标计算（30 分）──────────────────────────
def e2e_latency_ms(ttft_ms, tpot_ms, n_out):
    """端到端延迟 = TTFT + TPOT × (输出 token 数 − 1)。"""
    # TODO: 一行（n_out=1 时应恰为 ttft_ms）
    return None


def throughput_tokens_per_s(n_requests, n_out_tokens, wall_seconds):
    """吞吐 = 全部请求的 token 总数 / 墙钟时间。"""
    # TODO: 一行
    return None


# ── 题 2：KV 容量账（35 分）──────────────────────────────────
def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    """KV 显存 = 2(K+V) × layers × kv_heads × head_dim × seq × batch × bytes（GB）。"""
    # TODO: Part 8 06 章同款公式（除以 1e9）
    return None


def max_batch_for_vram(kv_gb_per_seq, vram_gb=24.0, model_gb=4.0, headroom_gb=2.0):
    """给定单序列 KV 与显存预算，求最大并发 batch：
    可用于 KV 的显存 = vram - 模型 - 预留；batch = floor(可用 / 每序列)。
    Returns:
        int（至少 1）
    """
    # TODO: 三行
    return None


# ── 题 3：连续批处理 vs 静态批处理（35 分）──────────────────
def static_batch_waste(jobs, pad_to_max=True):
    """静态批处理把整个 batch pad 到最长 job：浪费 = pad 掉的 token 份额。
    Args:
        jobs: list[int]（各请求输出 token 数）
        pad_to_max: True 按 max；False 按"逐请求"即无浪费（连续批处理的理想）
    Returns:
        float：浪费率（0~1）= 1 - 实际token / 分配token
    """
    # TODO:
    #   pad_to_max=True: 分配 = len(jobs) × max(jobs)
    #   False: 分配 = sum(jobs)
    return None
