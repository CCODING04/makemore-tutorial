"""Assignment 14 参考实现：题 1-4 核心 + 题 5 stretch 全部实现。

签名与 assignments/assignment_14/serving_exercises.py 完全一致；
跑 assignment_reference/assignment_14/test_serving_exercises.py 应全过。
"""

import math


# ── 题 1：serving 指标 ────────────────────────────────────────────────
def e2e_latency_ms(ttft_ms, tpot_ms, n_out):
    return ttft_ms + tpot_ms * (n_out - 1)


def throughput_tokens_per_s(n_requests, n_out_tokens, wall_seconds):
    return n_requests * n_out_tokens / wall_seconds


# ── 题 2：KV 容量账 ──────────────────────────────────────────────────
def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem / 1e9


def max_batch_for_vram(kv_gb_per_seq, vram_gb=24.0, model_gb=4.0, headroom_gb=2.0):
    avail = vram_gb - model_gb - headroom_gb
    return max(1, math.floor(avail / kv_gb_per_seq))


# ── 题 3：静态批处理浪费 ─────────────────────────────────────────────
def static_batch_waste(jobs, pad_to_max=True):
    actual = sum(jobs)
    alloc = len(jobs) * max(jobs) if pad_to_max else actual
    return 1 - actual / alloc


# ── 题 4：投机解码——接受率与有效加速 ─────────────────────────────────
def spec_tokens_per_cycle(alpha, gamma):
    # α=1 时公式为 0/0，取等比级数极限 1+α+…+α^γ = γ+1
    if alpha >= 1:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def spec_decode_speedup(alpha, gamma, draft_overhead=0.0):
    return spec_tokens_per_cycle(alpha, gamma) / (1 + draft_overhead)


# ── 题 5 🌟 stretch：连续批处理调度模拟器 ────────────────────────────
def simulate_batching(arrivals, gen_lens, max_batch=8, mode="continuous"):
    n = len(arrivals)
    if len(gen_lens) != n:
        raise ValueError("arrivals 与 gen_lens 长度不一致")
    if mode not in ("static", "continuous"):
        raise ValueError(f"mode 只支持 static/continuous，收到 {mode!r}")
    if n == 0:
        return {"makespan": 0, "slot_steps": 0, "tokens": 0, "waste_rate": 0.0}

    tokens = sum(gen_lens)

    if mode == "static":
        # 按到达顺序凑批：批开始时刻已到达的最多 max_batch 个请求组一队，
        # 整批跑到最慢的结束（每个槽位都占 max(gen_lens) 步）。
        makespan = 0
        slot_steps = 0
        i = 0
        while i < n:
            start = max(makespan, arrivals[i])
            j = i
            while j < n and j - i < max_batch and arrivals[j] <= start:
                j += 1
            dur = max(gen_lens[i:j])
            slot_steps += (j - i) * dur
            makespan = start + dur
            i = j
    else:
        # 连续批处理：每步先把已到达且有空位的请求装入；全体在跑请求各出
        # 1 token；完成即腾位。GPU 全空则空转到下一个到达时刻。
        remaining = list(gen_lens)
        waiting = list(range(n))
        running = []
        t = 0
        slot_steps = 0
        while waiting or running:
            while waiting and len(running) < max_batch and arrivals[waiting[0]] <= t:
                running.append(waiting.pop(0))
            if not running:                 # GPU 空转，跳到下一个到达
                t = arrivals[waiting[0]]
                continue
            slot_steps += len(running)      # 一步 decode：每个占用槽位各出 1 token
            t += 1
            nxt = []
            for r in running:
                remaining[r] -= 1
                if remaining[r] > 0:
                    nxt.append(r)
            running = nxt
        makespan = t

    waste = 1 - tokens / slot_steps if slot_steps else 0.0
    return {"makespan": makespan, "slot_steps": slot_steps,
            "tokens": tokens, "waste_rate": waste}
