import math
def e2e_latency_ms(ttft_ms, tpot_ms, n_out): return ttft_ms + tpot_ms*(n_out-1)
def throughput_tokens_per_s(n_requests, n_out_tokens, wall_seconds):
    return n_requests*n_out_tokens/wall_seconds
def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    return 2*n_layers*n_kv_heads*head_dim*seq_len*batch*bytes_per_elem/1e9
def max_batch_for_vram(kv_gb_per_seq, vram_gb=24.0, model_gb=4.0, headroom_gb=2.0):
    return max(1, math.floor((vram_gb - model_gb - headroom_gb)/kv_gb_per_seq))
def static_batch_waste(jobs, pad_to_max=True):
    actual = sum(jobs)
    alloc = len(jobs)*max(jobs) if pad_to_max else actual
    return 1 - actual/alloc
