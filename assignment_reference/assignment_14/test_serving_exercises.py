#!/usr/bin/env python3
"""Part 14 作业测试。独立运行：python test_serving_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serving_exercises import *  # noqa: F401,F403


def test_ex1_metrics():
    assert e2e_latency_ms is not None, "e2e_latency_ms 未实现"
    # TTFT=200ms, TPOT=50ms, 输出 5 个 token → 200+50×4=400
    assert math.isclose(e2e_latency_ms(200, 50, 5), 400.0), "E2E = TTFT+TPOT×(n-1)"
    assert math.isclose(e2e_latency_ms(200, 50, 1), 200.0), "单 token 输出 = TTFT"
    assert throughput_tokens_per_s is not None
    assert math.isclose(throughput_tokens_per_s(8, 32, 2.0), 128.0), "8×32/2s=128"


def test_ex2_kv():
    assert kv_cache_gb is not None, "kv_cache_gb 未实现"
    # LLaMA-7B fp16 seq2048 bs1：2×32×32×128×2048×2 = 1.07GB
    v = kv_cache_gb(32, 32, 128, 2048, 1)
    assert v is not None and abs(v - 2 * 32 * 32 * 128 * 2048 * 2 / 1e9) < 1e-9
    # GQA kv=8 → 恰好 1/4
    assert abs(kv_cache_gb(32, 8, 128, 2048, 1) - v / 4) < 1e-9, "GQA 应线性省 kv_heads"
    assert max_batch_for_vram is not None
    assert max_batch_for_vram(0.27, 24, 4, 2) == 66, "可用 18GB / 0.27GB 每序列 = 66"
    assert max_batch_for_vram(1.0, 24, 4, 2) == 18, "可用 18GB / 1.0GB = 18"


def test_ex3_waste():
    assert static_batch_waste is not None, "static_batch_waste 未实现"
    jobs = [10, 10, 10, 10]
    assert abs(static_batch_waste(jobs)) < 1e-12, "等长无浪费"
    jobs2 = [100, 10, 10, 10]
    w = static_batch_waste(jobs2)
    assert abs(w - (1 - 130 / 400)) < 1e-12, "max=400 实际130 → 浪费 67.5%"
    assert static_batch_waste(jobs2, pad_to_max=False) == 0.0, "连续批处理理想无浪费"


_TESTS = [("题1 serving 指标", test_ex1_metrics), ("题2 KV 容量账", test_ex2_kv),
          ("题3 批处理浪费", test_ex3_waste)]


def main():
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 serving_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
