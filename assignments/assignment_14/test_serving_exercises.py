#!/usr/bin/env python3
"""Part 14 作业测试。

两种运行方式：
  1. 独立运行：  python test_serving_exercises.py
  2. pytest：    pytest test_serving_exercises.py

行为约定：题 1-4 未实现（返回 None）会失败（❌，提示先实现）；
题 5 🌟 为 stretch 选做，未实现（返回 None）时优雅跳过（⏭️）。
"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serving_exercises import *  # noqa: F401,F403

try:
    import pytest
except ImportError:
    pytest = None


class _Skipped(Exception):
    """独立运行时的跳过信号（pytest 下用 pytest.skip）。"""


def _skip(reason):
    if pytest is not None:
        pytest.skip(reason)          # pytest 运行 → 正规 SKIP（显示 s）
    raise _Skipped(reason)           # 独立运行 → main() 捕获后打印 ⏭️


# ══════════════════════════════════════════════════════════════════════
#  题 1：serving 指标
# ══════════════════════════════════════════════════════════════════════

def test_ex1_metrics():
    r = e2e_latency_ms(200, 50, 5)
    assert r is not None, "e2e_latency_ms 未实现"
    # TTFT=200ms, TPOT=50ms, 输出 5 个 token → 200+50×4=400
    assert math.isclose(r, 400.0), "E2E = TTFT+TPOT×(n-1)"
    assert math.isclose(e2e_latency_ms(200, 50, 1), 200.0), "单 token 输出 = TTFT"
    # 输出翻倍 → E2E 增量恰为 TPOT×(n_out 差)
    d = e2e_latency_ms(200, 50, 9) - e2e_latency_ms(200, 50, 5)
    assert math.isclose(d, 50.0 * 4), "每多 1 个 token 多 1 个 TPOT"
    th = throughput_tokens_per_s(8, 32, 2.0)
    assert th is not None, "throughput_tokens_per_s 未实现"
    assert math.isclose(th, 128.0), "8×32/2s=128"


# ══════════════════════════════════════════════════════════════════════
#  题 2：KV 容量账
# ══════════════════════════════════════════════════════════════════════

def test_ex2_kv():
    v = kv_cache_gb(32, 32, 128, 2048, 1)
    assert v is not None, "kv_cache_gb 未实现"
    # LLaMA-7B fp16 seq2048 bs1：2×32×32×128×2048×2 = 1.07GB
    assert abs(v - 2 * 32 * 32 * 128 * 2048 * 2 / 1e9) < 1e-9, \
        "KV = 2(K+V)×layers×kv_heads×head_dim×seq×batch×bytes（别忘了 K/V 的 ×2）"
    # GQA kv=8 → 恰好 1/4
    assert abs(kv_cache_gb(32, 8, 128, 2048, 1) - v / 4) < 1e-9, "GQA 应线性省 kv_heads"
    # batch 与 seq_len 都应线性缩放（不变量，不查具体实现）
    assert abs(kv_cache_gb(32, 32, 128, 2048, 4) - v * 4) < 1e-9, "KV 随 batch 线性"
    assert abs(kv_cache_gb(32, 32, 128, 4096, 1) - v * 2) < 1e-9, "KV 随 seq 线性"
    mb = max_batch_for_vram(0.27, 24, 4, 2)
    assert mb is not None, "max_batch_for_vram 未实现"
    assert mb == 66, "可用 18GB / 0.27GB 每序列 = 66"
    assert max_batch_for_vram(1.0, 24, 4, 2) == 18, "可用 18GB / 1.0GB = 18"
    assert max_batch_for_vram(100.0, 24, 4, 2) >= 1, "预算不足也至少返回 1"


# ══════════════════════════════════════════════════════════════════════
#  题 3：静态批处理浪费
# ══════════════════════════════════════════════════════════════════════

def test_ex3_waste():
    w0 = static_batch_waste([10, 10, 10, 10])
    assert w0 is not None, "static_batch_waste 未实现"
    assert abs(w0) < 1e-12, "等长无浪费"
    jobs2 = [100, 10, 10, 10]
    w = static_batch_waste(jobs2)
    assert abs(w - (1 - 130 / 400)) < 1e-12, "max=400 实际130 → 浪费 67.5%"
    assert static_batch_waste(jobs2, pad_to_max=False) == 0.0, "连续批处理理想无浪费"
    # 浪费率 ∈ [0,1)：只要有一个非零 job 就不可能全是浪费
    assert 0.0 <= w < 1.0


# ══════════════════════════════════════════════════════════════════════
#  题 4：投机解码——接受率与有效加速
# ══════════════════════════════════════════════════════════════════════

def test_ex4_spec_decode():
    r0 = spec_tokens_per_cycle(0.0, 4)
    assert r0 is not None, "spec_tokens_per_cycle 未实现"
    tpc = spec_tokens_per_cycle
    # 边界：全拒（α=0）每周期仍白赚 1 个 token；全收（α=1）= γ+1（公式的极限值）
    assert math.isclose(r0, 1.0), "α=0 → 每周期 1 token"
    assert math.isclose(tpc(1.0, 4), 5.0), "α=1 → γ+1（0/0 的极限）"
    # Part 8 06 章实测口径：α=0.60, γ=4 → 理论 2.31
    v = tpc(0.6, 4)
    assert abs(v - (1 - 0.6 ** 5) / 0.4) < 1e-9, "E=(1-α^(γ+1))/(1-α)"
    assert abs(v - 2.3056) < 1e-3
    # 数学不变量 1：对 α 单调递增（接受率越高每周期产出越多）
    assert tpc(0.3, 4) < tpc(0.6, 4) < tpc(0.9, 4), "应对 α 单调"
    # 数学不变量 2：等比级数恒等式 (1-α^{γ+1})/(1-α) == Σ_{k=0}^{γ} α^k
    assert abs(tpc(0.7, 6) - sum(0.7 ** k for k in range(7))) < 1e-9, "等比级数恒等式"
    # 数学不变量 3：γ 越长产出越多，且上界是 γ+1
    assert tpc(0.6, 2) < tpc(0.6, 4) < tpc(0.6, 8) <= 9
    for a in (0.2, 0.6, 0.95):
        assert tpc(a, 4) <= 5.0 + 1e-12, f"上界 γ+1=5（α={a}）"

    s0 = spec_decode_speedup(0.6, 4, 0.0)
    assert s0 is not None, "spec_decode_speedup 未实现"
    sp = spec_decode_speedup
    # 草稿零开销 → 加速比上界就是 tokens/cycle
    assert math.isclose(s0, tpc(0.6, 4)), "overhead=0 → 上界=tokens/cycle"
    # draft 越贵加速越小（单调），且 α=0 + 贵草稿 = 负收益（<1）
    assert sp(0.6, 4, 0.5) < sp(0.6, 4, 0.25) < sp(0.6, 4, 0.0), "对 overhead 单调递减"
    assert math.isclose(sp(0.0, 4, 0.5), 1.0 / 1.5), "α=0, 开销 0.5 → 2/3"
    assert sp(0.0, 4, 0.5) < 1.0, "低接受率 + 贵草稿 → 越推越慢"


# ══════════════════════════════════════════════════════════════════════
#  题 5 🌟 stretch：连续批处理调度模拟器（未实现 → 优雅 SKIP）
# ══════════════════════════════════════════════════════════════════════

def test_ex5_stretch_sim():
    probe = simulate_batching([0, 0], [10, 2], max_batch=2, mode="static")
    if probe is None:
        _skip("题5 未实现（返回 None）—— stretch 选做")

    # 情形 A：同批内长短不齐。静态批：整批陪跑 max → 40% 浪费；连续批：完成即腾位 → 0
    st = simulate_batching([0, 0], [10, 2], max_batch=2, mode="static")
    ct = simulate_batching([0, 0], [10, 2], max_batch=2, mode="continuous")
    assert st["makespan"] == 10 and st["slot_steps"] == 20, "静态批 slot=2×10"
    assert abs(st["waste_rate"] - 0.4) < 1e-9, "静态批浪费 1-12/20=40%"
    assert ct["makespan"] == 10 and ct["slot_steps"] == 12, "连续批 slot=10+2"
    assert abs(ct["waste_rate"]) < 1e-12, "理想连续批零浪费"

    # 情形 B：批跑期间新请求到达。静态批必须等整批结束 → makespan 15；
    #         连续批立刻补位 → makespan 10（Orca 论文的动机，亲手算出来）
    st2 = simulate_batching([0, 0, 2], [10, 2, 5], max_batch=2, mode="static")
    ct2 = simulate_batching([0, 0, 2], [10, 2, 5], max_batch=2, mode="continuous")
    assert st2["makespan"] == 15, "静态批：等整批结束再跑新请求（10+5）"
    assert ct2["makespan"] == 10, "连续批：第 2 步腾出的槽位立刻给新请求"
    assert ct2["makespan"] <= st2["makespan"], "连续批 makespan 不劣于静态批"

    # 通用不变量：两种模式的实际 token 数都等于 sum(gen_lens)，浪费率 ∈ [0,1)
    for m in ("static", "continuous"):
        r = simulate_batching([0, 0, 2], [10, 2, 5], max_batch=2, mode=m)
        assert r["tokens"] == 17, f"{m}: tokens 应恰为 sum(gen_lens)"
        assert 0.0 <= r["waste_rate"] < 1.0

    # 边界：并发=1 且全部单 token → 串行，两种模式完全一致
    for m in ("static", "continuous"):
        r = simulate_batching([0, 0, 0], [1, 1, 1], max_batch=1, mode=m)
        assert r["makespan"] == 3 and abs(r["waste_rate"]) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════════════════════════

_TESTS = [
    ("题1 serving 指标", test_ex1_metrics),
    ("题2 KV 容量账", test_ex2_kv),
    ("题3 批处理浪费", test_ex3_waste),
    ("题4 投机解码数学", test_ex4_spec_decode),
    ("题5 🌟 连续批调度模拟器（stretch）", test_ex5_stretch_sim),
]

_SKIP_EXCS = (_Skipped, pytest.skip.Exception) if pytest is not None else (_Skipped,)


def main():
    passed = failed = skipped = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except _SKIP_EXCS as e:
            print(f"  ⏭️ {name} — SKIP: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}")
            failed += 1
    total = passed + failed + skipped
    print(f"\n  通过: {passed}/{total}, 失败: {failed}, 跳过: {skipped}"
          + ("  🎉" if failed == 0 and skipped == 0 else
             ("  ✨ 核心 4 题全过（stretch 未实现已跳过）" if failed == 0 else "  💡 先实现 serving_exercises.py")))
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
