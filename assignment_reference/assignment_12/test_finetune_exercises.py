#!/usr/bin/env python3
"""Part 12 作业测试。独立运行：python test_finetune_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_exercises import *  # noqa: F401,F403

# ─── 跳过机制（🌟 stretch 未实现时优雅 SKIP；兼容独立运行与 pytest）───
try:
    import pytest
except ImportError:
    pytest = None


class _Skipped(Exception):
    pass


def _skip(reason):
    """pytest 运行时走原生 skip；独立运行抛自定义异常由 main() 统计为跳过。"""
    if pytest is not None and "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)          # pytest.Skipped 继承 BaseException，别在独立模式触发
    raise _Skipped(reason)


def test_ex1_lora_params():
    assert lora_params is not None, "lora_params 未实现"
    v = lora_params([(288, 96), (96, 288)], 4)
    assert v is not None, "lora_params 未实现（返回 None）"
    want = 4 * (288 + 96) + 4 * (96 + 288)
    assert v == want, f"Σ r*(out+in) 应为 {want}，得到 {v}"
    r = lora_ratio([(288, 96), (96, 288)], 4, 200_000)
    assert r is not None, "lora_ratio 未实现（返回 None）"
    assert abs(r - want / 200_000) < 1e-12, f"比例应为 {want / 200_000}，得到 {r}"


def test_ex2_merge():
    import torch
    torch.manual_seed(0)
    W = torch.randn(8, 6)
    A = torch.randn(3, 6) / math.sqrt(3)
    B = torch.randn(8, 3) * 0.1
    x = torch.randn(6)
    assert merged_weight is not None, "merged_weight 未实现"
    W2 = merged_weight(W, A, B, alpha=8.0, r=3)
    assert W2 is not None and W2.shape == (8, 6), "合并结果应为 (8, 6)"
    assert torch.allclose(W2, W + (8.0 / 3) * B @ A, atol=1e-6), "合并公式错"
    assert merge_changes_output is not None, "merge_changes_output 未实现"
    assert merge_changes_output(W, A, B, 8.0, 3, x), "合并前后前向应一致"


def test_ex3_zero_init():
    import torch
    assert initial_delta_norm is not None, "initial_delta_norm 未实现"
    A = torch.randn(4, 8)
    B = torch.zeros(8, 4)
    v = initial_delta_norm(A, B)
    assert v is not None and abs(v) < 1e-9, "B=0 时 ΔW 范数应为 0（起点无损）"


def test_ex4_vram():
    assert qlora_vram_gb is not None, "qlora_vram_gb 未实现"
    v = qlora_vram_gb(7.0, 4, 20_000_000)
    assert v is not None, "qlora_vram_gb 未实现（返回 None）"
    want = (7e9 * 4 / 8 + 20_000_000 * 12) / 1e9
    assert abs(v - want) / want < 1e-9, f"应约 {want:.2f} GB，得到 {v}"


def test_ex5_rank_sweep_stretch():
    """🌟 题 5 stretch：未实现返回 None → SKIP ⏭️（不扣分）。"""
    if lora_rank_sweep is None:
        _skip("题 5 stretch 未实现")
    try:
        import torch  # noqa: F401
    except ImportError:
        _skip("题 5 需要 torch，当前环境不可用")
    res = lora_rank_sweep()          # 默认参数：ranks=(1,2,4,8), out_f=32, in_f=16
    if res is None:
        _skip("题 5 stretch 未实现（返回 None）")
    out_f, in_f, ranks = 32, 16, (1, 2, 4, 8)
    assert set(res.keys()) == set(ranks), f"返回的 keys 应恰为 {ranks}"
    for r in ranks:
        v = res[r]
        assert v["params"] == r * (out_f + in_f), \
            f"r={r}: params 应为 {r * (out_f + in_f)}（r·(out+in)），得到 {v['params']}"
        assert v["loss_end"] < v["loss_start"], f"r={r}: loss 应下降"
    starts = [res[r]["loss_start"] for r in ranks]
    assert max(starts) - min(starts) < 1e-6, "B=0 ⇒ 各 rank 的起点 loss 应相同"
    assert res[8]["loss_end"] <= res[1]["loss_end"], "rank 越大表达力越强：r=8 不应差于 r=1"
    assert res[4]["loss_end"] < 0.1 * res[4]["loss_start"], \
        "r=4 ≥ 目标秩(4)：应能把 loss 打到起点的 10% 以下"


_TESTS = [("题1 LoRA参数账", test_ex1_lora_params), ("题2 合并数学", test_ex2_merge),
          ("题3 零初始化", test_ex3_zero_init), ("题4 QLoRA显存", test_ex4_vram),
          ("题5 🌟 多rank对比(stretch)", test_ex5_rank_sweep_stretch)]


def main():
    p = f = s = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except _Skipped as e:
            print(f"  ⏭️  {name} — SKIP: {e}"); s += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    msg = f"\n  通过: {p}/{p + f + s}"
    if s:
        msg += f"（另 SKIP {s} 项 ⏭️）"
    msg += "  🎉" if f == 0 else "  💡 先实现 finetune_exercises.py"
    print(msg)
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
