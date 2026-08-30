#!/usr/bin/env python3
"""Part 12 作业测试。独立运行：python test_finetune_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_exercises import *  # noqa: F401,F403


def test_ex1_lora_params():
    assert lora_params is not None and lora_params([(288, 96), (96, 288)], 4) is not None
    v = lora_params([(288, 96), (96, 288)], 4)
    want = 4 * (288 + 96) + 4 * (96 + 288)
    assert v == want, f"Σ r*(out+in) 应为 {want}，得到 {v}"
    r = lora_ratio([(288, 96), (96, 288)], 4, 200_000)
    assert abs(r - want / 200_000) < 1e-12


def test_ex2_merge():
    import torch
    torch.manual_seed(0)
    W = torch.randn(8, 6)
    A = torch.randn(3, 6) / math.sqrt(3)
    B = torch.randn(8, 3) * 0.1
    x = torch.randn(6)
    assert merged_weight is not None, "merged_weight 未实现"
    W2 = merged_weight(W, A, B, alpha=8.0, r=3)
    assert W2 is not None and W2.shape == (8, 6)
    assert torch.allclose(W2, W + (8.0 / 3) * B @ A, atol=1e-6), "合并公式错"
    assert merge_changes_output is not None and merge_changes_output(W, A, B, 8.0, 3, x)


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
    assert v is not None
    want = (7e9 * 4 / 8 + 20_000_000 * 12) / 1e9
    assert abs(v - want) / want < 1e-9, f"应约 {want:.2f} GB，得到 {v}"


_TESTS = [("题1 LoRA参数账", test_ex1_lora_params), ("题2 合并数学", test_ex2_merge),
          ("题3 零初始化", test_ex3_zero_init), ("题4 QLoRA显存", test_ex4_vram)]


def main():
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 finetune_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
