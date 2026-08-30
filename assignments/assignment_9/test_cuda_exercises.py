#!/usr/bin/env python3
"""
Part 9 作业测试：CUDA 内核编程

两种运行方式：
  1. 独立运行：  python test_cuda_exercises.py
  2. pytest：    pytest test_cuda_exercises.py

行为约定：题 1-4 未实现（返回 None / 抛错）会失败——实现后即通过；
题 5 需要 GPU + triton，未实现或环境不支持时优雅跳过。
"""

import os
import sys
import math

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from cuda_exercises import *  # noqa: F401,F403

try:
    import pytest
except ImportError:
    pytest = None

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import triton  # noqa: F403
    _HAS_TRITON = _HAS_TORCH and torch.cuda.is_available()
except Exception:
    _HAS_TRITON = False


class _Skipped(Exception):
    pass


def _skip(reason):
    raise _Skipped(reason)


def _require_torch():
    if not _HAS_TORCH:
        _skip("未安装 torch（题 2 需要 torch 做对照）")


# ═════════════════════════════════════════════════════════════════════
#  题 1：全局线程索引
# ═════════════════════════════════════════════════════════════════════

def test_exercise_1_global_index():
    # --- 1D：手工可验的几组 ---
    assert global_index_1d is not None, "global_index_1d 未实现"
    assert global_index_1d(0, 0, 256) == 0, "block0/thread0 应为 0"
    assert global_index_1d(0, 255, 256) == 255, "block0/thread255 应为 255"
    assert global_index_1d(1, 0, 256) == 256, "block1/thread0 应为 256"
    assert global_index_1d(3, 17, 256) == 3 * 256 + 17, "公式：block_idx*block_size + thread_idx"
    assert global_index_1d(39, 0, 256) == 9984, "block_idx*block_size + thread_idx"

    # --- launch_config：向上取整 ---
    assert launch_config is not None, "launch_config 未实现"
    result = launch_config(1000, 256)
    assert result is not None, "launch_config 未实现（返回 None）"
    num_blocks, total = result
    assert num_blocks == 4, f"1000/256 应向上取整为 4，得到 {num_blocks}"
    assert total == 1024 and total > 1000, "total_threads 应为 4*256=1024（> n）"
    num_blocks, total = launch_config(1024, 256)
    assert num_blocks == 4 and total == 1024, "整除时恰好 4 块"
    num_blocks, total = launch_config(1, 256)
    assert num_blocks == 1 and total == 256, "1 个元素也要 1 个块"

    # --- 2D：与脚本 02 的 square_2d 索引一致 ---
    assert global_index_2d is not None, "global_index_2d 未实现"
    result = global_index_2d((0, 0), (0, 0), (32, 32), 512)
    assert result is not None, "global_index_2d 未实现（返回 None）"
    # block(0,0) thread(0,0)：左上角
    assert result == (0, 0, 0)
    # block(0,0) thread(1,2)（tx=1, ty=2）：row=2, col=1
    row, col, idx = global_index_2d((0, 0), (1, 2), (32, 32), 512)
    assert (row, col) == (2, 1) and idx == 2 * 512 + 1
    # block(2,3) thread(5,7)，block 32x32，行宽 512：
    #   col = 2*32+5 = 69, row = 3*32+7 = 103
    row, col, idx = global_index_2d((2, 3), (5, 7), (32, 32), 512)
    assert (row, col, idx) == (103, 69, 103 * 512 + 69)


# ═════════════════════════════════════════════════════════════════════
#  题 2：行/列主序 + CPU matmul
# ═════════════════════════════════════════════════════════════════════

def test_exercise_2_indexing_and_matmul():
    _require_torch()
    assert row_major_index is not None and col_major_index is not None, "索引函数未实现"
    # 同一元素，两种存储下标不同：
    assert row_major_index(1, 2, 4) == 1 * 4 + 2
    assert col_major_index(1, 2, 4) == 2 * 4 + 1

    assert matmul_cpu is not None, "matmul_cpu 未实现"
    # 已知答案的小例子（脚本 03 头注释同款）：
    # A=[[1,2],[3,4],[5,6]] (3x2) @ B=[[7,8,9,10],[11,12,13,14]] (2x4)
    A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    B = [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]]
    expect = [[29.0, 32.0, 35.0, 38.0],
              [65.0, 72.0, 79.0, 86.0],
              [101.0, 112.0, 123.0, 134.0]]
    C = matmul_cpu(A, B)
    assert len(C) == 3 and len(C[0]) == 4, f"输出形状应为 (3,4)，得到 {len(C)}x{len(C[0])}"
    for i in range(3):
        for j in range(4):
            assert math.isclose(C[i][j], expect[i][j], rel_tol=1e-6), \
                f"C[{i}][{j}]={C[i][j]} 期望 {expect[i][j]}"

    # 与 torch.matmul 对照（数值性质测试，不要求手写性能）
    torch.manual_seed(42)
    a = torch.randn(16, 8).tolist()
    b = torch.randn(8, 5).tolist()
    got = torch.tensor(matmul_cpu(a, b))
    want = torch.tensor(a) @ torch.tensor(b)
    assert torch.allclose(got, want, atol=1e-5), "与 torch.matmul 数值不一致"


# ═════════════════════════════════════════════════════════════════════
#  题 3：tiling 访存分析
# ═════════════════════════════════════════════════════════════════════

def test_exercise_3_tiling_reads():
    assert count_global_reads is not None, "count_global_reads 未实现"

    naive, tiled = count_global_reads(512, 512, 512, 32)
    assert naive == 512 * 512 * 2 * 512, "naive 应为 M*N*2*K"
    # tiled = (512/32)*(512/32) * 2*32*512 = 256 * 32768
    assert tiled == 16 * 16 * 2 * 32 * 512
    assert tiled < naive, "tiling 必须读得更少，否则公式写反了"

    assert tiled_speedup_ratio is not None, "tiled_speedup_ratio 未实现"
    for tile in (8, 16, 32, 64):
        ratio = tiled_speedup_ratio(512, 512, 512, tile)
        assert math.isclose(ratio, tile, rel_tol=1e-9), \
            f"读次数比值应恰好等于 tile={tile}，得到 {ratio}"


# ═════════════════════════════════════════════════════════════════════
#  题 4：GFLOPS 报告
# ═════════════════════════════════════════════════════════════════════

def test_exercise_4_gflops():
    assert gflops_report is not None, "gflops_report 未实现"

    # 已知数字：512^3 matmul 用 1ms -> GFLOPS = 2*512^3 / 0.001 / 1e9 ≈ 268.4
    rep = gflops_report(1.0, 512, 512, 512)
    assert isinstance(rep, dict), "应返回 dict"
    assert rep['flops'] == 2 * 512 ** 3
    assert math.isclose(rep['gflops'], 2 * 512 ** 3 / 1e-3 / 1e9, rel_tol=1e-9)
    assert rep['pct_of_peak'] is None, "未给 peak 时 pct_of_peak 应为 None"

    # 给峰值：脚本 04 实测 L5 ≈ 8000 GFLOPS @ 4090，fp32 峰值约 82.6 TFLOPS
    rep = gflops_report(0.031, 512, 512, 512, peak_gflops=82600)
    assert 0 < rep['pct_of_peak'] <= 100, "百分比应在 (0, 100]"
    assert math.isclose(rep['pct_of_peak'], rep['gflops'] / 82600 * 100, rel_tol=1e-9)


# ═════════════════════════════════════════════════════════════════════
#  题 5：Triton softmax（需要 GPU）
# ═════════════════════════════════════════════════════════════════════

def test_exercise_5_triton_softmax():
    if not _HAS_TRITON:
        _skip("需要 GPU + triton（无 GPU 同学可跳过本题，代码仍建议补全）")

    import torch
    assert triton_softmax is not None and callable(triton_softmax)
    torch.manual_seed(42)
    logits = torch.randn(128, 512, device='cuda')
    out = triton_softmax(logits)
    if out is None:
        _skip("triton_softmax 返回 None（未实现）")

    want = torch.softmax(logits, dim=1)
    assert out.shape == logits.shape
    assert torch.allclose(out, want, atol=1e-5), "与 torch.softmax 数值不一致"
    # 每行和为 1（softmax 的数学不变量）
    assert torch.allclose(out.sum(dim=1), torch.ones(128, device='cuda'), atol=1e-4)


# ─── 汇总 ─────────────────────────────────────────────────────────────

_TESTS = [
    ("题 1: 全局线程索引", test_exercise_1_global_index),
    ("题 2: 行/列主序 + CPU matmul", test_exercise_2_indexing_and_matmul),
    ("题 3: tiling 访存分析", test_exercise_3_tiling_reads),
    ("题 4: GFLOPS 报告", test_exercise_4_gflops),
    ("题 5: Triton softmax", test_exercise_5_triton_softmax),
]


def main():
    passed = failed = skipped = 0
    for name, test_fn in _TESTS:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except _Skipped as e:
            print(f"  ⏭️  {name} — SKIP: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"  ❌ {name} — FAIL: {e}")
            failed += 1
        except Exception as e:
            if isinstance(e, _Skipped):
                print(f"  ⏭️  {name} — SKIP: {e}")
                skipped += 1
            else:
                print(f"  ❌ {name} — ERROR: {e}")
                failed += 1

    total = passed + failed + skipped
    print(f"\n{'=' * 50}")
    print(f"  通过: {passed}/{total}, 失败: {failed}, 跳过: {skipped}")
    if failed == 0 and passed > 0:
        print(f"  🎉 全部通过！")
    elif passed == 0:
        print(f"  💡 所有题目都被跳过，请先实现 cuda_exercises.py 中的函数")
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
