#!/usr/bin/env python3
"""
Part 10 作业测试：分布式训练

两种运行方式：
  1. 独立运行：  python test_distributed_exercises.py
  2. pytest：    pytest test_distributed_exercises.py

题 1-4 未实现（返回 None）会失败；全部纯 CPU 可验证，无需 GPU。
"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from distributed_exercises import *  # noqa: F401,F403

try:
    import pytest
except ImportError:
    pytest = None

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _Skipped(Exception):
    pass


def _skip(reason):
    raise _Skipped(reason)


def _require_torch():
    if not _HAS_TORCH:
        _skip("未安装 torch（题 4 需要 torch 做张量运算）")


# ═════════════════════════════════════════════════════════════════════
#  题 1：all_reduce 平均语义
# ═════════════════════════════════════════════════════════════════════

def test_exercise_1_allreduce_mean():
    assert ddp_gradient is not None, "ddp_gradient 未实现"
    r = ddp_gradient([1.0, 2.0, 3.0, 4.0])
    assert r is not None, "ddp_gradient 未实现（返回 None）"
    assert math.isclose(r, 2.5, rel_tol=1e-9), f"4 rank 梯度 [1,2,3,4] 的平均应为 2.5，得到 {r}"
    assert math.isclose(ddp_gradient([6.0, 0.0]), 3.0), "负例验证失败"

    assert effective_batch is not None, "effective_batch 未实现"
    assert effective_batch(16, 2, 2) == 64, "16×accum2×2卡 = 64"
    assert effective_batch(32, 1, 8) == 256, "单 accum 8 卡 = 256"


# ═════════════════════════════════════════════════════════════════════
#  题 2：显存账本
# ═════════════════════════════════════════════════════════════════════

def test_exercise_2_memory():
    assert model_state_bytes is not None, "model_state_bytes 未实现"
    psi, n = 1_000_000_000, 8          # 1B 参数，8 卡

    assert math.isclose(model_state_bytes(psi, n, 'ddp'), 16 * psi, rel_tol=1e-9)
    assert math.isclose(model_state_bytes(psi, n, 'zero1'), 4 * psi + 12 * psi / n, rel_tol=1e-9)
    assert math.isclose(model_state_bytes(psi, n, 'zero2'), 8 * psi + 4 * psi / n, rel_tol=1e-9)
    assert math.isclose(model_state_bytes(psi, n, 'zero3'), 16 * psi / n, rel_tol=1e-9)
    # DDP 最大、ZeRO-3 最小；但 zero1 与 zero2 的大小取决于 N：
    #   zero1=4+12/N, zero2=8+4/N → N>2 时 zero1 更小，N=2 时 zero2 更小（反直觉！）
    v = {s: model_state_bytes(psi, n, s) for s in ('ddp', 'zero1', 'zero2', 'zero3')}
    assert v['ddp'] == max(v.values()), "DDP 应该是显存上限"
    assert v['zero3'] == min(v.values()), "ZeRO-3 应该最省"
    try:
        model_state_bytes(psi, n, 'unknown')
        assert False, "未知 stage 应抛 ValueError"
    except ValueError:
        pass

    # 7B + ZeRO-3 在 8 张 24GB 卡上：16*7e9/8 = 14GB；+6GB 激活 = 20GB < 24GB → 可训
    assert can_train_7b_on_24gb is not None, "can_train_7b_on_24gb 未实现"
    r = can_train_7b_on_24gb(8)
    assert r is not None, "can_train_7b_on_24gb 未实现（返回 None）"
    assert r is True, "8 卡 ZeRO-3：14GB+6GB=20GB < 24GB，应为 True"
    # 2 卡：56GB 状态直接爆
    assert can_train_7b_on_24gb(2) is False, "2 卡 ZeRO-3：56GB+6GB >> 24GB，应为 False"


# ═════════════════════════════════════════════════════════════════════
#  题 3：DistributedSampler 性质
# ═════════════════════════════════════════════════════════════════════

def test_exercise_3_sampler():
    assert sampler_indices is not None, "sampler_indices 未实现"
    n, world = 10, 3
    idx = sampler_indices(n, world, 0, seed=0)
    assert idx is not None, "sampler_indices 未实现（返回 None）"
    total = math.ceil(n / world) * world
    assert len(idx) == total // world, f"每 rank 长度应 = total/world = {total // world}，得到 {len(idx)}"
    assert all(0 <= i < n for i in idx), "索引必须落在 [0, n) 内（补齐样本除外）"
    # 相同种子 → 相同分片；不同 rank → 不同分片
    assert idx == sampler_indices(n, world, 0, seed=0), "同种子应可复现"
    idx1 = sampler_indices(n, world, 1, seed=0)
    assert idx != idx1, "不同 rank 的分片应不同"
    # 不同 epoch（种子）→ 不同分片
    assert idx != sampler_indices(n, world, 0, seed=1), "不同种子应产生不同分片（set_epoch 的意义）"

    assert sampler_coverage_ok is not None, "sampler_coverage_ok 未实现"
    r = sampler_coverage_ok(n, world)
    assert r is not None, "sampler_coverage_ok 未实现（返回 None）"
    assert r is True, "(n=10, world=3) 应满足：全覆盖 + 各 rank 内部无重复"
    for n_, w_ in [(7, 2), (16, 4), (5, 4), (100, 8)]:
        assert sampler_coverage_ok(n_, w_, seed=3) is True, f"(n={n_}, world={w_}) 覆盖检查失败"


# ═════════════════════════════════════════════════════════════════════
#  题 4：TP 分块数学
# ═════════════════════════════════════════════════════════════════════

def test_exercise_4_tp_math():
    _require_torch()
    assert tp_mlp_max_error is not None, "tp_mlp_max_error 未实现"
    torch.manual_seed(42)
    B, IN, H4, OUT = 8, 64, 128, 32
    X = torch.randn(B, IN)
    W1 = torch.randn(H4, IN) / IN ** 0.5
    W2 = torch.randn(OUT, H4) / H4 ** 0.5
    err = tp_mlp_max_error(X, W1, W2, n_shards=2)
    assert err is not None, "tp_mlp_max_error 未实现（返回 None）"
    assert err < 1e-5, f"列/行并行 vs 稠密的最大误差应 < 1e-5（脚本 05 实测 ~6e-7），得到 {err}"
    err4 = tp_mlp_max_error(X, W1, W2, n_shards=4)
    assert err4 < 1e-5, f"4 分片也应一致，得到 {err4}"


# ═════════════════════════════════════════════════════════════════════
#  题 5：🌟 流水线气泡
# ═════════════════════════════════════════════════════════════════════

def test_exercise_5_bubble():
    assert pipeline_bubble_fraction is not None, "pipeline_bubble_fraction 未实现"
    r = pipeline_bubble_fraction(2, 4)
    assert r is not None, "pipeline_bubble_fraction 未实现（返回 None）"
    assert math.isclose(r, 1 / 5, rel_tol=1e-9), f"p=2,m=4 → (2-1)/(4+2-1)=0.2，得到 {r}"
    assert math.isclose(pipeline_bubble_fraction(8, 64), 7 / 71, rel_tol=1e-9)
    assert pipeline_bubble_fraction(4, 4) > pipeline_bubble_fraction(4, 16), \
        "micro-batch 越多气泡应越小"

    # in_flight_activations 需要补全签名为 (schedule, m, p)
    import inspect
    assert in_flight_activations is not None, "in_flight_activations 未实现"
    try:
        sig = inspect.signature(in_flight_activations)
        result = in_flight_activations('gpipe', 8, 4)
        assert result is not None, "in_flight_activations 未实现（返回 None）"
        assert result == 8, "gpipe 的激活驻留 = m = 8"
        assert in_flight_activations('1f1b', 8, 4) == 4, "1f1b 的激活驻留 = p = 4"
    except TypeError:
        _skip("请把签名补全为 in_flight_activations(schedule, m, p)")


# ─── 汇总 ─────────────────────────────────────────────────────────────

_TESTS = [
    ("题 1: all_reduce 平均语义", test_exercise_1_allreduce_mean),
    ("题 2: 显存账本计算器", test_exercise_2_memory),
    ("题 3: DistributedSampler 不重不漏", test_exercise_3_sampler),
    ("题 4: TP 分块数学", test_exercise_4_tp_math),
    ("题 5: 🌟 流水线气泡", test_exercise_5_bubble),
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
        print(f"  💡 所有题目都被跳过，请先实现 distributed_exercises.py 中的函数")
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
