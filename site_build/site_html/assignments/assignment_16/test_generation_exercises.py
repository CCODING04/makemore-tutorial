#!/usr/bin/env python3
"""Assignment 16 测试。独立运行：python test_generation_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generation_exercises import *  # noqa: F401,F403


def test_ex1_q_sample():
    assert q_sample is not None, "q_sample 未实现"
    torch.manual_seed(0)
    T = 50
    betas = torch.linspace(1e-4, 0.02, T)
    ac = torch.cumprod(1 - betas, dim=0)
    x0 = torch.randn(4, 16)
    t = torch.tensor([0, 10, 25, 49])
    noise = torch.randn(4, 16)
    x_t = q_sample(x0, ac, t, noise)
    assert x_t is not None and x_t.shape == x0.shape, "q_sample 应返回与 x0 同形状的 (B, D) 张量"
    # 手动按行验证闭式
    for i in range(4):
        s = math.sqrt(ac[t[i]])
        want = s * x0[i] + math.sqrt(1 - ac[t[i]]) * noise[i]
        assert torch.allclose(x_t[i], want, atol=1e-6), f"第 {i} 行闭式不符"
    # 信号比例单调递减
    assert signal_ratio is not None and signal_ratio(0, betas) > signal_ratio(T - 1, betas)
    assert abs(signal_ratio(0, betas) - 1.0) < 1e-4, "t=0 时几乎无损"


def test_ex2_cfg():
    assert cfg is not None, "cfg 未实现"
    u, c = torch.zeros(2, 4), torch.ones(2, 4)
    r = cfg(u, c, 7.5)
    assert r is not None and torch.allclose(r, torch.full((2, 4), 7.5)), "uncond=0,cond=1,w=7.5 → 7.5"
    assert torch.allclose(cfg(u, c, 0.0), u), "w=0 退化为无条件"


def test_ex3_strength():
    assert img2img_start_step is not None, "img2img_start_step 未实现"
    assert img2img_start_step(1.0, 50) == 50, "strength=1 → 全步（纯文生图）"
    assert img2img_start_step(0.5, 50) == 25, "0.5 → 25"
    assert img2img_start_step(0.02, 50) == 1, "极小 strength → 几乎照抄参考图"


def test_ex4_ipa():
    """🌟 Stretch：未实现（返回 None）时优雅 SKIP，不判 FAIL。"""
    torch.manual_seed(0)
    Q = torch.randn(2, 16, 32)
    Kt, Vt = torch.randn(2, 8, 32), torch.randn(2, 8, 32)
    Kr, Vr = torch.randn(2, 4, 32), torch.randn(2, 4, 32)
    out0 = decoupled_cross_attn(Q, Kt, Vt, Kr, Vr, scale=0.0)
    if out0 is None:  # 未实现 → SKIP（pytest 下 pytest.skip，独立运行下返回标记）
        if "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip("🌟 Stretch 未实现（返回 None）——实现后此测试自动生效")
        return "SKIP"
    out1 = decoupled_cross_attn(Q, Kt, Vt, Kr, Vr, scale=1.0)
    assert out0.shape == (2, 16, 32), f"输出形状应为 (2,16,32)，got {tuple(out0.shape)}"
    # scale=0 → 纯文本注意力；与参考分支独立可加
    txt_only = F.softmax(Q @ Kt.transpose(-2, -1) / 32 ** 0.5, -1) @ Vt
    assert torch.allclose(out0, txt_only, atol=1e-6), "scale=0 应等于纯文本注意力"
    ref_branch = F.softmax(Q @ Kr.transpose(-2, -1) / 32 ** 0.5, -1) @ Vr
    assert torch.allclose(out1, txt_only + ref_branch, atol=1e-5), "解耦注入应为线性可加"


_TESTS = [("题1 DDPM 闭式", test_ex1_q_sample), ("题2 CFG", test_ex2_cfg),
          ("题3 img2img strength", test_ex3_strength),
          ("题4 🌟 Stretch 解耦交叉注意力", test_ex4_ipa)]


def main():
    p = f = s = 0
    for name, fn in _TESTS:
        try:
            r = fn()
            if r == "SKIP":
                print(f"  ⏭️  {name} — SKIP（未实现，不扣分）"); s += 1
            else:
                print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    msg = f"\n  通过: {p}/{p + f}" + (f"（另 {s} 项 SKIP ⏭️）" if s else "")
    print(msg + ("  🎉" if f == 0 else "  💡 先实现 generation_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
