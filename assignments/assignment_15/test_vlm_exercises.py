#!/usr/bin/env python3
"""Assignment 15 测试。独立运行：python test_vlm_exercises.py；或 pytest。"""

import os
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_exercises import *  # noqa: F401,F403


def test_ex1_patch():
    assert patch_tokens is not None, "patch_tokens 未实现"
    assert patch_tokens(224, 224, 14) == 256, "224/14=16 → 16×16=256（LLaVA/CLIP）"
    assert patch_tokens(336, 336, 14) == 576, "LLaVA-1.5 的 336² = 576 token"
    assert patch_tokens(1024, 768, 14) == 4070, "1024/14→74, 768/14→55 → 74×55=4070"
    s = vit_out_shape(576, 1024) if vit_out_shape else None
    assert s is not None and s == (576, 1024), "ViT 不改变 token 数与维度"


def test_ex2_infonce():
    if torch is None:
        raise AssertionError("需要 torch")
    assert infonce_loss is not None, "infonce_loss 未实现"
    torch.manual_seed(0)
    f_img = F.normalize(torch.randn(8, 16), dim=-1)
    f_txt = F.normalize(torch.randn(8, 16), dim=-1)
    loss = infonce_loss(f_img, f_txt, 10.0)
    assert loss is not None and loss.item() > 0, "InfoNCE 应为正标量"
    # 对称性：转置后的双重 CE 均值 = 手动对称实现
    logits = 10.0 * f_img @ f_txt.T
    labels = torch.arange(8)
    manual = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    assert torch.allclose(loss, manual, atol=1e-6), "应实现对称双方向"
    # 配对完美时（单位阵塔输出）loss → 0
    eye = F.normalize(torch.eye(8), dim=-1)
    assert infonce_loss(eye, eye, 10.0).item() < 1e-3, "完美配对时 loss≈0"


def test_ex3_projector():
    assert mlp2x_params is not None, "mlp2x_params 未实现"
    v = mlp2x_params(1024, 4096)
    want = (1024 * 4096 + 4096) + (4096 * 4096 + 4096)
    assert v == want, f"mlp2x 参数应 = {want}，得到 {v}"


def test_ex4_dynamic():
    assert dynamic_tokens is not None, "dynamic_tokens 未实现"
    # 小图不触发预算：1024×768 → 4070 raw → /4 ≈ 1017（<2560 预算）
    v = dynamic_tokens(1024, 768, patch=14, compress=4, max_tokens=2560)
    assert v is not None and v == 1017, f"4070/4 = 1017.5 → floor 1017，得到 {v}"
    # 超预算触发缩放：4× 大图 raw 16 倍 → 必须缩到 ≤ max_tokens
    v2 = dynamic_tokens(4096, 3072, patch=14, compress=4, max_tokens=2560)
    assert v2 is not None and 0 < v2 <= 2560, f"预算控制失败: {v2}"


_TESTS = [("题1 patch/形状", test_ex1_patch), ("题2 InfoNCE", test_ex2_infonce),
          ("题3 投影器参数", test_ex3_projector), ("题4 🌟动态分辨率", test_ex4_dynamic)]


def main():
    p = f = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    print(f"\n  通过: {p}/{p + f}" + ("  🎉" if f == 0 else "  💡 先实现 vlm_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
