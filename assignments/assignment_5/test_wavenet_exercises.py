#!/usr/bin/env python3
"""Part 5 作业测试"""

import torch
import sys
import os

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from wavenet_exercises import *

# 数据文件路径
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')


def test_flatten_consecutive():
    """测试 FlattenConsecutive：输出 shape 正确"""
    # 基本测试
    fc = FlattenConsecutive(2)
    x = torch.randn(4, 8, 10)
    y = fc(x)

    assert y.shape == (4, 4, 20), (
        f"FlattenConsecutive(2) 应把 (4,8,10) 变成 (4,4,20)，得到 {y.shape}"
    )
    print(f"  (4,8,10) → FC(2) → {y.shape} ✅")

    # n=4
    fc4 = FlattenConsecutive(4)
    x = torch.randn(4, 8, 10)
    y = fc4(x)
    assert y.shape == (4, 2, 40), (
        f"FlattenConsecutive(4) 应把 (4,8,10) 变成 (4,2,40)，得到 {y.shape}"
    )
    print(f"  (4,8,10) → FC(4) → {y.shape} ✅")

    # 参数为空
    assert fc.parameters() == [], "FlattenConsecutive 应无可训练参数"
    print(f"  parameters() = [] ✅")

    # 不可整除时应报错
    try:
        fc_bad = FlattenConsecutive(3)
        fc_bad(torch.randn(4, 7, 10))  # 7 % 3 != 0
        assert False, "T 不能被 n 整除时应报错"
    except (AssertionError, RuntimeError):
        print(f"  T%n!=0 时报错 ✅")

    print("  ✅ test_flatten_consecutive")


def test_batchnorm_3d():
    """测试 BatchNorm1d3D：2D 和 3D 输入都正确"""
    dim = 10

    # === 2D 测试 ===
    bn = BatchNorm1d3D(dim)
    assert hasattr(bn, 'training'), "应有 training 属性"
    assert bn.training == True, "默认应为 training 模式"

    x2d = torch.randn(32, dim) * 3 + 2
    y2d = bn(x2d)

    assert y2d.shape == (32, dim), f"2D 输出 shape 错误: {y2d.shape}"
    assert bn.running_mean.shape == (dim,), (
        f"running_mean 应为 1D ({dim},)，得到 {bn.running_mean.shape}"
    )
    # running_mean 应该被更新（不再是全 0）
    assert not torch.allclose(bn.running_mean, torch.zeros(dim)), (
        "训练后 running_mean 应被更新"
    )
    print(f"  2D: running_mean shape = {bn.running_mean.shape} ✅")

    # === 3D 测试 ===
    bn3 = BatchNorm1d3D(dim)
    x3d = torch.randn(32, 4, dim) * 3 + 2
    y3d = bn3(x3d)

    assert y3d.shape == (32, 4, dim), f"3D 输出 shape 错误: {y3d.shape}"
    assert bn3.running_mean.shape == (dim,), (
        f"3D 输入后 running_mean 应仍为 1D ({dim},)，得到 {bn3.running_mean.shape}"
    )

    # running_mean 应该被更新
    assert not torch.allclose(bn3.running_mean, torch.zeros(dim)), (
        "3D 训练后 running_mean 应被更新"
    )
    print(f"  3D: running_mean shape = {bn3.running_mean.shape} ✅")

    # === Eval 模式 ===
    bn3.training = False
    x3d_new = torch.randn(16, 4, dim)
    y3d_eval = bn3(x3d_new)

    assert y3d_eval.shape == (16, 4, dim), f"eval 3D shape 错误: {y3d_eval.shape}"

    # eval 模式不应更新 running stats
    rm_before = bn3.running_mean.clone()
    _ = bn3(x3d_new)
    assert torch.equal(bn3.running_mean, rm_before), (
        "eval 模式不应更新 running_mean"
    )
    print(f"  eval 模式不更新 stats ✅")

    # === parameters ===
    params = bn3.parameters()
    assert len(params) == 2, f"应有 2 个参数，得到 {len(params)}"
    print(f"  parameters() 返回 {len(params)} 个参数 ✅")

    print("  ✅ test_batchnorm_3d")


def test_build_wavenet():
    """测试 build_wavenet：模型能 forward，输出 shape 正确"""
    model = build_wavenet(vocab_size=27, n_embd=10, n_hidden=68, block_size=8)

    if model is None:
        print("  ⏭️ build_wavenet 未实现，跳过")
        return

    # Forward 测试
    x = torch.randint(0, 27, (4, 8))
    logits = model(x)

    assert logits.shape == (4, 27), (
        f"输出 shape 应为 (4, 27)，得到 {logits.shape}"
    )
    print(f"  Forward: (4, 8) → {logits.shape} ✅")

    # 参数量检查
    total_params = sum(p.nelement() for p in model.parameters())
    assert total_params > 0, "模型应有参数"
    print(f"  参数量: {total_params:,} ✅")

    # 所有参数有梯度
    for p in model.parameters():
        assert p.requires_grad, "所有参数应 requires_grad=True"
    print(f"  所有参数 requires_grad=True ✅")

    print("  ✅ test_build_wavenet")


def test_verify_shapes():
    """测试 verify_shapes：返回正确的 shape 列表"""
    model = build_wavenet(vocab_size=27, n_embd=10, n_hidden=68, block_size=8)

    if model is None:
        print("  ⏭️ 依赖 build_wavenet，跳过")
        return

    shapes = verify_shapes(model, vocab_size=27, block_size=8, batch_size=4)

    if shapes is None or len(shapes) == 0:
        print("  ⏭️ verify_shapes 未实现，跳过")
        return

    # 检查返回格式
    assert isinstance(shapes, list), "应返回列表"
    for item in shapes:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"每个元素应为 (name, shape) 元组，得到 {item}"
        )

    # 最终 shape 应该是 (4, 27)
    last_name, last_shape = shapes[-1]
    assert last_shape == (4, 27), (
        f"最终输出 shape 应为 (4, 27)，得到 {last_shape}"
    )

    print(f"  共 {len(shapes)} 层:")
    for name, shape in shapes:
        print(f"    {name}: {shape}")
    print("  ✅ test_verify_shapes")


def test_train_wavenet():
    """拓展题：训练 WaveNet，验证 dev loss"""
    words = open(_DATA_PATH, 'r').read().splitlines()

    result = train_wavenet(words, n_embd=24, n_hidden=128, block_size=8,
                           steps=10000, seed=42)
    if result is None:
        print("  ⏭️ 拓展题未实现，跳过")
        return

    dev_loss = result[0]

    assert isinstance(dev_loss, float), f"dev_loss 应为 float，得到 {type(dev_loss)}"
    assert dev_loss > 0, f"dev_loss 应 > 0，得到 {dev_loss}"

    # 宽松阈值：10000 步后应明显低于随机
    assert dev_loss < 2.5, f"10000 步后 dev_loss 应 < 2.5，得到 {dev_loss:.4f}"

    print(f"  dev_loss (10000 steps): {dev_loss:.4f}")
    print("  ✅ test_train_wavenet（拓展）")


if __name__ == '__main__':
    print("=" * 50)
    print("Part 5 作业测试")
    print("=" * 50)

    tests = [
        test_flatten_consecutive,
        test_batchnorm_3d,
        test_build_wavenet,
        test_verify_shapes,
        test_train_wavenet,
    ]

    passed = 0
    failed = 0
    for test in tests:
        print(f"\n▶ {test.__name__}")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"结果: {passed} 通过, {failed} 失败")
    if failed == 0:
        print("🎉 全部通过！")
    else:
        print("还有题目需要完成哦～")
