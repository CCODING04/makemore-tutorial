#!/usr/bin/env python3
"""Part 3 作业测试"""

import torch
import sys
import os

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from batchnorm_exercises import *

# 数据文件路径（基于脚本位置解析）
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')


def test_diagnose_initial_loss():
    """测试 diagnose_initial_loss：loss 应该远大于 ln(27) ≈ 3.298"""
    words = open(_DATA_PATH, 'r').read().splitlines()

    loss = diagnose_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647)

    # 返回类型
    assert isinstance(loss, float), f"loss 应该是 float，得到 {type(loss)}"

    # loss 应该 > 3.0（未经修正时明显偏高）
    assert loss > 3.0, f"未经修正的初始 loss 应该 > 3.0，得到 {loss:.4f}"

    # loss 应该在合理范围内（不会爆炸）
    assert loss < 10.0, f"初始 loss 不应太大（< 10），得到 {loss:.4f}"

    print(f"  未经修正的初始 loss: {loss:.4f} (ln(27) ≈ 3.298)")
    print("  ✅ test_diagnose_initial_loss")


def test_fix_initial_loss():
    """测试 fix_initial_loss：loss 应该接近 ln(27) ≈ 3.298"""
    words = open(_DATA_PATH, 'r').read().splitlines()

    loss = fix_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647)

    # 返回类型
    assert isinstance(loss, float), f"loss 应该是 float，得到 {type(loss)}"

    # loss 应该接近 ln(27) ≈ 3.298
    assert abs(loss - 3.29) < 0.5, (
        f"修正后的初始 loss 应该接近 3.29（ln(27)），得到 {loss:.4f}"
    )

    # 修正后的 loss 应该明显小于未修正的
    loss_unfixed = diagnose_initial_loss(words, block_size=3, n_embd=10, n_hidden=200, seed=2147483647)
    assert loss < loss_unfixed, (
        f"修正后 loss ({loss:.4f}) 应该小于未修正 ({loss_unfixed:.4f})"
    )

    print(f"  修正后的初始 loss: {loss:.4f} (ln(27) ≈ 3.298)")
    print("  ✅ test_fix_initial_loss")


def test_batchnorm1d():
    """测试 BatchNorm1d：shape、running statistics、training/eval 行为"""
    torch.manual_seed(42)

    dim = 10
    bn = BatchNorm1d(dim, eps=1e-5, momentum=0.1)

    # === 基本属性检查 ===
    assert hasattr(bn, 'training'), "BatchNorm1d 应该有 training 属性"
    assert bn.training == True, "新建的 BN 层 training 应该为 True"

    assert hasattr(bn, 'gamma'), "BatchNorm1d 应该有 gamma 属性"
    assert hasattr(bn, 'beta'), "BatchNorm1d 应该有 beta 属性"
    assert hasattr(bn, 'running_mean'), "BatchNorm1d 应该有 running_mean 属性"
    assert hasattr(bn, 'running_var'), "BatchNorm1d 应该有 running_var 属性"

    # gamma 和 beta 的 shape
    assert bn.gamma.shape == (dim,), f"gamma shape 错误: {bn.gamma.shape}"
    assert bn.beta.shape == (dim,), f"beta shape 错误: {bn.beta.shape}"

    # gamma 初始化为 1，beta 初始化为 0
    assert torch.allclose(bn.gamma, torch.ones(dim)), "gamma 应初始化为全 1"
    assert torch.allclose(bn.beta, torch.zeros(dim)), "beta 应初始化为全 0"

    # running_mean 和 running_var 的 shape
    assert bn.running_mean.shape == (dim,), f"running_mean shape 错误"
    assert bn.running_var.shape == (dim,), f"running_var shape 错误"

    # parameters() 返回正确
    params = bn.parameters()
    assert len(params) == 2, f"parameters() 应返回 2 个参数，得到 {len(params)}"
    assert params[0] is bn.gamma, "parameters()[0] 应该是 gamma"
    assert params[1] is bn.beta, "parameters()[1] 应该是 beta"

    # === Training 模式 ===
    bn.training = True
    x = torch.randn(32, dim) * 3 + 2  # 非标准分布

    y = bn(x)

    # 输出 shape 正确
    assert y.shape == (32, dim), f"输出 shape 错误: {y.shape}, 期望 (32, {dim})"

    # 训练模式下输出应该被归一化（均值 ≈ beta=0，方差 ≈ gamma²=1）
    # 由于 batch 较小，容差放宽
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0)
    assert y_mean.abs().max() < 0.5, (
        f"训练模式下输出均值应接近 0，最大偏差: {y_mean.abs().max():.4f}"
    )

    # Running statistics 应该被更新（不再全 0 / 全 1）
    assert not torch.allclose(bn.running_mean, torch.zeros(dim)), (
        "训练后 running_mean 应该被更新"
    )

    # === Eval 模式 ===
    bn.training = False
    x2 = torch.randn(16, dim) * 3 + 2

    y2 = bn(x2)

    # 输出 shape 正确
    assert y2.shape == (16, dim), f"eval 模式输出 shape 错误: {y2.shape}"

    # 记录 eval 模式下的 running_mean
    running_mean_before = bn.running_mean.clone()
    running_var_before = bn.running_var.clone()

    # eval 模式不应更新 running statistics
    _ = bn(x2)
    assert torch.equal(bn.running_mean, running_mean_before), (
        "eval 模式不应更新 running_mean"
    )
    assert torch.equal(bn.running_var, running_var_before), (
        "eval 模式不应更新 running_var"
    )

    print(f"  BN training 输出均值范围: [{y_mean.min():.4f}, {y_mean.max():.4f}]")
    print(f"  BN running_mean 非零: {not torch.allclose(bn.running_mean, torch.zeros(dim))}")
    print("  ✅ test_batchnorm1d")


def test_tanh_saturation():
    """测试 diagnose_tanh_saturation：返回值在 [0, 1] 之间"""
    torch.manual_seed(42)

    # === 情况 1：几乎无饱和（小值输入）===
    hpreact_small = torch.randn(100, 200) * 0.5
    ratio_small = diagnose_tanh_saturation(hpreact_small)

    assert isinstance(ratio_small, float), f"返回值应该是 float，得到 {type(ratio_small)}"
    assert 0 <= ratio_small <= 1, f"饱和比例应在 [0, 1]，得到 {ratio_small}"
    assert ratio_small < 0.1, f"小值输入时饱和比例应很低，得到 {ratio_small:.4f}"

    # === 情况 2：严重饱和（大值输入）===
    hpreact_large = torch.randn(100, 200) * 5.0
    ratio_large = diagnose_tanh_saturation(hpreact_large)

    assert isinstance(ratio_large, float), f"返回值应该是 float，得到 {type(ratio_large)}"
    assert 0 <= ratio_large <= 1, f"饱和比例应在 [0, 1]，得到 {ratio_large}"
    assert ratio_large > 0.3, f"大值输入时饱和比例应较高，得到 {ratio_large:.4f}"

    # === 情况 3：全部饱和（极大值）===
    hpreact_extreme = torch.ones(50, 100) * 10.0
    ratio_extreme = diagnose_tanh_saturation(hpreact_extreme)

    assert ratio_extreme > 0.99, f"极大值输入时几乎全部饱和，得到 {ratio_extreme:.4f}"

    print(f"  小值输入 (×0.5): 饱和比例 {ratio_small:.4f}")
    print(f"  大值输入 (×5.0): 饱和比例 {ratio_large:.4f}")
    print(f"  极大值 (×10.0):  饱和比例 {ratio_extreme:.4f}")
    print("  ✅ test_tanh_saturation")


def test_deep_bn():
    """拓展题：含 BN 的深层网络，验证 dev loss"""
    words = open(_DATA_PATH, 'r').read().splitlines()

    result = train_deep_bn(words, block_size=3, n_embd=10, n_hidden=200,
                           steps=10000, seed=2147483647)
    if result is None:
        print("  ⏭️ 拓展题未实现，跳过")
        return

    dev_loss = result[0]

    assert isinstance(dev_loss, float), f"dev_loss 应该是 float，得到 {type(dev_loss)}"
    assert dev_loss > 0, f"dev_loss 应该 > 0，得到 {dev_loss}"

    # 宽松的阈值：10000 步后 loss 应该明显低于随机水平
    assert dev_loss < 2.5, f"10000 步后 dev_loss 应该 < 2.5，得到 {dev_loss:.4f}"

    print(f"  dev_loss (10000 steps): {dev_loss:.4f}")
    print("  ✅ test_deep_bn（拓展）")


if __name__ == '__main__':
    print("=" * 50)
    print("Part 3 作业测试")
    print("=" * 50)

    tests = [
        test_diagnose_initial_loss,
        test_fix_initial_loss,
        test_batchnorm1d,
        test_tanh_saturation,
        test_deep_bn,
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
