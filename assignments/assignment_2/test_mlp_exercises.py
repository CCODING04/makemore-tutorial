#!/usr/bin/env python3
"""Part 2 作业测试"""

import torch
import sys
import os

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from mlp_exercises import *

# 数据文件路径（基于脚本位置解析）
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')

# 固定随机种子
torch.manual_seed(2147483647)


def test_build_dataset():
    """测试 build_dataset：shape、dtype、值范围"""
    # 小数据集
    words = ['emma']
    X, Y = build_dataset(words, block_size=3)

    # Shape
    assert X.shape == (5, 3), f"X.shape 错误: {X.shape}, 期望 (5, 3)"
    assert Y.shape == (5,), f"Y.shape 错误: {Y.shape}, 期望 (5,)"

    # Dtype
    assert X.dtype == torch.int64, f"X.dtype 错误: {X.dtype}, 期望 torch.int64"
    assert Y.dtype == torch.int64, f"Y.dtype 错误: {Y.dtype}, 期望 torch.int64"

    # 值范围 [0, 26]
    assert X.min() >= 0 and X.max() <= 26, f"X 值范围错误: [{X.min()}, {X.max()}]"
    assert Y.min() >= 0 and Y.max() <= 26, f"Y 值范围错误: [{Y.min()}, {Y.max()}]"

    # 第一个样本：上下文全 0，目标 'e'=5
    assert (X[0] == 0).all(), f"第一个样本的上下文应该全为 0，得到 {X[0]}"
    assert Y[0] == 5, f"第一个目标应该是 5 ('e')，得到 {Y[0]}"

    # 最后一个样本：'mma' → '.'
    assert Y[-1] == 0, f"最后一个目标应该是 0 ('.')，得到 {Y[-1]}"

    # 大数据集
    words = open(_DATA_PATH, 'r').read().splitlines()
    X2, Y2 = build_dataset(words, block_size=3)
    assert X2.shape[0] == Y2.shape[0], "X 和 Y 的样本数应一致"
    assert X2.shape[1] == 3, f"block_size=3 时 X 应有 3 列，得到 {X2.shape[1]}"

    print("  ✅ test_build_dataset")


def test_mlp_forward():
    """测试 mlp_forward：logits shape (N, 27)、不是 NaN"""
    torch.manual_seed(2147483647)

    words = open(_DATA_PATH, 'r').read().splitlines()
    X, Y = build_dataset(words[:5], block_size=3)

    n_embd, n_hidden, block_size = 10, 100, 3
    C = torch.randn(27, n_embd)
    W1 = torch.randn(block_size * n_embd, n_hidden)
    b1 = torch.randn(n_hidden)
    W2 = torch.randn(n_hidden, 27)
    b2 = torch.randn(27)

    logits = mlp_forward(X, C, W1, b1, W2, b2)

    # Shape
    N = X.shape[0]
    assert logits.shape == (N, 27), f"logits shape 错误: {logits.shape}, 期望 ({N}, 27)"

    # 不是 NaN
    assert not torch.isnan(logits).any(), "logits 中有 NaN"

    # 不是全零
    assert not (logits == 0).all(), "logits 不应该全为零"

    print("  ✅ test_mlp_forward")


def test_train_step():
    """测试 train_step：loss 是标量、> 0、梯度存在"""
    torch.manual_seed(2147483647)

    words = open(_DATA_PATH, 'r').read().splitlines()
    X, Y = build_dataset(words[:10], block_size=3)

    n_embd, n_hidden, block_size = 10, 100, 3
    C = torch.randn(27, n_embd, requires_grad=True)
    W1 = torch.randn(block_size * n_embd, n_hidden, requires_grad=True)
    b1 = torch.randn(n_hidden, requires_grad=True)
    W2 = torch.randn(n_hidden, 27, requires_grad=True)
    b2 = torch.randn(27, requires_grad=True)

    loss = train_step(X, Y, C, W1, b1, W2, b2, lr=0.1)

    # Loss 是标量 float
    assert isinstance(loss, float), f"loss 应该是 float，得到 {type(loss)}"

    # Loss > 0
    assert loss > 0, f"loss 应该 > 0，得到 {loss}"

    # Loss 合理范围（交叉熵初始随机 ≈ log(27) ≈ 3.3，不应太大）
    assert loss < 10, f"loss 不应太大（< 10），得到 {loss}"

    # 梯度存在且非零
    for name, param in [('C', C), ('W1', W1), ('b1', b1), ('W2', W2), ('b2', b2)]:
        assert param.grad is not None, f"{name} 的梯度为 None"
        assert not (param.grad == 0).all(), f"{name} 的梯度全为零"

    print(f"  train_step loss: {loss:.4f}")
    print("  ✅ test_train_step")


def test_evaluate():
    """测试 evaluate：返回标量 loss，不修改参数"""
    torch.manual_seed(2147483647)

    words = open(_DATA_PATH, 'r').read().splitlines()
    X, Y = build_dataset(words[:10], block_size=3)

    n_embd, n_hidden, block_size = 10, 100, 3
    C = torch.randn(27, n_embd, requires_grad=True)
    W1 = torch.randn(block_size * n_embd, n_hidden, requires_grad=True)
    b1 = torch.randn(n_hidden, requires_grad=True)
    W2 = torch.randn(n_hidden, 27, requires_grad=True)
    b2 = torch.randn(27, requires_grad=True)

    # 记录参数原始值
    C_orig = C.data.clone()
    W1_orig = W1.data.clone()

    loss = evaluate(X, Y, C, W1, b1, W2, b2)

    # 返回 float
    assert isinstance(loss, float), f"loss 应该是 float，得到 {type(loss)}"

    # Loss > 0
    assert loss > 0, f"loss 应该 > 0，得到 {loss}"

    # 参数不应该被修改
    assert torch.equal(C.data, C_orig), "evaluate 不应该修改 C"
    assert torch.equal(W1.data, W1_orig), "evaluate 不应该修改 W1"

    print(f"  evaluate loss: {loss:.4f}")
    print("  ✅ test_evaluate")


def test_tuning():
    """拓展题：调参实验，验证 loss < 2.5"""
    words = open(_DATA_PATH, 'r').read().splitlines()

    result = tuning_experiment(words, block_size=3, n_embd=10, n_hidden=100,
                              steps=1000, lr=0.1, seed=2147483647)
    if result is None:
        print("  ⏭️ 拓展题未实现，跳过")
        return

    val_loss, C, W1, b1, W2, b2 = result

    assert isinstance(val_loss, float), f"val_loss 应该是 float"
    assert val_loss > 0, f"val_loss 应该 > 0，得到 {val_loss}"

    # 宽松的阈值：1000 步后只要不是随机水平就行
    assert val_loss < 2.5, f"1000 步后 val_loss 应该 < 2.5，得到 {val_loss}"

    print(f"  tuning val_loss: {val_loss:.4f}")
    print("  ✅ test_tuning（拓展）")


if __name__ == '__main__':
    print("=" * 50)
    print("Part 2 作业测试")
    print("=" * 50)

    tests = [
        test_build_dataset,
        test_mlp_forward,
        test_train_step,
        test_evaluate,
        test_tuning,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print()
    print(f"结果: {passed} 通过, {failed} 失败")
    if failed == 0:
        print("🎉 全部通过！")
    else:
        print("还有题目需要完成哦～")
