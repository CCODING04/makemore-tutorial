#!/usr/bin/env python3
"""Part 1 作业测试"""

import torch
import sys
import os

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from bigram_exercises import *

# 数据文件路径（基于脚本位置解析）
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')

# 固定随机种子
torch.manual_seed(2147483647)


def test_build_bigram_matrix():
    words = ['emma', 'olivia', 'ava']
    N = build_bigram_matrix(words)
    assert N.shape == (27, 27), f"形状错误: {N.shape}"
    assert N.dtype == torch.int32, f"dtype 错误: {N.dtype}"
    assert N.sum() > 0, "矩阵应该有非零元素"
    # 检查 '.' 后面跟 'e' 的计数
    stoi = {s: i+1 for i, s in enumerate('abcdefghijklmnopqrstuvwxyz')}
    stoi['.'] = 0
    assert N[0, stoi['e']] > 0, "'.e' 应该有计数"
    print("  ✅ test_build_bigram_matrix")


def test_compute_probabilities():
    words = ['emma', 'olivia', 'ava']
    N = build_bigram_matrix(words)
    P = compute_probabilities(N)
    assert P.shape == (27, 27), f"形状错误: {P.shape}"
    # 每行概率和应该为 1
    row_sums = P.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(27), atol=1e-6), f"行和不为1: {row_sums}"
    # 所有概率应该 > 0 (因为 smoothing)
    assert (P > 0).all(), "有零概率，检查 smoothing"
    print("  ✅ test_compute_probabilities")


def test_generate_names():
    words = open(_DATA_PATH, 'r').read().splitlines()
    N = build_bigram_matrix(words)
    P = compute_probabilities(N)
    names = generate_names(P, n=5, seed=42)
    assert len(names) == 5, f"应该生成5个名字，得到 {len(names)}"
    assert all(isinstance(n, str) for n in names), "名字应该是字符串"
    assert all(len(n) > 0 for n in names), "名字不能为空"
    print(f"  生成结果: {names}")
    print("  ✅ test_generate_names")


def test_compute_nll_loss():
    words = open(_DATA_PATH, 'r').read().splitlines()
    N = build_bigram_matrix(words)
    P = compute_probabilities(N)
    loss = compute_nll_loss(P, words)
    assert isinstance(loss, float) or (isinstance(loss, torch.Tensor) and loss.ndim == 0), \
        f"loss 应该是标量"
    loss_val = loss if isinstance(loss, float) else loss.item()
    assert loss_val > 0, f"loss 应该 > 0，得到 {loss_val}"
    assert loss_val < 10, f"loss 不应该太大（< 10），得到 {loss_val}"
    # 合理范围：均匀分布的 NLL = log(27) ≈ 3.3，好的模型应该更好
    print(f"  平均 NLL: {loss_val:.4f}")
    print("  ✅ test_compute_nll_loss")


def test_train_bigram_nn():
    words = open(_DATA_PATH, 'r').read().splitlines()
    result = train_bigram_nn(words, epochs=50, lr=50, seed=2147483647)
    if result is None:
        print("  ⏭️ 拓展题未实现，跳过")
        return
    W, final_loss = result
    assert W.shape == (27, 27), f"W 形状错误: {W.shape}"
    loss_val = final_loss if isinstance(final_loss, float) else final_loss.item()
    assert loss_val > 0, f"loss 应该 > 0"
    # 50 步后 loss 应该在合理范围
    print(f"  50 步后 loss: {loss_val:.4f}")
    print("  ✅ test_train_bigram_nn（拓展）")


if __name__ == '__main__':
    print("=" * 50)
    print("Part 1 作业测试")
    print("=" * 50)

    tests = [
        test_build_bigram_matrix,
        test_compute_probabilities,
        test_generate_names,
        test_compute_nll_loss,
        test_train_bigram_nn,
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
