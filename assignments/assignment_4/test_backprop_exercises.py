"""
test_backprop_exercises.py - Assignment 4 测试

运行方式：
    python test_backprop_exercises.py          # 直跑，末尾给得分
    pytest test_backprop_exercises.py -v      # pytest 下逻辑失败会变红

测试内容：
    - Q1: forward_pass 结果与 F.cross_entropy 对比
    - Q2: backward_tanh / linear / softmax_ce 梯度与 autograd 对比（误差 < 1e-5）
    - Q3: cross_entropy_backward 梯度误差 < 1e-5
    - Q4: batchnorm_backward 梯度误差 < 1e-5
    - Q5 (拓展): manual_train 最终 loss 检查

设计说明：
    - 所有检查最终落到 assert：pytest 下任何逻辑失败都会红，直跑下计为未通过
    - 前置题未完成（如 Q1 未实现导致 Q3 拿不到 logits）时优雅跳过（SKIP），
      而不是 KeyError 崩溃
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

# 确保能 import exercises
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8")

from backprop_exercises import (
    forward_pass, backward_tanh, backward_linear, backward_bn_scale,
    backward_softmax_ce, cross_entropy_backward, batchnorm_backward,
    manual_train, get_test_batch, get_test_params, cmp,
    Xtr, Ytr, Xdev, Ydev, vocab_size, block_size
)


class _Checks:
    """收集一个测试内的所有检查结果，最后统一 assert"""

    def __init__(self, section):
        self.section = section
        self.problems = []

    def check(self, cond, msg):
        print(f"  {'✅' if cond else '❌'} {msg}")
        if not cond:
            self.problems.append(msg)

    def assert_all(self):
        assert not self.problems, f"{self.section} 有 {len(self.problems)} 项未通过: " + "；".join(self.problems)


class _SkipAssignment(Exception):
    """前置题未完成时的跳过信号（直跑模式）"""


def _skip_unready(msg):
    """Q1 未实现导致后续测试没有输入：优雅跳过，而不是 KeyError 崩溃"""
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest
        pytest.skip(msg)
    raise _SkipAssignment(msg)


def test_q1_forward_pass():
    """测试 Q1: forward_pass"""
    print("\n" + "=" * 60)
    print("Q1: forward_pass 测试")
    print("=" * 60)

    ck = _Checks("Q1")
    params = get_test_params()
    Xb, Yb = get_test_batch()

    cache = forward_pass(params, Xb, Yb)

    ck.check(cache is not None and 'loss' in cache, "forward_pass 返回 cache 且含 loss")
    if ck.problems:
        ck.assert_all()  # 立刻红：Q1 本身未实现

    # 检查必要字段
    required_keys = ['emb', 'embcat', 'hprebn', 'bnraw', 'hpreact', 'h', 'logits', 'loss']
    missing = [k for k in required_keys if k not in cache]
    ck.check(not missing, f"必要字段齐全" if not missing else f"缺少字段: {missing}")

    # 检查 loss 值
    logits = cache['logits']
    loss_ref = F.cross_entropy(logits, Yb)
    loss_diff = abs(cache['loss'].item() - loss_ref.item())
    ck.check(loss_diff < 1e-4, f"loss 误差: {loss_diff:.2e} (应 < 1e-4)")

    # 检查形状
    B = Xb.shape[0]
    n_embd = params['C'].shape[1]
    n_hidden = params['W1'].shape[1]

    shape_checks = [
        ('emb', (B, block_size, n_embd)),
        ('embcat', (B, n_embd * block_size)),
        ('hprebn', (B, n_hidden)),
        ('h', (B, n_hidden)),
        ('logits', (B, vocab_size)),
    ]

    for name, expected_shape in shape_checks:
        actual_shape = tuple(cache[name].shape)
        ck.check(actual_shape == expected_shape,
                 f"{name} 形状: {actual_shape} (期望 {expected_shape})")

    ck.assert_all()


def test_q2_backward_step():
    """测试 Q2: backward_step（backward_tanh / linear / softmax_ce）"""
    print("\n" + "=" * 60)
    print("Q2: backward_step 测试")
    print("=" * 60)

    ck = _Checks("Q2")
    params = get_test_params()
    Xb, Yb = get_test_batch()
    B = Xb.shape[0]

    # 前向传播
    cache = forward_pass(params, Xb, Yb)
    if cache is None or 'h' not in cache or 'embcat' not in cache or 'logits' not in cache:
        _skip_unready("Q1 forward_pass 未实现（cache 缺字段），Q2 没有输入可用")

    # ── 测试 backward_tanh ──
    try:
        dh = torch.randn_like(cache['h'])  # 模拟上游梯度
        h_no_grad = cache['h'].detach()
        dhpreact = backward_tanh(dh, h_no_grad)

        if dhpreact is None:
            ck.check(False, "backward_tanh 未实现（返回 None）")
        else:
            # 手动验证：dhpreact = dh * (1 - h^2)
            dhpreact_ref = dh * (1.0 - h_no_grad ** 2)
            diff = (dhpreact - dhpreact_ref).abs().max().item()
            ck.check(diff < 1e-5, f"backward_tanh: max diff = {diff:.2e}")
    except Exception as e:
        ck.problems.append(f"backward_tanh 异常: {e}")
        print(f"  ❌ backward_tanh 异常: {e}")

    # ── 测试 backward_linear ──
    try:
        input_t = cache['embcat'].detach()
        W = params['W2']
        dout = torch.randn(B, vocab_size)

        result = backward_linear(dout, input_t, W)
        if result is None or result[0] is None:
            ck.check(False, "backward_linear 未实现（返回 None）")
        else:
            dinput, dweight = result
            # 参考值
            dinput_ref = dout @ W.T
            dweight_ref = input_t.T @ dout

            diff_input = (dinput - dinput_ref).abs().max().item()
            diff_weight = (dweight - dweight_ref).abs().max().item()
            ok = diff_input < 1e-5 and diff_weight < 1e-5
            ck.check(ok, f"backward_linear: dinput diff={diff_input:.2e}, dweight diff={diff_weight:.2e}")
    except Exception as e:
        ck.problems.append(f"backward_linear 异常: {e}")
        print(f"  ❌ backward_linear 异常: {e}")

    # ── 测试 backward_softmax_ce ──
    try:
        logits = cache['logits'].detach()
        dlogits = backward_softmax_ce(logits, Yb)

        if dlogits is None:
            ck.check(False, "backward_softmax_ce 未实现（返回 None）")
        else:
            # autograd 参考值
            logits_ref = logits.clone().requires_grad_(True)
            loss_ref = F.cross_entropy(logits_ref, Yb)
            loss_ref.backward()
            dlogits_ref = logits_ref.grad

            diff = (dlogits - dlogits_ref).abs().max().item()
            ck.check(diff < 1e-5, f"backward_softmax_ce: max diff = {diff:.2e}")
    except Exception as e:
        ck.problems.append(f"backward_softmax_ce 异常: {e}")
        print(f"  ❌ backward_softmax_ce 异常: {e}")

    ck.assert_all()


def test_q3_cross_entropy_backward():
    """测试 Q3: cross_entropy_backward"""
    print("\n" + "=" * 60)
    print("Q3: cross_entropy_backward 测试")
    print("=" * 60)

    ck = _Checks("Q3")
    params = get_test_params()
    Xb, Yb = get_test_batch()

    cache = forward_pass(params, Xb, Yb)
    if cache is None or 'logits' not in cache:
        _skip_unready("Q1 forward_pass 未实现（cache 缺 logits），Q3 没有输入可用")

    logits = cache['logits'].detach()
    dlogits = cross_entropy_backward(logits, Yb)

    if dlogits is None:
        ck.check(False, "cross_entropy_backward 未实现（返回 None）")
        ck.assert_all()

    # autograd 参考
    logits_ref = logits.clone().requires_grad_(True)
    loss_ref = F.cross_entropy(logits_ref, Yb)
    loss_ref.backward()
    dlogits_ref = logits_ref.grad

    # 比较形状
    shape_ok = tuple(dlogits.shape) == tuple(dlogits_ref.shape)
    ck.check(shape_ok, f"形状: {tuple(dlogits.shape)} (期望 {tuple(dlogits_ref.shape)})")

    # 比较梯度值
    diff = (dlogits - dlogits_ref).abs().max().item()
    mean_diff = (dlogits - dlogits_ref).abs().mean().item()
    ck.check(diff < 1e-5, f"max diff = {diff:.2e}, mean diff = {mean_diff:.2e} (应 < 1e-5)")

    ck.assert_all()


def test_q4_batchnorm_backward():
    """测试 Q4: batchnorm_backward"""
    print("\n" + "=" * 60)
    print("Q4: batchnorm_backward 测试")
    print("=" * 60)

    ck = _Checks("Q4")
    params = get_test_params()
    Xb, Yb = get_test_batch()

    # 重新做前向，保留 BN 中间变量
    C, W1 = params['C'], params['W1']
    bngain, bnbias = params['bngain'], params['bnbias']
    W2, b2 = params['W2'], params['b2']
    B = Xb.shape[0]

    # 参数必须是带梯度的叶子，否则 loss.backward() 会报 "does not require grad"
    for p_ in params.values():
        p_.requires_grad_(True)

    emb = C[Xb]
    embcat = emb.view(B, -1)
    hprebn = embcat @ W1 + params['b1']

    # BatchNorm
    bnmeani = hprebn.mean(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = bndiff2.mean(0, keepdim=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    hpreact.retain_grad()   # ⚠️ 非叶子节点的 .grad 默认不保存，必须 retain_grad

    # autograd 参考（重建一条独立的 BN 子图）
    hprebn_retain = hprebn.detach().requires_grad_(True)
    bnmeani2 = hprebn_retain.mean(0, keepdim=True)
    bndiff2_2 = hprebn_retain - bnmeani2
    bndiff2_sq = bndiff2_2 ** 2
    bnvar2 = bndiff2_sq.mean(0, keepdim=True)
    bnvar_inv2 = (bnvar2 + 1e-5) ** -0.5
    bnraw2 = bndiff2_2 * bnvar_inv2
    hpreact2 = bngain * bnraw2 + bnbias
    h2 = torch.tanh(hpreact2)
    logits2 = h2 @ W2 + b2
    loss2 = F.cross_entropy(logits2, Yb)
    loss2.backward()
    dhprebn_ref = hprebn_retain.grad

    # 拿到 dhpreact
    loss.backward()
    dhpreact = hpreact.grad.clone()

    # 测试 batchnorm_backward
    dhprebn = batchnorm_backward(dhpreact, bnraw.detach(), bngain, bnvar_inv.detach())

    if dhprebn is None:
        ck.check(False, "batchnorm_backward 未实现（返回 None）")
        ck.assert_all()

    # 比较形状
    shape_ok = tuple(dhprebn.shape) == tuple(dhprebn_ref.shape)
    ck.check(shape_ok, f"形状: {tuple(dhprebn.shape)} (期望 {tuple(dhprebn_ref.shape)})")

    # 比较梯度值。前向方差是 1/n 有偏口径，正确的简化公式第三项系数为 1，
    # 与 autograd 对拍 max diff 约 1e-9 量级，1e-5 阈值足够
    # （旧版公式误用 n/(n-1) 系数时实测 ≈4.8e-05，会被此阈值正确判红）
    diff = (dhprebn - dhprebn_ref).abs().max().item()
    mean_diff = (dhprebn - dhprebn_ref).abs().mean().item()
    ck.check(diff < 1e-5, f"max diff = {diff:.2e}, mean diff = {mean_diff:.2e} (应 < 1e-5)")

    ck.assert_all()


def test_q5_manual_train():
    """测试 Q5 (拓展): manual_train"""
    print("\n" + "=" * 60)
    print("Q5 (拓展): manual_train 测试")
    print("=" * 60)

    ck = _Checks("Q5 (拓展)")

    try:
        result = manual_train(
            n_embd=10, n_hidden=64,
            max_steps=1000,  # 少量步骤快速测试
            batch_size=32, lr=0.1, seed=42
        )
    except Exception as e:
        ck.problems.append(f"manual_train 异常: {e}")
        print(f"  ❌ manual_train 异常: {e}")
        ck.assert_all()
        return

    ck.check(result is not None, "manual_train 返回结果")
    if result is None:
        ck.assert_all()

    lossi = result.get('lossi', [])
    ck.check(bool(lossi), f"有 loss 记录（{len(lossi)} 步）")
    if not lossi:
        ck.assert_all()

    print(f"  📊 训练 {len(lossi)} 步")
    print(f"     初始 loss ≈ {lossi[0]:.4f}")
    print(f"     最终 loss ≈ {lossi[-1]:.4f}")

    # 检查 loss 是否在下降
    first_100_avg = sum(lossi[:100]) / len(lossi[:100]) if len(lossi) >= 100 else lossi[0]
    last_100_avg = sum(lossi[-100:]) / len(lossi[-100:]) if len(lossi) >= 100 else lossi[-1]

    ck.check(last_100_avg < first_100_avg,
             f"loss 在下降 ({first_100_avg:.4f} → {last_100_avg:.4f})")

    ck.assert_all()


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("📝 Assignment 4 测试")
    print("=" * 60)

    results = {}

    for name, fn in [
        ('Q1', test_q1_forward_pass),
        ('Q2', test_q2_backward_step),
        ('Q3', test_q3_cross_entropy_backward),
        ('Q4', test_q4_batchnorm_backward),
        ('Q5', test_q5_manual_train),
    ]:
        try:
            fn()
            results[name] = 'pass'
        except _SkipAssignment as e:
            print(f"\n  ⏭️ {name} 跳过: {e}")
            results[name] = 'skip'
        except AssertionError as e:
            print(f"\n  ❌ {name} 未通过: {e}")
            results[name] = 'fail'

    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)

    for name, status in results.items():
        mark = {'pass': "✅ 通过", 'fail': "❌ 未通过", 'skip': "⏭️ 跳过（前置题未完成）"}[status]
        suffix = " (拓展)" if name == "Q5" else ""
        print(f"  {name}{suffix}: {mark}")

    n_pass = sum(1 for v in results.values() if v == 'pass')
    n_done = sum(1 for v in results.values() if v != 'skip')
    print(f"\n  得分: {n_pass}/{n_done}（跳过 {len(results) - n_done} 题不计入）")

    if n_pass == n_done:
        print("\n🎉 全部通过！你对反向传播的理解已经很扎实了！")
    elif n_pass >= n_done - 1:
        print("\n👍 基本通过！拓展题继续加油！")
    else:
        print("\n💪 再检查一下哪些题没通过，参考教程重新推导一遍。")

    sys.exit(0 if n_pass == n_done else 1)
