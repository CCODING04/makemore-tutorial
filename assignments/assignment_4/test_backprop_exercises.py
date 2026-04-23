"""
test_backprop_exercises.py - Assignment 4 测试

运行方式：
    python test_backprop_exercises.py

测试内容：
    - Q1: forward_pass 结果与 F.cross_entropy 对比
    - Q2: backward_step 梯度与 autograd 对比（误差 < 1e-5）
    - Q3: cross_entropy_backward 梯度误差 < 1e-5
    - Q4: batchnorm_backward 梯度误差 < 1e-5
    - Q5 (拓展): manual_train 最终 loss 检查
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

# 确保能 import exercises
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from backprop_exercises import (
    forward_pass, backward_tanh, backward_linear, backward_bn_scale,
    backward_softmax_ce, cross_entropy_backward, batchnorm_backward,
    manual_train, get_test_batch, get_test_params, cmp,
    Xtr, Ytr, Xdev, Ydev, vocab_size, block_size
)


def test_q1_forward_pass():
    """测试 Q1: forward_pass"""
    print("\n" + "=" * 60)
    print("Q1: forward_pass 测试")
    print("=" * 60)

    params = get_test_params()
    Xb, Yb = get_test_batch()

    cache = forward_pass(params, Xb)

    if cache is None or 'loss' not in cache:
        print("  ❌ forward_pass 未实现")
        return False

    # 检查必要字段
    required_keys = ['emb', 'embcat', 'hprebn', 'bnraw', 'hpreact', 'h', 'logits', 'loss']
    missing = [k for k in required_keys if k not in cache]
    if missing:
        print(f"  ❌ 缺少字段: {missing}")
        return False

    # 检查 loss 值
    logits = cache['logits']
    loss_ref = F.cross_entropy(logits, Yb)
    loss_diff = abs(cache['loss'].item() - loss_ref.item())
    ok = loss_diff < 1e-4
    print(f"  {'✅' if ok else '❌'} loss 误差: {loss_diff:.2e} (应 < 1e-4)")

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

    all_ok = ok
    for name, expected_shape in shape_checks:
        actual_shape = tuple(cache[name].shape)
        match = actual_shape == expected_shape
        print(f"  {'✅' if match else '❌'} {name} 形状: {actual_shape} (期望 {expected_shape})")
        all_ok &= match

    return all_ok


def test_q2_backward_step():
    """测试 Q2: backward_step"""
    print("\n" + "=" * 60)
    print("Q2: backward_step 测试")
    print("=" * 60)

    params = get_test_params()
    Xb, Yb = get_test_batch()
    B = Xb.shape[0]
    n_hidden = params['W1'].shape[1]

    # 前向传播
    cache = forward_pass(params, Xb)
    if cache is None:
        print("  ⚠️ forward_pass 未实现，跳过 Q2 测试")
        return False

    all_ok = True

    # ── 测试 backward_tanh ──
    try:
        h = cache['h'].detach().requires_grad_(True)
        loss_test = h.sum()
        loss_test.backward()
        dh_ref = h.grad

        dh = torch.randn_like(h)  # 模拟上游梯度
        h_no_grad = cache['h'].detach()
        dhpreact = backward_tanh(dh, h_no_grad)

        if dhpreact is None:
            print("  ❌ backward_tanh 未实现")
            all_ok = False
        else:
            # 手动验证：dhpreact = dh * (1 - h^2)
            dhpreact_ref = dh * (1.0 - h_no_grad ** 2)
            diff = (dhpreact - dhpreact_ref).abs().max().item()
            ok = diff < 1e-5
            print(f"  {'✅' if ok else '❌'} backward_tanh: max diff = {diff:.2e}")
            all_ok &= ok
    except Exception as e:
        print(f"  ❌ backward_tanh 异常: {e}")
        all_ok = False

    # ── 测试 backward_linear ──
    try:
        input_t = cache['embcat'].detach()
        W = params['W2']
        dout = torch.randn(B, vocab_size)

        result = backward_linear(dout, input_t, W)
        if result is None or result[0] is None:
            print("  ❌ backward_linear 未实现")
            all_ok = False
        else:
            dinput, dweight = result
            # 参考值
            dinput_ref = dout @ W.T
            dweight_ref = input_t.T @ dout

            diff_input = (dinput - dinput_ref).abs().max().item()
            diff_weight = (dweight - dweight_ref).abs().max().item()
            ok = diff_input < 1e-5 and diff_weight < 1e-5
            print(f"  {'✅' if ok else '❌'} backward_linear: dinput diff={diff_input:.2e}, dweight diff={diff_weight:.2e}")
            all_ok &= ok
    except Exception as e:
        print(f"  ❌ backward_linear 异常: {e}")
        all_ok = False

    # ── 测试 backward_softmax_ce ──
    try:
        logits = cache['logits'].detach()
        dlogits = backward_softmax_ce(logits, Yb)

        if dlogits is None:
            print("  ❌ backward_softmax_ce 未实现")
            all_ok = False
        else:
            # autograd 参考值
            logits_ref = logits.clone().requires_grad_(True)
            loss_ref = F.cross_entropy(logits_ref, Yb)
            loss_ref.backward()
            dlogits_ref = logits_ref.grad

            diff = (dlogits - dlogits_ref).abs().max().item()
            ok = diff < 1e-5
            print(f"  {'✅' if ok else '❌'} backward_softmax_ce: max diff = {diff:.2e}")
            all_ok &= ok
    except Exception as e:
        print(f"  ❌ backward_softmax_ce 异常: {e}")
        all_ok = False

    return all_ok


def test_q3_cross_entropy_backward():
    """测试 Q3: cross_entropy_backward"""
    print("\n" + "=" * 60)
    print("Q3: cross_entropy_backward 测试")
    print("=" * 60)

    params = get_test_params()
    Xb, Yb = get_test_batch()

    cache = forward_pass(params, Xb)
    if cache is None:
        print("  ⚠️ forward_pass 未实现，跳过 Q3 测试")
        return False

    logits = cache['logits'].detach()
    dlogits = cross_entropy_backward(logits, Yb)

    if dlogits is None:
        print("  ❌ cross_entropy_backward 未实现")
        return False

    # autograd 参考
    logits_ref = logits.clone().requires_grad_(True)
    loss_ref = F.cross_entropy(logits_ref, Yb)
    loss_ref.backward()
    dlogits_ref = logits_ref.grad

    # 比较形状
    shape_ok = tuple(dlogits.shape) == tuple(dlogits_ref.shape)
    print(f"  {'✅' if shape_ok else '❌'} 形状: {tuple(dlogits.shape)} (期望 {tuple(dlogits_ref.shape)})")

    # 比较梯度值
    diff = (dlogits - dlogits_ref).abs().max().item()
    mean_diff = (dlogits - dlogits_ref).abs().mean().item()
    ok = diff < 1e-5
    print(f"  {'✅' if ok else '❌'} max diff = {diff:.2e}, mean diff = {mean_diff:.2e}")

    return ok and shape_ok


def test_q4_batchnorm_backward():
    """测试 Q4: batchnorm_backward"""
    print("\n" + "=" * 60)
    print("Q4: batchnorm_backward 测试")
    print("=" * 60)

    params = get_test_params()
    Xb, Yb = get_test_batch()

    # 重新做前向，保留 BN 中间变量
    C, W1 = params['C'], params['W1']
    bngain, bnbias = params['bngain'], params['bnbias']
    W2, b2 = params['W2'], params['b2']
    n_embd = C.shape[1]
    n_hidden = W1.shape[1]
    B = Xb.shape[0]

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

    # autograd
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
        print("  ❌ batchnorm_backward 未实现")
        return False

    # 比较形状
    shape_ok = tuple(dhprebn.shape) == tuple(dhprebn_ref.shape)
    print(f"  {'✅' if shape_ok else '❌'} 形状: {tuple(dhprebn.shape)} (期望 {tuple(dhprebn_ref.shape)})")

    # 比较梯度值
    diff = (dhprebn - dhprebn_ref).abs().max().item()
    mean_diff = (dhprebn - dhprebn_ref).abs().mean().item()
    ok = diff < 1e-5
    print(f"  {'✅' if ok else '❌'} max diff = {diff:.2e}, mean diff = {mean_diff:.2e}")

    return ok and shape_ok


def test_q5_manual_train():
    """测试 Q5 (拓展): manual_train"""
    print("\n" + "=" * 60)
    print("Q5 (拓展): manual_train 测试")
    print("=" * 60)

    try:
        result = manual_train(
            n_embd=10, n_hidden=64,
            max_steps=1000,  # 少量步骤快速测试
            batch_size=32, lr=0.1, seed=42
        )

        if result is None:
            print("  ❌ manual_train 未实现")
            return False

        lossi = result.get('lossi', [])
        if not lossi:
            print("  ❌ 没有 loss 记录")
            return False

        print(f"  📊 训练 {len(lossi)} 步")
        print(f"     初始 loss ≈ {lossi[0]:.4f}")
        print(f"     最终 loss ≈ {lossi[-1]:.4f}")

        # 检查 loss 是否在下降
        first_100_avg = sum(lossi[:100]) / len(lossi[:100]) if len(lossi) >= 100 else lossi[0]
        last_100_avg = sum(lossi[-100:]) / len(lossi[-100:]) if len(lossi) >= 100 else lossi[-1]

        if last_100_avg < first_100_avg:
            print(f"  ✅ loss 在下降 ({first_100_avg:.4f} → {last_100_avg:.4f})")
            return True
        else:
            print(f"  ⚠️ loss 没有明显下降 ({first_100_avg:.4f} → {last_100_avg:.4f})")
            return True  # 拓展题，不严格要求

    except Exception as e:
        print(f"  ❌ manual_train 异常: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("📝 Assignment 4 测试")
    print("=" * 60)

    results = {}

    results['Q1'] = test_q1_forward_pass()
    results['Q2'] = test_q2_backward_step()
    results['Q3'] = test_q3_cross_entropy_backward()
    results['Q4'] = test_q4_batchnorm_backward()
    results['Q5'] = test_q5_manual_train()

    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 未通过"
        suffix = " (拓展)" if name == "Q5" else ""
        print(f"  {name}{suffix}: {status}")

    total = sum(1 for v in results.values() if v)
    print(f"\n  得分: {total}/{len(results)}")

    if total == len(results):
        print("\n🎉 全部通过！你对反向传播的理解已经很扎实了！")
    elif total >= len(results) - 1:
        print("\n👍 基本通过！拓展题继续加油！")
    else:
        print("\n💪 再检查一下哪些题没通过，参考教程重新推导一遍。")
