#!/usr/bin/env python3
"""
Part 8 作业测试：从零训练 LLM —— 后训练全流程

两种运行方式：
  1. 独立运行：  python test_post_training_exercises.py
  2. pytest：    pytest test_post_training_exercises.py

未实现的题目（函数返回 None）会优雅跳过而非报错。
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使运行更快更稳定
torch.set_num_threads(1)

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from post_training_exercises import *  # noqa: F401,F403


# ─── 跳过机制（兼容独立运行 与 pytest）─────────────────────────────
try:
    import pytest
except ImportError:
    pytest = None


class _Skipped(Exception):
    pass


def _skip(reason):
    """抛出自定义异常让主函数统计为跳过。"""
    raise _Skipped(reason)


def _is_skip(e):
    return isinstance(e, _Skipped)


# ═══════════════════════════════════════════════════════════════════
#  题 1：Causal Self-Attention Head
# ═══════════════════════════════════════════════════════════════════

def test_exercise_1_causal_head():
    """题 1：因果注意力头的 shape 和因果性"""
    torch.manual_seed(1337)

    B, T, n_embed, n_head = 2, 8, 32, 4
    head_size = n_embed // n_head

    try:
        head = Head(head_size, n_embed, context_length=16)
    except Exception:
        _skip("题 1 未实现：Head 类无法实例化")

    x = torch.randn(B, T, n_embed)
    try:
        out = head(x)
    except Exception:
        _skip("题 1 未实现：Head.forward 报错")

    if out is None:
        _skip("题 1 未实现（forward 返回 None）")

    assert out.shape == (B, T, head_size), \
        f"输出 shape 应为 {(B, T, head_size)}，得到 {out.shape}"

    # 因果性验证：改变 token t 的输入不应影响 token t-1 的输出
    x2 = x.clone()
    x2[:, 3, :] = torch.randn(B, n_embed)  # 改变 token 3
    out2 = head(x2)
    if out2 is None:
        _skip("题 1 未实现（forward 返回 None）")

    # token 0, 1, 2 的输出不应改变（因为 causal mask）
    assert torch.allclose(out[:, :3, :], out2[:, :3, :], atol=1e-5), \
        "因果性违反：改变 token 3 影响了 token 0-2 的输出"


# ═══════════════════════════════════════════════════════════════════
#  题 2：Pre-LN Transformer Block
# ═══════════════════════════════════════════════════════════════════

def test_exercise_2_preln_block():
    """题 2：Pre-LN Block 的 shape 一致性"""
    torch.manual_seed(1337)

    B, T, n_embed, n_head = 2, 8, 32, 4

    try:
        block = Block(n_head, n_embed, context_length=16)
    except Exception:
        _skip("题 2 未实现：Block 类无法实例化")

    x = torch.randn(B, T, n_embed)
    try:
        out = block(x)
    except Exception:
        _skip("题 2 未实现：Block.forward 报错")

    if out is None:
        _skip("题 2 未实现（forward 返回 None）")

    assert out.shape == (B, T, n_embed), \
        f"输出 shape 应为 {(B, T, n_embed)}，得到 {out.shape}"

    # 残差连接验证：block 输出不应等于零（说明有残差连接）
    assert out.abs().mean() > 0.01, \
        "输出接近零，可能缺少残差连接"


# ═══════════════════════════════════════════════════════════════════
#  题 3：Prompt-Masked SFT Loss
# ═══════════════════════════════════════════════════════════════════

def test_exercise_3_sft_loss():
    """题 3：SFT loss 的 mask 行为"""
    torch.manual_seed(1337)

    B, T, V = 2, 10, 100
    logits = torch.randn(B, T, V)
    tokens = torch.randint(0, V, (B, T))

    try:
        result = sft_loss(logits, tokens, torch.ones(B, T))
    except Exception:
        _skip("题 3 未实现：sft_loss 报错")

    if result is None:
        _skip("题 3 未实现（返回 None）")
    assert result.dim() == 0, f"loss 应为标量，得到 dim={result.dim()}"
    assert result.item() > 0, "loss 应为正数"

    # 全 mask 为 0 时 loss 应为 0
    loss_zero = sft_loss(logits, tokens, torch.zeros(B, T))
    if loss_zero is None:
        _skip("题 3 未实现（返回 None）")
    assert abs(loss_zero.item()) < 1e-6, \
        f"全 mask 为 0 时 loss 应为 0，得到 {loss_zero.item()}"

    # 全 mask 为 1 时等价于普通 CE
    loss_full = sft_loss(logits, tokens, torch.ones(B, T))
    if loss_full is None:
        _skip("题 3 未实现（返回 None）")
    # 手动计算普通 CE（shift 后）
    ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), tokens[:, 1:].reshape(-1))
    assert abs(loss_full.item() - ce.item()) < 1e-4, \
        f"全 mask 为 1 时 loss 应等于普通 CE ({ce.item():.4f})，得到 {loss_full.item():.4f}"

    # 部分 mask 测试
    mask = torch.ones(B, T)
    mask[:, :5] = 0  # 只在后 5 个位置算 loss
    loss_partial = sft_loss(logits, tokens, mask)
    if loss_partial is None:
        _skip("题 3 未实现（返回 None）")
    assert loss_partial.item() > 0, "部分 mask 的 loss 应为正数"


# ═══════════════════════════════════════════════════════════════════
#  题 4：Bradley-Terry Reward Loss
# ═══════════════════════════════════════════════════════════════════

def test_exercise_4_reward_loss():
    """题 4：Bradley-Terry loss 的基本性质"""
    torch.manual_seed(1337)

    B = 4
    r_chosen = torch.randn(B)
    r_rejected = torch.randn(B)

    try:
        result = reward_loss(r_chosen, r_rejected)
    except Exception:
        _skip("题 4 未实现：reward_loss 报错")

    if result is None:
        _skip("题 4 未实现（返回 None）")
    assert result.dim() == 0, f"loss 应为标量，得到 dim={result.dim()}"

    # 相等时 loss = ln(2) ≈ 0.693
    r_equal = torch.ones(B)
    loss_equal = reward_loss(r_equal, r_equal)
    if loss_equal is None:
        _skip("题 4 未实现（返回 None）")
    assert abs(loss_equal.item() - math.log(2)) < 0.01, \
        f"相等时 loss 应为 ln(2)={math.log(2):.4f}，得到 {loss_equal.item():.4f}"

    # chosen >> rejected 时 loss 应很小
    r_ch = torch.ones(B) * 10
    r_rej = torch.ones(B) * -10
    loss_good = reward_loss(r_ch, r_rej)
    if loss_good is None:
        _skip("题 4 未实现（返回 None）")
    assert loss_good.item() < 0.01, \
        f"chosen >> rejected 时 loss 应接近 0，得到 {loss_good.item():.4f}"

    # chosen << rejected 时 loss 应很大
    loss_bad = reward_loss(r_rej, r_ch)
    if loss_bad is None:
        _skip("题 4 未实现（返回 None）")
    assert loss_bad.item() > 10, \
        f"chosen << rejected 时 loss 应很大，得到 {loss_bad.item():.4f}"


# ═══════════════════════════════════════════════════════════════════
#  题 5：DPO Loss
# ═══════════════════════════════════════════════════════════════════

def test_exercise_5_dpo_loss():
    """题 5：DPO loss 的基本性质"""
    torch.manual_seed(1337)

    B = 4
    logps = torch.randn(B)

    try:
        result = dpo_loss(logps, logps - 0.5, logps, logps - 0.5)
    except Exception:
        _skip("题 5 未实现：dpo_loss 报错")

    if result is None:
        _skip("题 5 未实现（返回 None）")
    assert isinstance(result, tuple) and len(result) == 3, \
        f"应返回 (loss, chosen_reward, rejected_reward)，得到 {type(result)}"

    loss, chosen_reward, rejected_reward = result
    assert loss.dim() == 0, f"loss 应为标量，得到 dim={loss.dim()}"

    # pi == ref 时 loss = ln(2) ≈ 0.693
    pi_ch = torch.randn(B)
    pi_rej = torch.randn(B)
    loss_equal = dpo_loss(pi_ch, pi_rej, pi_ch, pi_rej)[0]
    if loss_equal is None:
        _skip("题 5 未实现（返回 None）")
    assert abs(loss_equal.item() - math.log(2)) < 0.01, \
        f"pi == ref 时 loss 应为 ln(2)={math.log(2):.4f}，得到 {loss_equal.item():.4f}"

    # policy 对 chosen 的 log-prob 相对更高时 loss 应下降
    pi_ch_good = pi_ch + 2.0  # 增大 chosen 的 log-prob
    loss_good = dpo_loss(pi_ch_good, pi_rej, pi_ch, pi_rej)[0]
    if loss_good is None:
        _skip("题 5 未实现（返回 None）")
    assert loss_good.item() < loss_equal.item(), \
        "policy 对 chosen 的 log-prob 更高时 loss 应下降"

    # chosen_reward 和 rejected_reward 应为 detached 的标量
    assert chosen_reward.shape == (B,), \
        f"chosen_reward shape 应为 ({B},)，得到 {chosen_reward.shape}"
    assert not chosen_reward.requires_grad, "chosen_reward 应该是 detached 的"


# ═══════════════════════════════════════════════════════════════════
#  题 6：GAE Advantage Estimation
# ═══════════════════════════════════════════════════════════════════

def test_exercise_6_gae():
    """题 6：GAE 的 λ=0 退化和 shape"""
    torch.manual_seed(1337)

    B, T = 2, 8
    rewards = torch.randn(B, T)
    values = torch.randn(B, T + 1)  # 包含 bootstrap

    try:
        adv = gae(rewards, values, gamma=1.0, lam=0.0)
    except Exception:
        _skip("题 6 未实现：gae 报错")

    if adv is None:
        _skip("题 6 未实现（返回 None）")
    assert adv.shape == (B, T), \
        f"advantage shape 应为 {(B, T)}，得到 {adv.shape}"

    # λ=0 时应退化为 TD error: A_t = r_t + γ * V(s_{t+1}) - V(s_t)
    expected_td = rewards + values[:, 1:] - values[:, :-1]
    assert torch.allclose(adv, expected_td, atol=1e-5), \
        "λ=0 时 GAE 应退化为 TD error"

    # λ=1 时应不同于 λ=0（除非 T=1）
    adv_lambda1 = gae(rewards, values, gamma=1.0, lam=1.0)
    if adv_lambda1 is None:
        _skip("题 6 未实现（返回 None）")
    if T > 1:
        assert not torch.allclose(adv, adv_lambda1, atol=1e-4), \
            "λ=0 和 λ=1 的 GAE 应该不同（除非 T=1）"


# ═══════════════════════════════════════════════════════════════════
#  题 7：PPO Clipped Loss
# ═══════════════════════════════════════════════════════════════════

def test_exercise_7_ppo_loss():
    """题 7：PPO clipped loss 的基本性质"""
    torch.manual_seed(1337)

    B, T = 2, 8
    logp = torch.randn(B, T)
    advantages = torch.randn(B, T)

    try:
        result = ppo_loss(logp, logp, advantages, eps=0.2)
    except Exception:
        _skip("题 7 未实现：ppo_loss 报错")

    if result is None:
        _skip("题 7 未实现（返回 None）")
    assert result.dim() == 0, f"loss 应为标量，得到 dim={result.dim()}"

    # logp_new == logp_old 时，ratio=1，loss = -mean(advantages)
    loss_same = ppo_loss(logp, logp, advantages, eps=0.2)
    if loss_same is None:
        _skip("题 7 未实现（返回 None）")
    expected = -advantages.mean()
    assert abs(loss_same.item() - expected.item()) < 1e-4, \
        f"ratio=1 时 loss 应为 -mean(adv)={expected.item():.4f}，得到 {loss_same.item():.4f}"

    # 裁剪验证：大幅改变 logp_new 时，loss 不应无限增大（被裁剪了）
    logp_big = logp + 10.0  # 大幅改变
    loss_clipped = ppo_loss(logp_big, logp, advantages, eps=0.2)
    if loss_clipped is None:
        _skip("题 7 未实现（返回 None）")
    # loss 不应为 inf 或 nan
    assert torch.isfinite(loss_clipped), \
        f"裁剪后 loss 不应为 inf/nan，得到 {loss_clipped.item()}"


# ═══════════════════════════════════════════════════════════════════
#  题 8：GRPO Group Advantage
# ═══════════════════════════════════════════════════════════════════

def test_exercise_8_group_advantages():
    """题 8：GRPO group advantage 的组内标准化"""
    torch.manual_seed(1337)

    num_prompts, group_size = 3, 4
    rewards = torch.randn(num_prompts * group_size)

    try:
        adv = group_advantages(rewards, group_size, eps=1e-8)
    except Exception:
        _skip("题 8 未实现：group_advantages 报错")

    if adv is None:
        _skip("题 8 未实现（返回 None）")
    assert adv.shape == rewards.shape, \
        f"advantage shape 应为 {rewards.shape}，得到 {adv.shape}"

    # 每组内 advantage 均值应为 0
    adv_groups = adv.view(num_prompts, group_size)
    group_means = adv_groups.mean(dim=1)
    assert torch.allclose(group_means, torch.zeros_like(group_means), atol=1e-4), \
        f"每组内 advantage 均值应为 0，得到 {group_means}"

    # 每组内 advantage 标准差应为 1（eps 可忽略时）
    group_stds = adv_groups.std(dim=1)
    assert torch.allclose(group_stds, torch.ones_like(group_stds), atol=0.1), \
        f"每组内 advantage 标准差应接近 1，得到 {group_stds}"

    # 组内所有奖励相同时 advantage 应为 0
    rewards_same = torch.ones(num_prompts * group_size) * 5.0
    adv_same = group_advantages(rewards_same, group_size, eps=1e-8)
    if adv_same is None:
        _skip("题 8 未实现（返回 None）")
    assert torch.allclose(adv_same, torch.zeros_like(adv_same), atol=1e-4), \
        f"组内奖励相同时 advantage 应为 0，得到 {adv_same}"


# ═══════════════════════════════════════════════════════════════════
#  运行所有测试
# ═══════════════════════════════════════════════════════════════════

_TESTS = [
    ("题 1: Causal Head", test_exercise_1_causal_head),
    ("题 2: Pre-LN Block", test_exercise_2_preln_block),
    ("题 3: SFT Loss", test_exercise_3_sft_loss),
    ("题 4: Reward Loss", test_exercise_4_reward_loss),
    ("题 5: DPO Loss", test_exercise_5_dpo_loss),
    ("题 6: GAE", test_exercise_6_gae),
    ("题 7: PPO Loss", test_exercise_7_ppo_loss),
    ("题 8: GRPO Advantage", test_exercise_8_group_advantages),
]


def main():
    passed = 0
    failed = 0
    skipped = 0

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
            if _is_skip(e):
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
        print(f"  💡 所有题目都被跳过，请先实现 post_training_exercises.py 中的函数")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
