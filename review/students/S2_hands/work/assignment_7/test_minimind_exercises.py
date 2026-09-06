#!/usr/bin/env python3
"""
Part 7 作业测试：从零复现 minimind —— 现代 LLM 六大组件

两种运行方式：
  1. 独立运行：  python test_minimind_exercises.py
  2. pytest：    pytest test_minimind_exercises.py

未实现的题目（函数返回 None）会优雅跳过而非报错。
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使运行更快更稳定
torch.set_num_threads(1)

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from minimind_exercises import *  # noqa: F401,F403


# ─── 跳过机制（兼容独立运行 与 pytest）─────────────────────────────
try:
    import pytest
except ImportError:
    pytest = None


class _Skipped(Exception):
    pass


def _skip(reason):
    """在 pytest 下优雅跳过；独立运行时抛自定义异常让主函数统计为跳过。"""
    if pytest is not None:
        pytest.skip(reason)
    raise _Skipped(reason)


def _is_skip(e):
    if isinstance(e, _Skipped):
        return True
    if pytest is not None and isinstance(e, pytest.skip.Exception):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
#  题 1：BPE 编码
# ═══════════════════════════════════════════════════════════════════

def test_exercise_1_bpe_encode():
    """题 1：rank 最小编合、合并顺序、id 映射、往返还原"""
    torch.manual_seed(1337)

    # 简单合并规则：先合并 (a,b) 成 'ab'，再合并 (ab,c) 成 'abc'
    merges = [('a', 'b'), ('ab', 'c')]
    # 词表：单字符 + 两个合并产物
    tokens = ['a', 'b', 'c', 'ab', 'abc']
    vocab = {t: i for i, t in enumerate(tokens)}

    result = exercise_1_bpe_encode('abcabc', merges, vocab)
    if result is None:
        _skip("题 1 未实现")

    assert isinstance(result, list), "返回应是 list[int]"

    # 期望：每 3 个字符合并成一个 'abc' token
    expected_ids = [vocab['abc'], vocab['abc']]
    assert result == expected_ids, \
        f"'abcabc' 应编码为 {expected_ids}（rank0 先并 ab，rank1 再并 abc），得到 {result}"

    # 编码后 decode 能还原：把所有 token 拼接回来 == 原文
    decoded = ''.join(tokens[i] for i in result)
    assert decoded == 'abcabc', f"encode 后应能还原原文，得到 {decoded!r}"

    # 无合并可做的纯字符（'z' 不在词表则跳过；这里只用词表内字符）
    result2 = exercise_1_bpe_encode('ab', merges, vocab)
    assert result2 == [vocab['ab']], f"'ab' 应合并为一个 token，得到 {result2}"

    print(f"  'abcabc' -> {result}, decode 还原 ✅")

    # 更复杂的：无重叠的多个合并
    merges2 = [('h', 'e'), ('he', 'llo')]  # 先 he 后 hello（llo 是单字符+两字符，这里简化为 he+l）
    tokens2 = ['h', 'e', 'l', 'o', 'he', 'hel', 'llo']
    vocab2 = {t: i for i, t in enumerate(tokens2)}
    # 'hello' -> ['h','e','l','l','o'] -> 并 ('h','e')->'he' -> ['he','l','l','o']
    # 找不到 ('he','l') 的合并（不在 merges 里），'he','l' 保持分离 → ['he','l','l','o']
    # 需要 ('l','l') 在 merges 里才继续。这里 merges2 没有，所以停在这里。
    res3 = exercise_1_bpe_encode('hello', [('h', 'e')], vocab2)
    assert res3 == [vocab2['he'], vocab2['l'], vocab2['l'], vocab2['o']], \
        f"'hello' 仅并 he 时应为 ['he','l','l','o']，得到 {res3}"

    print(f"  'hello'（仅并 he）-> {res3} ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 2：RMSNorm
# ═══════════════════════════════════════════════════════════════════

def test_exercise_2_rmsnorm():
    """题 2：形状、均方根≈weight、常数输入、无除零"""
    torch.manual_seed(1337)
    dim, B, T = 32, 4, 8

    norm = RMSNorm(dim)
    # 骨架状态（__init__ 未实现，weight 不存在）应优雅跳过
    if not hasattr(norm, 'weight') or norm.forward(torch.zeros(1, dim)) is None:
        _skip("题 2 未实现")

    x = torch.randn(B, T, dim)
    out = norm(x)

    # 形状保持
    assert out.shape == (B, T, dim), \
        f"输出形状应为 (B,T,dim)，得到 {tuple(out.shape)}"

    # 前向后再 normalize，均方根应≈weight（全 1）
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(B, T), atol=1e-4), \
        f"RMSNorm 输出的均方根应≈1（weight 全 1），得到 {rms.mean().item():.4f}"

    # 常数正输入：输出应≈weight/1 = 1（x 为全 1 时，mean(x²)=1，rms=1）
    x_const = torch.full((2, 4, dim), 2.0)
    out_const = norm(x_const)
    expected = torch.full_like(x_const, 2.0) / 2.0  # x / sqrt(4) = 1
    assert torch.allclose(out_const, expected, atol=1e-5), \
        f"全 2 输入应输出全 1，得到 {out_const[0,0,:3].tolist()}"

    # 零输入不崩溃（除零被 eps 保护）
    out_zero = norm(torch.zeros(2, 4, dim))
    assert torch.isfinite(out_zero).all(), "零输入不应产生 NaN/Inf"

    print(f"  shape={tuple(out.shape)}, 输出均方根≈1, 常数输入正确, 零输入安全 ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 3：RoPE 旋转位置编码
# ═══════════════════════════════════════════════════════════════════

def test_exercise_3_apply_rope():
    """题 3：形状、旋转后范数不变、单位旋转的实数公式对照"""
    torch.manual_seed(1337)
    B, T, H, hd = 2, 6, 4, 8
    q = torch.randn(B, T, H, hd)

    # 构造单位模长的旋转因子：freqs_cis[t, i] = exp(i * t * theta_i)
    theta = 10000.0 ** (-torch.arange(hd // 2).float() / (hd // 2))
    t = torch.arange(T).float()
    freqs_cis = torch.complex(torch.cos(t[:, None] * theta[None, :]),
                              torch.sin(t[:, None] * theta[None, :]))  # (T, hd//2)

    result = exercise_3_apply_rope(q, freqs_cis)
    if result is None:
        _skip("题 3 未实现")

    assert result.shape == q.shape, \
        f"旋转后形状应与输入相同 {tuple(q.shape)}，得到 {tuple(result.shape)}"

    # 范数不变（正交变换）：逐 (b,t,h) 验证范数近似相等
    q_norm = q.norm(dim=-1)
    r_norm = result.norm(dim=-1)
    assert torch.allclose(q_norm, r_norm, atol=1e-4), \
        f"RoPE 旋转是正交变换，范数应不变，偏差 {((q_norm - r_norm).abs().max()).item():.2e}"

    # 与手动实数公式对照（取位置 t=3，验证一组角度）
    t_check = 3
    # 手动旋转：x0' = x0*cos - x1*sin, x1' = x0*sin + x1*cos（按每对 (2i, 2i+1)）
    x = q[0, t_check, 0]                       # (hd,)
    cos_t = freqs_cis[t_check].real            # (hd//2,)
    sin_t = freqs_cis[t_check].imag
    xr = x.view(hd // 2, 2)
    manual = torch.stack([
        xr[:, 0] * cos_t - xr[:, 1] * sin_t,
        xr[:, 0] * sin_t + xr[:, 1] * cos_t,
    ], dim=-1).reshape(hd)
    assert torch.allclose(result[0, t_check, 0], manual, atol=1e-5), \
        "位置 t=3 的第一组旋转应与手动公式一致"

    print(f"  shape={tuple(result.shape)}, 范数不变 ✓, 手动公式对照 ✓ ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 4：GQA repeat_kv
# ═══════════════════════════════════════════════════════════════════

def test_exercise_4_repeat_kv():
    """题 4：形状、复制顺序（第 i 组 == 第 i//n_rep 个原始头）、数据一致性"""
    torch.manual_seed(1337)
    B, T, hd = 3, 7, 16
    n_kv, n_rep = 4, 2   # 例如 8 个 Q 头 / 4 个 KV 头

    x = torch.randn(B, n_kv, T, hd)
    result = exercise_4_repeat_kv(x, n_rep)
    if result is None:
        _skip("题 4 未实现")

    out_n = n_kv * n_rep
    assert result.shape == (B, out_n, T, hd), \
        f"输出形状应为 ({B},{out_n},{T},{hd})，得到 {tuple(result.shape)}"

    # 关键不变量：第 i 个输出头 == 第 i // n_rep 个原始头
    for i in range(out_n):
        src = i // n_rep
        assert torch.allclose(result[:, i], x[:, src], atol=1e-6), \
            f"第 {i} 个输出头应等于第 {src} 个原始头（{src} = {i}//{n_rep}）"

    # n_rep=1 时应原样返回
    result1 = exercise_4_repeat_kv(x, 1)
    assert torch.allclose(result1, x, atol=1e-6), "n_rep=1 时应返回原张量"

    # n_rep=3（如 12 Q 头 / 4 KV 头）也应工作
    result3 = exercise_4_repeat_kv(x, 3)
    assert result3.shape == (B, n_kv * 3, T, hd), \
        f"n_rep=3 时形状应为 ({B},{n_kv*3},{T},{hd})，得到 {tuple(result3.shape)}"

    print(f"  ({B},{n_kv},{T},{hd}) --n_rep={n_rep}--> {tuple(result.shape)}, 复制顺序正确 ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 5：SwiGLU 前馈网络
# ═══════════════════════════════════════════════════════════════════

def test_exercise_5_swiglu():
    """题 5：形状、silu 正性、与手动公式一致、梯度回流"""
    torch.manual_seed(1337)
    dim, B, T = 32, 4, 8
    hidden = 4 * dim

    ffn = SwiGLU(dim)
    # 骨架状态（__init__ 未实现，投影层不存在）应优雅跳过
    if not hasattr(ffn, 'gate_proj'):
        _skip("题 5 未实现")

    # 检查三个投影层是否存在
    for attr in ['gate_proj', 'up_proj', 'down_proj']:
        assert hasattr(ffn, attr), f"SwiGLU 应有 {attr} 投影层"

    x = torch.randn(B, T, dim)
    out = ffn(x)
    if out is None:
        _skip("题 5 未实现")

    # 形状保持
    assert out.shape == (B, T, dim), \
        f"输出形状应为 (B,T,dim)，得到 {tuple(out.shape)}"

    # 与手动公式一致（gate/up/down 已知）
    with torch.no_grad():
        manual = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert torch.allclose(out, manual, atol=1e-5), "前向结果应与 gate/up/down 手动组合一致"

    # 梯度回流到所有投影
    o = ffn(x).sum()
    o.backward()
    for p in ffn.parameters():
        assert p.grad is not None, "所有投影层都应收到梯度"

    print(f"  shape={tuple(out.shape)}, 手动公式一致 ✓, 梯度回流 ✓ ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 6：DPO 损失
# ═══════════════════════════════════════════════════════════════════

def test_exercise_6_dpo_loss():
    """题 6：标量、chosen 更优时 loss 更小、方向正确、有限"""
    torch.manual_seed(1337)
    N = 8
    beta = 0.1

    # 构造两组 log-prob：一组 chosen 明显更优，一组 chosen 明显更差
    # 用随机但带结构的数据
    pi_c_good = torch.randn(N) * 0.5 + 1.0     # chosen 偏高
    pi_r_good = torch.randn(N) * 0.5 - 1.0     # rejected 偏低
    ref_c = torch.randn(N) * 0.3
    ref_r = torch.randn(N) * 0.3

    loss_good = exercise_6_dpo_loss(pi_c_good, pi_r_good, ref_c, ref_r, beta)
    if loss_good is None:
        _skip("题 6 未实现")

    # 标量、有限
    assert loss_good.ndim == 0, f"loss 应为标量，得到 {loss_good.ndim} 维"
    assert torch.isfinite(loss_good), "loss 应有限"

    # 反向数据：chosen 比 rejected 差 → loss 应更大
    loss_bad = exercise_6_dpo_loss(pi_r_good, pi_c_good, ref_c, ref_r, beta)
    assert loss_good < loss_bad, \
        f"chosen 更优时 loss 应更小（{loss_good.item():.4f} < {loss_bad.item():.4f}）"

    # 全相等时 loss ≈ -log sigmoid(0) = -log(0.5) = ln2 ≈ 0.6931
    z = torch.zeros(N)
    loss_eq = exercise_6_dpo_loss(z, z, z, z, beta)
    assert abs(loss_eq.item() - math.log(2)) < 1e-4, \
        f"chosen==rejected 时 loss 应≈ln2≈{math.log(2):.4f}，得到 {loss_eq.item():.4f}"

    print(f"  chosen 优: loss={loss_good.item():.4f}, chosen 差: loss={loss_bad.item():.4f}, "
          f"全等: loss={loss_eq.item():.4f}(≈ln2) ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 7（🌟 拓展）：KV Cache
# ═══════════════════════════════════════════════════════════════════

def test_exercise_7_kv_cache():
    """题 7：None 缓存原样返回、拼接形状、时间维顺序"""
    torch.manual_seed(1337)
    B, heads, hd = 2, 4, 16

    # 首次调用：past 为 None，应返回 k,v 本身
    k0 = torch.randn(B, heads, 3, hd)
    v0 = torch.randn(B, heads, 3, hd)
    res = exercise_7_kv_cache(k0, v0, None, None)
    if res is None or not isinstance(res, tuple) or len(res) != 2:
        _skip("题 7 未实现")
    pk, pv = res

    assert torch.equal(pk, k0) and torch.equal(pv, v0), "past=None 时应返回 (k, v)"

    # 第二次调用：新 token 追加到缓存末尾
    k1 = torch.randn(B, heads, 1, hd)
    v1 = torch.randn(B, heads, 1, hd)
    pk2, pv2 = exercise_7_kv_cache(k1, v1, pk, pv)

    assert pk2.shape == (B, heads, 4, hd), \
        f"拼接后时间维应为 3+1=4，得到 {tuple(pk2.shape)}"

    # 前 3 个时间步保留历史，最后 1 步是新 token
    assert torch.equal(pk2[..., :3, :], k0), "缓存前 3 步应保留历史 K"
    assert torch.equal(pk2[..., 3:, :], k1), "缓存最后 1 步应是新 K"
    assert torch.equal(pv2[..., 3:, :], v1), "V 缓存同理"

    print(f"  None->({tuple(pk.shape)}), 拼接->({tuple(pk2.shape)}), 顺序正确 ✅")


# ═══════════════════════════════════════════════════════════════════
#  主函数（独立运行时调用）
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 56)
    print("Part 7 作业测试：从零复现 minimind —— 现代 LLM 六大组件")
    print("=" * 56)

    tests = [
        test_exercise_1_bpe_encode,
        test_exercise_2_rmsnorm,
        test_exercise_3_apply_rope,
        test_exercise_4_repeat_kv,
        test_exercise_5_swiglu,
        test_exercise_6_dpo_loss,
        test_exercise_7_kv_cache,
    ]

    passed = skipped = failed = 0
    for t in tests:
        print(f"\n▶ {t.__name__}")
        try:
            t()
            passed += 1
        except BaseException as e:  # noqa: BLE001
            if _is_skip(e):
                skipped += 1
                print(f"  ⏭️ {t.__name__}: {e}")
            else:
                failed += 1
                print(f"  ❌ {t.__name__}: {e}")

    print()
    print("=" * 56)
    print(f"结果: {passed} 通过, {skipped} 跳过, {failed} 失败")
    if failed == 0:
        print("🎉 全部通过！" if passed > 0 else "⏭️ 都还没实现，按 TODO 提示完成吧～")
    else:
        print("还有题目需要完成哦～")
