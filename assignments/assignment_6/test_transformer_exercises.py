#!/usr/bin/env python3
"""
Part 6 作业测试：Transformer/GPT

两种运行方式：
  1. 独立运行：  python test_transformer_exercises.py
  2. pytest：    pytest test_transformer_exercises.py

未实现的题目（函数返回 None）会优雅跳过而非报错。
"""

import os
import sys
import math
import torch

# 强制 stdout 使用 UTF-8，避免 Windows 控制台按 GBK 输出导致中文乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 小模型在 CPU 上多线程调度开销大于收益，固定单线程使运行更快更稳定
torch.set_num_threads(1)

# 确保能找到作业文件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from transformer_exercises import *  # noqa: F401,F403

# 数据文件路径
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'input.txt')


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


def _load_text():
    with open(_DATA_PATH, 'r', encoding='utf-8') as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════
#  题 1：Tokenize 与数据划分
# ═══════════════════════════════════════════════════════════════════

def test_exercise_1_tokenize():
    """题 1：vocab_size、往返一致性、shape/dtype、90/10 划分"""
    torch.manual_seed(1337)
    text = _load_text()
    result = exercise_1_tokenize(text)
    if result is None:
        _skip("题 1 未实现")

    vocab_size = result['vocab_size']
    data = result['data']
    train_data = result['train_data']
    val_data = result['val_data']
    encode = result['encode']
    decode = result['decode']

    # tiny Shakespeare 有 65 个唯一字符
    assert vocab_size == 65, f"vocab_size 应为 65，得到 {vocab_size}"

    # encode/decode 往返一致
    for s in ["Hello, world!\n", "To be, or not to be", "hi there"]:
        assert decode(encode(s)) == s, f"往返一致失败: {s!r}"

    # data 类型与维度
    assert data.dtype == torch.long, f"data dtype 应为 long，得到 {data.dtype}"
    assert data.ndim == 1, f"data 应为 1D，得到 {data.ndim} 维"

    # 索引范围合法
    assert data.min().item() >= 0 and data.max().item() < vocab_size, \
        "data 的取值应落在 [0, vocab_size) 内"

    # 90/10 划分：train + val 覆盖全部，比例 ≈ 0.9，且划分连续
    assert len(train_data) + len(val_data) == len(data), "train + val 应等于 data"
    ratio = len(train_data) / len(data)
    assert abs(ratio - 0.9) < 0.01, f"train 比例应≈0.9，得到 {ratio:.4f}"
    n = len(train_data)
    assert torch.equal(train_data, data[:n]), "train_data 应为 data 前 n 个元素"
    assert torch.equal(val_data, data[n:]), "val_data 应为 data 后 len-n 个元素"

    print(f"  vocab_size={vocab_size}, data={tuple(data.shape)}, "
          f"train/val={len(train_data)}/{len(val_data)} ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 2：get_batch
# ═══════════════════════════════════════════════════════════════════

def test_exercise_2_get_batch():
    """题 2：shape、dtype、y=x 后移一位、chunk 是 data 连续子串、种子可复现"""
    torch.manual_seed(1337)
    text = _load_text()

    # 优先复用题 1 的 data；若题 1 未实现则内联 tokenizer（保证测试独立）
    res = exercise_1_tokenize(text)
    if res is None:
        chars = sorted(list(set(text)))
        stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    else:
        data = res['data']

    B, T = 4, 8
    result = exercise_2_get_batch(data, block_size=T, batch_size=B, seed=1337)
    if result is None:
        _skip("题 2 未实现")
    x, y = result

    assert x.shape == (B, T) and y.shape == (B, T), \
        f"shape 错误: x={tuple(x.shape)}, y={tuple(y.shape)}，应为 ({B},{T})"
    assert x.dtype == torch.long and y.dtype == torch.long, \
        f"dtype 应为 long，得到 x={x.dtype}, y={y.dtype}"

    # 关键不变量：y 是 x 后移一位
    assert torch.equal(y[:, :-1], x[:, 1:]), "y 应为 x 后移一位"

    # 每个 chunk 必须是 data 的连续子串（不能随机取散点）
    windows = data.unfold(0, T, 1)
    for i in range(B):
        assert (windows == x[i]).all(dim=1).any(), \
            f"第 {i} 行不是 data 的连续子串"

    # 相同种子可复现
    x2, y2 = exercise_2_get_batch(data, block_size=T, batch_size=B, seed=1337)
    assert torch.equal(x, x2) and torch.equal(y, y2), "相同种子应可复现"

    print(f"  X/Y shape={tuple(x.shape)}, 偏移一致, 4 个 chunk 均来自 data ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 3：Bigram + 交叉熵 + generate
# ═══════════════════════════════════════════════════════════════════

def test_exercise_3_bigram_loss():
    """题 3：logits/loss 的 shape 与 dtype、初始 loss 接近 ln(vocab_size)"""
    torch.manual_seed(1337)
    model = exercise_3_bigram_model(65)
    if model is None:
        _skip("题 3 未实现")

    B, T = 32, 8
    xb = torch.randint(0, 65, (B, T))
    yb = torch.randint(0, 65, (B, T))
    logits, loss = model(xb, yb)

    # 与教程/脚本一致：有 targets 时 forward 返回的是为交叉熵 reshape 后的 logits (B*T, vocab)
    assert logits.shape == (B * T, 65), \
        f"有 targets 时 logits shape 应为 (B*T,vocab)=({B*T},65)，得到 {tuple(logits.shape)}"
    assert loss is not None, "有 targets 时应返回 loss"
    assert loss.ndim == 0, f"loss 应为标量，得到 {loss.ndim} 维"
    assert torch.isfinite(loss), "loss 应有限"

    # 初始 loss 应接近 ln(vocab_size)≈4.17（随机初始化允许合理浮动）
    ln_v = math.log(65)
    assert abs(loss.item() - ln_v) < 1.5, \
        f"初始 loss 应接近 ln65≈{ln_v:.2f}，得到 {loss.item():.4f}"

    # targets=None 时 loss 应为 None，且 logits 保持 (B, T, vocab)
    logits_none, loss_none = model(xb)
    assert loss_none is None, "targets=None 时 loss 应为 None"
    assert logits_none.shape == (B, T, 65), \
        f"无 targets 时 logits shape 应保持 (B,T,vocab)=({B},{T},65)，得到 {tuple(logits_none.shape)}"

    print(f"  初始 loss={loss.item():.4f}（ln65≈{ln_v:.2f}），有targets时logits={tuple(logits.shape)} ✅")


def test_exercise_3_generate():
    """题 3：generate 的 shape、dtype、取值区间"""
    torch.manual_seed(1337)
    model = exercise_3_bigram_model(65)
    if model is None:
        _skip("题 3 未实现")

    ctx = torch.zeros((1, 1), dtype=torch.long)
    n_new = 20
    out = model.generate(ctx, max_new_tokens=n_new)

    assert out.shape == (1, 1 + n_new), \
        f"generate 输出 shape 应为 (B, T+{n_new})，得到 {tuple(out.shape)}"
    assert out.dtype == torch.long, f"generate 输出 dtype 应为 long，得到 {out.dtype}"
    assert out.min().item() >= 0 and out.max().item() < 65, "生成 token 应在 [0, vocab) 内"

    # batch>1 也工作
    out2 = model.generate(torch.zeros((3, 5), dtype=torch.long), max_new_tokens=10)
    assert out2.shape == (3, 15), f"多 batch 生成 shape 应为 (3,15)，得到 {tuple(out2.shape)}"

    print(f"  generate (1,1)->{tuple(out.shape)}, 多batch (3,5)->{tuple(out2.shape)} ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 4：单头 Self-Attention
# ═══════════════════════════════════════════════════════════════════

def test_exercise_4_scaling():
    """题 4(a)：scaled attention 的方差推导（note 6）"""
    torch.manual_seed(1337)
    hs, B, T = 16, 8, 16
    q = torch.randn(B, T, hs)   # unit gaussian
    k = torch.randn(B, T, hs)

    raw = q @ k.transpose(-2, -1)            # 未缩放：方差 ≈ head_size
    scaled = scaled_dot_product_affinity(q, k)
    if scaled is None:
        _skip("题 4 scaled_dot_product_affinity 未实现")

    assert scaled.shape == (B, T, T), \
        f"亲和力 shape 应为 (B,T,T)，得到 {tuple(scaled.shape)}"

    # 未缩放时 q@k^T 的标准差 ≈ sqrt(head_size)=4
    assert 2.0 < raw.std().item() < 6.0, \
        f"未缩放的 q@k^T 标准差应≈sqrt(head_size)，得到 {raw.std().item():.3f}"
    # 缩放后方差 ≈ 1
    assert 0.5 < scaled.std().item() < 1.5, \
        f"除以 sqrt(head_size) 后 wei 标准差应≈1，得到 {scaled.std().item():.3f}"

    print(f"  q@k^T std={raw.std().item():.3f}（≈sqrt(16)=4）→ 缩放后 std="
          f"{scaled.std().item():.3f}（≈1）✅")


def test_exercise_4_head():
    """题 4(b)：输出 shape、wei 因果遮罩（严格下三角）、行和为 1"""
    torch.manual_seed(1337)
    head = exercise_4_head(head_size=16, n_embd=32, block_size=8)
    if head is None:
        _skip("题 4 SelfAttentionHead 未实现")

    B, T = 4, 8
    x = torch.randn(B, T, 32)
    out = head(x)

    assert out.shape == (B, T, 16), \
        f"输出 shape 应为 (B,T,head_size)=(4,8,16)，得到 {tuple(out.shape)}"

    assert hasattr(head, 'wei'), "forward 后应有 self.wei（保存注意力权重）"
    wei = head.wei
    assert wei.shape == (B, T, T), \
        f"self.wei shape 应为 (B,T,T)，得到 {tuple(wei.shape)}"

    # softmax 后每行和为 1
    assert torch.allclose(wei.sum(dim=-1), torch.ones(B, T), atol=1e-5), \
        "wei 每行 softmax 后应和为 1"

    # 因果遮罩：严格上三角为 0（未来不能看向过去）
    triu = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert torch.all(wei[..., triu] == 0), "wei 的严格上三角应为 0"

    # 概率值在 [0, 1]
    assert wei.min().item() >= 0 and wei.max().item() <= 1, "wei 应为概率值"

    print(f"  输出 shape={tuple(out.shape)}, wei 行和=1, 严格下三角遮罩 ✅")


# ═══════════════════════════════════════════════════════════════════
#  题 5（🌟 拓展）：完整 Transformer Block
# ═══════════════════════════════════════════════════════════════════

def test_exercise_5_transformer_block():
    """题 5（🌟 拓展）：shape 保持、残差恒等性、梯度回流"""
    torch.manual_seed(1337)
    block = exercise_5_transformer_block(n_embd=32, n_head=4, block_size=8)
    if block is None:
        _skip("拓展题未实现")

    block.eval()
    x = torch.randn(4, 8, 32)
    out = block(x)

    assert out.shape == (4, 8, 32), \
        f"Block 输出 shape 应保持 (B,T,n_embd)，得到 {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Block 输出应有限"

    # 组件齐全：多头注意力 / 前馈 / 两层 LayerNorm
    for attr in ['sa', 'ffwd', 'ln1', 'ln2']:
        assert hasattr(block, attr), f"Block 应有 {attr} 属性（MultiHead/FeedForward/LayerNorm）"

    # 残差连接：把子模块参数清零后，block 应近似恒等（x + 0 + 0）
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
        out_zero = block(x)
    assert torch.allclose(out_zero, x, atol=1e-5), \
        "残差结构应保证：参数清零时 block(x)≈x"

    # 梯度能流回所有参数
    block2 = exercise_5_transformer_block(n_embd=32, n_head=4, block_size=8)
    block2.train()
    o = block2(torch.randn(2, 8, 32))
    o.sum().backward()
    for p in block2.parameters():
        assert p.grad is not None, "所有参数都应收到梯度"
        assert torch.isfinite(p.grad).all(), "所有梯度应有限"

    print(f"  Block 输出={tuple(out.shape)}, 残差恒等 ✓, 梯度回流 ✓ ✅")


# ═══════════════════════════════════════════════════════════════════
#  主函数（独立运行时调用）
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 56)
    print("Part 6 作业测试：Transformer/GPT")
    print("=" * 56)

    tests = [
        test_exercise_1_tokenize,
        test_exercise_2_get_batch,
        test_exercise_3_bigram_loss,
        test_exercise_3_generate,
        test_exercise_4_scaling,
        test_exercise_4_head,
        test_exercise_5_transformer_block,
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
