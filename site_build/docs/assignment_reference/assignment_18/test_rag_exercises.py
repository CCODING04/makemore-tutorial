#!/usr/bin/env python3
"""Part 18 作业测试。独立运行：python test_rag_exercises.py；或 pytest。
只测性质不测精确值（除确定性纯函数的边界）。"""

import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_exercises import *  # noqa: F401,F403


def _strip_ws(s):
    return re.sub(r'\s+', '', s)


# ─────────────────────── 题 1：递归分块 ───────────────────────
def test_ex1_recursive_chunk():
    assert recursive_chunk("", 512, 64) == [], "空文本应返回 []"
    assert recursive_chunk("   \n\n \t ", 512, 64) == [], "纯空白应返回 []"

    one = recursive_chunk("hello world", 512, 64)
    assert one and len(one) == 1 and 'hello' in one[0] and 'world' in one[0]

    # 无分隔符超长文本：硬切路径
    text = 'A' * 2000
    chunks = recursive_chunk(text, 256, 32)
    assert chunks and all(len(c) <= 256 for c in chunks), \
        f"不变量① len<=size 被破坏: {[len(c) for c in chunks]}"
    for i in range(1, len(chunks)):
        assert chunks[i].startswith(chunks[i - 1][-32:]), \
            f"不变量② 第 {i} 块未以上一块尾部 overlap 字符开头"
    joined = chunks[0] + ''.join(c[32:] for c in chunks[1:])
    assert _strip_ws(joined) == _strip_ws(text), "不变量③ 丢字符"

    # 真实中英混排 + 自定义参数：递归下钻路径
    doc = ("# 标题\n\n第一段讲分块策略。句子二号讲重叠。\n\n" +
           "第二段有英文 BM25 and RRF fusion details. " * 12 + "\n" +
           "第三段。短句。又一句。" * 20)
    for size, ov in [(512, 64), (128, 16)]:
        cs = recursive_chunk(doc, size, ov)
        assert cs and all(len(c) <= size for c in cs), f"size={size} 违反不变量①"
        for i in range(1, len(cs)):
            assert cs[i].startswith(cs[i - 1][-ov:]), \
                f"size={size} 第 {i} 块重叠桥断裂"
        joined = cs[0] + ''.join(c[ov:] for c in cs[1:])
        assert _strip_ws(joined) == _strip_ws(doc), \
            f"size={size} 不变量③：拼接后丢/多了字符"


# ─────────────────────── 题 2：BM25 ───────────────────────
def test_ex2_bm25():
    chunks = ["apple pear apple plum", "apple pear plum quince",
              "quantum apple plum pear"]
    s = bm25_scores("apple", chunks)
    assert s is not None, "bm25_scores 未实现"
    assert len(s) == len(chunks) and all(x >= 0 for x in s), \
        "返回与 chunks 等长的非负分数"
    assert bm25_scores("zebra", chunks) == [0.0, 0.0, 0.0], "无命中词应为全 0"

    # 性质①：稀有词（df=1）的贡献 > 到处都有的词（df=N）——IDF 判别力
    rare = max(bm25_scores("quantum", chunks))
    common = max(bm25_scores("apple", chunks))
    assert rare > common * 3, \
        f"IDF 性质：稀有词最高分({rare:.3f}) 应远超常见词({common:.3f})"

    # 性质②：TF 单调——同一查询词出现更多的 chunk 得分更高（长度相近时）
    chunks2 = ["cat cat cat dog fish", "cat dog fish bird"]
    s2 = bm25_scores("cat", chunks2)
    assert s2[0] > s2[1], f"TF 单调性被长度归一破坏: {s2}"


# ─────────────────────── 题 3：RRF ───────────────────────
def test_ex3_rrf():
    assert rrf_fuse([], []) == [], "空输入应返回 []"
    assert rrf_fuse([3, 1], []) == [3, 1], "单榜非空应保留其顺序"

    # 基础：双榜都有的元素排最前
    r = rrf_fuse(['a', 'b'], ['b', 'c'])
    assert r is not None and r[0] == 'b' and sorted(r) == ['a', 'b', 'c']

    # k→∞：退化为"入选榜单数优先、名次和次之"的计数排序
    big = 10 ** 9
    r = rrf_fuse([3, 1, 2], [2, 4], k=big)
    assert sorted(r) == [1, 2, 3, 4]
    assert r[0] == 2, "双榜入选者（2）应第一"
    assert r[1] == 3, "单榜 rank1（3，名次和 1）应第二"
    assert r.index(4) > r.index(1) and r.index(4) > r.index(3), \
        "单榜 rank2（4，名次和 2）应排在 rank1 元素之后"

    # 并列名次：完全对称的输入允许任意稳定顺序，但必须是原元素的排列
    r = rrf_fuse(['x', 'y'], ['y', 'x'])
    assert sorted(r) == ['x', 'y'] and r[0] in ('x', 'y')


# ─────────────────────── 题 4：faithfulness ───────────────────────
def test_ex4_faithfulness():
    ctxs = ["GRPO 出自 DeepSeekMath 论文。", "verl 框架支持 GRPO 训练。"]
    ans = "GRPO 出自 DeepSeekMath 论文。课程用 verl 跑实战。"

    yes_judge = lambda p: 'yes'          # noqa: E731
    no_judge = lambda p: 'no'            # noqa: E731
    unsure_judge = lambda p: 'unsure'    # noqa: E731
    f = faithfulness(ans, ctxs, yes_judge)
    assert f is not None, "faithfulness 未实现"
    assert abs(f - 1.0) < 1e-9, "全 yes 裁判应得 1.0"
    assert faithfulness(ans, ctxs, no_judge) == 0.0, "全 no 裁判应得 0.0"
    assert faithfulness(ans, ctxs, unsure_judge) == 0.0, \
        "文档化口径：unsure 一律按不支持计 0"

    # 逐条判定：只对含标记词 ALPHA 的 claim 说 yes → 3 句中 1 句支持 = 1/3
    ans3 = "这是第一句话。这句包含ALPHA标记。这是第三句话。"
    marker_judge = lambda p: 'yes' if 'ALPHA' in p else 'no'  # noqa: E731
    assert abs(faithfulness(ans3, ctxs, marker_judge) - 1 / 3) < 1e-9, \
        "应逐条调用 judge（prompt 含当前 claim 文本）并按比例计分"

    assert faithfulness("", ctxs, yes_judge) is None, "空答案（无 claims）返回 None"


# ─────────────────────── 🌟 题 5：权重网格搜索 ───────────────────────
def _curve_lookup(curve, w):
    """在曲线里找最接近 w 的点对应的 recall（容忍网格端点的浮点误差）。"""
    best = min(curve, key=lambda item: abs(item[0] - w))
    return best[1]


def test_ex5_stretch():
    """🌟 Stretch：未实现（返回 None）时优雅 SKIP，不判 FAIL。"""
    # 构造：2 个 query × 10 个 chunk，|rel| = k = 5（相关 chunk 必须全部进 top5
    # 才有 recall=1.0，避免"掺一点 dense 就饱和"）。dense 完美、sparse 专捧干扰项
    # → 最优 w 落在 dense 侧（≥0.5）；角色互换后最优 w 落在 sparse 侧（≤0.5）
    rel = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    dense = [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    sparse = [[0.0, 0.0, 0.0, 0.0, 0.0, 9.9, 7.0, 5.0, 3.0, 1.0],
              [9.9, 7.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    result = hybrid_weight_sweep(dense, sparse, rel, k=5)
    if result is None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip("🌟 Stretch 未实现（返回 None）——实现后此测试自动生效")
        return "SKIP"
    best_w, curve = result
    ws = [w for w, _ in curve] if curve else []
    rcs = [r for _, r in curve] if curve else []
    assert len(curve) >= 5 and len(set(ws)) == len(ws), "曲线应有 ≥5 个不同 w 的点"
    assert abs(_curve_lookup(curve, 1.0) - 1.0) < 1e-9, "w≈1.0（纯 dense）应 recall=1.0"
    assert _curve_lookup(curve, 0.0) < 0.5, "w≈0.0（纯 sparse）应 recall 很低"
    assert best_w >= 0.5 - 1e-9, f"dense 完美/sparse 捣乱时最优 w 应在 dense 侧（≥0.5），got {best_w}"
    assert max(rcs) == _curve_lookup(curve, best_w), "best_w 的 recall 应为曲线最大值"
    # 交换角色：dense 捣乱、sparse 完美 → 最优 w 落在 sparse 侧（≤0.5）
    result2 = hybrid_weight_sweep(sparse, dense, rel, k=5)
    if result2 is not None:
        best_w2, _ = result2
        assert best_w2 <= 0.5 + 1e-9, f"角色互换后最优 w 应在 sparse 侧（≤0.5），got {best_w2}"


_TESTS = [("题1 递归分块", test_ex1_recursive_chunk), ("题2 手写 BM25", test_ex2_bm25),
          ("题3 RRF 融合", test_ex3_rrf), ("题4 faithfulness", test_ex4_faithfulness),
          ("题5 🌟 Stretch 权重网格搜索", test_ex5_stretch)]


def main():
    p = f = s = 0
    for name, fn in _TESTS:
        try:
            r = fn()
            if r == "SKIP":
                print(f"  ⏭️  {name} — SKIP（未实现，不扣分）"); s += 1
            else:
                print(f"  ✅ {name}"); p += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {e}"); f += 1
    msg = f"\n  通过: {p}/{p + f}" + (f"（另 {s} 项 SKIP ⏭️）" if s else "")
    print(msg + ("  🎉" if f == 0 else "  💡 先实现 rag_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
