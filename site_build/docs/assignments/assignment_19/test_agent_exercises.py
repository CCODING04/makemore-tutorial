#!/usr/bin/env python3
"""Assignment 19 测试。独立运行：python test_agent_exercises.py；或 pytest。
题 5 为 🌟 stretch：未实现（返回 None）时优雅 SKIP ⏭️，不算失败。"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_exercises import *  # noqa: F401,F403


class SkipTest(Exception):
    """🌟 stretch 题未实现（返回 None）时的优雅跳过信号（独立运行模式）。"""


def _skip(msg):
    """优雅跳过：pytest 环境用 pytest.skip；独立运行抛 SkipTest 由 main 捕获。"""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise SkipTest(msg)


# ── 题 1 测试用具：一个真函数 + 一个描述 dict ──
def _sample_fn(order_id, reason="none given"):
    """Refund an order by id. reason is optional."""
    return order_id


_DESC = {"name": "get_weather",
         "description": "Query current weather of a city.",
         "params": {"city": {"type": "string", "description": "City name"},
                    "units": {"type": "bogus", "description": "c or f"}}}


def test_ex1_tool_spec():
    assert tool_spec is not None, "tool_spec 未实现"
    # 路径 A：函数输入
    spec = tool_spec(_sample_fn)
    assert spec is not None, "未实现（返回 None）"
    fn = spec["function"]
    assert spec["type"] == "function"
    assert fn["name"] == "_sample_fn" and fn["name"], "name 应取函数名且非空"
    assert "Refund" in fn["description"], "description 应取 docstring 首行"
    assert fn["parameters"]["type"] == "object"
    assert set(fn["parameters"]["required"]) == {"order_id"}, \
        "required 应只含无默认值参数（order_id），不含 reason"
    assert set(fn["parameters"]["properties"]) == {"order_id", "reason"}, \
        "两个参数都应出现在 properties"
    # 路径 B：描述 dict 输入 + 非法 type 回退
    spec2 = tool_spec(_DESC)
    fn2 = spec2["function"]
    assert fn2["name"] == "get_weather"
    assert set(fn2["parameters"]["required"]) == {"city", "units"}, \
        "dict 路径所有参数都进 required"
    assert fn2["parameters"]["properties"]["units"]["type"] == "string", \
        "非法 type 'bogus' 应回退为 'string'"
    for p in fn2["parameters"]["properties"].values():
        assert p["type"] in {"string", "integer", "number", "boolean", "array", "object"}, \
            "参数 type 必须是合法 JSON schema 类型"


def test_ex2_parse_tool_calls():
    assert parse_tool_calls is not None, "parse_tool_calls 未实现"
    # ① 合法单调用
    single = ('Hmm... <tool_call>{"name": "calculator", '
              '"arguments": {"expression": "1+1"}}</tool_call> done.')
    calls = parse_tool_calls(single)
    assert calls is not None and calls == [{"name": "calculator",
                                            "arguments": {"expression": "1+1"}}], \
        "未实现（返回 None）或单调用解析错误"
    # ② 合法多调用（按出现顺序）
    multi = ('<tool_call>{"name": "file_read", "arguments": {"path": "/tmp/a.txt"}}</tool_call>'
             ' between '
             '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>')
    calls = parse_tool_calls(multi)
    assert len(calls) == 2 and calls[0]["name"] == "file_read" and calls[1]["name"] == "bash", \
        "多调用应按出现顺序全部解析出"
    # ③ 畸形 JSON：截断 → []；尾逗号 → 可修复
    assert parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expr') == [], \
        "截断的畸形 JSON 应返回 []（不抛异常）"
    fixed = parse_tool_calls('<tool_call>{"name": "a", "arguments": {"x": 1,}}</tool_call>')
    assert fixed == [{"name": "a", "arguments": {"x": 1}}], "尾逗号应被修复解析出来"
    # ④ 无调用
    assert parse_tool_calls("The answer is 42.") == [], "无调用应返回 []（不是 None）"


def test_ex3_should_stop():
    assert should_stop is not None, "should_stop 未实现"
    base = {"turns": 1, "max_turns": 4, "last_has_calls": True, "tool_history": ["bash"]}
    r = should_stop(dict(base))
    assert r is not None and r == (False, None), \
        "未实现（返回 None）或正常进行中应返回 (False, None)"
    # ① max_turns
    assert should_stop({**base, "turns": 4, "max_turns": 4}) == (True, "max_turns")
    # ② loop_detected：末尾 3 个同名（优先级高于 no_tool_calls）
    assert should_stop({**base, "last_has_calls": False,
                        "tool_history": ["file_read", "file_read", "file_read", "file_read"]}) \
        == (True, "loop_detected"), "末尾 3 个同工具应判 loop（优先级高于 no_tool_calls）"
    # 同工具但没连着 3 次 → 不算循环
    assert should_stop({**base, "tool_history": ["file_read", "bash", "file_read"]}) == (False, None)
    # ③ no_tool_calls
    assert should_stop({**base, "last_has_calls": False}) == (True, "no_tool_calls")


def test_ex4_pass_at_1():
    assert pass_at_1 is not None, "pass_at_1 未实现"
    r = pass_at_1([True, True, True, True])
    assert r is not None and abs(r - 1.0) < 1e-9, "全过 → 1.0"
    assert pass_at_1([False]) == 0.0, "全挂 → 0.0"
    assert abs(pass_at_1([True, True, False]) - 2 / 3) < 1e-9
    assert pass_at_1([]) is None, "空列表 → None（区分'全挂'与'没跑'）"
    # 数学性质：分组加权平均 == 整体均值（脚本 03 的逐任务/总体汇总依赖此性质）
    runs = [True, False, True, True, False, False, True]
    g1, g2 = runs[:3], runs[3:]
    weighted = (len(g1) * pass_at_1(g1) + len(g2) * pass_at_1(g2)) / len(runs)
    assert abs(weighted - pass_at_1(runs)) < 1e-9
    assert pass_at_1(runs) == sum(runs) / len(runs)


def test_ex5_mini_mcp_call():
    class MockTransport:
        """脚本化 transport：记录所有发出的消息，按序回放响应。"""

        def __init__(self, final_response):
            self.sent = []        # 全部消息（请求 + 通知）
            self._responses = [
                {"jsonrpc": "2.0", "id": 1, "result":
                 {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}},
                final_response,
            ]

        def send(self, request):
            self.sent.append(request)
            return self._responses.pop(0)

        def notify(self, msg):
            self.sent.append(msg)

    # 🌟 stretch：骨架返回 None → 优雅 SKIP（不算失败）。
    # 直接用正常剧本探测：正确实现返回第三步 result；未实现返回 None。
    ok = MockTransport({"jsonrpc": "2.0", "id": 2,
                        "result": {"content": [{"type": "text", "text": "echo: hi"}]}})
    try:
        r = mini_mcp_call(ok, "tools/call", {"name": "echo", "arguments": {"text": "hi"}})
    except Exception as e:                       # 实现了但行为错误（如发错时序）→ 算失败
        raise AssertionError(f"实现存在但运行出错: {type(e).__name__}: {e}")
    if r is None:
        _skip("🌟 未实现（mini_mcp_call 返回 None），先完成题 1-4 再挑战")
    assert r["content"][0]["text"] == "echo: hi", "应返回第三步的 result"
    # 时序断言：2 请求 + 1 通知；通知无 id；两请求 id 互不相同
    requests = [m for m in ok.sent if "id" in m]
    notifications = [m for m in ok.sent if "id" not in m]
    assert len(requests) == 2 and len(notifications) == 1, \
        "应恰好 2 条请求（initialize + method）与 1 条通知（initialized）"
    assert requests[0]["method"] == "initialize" and requests[1]["method"] == "tools/call"
    assert requests[0]["id"] != requests[1]["id"], "两次请求的 id 必须互不相同"
    assert notifications[0]["method"] == "notifications/initialized"
    assert requests[1]["params"] == {"name": "echo", "arguments": {"text": "hi"}}
    # error 响应 → None（不抛异常）
    bad = MockTransport({"jsonrpc": "2.0", "id": 2,
                         "error": {"code": -32601, "message": "no such method"}})
    assert mini_mcp_call(bad, "resources/list") is None, "error 响应应返回 None"


_TESTS = [("题1 tool_spec", test_ex1_tool_spec), ("题2 parse_tool_calls", test_ex2_parse_tool_calls),
          ("题3 should_stop", test_ex3_should_stop), ("题4 pass_at_1", test_ex4_pass_at_1),
          ("题5 🌟 mini_mcp_call", test_ex5_mini_mcp_call)]


def main():
    p = f = s = 0
    for name, fn in _TESTS:
        try:
            fn(); print(f"  ✅ {name}"); p += 1
        except SkipTest as e:
            print(f"  ⏭️ {name} — SKIP: {e}"); s += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}"); f += 1
        except Exception as e:
            print(f"  ❌ {name} — ERROR: {type(e).__name__}: {e}"); f += 1
    summary = f"\n  通过: {p}/{p + f}"
    if s:
        summary += f"（另 {s} 题 SKIP ⏭️）"
    print(summary + ("  🎉" if f == 0 else "  💡 先实现 agent_exercises.py"))
    return 0 if f == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
