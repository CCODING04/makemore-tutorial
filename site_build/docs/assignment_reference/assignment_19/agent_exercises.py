#!/usr/bin/env python3
"""Assignment 19 参考答案：Agent 与 Function Calling。纯 CPU 可验证。"""

import inspect
import json
import re

_LEGAL_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def tool_spec(func_or_desc):
    """函数/描述 dict → OpenAI tools schema 元素（参考实现）。"""
    if callable(func_or_desc):
        # 路径 A：真函数——inspect 读签名，docstring 首行做 description
        name = func_or_desc.__name__
        doc = inspect.getdoc(func_or_desc)
        description = doc.splitlines()[0].strip() if doc else f"Call the function {name}."
        properties, required = {}, []
        sig = inspect.signature(func_or_desc)
        for pname, param in sig.parameters.items():
            # 注解映射：int→integer，float→number，其余 str 兜底
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
            ptype = {int: "integer", float: "number", bool: "boolean", str: "string"}.get(ann, "string")
            properties[pname] = {"type": ptype, "description": f"Argument {pname} of {name}."}
            if param.default is inspect.Parameter.empty:      # 无默认值 → 必填
                required.append(pname)
    elif isinstance(func_or_desc, dict):
        # 路径 B：描述 dict——全部参数进 required，非法 type 回退 "string"
        name = func_or_desc["name"]
        description = func_or_desc.get("description", f"Call {name}.")
        properties, required = {}, []
        for pname, meta in func_or_desc.get("params", {}).items():
            ptype = meta.get("type", "string")
            if ptype not in _LEGAL_TYPES:
                ptype = "string"
            properties[pname] = {"type": ptype,
                                 "description": meta.get("description", "")}
            required.append(pname)
    else:
        raise TypeError("func_or_desc must be a callable or a dict")

    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object",
                                        "properties": properties,
                                        "required": required}}}


def parse_tool_calls(text):
    """解析全部 <tool_call> 块，畸形绝不抛异常（参考实现）。"""
    if not isinstance(text, str):
        return []
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        raw = m.group(1)
        parsed = None
        for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):   # 原样 → 修尾逗号
            try:
                parsed = json.loads(candidate)
                break
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(parsed, dict) and "name" in parsed:
            calls.append({"name": parsed["name"],
                          "arguments": parsed.get("arguments") or {}})
    return calls


def should_stop(state):
    """终止状态机：max_turns > loop_detected > no_tool_calls（参考实现）。"""
    if state["turns"] >= state["max_turns"]:
        return (True, "max_turns")
    hist = state.get("tool_history", [])
    if len(hist) >= 3 and len(set(hist[-3:])) == 1:
        return (True, "loop_detected")
    if not state.get("last_has_calls", False):
        return (True, "no_tool_calls")
    return (False, None)


def pass_at_1(runs):
    """pass^1 = 均值；空列表 → None（参考实现）。"""
    if not runs:
        return None
    return sum(runs) / len(runs)


def mini_mcp_call(transport, method, params=None):
    """JSON-RPC 三步握手 + 调用（参考实现）。"""
    next_id = [1]                                    # 闭包计数器：①③ 的 id 互不相同

    def _send(m, p):
        req = {"jsonrpc": "2.0", "id": next_id[0], "method": m, "params": p}
        next_id[0] += 1
        return transport.send(req)

    init = _send("initialize", {"protocolVersion": "2024-11-05",
                                "clientInfo": {"name": "assignment-19", "version": "1.0"},
                                "capabilities": {}})
    if "error" in init:                              # 握手失败 → None（不抛异常）
        return None
    transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})
    resp = _send(method, params or {})
    if "error" in resp:
        return None
    return resp.get("result")
