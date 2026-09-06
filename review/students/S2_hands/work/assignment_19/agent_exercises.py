#!/usr/bin/env python3
"""Assignment 19 作业骨架：Agent 与 Function Calling。在此文件实现五道题，用同目录
test_agent_exercises.py 验证（python 直跑或 pytest 均可）。全部纯 CPU、零模型依赖；
题 5 为 🌟 弹性题——不实现保持 return None 即可，测试会优雅 SKIP ⏭️。
题目说明与验收清单见同目录 assignment.md。"""

import inspect
import json
import re


# ── 题 1：tool_spec —— 函数/描述 → OpenAI tools JSON schema（25 分）──
_LEGAL_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def tool_spec(func_or_desc):
    """把 Python 函数或描述 dict 转成一个 OpenAI tools schema 元素。

    Args:
        func_or_desc: callable —— 用 inspect 读参数名（有默认值=可选，无=required），
                      docstring 首行做 description；
                      或 dict —— {"name": str, "description": str,
                                   "params": {参数名: {"type": str, "description": str}}}
                      （params 里所有参数默认 required；type 非法时回退 "string"）
    Returns:
        dict: {"type": "function", "function": {"name", "description",
               "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
        required 包含全部无默认值参数（顺序不限，测试用集合比较）；无必填参数时为空列表。
    """
    # 路径 A：callable → inspect 读签名与 docstring 首行
    if callable(func_or_desc):
        sig = inspect.signature(func_or_desc)
        doc = (inspect.getdoc(func_or_desc) or "").strip().splitlines()
        description = doc[0].strip() if doc else "No description provided."
        properties, required = {}, []
        for pname, p in sig.parameters.items():
            # 从 Python 注解推 type，推不出回退 "string"
            ann = p.annotation
            ptype = ann if isinstance(ann, str) and ann in _LEGAL_TYPES else "string"
            properties[pname] = {"type": ptype, "description": f"Parameter {pname}"}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "function", "function": {
            "name": func_or_desc.__name__, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required}}}
    # 路径 B：描述 dict → 直接读字段（全部参数 required，非法 type 回退 string）
    properties = {}
    for pname, pinfo in func_or_desc.get("params", {}).items():
        ptype = pinfo.get("type", "string")
        if ptype not in _LEGAL_TYPES:
            ptype = "string"
        properties[pname] = {"type": ptype,
                             "description": pinfo.get("description", "")}
    return {"type": "function", "function": {
        "name": func_or_desc["name"],
        "description": func_or_desc.get("description", ""),
        "parameters": {"type": "object", "properties": properties,
                       "required": list(func_or_desc.get("params", {}).keys())}}}


# ── 题 2：parse_tool_calls —— 解析模型输出（25 分，与课程脚本 01 同名同语义）──
def parse_tool_calls(text):
    """解析文本中全部 <tool_call>{"name": ..., "arguments": {...}}</tool_call>。

    兜底策略（解析失败绝不抛异常）：
      - 每个 JSON 块先原样 json.loads，失败再修尾逗号（r",\\s*([}\\]])" → r"\\1"）
      - 截断等不可修复的畸形 → 跳过该块
      - 全部失败 / 无调用 → 返回 []
    Returns:
        list[dict]: [{"name": str, "arguments": dict}, ...]（按出现顺序）
    """
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S):
        candidate = m.group(1)
        parsed = None
        # 宽容解析：先原样，再修尾逗号；都失败则跳过该块（绝不抛异常）
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
                break
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(parsed, dict) and "name" in parsed:
            calls.append({"name": parsed["name"],
                          "arguments": parsed.get("arguments", {}) or {}})
    return calls


# ── 题 3：should_stop —— agent 终止状态机（20 分）──
def should_stop(state):
    """判定 agent loop 是否终止。优先级：max_turns > loop_detected > no_tool_calls。

    Args:
        state: {"turns": int,               # 已用轮数
                "max_turns": int,           # 轮数上限
                "last_has_calls": bool,     # 最新一轮模型是否发出了工具调用
                "tool_history": [str, ...]} # 按序的工具名列表
    Returns:
        (stop: bool, reason: str | None)：
          turns >= max_turns          → (True, "max_turns")
          末尾 3 个工具名相同          → (True, "loop_detected")
          last_has_calls is False     → (True, "no_tool_calls")
          其余                         → (False, None)
    """
    if state["turns"] >= state["max_turns"]:
        return True, "max_turns"
    history = state.get("tool_history", [])
    if len(history) >= 3 and len(set(history[-3:])) == 1:
        return True, "loop_detected"
    if not state.get("last_has_calls", True):
        return True, "no_tool_calls"
    return False, None


# ── 题 4：pass_at_1 —— 一次通过率（15 分，与课程脚本 03 同名同语义）──
def pass_at_1(runs):
    """pass^1 = 通过次数 / 总次数；空列表返回 None（区分"全挂"与"没跑"）。

    Args:
        runs: list[bool] —— 每次独立运行的 pass 与否
    Returns:
        float | None
    """
    if not runs:
        return None
    return sum(runs) / len(runs)


# ── 题 5（🌟）：mini_mcp_call —— JSON-RPC 三步握手（15 分，弹性题）──
def mini_mcp_call(transport, method, params=None):
    """用 mock transport 完成 MCP 三步握手并调用 method。

    Args:
        transport: mock 对象，接口约定：
            transport.send(request: dict) -> response: dict   # 发请求、收响应
            transport.notify(msg: dict) -> None               # 发通知（无响应）
        method: str  —— 第三步要调的 JSON-RPC 方法（如 "tools/call"）
        params: dict —— 第三步的参数（缺省 {}）
    Returns:
        dict: 第三步响应的 result 字段；响应带 error（或任何一步出错）时返回 None。
    步骤：
        ① send initialize 请求（params 含 protocolVersion/clientInfo，带自增 id）
        ② notify notifications/initialized（消息【无 id】——JSON-RPC notification）
        ③ send method 调用（新的 id），返回其 result
    """
    try:
        # ① initialize 请求（带 id=1）
        init_resp = transport.send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05",
                       "clientInfo": {"name": "mini-client", "version": "0.1.0"},
                       "capabilities": {}}})
        if "error" in init_resp:
            return None
        # ② notifications/initialized：通知，无 id、不收响应
        transport.notify({"jsonrpc": "2.0",
                          "method": "notifications/initialized"})
        # ③ 真正的方法调用（新的 id=2）
        call_resp = transport.send({
            "jsonrpc": "2.0", "id": 2, "method": method,
            "params": params if params is not None else {}})
        if "error" in call_resp:
            return None
        return call_resp.get("result")
    except Exception:
        return None
