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
        required 按参数定义顺序排列；无必填参数时 required 为空列表。
    """
    # TODO: 分两条路径（callable → inspect.signature；dict → 直接读字段）
    # TODO: required = 无默认值的参数；dict 路径全部参数进 required
    # TODO: type 非法（不在 _LEGAL_TYPES）回退 "string"
    return None


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
    # TODO: re.finditer 找全部 <tool_call>...</tool_call> 块
    # TODO: 逐块宽容 JSON 解析（原样 → 修尾逗号），dict 且含 name 才收录
    return None


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
    # TODO: 按 max_turns → loop（tool_history[-3:] 全同）→ no_tool_calls 顺序判定
    return None


# ── 题 4：pass_at_1 —— 一次通过率（15 分，与课程脚本 03 同名同语义）──
def pass_at_1(runs):
    """pass^1 = 通过次数 / 总次数；空列表返回 None（区分"全挂"与"没跑"）。

    Args:
        runs: list[bool] —— 每次独立运行的 pass 与否
    Returns:
        float | None
    """
    # TODO: 空列表 → None；否则 sum(runs)/len(runs)
    return None


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
    # TODO: 三步时序；①③ 的 id 必须互不相同；任何响应含 "error" → return None
    return None
