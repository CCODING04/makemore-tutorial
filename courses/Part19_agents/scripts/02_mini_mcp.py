#!/usr/bin/env python3
"""
Part 19 - 脚本 02: Mini-MCP —— 100 行看穿 Model Context Protocol
目标：同文件内实现一个 toy MCP server（stdin/stdout JSON-RPC 2.0，暴露 echo/add
  两工具）+ 一个 mini client（握手 → 列工具 → 调工具），把"协议标准化"从名词
  变成可调试的代码。

真实 MCP 规范对照（https://modelcontextprotocol.io）：
  ① JSON-RPC 2.0 消息格式   —— 本脚本的 request/response 一比一实现
  ② initialize 握手          —— client/server 交换协议版本与能力（capabilities）
  ③ tools/list              —— server 声明自己有哪些工具（JSON schema，与脚本 01
                                的 TOOL_SPECS 同一格式——MCP 就是把这层标准化了）
  ④ tools/call              —— 调用工具，结果包在 content 数组里返回
  真实 MCP 还有 resources/prompts/prompts 能力协商、stdio 之外的 SSE/streamable
  HTTP 传输等——本脚本只保留教学主干。

A2A（https://a2a-protocol.org）：agent↔agent 协议，靠"Agent Card"（一个 JSON
  文档描述本 agent 的能力/端点）互相发现，传输用 JSON-RPC（含 SSE 流）。
  本脚本不实现 A2A——三层分工见教程 02 章表格：MCP=agent↔工具，A2A=agent↔agent，
  AGENTS.md=agent↔代码库。

运行（纯 CPU，秒级，零模型依赖）：python 02_mini_mcp.py
原理：main() 以 client 身份用 subprocess 启动【自身】加 --server 参数作为
  server 进程，两边通过 stdin/stdout 互发 JSON-RPC 消息（MCP stdio 传输同款）。
"""

import json
import subprocess
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROTOCOL_VERSION = "2024-11-05"     # MCP 真实版本号（示例用；真实 server 会协商）
SERVER_INFO = {"name": "mini-mcp", "version": "0.1.0"}


# ═══ 1. Server 侧：三个 JSON-RPC 方法 ═══
# MCP server 的工具定义与脚本 01 的 TOOL_SPECS 完全同构（OpenAI tools JSON schema）
SERVER_TOOLS = [
    {"name": "echo",
     "description": "Echo back the input text.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string", "description": "Text to echo"}},
         "required": ["text"]}},
    {"name": "add",
     "description": "Add two integers.",
     "inputSchema": {"type": "object", "properties": {
         "a": {"type": "integer"}, "b": {"type": "integer"}},
         "required": ["a", "b"]}},
]


def dispatch(method, params):
    """JSON-RPC 方法分发：initialize / tools/list / tools/call。
    返回值 = response 的 result 字段；未知方法抛错 → JSON-RPC error（协议要求）。"""
    if method == "initialize":
        # 握手：回应协议版本 + server 信息 + 能力声明（本 server 只有 tools）
        return {"protocolVersion": PROTOCOL_VERSION,
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {}}}
    if method == "tools/list":
        # 列工具：真实 MCP 还支持分页 cursor，这里一次给全
        return {"tools": SERVER_TOOLS}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name == "echo":
            text = args.get("text", "")
            # ⭐ MCP 工具结果格式：content 数组（type=text）+ isError 标志
            return {"content": [{"type": "text", "text": f"echo: {text}"}], "isError": False}
        if name == "add":
            a, b = args.get("a"), args.get("b")
            if not isinstance(a, int) or not isinstance(b, int):
                # 工具级错误用 isError=True 表达（区别于协议级 JSON-RPC error）
                return {"content": [{"type": "text",
                                     "text": f"Error: a and b must be integers, got {a!r}, {b!r}"}],
                        "isError": True}
            return {"content": [{"type": "text", "text": str(a + b)}], "isError": False}
        raise ValueError(f"unknown tool: {name}")
    raise ValueError(f"method not found: {method}")


def serve():
    """server 主循环：逐行读 stdin（每行一条 JSON-RPC 消息），逐行写 stdout。
    这就是 MCP stdio 传输的最小形态——真实的 server 用官方 SDK，但线上的字节
    格式与这里相同（newline-delimited JSON）。"""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)   # 行缓冲：立即发给 client
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            # ⭐ JSON-RPC 2.0：没有 id 的消息是 notification——【禁止回应】。
            # （第一版 bug 实录：给 notifications/initialized 回了 error，client 把它
            #  当成下一条 tools/list 的响应读走 → 两条消息读串——协议细节即工程bug）
            if "id" not in req:
                continue
            result = dispatch(req["method"], req.get("params", {}))
            resp = {"jsonrpc": "2.0", "id": req["id"], "result": result}
        except Exception as e:      # 协议级错误：code + message（JSON-RPC 2.0 规范）
            resp = {"jsonrpc": "2.0", "id": req.get("id"),
                    "error": {"code": -32601, "message": str(e)}}
        print(json.dumps(resp), flush=True)   # flush 必须：不 flush client 会死等


# ═══ 2. Client 侧：三步握手 + 调用 ═══
class MiniMCPClient:
    """最小 MCP client：subprocess 启动 server（自身 --server），按序发 JSON-RPC。"""

    def __init__(self):
        # 用自身当 server：真实场景这里是 `npx mcp-server-xxx` 之类的命令
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._id = 0
        self.log = []                       # 握手/调用日志（教学展示用）

    def request(self, method, params=None):
        """发一条 JSON-RPC request，阻塞读一行 response。"""
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method,
               "params": params or {}}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        self.log.append((method, req["params"], resp))
        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error on {method}: {resp['error']}")
        return resp["result"]

    def handshake(self):
        """MCP 三步：initialize → （收到结果）→ notifications/initialized。
        第二步是 notification（无 id、无需回应）——真实 client 必须发，本玩具保留
        以还原时序。"""
        info = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": "mini-client", "version": "0.1.0"},
            "capabilities": {}})
        # notifications/initialized：通知性质，无 id → server 不回应（写完即继续）
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()
        return info

    def list_tools(self):
        return self.request("tools/list")["tools"]

    def call(self, name, arguments):
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        return result["content"][0]["text"], result["isError"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


def main():
    print("═══ Part 19 脚本 02：Mini-MCP（JSON-RPC 2.0 over stdio）═══")
    print(f"  server: {SERVER_INFO}，tools: {[t['name'] for t in SERVER_TOOLS]}\n")

    client = MiniMCPClient()
    print("── Step 1: initialize 握手 ──")
    info = client.handshake()
    print(f"  → server 回应: protocolVersion={info['protocolVersion']}, "
          f"server={info['serverInfo']}, capabilities={list(info['capabilities'])}")

    print("\n── Step 2: tools/list ──")
    tools = client.list_tools()
    for t in tools:
        print(f"  → {t['name']}: {t['description']} "
              f"schema.required={t['inputSchema'].get('required')}")

    print("\n── Step 3: tools/call ──")
    text, err = client.call("echo", {"text": "hello mcp"})
    print(f"  → echo(hello mcp)      = {text!r} (isError={err})")
    text, err = client.call("add", {"a": 23, "b": 47})
    print(f"  → add(23, 47)          = {text!r} (isError={err})")
    text, err = client.call("add", {"a": "x", "b": 1})
    print(f"  → add('x', 1)          = {text!r} (isError={err})   ← 工具级错误")

    print("\n── Step 4: 协议级错误（未知方法）──")
    try:
        client.request("resources/list")     # 本 server 未实现的方法
    except RuntimeError as e:
        print(f"  → {e}")
    try:
        client.call("multiply", {"a": 2, "b": 3})
    except RuntimeError as e:
        print(f"  → {e}")

    print("\n── 线上字节流（client 视角，前 5 条）──")
    for method, params, resp in client.log[:5]:
        req_preview = json.dumps(params, ensure_ascii=False)[:60]
        resp_preview = json.dumps(resp, ensure_ascii=False)[:76]
        print(f"  {method:<22} params={req_preview:<60} → {resp_preview}")

    client.close()

    print("""
  对照与解读（教程 02 章）：
  - MCP 把脚本 01 里"TOOLS schema + 执行器"这一层标准化成了跨进程协议：
    tools/list 之于 TOOL_SPECS，tools/call 之于 execute_tool——agent 不再自己
    实现工具，而是连任意 MCP server（文件系统/GitHub/数据库……官方与社区有大量
    现成 server）；
  - 两级错误要分清：工具级错误走 isError=True（调用失败但协议正常），协议级
    错误走 JSON-RPC error（方法不存在/参数非法）——混用会让 client 无法重试；
  - MCP 是开放标准、多厂商支持（官方规范：https://modelcontextprotocol.io）。""")


if __name__ == "__main__":
    if "--server" in sys.argv:
        serve()          # 被 client 以子进程方式拉起
    else:
        main()
