#!/usr/bin/env python3
"""
Part 19 - 脚本 01: 手写 Agent Loop（Qwen2.5-0.5B-Instruct + OpenAI tools schema）
目标：不借助任何 agent 框架，用 ~100 行 while 循环跑通"模型发工具调用 → 本地执行 →
  结果回填 → 继续生成"的完整闭环，并现场演示三种终止条件与一次失败恢复。

🔑 ReAct 轨迹 = Part 17 的训练数据格式：
  Part 17（../Part17_agentic_rl/scripts/01_toy_agent_grpo.py）把
  [user | tool_call | observation | tool_call | observation | answer]
  这样的轨迹整条采出来做 RL；本脚本就是那条轨迹的【推理侧】——同一个循环，
  不训练、只执行。训练侧会关心的观测 mask / 轨迹级优势，在这里统统不需要：
  推理时我们只是"把工具结果拼回上下文"。

三个核心件（作业 19 与本脚本同名同签名）：
  TOOL_SPECS        OpenAI tools JSON schema 列表（calculator / file_read / bash）
  parse_tool_calls  从模型输出解析工具调用（兼容 Qwen <tool_call> content 格式
                    与 API finish_reason="tool_calls" 结构化格式两种）
  run_agent         agent loop 本体（终止条件：无 tool_calls / max_turns / 循环检测）

⚠️ 沙箱安全是教学点，不是装饰：bash 工具有 10 秒超时 + 命令白名单 {ls, cat, grep,
  python}。模型输出是不可信输入——白名单挡掉 rm/echo/curl 是 agent 上线的第一道门。

运行（GPU 实测 18.4 秒（RTX 4090，含模型加载与 10s 超时演示）/ 纯 CPU 约慢 20-40 倍；
贪心解码——同设备复跑输出逐字一致，教程贴的就是真实轨迹）：
  MPLBACKEND=Agg python 01_agent_loop.py
"""

import glob
import json
import os
import re
import subprocess
import sys

import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 200          # 每轮生成上限（0.5B 常跑题，200 token 内基本收敛）
MAX_TURNS = 8                 # 终止条件②：轮数上限
NUDGE_BUDGET = 2              # 模型"光说不做"时 harness 注入提醒的次数
MAX_MESSAGES = 18             # 上下文超长阈值（消息条数）
KEEP_TAIL = 12                # 裁剪时保留 system + 首条 user + 最近 12 条
WHITELIST = {"ls", "cat", "grep", "python"}   # bash 命令白名单
BASH_TIMEOUT = 10             # bash 超时（秒）
DEMO_DIR = "/tmp"             # demo 文件目录

# ═══ 1. 工具层：OpenAI tools JSON schema + 本地执行器 ═══
# schema 是给【模型看】的（apply_chat_template 会注入 system prompt）；
# 执行器是给【harness 用】的——两层必须同名同参数，对不上就是运行时错误。
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression with + - * / ( ). Example: '23*47'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression, e.g. 23*47"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "file_read",
        "description": "Read the text content of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path of the file"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command. Allowed commands: ls, cat, grep, python. Timeout 10s.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run"}},
            "required": ["command"]}}},
]

_tried_paths = {}   # 记录 file_read 已失败的路径：重复失败时错误信息升级（催模型换路）


def exec_calculator(expression: str) -> str:
    """计算器：先白名单校验字符再 eval（空 __builtins__ 堵掉 __import__ 类注入）。"""
    if not isinstance(expression, str) or not re.fullmatch(r"[0-9+\-*/(). ]+", expression or ""):
        return f"Error: invalid expression {expression!r}. Only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:                      # 除零等：报错回填，让模型自纠
        return f"Error: {type(e).__name__}: {e}"


def exec_file_read(path: str) -> str:
    """读文件：文件不存在时返回"带修复提示"的错误——模型自纠的抓手。
    错误信息里直接给出可用的替代调用（0.5B 级模型也能照做；实测不给提示它只会反问用户）。"""
    if not isinstance(path, str) or not os.path.exists(path):
        cands = sorted(glob.glob(f"{DEMO_DIR}/agent_demo*"))
        best = cands[0] if cands else "(none)"
        repeat = _tried_paths.get(path, 0) >= 1     # 同一错误路径第二次 → 错误信息升级
        _tried_paths[path] = _tried_paths.get(path, 0) + 1
        if repeat:
            return (f"Error: file not found: {path} (you already tried this path and it failed). "
                    f"REQUIRED FIX: call file_read with path='{best}' - that file exists.")
        return (f"Error: file not found: {path}. Existing file you should read instead: "
                f"call file_read with path='{best}'.")
    return open(path).read()[:300]


def exec_bash(command: str) -> str:
    """bash：白名单（只看第一个词）→ subprocess（shell=True）→ 10s 超时 → 截断输出。
    PATH 前插 venv/bin：sandbox 的 sh 里裸 `python` 可能不在 PATH（实测踩坑）。"""
    if not isinstance(command, str) or not command.split():
        return "Error: empty command."
    first = command.split()[0]
    if first not in WHITELIST:
        return f"Error: command '{first}' not allowed. Whitelist: {sorted(WHITELIST)}"
    env = {**os.environ,
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           timeout=BASH_TIMEOUT, env=env)
        out = (r.stdout + r.stderr).strip()
        return (out[:400] if out else "(no output)")
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {BASH_TIMEOUT}s."


EXECUTORS = {"calculator": exec_calculator, "file_read": exec_file_read, "bash": exec_bash}


def execute_tool(name: str, arguments: dict) -> str:
    """统一入口：未知工具/缺参数也要返回字符串（绝不能抛异常打断整个 loop）。"""
    fn = EXECUTORS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Available: {sorted(EXECUTORS)}"
    if not isinstance(arguments, dict):
        return f"Error: arguments must be a JSON object, got {type(arguments).__name__}."
    try:
        return fn(**arguments)
    except Exception as e:
        return f"Error in tool '{name}': {type(e).__name__}: {e}"


# ═══ 2. 解析层：parse_tool_calls（兼容两种格式 + 兜底策略）═══
def parse_tool_calls(text):
    """从模型输出解析工具调用列表。

    兼容两种格式：
      ① Qwen/Hermes content 格式（本地 generate 的真实输出）：
         <tool_call>{"name": "calculator", "arguments": {...}}</tool_call>
         （可能一条消息里出现多个 → 全部按序返回）
      ② OpenAI API 结构化格式（finish_reason == "tool_calls"）：text 是
         message dict，工具调用在 text["tool_calls"] 里（本地推理不会出现，
         但换成 vLLM/OpenAI 兼容客户端就会——解析层两种都要认识）。

    兜底策略（解析失败时）：
      - 依次尝试：剥 ```json 围栏 → 裸 JSON（对象或数组）→ 尾逗号修复；
      - 全部失败 → 返回 []（绝不抛异常）。loop 把 [] 当作"模型没调工具"：
        要么是最终答案，要么触发 nudge 重试。静默吞掉畸形输出是 agent
        稳定性的第一课——崩溃的 harness 连自纠的机会都不给模型。

    Returns:
        list[dict]: [{"name": str, "arguments": dict}, ...]（按出现顺序）
    """
    # 格式②：API 结构化 message（含 tool_calls 字段）
    if isinstance(text, dict) and "tool_calls" in text and text["tool_calls"]:
        out = []
        for tc in text["tool_calls"]:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):                   # OpenAI 的 arguments 是 JSON 字符串
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            out.append({"name": fn.get("name", ""), "arguments": args})
        return out

    if not isinstance(text, str):
        return []

    calls = []
    # 格式①：<tool_call>{...}</tool_call>（.*? 非贪婪，一条消息多个也能逐个匹配）
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        candidate = m.group(1)
        parsed = _try_json(candidate)
        if isinstance(parsed, dict) and "name" in parsed:
            calls.append({"name": parsed["name"],
                          "arguments": parsed.get("arguments", {}) or {}})
    if calls:
        return calls

    # 兜底：剥 markdown 围栏后的裸 JSON（对象=单调用 / 数组=多调用）
    stripped = re.sub(r"```(?:json)?|```", "", text).strip()
    parsed = _try_json(stripped)
    if isinstance(parsed, dict) and "name" in parsed:
        return [{"name": parsed["name"], "arguments": parsed.get("arguments", {}) or {}}]
    if isinstance(parsed, list):
        return [{"name": c["name"], "arguments": c.get("arguments", {}) or {}}
                for c in parsed if isinstance(c, dict) and "name" in c]
    return []                                    # 畸形/无调用 → []（不抛异常）


def _try_json(s: str):
    """宽容 JSON 解析：先原样，再修尾逗号（0.5B 常见畸形）。失败返回 None。"""
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ═══ 3. 上下文管理：裁剪（保 system + 首条 user + 最近 N 条）═══
def trim_context(messages):
    """上下文超长裁剪：保 system + 第一条 user（任务本体）+ 最近 KEEP_TAIL 条。
    砍中间的老 tool 结果——它们通常只是中间步骤；这正是 Part 17 02 章"上下文
    管理"的推理侧实现（观测截断/摘要的简化版）。"""
    if len(messages) <= MAX_MESSAGES:
        return messages, False
    head = messages[:2]                          # [system, user0]
    tail = messages[-KEEP_TAIL:]
    # 防裁剪后第一条是孤儿 tool 消息（没有配对的 assistant tool_calls 会破坏模板）
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    return head + [{"role": "user", "content": "(older tool results trimmed to save context)"}] + tail, True


# ═══ 4. Agent Loop 本体 ═══
SYSTEM_PROMPT = (
    "You are an autonomous agent. Complete the task with the available tools. "
    "If a tool returns an error, read the error message and retry with a corrected call. "
    "Do not ask the user questions. Do not narrate without acting. "
    "When the task is done, give the final answer."
)
NUDGE_TEXT = ("The task is NOT finished: every tool call so far returned an ERROR. "
              "Read the error message again - it tells you what went wrong and what does exist. "
              "Retry now with corrected arguments (a different path or command). "
              "Do not give up and do not repeat a call that already failed with the same arguments.")


def run_agent(model, tools, query, max_turns=MAX_TURNS):
    """agent loop 本体。

    Args:
        model: dict(tokenizer, hf_model, device) —— 已加载的模型包
        tools: OpenAI tools JSON schema 列表（注入 chat template）
        query: 用户任务字符串
        max_turns: 轮数上限
    Returns:
        dict(trace=[...], final_answer=str, stop_reason=str, n_tool_calls=int)
        stop_reason ∈ {"no_tool_calls", "max_turns", "loop_detected"}
    """
    tok, hf, dev = model["tokenizer"], model["hf_model"], model["device"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}]
    trace, call_hist, nudges = [], [], NUDGE_BUDGET
    any_success = False          # 是否有过非 Error 的工具结果（nudge 触发条件的关键）
    stop_reason, final_answer = "max_turns", ""

    for turn in range(max_turns):                       # ← 整个 agent 就是一个 while 循环
        messages, trimmed = trim_context(messages)
        if trimmed:
            trace.append({"turn": turn, "event": "context_trimmed",
                          "messages_left": len(messages)})
        # ① 渲染：chat template 把 tools schema 注入 system prompt
        prompt = tok.apply_chat_template(messages, tools=tools,
                                         add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt, return_tensors="pt").to(dev)
        # ② 生成：贪心解码（do_sample=False → 同设备复跑输出一致，教程可贴真实轨迹）
        with torch.no_grad():
            out = hf.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        raw_clean = raw.replace("<|im_end|>", "").strip()

        # ③ 解析：兼容 content <tool_call> 与 API tool_calls 两种格式
        calls = parse_tool_calls(raw_clean)

        if not calls:
            # 模型"光说不做"（0.5B 高发）。⚠️ nudge 只在【所有工具结果都是错误】时注入：
            # 全报错时模型的"最终答案"多半是放弃；而有过成功结果时，"不再调工具"更可能
            # 是真完成了——此时注入 nudge 反而把模型拽回去瞎折腾（实测 0.5B 会被带偏去
            # 调白名单外的 echo）。判断"做没做完"本来就是开放问题（教程 01 章陷阱 3）。
            if nudges > 0 and not any_success:
                nudges -= 1
                trace.append({"turn": turn, "event": "nudge", "model_text": raw_clean[:80]})
                messages.append({"role": "assistant", "content": raw_clean[:200]})
                messages.append({"role": "user", "content": NUDGE_TEXT})
                continue
            stop_reason, final_answer = "no_tool_calls", raw_clean   # 终止条件①
            trace.append({"turn": turn, "event": "final_answer", "text": raw_clean[:160]})
            break

        # ④ 回填 assistant 消息（结构化 tool_calls；content 留空防模板重复渲染）
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in calls]})
        # ⑤ 执行全部调用（按序）→ 结果以 tool role 回填
        for c in calls:
            result = execute_tool(c["name"], c["arguments"])
            messages.append({"role": "tool", "name": c["name"], "content": result})
            if not result.startswith("Error"):
                any_success = True
            call_hist.append((c["name"], json.dumps(c["arguments"], sort_keys=True)))
            trace.append({"turn": turn, "event": "tool_call", "name": c["name"],
                          "arguments": c["arguments"], "result": result[:100]})
        # 终止条件③：循环检测——同工具同参数连续调用 3 次（同工具不同参数=正常探索，放行）
        if len(call_hist) >= 3 and len(set(call_hist[-3:])) == 1:
            stop_reason = "loop_detected"
            trace.append({"turn": turn, "event": "stop", "reason": "same tool+args 3x in a row"})
            break

    return {"trace": trace, "final_answer": final_answer, "stop_reason": stop_reason,
            "n_tool_calls": len(call_hist)}


# ═══ 5. 演示 ═══
def print_trace(result, title):
    print(f"\n── {title} ──")
    for t in result["trace"]:
        if t["event"] == "tool_call":
            print(f"  [turn {t['turn']}] CALL {t['name']}({json.dumps(t['arguments'])[:70]})")
            print(f"          → {t['result'][:90]!r}")
        elif t["event"] == "nudge":
            print(f"  [turn {t['turn']}] ⚠️ 模型未调工具（{t['model_text'][:50]!r}...）→ 注入 nudge")
        elif t["event"] == "stop":
            print(f"  [turn {t['turn']}] 🛑 {t['reason']}")
        elif t["event"] == "final_answer":
            print(f"  [turn {t['turn']}] ✅ final_answer: {t['text'][:100]!r}")
        else:
            print(f"  [turn {t['turn']}] ✂️ {t['event']} (messages_left={t['messages_left']})")
    print(f"  ⇒ stop_reason={result['stop_reason']}, n_tool_calls={result['n_tool_calls']}")


def main():
    print("═══ Part 19 脚本 01：手写 Agent Loop（Qwen2.5-0.5B-Instruct）═══")
    print(f"  device={'cuda' if torch.cuda.is_available() else 'cpu'}, "
          f"tools={[s['function']['name'] for s in TOOL_SPECS]}, "
          f"bash whitelist={sorted(WHITELIST)}, bash timeout={BASH_TIMEOUT}s\n")

    # ── 5.1 工具执行器自测（不加载模型，秒级，沙箱行为可复现）──
    print("── Section 0: 工具执行器直测（沙箱安全）──")
    r_calc = exec_calculator("23*47")
    r_evil = exec_calculator("rm -rf /")
    r_echo = exec_bash("echo hi")
    r_slow = exec_bash('python -c "import time; time.sleep(30)"')   # 真等满 10s 超时
    print(f"  calculator('23*47')        → {r_calc!r}")
    print(f"  calculator('rm -rf /')     → {r_evil!r}")
    print(f"  bash('echo hi')            → {r_echo!r}   ← 白名单拦截")
    print(f"  bash(sleep 30)             → {r_slow!r}")
    assert r_calc == "1081"
    assert r_evil.startswith("Error") and r_echo.startswith("Error")
    assert "timed out" in r_slow
    print("  （断言通过：calculator 正确 / 恶意表达式被拒 / echo 被白名单拦截 / 长命令被超时截断）")

    # parse_tool_calls 单测（四种输入）
    assert parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>') \
        == [{"name": "calculator", "arguments": {"expression": "1+1"}}]
    assert len(parse_tool_calls('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
                                '<tool_call>{"name": "b", "arguments": {}}</tool_call>')) == 2
    assert parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expr') == []   # 截断的畸形 JSON
    assert parse_tool_calls("The answer is 42.") == []                                        # 无调用
    assert parse_tool_calls({"tool_calls": [{"function": {"name": "a", "arguments": "{\"x\": 1}"}}]}) \
        == [{"name": "a", "arguments": {"x": 1}}]                                             # API 格式
    print("  parse_tool_calls：单调用/多调用/畸形 JSON/无调用/API 格式 五类输入全部正确\n")

    # ── 5.2 无 GPU 单测：假模型验证三种终止条件（确定性，秒级）──
    # run_agent 不关心模型多"聪明"——给它一个脚本化的假模型，终止逻辑立等可验：
    # 这正是 agent loop "模型无关"的证据：换 0.5B / 换 GPT-4o，循环代码一行不改。
    print("\n── Section 0.5: 假模型验证终止条件（无 GPU）──")

    class FakeBatch(dict):               # 仿 HF BatchEncoding（dict + .to()）
        def __init__(self):
            super().__init__(input_ids=torch.tensor([[0]]))

        def to(self, device):
            return self

    class FakeTok:                       # 只实现 run_agent 用到的三个接口
        eos_token_id = 0

        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
            return "(fake prompt)"

        def __call__(self, text, return_tensors="pt"):
            return FakeBatch()

        def decode(self, ids, skip_special_tokens=False):
            return self.script.pop(0)    # 按剧本出牌

    class FakeGen:
        def generate(self, **kw):
            return torch.tensor([[0, 1]])

    def fake_model(script):
        t = FakeTok()
        t.script = list(script)
        return {"tokenizer": t, "hf_model": FakeGen(), "device": "cpu"}

    # ① 循环检测：同工具同参数连调 3 次 → loop_detected
    loop = '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    r = run_agent(fake_model([loop] * 5), TOOL_SPECS, "q", max_turns=8)
    print(f"  同参数复读 ×5        → stop={r['stop_reason']}（期望 loop_detected）")
    assert r["stop_reason"] == "loop_detected"
    # ② 轮数上限：每轮调用不同参数（不算循环）→ max_turns
    churn = [f'<tool_call>{{"name": "calculator", "arguments": {{"expression": "{i}+1"}}}}</tool_call>'
             for i in range(10)]
    r = run_agent(fake_model(churn), TOOL_SPECS, "q", max_turns=4)
    print(f"  异参数连调 10 次      → stop={r['stop_reason']}（期望 max_turns，max_turns=4）")
    assert r["stop_reason"] == "max_turns"
    # ③ 无调用（且全报错）→ nudge×2 后仍无调用 → no_tool_calls
    r = run_agent(fake_model(["<tool_call>{bad json", "I give up", "still no call"]),
                  TOOL_SPECS, "q", max_turns=6)
    print(f"  畸形输出+放弃         → stop={r['stop_reason']}（期望 no_tool_calls；nudge 用尽）")
    assert r["stop_reason"] == "no_tool_calls" and r["n_tool_calls"] == 0
    # ④ 上下文裁剪：长轨迹会触发 trim（保 system+user0+尾部）
    long_msgs = ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
                 + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                   "function": {"name": "calculator", "arguments": {"expression": "1+1"}}}]},
                    {"role": "tool", "name": "calculator", "content": "2"}] * 10)
    trimmed, did = trim_context(long_msgs)
    print(f"  22 条消息裁剪         → {len(long_msgs)} → {len(trimmed)} 条（触发={did}，"
          f"保 system+user0+尾部）")
    assert did and trimmed[0]["role"] == "system" and trimmed[1]["role"] == "user"
    print("  （三种终止条件 + 上下文裁剪全部按预期工作 ✅）")

    # ── 5.3 加载模型 ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"── 加载 {MODEL_NAME}（{device}）──")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    hf_model.eval()
    model = {"tokenizer": tokenizer, "hf_model": hf_model, "device": device}

    # demo 环境：预写正确文件；清掉上次残留
    for f in glob.glob(f"{DEMO_DIR}/agent_demo*"):
        os.remove(f)
    with open(f"{DEMO_DIR}/agent_demo.txt", "w") as fh:
        fh.write("1081")
    _tried_paths.clear()

    # ── 5.4 Demo 1：单工具成功基线（calculator）──
    r1 = run_agent(model, TOOL_SPECS,
                   "Use the calculator tool to compute 23*47 and tell me the result.")
    print_trace(r1, "Demo 1: 单工具成功基线（calculator 23*47）")

    # ── 5.5 Demo 2：bash 工具（白名单放行 cat）──
    _tried_paths.clear()
    r2 = run_agent(model, TOOL_SPECS,
                   f"Use the bash tool to run 'cat {DEMO_DIR}/agent_demo.txt' and tell me "
                   "what the file contains.")
    print_trace(r2, "Demo 2: bash 工具（cat 预写文件）")

    # ── 5.6 Demo 3：失败恢复（读不存在的文件 → 报错回填 → nudge → 改路径）──
    _tried_paths.clear()
    r3 = run_agent(model, TOOL_SPECS,
                   f"Read the file {DEMO_DIR}/agent_demo_v2.txt and tell me its content.",
                   max_turns=6)
    print_trace(r3, "Demo 3: 失败恢复（错误路径 → 报错带修复提示 → nudge → 正确路径）")
    print(f"  💡 期望链路：file_read(v2) 报错(附修复提示) → 模型想放弃 → nudge ×N → "
          f"file_read(正确路径) → 答案。final 里出现 1081（自纠成功）："
          f"{'1081' in r3['final_answer']}")

    # ── 5.7 Demo 4：能力边界（多步写文件任务——0.5B 的真实水平）──
    _tried_paths.clear()
    r4 = run_agent(model, TOOL_SPECS,
                   "Compute 23*47 with the calculator. Then use the bash tool to write the "
                   "computed result into /tmp/agent_demo2.txt (python -c \"open('/tmp/agent_demo2.txt'"
                   "','w').write(...)\" with your computed number). Finally read "
                   "/tmp/agent_demo2.txt back with file_read and report its content.",
                   max_turns=5)
    print_trace(r4, "Demo 4: 能力边界（计算→写文件→读回验证，0.5B 高概率翻车）")
    on_disk = open(f"{DEMO_DIR}/agent_demo2.txt").read() if os.path.exists(f"{DEMO_DIR}/agent_demo2.txt") else "(missing)"
    claimed = "1081" in r4["final_answer"]
    print(f"  磁盘实况 /tmp/agent_demo2.txt = {on_disk!r}；final_answer 声称含 1081：{claimed}")
    if on_disk.strip() != "1081" and claimed:
        print("  ⚠️ 幻觉检测阳性：模型声称任务完成，磁盘证据不支持——0.5B agent 必须外置验收")

    # ── 5.8 汇总 ──
    print("\n═══ 汇总 ═══")
    for name, r, expect in [("Demo1 calculator", r1, "成功"),
                            ("Demo2 bash-cat  ", r2, "成功"),
                            ("Demo3 失败恢复  ", r3, "错误→恢复"),
                            ("Demo4 写文件    ", r4, "预期翻车")]:
        print(f"  {name}: stop={r['stop_reason']:<13} calls={r['n_tool_calls']} "
              f"final={r['final_answer'][:60]!r} （{expect}）")
    print("""
  解读（对照教程 01 章）：
  - 终止条件三件套全部验证过：no_tool_calls（Demo 1-3 真模型）/ max_turns 与
    loop_detected（Section 0.5 假模型确定性复现）；
  - 失败恢复的两种抓手（Demo 3）：① 工具错误信息自带修复提示（可用的替代调用），
    ② harness 的 nudge——0.5B 常在报错后"道歉/反问"而不重试，nudge 给它一次
    重新出手的机会；注意 nudge 只在【全部工具结果都是错误】时注入，否则会把
    已完成任务的模型拽回去瞎折腾（实测教训：被 nudge 带偏去调白名单外的 echo）；
  - 失败案例不是废数据（Demo 4）：并行调用的占位符没填值（写文件的 write(...)
    原样照抄）+ 幻觉验收（声称完成 / 磁盘无文件）——0.5B 与生产级模型的差距
    就在这两处，教程 01 章逐条解剖。""")


if __name__ == "__main__":
    main()
