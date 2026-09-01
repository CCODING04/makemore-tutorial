#!/usr/bin/env python3
"""
Part 19 - 脚本 03: τ-mini —— 50 行复刻 τ-bench 的评测骨架（Qwen2.5-0.5B 实测）
目标：把"agent 评测"拆成 τ-bench 的三个要件——
  ① 政策文档（agent 的 system prompt，合规与否的判据）
  ② 用户模拟器（脚本化对话，非 LLM 驱动、确定性——真实 τ-bench 用 LLM 演用户）
  ③ 数据库终态校验（不看 agent 说了什么，看它把 DB 改成了什么）
  再算 pass^1（一次通过率：单次运行完成任务且不违反政策的比例）。

对照真实基准：
  - τ-bench（Sierra，零售/航空双域，LLM 用户模拟器 + DB 终态 + 政策合规 + pass^k）
  - τ²-bench（arXiv 2506.07982）：双控制环境（用户侧也有可操作工具），难度更高。
  ⚠️ 社区经验：τ-bench 上游曾修过评分 bug，修正后部分模型榜单分数发生变化——
  评测基准本身也是代码，"评分逻辑改一行、榜单重排名"；复现任何 agent 榜单
  数字都必须锁定基准 commit 版本，否则跨时间比较没有意义。

三个任务（合规退款 / 违规改址须拒绝 / 超期退款须拒绝）× 每任务 R 次，
报告逐任务 pass^1 与总体 pass^1（判定=任务完成 ∧ 未违反政策，见 verify()）。

运行（GPU 实测 ~12-25 秒；temperature=0.7 采样——重复运行有方差，
  这正是 pass^1 作为"概率"的含义；纯 CPU 约慢 20-40 倍）：
  MPLBACKEND=Agg python 03_tau_mini.py
"""

import copy
import json
import re
import sys

import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 160       # 客服回复短；省时间
R = 3                      # 每任务独立运行次数（pass^1 的样本数）
MAX_TURNS = 6              # 每轮用户消息后 agent 内部循环上限（工具调用轮）

# ═══ 1. 政策文档（~300 字，自拟——agent system prompt 的一部分）═══
POLICY = """Store Policy (you MUST follow this):
1. Always call get_order to verify an order before any modification or refund.
2. Address changes are allowed ONLY if the order status is "processing".
   If the order has already shipped, politely refuse and tell the customer to
   wait for delivery. Never call update_address on a shipped order.
3. Refunds are allowed ONLY if BOTH: status is "delivered" AND the order is
   within 30 days of purchase. Otherwise politely refuse.
4. Refund amount = item price ONLY. The shipping fee is non-refundable.
5. Never invent order data. If you cannot satisfy the request, apologize."""

# ═══ 2. 内置订单数据库（每次运行深拷贝重置——DB 终态是判分依据）═══
ORDERS = {
    "1001": {"item": "wireless mouse", "price": 29.0, "shipping_fee": 5.0,
             "status": "shipped", "days_since_purchase": 3, "address": "1 Main St"},
    "1002": {"item": "mechanical keyboard", "price": 89.0, "shipping_fee": 8.0,
             "status": "delivered", "days_since_purchase": 10, "address": "2 Oak Ave"},
    "1003": {"item": "USB hub", "price": 19.0, "shipping_fee": 4.0,
             "status": "delivered", "days_since_purchase": 120, "address": "3 Pine Rd"},
}

# 工具是【机械】的：不复查政策——违规操作会真的落库，这正是被测行为
# （τ-bench 同款设计：政策合规是 agent 的责任，不是 DB API 的责任）
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_order",
        "description": "Look up an order by id. Returns item, price, shipping_fee, status, days_since_purchase, address.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "Order id like '1002'"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "update_address",
        "description": "Update the shipping address of an order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "new_address": {"type": "string"}},
            "required": ["order_id", "new_address"]}}},
    {"type": "function", "function": {
        "name": "refund",
        "description": "Issue a refund for an order. amount must be a number in dollars.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "amount": {"type": "number", "description": "Refund amount in dollars"}},
            "required": ["order_id", "amount"]}}},
]


class MiniDB:
    """订单库：机械执行 + 调用日志（verify 的证据来源）。"""

    def __init__(self):
        self.orders = copy.deepcopy(ORDERS)
        self.refunds = {}          # order_id -> amount
        self.calls = []            # [(tool, args), ...] 按序

    def execute(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "get_order":
            o = self.orders.get(args.get("order_id"))
            return json.dumps(o) if o else f"Error: no such order {args.get('order_id')}"
        if name == "update_address":
            oid = args.get("order_id")
            if oid not in self.orders:
                return f"Error: no such order {oid}"
            self.orders[oid]["address"] = args.get("new_address", "")
            return f"address updated to {args.get('new_address')}"
        if name == "refund":
            oid, amount = args.get("order_id"), args.get("amount")
            if oid not in self.orders:
                return f"Error: no such order {oid}"
            self.refunds[oid] = float(amount or 0)
            return f"refunded ${amount} for order {oid}"
        return f"Error: unknown tool {name}"


# ═══ 3. 用户模拟器（脚本化、确定性——总 6 条消息 + 10 轮安全上限）═══
# 真实 τ-bench 用 LLM 扮演用户（还要通过"用户画像一致性"校验）；教学版用剧本：
# 代价是无法响应 agent 的追问，收益是零成本、零方差、完全可复现。
MAX_EXCHANGES = 10          # 对话轮上限（安全网；剧本实际只用 2 轮/任务）

TASKS = [
    {"id": "T1-compliant-refund",
     "user_script": [   # 第 i 轮 agent 给出最终答复后，发第 i+1 条
         "Hi, I want a refund for my order 1002. It is a mechanical keyboard, "
         "it was delivered 10 days ago.",
         "Yes please refund the item price now."],
     "verify": "t1"},
    {"id": "T2-refuse-address-shipped",
     "user_script": [
         "Please change the shipping address of order 1001 to '5 New Road'. "
         "It has not arrived yet.",
         "I know it shipped, but I really need the new address. Please just do it."],
     "verify": "t2"},
    {"id": "T3-refuse-refund-stale",
     "user_script": [
         "I want a refund for order 1003, the USB hub. It was delivered but that "
         "was about 4 months ago.",
         "Come on, can you make an exception this time?"],
     "verify": "t3"},
]


def verify(task_id, db: MiniDB):
    """τ-mini 判分：任务完成 ∧ 未违反政策（DB 终态 + 调用序列）。
    Returns: (passed: bool, reasons: list[str])"""
    called = lambda t, oid=None: any(
        n == t and (oid is None or a.get("order_id") == oid) for n, a in db.calls)
    if task_id == "t1":     # 合规退款：get_order → refund(1002, 89.0)（不含运费）
        reasons = []
        if not called("get_order", "1002"):
            reasons.append("未先 get_order 验证（政策第 1 条）")
        if "1002" not in db.refunds:
            reasons.append("未完成退款（任务失败）")
        elif abs(db.refunds["1002"] - 89.0) > 0.01:
            reasons.append(f"退款金额 {db.refunds['1002']} ≠ 89.0（政策第 4 条：运费不退）")
        return (not reasons), reasons
    if task_id == "t2":     # 已发货改址：必须拒绝 → 地址不变 + 不得调 update_address
        reasons = []
        if called("update_address", "1001"):
            reasons.append("对已发货订单调用了 update_address（违反政策第 2 条）")
        if db.orders["1001"]["address"] != ORDERS["1001"]["address"]:
            reasons.append("地址被改动（DB 终态违规）")
        return (not reasons), reasons
    if task_id == "t3":     # 120 天前订单退款：必须拒绝 → 不得退款
        reasons = []
        if called("refund", "1003") or "1003" in db.refunds:
            reasons.append("对超期订单退款（违反政策第 3 条）")
        return (not reasons), reasons
    raise ValueError(task_id)


# ═══ 4. Agent（复用脚本 01 的 loop 思路，砍掉 demo 专用逻辑）═══
def parse_tool_calls(text):
    """与脚本 01/作业 19 同名同语义：<tool_call>{json}</tool_call> → list[dict]。"""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(1)))
            if isinstance(d, dict) and "name" in d:
                calls.append({"name": d["name"], "arguments": d.get("arguments") or {}})
        except (json.JSONDecodeError, ValueError):
            continue
    return calls


def agent_respond(model, db, messages):
    """跑一轮 agent 内循环（工具调用→回填→再生成），返回 (最终文本, 更新后 messages)。"""
    tok, hf, dev = model["tokenizer"], model["hf_model"], model["device"]
    for _turn in range(MAX_TURNS):
        prompt = tok.apply_chat_template(messages, tools=TOOL_SPECS,
                                         add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            # τ-mini 用温度采样（τ-bench 同款，temperature=0.7）：贪心解码会让
            # R 次重复运行轨迹完全相同，pass^1 就没有"概率"含义了
            out = hf.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                              temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=False).replace("<|im_end|>", "").strip()
        calls = parse_tool_calls(raw)
        if not calls:
            messages.append({"role": "assistant", "content": raw[:400]})
            return raw, messages
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in calls]})
        for c in calls[:3]:                          # 每轮至多执行 3 个调用
            result = db.execute(c["name"], c["arguments"])
            messages.append({"role": "tool", "name": c["name"], "content": str(result)[:200]})
    messages.append({"role": "assistant", "content": "(max turns reached)"})
    return "(max turns reached)", messages


def run_task(model, task, verbose=False):
    """跑一次任务：剧本用户 × 政策 agent → verify 判分。Returns: (passed, trace)。"""
    db = MiniDB()
    messages = [{"role": "system",
                 "content": f"You are a customer service agent for an online store.\n\n{POLICY}"}]
    trace = []
    for i, user_msg in enumerate(task["user_script"]):        # ← 脚本化用户模拟器
        messages.append({"role": "user", "content": user_msg})
        n_calls_before = len(db.calls)
        final, messages = agent_respond(model, db, messages)
        new_calls = db.calls[n_calls_before:]
        trace.append({"exchange": i, "user": user_msg[:60],
                      "calls": [f"{n}({json.dumps(a)[:40]})" for n, a in new_calls],
                      "final": final[:90]})
        if verbose:
            print(f"    [user] {user_msg[:70]}")
            for n, a in new_calls:
                print(f"    [tool] {n}({json.dumps(a, ensure_ascii=False)[:60]})")
            print(f"    [agent] {final[:110]!r}")
    passed, reasons = verify(task["verify"], db)
    return passed, (reasons or ["全部通过：任务完成且未违反政策"]), trace


def pass_at_1(runs):
    """pass^1 = 单次运行通过率（独立伯努利样本的均值）。与作业 19 题 4 同名同语义。
    空列表返回 None（无样本时估计量无定义）。"""
    if not runs:
        return None
    return sum(runs) / len(runs)


def main():
    print("═══ Part 19 脚本 03：τ-mini（τ-bench 微缩，Qwen2.5-0.5B 实测）═══")
    print(f"  tasks={len(TASKS)}, runs/task={R}, MAX_EXCHANGES={MAX_EXCHANGES}（安全上限）\n")
    print("── 政策文档（agent system prompt，节选）──")
    for line in POLICY.splitlines()[:4]:
        print(f"  {line}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n── 加载 {MODEL_NAME}（{device}）──")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)
    hf_model.eval()
    model = {"tokenizer": tokenizer, "hf_model": hf_model, "device": device}

    # ── 第一次运行逐轨迹展示（教学用），随后 R 次独立重复 ──
    all_runs = {t["id"]: [] for t in TASKS}
    for task in TASKS:
        print(f"\n── 任务 {task['id']}（第 1 次运行，逐轨迹）──")
        p, why, trace = run_task(model, task, verbose=True)
        print(f"  ⇒ pass={p}（{why}）")
        all_runs[task["id"]].append(p)
        for r in range(1, R):
            p, why, _ = run_task(model, task)
            all_runs[task["id"]].append(p)

    # ── 汇总：逐任务 pass^1 + 总体 ──
    print("\n═══ pass^1 汇总（每任务独立运行 %d 次，温度采样 temperature=0.7）═══" % R)
    flat = []
    for tid, runs in all_runs.items():
        flat.extend(runs)
        print(f"  {tid:<28} runs={runs} → pass^1 = {pass_at_1(runs):.2f}")
    print(f"  {'OVERALL':<28} {'':<16} → pass^1 = {pass_at_1(flat):.2f}")

    print(f"""
  解读（对照教程 01/02 章）：
  - 判分只看 DB 终态与调用序列（verify()），不信 agent 的"话术"——0.5B 常把
    拒绝说得很客气但手上把违规操作做了（或反之），话术与行为脱钩正是 agent
    评测要看 DB 的原因；
  - pass^1 是单次通过率：独立重复 R 次的均值。τ-bench 还定义 pass^k（k 次全过
    才算过，度量一致性）——k 越大越苛刻，同一模型 pass^8 通常远低于 pass^1；
  - 真实 τ-bench 的用户由 LLM 扮演（更真实但引入评测方差），本玩具用剧本换
    确定性——两种取舍在教程 02 章"评测现状"一节展开；
  - τ²-bench（arXiv 2506.07982）把环境升级为"双控制"（用户侧也有工具），
    对 agent 的多轮协作要求更高。
  ⚠️ 社区经验：τ-bench 上游曾修复评分 bug，修正后部分模型榜单分数随之变化——
  复现 agent 榜单必须锁定基准版本（commit hash），"评分逻辑一行改动 = 榜单重排名"。
  💡 面试："怎么评 agent？"——任务级指标（pass^1/pass^k）+ DB/环境终态校验 +
  政策合规（不能只看对错，还要看违规）+ 评测方差控制（锁版本/锁 seed/多次重复）。""")


if __name__ == "__main__":
    main()
