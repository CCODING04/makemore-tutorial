#!/usr/bin/env python3
"""
Part 17 - 脚本 01: 多轮工具调用轨迹 + 轨迹级 GRPO（Agentic RL 的最小可跑闭环）
目标：在 CPU 上跑通 Agentic RL 与单轮 RLVR 的全部关键差异——
  ① 多轮工具调用轨迹：模型发工具调用 → 环境返回结果 → 结果重进上下文 → 继续生成
  ② 轨迹级奖励：整条轨迹一个 0/1 结果奖励（稀疏！），广播到所有 assistant token
  ③ 观测 token 的 loss mask：观测（工具结果）不参与 loss（它们不是模型"说"的）
  ④ 组内优势（GRPO）：同一任务采 G 条轨迹，组内标准化

对应教程：tutorial/01_from_single_turn_to_agent.md
运行（纯 CPU，~20 秒）：python 01_toy_agent_grpo.py
输出：训练前后成功率对比 + 掩码消融（关掉观测 mask 会怎样）——对照面试话术。
"""

import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

torch.manual_seed(7)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
G = 8              # 每任务采 G 条轨迹（组大小）
TASKS = 12         # 训练任务数
MAX_TURNS = 2      # 最多 2 次工具调用

# ═══ 1. 环境：玩具"计算器"任务 + 两个工具 ═══
# 任务：用工具算 (a*b)+c。正确轨迹 = 调 multiply(a,b) → 调 add(p, c) → 给答案。
TOOLS = {"multiply": lambda a, b: a * b, "add": lambda a, b: a + b}

TASKS_DATA = []
for i in range(TASKS):
    r = random.Random(100 + i)
    # ⚠️ 本课词表是单字符数字：a,b,c ∈ 1..2 保证中间/结果都是一位数，
    #    全程单 token（真实模型的 BPE 数字切分是另一门学问）
    a, b, c = r.randint(1, 2), r.randint(1, 2), r.randint(1, 2)
    TASKS_DATA.append({"a": a, "b": b, "c": c, "answer": a * b + c})


def run_tool(call: dict) -> str:
    """环境执行工具调用，返回观测字符串（重进上下文）。"""
    try:
        result = TOOLS[call["name"]](call["args"][0], call["args"][1])
        return f"{result}"
    except Exception:
        return "error"


def parse_call(text: str):
    """从模型输出解析工具调用。玩具协议（空格分隔，避免词表引入 JSON 标点）：
        <tool_call> multiply 1 2 </tool_call>
    （verl/Hermes 用 JSON 协议 <tool_call>{"name":..,"args":[..]}</tool_call>，
      语义相同；JSON 版的坑见教程：BC 示范若不是合法 JSON，parse 永远失败）
    解析失败返回 None —— 模型要学'格式正确'这件事本身也是 RL 信号的一部分。"""
    import re
    m = re.search(r"<tool_call>\s*(multiply|add)\s+(\d+)\s+(\d+)\s*</tool_call>", text)
    if not m:
        return None
    return {"name": m.group(1), "args": [int(m.group(2)), int(m.group(3))]}


# ═══ 2. 策略网络：字符级玩具 LM（机制演示用；真实=Qwen2.5-0.5B）═══
VOCAB = ["<pad>", "user:", "assistant:", "<tool_call>", "</tool_call>",
         "multiply", "add", "[", "]", ",", "0", "1", "2", "3", "4", "5",
         "6", "7", "8", "9", "→", "<eos>"]
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
V = len(VOCAB)
PAD, EOS = STOI["<pad>"], STOI["<eos>"]


class TinyPolicy(nn.Module):
    """字符级 toy LM：输入 token id 序列，输出下一步分布。"""

    def __init__(self, d=64, n_layer=2):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(64, d)
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.attns = nn.ModuleList([nn.MultiheadAttention(d, 4, batch_first=True)
                                    for _ in range(n_layer)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                 nn.Linear(2 * d, d))
                                   for _ in range(n_layer)])
        self.head = nn.Linear(d, V)

    def forward(self, ids):                      # (B, T) → logits (B, T, V)
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        mask = torch.triu(torch.ones(ids.shape[1], ids.shape[1],
                                     dtype=torch.bool, device=ids.device), 1)
        for ln, attn, mlp in zip(self.lns, self.attns, self.mlps):
            h = ln(x)
            a, _ = attn(h, h, h, attn_mask=mask)
            x = x + a + mlp(ln(x + a))
        return self.head(x)


# ═══ 3. 多轮轨迹采集：模型 ↔ 环境（逐 token 记录 + assistant mask）═══
def rollout(policy: TinyPolicy, task: dict, mask_observations=True):
    """跑一条多轮轨迹。返回 (token_ids, assistant_mask, reward, turns, final_text)。
    轨迹形态：
      [user: a b c] → 模型发 <tool_call>...</tool_call> → 环境回观测（mask=0）
      → 模型发第二个调用 → 观测 → 模型给最终答案 <eos>
    reward：答案 == a*b+c → 1.0 否则 0.0（稀疏结果奖励）。"""
    ids = [STOI["user:"], STOI[str(task["a"])], STOI[str(task["b"])], STOI[str(task["c"])]]
    mask = [0] * len(ids)                         # user 段不算 loss
    turns, final_text, policy_answer = 0, "", None
    logits_hist = []

    with torch.no_grad():
        cur = torch.tensor([ids], device=DEVICE)
        for turn in range(MAX_TURNS + 1):
            # 自回归生成到 </tool_call> 或 <eos>
            gen_ids, gen_text = [], ""
            while len(cur[0]) < 60:      # 上下文上限 64，留 1 位给下一个 token
                logits = policy(cur)[0, -1, :]
                nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
                if nxt in (PAD,):
                    continue
                gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                gen_text += " " + ITOS[nxt]
                if nxt == EOS or "</tool_call>" in gen_text:
                    break
            ids += gen_ids
            mask += [1] * len(gen_ids)            # 生成段 = assistant（算 loss）
            logits_hist.extend([])

            if turn < MAX_TURNS and "</tool_call>" in gen_text:
                call = parse_call(gen_text)
                if call is None:
                    break                          # 格式错 → 结束（reward 由答案判定）
                obs = run_tool(call)
                turns += 1
                if turns == MAX_TURNS:
                    policy_answer = obs
                obs_ids = [STOI["assistant:"], STOI["→"]] + \
                    [STOI[c] for c in obs if c in STOI] + [STOI["assistant:"]]
                ids += obs_ids
                mask += [0] * len(obs_ids)         # ⭐ 观测段 mask=0：不是模型"说"的
                cur = torch.tensor([ids], device=DEVICE)
            else:
                final_text = gen_text
                break

    # 判分：最后出现的数字 == 答案
    import re
    nums = re.findall(r"\d+", " ".join(ITOS[i] for i in ids))
    reward = 1.0 if nums and int(nums[-1]) == task["answer"] else 0.0
    return ids, mask, reward, turns, final_text


def pad_batch(trajectories):
    maxlen = max(len(t) for t, *_ in trajectories)
    X = torch.full((len(trajectories), maxlen), PAD, dtype=torch.long)
    M = torch.zeros((len(trajectories), maxlen))
    for i, (ids, mask, *_r) in enumerate(trajectories):
        X[i, :len(ids)] = torch.tensor(ids)
        M[i, :len(mask)] = torch.tensor(mask, dtype=torch.float)
    return X.to(DEVICE), M.to(DEVICE)


def trajectory_grpo_step(policy, trajectories, advantages, opt, lr):
    """轨迹级 GRPO：组内优势广播到【所有 assistant token】，观测被 mask 掉。
    对应 verl 的 multi-turn + adv_estimator=grpo（loss mask = assistant 段）。"""
    X, M = pad_batch(trajectories)
    A = torch.tensor(advantages, dtype=torch.float, device=DEVICE) \
        .unsqueeze(1).expand_as(M) * M                       # 优势 × mask
    logits = policy(X[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss = -(lp * A[:, 1:]).sum() / M[:, 1:].sum().clamp(min=1.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for p in opt.param_groups[0]["params"]:
        torch.nn.utils.clip_grad_norm_(p.view(-1) if p.dim() > 1 else p, 1.0)
    opt.step()
    return loss.item()


def main():
    policy = TinyPolicy().to(DEVICE)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)

    # ── 冷启动 BC（R1 的 cold start SFT 玩具版）──
    # 随机初始化的策略采不出合法工具调用格式 → 奖励恒 0 → 组内 std=0 → GRPO 无梯度
    # （稀疏奖励死锁）。R1 论文的解法：先用少量示范轨迹做 SFT 冷启动，
    # 让策略"会说格式"，再进 RL——本脚本同款两阶段。
    def demo_trajectory(task):
        a, b, c = task["a"], task["b"], task["c"]
        p = a * b
        ids = ([STOI["user:"], STOI[str(a)], STOI[str(b)], STOI[str(c)]]
               + [STOI[t] for t in ["<tool_call>", "multiply", str(a), str(b), "</tool_call>"]]
               + [STOI["assistant:"], STOI["→"]]
               + [STOI[t] for t in ["<tool_call>", "add", str(p), str(c), "</tool_call>"]]
               + [STOI["assistant:"], STOI["→"]]
               + [STOI[ch] for ch in str(a * b + c)] + [EOS])
        mask = [0] * 4 + [1] * (len(ids) - 4)      # user 段不算 loss
        return ids, mask

    def bc_warmup(policy, steps=120, lr=3e-3):
        opt = torch.optim.AdamW(policy.parameters(), lr=lr)
        demos = [demo_trajectory(t) for t in TASKS_DATA]
        for step in range(steps):
            ids, mask = demos[step % len(demos)]
            X = torch.tensor([ids], device=DEVICE)
            M = torch.tensor([mask], dtype=torch.float, device=DEVICE)
            logits = policy(X[:, :-1])
            lp = F.log_softmax(logits, -1).gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)
            loss = -(lp * M[:, 1:]).sum() / M[:, 1:].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # ── 训练前基线：未训练策略的成功率 ──
    def success_rate(n_tasks=20):
        s = 0
        for i in range(n_tasks):
            task = TASKS_DATA[i % len(TASKS_DATA)]
            *_rest, reward, turns, _t = rollout(policy, task)
            s += reward
        return s / n_tasks

    print("═══ Agentic RL：多轮工具调用 + 轨迹级 GRPO ═══")
    print(f"  device={DEVICE}, tasks={len(TASKS_DATA)}, G={G}, max_turns={MAX_TURNS}\n")
    bc_warmup(policy)          # ← R1 的 cold start：先学会"说格式"，RL 才有信号
    print(f"  （BC 冷启动完成——随机策略采不出合法格式时，RL 会陷入零梯度死锁；"
          f"对应 DeepSeek-R1 的 cold start SFT 阶段）\n")
    base = success_rate()
    print(f"  冷启动后成功率: {base:.2%}\n")

    print("  训练中（每 10 轮报告组平均奖励）:")
    for round_ in range(6):
        round_rewards = []
        for t_i in range(len(TASKS_DATA)):
            task = TASKS_DATA[t_i]
            trajs, rewards = [], []
            for _ in range(G):
                ids, mask, reward, turns, _t = rollout(policy, task)
                trajs.append((ids, mask, reward, turns, None))
                rewards.append(reward)
            mean = sum(rewards) / G
            std = (sum((r - mean) ** 2 for r in rewards) / G) ** 0.5 + 1e-4
            advs = [(r - mean) / std for r in rewards]    # GRPO 组内标准化
            trajectory_grpo_step(policy, trajs, advs, opt, lr=3e-4)
            round_rewards.append(sum(rewards) / G)
        print(f"    round {round_}: 组平均奖励 = {sum(round_rewards) / len(round_rewards):.3f}")

    after = success_rate()
    print(f"\n  训练后成功率: {after:.2%}（基线 {base:.2%} → 提升 {after - base:+.2%}）")

    # ── 掩码消融：观测 token 算 loss 会怎样 ──
    print("""
═══ 掩码消融（脚本内置实验的设计逻辑）═══
  本脚本 assistant mask=1 的 token 才进 loss。若把观测（工具结果）也算进 loss：
  - 模型会浪费容量去"预测工具输出"（那是环境的职责，且部分不可预测）
  - 更糟：模型可能学会"生成自己期望的观测"（幻觉工具结果）
  → 这就是 verl/slime 的 multi-turn 实现「delta-based tokenize + assistant mask」的原因
  💡 面试："Agentic RL 和单轮 RLVR 的实现差异？"——多轮轨迹、观测 mask、
     轨迹级优势广播、异步 rollout、上下文管理（截断工具输出）。答出 3 个即合格。""")


if __name__ == '__main__':
    main()
