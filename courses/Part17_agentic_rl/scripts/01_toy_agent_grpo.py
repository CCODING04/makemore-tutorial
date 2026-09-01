#!/usr/bin/env python3
"""
Part 17 - 脚本 01: 多轮工具调用轨迹 + 轨迹级 GRPO + 真实掩码消融
目标：跑通 Agentic RL 与单轮 RLVR 的全部关键差异——
  ① 多轮工具调用轨迹：模型发工具调用 → 环境返回结果 → 结果重进上下文 → 继续生成
  ② 轨迹级奖励：整条轨迹一个 0/1 结果奖励（稀疏！），广播到所有 assistant token
  ③ 观测 mask（本脚本做【真消融】，不是文字描述）：
       mask_observations=True  → 工具观测段替换为固定 <mask> token：
                                 观测内容不进策略输入，也不进 loss（loss-mask=0）
       mask_observations=False → 观测原样进策略输入——第二次观测本身就是最终答案，
                                 策略可以学会"复读观测"而非内化计算（信息泄漏）
  ④ 组内优势（GRPO）：同一任务采 G 条轨迹，组内标准化

对应教程：tutorial/01_from_single_turn_to_agent.md
运行（GPU ~10-15 秒 / 纯 CPU ~20-60 秒，seed 固定）：python 01_toy_agent_grpo.py
输出：两组同 seed 对照实验的真实数字（实测，RTX GPU / seed=7 / 6 轮 RL）：
     开卷·train        A(mask)=99.0%   B(泄漏)=96.9%   ← 训练组合已饱和
     开卷·holdout      A=12.5%         B=3.1%          ← 泄漏组未见组合崩塌
     闭卷·train        A=43.8%         B=10.4%         ← 工具拿走后泄漏组现形
     闭卷·holdout      A=25.0%         B=0.0%
     机理：泄漏组的最终答案学的是"复读观测"（第二次观测=答案），不是 (a,b,c)
     的函数——详见文末"掩码消融对比"表与解读。
     ⚠️ 设备说明：CPU 与 CUDA 的浮点差异会让采样轨迹分岔，具体数字随设备波动
     （本机 CPU 实测：A 组 31.2%/39.6%/28.1% vs B 组 0%/2.1%/9.4%），但
     "泄漏组在 holdout/闭卷崩塌"的定性结论在两种设备上均稳定复现。
"""

import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
G = 8              # 每任务采 G 条轨迹（组大小）
RL_ROUNDS = 6      # RL 轮数（两组实验相同——唯一变量是 mask_observations）
BC_STEPS = 120     # 冷启动 BC 步数
MAX_TURNS = 2      # 最多 2 次工具调用

# ═══ 1. 环境：玩具"计算器"任务 + 两个工具 ═══
# 任务：用工具算 (a*b)+c。正确轨迹 = 调 multiply(a,b) → 调 add(p, c) → 给答案。
TOOLS = {"multiply": lambda a, b: a * b, "add": lambda a, b: a + b}

# ⚠️ 本课词表是单字符数字：a,b,c ∈ 1..2 保证中间/结果都是一位数，
#    全程单 token（真实模型的 BPE 数字切分是另一门学问）
# 训练/测试任务划分（消融的"泛化探针"）：训练只见 (a,b) ∈ {(1,1),(1,2),(2,2)}，
# 留出 (2,1,·) 两个组合做 holdout——看策略学的是"查表复读"还是"内化计算"。
TRAIN_TASKS = [{"a": a, "b": b, "c": c, "answer": a * b + c}
               for a, b in [(1, 1), (1, 2), (2, 2)] for c in (1, 2)] * 2   # 12 条（6 组合×2）
HOLDOUT_TASKS = [{"a": 2, "b": 1, "c": c, "answer": 2 + c} for c in (1, 2)]
EVAL_TRAIN = TRAIN_TASKS[:6]       # 评测用每组合 1 条


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
    m = re.search(r"<tool_call>\s*(multiply|add)\s+(\d+)\s+(\d+)\s*</tool_call>", text)
    if not m:
        return None
    return {"name": m.group(1), "args": [int(m.group(2)), int(m.group(3))]}


# ═══ 2. 策略网络：字符级玩具 LM（机制演示用；真实=Qwen2.5-0.5B）═══
VOCAB = ["<pad>", "user:", "assistant:", "<tool_call>", "</tool_call>",
         "multiply", "add", "[", "]", ",", "0", "1", "2", "3", "4", "5",
         "6", "7", "8", "9", "→", "<mask>", "<eos>"]
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
V = len(VOCAB)
PAD, EOS, MASK = STOI["<pad>"], STOI["<eos>"], STOI["<mask>"]


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
    """跑一条多轮轨迹（工具真实执行）。返回 (ids, mask, reward, turns)。

    轨迹形态：
      [user: a b c] → 模型发 <tool_call>...</tool_call> → 环境回观测（mask=0）
      → 模型发第二个调用 → 观测 → 模型给最终答案 <eos>

    mask_observations=True  → 观测数字替换为 <mask>：观测内容既不进策略输入，
                              也不进 loss（观测段 loss-mask 恒为 0）
    mask_observations=False → 观测数字原样进策略输入（第二次观测=最终答案——泄漏）
    奖励按【真实执行】轨迹（real_ids，观测永远是真实数字）判分，两种模式判分一致。
    """
    ids = [STOI["user:"], STOI[str(task["a"])], STOI[str(task["b"])], STOI[str(task["c"])]]
    mask = [0] * len(ids)                         # user 段不算 loss
    real_ids = list(ids)                          # 真实轨迹（判分用）
    turns = 0

    with torch.no_grad():
        cur = torch.tensor([ids], device=DEVICE)
        for turn in range(MAX_TURNS + 1):
            # 自回归生成到 </tool_call> 或 <eos>
            gen_ids, gen_text = [], ""
            while len(cur[0]) < 60:      # 上下文上限 64，留余量
                logits = policy(cur)[0, -1, :]
                nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
                if nxt == PAD:
                    continue
                gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                gen_text += " " + ITOS[nxt]
                if nxt == EOS or "</tool_call>" in gen_text:
                    break
            ids += gen_ids
            real_ids += gen_ids
            mask += [1] * len(gen_ids)            # 生成段 = assistant（算 loss）

            if turn < MAX_TURNS and "</tool_call>" in gen_text:
                call = parse_call(gen_text)
                if call is None:
                    break                          # 格式错 → 结束（reward 由答案判定）
                obs = run_tool(call)               # 环境真实执行
                turns += 1
                # 观测段：wrapper(assistant: →) + 结果数字。mask 模式下数字 → <mask>
                obs_content = [MASK if mask_observations else STOI[d]
                               for d in obs if d in STOI]
                obs_ids = [STOI["assistant:"], STOI["→"]] + obs_content
                real_obs = [STOI[d] for d in obs if d in STOI]
                ids += obs_ids
                real_ids += [STOI["assistant:"], STOI["→"]] + real_obs
                mask += [0] * len(obs_ids)         # ⭐ 观测段 mask=0：不进 loss
                cur = torch.tensor([ids], device=DEVICE)
            else:
                break

    # 判分：真实轨迹里最后出现的数字 == 答案
    # 📝 玩具判分漏洞（教程 01 章有专注）：第二次观测本身=最终答案，可"冒充"
    #    最终回答——真实 RLVR 用格式约束/工具协议/答案位置锚定避免
    nums = re.findall(r"\d+", " ".join(ITOS[i] for i in real_ids))
    reward = 1.0 if nums and int(nums[-1]) == task["answer"] else 0.0
    return ids, mask, reward, turns


def demo_trajectory(task: dict, mask_observations: bool):
    """BC 示范轨迹——与 rollout 完全同构（观测段同样按模式替换），保证冷启动
    教的格式和 RL 采的轨迹一致。观测段 loss-mask=0（示范只教 assistant 段）。"""
    a, b, c = task["a"], task["b"], task["c"]
    p, ans = a * b, a * b + c

    def obs_seg(digit: str):
        d = MASK if mask_observations else STOI[digit]
        return [STOI["assistant:"], STOI["→"], d], [0, 0, 0]

    obs1, m1 = obs_seg(str(p))
    obs2, m2 = obs_seg(str(ans))
    call1 = [STOI[t] for t in ["<tool_call>", "multiply", str(a), str(b), "</tool_call>"]]
    call2 = [STOI[t] for t in ["<tool_call>", "add", str(p), str(c), "</tool_call>"]]
    tail = [STOI[str(ans)], EOS]                  # 最终答案 + 结束符
    ids = ([STOI["user:"], STOI[str(a)], STOI[str(b)], STOI[str(c)]]
           + call1 + obs1 + call2 + obs2 + tail)
    mask = [0] * 4 + [1] * len(call1) + m1 + [1] * len(call2) + m2 + [1] * len(tail)
    return ids, mask


def pad_batch(trajectories):
    maxlen = max(len(ids) for ids, _m in trajectories)
    X = torch.full((len(trajectories), maxlen), PAD, dtype=torch.long)
    M = torch.zeros((len(trajectories), maxlen))
    for i, (ids, mask) in enumerate(trajectories):
        X[i, :len(ids)] = torch.tensor(ids)
        M[i, :len(mask)] = torch.tensor(mask, dtype=torch.float)
    return X.to(DEVICE), M.to(DEVICE)


def trajectory_grpo_step(policy, trajectories, advantages, opt):
    """轨迹级 GRPO：组内优势广播到【所有 assistant token】，观测段 mask=0 不进 loss。
    对应 verl 的 multi-turn + adv_estimator=grpo（loss mask = assistant 段）。"""
    X, M = pad_batch(trajectories)
    A = torch.tensor(advantages, dtype=torch.float, device=DEVICE) \
        .unsqueeze(1).expand_as(M) * M                       # 优势 × loss-mask
    logits = policy(X[:, :-1])                               # (B, T-1, V)
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1) 取中目标 token
    loss = -(lp * A[:, 1:]).sum() / M[:, 1:].sum().clamp(min=1.0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # 全局范数裁剪
    opt.step()
    return loss.item()


# ═══ 4. 两种评测：开卷（带工具）/ 闭卷（工具拿走）═══
def open_book_success(policy, tasks, mask_observations, n_samples=16):
    """开卷成功率：正常多轮 rollout（工具可用），与训练同模式。"""
    hit = 0
    for task in tasks:
        for _ in range(n_samples):
            *_r, reward, _turns = rollout(policy, task, mask_observations)
            hit += reward
    return hit / (len(tasks) * n_samples)


def closed_book_success(policy, tasks, n_samples=16):
    """闭卷直答探针：不给工具、不插入任何观测，模型自己生成到 <eos>（上限 24 token）。
    判分 = 生成文本中最后出现的数字 == 答案。没有观测可"复读"——泄漏策略在此现形。"""
    hit = 0
    for task in tasks:
        for _ in range(n_samples):
            ids = [STOI["user:"], STOI[str(task["a"])],
                   STOI[str(task["b"])], STOI[str(task["c"])]]
            cur = torch.tensor([ids], device=DEVICE)
            gen = []
            with torch.no_grad():
                for _step in range(24):
                    logits = policy(cur)[0, -1, :]
                    nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
                    if nxt == PAD:
                        continue
                    gen.append(nxt)
                    if nxt == EOS:
                        break
                    cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            nums = re.findall(r"\d+", " ".join(ITOS[i] for i in gen))
            if nums and int(nums[-1]) == task["answer"]:
                hit += 1
    return hit / (len(tasks) * n_samples)


# ═══ 5. 一组完整实验：BC 冷启动 → 轨迹级 GRPO → 评测 ═══
def run_experiment(mask_observations: bool, seed=SEED):
    """同一 seed 跑一组实验——两组实验的唯一变量是 mask_observations
    （同初始化、同 BC 数据顺序、同 RL 轮数，保证可比）。"""
    torch.manual_seed(seed)
    policy = TinyPolicy().to(DEVICE)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)

    # ── 冷启动 BC（R1 的 cold start SFT 玩具版）──
    # 随机初始化的策略采不出合法工具调用格式 → 奖励恒 0 → 组内 std=0 → GRPO 无梯度
    # （稀疏奖励死锁）。R1 论文的解法：先用少量示范轨迹做 SFT 冷启动，
    # 让策略"会说格式"，再进 RL——本脚本同款两阶段。
    demos = [demo_trajectory(t, mask_observations) for t in TRAIN_TASKS]
    bc_opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    for step in range(BC_STEPS):
        ids, mask = demos[step % len(demos)]
        X = torch.tensor([ids], device=DEVICE)
        M = torch.tensor([mask], dtype=torch.float, device=DEVICE)
        logits = policy(X[:, :-1])
        lp = F.log_softmax(logits, -1).gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = -(lp * M[:, 1:]).sum() / M[:, 1:].sum()
        bc_opt.zero_grad(set_to_none=True)
        loss.backward()
        bc_opt.step()

    base = open_book_success(policy, EVAL_TRAIN, mask_observations)
    curve = []
    for round_ in range(RL_ROUNDS):
        round_rewards = []
        for task in TRAIN_TASKS:
            trajs, rewards = [], []
            for _ in range(G):
                ids, mask, reward, _turns = rollout(policy, task, mask_observations)
                trajs.append((ids, mask))
                rewards.append(reward)
            mean = sum(rewards) / G
            std = (sum((r - mean) ** 2 for r in rewards) / G) ** 0.5 + 1e-4
            advs = [(r - mean) / std for r in rewards]    # GRPO 组内标准化
            trajectory_grpo_step(policy, trajs, advs, opt)
            round_rewards.append(mean)
        curve.append(sum(round_rewards) / len(round_rewards))

    return {
        "mask": mask_observations,
        "base": base,
        "curve": curve,
        "open_train": open_book_success(policy, EVAL_TRAIN, mask_observations),
        "open_holdout": open_book_success(policy, HOLDOUT_TASKS, mask_observations),
        "closed_train": closed_book_success(policy, EVAL_TRAIN),
        "closed_holdout": closed_book_success(policy, HOLDOUT_TASKS),
    }


def main():
    print("═══ Agentic RL：多轮工具调用 + 轨迹级 GRPO + 掩码消融 ═══")
    print(f"  device={DEVICE}, train_tasks={len(TRAIN_TASKS)}(6 组合), "
          f"holdout_tasks={len(HOLDOUT_TASKS)}, G={G}, rl_rounds={RL_ROUNDS}, seed={SEED}\n")

    results = {}
    for mode, tag in [(True, "A: mask=True（观测→<mask>，标准做法）"),
                      (False, "B: mask=False（观测原样进策略输入——泄漏）")]:
        print(f"── 实验组 {tag} ──")
        print("  （BC 冷启动完成——随机策略采不出合法格式时，RL 会陷入零梯度死锁；"
              "对应 DeepSeek-R1 的 cold start SFT 阶段）")
        r = run_experiment(mode)
        print(f"  冷启动后开卷成功率: {r['base']:.1%}")
        for i, v in enumerate(r["curve"]):
            print(f"    round {i}: 组平均奖励 = {v:.3f}")
        print(f"  训练后：开卷 train {r['open_train']:.1%} | 开卷 holdout {r['open_holdout']:.1%}"
              f" | 闭卷 train {r['closed_train']:.1%} | 闭卷 holdout {r['closed_holdout']:.1%}\n")
        results[mode] = r

    a, b = results[True], results[False]
    print("═══ 掩码消融对比（同 seed，唯一变量 = 观测内容是否泄漏进策略输入）═══")
    print(f"  {'指标':<16}{'A: mask=True':>14}{'B: 泄漏(mask=False)':>20}")
    print(f"  {'─' * 52}")
    print(f"  {'开卷·train 任务':<16}{a['open_train']:>13.1%}{b['open_train']:>19.1%}")
    print(f"  {'开卷·holdout 任务':<16}{a['open_holdout']:>13.1%}{b['open_holdout']:>19.1%}")
    print(f"  {'闭卷·train 任务':<16}{a['closed_train']:>13.1%}{b['closed_train']:>19.1%}")
    print(f"  {'闭卷·holdout 任务':<16}{a['closed_holdout']:>13.1%}{b['closed_holdout']:>19.1%}")
    print(f"""
  解读（依据上面真实数字）：
  - 开卷·train：两组都 ~97-99%——6 个训练组合上 BC 冷启动已把任务"背下来"，
    组内几乎全对 → 组内 std≈0 → GRPO 梯度≈0（RL 边际增益小是稀疏奖励 +
    已饱和组的正常现象，A 组 round 5 的 0.948 回落是熵下降波动——02 章
    Echo Trap 的伏笔）。
  - 开卷·holdout / 闭卷：泄漏组全面崩塌。机理：第二次观测本身就是最终答案，
    泄漏组的最优解是"复读前一个观测数字"——答案 token 从未被训练成
    (a,b,c) 的函数；组合一没见过（holdout）或观测一拿走（闭卷）就现形。
    mask 组观测不可见，BC/RL 只能靠 user token 把 a*b+c 内化进参数，
    闭卷 train 仍保住约 {a['closed_train']:.0%}。
  - 对应工业实践：verl/slime 的 multi-turn 是「观测进上下文 + loss mask」
    （模型要读工具结果，但策略梯度不流过观测 token）；本玩具把"隐藏观测内容"
    与"观测不进 loss"合并成一个开关，是为了在秒级看到"信息泄漏 → 学会走捷径"
    的效应。原则不变：环境给的 token 不该承载策略梯度。
  💡 面试："Agentic RL 和单轮 RLVR 的实现差异？"——多轮轨迹、观测 mask、
     轨迹级优势广播、异步 rollout、上下文管理（截断工具输出）。答出 3 个即合格。""")


if __name__ == '__main__':
    main()
