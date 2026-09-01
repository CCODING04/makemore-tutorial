#!/usr/bin/env python3
"""
Part 11 - 脚本 02: 玩具 GRPO 训练循环（脚本 01 的三零件拼成完整 RL 循环）

目标：
  脚本 01 验证了三个零件（奖励函数 / 组内优势 / k3 KL）；本脚本把它们装配成
  一个真正会学习的 GRPO 训练循环（玩具规模，CPU <5 秒（实测约 1.3s）），并展示三件事：
    (a) 平均奖励随训练上升（RLVR 的 0/1 信号足以学会任务）
    (b) 全对/全错组的优势全零、不产生梯度——GRPO 天然跳过已掌握/学不会的样本
    (c) 与 BC（行为克隆，需要标准答案标签）基线对比——GRPO 只需要"验证器"

  这正是 verl 里 actor_rollout_ref 三角色 + adv_estimator=grpo +
  algorithm.kl_penalty 配置背后的代码语义（02 章 Docker 实操的手写对照物）。

对应教程：tutorial/01_handwritten_to_verl.md
运行：python 02_grpo_toy_train.py（CPU 即可，<5 秒（实测约 1.3s））

数据流追踪（每个 step）：
  prompts: (P,) int                        # P 道题，每题有隐藏答案 target[p]
    ↓ policy 前向（Embedding → Linear）
  logits: (P, V)                           # 每道题在 V 个候选数字上的打分
    ↓ Categorical 采样 G 次（= rollout）
  actions: (G, P) int                      # G 次采样 × P 道题；每列是一道题的组
    ↓ 拼成 "\\boxed{d}" 字符串 → 规则奖励（脚本 01 同款抽取链）
  rewards: (G, P) float                    # 0/1
    ↓ 转置成 (P, G) 后组内标准化（每行一个组，= adv_estimator=grpo）
  advs: (G, P) float                       # 转回与 logp 对齐；每列和=0，全对/全错列全 0
    ↓ loss = -(logp * adv).mean() + β * k3_kl
  标量 loss → backward → 更新 policy
"""

import math
import re
import sys

import torch
import torch.nn as nn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 可验证奖励（脚本 01 的 gsm8k_reward 同款抽取链，内联以保持自包含）
# ═══════════════════════════════════════════════════════════════════════════════

def math_reward(response: str, ground_truth: str) -> float:
    """RLVR 规则奖励：\\boxed{} → '#### x' → 最后一个数字（与脚本 01 相同）。"""
    for pat in (r"\\boxed\{(-?[\d,\.]+)\}", r"####\s*(-?[\d,\.]+)",
                r"(-?\d[\d,]*(?:\.\d+)?)"):
        m = re.findall(pat, response, flags=re.I)
        if m:
            try:
                pred = float(m[-1].replace(",", "").rstrip("."))
                return 1.0 if abs(pred - float(ground_truth)) < 1e-4 else 0.0
            except ValueError:
                continue
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GRPO 三零件（脚本 01 同款：批量组内优势 + k3 KL）
# ═══════════════════════════════════════════════════════════════════════════════

def group_advantages(rewards_per_prompt, eps=1e-6):
    """批量版组内优势：输入 (P, G)，每行独立标准化，全同行 → 全 0。"""
    advs = []
    for group in rewards_per_prompt:
        n = len(group)
        mean = sum(group) / n                             # shape: scalar
        var = sum((x - mean) ** 2 for x in group) / n     # shape: scalar
        std = max(var ** 0.5, eps)                        # eps 兜底防除零
        advs.append([(x - mean) / std for x in group])    # shape: (G,)
    return advs


def zero_adv_groups(reward_matrix):
    """全同奖励组下标（这些组优势全零、无梯度——本轮浪费的 rollout）。"""
    return [i for i, g in enumerate(reward_matrix) if max(g) == min(g)]


def k3_kl(logp_ref, logp_new):
    """k3 估计器（脚本 01 同款签名）：mean(exp(d) - d - 1)，d = logp_ref - logp_new。"""
    return sum(math.exp(a - b) - (a - b) - 1
               for a, b in zip(logp_ref, logp_new)) / len(logp_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 玩具任务与策略模型
# ═══════════════════════════════════════════════════════════════════════════════

class ToyPolicy(nn.Module):
    """P 道题的"答题策略"：Embedding(P, d) → Linear(d, V)。

    logits[p, v] = 策略认为第 p 题答案是 v 的倾向；每题独立 softmax 后采样。
    这是一个最小的"语言模型"替身——真实 LLM 对 token 序列建模，
    这里单 token 就足以演示 GRPO 的全部数学。
    """

    def __init__(self, n_prompts, vocab, d_model=16):
        super().__init__()
        self.embed = nn.Embedding(n_prompts, d_model)   # (P,) → (P, d_model)
        self.head = nn.Linear(d_model, vocab)           # (P, d_model) → (P, V)

    def forward(self, prompt_ids):
        # prompt_ids: (P,) int → logits: (P, V) float
        return self.head(self.embed(prompt_ids))


def rollout_and_score(policy, prompt_ids, targets, group_size):
    """一次 rollout：采样 G 个回答 → 拼字符串 → 规则奖励打分（字符串进、分数出）。

    Returns:
        actions: (G, P) int     每列是一道题的 G 个回答（digit）
        logp:    (G, P) float   采样时的对数概率（用于 loss）
        rewards: (G, P) float   0/1
    """
    logits = policy(prompt_ids)                          # (P, V)
    dist = torch.distributions.Categorical(logits=logits)   # batch_shape: (P,)
    actions = dist.sample((group_size,))                 # (G, P)：G 次采样 × P 道题
    logp = dist.log_prob(actions)                        # (G, P)
    rewards = torch.zeros(actions.shape, dtype=torch.float)   # (G, P)
    for g in range(actions.shape[0]):
        for p in range(actions.shape[1]):
            # RLVR 打分走真实接口：模型"说话"，验证器"听话"（脚本 01 同一条链）
            resp = f"the answer is \\boxed{{{int(actions[g, p])}}}"
            rewards[g, p] = math_reward(resp, str(int(targets[p])))
    return actions, logp, rewards


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 主函数：GRPO 训练循环 + BC 基线对比
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)   # 固定种子：每次运行输出一致

    # ── 配置 ──
    P, V, G = 6, 4, 4       # 6 道题 / 候选数字 0-3 / 每题采 4 个回答（组大小）
    N_STEPS = 60            # 训练步数
    LR, BETA = 0.5, 0.02    # 学习率 / KL 惩罚系数（verl: algorithm.kl_penalty）

    prompt_ids = torch.arange(P)                       # (P,)
    targets = torch.tensor([2, 0, 3, 1, 2, 0])         # (P,) 每题隐藏答案

    policy = ToyPolicy(P, V)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    # 参考策略 = 初始策略的冻结副本（= verl 的 ref 角色；KL 只防漂移不参与学习）
    ref_policy = ToyPolicy(P, V)
    ref_policy.load_state_dict(policy.state_dict())
    for p_ in ref_policy.parameters():
        p_.requires_grad_(False)

    print("═══ 玩具 GRPO 训练循环（脚本 01 三零件的完整装配）═══\n")
    print(f"任务：{P} 道猜数字题（候选 0-{V - 1}），每题采 G={G} 个回答")
    print(f"配置：steps={N_STEPS}, lr={LR}, KL 系数 β={BETA}\n")

    # ── [1] GRPO 训练循环 ──
    print("[1] GRPO 训练循环（rollout → 规则奖励 → 组内优势 → KL 惩罚 → 更新）:")
    print("    数据流: prompts(P,) → logits(P,V) → actions(G,P) → rewards(G,P) → advs(G,P)")
    print()

    hist_avg_reward = []       # 每步平均奖励（观察 (a) 上升趋势）
    n_skip_hist = []           # 每步零梯度组数（观察 (b) 跳过现象）

    for step in range(N_STEPS):
        # ── rollout（= verl 的 rollout 角色）──
        actions, logp, rewards = rollout_and_score(
            policy, prompt_ids, targets, G)             # (G, P) × 3

        # ── advantage（= verl 的 adv_estimator=grpo）──
        # rewards.t(): (G, P) → (P, G)，每行是一道题的 G 个奖励（= 一个组）
        groups = rewards.t().tolist()                   # (P, G)
        adv_t = torch.tensor(group_advantages(groups)).t()    # (P, G) → (G, P) 对齐 logp
        skip_ids = zero_adv_groups(groups)              # 全对/全错组下标

        # ── KL（= verl 的 ref 角色 + kl_penalty）──
        with torch.no_grad():
            ref_logits = ref_policy(prompt_ids)         # (P, V)
        ref_logp = torch.distributions.Categorical(
            logits=ref_logits).log_prob(actions)        # (G, P) 冻结，无梯度
        d_t = ref_logp - logp                           # (P, G) 可反传
        kl_pen = (d_t.exp() - d_t - 1).mean()           # k3 的 torch 形态，进 loss
        # 与脚本 01 的纯 math 版互验（同一公式、两种实现，应逐位一致）
        kl_math = k3_kl(ref_logp.flatten().tolist(),
                        logp.detach().flatten().tolist())
        assert abs(kl_pen.item() - kl_math) < 1e-6, "torch 版与 math 版 k3 不一致"

        # ── loss & 更新：-(logp * adv).mean() + β * k3 ──
        pg_loss = -(logp * adv_t).mean()                # 零优势项自动无贡献
        loss = pg_loss + BETA * kl_pen
        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_r = rewards.mean().item()
        hist_avg_reward.append(avg_r)
        n_skip_hist.append(len(skip_ids))

        # 分段 debug 打印：前 10 步密集（看学习曲线爬升）+ 之后每 15 步
        if step in (0, 4, 9) or (step + 1) % 15 == 0:
            n_all_right = sum(1 for r in groups if min(r) == 1.0)
            n_all_wrong = len(skip_ids) - n_all_right
            print(f"    step {step:>3}: 平均奖励={avg_r:.2f}  "
                  f"零梯度组={len(skip_ids)}/{P}（全对{n_all_right}+全错{n_all_wrong}）  "
                  f"KL={kl_pen.item():.3f}")

    print()
    print("    零梯度组数量变化（每 10 步）:",
          [n_skip_hist[i] for i in range(0, N_STEPS, 10)])

    # ── [2] (b) 全对/全错组被跳过的现场还原 ──
    print()
    print("[2] 零梯度组现场（训练后策略变自信；全对组=已掌握，全错组=卡死，优势都全 0）:")
    with torch.no_grad():
        _, _, rewards2 = rollout_and_score(policy, prompt_ids, targets, G)
    groups2 = rewards2.t().tolist()                     # (P, G)
    advs2 = group_advantages(groups2)
    skip2 = zero_adv_groups(groups2)
    for p in range(min(3, P)):
        r = groups2[p]
        a = [round(x, 2) for x in advs2[p]]
        tag = "← 零梯度组（跳过）" if p in skip2 else ""
        print(f"    prompt{p} rewards={r} → adv={a} {tag}")

    grpo_init, grpo_final = hist_avg_reward[0], hist_avg_reward[-1]
    print(f"\n    (a) 平均奖励: 初始 {grpo_init:.2f} → 最终 {grpo_final:.2f}")
    assert grpo_final > grpo_init + 0.15, \
        f"FAIL: 平均奖励应上升，{grpo_init:.2f} → {grpo_final:.2f}"

    # ── [3] BC 基线对比（有标准答案标签时的监督学习上限）──
    print()
    print("[3] BC（行为克隆）基线：同样的模型与步数，直接用标准答案做交叉熵:")
    torch.manual_seed(42)   # 重新播种 → 与 GRPO 完全相同的初始化，公平对比
    bc_policy = ToyPolicy(P, V)
    bc_opt = torch.optim.Adam(bc_policy.parameters(), lr=LR)
    bc_hist = []
    for step in range(N_STEPS):
        logits = bc_policy(prompt_ids)                  # (P, V)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()
        acc = (logits.argmax(dim=-1) == targets).float().mean().item()
        bc_hist.append(acc)
        if step in (0, N_STEPS - 1) or (step + 1) % 15 == 0:
            print(f"    step {step:>3}: 准确率={acc:.2f}")

    # ── 总结 ──
    grpo_tail = hist_avg_reward[-N_STEPS // 10:]   # 末 1/10 步的稳定平均
    bc_tail = bc_hist[-N_STEPS // 10:]
    print(f"""
═══ 总结 ═══

  (a) 平均奖励上升: GRPO {grpo_init:.2f} → {grpo_final:.2f}
      （奖励只来自规则验证器——模型没有见过任何标准答案标签）
  (b) 零梯度组的两种命运:
      - 全对组 = 已掌握 → 跳过是 feature（算力集中在有区分度的题上）
      - 全错组 = 永远学不会 → 跳过是盲区（本例就有 1 题卡在全错组，
        最终停在 {grpo_final:.2f} 而非 1.00——DAPO 的 dynamic sampling
        就是把这类组过滤掉重新采样，专门治这个病）
  (c) 末段平均: GRPO {sum(grpo_tail) / len(grpo_tail):.2f} vs BC {sum(bc_tail) / len(bc_tail):.2f}
      BC 有标签、信号稠密，是玩具任务的上限；GRPO 只要"能验证对错"，
      在没有标签、只有验证器（数学/代码/格式检查）的真实 RLVR 场景里依然可用。

  与 verl 的映射（02 章 Docker 实操时对照）：
    本脚本 rollout_and_score   → actor_rollout_ref.rollout（vLLM 生成）
    本脚本 math_reward         → custom reward function
    本脚本 group_advantages    → algorithm.adv_estimator=grpo
    本脚本 ref_policy + k3     → ref 角色 + algorithm.kl_penalty
    单进程 for 循环            → Ray 单控制器数据流

💡 面试要点：
   - RLVR 的奖励为什么够用？→ 对错可机器验证时，0/1 信号 + 组内对比即可学习
   - 全对/全错组为什么无梯度？→ 组内 std=0 → 优势全 0 → logp*adv 恒 0
   - KL 惩罚防什么？→ 策略漂离 SFT 起点太远（reward hacking 的常见前兆）
""")


if __name__ == '__main__':
    main()
