#!/usr/bin/env python3
"""
Part 8 - 脚本 09: 推理模型的手写机制——R1 两阶段 + test-time compute
目标：① 用 SFT 训一个能做单位数加法的玩具模型（作为"有能力的模型"基线）；
      ② 实测 test-time compute 的 self-consistency：n↑ → 准确率↑（算力换准确率）。
对应教程：tutorial/09_reasoning_models.md
运行（GPU ~30 秒 / CPU ~2 分钟）：python 09_reasoning_models.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══ 玩具任务：单位数加法 ═══
VOCAB = (["<pad>", "<think>", "</think>", "<answer>", "</answer>", "+", "="]
         + [str(i) for i in range(19)])
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
V = len(VOCAB)
ANS_S = STOI["<answer>"]


def make_task(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    prompt_ids = [STOI[str(a)], STOI["+"], STOI[str(b)], STOI["="]]
    return prompt_ids, a + b


class TinyLM(nn.Module):
    def __init__(self, vocab, d=128, n_layer=3, n_head=4, ctx=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([_Block(d, n_head, ctx) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


class _Block(nn.Module):
    def __init__(self, d, n_head, ctx):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.register_buffer('mask', torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T])
        return x + self.mlp(self.ln2(x + a))


@torch.no_grad()
def sample_ids(model, prompt_ids, max_new=4, temperature=1.0, n_samples=1):
    """批量采样 n_samples 条完整生成。"""
    model.eval()
    outputs = []
    for _ in range(n_samples):
        ids = list(prompt_ids)
        for _ in range(max_new):
            logits = model(torch.tensor([ids], device=DEVICE))[:, -1, :] / temperature
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
            ids.append(nxt)
        outputs.append(list(ids))
    return outputs


def extract_answer(ids):
    """找 ANS_S token 后的第一个数字 token。"""
    try:
        a0 = ids.index(ANS_S) + 1
        tok_str = ITOS[ids[a0]]
        return int(tok_str) if tok_str.isdigit() else None
    except (ValueError, IndexError):
        return None


def main():
    rng = random.Random(42)
    print("═══ 推理模型：R1 两阶段 + test-time compute ═══")
    print(f"  device={DEVICE}\n")

    # ── SFT：教模型"看到 a+b= 就输出答案"──
    print("[SFT] 训练单位数加法（5000 步）...")
    model = TinyLM(V).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(5000):
        rng_local = random.Random(step)
        a, b = rng_local.randint(1, 9), rng_local.randint(1, 9)
        result = a + b
        target = [STOI[str(a)], STOI["+"], STOI[str(b)], STOI["="], ANS_S, STOI[str(result)]]
        X = torch.tensor([target[:-1]], device=DEVICE)
        Y = torch.tensor([target[1:]], device=DEVICE)
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  step {step}: loss = {loss.item():.4f}")
    print(f"  SFT 完成，loss = {loss.item():.4f}\n")

    # ── test-time compute：self-consistency ──
    print("[self-consistency] 采 N 条轨迹 → 众数答案 vs 单次采样：")
    trials = 50
    for n in (1, 4, 8):
        correct = 0
        for _ in range(trials):
            rng_t = random.Random(rng.randint(0, 99999))
            a, b = rng_t.randint(1, 9), rng_t.randint(1, 9)
            result = a + b
            prompt = [STOI[str(a)], STOI["+"], STOI[str(b)], STOI["="]]
            answers = []
            for _ in range(n):
                model.eval()
                ids = list(prompt)
                for _ in range(2):  # answer 长最多 2 位
                    logits = model(torch.tensor([ids], device=DEVICE))[:, -1, :]
                    nxt = int(torch.multinomial(F.softmax(logits, -1), 1).item())
                    ids.append(nxt)
                # 提取答案 token（ANS_S 之后）
                try:
                    a0 = ids.index(ANS_S) + 1
                    tok = ITOS[ids[a0]]
                    if tok.isdigit():
                        answers.append(int(tok))
                except (ValueError, IndexError):
                    pass
            if answers:
                from collections import Counter
                top = Counter(answers).most_common(1)[0][0]
                correct += int(top == result)
        print(f"  n={n}: 准确率 {correct / trials:.2%}")
    print("""
═══ 预期与解读 ═══
  - self-consistency：n↑ → 众数答案更稳（多次采样+多数投票 = 算力换准确率）
  💡 R1 四阶段（教程 09 章详述）：cold start SFT → 推理 RL → 拒绝采样 SFT → 全场景 RL
  奖励 = 规则准确率 + 格式分（非 NM，防 reward hacking）""")


if __name__ == '__main__':
    main()
