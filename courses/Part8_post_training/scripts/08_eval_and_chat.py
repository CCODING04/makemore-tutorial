#!/usr/bin/env python3
"""
Part 8 - 脚本 8: 评估流水线 + 交互式 Chat
目标：实现完整的 LLM 评估流水线，比较各训练阶段的模型质量。
演示 GSM8K 风格评估、checkpoint 对比、生成质量分析、交互式对话。

覆盖知识点：
  - GSM8K 评估：生成答案 → 提取数字 → 对比金标 → 计算准确率
  - Checkpoint 管理：加载 pretrain/SFT/DPO/PPO/GRPO 各阶段模型
  - 生成质量对比：不同阶段模型在相同 prompt 下的输出差异
  - 温度/Top-K 效果：temperature 控制随机性，top_k 控制候选集
  - 完整训练流水线回顾：从预训练到最终部署的全链路

完整 LLM 训练流水线：
  预训练（续写能力）→ SFT（对话能力）→ 奖励模型（判断好坏）
  → DPO/PPO/GRPO（对齐人类偏好）→ 评估（量化效果）

torch API 速查：
  torch.load() — 加载 checkpoint
  model.eval() — 切换到评估模式（关闭 dropout 等）
  torch.no_grad() — 禁用梯度计算（推理时节省显存）
"""

import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.set_num_threads(1)

# ─── 模式选择 ──────────────────────────────────────────────
CPU_MODE = not torch.cuda.is_available()
if CPU_MODE:
    vocab_size = 256
    n_embed = 64
    n_head = 4
    n_blocks = 2
    context_length = 64
    batch_size = 2
    n_eval_problems = 10
    generate_len = 12
else:
    vocab_size = 50304
    n_embed = 512
    n_head = 8
    n_blocks = 12
    context_length = 512
    batch_size = 4
    n_eval_problems = 100
    generate_len = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


# ─── GPT 模型（内嵌，与 01~07 等价）───────────────────────
class Head(nn.Module):
    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        hs = n_embed // n_head
        self.heads = nn.ModuleList([Head(hs, n_embed, context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_head, n_embed, context_length, vocab_size, n_blocks):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(context_length, n_embed)
        self.blocks = nn.ModuleList([Block(n_head, n_embed, context_length) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos', torch.arange(context_length))

    def forward_hidden(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos[:T])
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)

    def forward(self, idx, targets=None):
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.context_length:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# ─── GSM8K 评估 ────────────────────────────────────────────
def extract_number(text):
    """从文本中提取最后一个数字。

    GSM8K 格式: "#### 42" — 答案在最后
    这里做简化：提取文本中出现的最后一个数字。
    """
    numbers = re.findall(r'-?\d+\.?\d*', str(text))
    if numbers:
        try:
            return float(numbers[-1])
        except Exception:
            return None
    return None


def generate_eval_problems(n_problems):
    """生成合成数学题用于评估。

    CPU: 简单加减法
    GPU: 更多样的算术题

    返回: list of (prompt_text, expected_answer)
    """
    import random
    random.seed(42)  # 固定种子，保证可复现
    problems = []
    for _ in range(n_problems):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(['+', '-'])
        if op == '+':
            answer = a + b
            prompt = f"{a}+{b}="
        else:
            if b > a:
                a, b = b, a
            answer = a - b
            prompt = f"{a}-{b}="
        problems.append((prompt, answer))
    return problems


def evaluate_model(model, stoi, itos, problems, max_tokens=10, temperature=0.8, top_k=10):
    """评估模型在数学题上的准确率。

    流程：
      1. 对每个问题，用模型生成回答
      2. 从回答中提取数字
      3. 与期望答案比较
      4. 计算准确率

    返回: (accuracy, results_list)
    """
    model.eval()
    correct = 0
    total = len(problems)
    results = []

    for prompt_text, expected in problems:
        prompt_ids = [stoi.get(c, 0) for c in prompt_text]
        prompt_tensor = torch.tensor([prompt_ids], device=device)

        with torch.no_grad():
            gen = model.generate(
                prompt_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        resp_ids = gen[0].tolist()[len(prompt_ids):]
        resp_text = ''.join(itos.get(tid, '?') for tid in resp_ids)
        predicted = extract_number(resp_text)

        is_correct = (predicted is not None) and abs(predicted - expected) < 0.01
        if is_correct:
            correct += 1

        results.append({
            'prompt': prompt_text,
            'expected': expected,
            'response': resp_text,
            'predicted': predicted,
            'correct': is_correct,
        })

    accuracy = correct / max(total, 1)
    return accuracy, results


# ─── 数据加载 ──────────────────────────────────────────────
def load_text_data():
    """加载文本数据。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'input.txt')
    if not os.path.exists(data_path):
        text = "Hello world. How are you? I am fine. " * 500
        return text, "合成数据"
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text, "input.txt"


def create_fresh_model():
    """创建一个新的 GPT 模型（不加载 checkpoint）。"""
    text, _ = load_text_data()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    actual_vocab = len(chars)
    model = GPT(n_head, n_embed, context_length, actual_vocab, n_blocks).to(device)
    return model, actual_vocab, stoi


def quick_pretrain(model, stoi, steps=20):
    """快速预训练一个模型。"""
    text, _ = load_text_data()
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        ix = torch.randint(len(data) - context_length, (batch_size,))
        xb = torch.stack([data[i:i + context_length] for i in ix]).to(device)
        yb = torch.stack([data[i + 1:i + context_length + 1] for i in ix]).to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def load_checkpoint(ckpt_path, model_class=GPT):
    """加载 checkpoint，返回 (model, config, stoi)。

    如果 checkpoint 不存在或加载失败，返回 None。
    """
    if not os.path.exists(ckpt_path):
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = model_class(
            config['n_head'], config['n_embed'], config['context_length'],
            config['vocab_size'], config['n_blocks']
        ).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        text, _ = load_text_data()
        chars = sorted(list(set(text)))
        stoi = {c: i for i, c in enumerate(chars)}
        return model, config, stoi
    except Exception as e:
        print(f"  加载失败: {e}")
        return None


# ─── Main ──────────────────────────────────────────────────
def main():
    print("═══ 评估流水线 + 交互式 Chat ═══")
    print(f"  模式: {'CPU' if CPU_MODE else 'GPU'}, device={device}")
    print(f"  架构: embed={n_embed}, heads={n_head}, blocks={n_blocks}, ctx={context_length}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # ── 1. 完整训练流水线回顾 ──
    print(f"\n── Step 1: 完整 LLM 训练流水线 ──")
    print(f"""
  完整 LLM 训练流水线：

  ┌─────────────────────────────────────────────────────────┐
  │  预训练（续写能力）                                      │
  │  目标：学习语言的统计规律                                │
  │  数据：大规模无标注文本                                  │
  │  损失：next-token prediction cross-entropy               │
  │  脚本：02_pretrain.py                                    │
  ├─────────────────────────────────────────────────────────┤
  │  SFT（对话能力）                                         │
  │  目标：学会遵循指令、生成有格式的回答                    │
  │  数据：(instruction, response) 对                        │
  │  损失：response 部分的 cross-entropy                     │
  │  脚本：03_sft.py                                         │
  ├─────────────────────────────────────────────────────────┤
  │  奖励模型（判断好坏）                                    │
  │  目标：学习人类偏好，给回答打分                          │
  │  数据：(chosen, rejected) 偏好对                         │
  │  损失：Bradley-Terry pairwise loss                       │
  │  脚本：04_reward_model.py                                │
  ├─────────────────────────────────────────────────────────┤
  │  DPO / PPO / GRPO（对齐人类偏好）                        │
  │  目标：让模型输出更符合人类期望                          │
  │  DPO:  直接用偏好对优化，无需 reward model               │
  │  PPO:  用 reward model 的分数做 RL                       │
  │  GRPO: 用 group-relative advantage 做 RL                 │
  │  脚本：05_dpo / 06_ppo / 07_grpo                        │
  ├─────────────────────────────────────────────────────────┤
  │  评估（量化效果）                                        │
  │  目标：量化各阶段的改进                                  │
  │  方法：GSM8K 准确率、生成质量对比、交互式测试           │
  │  脚本：08_eval_and_chat.py ← 本脚本                     │
  └─────────────────────────────────────────────────────────┘""")

    # ── 2. 生成评估问题 ──
    print(f"\n── Step 2: 生成 GSM8K 风格评估问题 ──")
    problems = generate_eval_problems(n_eval_problems)
    print(f"  生成 {len(problems)} 个数学题")
    for i in range(min(5, len(problems))):
        print(f"    [{i}] '{problems[i][0]}'  答案: {problems[i][1]}")

    # ── 3. 尝试加载各阶段 checkpoint ──
    print(f"\n── Step 3: 加载各阶段 checkpoint ──")
    stages = [
        ('pretrain', 'ckpt_pretrain.pt', '预训练模型'),
        ('sft', 'ckpt_sft.pt', 'SFT 模型'),
        ('dpo', 'ckpt_dpo.pt', 'DPO 对齐模型'),
        ('ppo', 'ckpt_ppo.pt', 'PPO 对齐模型'),
        ('grpo', 'ckpt_grpo.pt', 'GRPO 对齐模型'),
    ]

    stage_models = {}  # name -> (model, stoi)
    for name, filename, desc in stages:
        ckpt_path = os.path.join(temp_dir, filename)
        if os.path.exists(ckpt_path):
            result = load_checkpoint(ckpt_path)
            if result is not None:
                model, config, stoi = result
                n_params = sum(p.numel() for p in model.parameters())
                stage_models[name] = (model, stoi)
                print(f"  [OK] {desc}: vocab={config['vocab_size']}, params={n_params:,}")
            else:
                print(f"  [FAIL] {desc}: 加载失败")
        else:
            print(f"  [SKIP] {desc}: {filename} 不存在")

    # 如果没有任何 checkpoint，创建一个随机模型作为 baseline
    if not stage_models:
        print(f"\n  没有找到任何 checkpoint，创建随机模型作为 baseline...")
        model, actual_vocab, stoi = create_fresh_model()
        stage_models['random'] = (model, stoi)
        print(f"  随机模型创建完成")

    # ── 4. 评估各阶段模型 ──
    print(f"\n── Step 4: 评估各阶段模型（{n_eval_problems} 题）──")
    print(f"  {'阶段':<12} {'准确率':<10} {'正确/总数':<12}")
    print(f"  {'─'*36}")

    stage_results = {}
    for name, (model, stoi) in stage_models.items():
        itos = {i: c for c, i in stoi.items()}
        acc, results = evaluate_model(model, stoi, itos, problems,
                                      max_tokens=generate_len)
        stage_results[name] = (acc, results)
        correct = sum(1 for r in results if r['correct'])
        print(f"  {name:<12} {acc:<10.1%} {correct}/{len(results)}")

    # 找出最佳模型
    if stage_results:
        best_stage = max(stage_results, key=lambda k: stage_results[k][0])
        print(f"\n  最佳阶段: {best_stage} (准确率 {stage_results[best_stage][0]:.1%})")

    # ── 5. 生成质量对比 ──
    print(f"\n── Step 5: 各阶段生成质量对比 ──")
    test_prompts = ["Hello", "The"] if CPU_MODE else ["Explain AI", "What is"]
    for prompt in test_prompts:
        print(f"\n  Prompt: '{prompt}'")
        for name, (model, stoi) in stage_models.items():
            itos = {i: c for c, i in stoi.items()}
            prompt_ids = [stoi.get(c, 0) for c in prompt]
            prompt_tensor = torch.tensor([prompt_ids], device=device)
            model.eval()
            with torch.no_grad():
                gen = model.generate(prompt_tensor, max_new_tokens=20,
                                     temperature=0.8, top_k=10)
            gen_text = ''.join(itos.get(i, '?') for i in gen[0].tolist())
            print(f"    {name:<12}: '{gen_text[:60]}'")

    # ── 6. 温度 / Top-K 效果演示 ──
    print(f"\n── Step 6: Temperature / Top-K 效果演示 ──")

    # 选择一个模型做演示
    if stage_models:
        demo_name = list(stage_models.keys())[0]
        demo_model, demo_stoi = stage_models[demo_name]
        demo_itos = {i: c for c, i in demo_stoi.items()}
    else:
        demo_model, _, demo_stoi = create_fresh_model()
        demo_itos = {i: c for c, i in demo_stoi.items()}
        demo_name = "random"

    demo_prompt = "Hello"
    demo_ids = [demo_stoi.get(c, 0) for c in demo_prompt]
    demo_tensor = torch.tensor([demo_ids], device=device)

    print(f"  使用模型: {demo_name}, prompt: '{demo_prompt}'")

    # Temperature 效果
    print(f"\n  Temperature 效果（固定 top_k=10）:")
    print(f"  {'温度':<8} {'生成文本'}")
    print(f"  {'─'*50}")
    for temp in [0.3, 0.7, 1.0, 1.5]:
        demo_model.eval()
        with torch.no_grad():
            gen = demo_model.generate(demo_tensor.clone(), max_new_tokens=15,
                                      temperature=temp, top_k=10)
        gen_text = ''.join(demo_itos.get(i, '?') for i in gen[0].tolist())
        # 中文注释
        if temp < 0.5:
            note = "（低温度：确定性强，重复多）"
        elif temp < 1.0:
            note = "（中温度：平衡）"
        else:
            note = "（高温度：随机性强，多样但可能不通顺）"
        print(f"  {temp:<8} '{gen_text[:40]}'  {note}")

    # Top-K 效果
    print(f"\n  Top-K 效果（固定 temperature=0.8）:")
    print(f"  {'top_k':<8} {'生成文本'}")
    print(f"  {'─'*50}")
    for k in [1, 5, 10, 50]:
        demo_model.eval()
        with torch.no_grad():
            gen = demo_model.generate(demo_tensor.clone(), max_new_tokens=15,
                                      temperature=0.8, top_k=k)
        gen_text = ''.join(demo_itos.get(i, '?') for i in gen[0].tolist())
        if k == 1:
            note = "（贪心：每次选最可能的 token）"
        elif k <= 5:
            note = "（小 k：候选少，较保守）"
        else:
            note = "（大 k：候选多，更多样）"
        print(f"  {k:<8} '{gen_text[:40]}'  {note}")

    # ── 7. Top-P (Nucleus Sampling) 说明 ──
    print(f"\n── Step 7: Top-P (Nucleus Sampling) 说明 ──")
    print(f"""
  Top-P（Nucleus Sampling）是另一种解码策略：

  原理：
    1. 将 token 按概率从高到低排序
    2. 从概率最高的 token 开始累加，直到累积概率 >= p
    3. 只在这些 token 中采样

  与 Top-K 的区别：
    Top-K: 固定候选数量 K（不管概率分布形状）
    Top-P: 动态候选数量（概率集中时少选，分散时多选）

  常用设置：
    Top-P = 0.9: 保留累积概率 90% 的 token
    Top-P = 0.95: 更多样
    Top-P = 1.0: 退化为原始采样

  最佳实践（组合使用）：
    temperature=0.7, top_k=50, top_p=0.9
    — 三个参数共同控制生成的"创造性 vs 确定性"

  注意：本教程的 generate() 只实现了 temperature + top_k，
  未实现 top_p（留作练习）。""")

    # ── 8. 交互式 Chat 模拟 ──
    print(f"\n── Step 8: 交互式 Chat 模拟 ──")

    # 选择最佳模型
    if stage_models:
        # 优先用 grpo > ppo > dpo > sft > pretrain > random
        priority = ['grpo', 'ppo', 'dpo', 'sft', 'pretrain', 'random']
        chat_name = None
        for p in priority:
            if p in stage_models:
                chat_name = p
                break
        if chat_name is None:
            chat_name = list(stage_models.keys())[0]
        chat_model, chat_stoi = stage_models[chat_name]
    else:
        chat_model, _, chat_stoi = create_fresh_model()
        chat_name = "random"
    chat_itos = {i: c for c, i in chat_stoi.items()}

    print(f"  使用模型: {chat_name}")
    print(f"  模拟对话（自动生成 prompt + response）:")

    # 模拟几轮对话
    chat_prompts = ["Hello", "What is", "The sun is"] if CPU_MODE else \
                   ["Explain AI in one sentence.", "What is Python?",
                    "Tell me about the sun.", "How does a computer work?"]

    for i, prompt in enumerate(chat_prompts):
        prompt_ids = [chat_stoi.get(c, 0) for c in prompt]
        prompt_tensor = torch.tensor([prompt_ids], device=device)
        chat_model.eval()

        with torch.no_grad():
            gen = chat_model.generate(
                prompt_tensor,
                max_new_tokens=20,
                temperature=0.7,
                top_k=10,
            )

        resp_ids = gen[0].tolist()[len(prompt_ids):]
        resp_text = ''.join(chat_itos.get(tid, '?') for tid in resp_ids)

        print(f"\n  [User {i+1}] {prompt}")
        print(f"  [Bot]     {resp_text[:80]}")

    # ── 9. Checkpoint 管理建议 ──
    print(f"\n── Step 9: Checkpoint 管理建议 ──")

    # 列出 temp 目录下的 checkpoint
    print(f"\n  当前 checkpoint 状态:")
    all_ckpts = [
        'ckpt_pretrain.pt', 'ckpt_sft.pt', 'ckpt_dpo.pt',
        'ckpt_ppo.pt', 'ckpt_grpo.pt',
    ]
    for ckpt_name in all_ckpts:
        ckpt_path = os.path.join(temp_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            size_kb = os.path.getsize(ckpt_path) / 1024
            print(f"    [OK]   {ckpt_name} ({size_kb:.1f} KB)")
        else:
            print(f"    [MISS] {ckpt_name}")

    print(f"""
  Checkpoint 最佳实践：
    1. 每个阶段保存 checkpoint（便于回溯和对比）
    2. 记录训练超参数和指标（loss, accuracy, reward）
    3. 用 timestamp 或 step 命名（避免覆盖）
    4. 定期清理旧 checkpoint（节省磁盘空间）
    5. 部署前选择最佳 checkpoint（基于评估指标）

  推荐的训练流程：
    python 02_pretrain.py   → ckpt_pretrain.pt
    python 03_sft.py        → ckpt_sft.pt
    python 05_dpo.py        → ckpt_dpo.pt    (简单)
    # 或
    python 06_ppo.py        → ckpt_ppo.py    (更强大)
    # 或
    python 07_grpo.py       → ckpt_grpo.pt   (最简单)
    python 08_eval.py       → 评估对比所有阶段""")

    # ── 10. 推荐的超参数 ──
    print(f"\n── Step 10: 推荐的推理超参数 ──")
    print(f"""
  | 场景             | temperature | top_k | top_p  | 说明                     |
  |------------------|-------------|-------|--------|--------------------------|
  | 数学/代码        | 0.0~0.3     | 1~5   | 0.9    | 确定性高，减少错误       |
  | 日常对话         | 0.7~0.9     | 10~50 | 0.9    | 平衡多样性和连贯性       |
  | 创意写作         | 1.0~1.5     | 50+   | 0.95   | 高多样性，更多创意       |
  | 翻译/摘要        | 0.3~0.5     | 10~20 | 0.9    | 准确为主，少随机性       |

  解码策略选择：
    Greedy (top_k=1): 最确定，但可能重复
    Top-K: 固定候选数，简单有效
    Top-P: 动态候选数，更自适应
    Beam Search: 多条路径并行，找最优序列（本教程未实现）""")

    # ── 总结 ──
    print(f"""
═══ 总结 ═══

本脚本实现了完整的 LLM 评估流水线。

核心内容：

1. GSM8K 风格评估
   生成答案 → 提取数字 → 对比金标 → 计算准确率
   可比较各训练阶段（pretrain → SFT → DPO/PPO/GRPO）

2. Checkpoint 管理
   加载各阶段 checkpoint，对比生成质量
   优先级：GRPO > PPO > DPO > SFT > Pretrain

3. 生成策略分析
   Temperature: 控制随机性（低=确定，高=多样）
   Top-K: 控制候选集大小
   Top-P: 动态候选集（累积概率阈值）

4. 完整训练流水线
   预训练（续写能力）→ SFT（对话能力）→ 奖励模型（判断好坏）
   → DPO/PPO/GRPO（对齐人类偏好）→ 评估（量化效果）

本教程 Part 8 的完整脚本序列：
  01_gpt_model.py      — 构建 GPT 架构
  02_pretrain.py        — 预训练
  03_sft.py             — 监督微调
  04_reward_model.py    — 奖励模型
  05_dpo_alignment.py   — DPO/ORPO/KTO 对齐
  06_ppo_training.py    — PPO 强化学习
  07_grpo_training.py   — GRPO 强化学习
  08_eval_and_chat.py   — 评估 + Chat ← 本脚本

恭喜你完成了 Part 8 的全部学习！
你已经掌握了 LLM 后训练的完整流程。""")


if __name__ == '__main__':
    main()
