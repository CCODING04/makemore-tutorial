#!/usr/bin/env python3
"""
05 - Minibatch SGD 训练
使用小批量随机梯度下降训练 MLP 字符级语言模型。
- 训练 20000 步，batch_size=32（可用环境变量 STEPS 覆盖步数，如 STEPS=2000 快速冒烟）
- 学习率调度：前 15000 步 lr=0.1，之后 lr=0.01
- 记录 loss 曲线，评估 train/dev/test loss
"""

import os
import torch
import torch.nn.functional as F
import random

# ── 数据路径 ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

# ── 超参数 ────────────────────────────────────────────────
BLOCK_SIZE = 3
N_EMBD = 2
N_HIDDEN = 100
MAX_STEPS = int(os.environ.get('STEPS', '20000'))  # 默认 20000 步；STEPS=2000 可快速冒烟
BATCH_SIZE = 32
LR_INIT = 0.1
LR_DECAY = 0.01
LR_DECAY_STEP = 15000  # 在这一步之后切换到小学习率


def build_dataset(words: list[str], stoi: dict, block_size: int = BLOCK_SIZE):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(2147483647)

    # ── 数据准备 ──────────────────────────────────────────
    with open(data_path, 'r') as f:
        words = f.read().splitlines()
    chars = sorted(set(''.join(words)))
    # 字符映射写法与 Part 1 / 作业统一：a=1..z=26，'.'=0
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(stoi)

    # 划分数据集
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    X_train, Y_train = build_dataset(words[:n1], stoi)
    X_dev, Y_dev = build_dataset(words[n1:n2], stoi)
    X_test, Y_test = build_dataset(words[n2:], stoi)

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_dev.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print()

    # ── 初始化参数 ────────────────────────────────────────
    C = torch.randn((vocab_size, N_EMBD))
    W1 = torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN))
    b1 = torch.randn(N_HIDDEN)
    W2 = torch.randn((N_HIDDEN, vocab_size))
    b2 = torch.randn(vocab_size)
    parameters = [C, W1, b1, W2, b2]

    total_params = sum(p.numel() for p in parameters)
    print(f"总参数量: {total_params}")
    print()

    # 启用梯度追踪
    for p in parameters:
        p.requires_grad = True

    # ── 可选：学习率搜索（教程 03 章「学习率调度」对应的可运行版本） ──
    # 仅在 LR_SEARCH=1 时执行；默认关闭，默认档行为与输出完全不变。
    if os.environ.get('LR_SEARCH') == '1':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        lre = torch.linspace(-3, 0, 1000)   # 在指数上线性取 1000 个点
        lrs = 10 ** lre                     # 0.001 → 1.0，指数空间
        search_lri, search_lossi = [], []
        for i in range(1000):
            ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
            emb = C[X_train[ix]]
            h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
            logits = h @ W2 + b2
            search_loss = F.cross_entropy(logits, Y_train[ix])
            for p in parameters:
                p.grad = None
            search_loss.backward()
            for p in parameters:
                p.data += -lrs[i].item() * p.grad
            search_lri.append(lre[i].item())
            search_lossi.append(search_loss.item())
            if i % 100 == 0:
                print(f"  lr 搜索 {i:4d}/1000 | loss = {search_loss.item():.4f} | 10^lre = {lrs[i]:.4f}", flush=True)

        plt.figure(figsize=(8, 4))
        plt.plot(search_lri, search_lossi)
        plt.xlabel('log10(learning rate)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Search')
        plt.savefig(os.path.join(script_dir, 'lr_search.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("lr 搜索完成，曲线已保存到 lr_search.png（最低点附近即好学习率）", flush=True)

    # ── 训练循环 ──────────────────────────────────────────
    lossi = []

    for step in range(MAX_STEPS):
        # 1) 构造 mini-batch
        ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
        Xb, Yb = X_train[ix], Y_train[ix]

        # 2) 前向传播
        emb = C[Xb]                            # (32, 3, 2)
        emb_cat = emb.view(emb.shape[0], -1)   # (32, 6)
        h = torch.tanh(emb_cat @ W1 + b1)      # (32, 100)
        logits = h @ W2 + b2                   # (32, 27)
        loss = F.cross_entropy(logits, Yb)

        # 3) 反向传播
        for p in parameters:
            p.grad = None
        loss.backward()

        # 4) 更新参数（带学习率调度）
        lr = LR_INIT if step < LR_DECAY_STEP else LR_DECAY
        for p in parameters:
            p.data += -lr * p.grad

        # 记录 loss
        lossi.append(loss.item())

        # 每 2000 步打印一次
        if step % 2000 == 0:
            print(f"  步骤 {step:5d}/{MAX_STEPS} | loss = {loss.item():.4f} | lr = {lr}", flush=True)

    print()
    print(f"训练完成！最终 mini-batch loss: {lossi[-1]:.4f}", flush=True)

    # ── 评估完整数据集 loss ───────────────────────────────
    @torch.no_grad()
    def evaluate(X, Y, label: str):
        emb = C[X]
        h = torch.tanh(emb.view(-1, BLOCK_SIZE * N_EMBD) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)
        print(f"  {label} loss: {loss.item():.4f}", flush=True)

    print()
    print("=== 全数据集评估 ===")
    evaluate(X_train, Y_train, "Train")
    evaluate(X_dev, Y_dev, "Dev")
    evaluate(X_test, Y_test, "Test")
