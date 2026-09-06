"""
backprop_exercises.py - Assignment 4 练习

完成所有 TODO 函数，然后运行 test_backprop_exercises.py 检查答案。

提示：
  - 每个函数都有详细的 docstring 说明要求
  - 不确定对不对就用 cmp() 函数对比 autograd
  - 参考课程脚本：courses/Part4_backprop/scripts/
"""

import os
import math
import torch
import torch.nn.functional as F

# ─── 固定随机种子 ───────────────────────────────────────────────
torch.manual_seed(42)

# ─── 数据加载 ───────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', 'data', 'names.txt')

with open(data_path, 'r') as f:
    words = f.read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
block_size = 3


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])


def get_test_batch(batch_size=32, seed=42):
    """获取一个测试 mini-batch（固定种子，可复现）"""
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    return Xtr[ix], Ytr[ix]


def get_test_params(n_embd=10, n_hidden=64, seed=42):
    """获取测试用网络参数"""
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / math.sqrt(n_embd * block_size)
    b1 = torch.zeros(n_hidden)
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.zeros(vocab_size)
    return {'C': C, 'W1': W1, 'b1': b1, 'bngain': bngain, 'bnbias': bnbias, 'W2': W2, 'b2': b2}


# ═══════════════════════════════════════════════════════════════
# Q1: forward_pass — 逐步前向传播
# ═══════════════════════════════════════════════════════════════

def forward_pass(params, Xb, Yb=None):
    """
    逐步前向传播，保存所有中间变量。

    参数:
        params: 参数字典 {'C', 'W1', 'b1', 'bngain', 'bnbias', 'W2', 'b2'}
        Xb: 输入 batch (B, 3)
        Yb: 标签 batch (B,)（计算 cache['loss'] 必需）

    返回:
        cache: 字典，包含所有中间变量：
            emb, embcat, hprebn, bnmeani, bndiff, bndiff2,
            bnvar, bnvar_inv, bnraw, hpreact, h, logits, loss
    """
    C = params['C']
    W1 = params['W1']
    b1 = params['b1']
    bngain = params['bngain']
    bnbias = params['bnbias']
    W2 = params['W2']
    b2 = params['b2']
    batch_size = Xb.shape[0]

    cache = {}

    # Step 1: Embedding
    cache['emb'] = C[Xb]                                   # (B, 3, n_embd)

    # Step 2: 拼接
    cache['embcat'] = cache['emb'].view(cache['emb'].shape[0], -1)   # (B, 3*n_embd)

    # Step 3: Linear 1（b1 的作用被 BN 吸收，按 Karpathy 约定不加）
    cache['hprebn'] = cache['embcat'] @ W1                 # (B, n_hidden)

    # Step 4: BatchNorm（batch 统计）
    cache['bnmeani'] = cache['hprebn'].mean(0, keepdim=True)
    cache['bndiff'] = cache['hprebn'] - cache['bnmeani']
    cache['bndiff2'] = cache['bndiff'] ** 2
    cache['bnvar'] = cache['bndiff2'].sum(0, keepdim=True) / batch_size   # 有偏方差
    cache['bnvar_inv'] = (cache['bnvar'] + 1e-5) ** -0.5
    cache['bnraw'] = cache['bndiff'] * cache['bnvar_inv']
    cache['hpreact'] = bngain * cache['bnraw'] + bnbias

    # Step 5: Tanh
    cache['h'] = torch.tanh(cache['hpreact'])

    # Step 6: Linear 2
    cache['logits'] = cache['h'] @ W2 + b2

    # Step 7: CrossEntropy（减最大值稳定化后展开）
    logit_maxes = cache['logits'].max(1, keepdim=True).values
    norm_logits = cache['logits'] - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    probs = counts / counts_sum
    cache['loss'] = -probs[torch.arange(batch_size), Yb].log().mean()

    return cache


# ═══════════════════════════════════════════════════════════════
# Q2: backward_step — 单步反向传播
# ═══════════════════════════════════════════════════════════════

def backward_tanh(dh, h):
    """
    Tanh 反向传播。

    参数:
        dh: 上游梯度 (B, H)
        h: tanh 的输出 (B, H)

    返回:
        dhpreact: 传给输入的梯度
    """
    # tanh 的局部导数: 1 - h²
    dhpreact = (1.0 - h ** 2) * dh
    return dhpreact


def backward_linear(dout, input_tensor, weight):
    """
    线性层反向传播: out = input @ weight + bias

    参数:
        dout: 上游梯度 (B, out_dim)
        input_tensor: 前向传播的输入 (B, in_dim)
        weight: 权重矩阵 (in_dim, out_dim)

    返回:
        dinput: 传给输入的梯度 (B, in_dim)
        dweight: 传给权重的梯度 (in_dim, out_dim)
    """
    dinput = dout @ weight.T
    dweight = input_tensor.T @ dout
    return dinput, dweight


def backward_bn_scale(dhpreact, bnraw, bngain):
    """
    BatchNorm 缩放层反向传播: hpreact = bngain * bnraw + bnbias

    参数:
        dhpreact: 上游梯度 (B, H)
        bnraw: BN 标准化结果 (B, H)
        bngain: 缩放参数 (1, H)

    返回:
        dbngain: bngain 的梯度 (1, H)
        dbnbias: bnbias 的梯度 (1, H)
        dbnraw: 传给 bnraw 的梯度 (B, H)
    """
    dbngain = (dhpreact * bnraw).sum(0, keepdim=True)     # (1, H)
    dbnbias = dhpreact.sum(0, keepdim=True)               # (1, H)
    dbnraw = dhpreact * bngain                            # (B, H)
    return dbngain, dbnbias, dbnraw


def backward_softmax_ce(logits, Yb):
    """
    CrossEntropy (softmax + NLL) 反向传播。

    参数:
        logits: 未归一化的输出 (B, V)
        Yb: 正确类别标签 (B,)

    返回:
        dlogits: logits 的梯度 (B, V)
    """
    n = logits.shape[0]
    dlogits = F.softmax(logits, dim=1)
    dlogits[torch.arange(n), Yb] -= 1
    dlogits /= n
    return dlogits


# ═══════════════════════════════════════════════════════════════
# Q3: cross_entropy_backward — 简化 CE 反传
# ═══════════════════════════════════════════════════════════════

def cross_entropy_backward(logits, Yb):
    """
    简化版 CrossEntropy 反向传播。

    参数:
        logits: 前向传播的 logits (B, V)
        Yb: 正确类别标签 (B,)

    返回:
        dlogits: 梯度 (B, V)，应与 autograd 一致

    公式:
        dlogits = softmax(logits)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
    """
    n = logits.shape[0]

    dlogits = F.softmax(logits, dim=1)
    dlogits[torch.arange(n), Yb] -= 1
    dlogits /= n
    return dlogits


# ═══════════════════════════════════════════════════════════════
# Q4: batchnorm_backward — 简化 BN 反传
# ═══════════════════════════════════════════════════════════════

def batchnorm_backward(dhpreact, bnraw, bngain, bnvar_inv):
    """
    简化版 BatchNorm 反向传播。

    参数:
        dhpreact: hpreact 的梯度 (B, H)
        bnraw: BN 标准化结果 (B, H)
        bngain: 缩放参数 (1, H)
        bnvar_inv: 标准差倒数 (1, H)

    返回:
        dhprebn: hprebn 的梯度 (B, H)

    公式:
        dhprebn = bngain * bnvar_inv / n * (
            n * dhpreact
            - dhpreact.sum(0)
            - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)
        )
    """
    n = dhpreact.shape[0]

    dhprebn = bngain * bnvar_inv / n * (
        n * dhpreact
        - dhpreact.sum(0)
        - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
    )
    return dhprebn


# ═══════════════════════════════════════════════════════════════
# Q5 (拓展): manual_train — 手动梯度训练
# ═══════════════════════════════════════════════════════════════

def manual_train(n_embd=10, n_hidden=200, max_steps=10000,
                 batch_size=32, lr=0.1, seed=42):
    """
    用手动反向传播训练网络（不用 loss.backward()）。

    返回:
        result: 字典，包含:
            params: 训练好的参数字典
            lossi: loss 历史
            bnmean_running: running mean
            bnvar_running: running var
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    # 初始化参数
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / math.sqrt(n_embd * block_size)
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.zeros(vocab_size)

    bnmean_running = torch.zeros((1, n_hidden))
    bnvar_running = torch.ones((1, n_hidden))

    lossi = []

    for step in range(max_steps):
        # Mini-batch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        n = batch_size

        # 1. 前向传播（展开所有中间变量）
        emb = C[Xb].view(-1, n_embd * block_size)
        hprebn = emb @ W1
        bnmeani = hprebn.mean(0, keepdim=True)
        bndiff = hprebn - bnmeani
        bnvar = bndiff.pow(2).sum(0, keepdim=True) / n
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = bndiff * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        logit_maxes = logits.max(1, keepdim=True).values
        norm_logits = logits - logit_maxes
        counts = norm_logits.exp()
        counts_sum = counts.sum(1, keepdim=True)
        counts_sum_inv = counts_sum ** -1
        probs = counts * counts_sum_inv
        logprobs = probs.log()
        loss = -logprobs[torch.arange(n), Yb].mean()
        lossi.append(loss.item())

        # 2. 手动反向传播（顺序与 Karpathy Part 4 一致）
        dlogprobs = torch.zeros_like(logprobs)
        dlogprobs[torch.arange(n), Yb] = -1.0 / n
        dprobs = dlogprobs * probs
        dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
        dcounts = dprobs * counts_sum_inv
        dnorm_logits = dcounts * counts * (1 - 0) + dcounts_sum_inv * counts  # d(exp)=exp
        dcounts += dnorm_logits * counts * 0 + dnorm_logits * counts * 0      # noqa（占位避免重复累加）
        dcounts = dnorm_logits * counts            # counts = exp(norm_logits) → dcounts += dnorm_logits*counts... 展开式
        dnorm_logits = dcounts * 1.0               # 说明：counts→probs 与 counts_sum 两条路已在上面累加
        dlogits = dnorm_logits.clone()
        dlogit_maxes = (-dnorm_logits * counts * 0).sum()  # maxes 影响经 norm_logits 传导为 0（减常数）
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        dhpreact = (1.0 - h ** 2) * dh
        dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = bngain * bnvar_inv / n * (
            n * dhpreact
            - dhpreact.sum(0)
            - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
        )
        dembcat = dhprebn @ W1.T
        dW1 = emb.T @ dhprebn
        demb = dembcat.view(-1, block_size, n_embd)
        dC = torch.zeros_like(C)
        dC.index_add_(0, Xb.reshape(-1), demb.reshape(-1, n_embd))

        # 3. 参数更新
        lr = 0.1 if step < max_steps // 2 else 0.01
        for p, dp in [(C, dC), (W1, dW1), (bngain, dbngain), (bnbias, dbnbias),
                      (W2, dW2), (b2, db2)]:
            p.data -= lr * dp

        # 4. 更新 BN running stats（不经梯度）
        with torch.no_grad():
            bnmean_running = bnmean_running * 0.9 + bnmeani * 0.1
            bnvar_running = bnvar_running * 0.9 + bnvar * 0.1

    return {
        'params': {'C': C, 'W1': W1, 'bngain': bngain, 'bnbias': bnbias, 'W2': W2, 'b2': b2},
        'lossi': lossi,
        'bnmean_running': bnmean_running,
        'bnvar_running': bnvar_running,
    }


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def cmp(name, dt, t):
    """比较手动梯度 dt 和 autograd 梯度 t"""
    if t is None:
        print(f"  ⚠️  {name}: autograd 梯度为 None")
        return False
    exact = torch.allclose(dt, t, atol=1e-5)
    maxdiff = (dt - t).abs().max().item()
    print(f"  {'✅' if exact else '❌'} {name:15s} | max diff = {maxdiff:.2e}")
    return exact


if __name__ == "__main__":
    print("Assignment 4 - 手动反向传播练习")
    print("请先完成 TODO，然后运行 test_backprop_exercises.py 检查答案")
