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

def forward_pass(params, Xb):
    """
    逐步前向传播，保存所有中间变量。

    参数:
        params: 参数字典 {'C', 'W1', 'b1', 'bngain', 'bnbias', 'W2', 'b2'}
        Xb: 输入 batch (B, 3)

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

    # TODO: 实现逐步前向传播
    # Step 1: Embedding
    # cache['emb'] = ...

    # Step 2: 拼接
    # cache['embcat'] = ...

    # Step 3: Linear 1
    # cache['hprebn'] = ...

    # Step 4: BatchNorm
    # cache['bnmeani'] = ...
    # cache['bndiff'] = ...
    # cache['bndiff2'] = ...
    # cache['bnvar'] = ...
    # cache['bnvar_inv'] = ...
    # cache['bnraw'] = ...
    # cache['hpreact'] = ...

    # Step 5: Tanh
    # cache['h'] = ...

    # Step 6: Linear 2
    # cache['logits'] = ...

    # Step 7: CrossEntropy (展开)
    # cache['loss'] = ...

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
    # TODO: 实现 tanh 反向传播
    # dhpreact = ...
    pass


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
    # TODO: 实现线性层反向传播
    # dinput = ...
    # dweight = ...
    pass


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
    # TODO: 实现 BN 缩放层反向传播
    # dbngain = ...
    # dbnbias = ...
    # dbnraw = ...
    pass


def backward_softmax_ce(logits, Yb):
    """
    CrossEntropy (softmax + NLL) 反向传播。

    参数:
        logits: 未归一化的输出 (B, V)
        Yb: 正确类别标签 (B,)

    返回:
        dlogits: logits 的梯度 (B, V)
    """
    # TODO: 用简化公式实现
    # 提示：3 行代码
    pass


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

    # TODO: 实现简化 CE 反传（3 行代码）
    dlogits = None

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

    # TODO: 实现简化 BN 反传（一行公式）
    dhprebn = None

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

        # TODO: 实现训练循环
        # 1. 前向传播（展开所有中间变量）
        # 2. 手动反向传播
        # 3. 参数更新
        # 4. 更新 BN running stats
        # 5. 记录 loss

        pass

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
