#!/usr/bin/env python3
"""Part 1 作业参考答案：Bigram 语言模型（assignment_reference）"""

import torch
import torch.nn.functional as F


def build_bigram_matrix(words):
    """构建 bigram 计数矩阵 (27,27)，'.'=0, 'a'..'z'=1..26。"""
    chars = [chr(ord('a') + i) for i in range(26)]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi['.'] = 0
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def compute_probabilities(N, smoothing=1):
    """Laplace 平滑后按行归一化。"""
    Nf = (N + smoothing).float()
    P = Nf / Nf.sum(dim=1, keepdim=True)
    return P


def generate_names(P, n=5, seed=2147483647):
    """从 P 逐字符采样；'.'（0）结束名字。"""
    g = torch.Generator().manual_seed(seed)
    itos = {i: c for c, i in {**{chr(ord('a') + i): i + 1 for i in range(26)}, '.': 0}.items()}
    names = []
    for _ in range(n):
        ix = 0
        name = []
        while True:
            p = P[ix]
            nxt = int(torch.multinomial(p, num_samples=1, generator=g).item())
            if nxt == 0:
                break
            name.append(itos[nxt])
            ix = nxt
        names.append(''.join(name))
    return names


def compute_nll_loss(P, words):
    """数据集平均 NLL。"""
    chars = [chr(ord('a') + i) for i in range(26)]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi['.'] = 0
    loglik = 0.0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            loglik += torch.log(P[stoi[ch1], stoi[ch2]]).item()
            n += 1
    return -loglik / n


def train_bigram_nn(words, epochs=100, lr=50, seed=2147483647):
    """梯度下降训练 bigram 神经网络（one-hot → W → softmax → NLL）。"""
    g = torch.Generator().manual_seed(seed)
    chars = [chr(ord('a') + i) for i in range(26)]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi['.'] = 0
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    for _ in range(epochs):
        logits = W[xs]                       # one-hot @ W 的等价查表写法
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(xs)), ys].log().mean()
        loss.backward()
        with torch.no_grad():
            W.data -= lr * W.grad
        W.grad = None
    return W, loss.item()


if __name__ == '__main__':
    import os
    _data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'names.txt')
    words = open(_data_path, 'r').read().splitlines()
    N = build_bigram_matrix(words)
    print(f"Bigram 矩阵形状: {N.shape}")
    P = compute_probabilities(N)
    print(f"概率矩阵行和: {P.sum(1)[:3]}")
    names = generate_names(P, n=5)
    print(f"生成的名字: {names}")
    loss = compute_nll_loss(P, words)
    print(f"平均 NLL: {loss:.4f}")
    W, final_loss = train_bigram_nn(words, epochs=100)
    print(f"神经网络最终 loss: {final_loss:.4f}")
