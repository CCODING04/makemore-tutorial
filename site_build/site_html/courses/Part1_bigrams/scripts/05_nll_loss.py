#!/usr/bin/env python3
"""
Part 1 - 脚本 05: 负对数似然损失 (NLL Loss)
目标：评估 bigram 模型的质量。
思路：最大化似然 → 最大化对数似然 → 最小化 NLL → 最小化平均 NLL。
"""

import os
import torch

if __name__ == '__main__':
    # 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    # 构建字符映射
    chars = sorted(set(''.join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    # 构建 bigram 计数矩阵（含平滑）
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1

    P = (N + 1).float()
    P = P / P.sum(1, keepdims=True)

    # ============ 计算整个训练集的 NLL ============
    # 目标：最小化平均 NLL（越小越好）
    log_likelihood = 0.0
    n = 0

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1

    nll = -log_likelihood
    print(f"=== 整个训练集的损失 ===")
    print(f"总 bigram 数量: {n}")
    print(f"log_likelihood: {log_likelihood.item():.4f}")
    print(f"nll (负对数似然): {nll.item():.4f}")
    print(f"平均 NLL (每个 bigram): {nll.item() / n:.4f}")

    # ============ 测试特殊名字 "andrejq" ============
    # "jq" 这个 bigram 在训练数据中很少见，所以 loss 应该很高
    test_name = "andrejq"
    log_likelihood_test = 0.0
    n_test = 0

    print(f"\n=== 测试名字 '{test_name}' 的损失 ===")
    chs = ['.'] + list(test_name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood_test += logprob
        n_test += 1
        print(f"  '{ch1}{ch2}': P({ch2}|{ch1}) = {prob.item():.4f}, log = {logprob.item():.4f}")

    nll_test = -log_likelihood_test
    print(f"log_likelihood: {log_likelihood_test.item():.4f}")
    print(f"nll: {nll_test.item():.4f}")
    print(f"平均 NLL: {nll_test.item() / n_test:.4f}")
    print(f"\n结论：'{test_name}' 的平均 NLL ({nll_test.item() / n_test:.4f}) 远高于整体 ({nll.item() / n:.4f})")
    print(f"  因为 'jq' 这个 bigram 极其罕见，模型认为它不太可能出现。")
