#!/usr/bin/env python3
"""
Part 7 作业：从零复现 minimind —— 现代 LLM 的六大核心组件

本作业带你从零实现现代 LLM（以 minimind 为蓝本）的六个关键组件：

  题 1. BPE 分词器的编码（subword tokenizer）
  题 2. RMSNorm（简化归一化，替代 LayerNorm）
  题 3. RoPE 旋转位置编码（替代可学习位置编码）
  题 4. GQA 分组的 K/V 头复制 repeat_kv
  题 5. SwiGLU 前馈网络（替代 ReLU FFN）
  题 6. DPO 直接偏好优化损失
  题 7.（🌟 拓展）KV Cache 推理缓存

所有函数/类定义在编写后应该能用下面的测试脚本验证：
  python test_minimind_exercises.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 数据路径：assignments/assignment_7/ 到 data 需要 2 级
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')


# ═══════════════════════════════════════════════════════════════════
#  题 1：BPE 编码（基础）
# ═══════════════════════════════════════════════════════════════════

def exercise_1_bpe_encode(text, merges, vocab):
    """BPE 编码：把文本编码成 token id 序列。

    Args:
        text: 输入字符串。
        merges: 有序列表，元素是 (token_a, token_b) 元组，**先出现的 rank 更小**。
               例：[('a', 'b'), ('ab', 'c')] 表示先把 'ab' 合并成一个 token，再合并 'abc'。
        vocab: 字典 {token_str: id}，包含所有单字符和所有 merges 产生的 token。

    Returns:
        list[int]: 编码后的 token id 列表。

    步骤:
        1. 把 text 拆成单字符列表，例如 'abc' -> ['a', 'b', 'c']
        2. 从左到右扫描，找 merges 中 **rank 最小**（列表里最靠前）且当前出现在
           tokens 里的相邻对 (a, b)；若多个候选并列，取最靠左的那对。
        3. 把该相邻对合并成新 token（删除 a、b，插入 a+b），重复步骤 2。
        4. 直到找不到可合并的对为止。
        5. 用 vocab 把最终 token 列表映射成 id 列表。

    提示:
        - 每次只合并"一个"对，然后重新扫描（因为合并可能产生新的可合并对）。
        - 例：merges=[('a','b'), ('ab','c')], text='abcabc'
          → 先合并 rank0 的 ('a','b') 得 ['ab','c','ab','c']
          → 再合并 rank1 的 ('ab','c') 得 ['abc','abc']
    """
    tokens = list(text)
    ranks = {pair: i for i, pair in enumerate(merges)}
    while len(tokens) >= 2:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        candidates = [(ranks[p], i) for i, p in enumerate(pairs) if p in ranks]
        if not candidates:
            break
        best_rank, best_i = min(candidates)
        a, b = pairs[best_i]
        tokens = tokens[:best_i] + [a + b] + tokens[best_i + 2:]
    return [vocab[t] for t in tokens]


# ═══════════════════════════════════════════════════════════════════
#  题 2：RMSNorm（基础）
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm 归一化层。

    公式: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    与 LayerNorm 的区别:
      - 不做均值中心化（不减 mean），只按均方根缩放
      - 没有 bias（beta）
      - 参数量减半，计算更省，Transformer 里效果相近且更稳

    Attributes:
        eps: 防除零的小常数（默认 1e-6）。
        weight: 可学习缩放，形状 (dim,)，初始全 1。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """输入 x: (..., dim)，输出与 x 同形状。"""
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ═══════════════════════════════════════════════════════════════════
#  题 3：RoPE 旋转位置编码（基础）
# ═══════════════════════════════════════════════════════════════════

def exercise_3_apply_rope(q, freqs_cis):
    """把旋转位置编码应用到 query/key 上。

    Args:
        q: 张量，形状 (B, T, num_heads, head_dim)，head_dim 为偶数。
        freqs_cis: 复数张量，形状 (T, head_dim // 2)，每项模长≈1（单位旋转）。

    Returns:
        Tensor: 旋转后的张量，形状与 q 相同 (B, T, num_heads, head_dim)。

    步骤:
        1. 把 head_dim 维每两个元素看作一个复数 (x0, x1) -> x0 + x1*i
           - 先 reshape 成 (B, T, num_heads, head_dim//2, 2)
        2. 用 torch.view_as_complex 把最后一维的 (x0, x1) 变成复数，
           reshape 成 (B, T, num_heads, head_dim//2)
        3. 与 freqs_cis（广播到该形状）逐元素复数相乘
        4. 用 torch.view_as_real 变回 (..., head_dim//2, 2)，再 reshape 回 (B, T, num_heads, head_dim)

    提示:
        - view_as_complex 要求最后一维大小恰好为 2。
        - 旋转后范数不变（单位复数的旋转是正交变换），测试会验证。
    """
    B, T, H, D = q.shape
    q_c = torch.view_as_complex(q.float().reshape(B, T, H, D // 2, 2))
    out = q_c * freqs_cis[None, :, None, :]
    out = torch.view_as_real(out).reshape(B, T, H, D)
    return out.type_as(q)


# ═══════════════════════════════════════════════════════════════════
#  题 4：GQA 分组的 K/V 头复制（基础）
# ═══════════════════════════════════════════════════════════════════

def exercise_4_repeat_kv(x, n_rep):
    """把 K/V 头复制到与 Q 头数量一致（GQA 的核心）。

    Args:
        x: 张量，形状 (B, num_kv_heads, T, head_dim)。
        n_rep: 整数，复制倍数 = num_heads // num_kv_heads。

    Returns:
        Tensor: 形状 (B, num_kv_heads * n_rep, T, head_dim)，
                **第 i 个输出头 == 第 i // n_rep 个原始头**。
                即原始头按 [h0,h0,h1,h1,...] 顺序排列。

    提示:
        - x[:, :, None, :, :] 增加一个维度，然后 expand/reshape
        - 或者直接 torch.repeat_interleave(x, n_rep, dim=1)
    """
    B, num_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, num_kv, n_rep, T, D)
    return x.reshape(B, num_kv * n_rep, T, D)


# ═══════════════════════════════════════════════════════════════════
#  题 5：SwiGLU 前馈网络（基础）
# ═══════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU 前馈网络（现代 LLM 的标准 FFN）。

    结构: gate_proj / up_proj / down_proj 三个线性层
        out = down_proj( silu(gate_proj(x)) * up_proj(x) )

    与 ReLU FFN 的区别:
      - ReLU FFN: down(relu(x @ W1) @ W2)，硬截断负值为 0
      - SwiGLU: 用 silu（平滑可微的"软门控"）决定信息通过量，梯度更平滑
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim   # 默认 4 倍宽度
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        """输入 x: (..., dim)，输出 (..., dim)。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════
#  题 6：DPO 直接偏好优化损失（基础）
# ═══════════════════════════════════════════════════════════════════

def exercise_6_dpo_loss(pi_logps_chosen, pi_logps_rejected,
                        ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """DPO 直接偏好优化损失。

    Args:
        pi_logps_chosen:  当前策略对"chosen（好回答）"的 log-prob，张量 (N,)。
        pi_logps_rejected: 当前策略对"rejected（差回答）"的 log-prob，张量 (N,)。
        ref_logps_chosen:  参考策略对 chosen 的 log-prob，张量 (N,)。参考策略冻结。
        ref_logps_rejected: 参考策略对 rejected 的 log-prob，张量 (N,)。
        beta: 温度系数，控制与参考策略的偏离程度（默认 0.1）。

    Returns:
        Tensor: 标量 loss（对 batch 求均值）。

    步骤:
        1. log_pi_chosen   = pi_logps_chosen - ref_logps_chosen
        2. log_pi_rejected = pi_logps_rejected - ref_logps_rejected
        3. logits = log_pi_chosen - log_pi_rejected
        4. loss = -F.logsigmoid(beta * logits).mean()

    提示:
        - 目标：最大化 chosen 相对 rejected 的策略优势（以 ref 为锚点，防跑偏）。
        - 参考策略 ref 不更新梯度（它在训练里 requires_grad_(False)）。
    """
    pi_c = pi_logps_chosen - ref_logps_chosen
    pi_r = pi_logps_rejected - ref_logps_rejected
    logits = pi_c - pi_r
    return -F.logsigmoid(beta * logits).mean()


# ═══════════════════════════════════════════════════════════════════
#  题 7（🌟 拓展）：KV Cache 推理缓存
# ═══════════════════════════════════════════════════════════════════

def exercise_7_kv_cache(k, v, past_k, past_v):
    """KV Cache：把历史 K/V 缓存起来，供自回归生成复用。

    Args:
        k: 当前步的 key，张量 (B, num_heads, T_new, head_dim)。
        v: 当前步的 value，张量 (B, num_heads, T_new, head_dim)。
        past_k: 历史 key 缓存，与 k 同前两维，或 None（首次调用）。
        past_v: 历史 value 缓存，或 None。

    Returns:
        (Tensor, Tensor): 更新后的 (new_past_k, new_past_v)，形状
                (B, num_heads, T_old + T_new, head_dim)。

    步骤:
        1. 若 past_k is None：直接返回 (k, v)（这是第一次调用）
        2. 否则: new_past_k = torch.cat([past_k, k], dim=2)  （在时间维拼接）
           new_past_v = torch.cat([past_v, v], dim=2)
        3. 返回 (new_past_k, new_past_v)

    提示:
        - 有了 KV Cache，自回归生成每一步只需算最后一个新 token 的注意力，
          而不必重新计算所有历史 token 的 Q/K/V，复杂度从 O(T²) 降到 O(T)。
        - 拼接维度是时间维（dim=2，因为 K/V 的形状是 (B, heads, T, head_dim)）。
    """
    if past_k is None:
        return k, v
    return torch.cat([past_k, k], dim=2), torch.cat([past_v, v], dim=2)
