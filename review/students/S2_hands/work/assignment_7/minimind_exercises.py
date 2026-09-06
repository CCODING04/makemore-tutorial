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
    ranks = {pair: r for r, pair in enumerate(merges)}
    while True:
        # 找当前 tokens 中 rank 最小的相邻对；并列时取最靠左（min 遍历顺序保证）
        best_rank, best_idx = None, None
        for i in range(len(tokens) - 1):
            r = ranks.get((tokens[i], tokens[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_idx = r, i
        if best_idx is None:
            break
        tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx + 1]] + tokens[best_idx + 2:]
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
    """
    # (B, T, H, hd) -> (B, T, H, hd//2, 2) -> 复数 (B, T, H, hd//2)
    q_complex = torch.view_as_complex(
        q.float().reshape(*q.shape[:-1], -1, 2)
    )
    # freqs_cis: (T, hd//2) -> (1, T, 1, hd//2) 广播
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    rotated = q_complex * freqs
    # 复数 -> (..., hd//2, 2) -> (B, T, H, hd)
    out = torch.view_as_real(rotated).reshape(q.shape)
    return out


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
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    # (B, n_kv, 1, T, hd) -> expand -> (B, n_kv, n_rep, T, hd) -> (B, n_kv*n_rep, T, hd)
    x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd)
    return x.reshape(B, n_kv * n_rep, T, hd)


# ═══════════════════════════════════════════════════════════════════
#  题 5：SwiGLU 前馈网络（基础）
# ═══════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU 前馈网络（现代 LLM 的标准 FFN）。

    结构: gate_proj / up_proj / down_proj 三个线性层
        out = down_proj( silu(gate_proj(x)) * up_proj(x) )
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

    步骤:
        1. log_pi_chosen   = pi_logps_chosen - ref_logps_chosen
        2. log_pi_rejected = pi_logps_rejected - ref_logps_rejected
        3. logits = log_pi_chosen - log_pi_rejected
        4. loss = -F.logsigmoid(beta * logits).mean()
    """
    log_pi_chosen = pi_logps_chosen - ref_logps_chosen
    log_pi_rejected = pi_logps_rejected - ref_logps_rejected
    logits = log_pi_chosen - log_pi_rejected
    return -F.logsigmoid(beta * logits).mean()


# ═══════════════════════════════════════════════════════════════════
#  题 7（🌟 拓展）：KV Cache 推理缓存
# ═══════════════════════════════════════════════════════════════════

def exercise_7_kv_cache(k, v, past_k, past_v):
    """KV Cache：把历史 K/V 缓存起来，供自回归生成复用。

    步骤:
        1. 若 past_k is None：直接返回 (k, v)
        2. 否则在时间维（dim=2）拼接后返回
    """
    if past_k is None:
        return k, v
    return torch.cat([past_k, k], dim=2), torch.cat([past_v, v], dim=2)
