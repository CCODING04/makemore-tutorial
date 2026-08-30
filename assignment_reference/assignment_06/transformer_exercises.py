"""
Assignment 6: Transformer/GPT — 从零实现一个 decoder-only Transformer
对应 Part 6 教程（Let's build GPT / makemore Part 6）。

本文件是你需要完成的练习骨架。每个函数/类都有详细的 TODO 步骤提示，
按提示实现即可。全部完成后，你将得到与 gpt.py 一致的组件：
  tokenizer → get_batch → Bigram 基线 → 单头 Self-Attention → 完整 Transformer Block
"""

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'input.txt')


# ═══════════════════════════════════════════════════════════════════
#  题 1：字符级 Tokenizer 与 train/val 划分
# ═══════════════════════════════════════════════════════════════════

def exercise_1_tokenize(text):
    """
    题 1：把原始文本变成可训练的数据。

    Args:
        text (str): 原始文本（tiny Shakespeare 全文，~1.1M 字符）

    Returns:
        dict，包含以下键：
            'chars'       (list of str): 排序后的唯一字符
            'vocab_size'  (int): 唯一字符数（tiny Shakespeare 为 65）
            'stoi'        (dict): 字符 -> 整数
            'itos'        (dict): 整数 -> 字符
            'encode'      (callable): str -> list[int]（字符串 -> 整数列表）
            'decode'      (callable): list[int] -> str（整数列表 -> 字符串）
            'data'        (torch.LongTensor, 1D): encode(text) 的整数序列
            'train_data'  (torch.LongTensor, 1D): 前 90%
            'val_data'    (torch.LongTensor, 1D): 后 10%

    步骤：
        1. chars = sorted(list(set(text)))        # 所有唯一字符，排序
        2. vocab_size = len(chars)
        3. stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 索引
        4. itos = {i: ch for i, ch in enumerate(chars)}   # 索引 -> 字符
        5. encode = lambda s: [stoi[c] for c in s]        # 字符串 -> 整数列表
        6. decode = lambda l: ''.join([itos[i] for i in l])  # 整数列表 -> 字符串
        7. data = torch.tensor(encode(text), dtype=torch.long)  # 1D LongTensor
        8. n = int(0.9 * len(data))   # 前 90% 训练
        9. train_data = data[:n];  val_data = data[n:]    # 后 10% 验证
        10. 按上面的键返回一个 dict

    提示：
        - 语言模型只能输出"见过"的字符：vocab 必须来自数据本身。
        - 其它 tokenizer 对比：sentencepiece(Google) / tiktoken-BPE(OpenAI, 50K)，
          权衡 = 词表大小 vs 序列长度。我们采用最简单的字符级。
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return {'chars': chars, 'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos,
            'encode': encode, 'decode': decode, 'data': data,
            'train_data': train_data, 'val_data': val_data}


# ═══════════════════════════════════════════════════════════════════
#  题 2：DataLoader — get_batch
# ═══════════════════════════════════════════════════════════════════

def exercise_2_get_batch(data, block_size, batch_size, seed=1337):
    """
    题 2：从 1D 整数序列中随机采样 batch 个 chunk。

    不把整篇文本喂入 Transformer：每次只随机采样若干长度为 block_size 的 chunk，
    每个 chunk 内含 block_size 个 (x, y) 训练样本（T, T+1 偏移）。

    Args:
        data (torch.LongTensor, 1D): token 化后的整数序列
        block_size (int): 最大上下文长度（每个 chunk 的长度）
        batch_size (int): 并行处理的独立序列数
        seed (int): 随机种子（固定以便复现）

    Returns:
        (x, y) 元组：
            x (torch.LongTensor): shape (batch_size, block_size)，输入
            y (torch.LongTensor): shape (batch_size, block_size)，目标（x 后移一位）

    步骤：
        1. torch.manual_seed(seed)
        2. ix = torch.randint(len(data) - block_size, (batch_size,))
             # batch_size 个随机起始偏移，取值范围保证 chunk 不越界
        3. x = torch.stack([data[i:i + block_size] for i in ix])
             # 每个偏移取一段长度 block_size 的连续子串，堆叠成 (B, T)
        4. y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
             # 偏移 +1：y 的每个位置是 x 同一位置的"下一个字符"
        5. return x, y

    提示：
        - y[b, t] == x[b, t+1]（后移一位）是本函数最关键的不变量。
        - 每个 chunk 必须是 data 的连续子串（不能随机取散点）。
    """
    torch.manual_seed(seed)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ═══════════════════════════════════════════════════════════════════
#  题 3：Bigram 基线 + 交叉熵 + generate
# ═══════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    题 3：最简单的语言模型基线 —— 每个 token 只看"我是谁"，token 之间不交流。

    直接用 vocab_size x vocab_size 的 Embedding 表当 logits：
    输入 token 索引，查表得到"下一个 token"的分数。
    """

    def __init__(self, vocab_size):
        """
        Args:
            vocab_size (int): 词汇表大小
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Args:
            idx (torch.LongTensor): shape (B, T)，输入 token 索引
            targets (torch.LongTensor or None): shape (B, T)，目标（可为 None）

        Returns:
            (logits, loss) 元组：
                logits (torch.Tensor): 有 targets 时 shape (B*T, vocab_size)
                                        （为交叉熵 reshape 后的版本）；
                                        无 targets 时保持 (B, T, vocab_size)
                loss (torch.Tensor or None): 标量损失；targets=None 时为 None

        步骤：
            1. logits = self.token_embedding_table(idx)   # (B, T, vocab_size)
            2. if targets is None: loss = None
            3. else:
                 a. B, T, C = logits.shape
                 b. logits = logits.view(B * T, C)        # 交叉熵要求 (B*T, C)
                 c. targets = targets.view(B * T)         # 以及 (B*T)
                 d. loss = F.cross_entropy(logits, targets)
            4. return logits, loss

        提示：
            - 与教程/脚本（gpt.py）保持一致：有 targets 时直接返回 reshape 后的
              logits (B*T, C)；无 targets 时（如 generate 里）返回原始 (B, T, C)。

        提示：
            - PyTorch 的 cross_entropy 要求把 (B, T, C) 展平成 (B*T, C)。
            - 初始 loss 应接近 ln(vocab_size) ≈ 4.17（均匀分布的负对数似然）。
        """
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        从当前上下文 idx 继续生成 max_new_tokens 个 token。

        Args:
            idx (torch.LongTensor): shape (B, T)，当前上下文
            max_new_tokens (int): 要新生成的 token 数

        Returns:
            torch.LongTensor: shape (B, T + max_new_tokens)，扩展后的序列

        步骤（循环 max_new_tokens 次）：
            1. logits, _ = self(idx)            # 注意这里 targets=None
            2. logits = logits[:, -1, :]        # 只取最后时间步 (B, vocab)
            3. probs = F.softmax(logits, dim=-1)      # (B, vocab) 概率
            4. idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) 采样
            5. idx = torch.cat((idx, idx_next), dim=1)  # 拼接到时间维
        最后 return idx

        提示：
            - softmax 把分数变成概率，multinomial 按概率采样（引入随机性）。
            - bigram 模型只用最后 1 个 token 就能预测，所以无需裁剪。
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def exercise_3_bigram_model(vocab_size):
    """
    题 3 的入口：构建并返回一个 BigramLanguageModel 实例。

    Args:
        vocab_size (int): 词汇表大小（tiny Shakespeare 为 65）

    Returns:
        BigramLanguageModel 实例
    """
    return BigramLanguageModel(vocab_size)


# ═══════════════════════════════════════════════════════════════════
#  题 4：单头 Self-Attention（Transformer 核心）
# ═══════════════════════════════════════════════════════════════════

def scaled_dot_product_affinity(q, k):
    """
    题 4(a)：计算 scaled attention 的亲和力（尚未遮罩、尚未 softmax）。

    每个 token 发出 query（我在找什么）与 key（我有什么），
    亲和力 = query 与所有 key 的内积。除以 sqrt(head_size) 控制方差。

    Args:
        q (torch.Tensor): shape (B, T, head_size)，query
        k (torch.Tensor): shape (B, T, head_size)，key

    Returns:
        torch.Tensor: shape (B, T, T)，亲和力 wei = q @ k^T / sqrt(head_size)

    步骤：
        1. head_size = q.shape[-1]
        2. wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        3. return wei

    提示（为什么除以 sqrt(head_size)）：
        - 若 q/k 是 unit gaussian，则 q@k^T 的方差 ≈ head_size；
        - 不缩放时 softmax 会太尖锐（趋近 one-hot），初始化时每个 token
          只聚合一个 token；
        - 除以 sqrt(head_size) 后方差 ≈ 1，softmax 保持"扩散"。
    """
    head_size = q.shape[-1]
    return q @ k.transpose(-2, -1) * (head_size ** -0.5)


class SelfAttentionHead(nn.Module):
    """
    题 4(b)：单头自注意力。

    key/query/value 线性投影 → 亲和力(scaled) → 因果遮罩 → softmax → 加权聚合。
    """

    def __init__(self, head_size, n_embd, block_size):
        """
        Args:
            head_size (int): 该头的输出维度（key/query/value 的维度）
            n_embd (int): 输入 x 的特征维度
            block_size (int): 最大上下文长度（决定三角遮罩的尺寸）
        """
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, n_embd)

        Returns:
            torch.Tensor: shape (B, T, head_size)，加权聚合后的输出。
            同时应把 softmax 后的注意力权重存到 self.wei（shape (B, T, T)）。

        步骤：
            1. B, T, C = x.shape
            2. k = self.key(x)        # (B, T, head_size)
            3. q = self.query(x)      # (B, T, head_size)
            4. wei = scaled_dot_product_affinity(q, k)   # (B, T, T) 亲和力
            5. wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
                 # 因果遮罩：未来不能看向过去（decoder 三角遮罩）
            6. wei = F.softmax(wei, dim=-1)   # 每行 softmax，和为 1
            7. self.wei = wei                 # 保存注意力权重供测试/调试
            8. v = self.value(x)              # (B, T, head_size)
            9. out = wei @ v                  # 加权聚合 -> (B, T, head_size)
            10. return out

        提示：
            - tril 可能比 T 大：用 [:T, :T] 裁剪到当前长度。
            - self.wei 的严格上三角应为 0（被 -inf 遮罩后 softmax 归零）。
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = scaled_dot_product_affinity(q, k)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.wei = wei
        v = self.value(x)
        out = wei @ v
        return out


def exercise_4_head(head_size, n_embd, block_size):
    """
    题 4 的入口：构建并返回一个 SelfAttentionHead 实例。

    Args:
        head_size (int): 头输出维度
        n_embd (int): 输入特征维度
        block_size (int): 最大上下文长度

    Returns:
        SelfAttentionHead 实例
    """
    return SelfAttentionHead(head_size, n_embd, block_size)


# ═══════════════════════════════════════════════════════════════════
#  题 5（🌟 拓展）：完整 Transformer Block
# ═══════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    题 5(a)：多头注意力 —— 多个单头并行、通道维拼接、proj 投影回 n_embd。
    （类比分组卷积：多个小的独立通信通道。）
    """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        """
        Args:
            num_heads (int): 头数
            head_size (int): 每个头的输出维度
            n_embd (int): 输入特征维度
            block_size (int): 最大上下文长度
        """
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size, n_embd, block_size)
            for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, n_embd)
        Returns:
            torch.Tensor: shape (B, T, n_embd)
        步骤：
            1. out = torch.cat([h(x) for h in self.heads], dim=-1)  # 沿通道维拼接
            2. out = self.proj(out)                                 # 投影回 n_embd
            3. return out
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """
    题 5(b)：逐 token 的前馈网络 —— "通信之后各自思考"。
    内层 4 x n_embd（论文 512 -> 2048 的 4 倍规律）。
    """

    def __init__(self, n_embd):
        """
        Args:
            n_embd (int): 输入/输出特征维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, n_embd)
        Returns:
            torch.Tensor: shape (B, T, n_embd)
        提示：nn.Sequential / Linear / ReLU 会自动应用到最后一个维度。
        """
        return self.net(x)


class Block(nn.Module):
    """
    题 5(c)：Transformer Block —— 通信(sa) + 计算(ffwd)，pre-norm 结构。

    pre-norm（先 LayerNorm 再子模块，区别于原论文 post-norm）：
        x = x + sa(ln1(x))
        x = x + ffwd(ln2(x))

    残差连接：反传时加法把梯度均分给两个分支 → 梯度"超高速公路"直达输入。
    """

    def __init__(self, n_embd, n_head, block_size):
        """
        Args:
            n_embd (int): 特征维度
            n_head (int): 头数
            block_size (int): 最大上下文长度
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, n_embd)
        Returns:
            torch.Tensor: shape (B, T, n_embd)
        步骤：
            1. x = x + self.sa(self.ln1(x))    # pre-norm：先 LN 再 attention
            2. x = x + self.ffwd(self.ln2(x))  # pre-norm：先 LN 再 ffwd
            3. return x
        提示：残差路径只经过加法；子模块初始贡献小，训练中逐步"上线"。
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


def exercise_5_transformer_block(n_embd, n_head, block_size):
    """
    题 5（🌟 拓展）的入口：构建并返回一个完整 Transformer Block。

    Args:
        n_embd (int): 特征维度
        n_head (int): 头数
        block_size (int): 最大上下文长度

    Returns:
        Block 实例；若未实现，返回 None（测试会优雅跳过）
    """
    return Block(n_embd, n_head, block_size)
