#!/usr/bin/env python3
"""生成测验和闪卡内容为每个 Part。

基于课程结构生成 JSON 格式的测验题目（选择题、填空题、简答题）。
用法：python generate_quizzes.py
"""
import json
import os

THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS)

# 每个 Part 的核心知识点和测验题目
QUIZ_DATA = {
    1: {
        'title': 'Bigrams - 最简单的语言模型',
        'questions': [
            {
                'type': 'choice',
                'question': 'Bigram 模型的基本思想是什么？',
                'options': ['只看前一个字符预测下一个字符', '看所有前文预测下一个字符', '随机生成字符', '使用神经网络预测'],
                'answer': 0,
                'explanation': 'Bigram 模型只依赖前一个字符（一阶马尔可夫假设），通过统计条件概率来预测下一个字符。'
            },
            {
                'type': 'choice',
                'question': '为什么使用负对数似然（NLL）作为损失函数？',
                'options': ['因为它计算简单', '因为它可以将概率最大化转化为损失最小化', '因为它总是正数', '因为它不需要梯度'],
                'answer': 1,
                'explanation': 'NLL 将概率最大化问题（max P）转化为损失最小化问题（min -log P），便于使用梯度下降优化。'
            },
            {
                'type': 'fill',
                'question': '对于 27 个字符的词汇表，均匀分布的初始 loss 约等于 ______。',
                'answer': 'ln(27) ≈ 3.296',
                'explanation': '当模型对所有字符均匀预测时，每个字符概率为 1/27，loss = -ln(1/27) = ln(27) ≈ 3.296。'
            },
            {
                'type': 'short',
                'question': '解释 Laplace 平滑的作用和原理。',
                'answer': 'Laplace 平滑通过给每个计数加 1（N+1），避免零概率问题。原理：即使某个 bigram 从未出现，也不会得到概率 0，保证模型能生成所有可能的字符。',
                'explanation': '零概率会导致 log(0) = -∞，使训练崩溃。平滑后所有字符都有非零概率。'
            }
        ],
        'flashcards': [
            {'front': '什么是 Bigram 模型？', 'back': '只看前一个字符来预测下一个字符的简单语言模型，基于条件概率统计。'},
            {'front': 'NLL 损失函数的公式是什么？', 'back': 'NLL = -Σ log(P(target))，其中 P(target) 是模型对正确字符的预测概率。'},
            {'front': '为什么初始 loss 应该约等于 ln(vocab_size)？', 'back': '因为随机初始化时模型对所有字符均匀预测，每个字符概率为 1/vocab_size，loss = -ln(1/vocab_size) = ln(vocab_size)。'}
        ]
    },
    2: {
        'title': 'MLP - 多层感知机',
        'questions': [
            {
                'type': 'choice',
                'question': 'Embedding 层的作用是什么？',
                'options': ['将连续向量转为离散索引', '将离散字符映射为连续向量', '进行特征归一化', '实现注意力机制'],
                'answer': 1,
                'explanation': 'Embedding 通过查表将离散的字符索引映射为连续的稠密向量表示。'
            },
            {
                'type': 'choice',
                'question': '为什么要划分 train/dev/test 三个数据集？',
                'options': ['为了增加数据量', '为了防止用"考卷"调参导致过拟合测试集', '为了加快训练速度', '为了减少内存占用'],
                'answer': 1,
                'explanation': 'train 用于训练，dev 用于选模型和调参，test 只在最终评估时使用一次，防止间接过拟合测试集。'
            },
            {
                'type': 'fill',
                'question': 'MLP 中，将多个字符的 embedding 拼接在一起的操作是 ______。',
                'answer': 'emb.view(emb.shape[0], -1) 或 torch.cat',
                'explanation': '将 (N, block_size, n_embd) 的 embedding 张量 reshape 为 (N, block_size * n_embd)，送入全连接层。'
            },
            {
                'type': 'short',
                'question': '解释 block_size 对模型性能的影响。',
                'answer': 'block_size 越大，模型能看到更多上下文，性能通常更好（如 3→5 可降低 dev loss）。但输入维度线性增长（block_size × n_embd），计算成本增加，且收益递减。',
                'explanation': '课程实验：block_size 从 3 增加到 5，dev loss 从约 2.3 降到约 2.1。'
            }
        ],
        'flashcards': [
            {'front': 'Embedding 和 one-hot + 矩阵乘法的关系？', 'back': '数学等价：C[X] 等价于 one_hot(X) @ C，但查表更高效，无需构造稀疏的 one-hot 向量。'},
            {'front': '什么是 block_size？', 'back': '上下文窗口大小，即模型一次能看到的前文字符数量。block_size=3 表示看前 3 个字符。'},
            {'front': 'train/dev/test 三份数据各自的用途？', 'back': 'train: 训练模型；dev: 选择超参和模型；test: 最终一次性评估。'}
        ]
    },
    3: {
        'title': 'BatchNorm - 训练诊断与优化',
        'questions': [
            {
                'type': 'choice',
                'question': 'Kaiming 初始化的核心思想是什么？',
                'options': ['让所有权重为 1', '保持每层输出的方差稳定', '让梯度为 0', '随机初始化权重'],
                'answer': 1,
                'explanation': 'Kaiming 初始化通过缩放权重（gain/√fan_in），确保信号在深层网络中传播时方差保持稳定，避免梯度消失或爆炸。'
            },
            {
                'type': 'choice',
                'question': 'BatchNorm 的 running_mean 和 running_var 在什么时候使用？',
                'options': ['训练时', '推理时', '永远不用', '只在第一层使用'],
                'answer': 1,
                'explanation': '训练时使用 batch 统计量并更新 running stats，推理时使用 running_mean 和 running_var（固定值）。'
            },
            {
                'type': 'fill',
                'question': 'tanh 激活函数的 Kaiming gain 值是 ______。',
                'answer': '5/3',
                'explanation': '对于 tanh 激活函数，Kaiming gain = 5/3 ≈ 1.667，用于补偿 tanh 的非线性压缩效应。'
            },
            {
                'type': 'short',
                'question': '解释 BatchNorm 的 momentum 参数含义。',
                'answer': 'momentum 控制 running stats 的更新速度：running = (1-momentum) * running + momentum * batch_stat。momentum=0.1（PyTorch 默认）更新较快，momentum=0.001 更新较慢但更平滑。',
                'explanation': 'EMA 时间常数 τ = 1/momentum，约 3-5τ 后收敛。'
            }
        ],
        'flashcards': [
            {'front': '什么是 BatchNorm？', 'back': '对每个 mini-batch 的激活值进行归一化（减均值除标准差），再用可学习的 γ/β 缩放偏移。'},
            {'front': '为什么需要 BatchNorm？', 'back': '1) 稳定训练；2) 允许更大学习率；3) 减少对初始化的敏感度；4) 轻微正则化效果。'},
            {'front': '训练和推理时 BatchNorm 的区别？', 'back': '训练：用 batch 均值/方差，更新 running stats；推理：用固定的 running_mean/running_var。'}
        ]
    },
    4: {
        'title': 'Backpropagation - 手动反向传播',
        'questions': [
            {
                'type': 'choice',
                'question': '反向传播的核心数学原理是什么？',
                'options': ['矩阵乘法', '链式法则', '梯度下降', '概率论'],
                'answer': 1,
                'explanation': '反向传播基于链式法则（chain rule），从输出层向输入层逐层计算梯度。'
            },
            {
                'type': 'fill',
                'question': 'tanh 激活函数的导数公式是 ______。',
                'answer': '1 - tanh²(x) 或 1 - h²（用输出表示）',
                'explanation': 'tanh\'(x) = 1 - tanh²(x)，用输出 h 表示可以避免重算 tanh(x)。'
            },
            {
                'type': 'short',
                'question': '为什么非叶子张量的梯度默认不保留？',
                'answer': '为了节省内存。非叶子张量（如中间激活值）的梯度只在反向传播时临时使用，用完即释放。只有叶子张量（模型参数）需要累积梯度用于参数更新。',
                'explanation': '可以用 retain_grad() 强制保留非叶子张量的梯度，用于调试。'
            }
        ],
        'flashcards': [
            {'front': '什么是链式法则？', 'back': '复合函数的导数等于各层导数的乘积：dy/dx = dy/du * du/dx。'},
            {'front': 'autograd 中 retain_grad() 的作用？', 'back': '强制保存非叶子张量的梯度，否则只有叶子张量（参数）的梯度会被保留。'},
            {'front': '为什么手动实现反向传播有助于理解深度学习？', 'back': '能深入理解梯度流动、调试梯度问题、优化计算效率。'}
        ]
    },
    5: {
        'title': 'WaveNet - 层次化架构',
        'questions': [
            {
                'type': 'choice',
                'question': 'WaveNet 相比简单 MLP 的主要优势是什么？',
                'options': ['参数更少', '可以处理更长的上下文', '训练更快', '不需要梯度'],
                'answer': 1,
                'explanation': 'WaveNet 通过层次化的膨胀卷积，可以用更少的参数捕获更长距离的依赖关系。'
            },
            {
                'type': 'fill',
                'question': 'FlattenConsecutive 层的作用是 ______。',
                'answer': '将相邻的 token embedding 拼接在一起',
                'explanation': '实现层次融合：每层将相邻 2 个 token 合并，逐层扩大感受野。'
            }
        ],
        'flashcards': [
            {'front': 'WaveNet 的核心思想是什么？', 'back': '通过层次化的膨胀卷积（dilated convolution），逐层融合相邻 token，指数级扩大感受野。'},
            {'front': '什么是感受野？', 'back': '模型能"看到"的输入范围。WaveNet 每层感受野翻倍：1→2→4→8→...'}
        ]
    },
    6: {
        'title': 'Transformer/GPT - 从零构建',
        'questions': [
            {
                'type': 'choice',
                'question': 'Self-Attention 的核心计算是什么？',
                'options': ['矩阵加法', 'Q @ K^T / √d_k → softmax → @ V', '卷积运算', '循环递归'],
                'answer': 1,
                'explanation': 'Self-Attention：计算 Query 和 Key 的相似度，归一化后加权聚合 Value。'
            },
            {
                'type': 'choice',
                'question': '为什么 Attention 要除以 √d_k？',
                'options': ['为了简化计算', '为了防止 softmax 饱和', '为了减少参数', '为了加速训练'],
                'answer': 1,
                'explanation': '当 d_k 较大时，Q@K^T 的方差会很大，导致 softmax 输出接近 one-hot（梯度消失）。除以 √d_k 使方差稳定在 1 附近。'
            },
            {
                'type': 'fill',
                'question': 'Multi-Head Attention 中，每个 head 的维度是 ______。',
                'answer': 'd_model / n_heads',
                'explanation': '例如 d_model=512, n_heads=8，则每个 head 的维度是 64。'
            },
            {
                'type': 'short',
                'question': '解释 LayerNorm 和 BatchNorm 的区别。',
                'answer': 'BatchNorm 沿 batch 维度归一化（依赖 batch size），LayerNorm 沿特征维度归一化（不依赖 batch size）。Transformer 使用 LayerNorm 因为序列长度可变，batch 统计不稳定。',
                'explanation': 'LayerNorm 对每个样本独立归一化，适合变长序列。'
            }
        ],
        'flashcards': [
            {'front': 'Self-Attention 的三个矩阵 Q, K, V 分别代表什么？', 'back': 'Q (Query): 我在找什么；K (Key): 我有什么特征；V (Value): 我的实际内容。'},
            {'front': '什么是残差连接（Residual Connection）？', 'back': '将层的输入直接加到输出：output = x + f(x)。解决深层网络的梯度消失问题。'},
            {'front': '为什么 Transformer 使用 learned positional embedding？', 'back': '因为 Self-Attention 是置换不变的（permutation invariant），需要位置信息来区分 token 顺序。'}
        ]
    },
    7: {
        'title': 'Minimind 复现 - 现代 LLM 组件',
        'questions': [
            {
                'type': 'choice',
                'question': 'RoPE（旋转位置编码）的核心思想是什么？',
                'options': ['给每个位置一个可学习的向量', '通过旋转矩阵编码相对位置', '使用正弦余弦函数', '添加随机噪声'],
                'answer': 1,
                'explanation': 'RoPE 通过旋转 Q 和 K 向量，使内积自然包含相对位置信息，且支持长度外推。'
            },
            {
                'type': 'choice',
                'question': 'GQA（分组查询注意力）相比 MHA 的优势是什么？',
                'options': ['更准确', '减少 KV Cache 显存', '训练更快', '参数更少'],
                'answer': 1,
                'explanation': 'GQA 让多个 Q head 共享同一组 KV head，减少推理时 KV Cache 的显存占用。'
            },
            {
                'type': 'fill',
                'question': 'RMSNorm 相比 LayerNorm 去掉了 ______ 和 ______。',
                'answer': '均值中心化（减均值）和 bias（偏置项）',
                'explanation': 'RMSNorm 只用均方根（RMS）归一化，省去减均值和 bias，计算更快且效果相当。'
            },
            {
                'type': 'short',
                'question': '解释 DPO 的核心思想。',
                'answer': 'DPO 直接从偏好数据学习，无需训练奖励模型。通过 Bradley-Terry 模型将偏好转化为 loss，隐式优化 RLHF 目标。',
                'explanation': 'DPO 公式：loss = -log(σ(β·(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l))))'
            }
        ],
        'flashcards': [
            {'front': '什么是 KV Cache？', 'back': '推理时缓存已计算的 K 和 V，避免重复计算，使生成每个新 token 只需计算当前 token 的 Q。'},
            {'front': 'SwiGLU 激活函数的公式是什么？', 'back': 'SwiGLU(x) = (x @ W_gate) * swish(x @ W_up) @ W_down，其中 swish(x) = x * sigmoid(x)。'},
            {'front': 'MoE（混合专家）如何工作？', 'back': '路由器（Router）为每个 token 选择 top-k 个专家，只激活选中的专家，实现稀疏激活。'}
        ]
    },
    8: {
        'title': '后训练全流程 - SFT → RM → RL',
        'questions': [
            {
                'type': 'choice',
                'question': 'SFT 中 Prompt Masking 的作用是什么？',
                'options': ['隐藏 prompt 不让模型看到', '只计算 response 部分的 loss', '增加数据多样性', '加速训练'],
                'answer': 1,
                'explanation': 'Prompt Masking 将 prompt 部分的 loss 设为 0（ignore_index=-100），只训练模型学会生成 response。'
            },
            {
                'type': 'choice',
                'question': 'GRPO 与 PPO 的主要区别是什么？',
                'options': ['GRPO 不需要奖励模型', 'GRPO 去掉了 Value Network', 'GRPO 不需要 KL 惩罚', 'GRPO 只能用于文本'],
                'answer': 1,
                'explanation': 'GRPO 用组内标准化优势替代 Value Network，节省显存和计算，是 PPO 的轻量替代。'
            },
            {
                'type': 'fill',
                'question': 'DPO 的损失函数中，β 参数控制 ______。',
                'answer': '对参考模型的偏离程度（KL 约束强度）',
                'explanation': 'β 越大，约束越强，策略越接近参考模型；β 越小，允许偏离越多。'
            },
            {
                'type': 'short',
                'question': '解释 RLHF 的三个阶段。',
                'answer': '1) SFT：在指令数据上微调；2) RM：训练奖励模型学习人类偏好；3) RL：用 PPO/GRPO 优化策略，最大化奖励同时保持与 SFT 模型的 KL 距离。',
                'explanation': '三个阶段逐步将"会说话"的模型变成"说好话"的模型。'
            }
        ],
        'flashcards': [
            {'front': '什么是 Bradley-Terry 模型？', 'back': '将偏好建模为两个选项的比较：P(y_w > y_l) = σ(r(y_w) - r(y_l))，其中 r 是奖励函数。'},
            {'front': 'PPO 的 Clipped Surrogate 目标是什么？', 'back': 'L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)，防止策略更新过大。'},
            {'front': '什么是 Reward Hacking？', 'back': '模型学会"钻奖励函数的漏洞"，获得高分但实际质量下降。需要 KL 约束或奖励模型集成来缓解。'}
        ]
    },
    9: {
        'title': 'CUDA 内核编程 - GPU 架构',
        'questions': [
            {
                'type': 'choice',
                'question': 'GPU 的基本执行单元是什么？',
                'options': ['CPU 核心', 'Warp（32 线程）', 'Block', 'Grid'],
                'answer': 1,
                'explanation': 'Warp 是 GPU 调度的最小单位，包含 32 个线程，同步执行同一条指令（SIMT）。'
            },
            {
                'type': 'fill',
                'question': 'CUDA 中，共享内存（SMEM）的特点是 ______。',
                'answer': 'Block 内线程共享、速度接近寄存器、手动管理',
                'explanation': 'SMEM 由同一 Block 内的线程共享，访问速度比全局内存快 10-100 倍，需要手动声明和同步。'
            },
            {
                'type': 'short',
                'question': '解释合并访存（Coalesced Memory Access）的重要性。',
                'answer': 'GPU 全局内存访问以 128 字节为单位。当同一 Warp 的 32 个线程访问连续地址时，可以合并为一次事务；否则需要多次事务，带宽利用率下降。合并访存 vs 非合并访存可差 9 倍以上。',
                'explanation': '课程实测：合并访存做对 vs 做错差约 9 倍。'
            }
        ],
        'flashcards': [
            {'front': 'GPU 的内存层次结构是什么？', 'back': '寄存器 → 共享内存(SMEM) → L1/L2 Cache → 全局内存(HBM)。速度递减，容量递增。'},
            {'front': '什么是 Bank Conflict？', 'back': '同一 Warp 的多个线程访问 SMEM 的同一 Bank 时发生冲突，需要串行化。可以通过 padding 避免。'},
            {'front': 'Triton 是什么？', 'back': 'OpenAI 开发的 GPU 编程语言，用 Python 语法写 kernel，自动处理 tiling、内存合并等优化。'}
        ]
    },
    10: {
        'title': '分布式训练 - 从单卡到集群',
        'questions': [
            {
                'type': 'choice',
                'question': 'DDP（分布式数据并行）的核心思想是什么？',
                'options': ['把模型切到不同 GPU', '每个 GPU 持有完整模型，数据不同', '只在 CPU 上训练', '使用单卡训练'],
                'answer': 1,
                'explanation': 'DDP 中每个 GPU 持有完整的模型副本，处理不同的数据子集，通过 All-Reduce 同步梯度。'
            },
            {
                'type': 'fill',
                'question': 'ZeRO 的三个阶段分别优化 ______、______、______。',
                'answer': '优化器状态、梯度、参数',
                'explanation': 'ZeRO-1: 分片优化器状态；ZeRO-2: +分片梯度；ZeRO-3: +分片参数。每个阶段减少显存占用。'
            },
            {
                'type': 'short',
                'question': '解释张量并行（Tensor Parallelism）的原理。',
                'answer': '将矩阵乘法拆分到多个 GPU：例如将权重矩阵按列切分，每个 GPU 计算部分结果，再通过 All-Gather 合并。适合单层很大的模型（如 70B）。',
                'explanation': 'Megatron-LM 风格的张量并行：行切分和列切分交替使用，最小化通信。'
            }
        ],
        'flashcards': [
            {'front': 'All-Reduce 操作的作用是什么？', 'back': '将所有 GPU 的梯度求和/求平均，并将结果广播到所有 GPU，保证参数同步。'},
            {'front': '什么是流水线并行（Pipeline Parallelism）？', 'back': '将模型按层切分到不同 GPU，通过 micro-batch 流水线执行，减少气泡（bubble）。'},
            {'front': 'FSDP 和 ZeRO-3 的关系？', 'back': 'FSDP（Fully Sharded Data Parallel）是 PyTorch 对 ZeRO-3 的实现，将参数、梯度、优化器状态全部分片。'}
        ]
    },
    11: {
        'title': '对齐实战 - verl/GRPO',
        'questions': [
            {
                'type': 'choice',
                'question': 'GRPO 中"组内标准化优势"的作用是什么？',
                'options': ['增加奖励值', '减少方差，稳定训练', '加速收敛', '减少参数'],
                'answer': 1,
                'explanation': '对同一 prompt 的多个采样计算相对优势（减均值除标准差），减少奖励尺度的影响。'
            },
            {
                'type': 'short',
                'question': '解释 RLVR（Reinforcement Learning with Verifiable Rewards）。',
                'answer': '使用可验证的规则作为奖励（如数学题的答案是否正确），而非学习的奖励模型。优点是无需标注偏好数据，缺点是只适用于有标准答案的任务。',
                'explanation': 'RLVR 是 GRPO 的典型应用场景，DeepSeek-Math 首创。'
            }
        ],
        'flashcards': [
            {'front': 'verl 框架的三角色是什么？', 'back': 'Actor（策略模型）、Critic（价值模型，GRPO 可选）、Reference（参考模型，计算 KL）。'},
            {'front': '什么是 KL 散度约束？', 'back': '限制策略模型与参考模型的差异，防止 reward hacking。KL = Σ π_ref(y|x) * log(π_ref(y|x) / π(y|x))。'}
        ]
    },
    12: {
        'title': '微调实战 - LLaMA-Factory/LoRA',
        'questions': [
            {
                'type': 'choice',
                'question': 'LoRA 的核心思想是什么？',
                'options': ['全参数微调', '低秩分解注入', '冻结所有层', '只训练最后一层'],
                'answer': 1,
                'explanation': 'LoRA 冻结预训练权重，只训练低秩分解的 A 和 B 矩阵（W\' = W + B×A），大幅减少可训练参数。'
            },
            {
                'type': 'fill',
                'question': 'QLoRA 相比 LoRA 额外引入了 ______ 和 ______ 技术。',
                'answer': 'NF4 量化和双量化',
                'explanation': 'QLoRA 将基础模型量化到 4-bit（NF4），并用双量化进一步压缩量化参数，减少显存占用。'
            },
            {
                'type': 'short',
                'question': '为什么 LoRA 的 B 矩阵要零初始化？',
                'answer': 'B 零初始化保证第一步 W\' = W + B×A = W，即初始时模型与原模型完全相同，训练从原模型出发，避免随机扰动破坏预训练知识。',
                'explanation': '如果 B 不为零，初始时 W\' ≠ W，模型行为会突变，可能导致训练不稳定。'
            }
        ],
        'flashcards': [
            {'front': 'LoRA 的参数量怎么算？', 'back': '每个线性层增加 2 × rank × d 参数（A: d×rank, B: rank×d）。例如 rank=8, d=4096，增加 65K 参数。'},
            {'front': '什么时候用 LoRA，什么时候用全参微调？', 'back': 'LoRA: 资源有限、快速实验、数据少；全参: 数据充足、追求最佳效果、任务与预训练差异大。'},
            {'front': 'DPO-LoRA 是什么？', 'back': '用 LoRA 做 DPO 微调：同时训练策略模型和参考模型的 LoRA adapter，参考模型权重冻结。'}
        ]
    },
    13: {
        'title': '数据工程 - 去重与清洗',
        'questions': [
            {
                'type': 'choice',
                'question': 'MinHash 的作用是什么？',
                'options': ['精确去重', '近似去重（估计 Jaccard 相似度）', '数据加密', '特征提取'],
                'answer': 1,
                'explanation': 'MinHash 用多个哈希函数的最小值作为文档签名，签名相同的概率等于 Jaccard 相似度。'
            },
            {
                'type': 'fill',
                'question': 'LSH（局部敏感哈希）通过 ______ 技术加速相似文档查找。',
                'answer': '分带（band）',
                'explanation': '将签名分成 b 个带，每个带 r 行。只要有一个带完全匹配，就认为是候选对。'
            },
            {
                'type': 'short',
                'question': '为什么训练数据需要去重？',
                'answer': '重复数据会导致：1) 模型过拟合特定样本；2) 评估数据污染（训练集包含测试集）；3) 训练效率降低（梯度被重复样本主导）。FineWeb 实验显示去重可显著提升模型质量。',
                'explanation': '课程实测：MinHash+LSH 去重可将数据集压缩到原来的 60-80%。'
            }
        ],
        'flashcards': [
            {'front': '什么是 Jaccard 相似度？', 'back': '两个集合的交集大小除以并集大小：J(A,B) = |A∩B| / |A∪B|。范围 [0,1]，越大越相似。'},
            {'front': 'LSH 的分带参数如何选择？', 'back': '给定目标阈值 t，选择 b（带数）和 r（每帧行数）使得 S-curve 在 t 处陡峭。例如 14 band × 8 row ≈ 阈值 0.7。'},
            {'front': 'Data-Juicer 是什么？', 'back': '阿里巴巴开源的数据清洗框架，用 YAML 配置管线，支持 100+ 算子（去重、过滤、质量评分等）。'}
        ]
    },
    14: {
        'title': '推理部署 - vLLM',
        'questions': [
            {
                'type': 'choice',
                'question': 'vLLM 的 PagedAttention 解决了什么问题？',
                'options': ['训练速度', 'KV Cache 内存碎片化', '模型精度', '数据加载'],
                'answer': 1,
                'explanation': 'PagedAttention 将 KV Cache 分成固定大小的页，按需分配，避免预分配导致的内存浪费（从 60-80% 降到 <4%）。'
            },
            {
                'type': 'fill',
                'question': 'TTFT 是 ______ 的缩写，表示 ______。',
                'answer': 'Time To First Token，首 token 延迟',
                'explanation': 'TTFT 衡量从请求到生成第一个 token 的时间，反映 prefill 阶段的效率。'
            },
            {
                'type': 'short',
                'question': '解释投机解码（Speculative Decoding）的原理。',
                'answer': '用小模型快速生成多个候选 token，大模型一次性验证。如果候选被接受，节省了多次大模型推理；如果不接受，从分歧点重新生成。加速比取决于接受率 α。',
                'explanation': '课程实测：α≈0.65 时，实测 2.81 tokens/cycle vs 理论 2.53。'
            }
        ],
        'flashcards': [
            {'front': '什么是连续批处理（Continuous Batching）？', 'back': '不等一个 batch 全部完成，而是有请求就加入、完成就退出，提高 GPU 利用率。'},
            {'front': 'KV Cache 的显存怎么算？', 'back': 'KV 显存 = 2 × n_layers × n_heads × d_head × seq_len × dtype_size。例如 LLaMA-7B fp16 seq2048 ≈ 1.07GB。'},
            {'front': 'GQA 如何减少 KV Cache？', 'back': '多个 Q head 共享同一组 KV head。例如 KV heads=8（vs Q heads=32），KV Cache 减少 4 倍。'}
        ]
    },
    15: {
        'title': '多模态理解 - VLM',
        'questions': [
            {
                'type': 'choice',
                'question': 'VLM 中 Projector 的作用是什么？',
                'options': ['提取图像特征', '将视觉特征映射到语言空间', '生成图像', '分类图像'],
                'answer': 1,
                'explanation': 'Projector 将 ViT 提取的视觉特征投影到 LLM 的 embedding 空间，实现跨模态对齐。'
            },
            {
                'type': 'short',
                'question': '解释 CLIP 和 SigLIP 的区别。',
                'answer': 'CLIP 使用 softmax 交叉熵损失（对比学习），SigLIP 使用 sigmoid 二元交叉熵（独立分类）。SigLIP 不需要全局归一化，支持更大的 batch size，训练更稳定。',
                'explanation': 'SigLIP 是 Google 的改进版本，InfoNCE loss 的 sigmoid 变体。'
            }
        ],
        'flashcards': [
            {'front': '什么是 LLaVA 的两阶段训练？', 'back': '阶段一：只训练 Projector（冻结 ViT 和 LLM）；阶段二：解冻 LLM，联合微调。'},
            {'front': 'VLM 的三种架构方案是什么？', 'back': '1) 拼接式（concatenation）：视觉 token 拼接到文本前；2) 门控式（gated）：交叉注意力；3) 早期融合（early-fusion）：从底层就混合。'}
        ]
    },
    16: {
        'title': '图像/视频生成 - 扩散模型',
        'questions': [
            {
                'type': 'choice',
                'question': 'DDPM 的前向过程是什么？',
                'options': ['从噪声生成图像', '逐步给图像加噪声', '去噪', '编码图像'],
                'answer': 1,
                'explanation': '前向过程逐步向图像添加高斯噪声，直到变成纯噪声。这是固定的马尔可夫链，不需要学习。'
            },
            {
                'type': 'fill',
                'question': 'DDPM 的反向过程通过 ______ 学习去噪。',
                'answer': '预测噪声（noise prediction）',
                'explanation': '模型学习预测每步添加的噪声 ε，然后从当前状态减去预测的噪声，逐步还原图像。'
            },
            {
                'type': 'short',
                'question': '解释 Classifier-Free Guidance 的原理。',
                'answer': '同时训练有条件和无条件生成，推理时将两者结合：output = uncond + scale × (cond - uncond)。scale > 1 增强条件控制，但可能降低多样性。',
                'explanation': 'CFG 是 Stable Diffusion 等模型的关键技术。'
            }
        ],
        'flashcards': [
            {'front': '什么是 IP-Adapter？', 'back': '通过解耦的 Cross-Attention 将图像特征注入扩散模型，实现图像风格/内容控制，无需重新训练。'},
            {'front': 'ControlNet 如何工作？', 'back': '在 U-Net 旁添加条件分支，接收边缘图、深度图等条件，通过 zero convolution 注入主网络。'},
            {'front': '视频生成的主要挑战是什么？', 'back': '1) 时间一致性（帧间连贯）；2) 计算成本（3D 卷积/注意力）；3) 长视频生成（误差累积）。'}
        ]
    },
    17: {
        'title': 'Agentic RL - 多轮工具调用',
        'questions': [
            {
                'type': 'choice',
                'question': 'Agentic RL 与单轮 RLHF 的主要区别是什么？',
                'options': ['奖励函数不同', '轨迹更长，需要处理多轮交互', '不需要 KL 约束', '只能用于文本'],
                'answer': 1,
                'explanation': 'Agentic RL 处理多轮工具调用轨迹，需要考虑长期奖励、工具使用正确性、轨迹级别的优势估计。'
            },
            {
                'type': 'short',
                'question': '什么是 BC 冷启动？',
                'answer': '在 RL 训练前，先用行为克隆（Behavior Cloning）从示范数据学习基础策略，避免随机初始化导致的探索效率低下。',
                'explanation': 'BC 冷启动 → RL 微调是 Agentic RL 的常见训练流程。'
            }
        ],
        'flashcards': [
            {'front': '什么是 Echo Trap？', 'back': '模型陷入重复输出的循环，无法跳出。需要特殊的奖励设计或采样策略来打破。'},
            {'front': 'GiGPO 是什么？', 'back': 'Group-in-Group Policy Optimization，将轨迹分成子组，组内标准化优势，减少方差。'}
        ]
    },
    18: {
        'title': 'RAG 全链路 - 检索增强生成',
        'questions': [
            {
                'type': 'choice',
                'question': 'RAG 中"混合检索"指什么？',
                'options': ['混合不同模型', '同时使用稀疏检索和稠密检索', '混合不同数据', '混合不同任务'],
                'answer': 1,
                'explanation': '混合检索结合 BM25（稀疏/关键词）和向量检索（稠密/语义），通过 RRF 等方法融合结果，取长补短。'
            },
            {
                'type': 'fill',
                'question': 'RRF（Reciprocal Rank Fusion）的公式是 ______。',
                'answer': 'score = Σ 1 / (k + rank_i)，其中 k 通常为 60',
                'explanation': 'RRF 将多个排序列表的排名倒数求和，k 平滑常数防止排名靠前的项过度主导。'
            },
            {
                'type': 'short',
                'question': '解释"什么时候不该用 RAG"。',
                'answer': '1) 问题需要推理而非检索（如数学证明）；2) 知识库质量差或过时；3) 延迟要求极高（RAG 增加检索延迟）；4) 问题超出知识库范围。此时应直接用 LLM 或微调。',
                'explanation': 'RAG 不是万能的，需要根据任务特点选择。'
            }
        ],
        'flashcards': [
            {'front': '什么是 Contextual Retrieval？', 'back': 'Anthropic 提出的技术：为每个 chunk 添加上下文摘要，提升检索召回率。'},
            {'front': 'RAG 中的"幻觉"问题如何缓解？', 'back': '1) 引用来源；2) 使用 faithfulness 评测；3) 检索结果置信度过滤；4) 后处理验证。'},
            {'front': '什么是重排（Reranking）？', 'back': '用更精确的模型（如 Cross-Encoder）对初检结果重新排序，提升最终结果质量。'}
        ]
    },
    19: {
        'title': 'Agent 与 Function Calling',
        'questions': [
            {
                'type': 'choice',
                'question': 'Agent Loop 的三个终止条件是什么？',
                'options': ['超时、错误、用户取消', '任务完成、达到步数上限、模型拒绝', '工具调用失败、网络断开、内存不足', '答案正确、答案错误、无法回答'],
                'answer': 1,
                'explanation': 'Agent Loop 终止：1) 模型输出最终答案；2) 达到最大步数；3) 模型主动拒绝或请求人工介入。'
            },
            {
                'type': 'short',
                'question': '解释 MCP（Model Context Protocol）的三层架构。',
                'answer': '1) Transport 层：HTTP/SSE/WebSocket 通信；2) Protocol 层：JSON-RPC 2.0 消息格式；3) Tool 层：工具定义、调用、结果返回。MCP 让 Agent 与外部工具标准化交互。',
                'explanation': 'MCP 是 Anthropic 提出的开放标准，已成为事实标准。'
            }
        ],
        'flashcards': [
            {'front': '什么是 Function Calling？', 'back': 'LLM 输出结构化的工具调用请求（函数名+参数），由系统执行后将结果返回给模型，实现与外部世界的交互。'},
            {'front': 'τ-bench 测试什么？', 'back': '测试 Agent 在真实场景下的工具使用能力：给定政策文档和用户请求，Agent 需要正确调用工具完成任务。'},
            {'front': '多智能体系统的优势是什么？', 'back': '1) 任务分解（专家分工）；2) 并行处理；3) 互相验证（减少幻觉）；4) 更好的可扩展性。'}
        ]
    }
}


def generate_quiz_html(part_num, quiz_data):
    """生成单个 Part 的测验 HTML 页面。"""
    title = quiz_data['title']
    questions = quiz_data['questions']
    flashcards = quiz_data['flashcards']
    
    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Part {part_num} 测验 - {title}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f7f8fa; color: #1f2328; }}
h1 {{ color: #0969da; border-bottom: 2px solid #d0d7de; padding-bottom: 10px; }}
.quiz-section {{ background: #fff; border: 1px solid #d0d7de; border-radius: 12px; padding: 20px; margin: 20px 0; }}
.question {{ margin-bottom: 20px; }}
.question h3 {{ color: #1f2328; font-size: 16px; }}
.options {{ list-style: none; padding: 0; }}
.options li {{ padding: 10px 15px; margin: 5px 0; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; cursor: pointer; transition: all 0.2s; }}
.options li:hover {{ border-color: #0969da; background: #dbeafe; }}
.options li.selected {{ border-color: #0969da; background: #dbeafe; }}
.options li.correct {{ border-color: #2da44e; background: #dcfce7; }}
.options li.wrong {{ border-color: #cf222e; background: #fee2e2; }}
.explanation {{ display: none; padding: 10px; margin-top: 10px; background: #f0f3f6; border-radius: 8px; font-size: 14px; color: #57606a; }}
.explanation.show {{ display: block; }}
.flashcard {{ background: #fff; border: 1px solid #d0d7de; border-radius: 12px; padding: 20px; margin: 10px 0; cursor: pointer; transition: all 0.3s; }}
.flashcard:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
.flashcard .front {{ font-weight: 600; color: #0969da; }}
.flashcard .back {{ display: none; margin-top: 10px; padding-top: 10px; border-top: 1px dashed #d0d7de; color: #57606a; }}
.flashcard.show .back {{ display: block; }}
.toolbar {{ position: fixed; top: 10px; right: 10px; display: flex; gap: 8px; }}
.toolbar a {{ padding: 8px 16px; background: #f0f3f6; border: 1px solid #d0d7de; border-radius: 8px; text-decoration: none; color: #1f2328; font-size: 14px; }}
.score {{ font-size: 24px; font-weight: 600; color: #2da44e; text-align: center; padding: 20px; background: #dcfce7; border-radius: 12px; margin: 20px 0; display: none; }}
</style>
</head>
<body>
<div class="toolbar">
  <a href="/index.html">返回首页</a>
  <a href="/quizzes/quiz_index.html">全部测验</a>
</div>
<h1>Part {part_num}: {title}</h1>

<div class="quiz-section">
<h2>选择题</h2>
'''
    
    for i, q in enumerate(questions):
        if q['type'] == 'choice':
            html += f'''
<div class="question" data-answer="{q['answer']}">
<h3>{i+1}. {q['question']}</h3>
<ul class="options">
'''
            for j, opt in enumerate(q['options']):
                html += f'  <li data-idx="{j}" onclick="selectOption(this)">{opt}</li>\n'
            html += f'''</ul>
<div class="explanation">{q['explanation']}</div>
</div>
'''
        elif q['type'] == 'fill':
            html += f'''
<div class="question">
<h3>{i+1}. {q['question']}</h3>
<input type="text" placeholder="输入答案" style="width: 100%; padding: 10px; border: 1px solid #d0d7de; border-radius: 8px; font-size: 14px;">
<div class="explanation"><strong>答案：</strong>{q['answer']}<br>{q['explanation']}</div>
<button onclick="this.previousElementSibling.classList.toggle('show')" style="margin-top: 10px; padding: 8px 16px; background: #0969da; color: #fff; border: none; border-radius: 8px; cursor: pointer;">显示答案</button>
</div>
'''
        elif q['type'] == 'short':
            html += f'''
<div class="question">
<h3>{i+1}. {q['question']}</h3>
<textarea placeholder="输入你的答案" rows="4" style="width: 100%; padding: 10px; border: 1px solid #d0d7de; border-radius: 8px; font-size: 14px; resize: vertical;"></textarea>
<div class="explanation"><strong>参考答案：</strong><br>{q['answer']}<br><br>{q['explanation']}</div>
<button onclick="this.previousElementSibling.classList.toggle('show')" style="margin-top: 10px; padding: 8px 16px; background: #0969da; color: #fff; border: none; border-radius: 8px; cursor: pointer;">显示答案</button>
</div>
'''
    
    html += '</div>\n<div class="score" id="score"></div>\n'
    
    # 闪卡部分
    html += '''
<div class="quiz-section">
<h2>闪卡</h2>
<p style="color: #57606a; font-size: 14px;">点击卡片查看答案</p>
'''
    for fc in flashcards:
        html += f'''
<div class="flashcard" onclick="this.classList.toggle('show')">
<div class="front">{fc['front']}</div>
<div class="back">{fc['back']}</div>
</div>
'''
    
    html += '''
</div>

<script>
function selectOption(el) {
    const question = el.closest('.question');
    const options = question.querySelectorAll('.options li');
    const answer = parseInt(question.dataset.answer);
    const selected = parseInt(el.dataset.idx);
    
    options.forEach(opt => {
        opt.classList.remove('selected', 'correct', 'wrong');
        opt.style.pointerEvents = 'none';
    });
    
    el.classList.add('selected');
    if (selected === answer) {
        el.classList.add('correct');
    } else {
        el.classList.add('wrong');
        options[answer].classList.add('correct');
    }
    
    question.querySelector('.explanation').classList.add('show');
}
</script>
</body>
</html>'''
    
    return html


def main():
    quizzes_dir = os.path.join(REPO_ROOT, 'quizzes')
    os.makedirs(quizzes_dir, exist_ok=True)
    
    # 生成每个 Part 的测验
    for part_num, quiz_data in QUIZ_DATA.items():
        html = generate_quiz_html(part_num, quiz_data)
        html_path = os.path.join(quizzes_dir, f'quiz_part{part_num:02d}.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'  Part {part_num}: 生成 {len(quiz_data["questions"])} 题 + {len(quiz_data["flashcards"])} 闪卡')
    
    # 生成测验索引页
    index_html = '''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>测验系统 - LLM 面试课程</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f7f8fa; color: #1f2328; }
h1 { color: #0969da; border-bottom: 2px solid #d0d7de; padding-bottom: 10px; }
.quiz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 16px; margin: 20px 0; }
.quiz-card { background: #fff; border: 1px solid #d0d7de; border-radius: 12px; padding: 20px; text-decoration: none; color: #1f2328; transition: all 0.2s; }
.quiz-card:hover { border-color: #0969da; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px); }
.quiz-card h3 { color: #0969da; margin: 0 0 8px 0; font-size: 16px; }
.quiz-card p { color: #57606a; font-size: 14px; margin: 0; }
.toolbar { position: fixed; top: 10px; right: 10px; }
.toolbar a { padding: 8px 16px; background: #f0f3f6; border: 1px solid #d0d7de; border-radius: 8px; text-decoration: none; color: #1f2328; font-size: 14px; }
</style>
</head>
<body>
<div class="toolbar">
  <a href="/index.html">返回首页</a>
</div>
<h1>📝 测验系统</h1>
<p>选择一个 Part 开始测验，检验你的学习成果。</p>
<div class="quiz-grid">
'''
    
    for part_num, quiz_data in QUIZ_DATA.items():
        index_html += f'''
<a href="quiz_part{part_num:02d}.html" class="quiz-card">
<h3>Part {part_num}</h3>
<p>{quiz_data['title']}</p>
<p>{len(quiz_data['questions'])} 题 · {len(quiz_data['flashcards'])} 闪卡</p>
</a>
'''
    
    index_html += '''
</div>
</body>
</html>'''
    
    with open(os.path.join(quizzes_dir, 'quiz_index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    try:
        print(f'✅ 测验生成完成 → {quizzes_dir}')
    except UnicodeEncodeError:
        print(f'[OK] 测验生成完成 -> {quizzes_dir}')


if __name__ == '__main__':
    main()
