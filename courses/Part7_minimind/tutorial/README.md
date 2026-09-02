# Part 7: 现代 LLM / Minimind — 从 Part 6 的 GPT 到 RoPE、GQA、SwiGLU、DPO

> 🚀 把 Part 6 那个"会写莎士比亚的 mini-GPT"，升级成现代大语言模型（LLM）的完整形态——最终从零复现 minimind，一个约 26M 参数的 tiny LLM。

## 📚 章节导航

| 序号 | 章节 | 内容 | 对应脚本 |
|------|------|------|----------|
| 01 | [BPE Tokenizer](01_bpe_tokenizer.md) | 为什么需要 subword、BPE 算法原理、训练 6400 词表、压缩率对比、chat 格式预告 | `01` |
| 02 | [现代组件：RMSNorm 与 RoPE](02_modern_components.md) | LayerNorm 回顾、RMSNorm、RoPE 旋转位置编码、权重绑定 | `02` |
| 03 | [GQA 与 FFN：SwiGLU、KV Cache、MoE](03_gqa_and_ffn.md) | MHA 回顾、GQA/MQA、KV Cache、Flash Attention、SwiGLU、MoE | `03` `04` |
| 04 | [训练流水线：Pretrain → SFT → DPO](04_training_pipeline.md) | 预训练技巧、SFT + Loss Masking、DPO、完整流水线与部署 | `05` `06` `07` `08` |
| 05 | [复现 minimind 毕业指南](05_reproduce_minimind.md) | 课程脚本 ↔ 官方 trainer 对照、真实数据下载、四阶段超参、验收与成本；进阶实验：RoPE 外推四件套 + 迷你 RULER 长上下文评测 | `11` `13` |
| 06 | [注意力演进：MLA 与 NSA](06_attention_mla_nsa.md) | MLA 低秩 KV 压缩、NSA 三分支稀疏注意力 | `12` |

## 🧰 前置知识

本部分需要你已经掌握：

- **Part 6 全部内容**：字符级 tokenizer、Dataloader、self-attention、Multi-Head、残差连接、LayerNorm（pre-norm）、decoder-only GPT —— Part 7 是在它的骨架上"换零件"
- **Part 3 的 BatchNorm**：归一化的思想、可学习的缩放参数 γ/β —— 讲 RMSNorm 时会和它对照
- **Part 4 的反向传播直觉**："归一化层里的每个参数都是可学习的" —— 讲 RMSNorm 去掉 bias 时用到
- **Part 5 的非线性激活**：tanh —— 讲 SwiGLU 时拿它和 ReLU/tanh 对比
- **Part 6 04 章的 RLHF 概念**：SFT → 奖励模型 → PPO —— DPO 是"消灭 PPO"的替代方案

> 💡 如果你卡住了，随时回看前几章的 `tutorial/` 目录。Part 7 每一章开头都有「📖 前置知识」告诉你该回看哪。

## 🗺️ 学习路线图

```
Part 6 (Transformer / GPT：字符级、LayerNorm、learned PE、MHA、ReLU FFN、预训练)
    │
    │  "架构懂了，但 ChatGPT/GPT-3 里真正的 LLM 用的是另一套组件..."
    ▼
┌─────────────────────────────────────────────┐
│  Part 7: 现代 LLM / Minimind（~26M 参数）    │
│                                             │
│  ① BPE Tokenizer    — 字符级 → 6400 子词    │──→ 01_bpe_tokenizer.md
│  ② 现代组件          — RMSNorm + RoPE        │──→ 02_modern_components.md
│  ③ GQA 与 FFN       — GQA/KV Cache/SwiGLU/MoE│──→ 03_gqa_and_ffn.md
│  ④ 训练流水线       — Pretrain→SFT→DPO       │──→ 04_training_pipeline.md
│  ⑤ 复现毕业指南     — 官方仓库对照/进阶实验    │──→ 05_reproduce_minimind.md
│  ⑥ 注意力演进       — MLA + NSA               │──→ 06_attention_mla_nsa.md
│                                             │
└──────────────┬──────────────────────────────┘
               │
               │  "现在你拥有一个完整的现代 LLM 全家桶..."
               ▼
          等价于 minimind（从零复现完成）
```

## 📦 数据与依赖

**本课程完全自包含**：数据、权重都不需要提前下载，全部脚本可直接跑通——

| 需要的东西 | 说明 |
|------|------|
| 数据 | `data/input.txt`（tiny Shakespeare）已在仓库内，脚本 01–11 都用它（含 09 三阶段验收、10 MoE、11 RoPE 外推）；13 用合成 KV 检索任务（无需数据文件） |
| Python 依赖 | 仅脚本 01 的「真 BPE」需要 [`tokenizers`](https://pypi.org/project/tokenizers/)（已声明在 `requirements.txt`）；未安装时自动回退字符级分词 |
| 预训练权重 | 不需要 —— 所有权重（分词器 `temp/bpe_tokenizer.json`、`ckpt_*.pt`）都由脚本从零训练并自动生成 |

**可选：对照 minimind 官方中文数据与权重**（想用真实中文语料复现原版、或直接加载官方模型时）：

| 内容 | 来源 | 说明 |
|------|------|------|
| 训练数据 | [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) / [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) | 最小复现只需 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`（放进 `./dataset`）；RL 阶段用 `dpo.jsonl` |
| PyTorch 权重（.pth） | [ModelScope](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch) / [HuggingFace](https://huggingface.co/jingyaogong/minimind-3-pytorch) | 官方已训好的各尺寸权重，可直接推理 |
| Transformers 权重 | [ModelScope 合集](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47) / [HuggingFace 合集](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5) | 兼容 HF 生态，可直接 `from_pretrained` 加载 |

```bash
# 下载 PyTorch 权重（两种方式任选其一）
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
git clone https://huggingface.co/jingyaogong/minimind-3
```

> ⚠️ 官方权重依赖 minimind 仓库的模型/分词器代码结构，与本课程从零实现的脚本不完全等价，仅作对照学习，不要与本课程脚本混用。

## 🎯 学完这一部分你能...

- ✅ 手写/训练一个 **BPE tokenizer**，把字符级（65 词表）升级到 6400 子词，理解压缩率与 OOV 的取舍
- ✅ 从零实现 **RMSNorm**，讲清楚"为什么现代 LLM 抛弃了 LayerNorm 的均值中心化和 bias"
- ✅ 从零实现 **RoPE（旋转位置编码）**，理解"旋转正交、相对位置影响内积、可外推"
- ✅ 理解 **权重绑定（tie_word_embeddings）** 为什么能让 embedding 层"不花参数"
- ✅ 从零实现 **GQA（分组查询注意力）** 和 `repeat_kv`，讲清"为什么 KV 头可以比 Q 头少"
- ✅ 实现 **KV Cache**，理解生成时为什么只算最后一个 token、复用什么
- ✅ 从零实现 **SwiGLU FFN**（gate/up/down 三投影），对比 ReLU FFN
- ✅ 理解 **MoE（混合专家）** 的概念、路由器和负载均衡损失
- ✅ 跑通 **Pretrain → SFT → DPO** 完整流水线，讲清 loss masking、Bradley-Terry、参考模型冻结
- ✅ 用 **PI / NTK / YaRN** 给 RoPE 做长上下文外推（含 YaRN 温度因子 √(1/t)=0.1·ln(s)+1），并用迷你 RULER 的 needle 检索引擎验证"读得顺 ≠ 记得住"
- ✅ 对照 minimind 的 `train_tokenizer → train_pretrain → train_full_sft → train_dpo` 全流程

## 📈 演进路线：从"迷你 GPT"到"现代 LLM"

本教程会反复看到这张表——**Part 6 到 Part 7，换的不是架构骨架，而是每一层零件**：

| 维度 | Part 6 的 mini-GPT | Part 7 的现代 LLM（minimind） |
|------|:---:|:---:|
| Tokenizer | 字符级，65 词表 | **BPE**，6400 词表 |
| 归一化 | LayerNorm（mean/var + γ/β + bias） | **RMSNorm**（只算均方根，去 bias） |
| 位置编码 | learned positional embedding（可学习参数表） | **RoPE**（旋转编码，零参数、可外推） |
| 注意力 | MHA，每头独立 K/V | **GQA**，8 Q 头 / 4 KV 头 |
| KV Cache | 无 | **有**（推理加速） |
| FFN | Linear → ReLU → Linear（4×） | **SwiGLU**（gate/up/down，~3.2×） |
| 特殊 token | 无 | `<\|im_start\|>` / `<\|im_end\|>` chat 格式 |
| 训练 | 纯预训练 | **预训练 → SFT → DPO** |
| 参数量 | ~10M | **~26M** |

预训练阶段的损失数字**和 Part 6 不可直接比较**——词表从 65 变 6400，初始 loss 反而更高（`ln6400≈8.8` vs `ln65≈4.2`）；但模型更强、且子词比字符更好预测，**收敛后的 per-token loss 往往略低于字符级**（≈2.0 vs 2.23）。看表请抓住"趋势"而不是硬比数字：

| 阶段 | 目标 | 预期损失（CPU 缩小版，≈） |
|------|:---:|:---:|
| Part 6 最终（字符级 GPT） | 预测下一个字符 | val loss ≈ 2.23 |
| Part 7 预训练（BPE，~26M） | 预测下一个 token | val loss ≈ 2.0，ppl ≈ 7~12 |
| Part 7 SFT | 只对 assistant 回答算 loss | 微调 loss 下降、对话变"像话" |
| Part 7 DPO | 让回答更"讨喜" | 偏好奖励上升、dpo loss 下降 |

> ⚠️ 我们的脚本是 **CPU 缩小版**（更小的 hidden/dim、更少的步数、更短的上下文）。不同超参、不同随机种子，数字都会有差异，所以都带 ≈。**看趋势，别死记数字。** GPU 全量版请参考 minimind 仓库的超参。

## 📝 课后作业

每一章末尾有 2-3 道思考题（`<details>` 折叠答案）。全部学完后，去这里做动手练习：

👉 [Assignment 7](../../../assignments/assignment_7/)

## 🔗 相关资源

- 🐙 [minimind](https://github.com/jingyaogong/minimind) — 本部分复现的目标项目，一个约 26M 参数的 tiny LLM
- 📄 Sennrich et al. 2016：[Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909)
- 📄 Zhang & Sennrich 2019：[Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467)
- 📄 Su et al. 2021：[RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
- 📄 Ainslie et al. 2023：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- 📄 Shazeer 2020：[GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202)
- 📄 Shazeer et al. 2017：[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (MoE)](https://arxiv.org/abs/1701.06538)
- 📄 Rafailov et al. 2023：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/abs/2305.18290)
- 📄 Llama 2 论文（2023）：[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — 现代 LLM 组件的工业级集大成者

---

[← 上一章：Part 6 Transformer](../../Part6_transformer/tutorial/README.md)
