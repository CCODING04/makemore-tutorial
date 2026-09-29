# 08 — 组装对账：从 28.98M 的基线到 25.83M 的官方口径（v6）

> 🧭 零件的进化到本章收官。最后一块是归一化（LayerNorm→RMSNorm），然后把所有开关打开，
> **回到 02 章那四行账对账**：基线 28.98M → 现代版 25.83M，与官方 26M 口径对上——每一处
> 瘦身都能指出是哪个开关省的。最后跑通生成循环，它会输出乱码——**这是好事**：器官齐全、
> 还没学过话，学习是 09-11 章的事。

## 🎒 前置回忆包（不翻旧章也能读）

- **LayerNorm（基线位）**：`γ·(x−μ)/√(σ²+ε) + β`——均值中心化 + 方差缩放 + 可学习缩放偏置。
- **pre-norm 残差块**：`x = x + 子层(归一化(x))`，02 章以来没变过。
- **02 章的四行账**：embedding 3.277M (11.3%) ｜ pos_emb 0.524M ｜ attention 8.389M (28.9%)
  ｜ ffn 16.777M (57.9%)，合计 **28.98M**。

## 📍 你现在的位置

```text
v5 现代 FFN ──本章──▶ v6：开关全开的完整模型 25.83M（官方口径 ✓，生成乱码）
```

## 第 1 件：RMSNorm——LayerNorm 的"减法"

```python
class RMSNorm(nn.Module):
    """LayerNorm 减掉均值中心化(−μ)和偏置(β)，只留均方根缩放。"""
    def __init__(self, dim, eps=1e-5):
        self.weight = nn.Parameter(torch.ones(dim))          # 只有缩放，没有 bias

    def forward(self, x):
        return (self.weight * x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        #            ↑ fp32 里算再转回——bf16 下直接算会丢精度，一行代码里的混合精度意识
```

- **为什么减**：RMSNorm 论文的消融显示均值中心化对 Transformer 收益可忽略；减掉后每处 norm
  省 2×d 参数、少一步规约。**"工业界为什么这么用"的最干净样本：不是加法堆出来的创新，是
  减法减出来的效率。**（RMSNorm, Zhang & Sennrich 2019）
- 实验位：本章开关 `norm='rms'` 打开后，02 章 `make_norm` 工厂直接换零件——模型骨架依旧不动。

### 📐 RMSNorm 数学

公式逐项对着上面代码看：

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \varepsilon}} \odot \gamma$$

- `mean(x²)`：对 hidden 向量把每个元素的平方取平均——**注意不先减均值**；
- 开根号 + ε：得到激活的均方根（RMS），ε 只防除零；
- 除以它：整行的"尺度"归一到 ≈1；
- `weight`：可学习缩放 γ（不再有 β/bias）。

**手算：x = [1, 2, 3]**（忽略 γ 和 ε，右边同时算 LayerNorm 同输入做对照）：

```text
RMSNorm:                              LayerNorm（同样输入）:
  mean(x²) = (1+4+9)/3 = 14/3 ≈ 4.667   mean = (1+2+3)/3 = 2
  rms = √4.667 ≈ 2.160                   var  = (1+0+1)/3 = 2/3 ≈ 0.667
  y = x/2.160 ≈ [0.463, 0.926, 1.389]    y = (x−2)/√0.667 ≈ [−1.225, 0, 1.225]
  验证：输出自身的 RMS = √((0.463²+0.926²+1.389²)/3) = 1 ✓
```

- 💡 差异一眼可见：LayerNorm 先减均值，输出**必然以 0 为中心**（有正有负）；RMSNorm
  **保持原来的正负形状**，只把"尺度"压成 1——对激活本身的值更忠实，这正是"均值中心化
  信息量低、砍掉它"的直观体现。
- ⚠️ 手算忽略了 γ（初始化为 1，训练中自己学）；真实模型 `y = γ ⊙ (x/rms)` 逐元素缩放，
  与 LayerNorm 的 γ 作用相同。

| | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | `(x−μ)/√(σ²+ε)·γ + β` | `x/√(mean(x²)+ε)·γ` |
| 减均值 | 有 | **没有** |
| 可学习参数 | γ + β | **只有 γ** |
| bias | 有（可关） | **没有** |
| 计算量 | 算 mean + var | **只算 mean(x²)** |
| 训练稳定性 | 好 | **更好**（大模型普遍选择） |

**为什么更好**——三条，对应表里每一行差距：

1. **均值中心化在 Transformer 里信息量低**：隐藏层经残差连接多次混合，均值项几乎不携带
   有用信息，去掉后性能几乎不变（原论文 LayerNorm↔RMSNorm 对照实验相当）。
2. **省计算**：少一次减均值、少一次规约，每处 norm 少 2×d 参数（β 没了）——单处是小钱，
   但 block 里 2 处 × 8 层累积，超大规模下前向/反向的节省可观。
3. **训练更稳**：数值只依赖"平方的均值"，函数更平滑、梯度更干净——Llama 系全线选它的
   原因之一。

（q/k 上再各加一层 `head_dim` 维 RMSNorm 的"QK-Norm"，06 章已专节做过造病实验。）

## 第 2 件：三处"全线的沉默选择"——bias=False、dropout=0、normal(0.02)

```python
self.apply(self._init)                       # HF 基类默认：所有 Linear/Embedding
nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

| 选择 | 官方/现代 LLM | 为什么 |
|---|---|---|
| Linear 全线 `bias=False` | Llama 起惯例 | 省的参数是小钱；真收益是数值行为更干净、kernel 可融合 |
| `dropout=0.0` | 同上 | 现代 LLM 数据量 ≫ 参数量，正则交给数据多样性；dropout 只拖慢收敛 |
| 初始化 `normal(0, 0.02)` | HF 基类默认 | std=0.02 让初始 logits 方差落在合理区间，初始 loss ≈ ln(词表)——02 章的 8.76 就是它 |

## 第 3 件：参数对账——28.98M → 25.83M，每克脂肪都有出处

`--stage 6` 实测（对照 02 章的四行账）：

```text
基线（02 章）合计 28.98 M：            现代版（开关全开）合计 25.83 M：
    embedding    3.277 M  ( 11.3%)        embedding    3.277 M  ( 12.7%)
    pos_emb      0.524 M  (  1.8%)       attention    5.244 M  ( 20.3%)   ← GQA 砍掉 3.15M
    attention    8.389 M  ( 28.9%)        ffn         17.302 M  (  67.0%)   ← SwiGLU 守恒略胖
    ffn         16.777 M  ( 57.9%)        norm         0.009 M  (  0.0%)   ← RMSNorm 减法
    norm         0.017 M  (  0.1%)
总参数: 25.83 M                        ← "官方 ~26M" 的构成，对上了 ✓
```

对账清单：**pos_emb −0.52M**（RoPE 零参数）、**attention −3.15M**（GQA 的 K/V 砍半再砍半）、
**ffn +0.53M**（SwiGLU 守恒公式的 2.75d 略胖于 4d 的 2 倍投影）、**norm −0.008M**（减法）。
**净减 3.15M，还能指出每一克的去向**——这就是 02 章"先记账"的回报。

- 🔑 小模型参数大头在 FFN（67%）——"加容量"优先动 FFN（07 章的 MoE）而不是注意力。
- 💡 这笔账更上层的用法：**参数量定了，数据要多少？** Chinchilla 经验法则"每参数 ≈20 token"
  → 26M × 20 ≈ 5.2 亿。带着这个数去 09 章对账（官方 mini 语料 ×2 epoch 恰好在量级上）；
  把这条线拉长就是 [Part 13 Scaling Laws](../../Part13_data_engineering/tutorial/00_scaling_laws.md)。

## 第 4 件：生成循环——temperature / top-k / 停止

```python
@torch.no_grad()
def generate(self, input_ids, max_new_tokens=48, temperature=0.85, top_k=50, eos_id=None):
    for _ in range(max_new_tokens):
        idx = input_ids[:, -self.cfg['max_pos']:]         # 滑窗：最长只喂 max_pos
        logits = self(idx)[0][:, -1, :] / temperature     # 温度：<1 更保守，>1 更放飞
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')       # top-k：只留前 50 个候选
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)   # 按概率采样，不是贪心 argmax
        input_ids = torch.cat((input_ids, nxt), dim=1)
        if eos_id is not None and nxt.item() == eos_id:   # <|im_end|>：10 章训练后才会触发
            break
    return input_ids
```

- 为什么采样不取 argmax：贪心会陷入重复循环；温度+截断让生成"有 Railings 的随机"。
- 这版每步整段重算（教学版）。`idx[:, -max_pos:]` 这行滑窗对 learned PE 是**硬上限**（05 章
  的表长问题），对 RoPE 则只是效率问题——工业生成有 KV Cache/分页/连续批处理，06 章的
  GQA 就是为它省显存的。

🏭 **顺便认识你刚写出的东西**：这个"config 字典 + 模型类"就是 HuggingFace
`PreTrainedModel + config.json` 的手工版。官方 `MiniMindConfig(PretrainedConfig)` 能被
`from_pretrained` 加载、被 vLLM/llama.cpp 转换——因为生态约定了"配置与权重分离、字段名对齐"。
你自己的 minimind 想进生态，照抄字段名即可。

## ▶️ 运行本章成果

```bash
python my_minimind.py --stage 6
```

```text
基线（02 章）合计 28.98 M：…（见第 3 件）
现代版（05-08 章开关全开）合计 25.83 M：…（见第 3 件）
对照: 官方 26M 口径 ≈25.8M ✓
随机权重生成: '<|im_start|> land文件这有助于� personal topics邀知异常ording…'
↳ v6 达成：你的 minimind 完全体——架构现代、身材对账官方。训练在 09-11 章。
```

- 🔑 欣赏这段乱码：中英混杂、词形完整（token 层面是对的），语义为零。**"会说 token 层面的
  话"是架构给的，"说人话"是数据给的**——下一章开始训练。

## 🪝 引子

- 仪器（02）、速度（03）、现代零件（04-07）全齐——下一章把**全部工程件**投入第一次真正的
  预训练，目标 val ppl 从 5448 压到多少，跑完见分晓。

## 🎯 面试直通车

<details>
<summary>Q: 现代 LLM 相比 GPT-2 时代的 Transformer，"骨架"变了吗？</summary>
A: 骨架没变：pre-norm 残差块、注意力+FFN、自回归生成，Part 6 的结构原样成立（本课一个模型
类走到底就是证明）。变的全是零件：归一化 LN→RMSNorm、位置 learned→RoPE、注意力 MHA→GQA
(+QK-Norm)、FFN ReLU→SwiGLU→MoE、bias/dropout 关闭。所以"读懂一只模型"的正确姿势是列零件
清单对比，config 里每一项都能对着报。
</details>

## ✅ 本章验收

- [ ] `--stage 6` 跑通，能指着对账清单说出 3.15M 各去了哪
- [ ] 能解释 RMSNorm"减"掉了什么、为什么现代模型敢关 dropout
- [ ] 能说出随机权重生成"像话又不通"的原因（token 层面对 vs 语义层面空）

---

[← 上一章：FFN 进化](07_ffn_moe.md) | [课程 README](README.md) | [下一章：阶段一 Pretrain →](09_pretrain.md)
