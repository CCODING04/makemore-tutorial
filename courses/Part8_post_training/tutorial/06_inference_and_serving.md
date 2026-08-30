# 06 — 推理与服务：从"能聊"到"能上线"

> 🧭 05 章的 `08_eval_and_chat.py` 让模型能对话了，但"能聊"和"能上线"之间隔着一整门
> 工程学科：**怎么让它更小（量化）、更快（投机解码/批处理）、更省（KV 显存管理），
> 以及怎么用数字向别人证明它快（TTFT/TPOT）**。本章全部概念都有配套实测：
> 跑一遍 [scripts/09_quantize_and_serve.py](../scripts/09_quantize_and_serve.py)（CPU 可跑，2-3 分钟）。

## 📖 前置知识

- **Part 7 03 章**：KV Cache、GQA——本章算 KV 显存时直接用
- **Part 9 02 章**：memory-bound vs compute-bound——本章所有优化技巧的共同原理
- **05 章生成参数**（temperature/top_k）——投机解码建立在它之上

## 0. 一个原理打全部：decode 是 memory-bound

自回归生成的每一步都要把**全部权重**从显存读一遍，算术强度 ≈ 1 FLOP/字节——
远低于 GPU 的平衡点（4090 约 1:150）。所以 decode 阶段 GPU 的计算单元大部分时间在
**等权重到位**。整个推理优化业界的招数都能归结为一句话：

> 🔑 **让每一次权重搬运多干点活**：量化（搬运的字节变少）、批处理（一次搬运服务多个请求）、
> 投机解码（一次搬运验证多个 token）、KV 管理（别让没用的 KV 挤占搬运带宽）。

## 1. 权重量化：int8 / int4

**RTN（round-to-nearest）对称量化**，本课脚本的实现：

```
scale = max|W| / 127            # 逐输出通道（一行一个 scale）
W_int8 = round(W / scale)       # 反量化 Ŵ = W_int8 × scale，误差 ≤ scale/2
```

| 配置 | 有效 bits/权重 | 典型 ppl 代价（7B 级） |
|---|:---:|---|
| fp16 基线 | 16 | — |
| **int8 逐通道** | ~8.1 | **几乎无损**（LLM.int8()：<0.05） |
| int4 分组 g128 | ~4.1 | +0.1~0.3（GPTQ）/ +0.2~0.4（AWQ） |
| int4 无分组 | 4 | 大模型上会崩（离群通道） |

- 💡 **分组的意义**：group size=128 时每 128 个权重配一个 fp16 scale（摊 0.125 bit），
  离群的通道自己一组，不污染邻居。大模型存在少数"离群激活通道"（LLM.int8 的发现），
  逐张量一个 scale 会被离群值拖垮整体分辨率。
- **GPTQ**（Frantar 2022）：逐层用 Hessian 误差补偿——量化一列后把误差摊给未量化的列，
  不用反传、一遍校准数据搞定。**AWQ**（Lin 2023）：发现"重要通道"跟着激活幅值走，
  量化前按激活统计给通道做等价缩放（s=mean|x_act|^α），无反传可扩到 175B。
  一句话对比：**GPTQ 事后补偿误差，AWQ 事前保护重要通道**。
- ⚠️ 本课 2M 小模型的实测（脚本①，500 步训练）：int8 Δ≈+0.4、int4 g128 Δ≈+0.3
  （7B 论文里 int8 <0.05）——**模型越小、训练越不充分，对量化越敏感**；而且小模型没有
  离群通道，int4 分不分组差别不大——离群值是规模涌现的。把训练步数翻倍再看，Δ 会变小：
  量化损伤与训练充分度负相关，这本身就是个可写进面经的观察。

**真实模型量化环境自检**（bitsandbytes 是 CUDA 专用库，装错版本能卡一下午）：

```python
import torch, bitsandbytes as bnb            # 版本要匹配 torch 的 CUDA（cu121↔cu121）
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
        load_in_4bit=True, device_map="auto")  # 一行 4bit；Colab 免费 T4 可跑
```

## 2. KV Cache 显存：GQA 和量化各省多少

```
KV_bytes = 2(K+V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes
```

| 配置（LLaMA-7B 级，seq 2048，bs=1，fp16） | KV 显存 |
|---|---:|
| MHA（kv_heads=32） | 1.07 GB |
| GQA（kv_heads=8） | **0.27 GB** ← Part 7 GQA 的直接意义 |
| GQA + KV int8 | 0.13 GB |
| GQA + KV 2bit（KIVI） | ~0.07 GB |

- 🔑 **KIVI**（ICML 2024）的细节：**K 按通道（feature 维）量化、V 按 token 量化**——
  因为 K 存在少数固定的离群通道（所有 token 一致），必须按通道隔离；V 没有该现象。
  另外保留开头几个 + 最近 token 的全精度窗口（attention-sink）。
- 跑 [脚本②](../scripts/09_quantize_and_serve.py) 的计算器，把你自己模型的数字算出来。

## 3. PagedAttention 与连续批处理（vLLM 的两大支柱）

**问题**：给每个请求按 `max_seq_len` 预留一整块连续 KV，实际生成长度参差 →
vLLM 论文实测 **60-80% 的 KV 显存被浪费**（内部碎片：预留没用完；外部碎片：零散小空隙插不进新请求）。

**PagedAttention**：学操作系统虚拟内存——KV 切成 **16 token/块**，逻辑块表映射物理块，
按需分配、写时复制；相同前缀的请求共享物理块（prefix sharing）。浪费降到 **<4%**。

**连续批处理**（Orca, OSDI'22）：静态 batch 要等最长的请求跑完才能换人（早完成的请求占着
GPU 空转）；iteration-level 调度**每个 decode 步都允许新请求进/完成请求出**，batch 常满。
Orca 报告同延迟下吞吐 **36.9×**（对比 FasterTransformer）。

[脚本③](../scripts/09_quantize_and_serve.py) 用简化模拟量化了这个叙事（64 请求）：
整块预留浪费 **41%** → 分页 **5%**——方向与论文一致（60-80% 来自真实长尾负载 + 外部碎片）。

## 4. 投机解码：用小模型给大模型"代笔"

**机制**（Leviathan et al. 2023）：

```
循环：
  1. draft（小模型）自回归采样 γ 个 token（顺带记下每个位置的分布 pd）
  2. target（大模型）一次前向，并行给出 γ 个位置的分布 pt
  3. 逐个验证：u ~ U(0,1)，u < pt/pd → 接受；否则从 max(0, pt−pd) 重采样并终止
  证明：最终输出分布与 target 单独采样完全一致（无损！）
```

为什么快？memory-bound：验证 γ 个 token 的一次前向 ≈ 生成 1 个 token 的钱（权重只读一遍）。
期望每周期产出 `E = (1−α^(γ+1))/(1−α)`，α 是接受率。

[脚本④](../scripts/09_quantize_and_serve.py) 的实测（draft=减半宽单层，γ=4）：
**α≈0.65 → 实测 2.81 tokens/cycle，理论公式给 2.53**——公式与实测对上了。
⚠️ draft 太弱（α 低）会不赚反赔：draft 的 γ 次前向白花。

## 5. 部署指标与 vLLM 最小实操

| 指标 | 定义 | 谁在乎 |
|---|---|---|
| **TTFT** | 首 token 延迟（prefill 主导） | 用户体验"反应快不快" |
| **TPOT/TBT** | 平均每 token 间隔（decode 主导） | 打字机流式体验 |
| 吞吐 | 全部请求 tokens/s 合计 | 成本 |
| **goodput** | 满足 SLO 的吞吐（如 P99 TTFT≤200ms 且 TPOT≤50ms） | 生产答辩用 |

E2E 延迟 ≈ TTFT + TPOT × (输出 token 数 − 1)。批越大吞吐越高、但 TTFT/TPOT 变差——
这就是 goodput 存在的原因。

**10 行上手 vLLM**（模拟里的概念对号入座）：

```bash
pip install vllm                       # 需 Linux + NVIDIA GPU；Colab T4 可跑 0.5B 模型
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048   # 起一个 OpenAI 兼容服务
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

概念映射：`--max-model-len` → KV 池大小；PagedAttention → 自动开启（看日志 `# GPU blocks`）；
连续批处理 → 服务端自动；`--gpu-memory-utilization` → KV 池占比调参。
对比实验：同一批 100 个请求分别用 `transformers` 生成循环和 vLLM 打服务，测吞吐差（通常 5-20×）。

## 学完本章你能...

- ✅ 用 memory-bound 一句话解释量化/批处理/投机解码为什么有效
- ✅ 手算 KV 显存，说出 GQA 和 KV 量化各省多少
- ✅ 画出 PagedAttention 的块表，解释 60-80%→<4% 的来源
- ✅ 写出投机解码的接受判据和期望产出公式，并复述实测对照
- ✅ 说出 TTFT/TPOT/goodput 的定义与取舍，并起一个 vLLM 服务

**课后练习**

<details>
<summary>Q1: 为什么量化 int8 基本无损而 KV cache 量化更难？</summary>
A: 权重是静态的、可以离线校准分组，逐通道 scale 吸收离群；KV 是运行时逐 token 增长的，
每步都要量化-反量化（有额外 kernel 开销），且 K 的离群是"跨 token 一致的固定通道"，
必须按通道量化（KIVI），实现更绕。另外注意力对 KV 误差更敏感（softmax 后误差被放大）。
</details>

<details>
<summary>Q2: 一个 SLO 要求 P99 TTFT ≤ 200ms。批开得越大越接近还是越远？怎么办？</summary>
A: 越远——批大 → 排队 + prefill 变长 → TTFT 恶化。解法：admission control（限流进 batch）、
chunked prefill（把长 prompt 的 prefill 切片，decode 夹在里面）、按 SLO 分池。
这就是 goodput 指标存在的原因：吞吐要在"满足 SLO 的请求"上算。
</details>

<details>
<summary>Q3: 投机解码在 batch 很大时还赚吗？</summary>
A: 不赚。大 batch 时系统转为 compute-bound（算力饱和），target 一次验证 γ 个 token
不再"顺带免费"，draft 反而抢算力。投机解码是 decode、小 batch、内存墙场景的武器。
</details>

## 📝 课后作业

👉 [Assignment 8](../../../assignments/assignment_8/)（综合题不变）+ 跑通
[scripts/09_quantize_and_serve.py](../scripts/09_quantize_and_serve.py) 的四个实验并记录你的实测数字。

## 下一步

模型上线前还有最后一道关：**怎么科学地知道它变好/变坏了**？
下一章讲评估学——benchmark 污染、LLM-as-judge 的偏差、以及为什么 ppl 不能跨模型比较。

👉 [07 — 评估学：怎么科学地给模型打分](07_evaluation.md)
