# Part 7 · 手写 minimind：从零到一造一个 26M 的小型语言模型

> 🧭 Part 6 你已经手写过 GPT（字符级、65 词表、莎士比亚）。Part 7 把它**生产化**：
> 换上现代 LLM 的全部零件（BPE / RMSNorm / RoPE / GQA / SwiGLU / MoE），吃**真实数据**
> （官方 minimind mini 语料），走完**预训练 → SFT → DPO** 三阶段，最后端出一个会对话的 26M 模型。
>
> 教法学的是**实验驱动**：不直接写现代架构——先故意用"古董配置"（MHA + 可学习位置表 +
> ReLU + LayerNorm）训一个基线，然后一章一个开关地升级，每一步用实验数字证明升级值得。

## 📋 读者四问（开读前对齐）

- **目标读者是谁**：学过 Part 6（手写 GPT）、会用 PyTorch 的开发者；想搞懂现代开源小模型
  （Llama/minimind 同款架构）每个零件"为什么长这样"的人。
- **前置知识是什么**：必须掌握 Part 6 的自注意力、残差块、训练三行循环；建议了解 Part 3
  （BatchNorm 引出的归一化思想）与 Part 6 04 章（RLHF 图景）。
- **使用场景是什么**：跟读 + 跟跑（每章一个脚本，quick 档几分钟）；面试备战（每章有"面试直通车"）；
  之后衔接 Part 8-19 的任何一支。
- **哪些术语可进白名单**：GPU、token、embedding、softmax、loss、lr——这些 Part 6 已讲透；
  其余术语首现处都有白话解释（完整术语表见文末）。

## 📚 章节导航

| 章 | 标题 | 一句话 | 脚本 |
|---|---|---|---|
| 01 | [BPE Tokenizer](01_bpe_tokenizer.md) | 三难取舍、手推合并、官方词典体检、自训一版对照 | `00` `01` |
| 02 | [基线诞生](02_minimal_model.md) | 古董配置（MHA+learned PE+ReLU+LayerNorm）跑通 28.98M | `02_baseline.py` |
| 03 | [实验仪器](03_experiment_lab.md) | 验证集 ppl + CSV/TensorBoard 记录 → 问题清单 | `03_experiment_lab.py` |
| 04 | [稳定与提速](04_stability_speed.md) | fp32/fp16/bf16 实测、裁剪、累积、DDP | `04_stability_speed.py` |
| 05 | [位置编码进化](05_rope.md) | learned PE 的两个毛病 → RoPE 推导 → 外推实验 | `05_rope.py` |
| 06 | [注意力的两次手术](06_attention_gqa_qknorm.md) | KV Cache 账本与一致性 → GQA → QK-Norm | `06_gqa_qknorm.py` |
| 07 | [FFN 进化](07_ffn_moe.md) | ReLU→SwiGLU 同参对比、MoE 路由/aux/负载均衡 | `my_minimind.py --stage 5` |
| 08 | [组装对账](08_assemble_model.md) | RMSNorm + 参数账本 28.98M→25.83M 对上官方 26M | `my_minimind.py --stage 6` |
| 09 | [阶段一 Pretrain](09_pretrain.md) | 官方数据四步工序 + 全件训练，val ppl 5448→278 | `my_minimind.py --stage 7` |
| 10 | [阶段二 SFT](10_sft.md) | chat 模板渲染 + token 子序列扫描 mask，443→66 | `my_minimind.py --stage 8` |
| 11 | [阶段三 DPO](11_dpo.md) | 隐式奖励 + sum 口径 + 过拟合现场 | `my_minimind.py --stage 9` |
| 12 | [毕业指南](12_reproduce_minimind.md) | 课程脚本 ↔ 官方 trainer 对照、官方超参、成本、进阶实验 | `12` `13` |
| 13 | [MLA 与 NSA](13_attention_mla_nsa.md)（**选修**） | DeepSeek 的两条"省"路线：低秩 KV 与稀疏注意力 | `14_mla_nsa_accounting.py` |

`my_minimind.py` 是单文件"完全体"（一个模型类 + 全部组件开关），07-11 章的实验在它的
`--stage 5..9`；`--profile quick|full` 切换教学档/官方超参档。

## 🗺️ 学习路线图：一条生长链

```text
v0 空文件
 │ 02 基线诞生     官方 BPE 字典 + MHA + learned PE + ReLU + LayerNorm——60 步训起来
 ▼                "先跑通，再变强"
v1 最小可用（28.98M）
 │ 03 实验仪器     验证集 ppl + metric 记录（CSV/tensorboard）+ 超参速查表
 ▼                "没有测量就没有实验" → 拿到问题清单
v2 会做实验的基线
 │ 04 稳定提速     fp32/fp16+GradScaler/bf16 对比 ｜ 裁剪/累积 ｜ 双卡 DDP
 ▼                "快 1.8 倍、稳如老狗"
v3 训练工程齐备
 │ 05 位置编码进化 learned PE 的两个毛病 → RoPE 推导 → θ 深挖 → 外推实验
 ▼                "把绝对查表换成相对函数"
v4a 现代位置编码
 │ 06 注意力手术   KV Cache 账本与一致性验证 → GQA（质量≈无损、缓存÷4）→ QK-Norm
 ▼                "推理省 4 倍，训练保命"
v4 现代注意力
 │ 07 FFN 进化     ReLU→SwiGLU（同参 ppl -10%）→ MoE（路由/直通梯度/aux/负载均衡）
 ▼                "形状换收益，容量解耦算力"
v5 现代 FFN
 │ 08 组装对账     RMSNorm + 初始化 + 参数账本：28.98M → 25.83M，与官方 26M 对上
 ▼                "每一处瘦身都能指出是哪个开关省的"
v6 完整模型（乱码）
 │ 09 阶段一 Pretrain   官方 pretrain_t2t_mini.jsonl，数据四步 + 训练机器全件上阵
 ▼                     val ppl 5448 → 278
v7 会续写（base）
 │ 10 阶段二 SFT        chat 模板渲染 + token 子序列扫描 mask + lr 骤降
 ▼                     val ppl 443 → 66，会一问一答
v8 会对话
 │ 11 阶段三 DPO        隐式奖励 + sum 口径 + ref 冻结 + 泛化监控
 ▼                     训练池 acc 56%→80%，heldout 62%→45%（过拟合现场！）
v9 你的 minimind 🎉    ── 12 章毕业指南：官方仓库对照、成本、进阶实验 ──
```

每章都是同一个节奏：**先跑 → 看到什么问题 → 为什么 → 换成什么 → 实验前后对比 → 论文怎么说 →
工业界怎么做**。所有对比实验由同一份脚本完成，同一份数据、同一个种子——你看到的就是消融实验。

## 📦 数据与依赖（第 0 步）

```bash
cd courses/Part7_minimind/scripts
python 00_download_data.py --sample-only --no-full   # 只要小样本（~48MB，3-5 分钟）
python 00_download_data.py                           # 或全量 2.9GB（可断点续传）
```

| 文件 | 大小 | 行数 | 用途 |
|---|---|---|---|
| `pretrain_t2t_mini[_sample].jsonl` | 1.2GB / 15MB | 127 万 / 2 万 | 09 章预训练 |
| `sft_t2t_mini[_sample].jsonl` | 1.7GB / 17MB | 90 万 / 6 千 | 10 章 SFT |
| `dpo[_sample].jsonl` | 54MB / 16MB | 1.7 万 / 4 千 | 11 章 DPO |
| `tokenizer/` | 0.5MB | — | 官方 6400 词表 BPE（不建议重训，01 章有对照实验） |

- **quick 档**（默认）：sample 数据 + 官方 26M 配置，每章几分钟（RTX 4090 实测）；
  **无 GPU 自动降 toy 档**（hidden 64），只保流程，数字与 GPU 档不可比。
- **full 档**：`--profile full` = 官方默认超参 + 全量数据（09-11 章与官方 argparse 逐项对齐）。
- 环境变量：`P7_STEPS=N` 压步数冒烟、`P7_NO_TB=1` 关 tensorboard、`P7_SWANLAB=1` 开云端记录（默认关）。
- 实验记录恒定写入 `temp/out/logs/`（训练曲线 CSV + tensorboard + 消融证据 `exp*.csv`，
  教程引用的数字都以这里的落盘为准，可重跑复现）。

## 🧰 前置知识

- **必须掌握**：[Part 6 全部内容](../../Part6_transformer/tutorial/README.md)——字符级 tokenizer、
  self-attention、Multi-Head、残差块、pre-norm、decoder-only GPT。每章开头有「🎒 前置回忆包」，
  5 行内自带复习。
- **建议掌握**：[Part 3 的 BatchNorm](../../Part3_batchnorm/tutorial/02_batchnorm.md)（归一化的思想——
  08 章 RMSNorm 和它对照）；[Part 6 04 章的 RLHF 概念](../../Part6_transformer/tutorial/04_beyond_transformer.md)
  （SFT → 奖励模型 → PPO——11 章 DPO 是"消灭 PPO"的替代方案）。

## 📈 Part 6 → Part 7：换零件，不改骨架

| 零件 | Part 6（莎士比亚 GPT） | Part 7（minimind） |
|---|---|---|
| tokenizer | 字符级，词表 65 | **BPE 子词**，词表 6400（官方词典） |
| 归一化 | LayerNorm | **RMSNorm**（08 章上线） |
| 位置编码 | 可学习位置表 | **RoPE** 旋转位置编码（05 章上线） |
| 注意力 | MHA | **GQA** + QK-Norm（06 章上线） |
| FFN | ReLU 两层 | **SwiGLU**（07 章上线）、MoE 可选 |
| 特殊 token | 无 | `<|im_start|>` / `<|im_end|>` chat 格式 |
| 训练 | 纯预训练 | **预训练 → SFT → DPO** 三阶段 |
| 参数量 | ~10M | **~26M**（官方 minimind2-small 口径） |

> 📝 **本课脚本口径（2026-09-19 对照官方源码历史核定）**：GPU 档即官方 **26M（minimind2-small）**
> 配置——hidden 512 / 8 层 / 8 Q 头 / 2 KV 头 / rope θ=1e6 / rms_norm_eps 1e-5 /
> intermediate 1408（`int(d·8/3)` 对齐 64，两代公式演变史见 [07 章](07_ffn_moe.md)）。
> 官方 **64M minimind-3**（现 master）= 8Q/4KV / eps 1e-6 / `ceil(π·d/64)·64`→2432。
> 教程中凡与官方不同处都标注"课程实现 vs 官方"，面试答题建议带版本限定词。

## 🎯 预期数字总览（quick 档，RTX 4090，2026-09 实测，证据在 `temp/out/logs/exp*.csv`）

| 里程碑 | 数字 | 出处 |
|---|---|---|
| 基线第一次训练 | loss 8.87 → 7.23（60 步，≈ln6400=8.76 起步） | 02 章 |
| AMP 三连 | fp32 6.4355 / fp16 6.5192 / bf16 6.5287；tok/s 125k→229k（**1.8×**） | 04 章 |
| RoPE vs learned | 外推 ppl@512：learned 416→431（+3.4%）vs rope 334→319（−4.4%） | 05 章 |
| MHA vs GQA | ppl 363.53 vs 352.44（≈无损），KV 缓存 ÷4 | 06 章 |
| ReLU vs SwiGLU | 同参预算 ppl 310.53 → 278.41（**−10%**） | 07 章 |
| 参数对账 | 28.98M → 25.83M ≈ 官方 26M | 08 章 |
| Pretrain | val ppl **5448.66 → 278.41** | 09 章 |
| SFT | val ppl **442.78 → 66.43**，会一问一答 | 10 章 |
| DPO | 训练池 acc 56.5%→80.5%，heldout 62.0%→45.0%（过拟合现场） | 11 章 |

> ⚠️ Part 7 的 loss 与 Part 6 **不可直接比**：词表从 65 变 6400，初始 loss 反而更高
> （ln6400≈8.8 vs ln65≈4.2）；但子词比字符更好预测，收敛后 per-token loss 往往更低。
> 看趋势，别硬比绝对值。

## ✅ 学完 Part 7 你能

- 手写现代 LLM 的每个零件并说清"为什么"（RMSNorm/RoPE/GQA/SwiGLU/MoE）
- 跑通三阶段训练并解读每条曲线（含 DPO 过拟合的诊断）
- 对照官方 minimind 仓库逐文件读懂 trainer，知道课程版放大成什么
- 用 ppl + needle 检索两层方法评测长上下文外推（12 章）

## 📝 课后作业

[Assignment 7](../../../assignments/assignment_7/assignment.md)：从零复现 minimind 六大组件
（BPE 编码器 / RMSNorm / RoPE / repeat_kv / SwiGLU / DPO loss + KV Cache 拓展题），
与本章脚本一一对应。

## 📖 术语表（首现章节｜全文统一叫法）

| 术语/英文 | 白话机制（条件→动作→结果） | 首现 | 统一叫法 |
|---|---|---|---|
| BPE（Byte Pair Encoding） | 反复合并语料中最高频的相邻字节对 → 高频词变短 token、罕见词拆字节兜底 → 无 OOV 且序列变短 | 01 | BPE |
| OOV（未知词） | 词表里没有的词 → 编码失败 → 词级模型的致命伤 | 01 | OOV |
| 压缩率 | 每个 token 平均代表几个字符 → 越高同算力看到越长上下文 | 01 | 压缩率 |
| 特殊 token | 训练时预留的固定 id（如 `<|im_end|>`）→ 充当结构标记 → 模型靠它学会格式 | 01 | 特殊 token |
| 权重绑定（tie） | 输入查表与输出打分共用同一张矩阵 → 省参数（26M 省 12.6%） | 02 | tie |
| ppl（困惑度） | exp(loss) → 模型每步平均在几个选项里犹豫 → 越低越准 | 03 | ppl |
| AMP（自动混合精度） | 前向/反向用低精度、权重更新用 fp32 → 显存减、速度升 | 04 | AMP |
| GradScaler | fp16 梯度易下溢 → 先放大 loss 再反传、更新前缩回 → fp16 也能稳训 | 04 | GradScaler |
| bf16 | 与 fp16 同宽但指数位更多 → 动态范围大 → 不需要 scaler | 04 | bf16 |
| 梯度累积 | 小 batch 多次前向反向、累积梯度再更新 → 显存不涨而有效 batch 变大 | 04 | 梯度累积 |
| DDP（分布式数据并行） | 每张卡各训一半数据、反向时梯度求平均 → 等效大 batch | 04 | DDP |
| learned PE（可学习位置表） | 每个位置学一个向量加进 embedding → 只见过训练长度 → 外推有硬上限 | 05 | learned PE |
| RoPE（旋转位置编码） | 把位置转成 q/k 的旋转角 → 内积只依赖相对距离 → 可外推、零参数 | 05 | RoPE |
| θ（RoPE 基频） | 频率公式的底（10000/1e6）→ 越大低频维波长越长 → 可分辨更远距离 | 05 | θ |
| KV Cache | 生成时缓存历史 K/V、每步只算新 token → K/V 计算量从每步 O(t) 降到 O(1) | 06 | KV Cache |
| GQA（分组查询注意力） | 多个 Q 头共享一组 K/V 头 → 参数与缓存÷4、质量≈无损 | 06 | GQA |
| QK-Norm | 对 q/k 各做一次 RMSNorm → 注意力打分尺度钉死 → 深层训练稳 | 06 | QK-Norm |
| SwiGLU | FFN 从两投影改三投影+门控 → 同参数预算下表达力更强 | 07 | SwiGLU |
| MoE（混合专家） | 路由器为每个 token 只选 k 个专家 → 总容量×N、单 token 算力不变 | 07 | MoE |
| aux loss（负载均衡损失） | 惩罚"专家贫富分化"的附加损失 → 所有专家都被用到 → 防退化成 dense | 07 | aux loss |
| RMSNorm | 只除均方根、不减均值无 bias → 比 LayerNorm 省算且效果相当 | 08 | RMSNorm |
| loss masking | 把不该学的 token 标成 -100 → 不进损失 → 模型只学回答不学复述 | 10 | loss masking |
| chat 模板 | 把多轮对话渲染成带角色标记的单序列 → 模型学会"轮到谁说话" | 10 | chat 模板 |
| DPO（直接偏好优化） | 用"chosen 与 rejected 的概率比相对参考模型的变化"当奖励 → 不建奖励模型直接调偏好 | 11 | DPO |
| 参考模型（ref） | SFT 权重的冻结副本 → 提供"改动前的基准" → 防 DPO 漂移 | 11 | ref |
| 隐式奖励 | log π(y)−log π_ref(y) → 当前模型相对 ref 多相信这个回答多少 | 11 | 隐式奖励 |
| MLA（多头潜在注意力） | 把 KV 压缩到低秩潜在向量再按需还原 → 缓存再降数倍 | 13 | MLA |
| NSA（原生稀疏注意力） | 训练时就只用压缩/选择/滑窗三分支 → 长上下文计算量大降 | 13 | NSA |

## 🔗 相关资源

- 🐙 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0，本课对照的官方仓库）
- 📦 [数据集 ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) / [HF](https://huggingface.co/collections/jingyaogong/minimind)
- ➡️ 下一站：[Part 8](../../Part8_post_training/tutorial/README.md)（后训练全家桶）｜
  [Part 10](../../Part10_distributed/tutorial/README.md)（分布式深入）｜
  [Part 14](../../Part14_inference_vllm/tutorial/README.md)（用 vLLM 部署你的模型）
