# 07 — 评估学：怎么科学地给模型打分

> 🧭 "我们的模型 GSM8K 提升了 3 个点"——这句话可能是真的进步，也可能是**背过题库**。
> 05 章我们跑过 GSM8K mini 评估，但评估的方法论（怎么测、信多少、怎么防作弊）是独立的
> 一门学问。这是对齐/后训练岗面试的常客，也是看论文时必备的"免疫力"。

## 📖 前置知识

- **05 章**：`08_eval_and_chat.py` 的 GSM8K 流程（出题→生成→抽答案→对答案）
- **06 章**：部署视角——评估是上线前的最后一道闸

## 1. 三种评估范式，各有各的坑

| 范式 | 代表 | 优点 | 坑 |
|---|---|---|---|
| **规则评估** | exact match、选择题 loglikelihood、ppl | 便宜、可复现、客观 | 只能测"有标准答案"的能力；容易被背题 |
| **人工评估** | Chatbot Arena（成对盲测→Elo） | 最接近真实偏好，难作弊 | 慢、贵、噪声大 |
| **LLM-as-judge** | MT-Bench、AlpacaEval | 便宜、可规模化 | 评审模型有**位置偏差、长度偏差、自偏好** |

- 🔑 LLM-judge 与人类的一致率其实不低：GPT-4 judge 与人类 ~85%（去平票后），甚至高于
  人类之间的一致率 81%。但必须做**位置交换**（A/B 轮流放前后）和**长度控制**
  （AlpacaEval 2.0 的 length-controlled win-rate）——否则"哪个答案长哪个赢"。
- **ppl 的陷阱**：perplexity 依赖 tokenizer，**不同词表的模型之间不可比**（6400 词表的
  ppl 11 和 15 万词表的 ppl 3 没有可比性）。跨模型请看同一语料的 bits-per-byte（bpc）。

## 2. lm-evaluation-harness：预训练评估的事实标准

EleutherAI 的 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
统一了 300+ 任务，只做两类请求：

- **loglikelihood**：选择题——算每个选项的归一化对数概率，选最高的（注意归一化方式：
  `acc` vs `acc_norm` 按字符长度归一，结果可能差好几个点）
- **generate_until**：生成题——生成后抽取/匹配答案（如 GSM8K 抽最后一个数字）

minimind 官方就是用它报告的（ceval 24.89 / cmmlu 25.38 / arc_easy 28.49 / piqa 50.65）。
复现 minimind 后跑同一套任务，你的模型分数落在 25-30%（多选题≈随机 25%）**是正常的**——
2.9GB 语料训出的 26M 模型本来就不是为刷榜而生的，看的是**相对变化**和 pipeline 是否可信。

## 3. HELM：一次看七个维度

Stanford [HELM](https://arxiv.org/abs/2211.09110) 的贡献是把"单一排行榜分数"拆成
**指标 × 场景矩阵**：accuracy 之外还有 **calibration**（模型说 70% 把握的事是否真有 70% 对）、
robustness、fairness、bias、toxicity、efficiency。面试里能说出"除准确率外我会看校准和
鲁棒性"就已经超出多数候选人。

## 4. Benchmark 污染：不是理论问题，是实测问题

**污染** = 测试题（或其近似）混进了训练数据 → 分数衡量的是记忆力而非能力。

- 🔑 硬证据 **GSM1k**（arXiv 2405.00332）：研究者造了一套"GSM8K 的镜像新题"（同难度同格式，
  确保没泄露），一批开源模型分数应声下跌：**Mistral −8%，Phi −21%**——说明部分"数学能力"
  是背出来的。
- 常规防御：训练数据 vs 测试集做 **n-gram 重叠去污染**（GPT-3 用 13-gram，Llama 用 20-gram）；
  进阶检测：embedding 检索相似题、Min-K% Prob（成员推断）、直接让模型复述测试题。
- 我们 Part 8 的做法诚实版：GSM8K 测试集**只用于评估**、训练用的是算术合成数据——
  但要小心：如果合成数据的题型和 GSM8K 高度同构，也是一种"软污染"。

## 5. 给自己的模型搭一条"最小可信评估线"

不需要大而全，四条就够（都是本课程已有的能力）：

```
1. ppl/bpc：held-out 文本，固定 tokenizer（Part 7 的 09_eval_demo.py）
2. 任务集：GSM8K/CEval 各抽 100 题固定种子（Part 8 的 08 脚本模式）
3. 对照组：每阶段 ckpt 都测（Base/SFT/DPO 分开报数——单点分数无意义）
4. 防污染声明：训练数据与评测集的重叠检查一句话写进实验记录
```

> 💡 面试叙事模板："我评估模型时固定种子与题集、分阶段对照、并做过 n-gram 重叠检查"——
> 这句话的含金量高于"我的模型 GSM8K 到了 X 分"。

## 🎯 面试直通车

<details>
<summary>Q1: 训练数据里混进了 benchmark 原题，模型分数虚高，怎么发现？</summary>
A: 三条路：① 训练语料与测试集做 n-gram（13/20-gram）重叠扫描，命中即污染；
② 行为探测——让模型补全测试题干，能一字不差复述答案的基本是背的；
③ 镜像集对照——GSM1k 思路：同分布新题重测，掉分幅度暴露过拟合程度。
</details>

<details>
<summary>Q2: LLM-as-judge 有哪些系统性偏差？怎么修？</summary>
A: 位置偏差（先出现的占优→交换 A/B 重判）、长度偏差（长答案占优→length-controlled
win-rate）、自偏好（评审模型偏爱自己家族的输出→换多个 judge 交叉）。MT-Bench 的
一致性数据（85% vs 人类 81%）说明可用，但要带这些修正。
</details>

<details>
<summary>Q3: 两个模型 tokenizer 不同，A 的 ppl=3.2、B 的 ppl=8.1，谁好？</summary>
A: 不可直接比。ppl=每 token 平均交叉熵的指数，token 越碎（词表越大）单 token 越好预测，
ppl 天然越低。可比做法：同一份评测语料换算 bits-per-byte（总 NLL / 总字节数再转 2 底指数），
或者干脆用同一批下游任务分数比。
</details>


> 📚 **延伸对照（LLMs-from-scratch）**：rasbt ch07 的 `ollama_evaluate.py` 是 **LLM-as-judge 的最小可运行实现**（本地 Ollama 评审模型打分）——想动手验证本节"LLM-as-judge 偏差"，从它开始。

## 课程总结：你现在的位置

```
Part 1-6  会训练一个语言模型          →  loss 下降曲线
Part 7-8  会走完现代 LLM 全流程       →  一个能对话的模型
Part 9    知道它跑在什么硬件上        →  优化方向的判断力
06 章     知道怎么让它更快更省        →  上线能力
07 章     知道怎么证明它真的更好      →  科研与工程的可信度
```

下一步（课程路线图 docs/course_roadmap_v2.md）：Part 10 分布式训练——单卡装不下模型的那天。

---

[← 上一章：推理与服务](06_inference_and_serving.md) | [Part 8 README](README.md)
