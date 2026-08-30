# Part 7 / Part 8 优化方案

> **目标**：在不推翻现有内容的前提下，解决三个问题——
> ① 如何更**贴合原仓库**（学生能从零到一真正复现 minimind / train-llm-from-scratch）；
> ② 如何**循序渐进、衔接 Part 1-6 的知识**；
> ③ 在合适的地方**拓展**（通向 Part 9 CUDA、面试、真实工程）。
>
> 优先级沿用项目惯例：P0 必须做 / P1 应该做 / P2 锦上添花。

---

## 一、现状诊断

### Part 7（minimind 复现）

| 维度 | 现状 | 差距 |
|------|------|------|
| 源头贴合 | 8 个脚本覆盖 minimind 六大组件 + 三阶段训练 | 课程用 tiny Shakespeare + 合成对话；minimind 真实链路是**中文语料 BPE → 中文问答 pretrain → 多轮对话 SFT → dpo.jsonl**，学生跑完课程脚本仍不知道"怎么跑原版" |
| 架构对齐 | RMSNorm/RoPE/GQA/SwiGLU 与 minimind 一致，教程注明省略 q_norm/k_norm | 缺一张「课程配置 ↔ minimind config.json」对照表（dim/layers/heads/vocab/rope_base），学生换回原版配置无从下手 |
| 衔接 | README 有 Part 6→7 换零件对照表（好） | ① BPE 用 HF `tokenizers` 库，Part 8 却用 tiktoken，两者关系无交代；② MoE 只在脚本 04 出现，与 dense 主线的关系一句话带过；③ 缺"训练完的模型怎么验收"（Part 8 有 GSM8K + 全阶段对比，Part 7 没有对应的最终评估） |
| 复现通道 | README 列了官方数据/权重下载链接 | 只有链接没有路径：没有「课程脚本 → minimind 四个官方脚本」的对照与迁移步骤 |

### Part 8（train-llm-from-scratch 复现）

| 维度 | 现状 | 差距 |
|------|------|------|
| 源头贴合 | 8 个脚本覆盖原仓库 notebook 全流程，且有 4090 GPU 配置（docs plan） | CPU 缩小版与原仓库 40M 配置之间缺"一键切换"：想跑原版规模要自己改超参 |
| 衔接 | 与 Part 7 独立入口（好），README 声明"重合即复习" | SFT/DPO 两处讲解重复度高，差异化定位（Part 7 = 实现、Part 8 = 推导+全家桶）可以更显性 |
| 拓展 | GRPO→DeepSeek-R1 叙事已有 | ① 缺"训练完之后"的工程视角（量化/vLLM 推理）引子；② 缺与 Part 9 的接口：教程里说"Flash Attention/KV Cache 是内核优化"，但学生不知道去哪学 |

### 共性问题

1. **复现断层**：课程 = 自包含缩小版；原仓库 = 真实数据 + 真实规模。中间缺一条"毕业通道"。
2. **验收缺失**：Part 7 训完 SFT/DPO 后只有脚本打印的 loss/示例输出，没有量化验收。
3. **面试衔接缺失**：Part 7/8 的知识点（RoPE/GQA/KV Cache/DPO/GRPO）全是 LLM 岗位高频面试题，但没有从"学会"到"会答"的桥。

---

## 二、优化项清单

### P0（必须做）

#### P0-1 给 Part 7 加「复现 minimind 毕业指南」

在 `courses/Part7_minimind/tutorial/` 新增 `05_reproduce_minimind.md`（或并入 04 章末尾），内容固定四步：

```
第 0 步 对照表：课程脚本 ↔ minimind 官方文件
  01_bpe_tokenizer.py     ↔ train_tokenizer.py（TrainingCorpus → 中文语料）
  05_full_model.py        ↔ model/model_minimind.py（MiniMindConfig 逐字段对照）
  06_pretrain_pipeline.py ↔ train_pretrain.py（pretrain_hq.jsonl，512 seq）
  07_sft_training.py      ↔ train_full_sft.py（sft_512.jsonl，im_start/im_end）
  08_dpo_alignment.py     ↔ train_dpo.py（dpo.jsonl，max_length 512）

第 1 步 数据：ModelScope/HF 下载 pretrain_hq_mini / sft_512_mini / dpo（给出最小子集）
第 2 步 超参对照表：dim=512/768、n_layers=8/16、vocab=6400、lr≈1e-4→3e-5 cosine、
        batch、epochs；标注课程缩小版各项的"放大方向"
第 3 步 验收：加载官方 .pth 权重（或自己训的）做对话冒烟测试 + 困惑度对比
```

> 验收标准：学生不用读 minimind 源码，也能照指南在 GPU 上跑通官方四阶段。

#### P0-2 给 Part 8 加「一键切到原版规模」配置

`02_pretrain.py`（及 03-08）已有 CPU/GPU 双模式，但 GPU 模式内部超参与原仓库不一致。补一张
`configs` 表（脚本内 dict 或教程表格）：

| 配置 | 参数量 | 数据 | 来源 |
|------|--------|------|------|
| cpu-toy（默认） | ~2M | 合成 | 课程自制 |
| gpu-course | ~40M | tiny Shakespeare + 合成 SFT/偏好对 | 课程自制 |
| **gpu-original** | ~40M（embed=560? 按原仓库） | The Pile / Alpaca / HH-RLHF / GSM8K | train-llm-from-scratch |

至少在 `08_eval_and_chat.py` 与教程 05 章给出"原版数据模式"的开关说明，让 `--data original`（或改一个常量）即可对齐原仓库实验。

#### P0-3 Part 7 补「最终验收」：加一个评估出口

新增 `courses/Part7_minimind/scripts/09_eval_demo.py`（~30s CPU）：
- Base（pretrain 后）vs SFT 后 vs DPO 后，同一批 prompt 的生成对比（脚本自动保存 ckpt 已具备）
- 对 `data/input.txt` held-out 部分算 ppl，画出三阶段对比
- 教程 04 章末尾链接它，对应 Part 8 的 `08_eval_and_chat.py`（两章形成对称的"验收"）

### P1（应该做）

#### P1-1 BPE 双实现对照小节（Part 7 第 1 章 + Part 8 第 1 章各加 5-10 行）

- Part 7 用 HF `tokenizers`（训练 BPE，词表 6400）
- Part 8 用 `tiktoken`（现成 GPT-2 词表 50257/50304）
- 讲清：BPE 是算法；HF tokenizers 能"训练"新词表；tiktoken 只能"用"已发布的词表；sentencepiece 是另一个工业实现（Llama 系）。加一张 3 列对照表。
- 呼应 Part 6 字符级：字符级 = 词表最小/序列最长；BPE 词表↑序列↓；压缩率实验已在 Part 7 01 章存在，补一句"tiktoken 对同一段文本的压缩率"让学生自己算。

#### P1-2 Part 7→8 重叠内容的差异化导览

在两边 README 各加一小节「Part 7 / Part 8 怎么选」：
- 只学一遍：时间少选 Part 8（推导全、算法全）；想懂"现代 LLM 每个组件"选 Part 7
- SFT：Part 7 = 全 token loss（简化），Part 8 = prompt masking（正确姿势）——Part 7 的对应位置加 ⚠️ 指向 Part 8 02 章
- DPO：Part 7 = 能跑的最小实现；Part 8 = 完整推导 + ORPO/KTO + 参考模型冻结细节
- 两边互相挂"另一视角"链接（单向已有，补成双向）

#### P1-3 组件替换 diff 式讲解（Part 7）

GQA、SwiGLU 等章在"从 MHA 到 GQA"处，除概念外增加**代码 diff 块**（只改哪几行），与 Part 6 脚本建立 diff 引用：
```
- self.k_proj = nn.Linear(n_embd, n_embd, bias=False)     # MHA：8 头各配 K
+ self.k_proj = nn.Linear(n_embd, n_kv_heads * head_dim, bias=False)  # GQA：4 头共享
+ kv = repeat_kv(kv, self.n_rep)                            # 关键新增
```
降低"新组件 = 全新代码"的认知负荷（README 表格保留，作为总览）。

#### P1-4 显存/耗时表进 README（Part 7 / Part 8）

把 Part 8 plan 文档里的显存估算表（40M/bf16 ≈ 2-3GB 等）摘进各 README「数据与依赖」处，并给 Part 7 补等价表（26M fp32/fp16 各多少）。学生在自己显卡上跑之前能预判。

#### P1-5 每章「面试直通车」小节（Part 7 / Part 8 全部章节）

每章末尾 3-5 条该章知识点的**面试问法**（不写答案，指向正文小节），例：
- Part 7 02 章：「RoPE 相对位置性质怎么证明？」「RMSNorm 为什么去掉均值中心化不掉点？」
- Part 8 04 章：「GRPO 和 PPO 的显存差异来自哪？」「为什么 RLVR 不需要奖励模型？」
- 汇总入口放两 README（并在项目级文档给完整面试地图，见本次交付的最终总结）。

#### P1-6 与 Part 9 的接口（Part 7 03 章 / Part 8 05 章）

KV Cache、Flash Attention、bf16 处只留一句"这是 GPU 内核层面的优化 → Part 9 会亲手写"，并挂链接。避免新增章节，只补"去哪学"。

### P2（锦上添花）

#### P2-1 GPU 冒烟脚本

`courses/Part7_minimind/scripts/run_gpu_smoke.sh`（Part 8 同）：一条命令在 4090 上按 GPU 配置跑 01→08/09 全流程（每脚本限步数），输出每步耗时/显存，供"我显卡行不行"快速自检。

#### P2-2 原仓库 commit/版本固定

两个 README 注明"内容基于 minimind @ commit vX / train-llm-from-scratch @ 2024-xx 版本"，防止上游漂移导致对不上（minimind 更新很勤）。

#### P2-3 assignment 实验题升级

assignment_7/8 各补 1 道"跑实验"题（改 β / 改 G 个采样数 / 关掉 prompt masking 对比 loss 曲线），提交一段结论而非代码——训练学生做 ablation 的习惯（面试加分项）。

---

## 三、执行顺序建议

```
1. P0-3（Part 7 验收脚本，半天） → 立刻补齐"学完有个交付物"
2. P0-1（minimind 毕业指南，半天）
3. P0-2（原版规模配置，1-2 小时）
4. P1-1 / P1-2 / P1-6（文档级小改动，合计半天）
5. P1-3（diff 式讲解，半天，重写两章局部）
6. P1-4 / P1-5（表格+面试小节，合计半天）
7. P2 按需
```

> 原则：**不重写已有章节主线**，所有优化以"新增小节/文档/脚本"为主，保证已通过双重审查的正文不被破坏。
