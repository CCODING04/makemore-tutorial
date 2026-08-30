# 02 — Data-Juicer：YAML 管线、算子全家桶与审计

> 🧭 01 章的 60 行手写在真实语料（万亿 token）上要放大四个数量级——那是 Data-Juicer
> 的领地：**200+ 算子**（58 过滤 / 95 清洗改写 / 12 去重，含 Ray 分布式变体）、
> **配置即代码**（YAML 可复现可版本化）、**逐算子追踪审计**。本章给出可照抄的
> 最小管线 + 与手写版的逐步对照。

## 📖 前置知识

- **01 章**：MinHash/LSH 四阶段（本章"工业放大"的对象）

## 1. 安装与最小管线（CPU 即可）

```bash
pip install py-data-juicer     # 重依赖（Ray/多模态/audio）是可选 extras
```

一个最小 YAML（等价于手写版"过滤 + 去重"）：

```yaml
# dedup_demo.yaml —— 对照手写版：① 过滤（words_num）② 去重（MinHash）
dataset_path: ./tiny_corpus.jsonl      # [{"text": "..."}, ...]
export_path: ./dedup_output.jsonl

process:
  - words_num_filter:                  # 过滤：太短的文档（C4/Gopher 启发式的一种）
      lang: en
      min_num: 20
      max_num: 100000
  - document_minhash_deduplicator:     # 与手写 60 行同款数学
      tokenization: character          # 或 space/punctuation
      window_size: 5                   # 5-gram shingles（FineWeb 同款）
      num_permutations: 256            # 签名维度（手写 64 的工业版）
      jaccard_threshold: 0.7           # FineWeb 的等效阈值
      num_bands: 14
      num_rows: 8
```

```bash
dj-process --config dedup_demo.yaml
# 产物：dedup_output.jsonl + 逐算子的 stats/ 追踪报告（每一步删了多少、为什么）
```

## 2. 手写 ↔ Data-Juicer 逐步对照（本章核心产出）

| 手写（01 章 60 行） | Data-Juicer | 放大点 |
|---|---|---|
| `shingles()` 正则分词 | 内置多语种分词（Cython/C++ 加速） | 万亿 token 吞吐 |
| 64 维签名循环 | C++ minhash + 矢量化，`num_permutations: 256` | 精度与吞吐 |
| 单机 dict 分桶 | **Ray 分布式** LSH（`document_minhash_deduplicator` 的分布式变体） | 千节点 |
| keep-first 丢弃 | 簇消解策略 + 可选"保留文本最长的" | 质量导向 |
| print 日志 | **逐 op 追踪**：每个算子前后样本数、被删样本的 HTML 报告 | 可审计（数据管线必须可审计！） |

- 🔑 最值得学的是**"配置即代码"哲学**：YAML 管线像代码一样 review/版本化/复现——
  这正是 Data-Juicer 把 Gopher/C4/FineWeb 式清洗规则做成 50+ 配方（data-juicer-hub，
  含 RedPajama/BLOOM 复现）的原因。
- 💡 彩蛋教学：Data-Juicer 2026-08-28 的最新提交恰是 `document_minhash_deduplicator`
  的 bugfix（空 token 样本）——去重算子至今仍在被打磨，读这个 PR 比读十页文档更能理解
  边界条件（空/超短文档、跨语种）。

## 3. 一条"真实感"的完整管线（照抄即用）

```yaml
process:
  - clean_html_mapper                    # 去 HTML 壳
  - fix_unicode_mapper                   # unicode 归一化
  - language_id_score_filter: {lang: en, min_score: 0.8}   # 语种过滤（fastText）
  - alphanumeric_filter: {tokenization: word, min_ratio: 0.7}  # 符号比异常
  - word_repetition_filter: {rep_len: 10, max_ratio: 0.6}      # 行级复读（LLM 吐复读机的饲料）
  - document_minhash_deduplicator: {...}  # 全局模糊去重（01 章）
```

对照 FineWeb 的叙述：抽取 → 语种 → 启发式质量（Gopher 规则：文档长度/符号词比/停用词比）
→ 全局 MinHash 去重 → 质量分类器。**每一类算子都对应上面一个真实条目。**

## 学完本部分你能...

- ✅ 用 YAML 搭起"清洗→过滤→去重"的完整管线并解读追踪报告
- ✅ 把 01 章手写算法映射到 Data-Juicer 的算子与参数（window_size/num_permutations/threshold）
- ✅ 说出"配置即代码 + 逐 op 审计"为什么是数据管线的工程底线
- ✅ 按 FineWeb/Gopher 的配方思路为自己的语料设计清洗规则

**课后练习**

<details>
<summary>Q1: 为什么去重要放在质量过滤之后？顺序换一下会怎样？</summary>
A: 先去重可省后续计算（同文档只算一次）；但质量过滤可能把"重复簇"删得只剩不同副本，
导致本应整体丢弃的低质量重复被保留一份。主流做法：轻过滤 → 去重 → 重过滤/质量打分
（FineWeb 的顺序），两个方向都有流派，关键是消融证明。
</details>

<details>
<summary>Q2: num_permutations 从 64 提到 256，代价和收益各是什么？</summary>
A: 签名计算与内存 ×4；Jaccard 估计方差更小 → LSH 命中更稳定、阈值附近的行为更平滑。
工业界 128-256 是常见档位；再高收益边际递减。
</details>

## 📝 课后作业

👉 [Assignment 13](../../../assignments/assignment_13/)

## 下一步

数据准备完、训练完，最后是部署：Part 14 用 vLLM 把模型真正"上线"并量化对比。

👉 [Part 14 vLLM 推理部署（拟开）](../../Part14_inference_vllm/tutorial/README.md)
