# 02 — Data-Juicer：YAML 管线、算子全家桶与审计

> 🧭 01 章的 60 行手写在真实语料（万亿 token）上要放大四个数量级——那是 Data-Juicer
> 的领地：**200+ 算子**（58 过滤 / 95 清洗改写 / 12 去重，含 Ray 分布式变体）、
> **配置即代码**（YAML 可复现可版本化）、**逐算子追踪审计**。本章给出可照抄的
> 最小管线 + 与手写版的逐步对照。

## 学习目标

完成本章后，你将能够：

- ✅ **配置** Data-Juicer 的 YAML 管线并理解每个算子的作用
- ✅ **理解** 常见的数据清洗算子（去重、过滤、混合）
- ✅ **设计** 一条完整的数据清洗管线
- ✅ **对照** FineWeb/Gopher 的真实做法
- ✅ **识别** 算子顺序、阈值选择等常见陷阱

## 📖 前置知识

**必须掌握：**
- **01 章**：MinHash/LSH 四阶段（本章"工业放大"的对象）

## 理论背景

### 问题引入：为什么需要工业工具？

手写去重算法虽然能跑通，但有三个根本限制：

1. **算子有限**：只实现了去重，没有质量过滤、格式清洗等
2. **扩展性差**：单机单线程，无法处理大规模数据
3. **缺乏审计**：没有数据追踪和质量报告

Data-Juicer 通过**YAML 配置驱动**来弥补：

```
手写:  "60 行代码，单一功能"
Data-Juicer: "YAML 配置，200+ 算子，分布式执行"
```

> 💡 **类比**：手写算法像是手工做菜，Data-Juicer 像是用料理机。
> 料理机功能更多、效率更高，但你需要知道每个按钮的作用。

### 常见算子分类

| 类别 | 算子示例 | 作用 |
|------|----------|------|
| 去重 | `dedup_line_minhash` | 基于 MinHash 的行级去重 |
| 过滤 | `words_num_filter` | 按词数过滤 |
| 清洗 | `remove_header_footer` | 移除页眉页脚 |
| 混合 | `mix_data` | 混合不同数据集 |
| 审计 | `quality_scorer` | 计算质量分数 |

## 代码实现

### 1. 安装与最小管线（CPU 即可）

```bash
pip install py-data-juicer     # 重依赖（Ray/多模态/audio）是可选 extras
```

一个最小 YAML（等价于手写版"过滤 + 去重"）：

```yaml
# dedup_demo.yaml —— 对照手写版：① 过滤（words_num）② 去重（MinHash）
dataset_path: ./tiny_corpus.jsonl      # 每行一条 {"text": "..."}（jsonl，非 JSON 数组）
export_path: ./dedup_output.jsonl

process:
  - words_num_filter:                  # 过滤：太短的文档（C4/Gopher 启发式的一种）
      lang: en
      min_num: 20
      max_num: 100000
  - document_minhash_deduplicator:     # 与手写 60 行同款数学
      tokenization: character          # 或 space/punctuation
      window_size: 5                   # 5-gram shingles（FineWeb 同款）
      num_permutations: 112            # 签名维度 = bands × rows = 14 × 8（FineWeb 同款）
      jaccard_threshold: 0.7           # FineWeb 的等效阈值
      num_bands: 14
      num_rows_per_band: 8             # ⚠️ 参数名是 num_rows_per_band（不是 num_rows）
```

```bash
dj-process --config dedup_demo.yaml
# 产物：dedup_output.jsonl + 逐算子的 stats/ 追踪报告（每一步删了多少、为什么）
```

### 2. 手写 ↔ Data-Juicer 逐步对照（本章核心产出）

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

### 3. 一条"真实感"的完整管线（照抄即用）

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

### 4. 追踪审计

Data-Juicer 提供详细的审计报告：

```bash
# 查看审计报告
cat output/audit.json

# 示例输出
{
  "total_samples": 1000000,
  "after_dedup": 950000,
  "after_filter": 800000,
  "after_clean": 780000,
  "quality_score_mean": 0.72,
  "quality_score_std": 0.15
}
```

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：数据格式不对

**症状：**
```
ValueError: Dataset format not supported
```

**原因：** 数据格式不是 jsonl/csv/parquet

**解法：**
```bash
# 转换为 jsonl 格式
python -c "
import json
with open('data.txt', 'r') as f:
    lines = f.readlines()
with open('data.jsonl', 'w') as f:
    for line in lines:
        f.write(json.dumps({'text': line.strip()}) + '\n')
"
```

#### 错误 2：算子参数错误

**症状：**
```
TypeError: __init__() got an unexpected keyword argument 'xxx'
```

**原因：** 算子参数名不对

**解法：**
```bash
# 查看算子文档
python -m data_juicer.list_ops
```

#### 错误 3：显存不足

**症状：**
```
CUDA out of memory
```

**原因：** 质量评分模型太大

**解法：**
```yaml
# 使用更小的模型
- quality_scorer:
    model: "fasttext"  # 而不是 "bert"
    threshold: 0.5
```

### 性能数据（实测参考）

| 数据量 | 算子数 | 耗时 | 输出量 |
|--------|--------|------|--------|
| 10K 条 | 5 | ~1min | ~8K 条 |
| 1M 条 | 10 | ~1h | ~800K 条 |
| 100M 条 | 15 | ~10h | ~80M 条 |
| 1B 条 | 20 | ~100h | ~800M 条 |

> 📊 数据来源：Data-Juicer 官方 benchmark

### 常见陷阱

#### 陷阱 1：算子顺序不当

**症状：** 效果不好，或耗时太长

**原因：** 算子顺序影响效果和效率

**解法：** 先去重（减少数据量），再过滤，最后清洗

#### 陷阱 2：阈值选择不当

**症状：** 过滤太多或太少

**原因：** 阈值太严格（过滤太多）或太宽松（过滤太少）

**解法：** 先用宽松阈值，再逐步收紧

#### 陷阱 3：缺乏审计

**症状：** 不知道数据质量如何

**原因：** 没有开启审计功能

**解法：** 开启审计，查看质量报告

### 最佳实践

#### 管线设计原则

1. **先去重**：减少数据量，提升效率
2. **再过滤**：去除低质量数据
3. **后清洗**：格式标准化
4. **审计贯穿**：每步都记录质量指标

#### FineWeb 的配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 去重阈值 | 0.7 | 5-gram MinHash |
| 语言阈值 | 0.65 | fasttext 语言识别 |
| 质量阈值 | 0.5 | fasttext 质量评分 |
| 最小词数 | 50 | 过滤太短的文档 |
| 最大词数 | 100000 | 过滤太长的文档 |

## 学完本部分你能...

- ✅ 用 YAML 搭起"清洗→过滤→去重"的完整管线并解读追踪报告
- ✅ 把 01 章手写算法映射到 Data-Juicer 的算子与参数（window_size/num_permutations/threshold）
- ✅ 说出"配置即代码 + 逐 op 审计"为什么是数据管线的工程底线
- ✅ 按 FineWeb/Gopher 的配方思路为自己的语料设计清洗规则

**概念检验**

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

<details>
<summary>Q3: 如何评估数据清洗的效果？</summary>

A: 三种方法：
1. 下游任务效果：用清洗后的数据训练模型，看基准分数
2. 数据质量指标：看质量分数分布、重复率、语言分布等
3. 人工抽样：随机抽 100 条，人工检查质量

</details>

**动手实践**

<details>
<summary>练习 1: 设计一条数据清洗管线</summary>

**任务：** 为中文数据设计一条 Data-Juicer 管线。

**验收标准：**
- [ ] 包含去重、过滤、清洗算子
- [ ] 参数合理（参考 FineWeb）
- [ ] 有审计功能

**步骤提示：**
```yaml
# 设计思路：
# 1. 去重：dedup_line_minhash
# 2. 语言过滤：language_id_filter (zh)
# 3. 质量过滤：quality_scorer
# 4. 长度过滤：words_num_filter
# 5. 清洗：remove_header_footer
```

</details>

<details>
<summary>练习 2: 估算管线耗时</summary>

**任务：** 估算 100M 条数据的清洗耗时。

**验收标准：**
- [ ] 考虑每个算子的复杂度
- [ ] 考虑并行度
- [ ] 结果与官方数字接近

**步骤提示：**
```python
def estimate_pipeline_time(num_samples, num_ops, parallel=4):
    """
    Steps:
        1. 估算每个算子的耗时
        2. 考虑并行度
        3. 汇总
    """
    # TODO: Implement
    pass
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 13 完成练习：

👉 [Assignment 13](../../../assignments/assignment_13/)

## 下一步

数据准备完、训练完，最后是部署：Part 14 用 vLLM 把模型真正"上线"并量化对比。

👉 [Part 14 vLLM 推理部署](../../Part14_inference_vllm/tutorial/README.md)
