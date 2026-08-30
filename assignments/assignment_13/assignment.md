# Assignment 13：数据工程

> 对应 Part 13 教程（[01 手写 MinHash/LSH](../../courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.md) / [02 Data-Juicer](../../courses/Part13_data_engineering/tutorial/02_data_juicer_pipeline.md)）。
> 四题纯标准库可完成。

## 题目（实现 `data_exercises.py`）

1. **Jaccard/shingling**（25 分）：k-gram 集合 + Jaccard（空集约定）
2. **签名一致率**（25 分）：MinHash 近似的有限样本形态
3. **分带 LSH 概率**（25 分）：`1-(1-J^r)^b` + 给定召回目标反推最小 bands
4. **簇消解**（25 分）：重复对 → 并查集连通簇 → 每簇保留"原顺序最先"（注意传递闭包）

## 实验题（观测型）

- 跑脚本 01，把 `jaccard_threshold` 0.5 降到 0.35：dupB/dupC（重改写）会被抓到吗？
  误报增加了吗？写下你对"阈值-召回-误报"三角的结论
- 装上 Data-Juicer，用 02 章 YAML 跑自己的 tiny_corpus，读一遍 stats 追踪报告里
  "每个算子删了多少"——面试讲数据工程时这就是你的实证

## 🎯 面试直通车

- "LSH 为什么不漏高相似对？"——P[候选|J]=1-(1-J^r)^b，J 高趋近 1；误报靠 Jaccard 验证挡
- "去重放在质量过滤前还是后？"——两流派都讲得出消融理由（02 章练习 Q1）
- "FineWeb 的去重配置？"——5-gram、14 band × 8 row、等效阈值≈0.7、3.3M CPU 小时
- "字面重复 vs 语义重复？"——MinHash 管字面；语义重复要 embedding（成本高一个量级）
