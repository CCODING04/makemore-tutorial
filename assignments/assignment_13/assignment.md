# Assignment 13：数据工程

> 对应 Part 13 教程（[01 手写 MinHash/LSH](../../courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.md) / [02 Data-Juicer](../../courses/Part13_data_engineering/tutorial/02_data_juicer_pipeline.md)）。
> 四题必做 + 一题 🌟 Stretch，纯标准库可完成。实现 `data_exercises.py` 后运行
> `python test_data_exercises.py`（或 pytest）。

## 📊 分值表

| 题号 | 主题 | 分值 | 对应测试 |
|------|------|------|----------|
| 1 | Jaccard / shingling | 25 | `test_ex1_jaccard` |
| 2 | 签名一致率（MinHash 近似） | 25 | `test_ex2_signature` |
| 3 | 分带 LSH 概率 + 反推 bands | 25 | `test_ex3_lsh_prob` |
| 4 | 簇消解（keep-first） | 25 | `test_ex4_cluster` |
| 🌟 5 | 簇消解（keep-longest，Stretch） | 附加 10（总分封顶 100） | `test_ex5_stretch`（未实现自动 SKIP ⏭️） |

## 题目（实现 `data_exercises.py`）

### 题 1：Jaccard 与 shingling（25 分）

k-gram 集合 + Jaccard（空集约定）。

**验收标准：**
- [ ] `shingles("The quick brown fox!", k=2)` 返回 set/frozenset，大小为 3，含 `"quick brown"`、`"brown fox"`
- [ ] 大小写与标点归一：`"The"` 参与的 shingle 与 `"the"` 一致（只留 `[a-z]+` 词）
- [ ] `jaccard({"a","b"}, {"a","b"}) == 1.0`；`jaccard({"a"}, {"b"}) == 0.0`
- [ ] `jaccard({"a","b","c"}, {"b","c","d"}) == 0.5`
- [ ] 两个空集返回 `1.0`，不抛 `ZeroDivisionError`

### 题 2：签名一致率（25 分）

MinHash 近似的有限样本形态：签名一致比例 ≈ Jaccard。

**验收标准：**
- [ ] `signature_agreement([1,2,3], [1,2,3]) == 1.0`
- [ ] `signature_agreement([1,2,3], [9,9,9]) == 0.0`
- [ ] `signature_agreement([1,2,3,4], [1,9,3,9]) == 0.5`
- [ ] 不依赖 numpy/torch（纯标准库）

### 题 3：分带 LSH 概率（25 分）

`P(成为候选) = 1-(1-J^r)^b`，并给定召回目标反推最小 bands。

**验收标准：**
- [ ] `lsh_hit_probability(1.0, 16, 4) == 1.0`；`lsh_hit_probability(0.0, 16, 4) == 0.0`
- [ ] `lsh_hit_probability(0.5, 3, 2) == 1 - 0.75**3`（手算可验）
- [ ] `choose_bands_for_recall(1.0, 4, 0.99) == 1`
- [ ] `choose_bands_for_recall(0.9, 4, 0.99)` 返回**最小**的 b：`P(0.9, b, 4) >= 0.99` 且 `P(0.9, b-1, 4) < 0.99`
- [ ] `j<=0` 或 `j>=1` 的边界返回 1（不死循环）

### 题 4：簇消解 keep-first（25 分）

重复对 → 并查集连通簇 → 每簇保留"原顺序最先"。**注意传递闭包**。

**验收标准：**
- [ ] `["a","b","c","d"]` + `[("a","b"),("b","c")]` → `(["a","d"], ["b","c"])`（a-b、b-c 传递成簇 {a,b,c}）
- [ ] `["x","a","b"]` + `[("a","x")]` → `(["x","b"], ["a"])`（对的书写顺序无关，保留列表顺序最前的 x）
- [ ] kept 保持 `doc_names` 原顺序
- [ ] 无重复对时全部保留、dropped 为空

### 🌟 题 5（Stretch，附加 10 分）：带偏好的簇消解 keep-longest

题 4 的 keep-first 有个工业上真实的缺陷：**重复簇里最靠前的未必质量最好**——
转载页常比原文更长更完整。Data-Juicer 的簇消解因此支持"保留文本最长的"
（02 章对照表第 4 行）。实现 `keep_best_per_cluster(doc_names, doc_lengths,
duplicate_pairs)`：每簇保留**长度最长**的文档；长度并列时保留**原顺序最先**的。

**验收标准：**
- [ ] 传递闭包正确（与题 4 同款输入语义）
- [ ] 簇 `{"a","b","c"}` 长度 10/5/20 → 保留 `c`，丢弃 `a`、`b`
- [ ] 长度并列（8 vs 8）→ 保留 `doc_names` 中先出现者
- [ ] kept / dropped 均保持 `doc_names` 原顺序
- [ ] 无重复对 → 全部保留
- [ ] 未实现（`return None`）时测试显示 `⏭️ SKIP` 而非 ❌ FAIL

**步骤提示：** 复用题 4 的并查集；按原顺序遍历簇成员，当前最优只被**严格更长**者替换
（遍历有序 + 严格大于 ⇒ 并列自动保留先出现者，无需额外比较器）。

## 实验题（观测型）

- 跑脚本 01（`courses/Part13_data_engineering/scripts/01_minhash_dedup.py`），把 L103/L108
  两处 `>= 0.5` 的判重阈值降到 `>= 0.35`：dupB/dupC（重改写）会被抓到吗？
  （提示：实测 J(doc03,dupC)=0.45、J(doc00,dupB)=0.24——一个会被抓、一个仍不会，
  这正是"阈值-召回-误报"三角的直观演示）误报增加了吗？写下你的结论
- 装上 Data-Juicer，用 02 章 YAML 跑自己的 tiny_corpus，读一遍 stats 追踪报告里
  "每个算子删了多少"——面试讲数据工程时这就是你的实证

## 🤔 思考题

**Q1：** 脚本 01 实测：LSH 候选 3 对、Jaccard 验证后剩 2 对——多出的 `('doc03','dupC')`
真实 Jaccard 只有 0.45（阈值之下）。这是 bug 吗？如果删掉最后的 Jaccard 验证步，会发生什么？

<details>
<summary>💡 答案</summary>

不是 bug，是设计。LSH 分带的概率 `P(候选|J)=1-(1-J^r)^b` 对中等相似度（0.4-0.6）
并不接近 0——doc03/dupC 的 64 维签名采样一致率是 0.53，碰巧撞上某个完全相同的 band
就成了候选。LSH 的哲学是**用高召回换误报**（宁可多报、绝不漏报：真重复 2 对零漏召），
确定性交给验证步兜底（精算 Jaccard=0.45 < 0.5 → 拦下，最终零误报）。
若删掉验证步：候选对直接判重 → doc03 或 dupC 会被**误删**，语料凭空少一篇好文档；
在十亿级语料上，这类误删会成批发生——所以工业管线（FineWeb/Data-Juicer）全部保留
"粗筛 + 精验"两段。

</details>

**Q2：** 题 4 为什么必须求传递闭包（并查集/BFS）？"对每个重复对直接丢掉其中后一个"
会错在哪？举一个具体反例。

<details>
<summary>💡 答案</summary>

反例：`names=["a","b","c"]`，`pairs=[("a","b"),("c","b")]`。"逐对丢第二个"会丢 `b`（第一对）
和 `b`（第二对，已丢）→ 保留 `a` 和 `c`。但 a-b、c-b 通过 b 传递成**同一簇** {a,b,c}，
正确结果只保留一个（`a`）。根因：重复关系是**等价关系**（对称 + 传递），传递性使
"对"的局部信息必须合并成"簇"的全局信息，否则同簇文档会漏删。并查集是求连通分量的
标准做法，near-O(α(n)) 的合并/查询让它在亿级重复对上仍然可行。

</details>

**Q3：** 题 1 约定 `jaccard(∅, ∅) = 1.0`（避免除零）。这个约定在什么真实场景下会
造成事故？（提示：想想词数少于 k 的文档）

<details>
<summary>💡 答案</summary>

短文档边界。词数 < k 的文档 shingle 集合是**空集**（`range(len(words)-k+1)` 为空），
于是**所有短于 k 个词的文档两两 Jaccard = 1.0**，被判"完全重复"互相吞并——实测
`jaccard(shingles("hi there"), shingles("ok bye")) == 1.0`，两篇毫无关系的短文被判重。
在爬虫语料里短文档（导航页、空壳页）海量，事故形态是"去重一步删掉一大片"。
解法：shingling 前先过滤超短文档（工业界 `words_num_filter` 的 `min_num`），
或对空集合返回哨兵值表示"不可比较"。

</details>

**Q4：** FineWeb 配置 14 band × 8 row 的"等效阈值 ≈ 0.7"是怎么来的？如果把 rows
从 8 改成 4（保持签名维度 112，即 28 band × 4 row），管线的召回-误报会怎么移动？

<details>
<summary>💡 答案</summary>

S 曲线拐点近似 `(1/b)^(1/r)`：`(1/14)^(1/8) ≈ 0.72`——相似度
高于它的对大概率成候选、低于的大概率被过滤。改成 28×4 后拐点 ≈ `(1/28)^(1/4) ≈ 0.44`：
阈值左移，**召回变高**（更低相似度的对也被抓到）但**误报变多**（验证步工作量上升，
且 0.4-0.5 的"转载改写"会被判重——是不是想要的取决于业务）。b/r 是同一枚硬币的两面：
b 升召回、r 升严格度，乘积受签名维度预算约束。

</details>

## 🎯 面试直通车

- "LSH 为什么不漏高相似对？"——P[候选|J]=1-(1-J^r)^b，J 高趋近 1；误报靠 Jaccard 验证挡
- "去重放在质量过滤前还是后？"——两流派都讲得出消融理由（02 章练习 Q1）
- "FineWeb 的去重配置？"——5-gram、14 band × 8 row、等效阈值≈0.7、3.3M CPU 小时
- "字面重复 vs 语义重复？"——MinHash 管字面；语义重复要 embedding（成本高一个量级）
- "簇消解保留哪一篇？"——keep-first / keep-longest / 带质量分的 keep-best（🌟 题 5 就是它）
