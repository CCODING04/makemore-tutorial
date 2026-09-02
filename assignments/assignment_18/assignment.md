# Assignment 18：RAG 全链路

> 对应 Part 18 教程（[01 手写五件套](../../courses/Part18_rag/tutorial/01_naive_to_hybrid.md) /
> [02 高级 RAG](../../courses/Part18_rag/tutorial/02_advanced_rag.md)）。
> 四题必做 + 一题 🌟 Stretch，纯标准库可完成（不需要 torch/模型/GPU）。
> 实现本目录 `rag_exercises.py` 后运行 `python test_rag_exercises.py`（或 pytest）。
> 四个必做函数与课程脚本**同名同签名**——写完可直接回
> `courses/Part18_rag/scripts/01_minimal_rag.py`、`03_rag_eval.py` 对照。

## 📊 分值表

| 题号 | 主题 | 分值 | 对应测试 |
|------|------|------|----------|
| 1 | 递归分块（三条不变量） | 25 | `test_ex1_recursive_chunk` |
| 2 | 手写 BM25（IDF 判别力 + TF 单调） | 25 | `test_ex2_bm25` |
| 3 | RRF 融合（含 k→∞ 退化性质） | 25 | `test_ex3_rrf` |
| 4 | 手写 faithfulness（judge 可注入） | 25 | `test_ex4_faithfulness` |
| 🌟 5 | hybrid 权重网格搜索（Stretch） | 附加 10（总分封顶 100） | `test_ex5_stretch`（未实现自动 SKIP ⏭️） |

## 题目（实现 `rag_exercises.py`）

### 题 1：递归分块（25 分）

按 `['\n\n', '\n', '。', ' ']` 递归切原子段，贪心装进 ≤ `size` 字符的 chunk，
相邻 chunk 共享前一块尾部 `overlap` 个字符。

**验收标准：**
- [ ] 空文本 / 纯空白文本返回 `[]`；短文本返回单块且内容完整
- [ ] 不变量①：所有 chunk 长度 ≤ `size`（含无分隔符长文本的硬切路径）
- [ ] 不变量②：除首块外，每块以上一块尾部 `overlap` 个字符开头
- [ ] 不变量③：每块去掉开头 overlap 段后按序拼接，**去空白后**与原文去空白完全一致（不丢字符）
- [ ] `size=512/overlap=64` 与 `size=128/overlap=16` 两组参数都成立（参数要真的被用上）

> ⚠️ 最经典的坑：`s.split('。')` 会**丢掉句号**——不变量③在长中文段落上直接爆。
> 用捕获组 `re.split('(。)', s)` 把句号保留成独立原子段。

### 题 2：手写 BM25（25 分）

Okapi BM25，`k1=1.2, b=0.75`，查询词按出现次数累加，IDF 用恒正平滑式
（公式与计分约定见 `rag_exercises.py` docstring；分词器 `_tokens` 已提供）。

**验收标准：**
- [ ] 返回与 `chunks` 等长、位置对齐的 `list[float]`，全部 ≥ 0
- [ ] 查询词不在任何 chunk → 全 0 分
- [ ] IDF 判别力：只出现在 1 个 chunk 的稀有词，其最高分远超出现在所有 chunk 的常见词
- [ ] TF 单调性（长度归一没写反）：同一查询词出现 3 次的 chunk 得分高于出现 1 次、
  长度相近的 chunk

### 题 3：RRF 融合（25 分）

`score(item) = Σ_{两榜} 1/(k + rank)`，rank 从 1 起；只融合名次不融合分值；
并列按首次出现序稳定输出。

**验收标准：**
- [ ] 两榜皆空返回 `[]`；单榜非空时保留该榜全部元素
- [ ] 双榜都出现的元素排在所有单榜元素之前
- [ ] **k→∞ 退化**（测试用 `k=10**9`）：退化为"入选榜单数优先、名次和次之"的
  计数排序——`rrf_fuse([3,1,2],[2,4], k=10**9)` 首位是 2（双榜），
  次位是 3（单榜 rank1），单榜 rank2 的 4 排在 1 和 3 之后
- [ ] 完全对称输入（`['x','y']` vs `['y','x']`）返回原元素的某个排列（允许任一稳定顺序）

### 题 4：手写 faithfulness（25 分）

RAGAS faithfulness 的手写版：`split_claims`（已提供）拆句 → 逐条构造 prompt
（**含上下文与当前 claim 文本**）调 `judge` → 解析 yes/no/unsure → 返回
支持数 / 总 claims。**judge 是可调用对象**（输入 prompt 字符串返回文本）——
这正是为了让你能用 mock 离线测试（生产里它包着一个 LLM，课程脚本 03 里它包着
Qwen2.5-0.5B）。

**计分口径（文档化，测试按此校验）：** yes 计 1；no 与 unsure 计 0
（unsure 一律按"不支持"——保守口径，宁可低估不虚高）；解析不出三选一 → unsure；
无有效 claims（空答案）返回 `None`。

**验收标准：**
- [ ] 全 yes 的 mock judge → 返回 `1.0`
- [ ] 全 no / 全 unsure 的 mock judge → 返回 `0.0`
- [ ] 只对含标记词的 claim 答 yes 的 mock judge（3 句答案）→ 返回 `1/3`
      （证明你**逐条**调用了 judge，且 claim 文本出现在 prompt 里）
- [ ] 空答案 → `None`

### 🌟 题 5（Stretch，附加 10 分）：hybrid 权重网格搜索

01 章实测过"融合不是免费的"（Q2：BM25 单路 1.00 → hybrid 0.80）——RRF 是免调参
的稳健默认，但不是最优。实现 `hybrid_weight_sweep(dense_scores, sparse_scores,
relevant_sets, weights=None, k=5)`：对每个 `w` 计算加权混合
`w·dense + (1-w)·sparse`（**两路分数先各自 min-max 归一化到 [0,1]**，否则 BM25
的无界正分会统治 cosine）后的平均 recall@k，返回 `(best_w, curve)`；
`curve = [(w, mean_recall), ...]`，`best_w` 取网格中**最先**达到最大 recall 的 w。

**验收标准：**
- [ ] 曲线含 ≥5 个不同 w 的点；`best_w` 处 recall 为曲线最大值
- [ ] dense 完美 / sparse 专捧干扰项的构造下：`w=1.0` 处 recall=1.0、`w=0.0`
      处 recall < 0.5，且 `best_w ≥ 0.5`（落在 dense 侧）
- [ ] 角色互换（sparse 完美 / dense 捣乱）后 `best_w ≤ 0.5`（证明不是只会偏向 dense）
- [ ] 未实现（`return None`）时测试显示 `⏭️ SKIP` 而非 ❌ FAIL
- [ ] （不参与测试）用 matplotlib（`Agg` 后端）把 recall-w 曲线画出来，
      看看最优权重是不是总在端点——想想什么时候它会在中间

**步骤提示：**
```python
# 1. 逐 query 对两路分数 min-max 归一化（注意除零：max==min 时返回全 0）
# 2. 对每个 w：fused = [w*a + (1-w)*b for a, b in zip(nd, ns)]
# 3. 排名取 top-k，recall@k = |topk ∩ rel| / min(|rel|, k)，对全部 query 取平均
# 4. 扫完网格返回 (best_w, curve)；并列取先出现者（用严格大于更新最优）
```

## 实验题（观测型）

- 跑 `RAG18_FORCE_FALLBACK=1 python courses/Part18_rag/scripts/01_minimal_rag.py`，
  对比主模式：dense 列 recall 从多少掉到多少？用一句话解释 hashing trick 与真嵌入
  的差距来源（提示：字面碰撞 vs 语义泛化）
- 把课程脚本 01 的 `RRF_K` 从 60 改成 6000 再改成 6，观察 Q1-Q3 的 hybrid 列怎么动；
  对照题 3 的 k→∞ 退化性质解释你看到的方向
- （选做）装 ragas：`uv pip install --python .venv ragas`，跑通脚本 03 的对照段，
  与手写 faithfulness 比"幻觉句"被压低的幅度

## 🤔 思考题

**Q1：** 题 3 里 k→∞ 时 RRF 退化成"计数排序"。这个极限性质在工程上什么时候有用、
什么时候有害？

<details>
<summary>💡 答案</summary>

有用面：当两个榜单的分数尺度/分布完全不可信（比如一路是冷启动模型、分数未校准），
只信"是否入选"这一比特信息时，大 k 让 RRF 接近投票制——极端稳健。
有害面：名次信息被压平后，"两榜都认可的第 1 名"和"两榜都勉强入选的第 50 名"
得分几乎一样（都接近 2/k），区分度坍塌；k=60 的默认值就是把"名次和"与
"入选数"的权重调在一个务实折中上。本质：k 控制的是名次信息的带宽。
</details>

**Q2：** 题 4 的 faithfulness 里，unsure 计 0（保守）还是从分母里剔除（宽松），
哪种口径更好？两种口径分别会把什么真实问题藏起来？

<details>
<summary>💡 答案</summary>

计 0（保守）：分数被裁判的"不确定"压低——低分可能来自答案真烂，也可能来自
上下文没覆盖（检索的锅被记到生成的头上）；好处是永不虚高，灰度发布时宁可错杀。
剔除（宽松）：分数只反映"有明确证据"的部分，裁判不敢判的陈述成了法外之地——
模型含糊其辞（既不支持也不矛盾的话术）反而拿到高分，会被 prompt 注入式攻击
利用。RAGAS 的实现接近前者（不支持即不计入 faithful claims）。工程折中：
把 unsure 单独报一列（三明治口径：supported / refuted / unknown 三数齐报），
别让一个标量把三种状态压扁——课程脚本 03 的讨论块就是这个意思。
</details>

**Q3：** 题 5 的网格搜索里，为什么必须先 min-max 归一化？如果不归一化，
最优 w 会怎么漂？

<details>
<summary>💡 答案</summary>

cosine ∈ [-1,1]（方差通常 ~0.1），BM25 无界正分（一个稀有词命中就能到 5+）。
不归一化时 `w·dense + (1-w)·sparse` 的排序几乎完全由 sparse 决定——除非 w 逼近 1，
dense 的贡献都被 sparse 的量级淹没；于是 recall-w 曲线会在 w 接近 1 的窄区间里
跳变，"最优 w"看起来像 0.9+，给你"dense 占九成"的错觉，实际上那只是
**量纲补偿**而非信息权重。归一化后 w 才是真正的"信任分配"。这也是为什么
RRF（只看名次）在工业界更受欢迎：它对尺度天然免疫。
</details>

**Q4：** 老板给你一批内部文档（约 50k token）和一个 128k 上下文的模型，
要做问答机器人："RAG、微调、长上下文直塞，怎么选？"

<details>
<summary>💡 答案</summary>

先看量级：50k token 远小于上下文预算，**首推直接塞 prompt**（context engineering
共识：装得下就别绕路）——零检索误差、零额外组件、引用直接给原文段落。
但要过三关再拍板：① 更新频率——文档每天变？重发全量 prompt 的成本 vs 重嵌入
增量 chunk（此时 RAG 赢）；② 多租户/权限——每个用户只能看自己的文档时，
按查询检索（RAG）比把所有文档塞进共享上下文安全得多；③ 成本规模——
Self-Route（arXiv 2407.16833）实测 RAG 的 token 消耗仅约为长上下文直塞的 1/6，
QPS 上来后这个差距会翻成真金白银。微调在这里**根本不对题**：它改变的是模型的
风格/能力，不是知识的新鲜度（LaRA 2502.09977 的结论也是两者各有所长、无银弹）。
一句话回答面试官：知识问题用检索或长上下文（按预算与更新频率二选一），
行为问题才用微调，混合需求就叠着用。
</details>

## 🎯 面试直通车

- "手写一个 BM25"——IDF 平滑式 + TF 饱和 + 长度归一，三个公式白板可写（题 2）
- "hybrid 搜索两路分数怎么融合？"——RRF 免标定是默认答案；追问就讲加权 + 归一化 +
  网格搜（题 5 的完整故事），并给出"融合不是免费"的实测（01 章 Q2）
- "RAG 怎么评测？"——检索侧 recall@k（要 ground truth），生成侧 RAGAS 四指标；
  重点讲 faithfulness 的 claims 分解与**评测器噪声**（等价问法即可让小裁判翻转）
- "chunk 大小怎么定？"——没有银弹，256-1024 起步 + 按下游指标调；碎块的解法是
  overlap、contextual retrieval（先切后补）或 late chunking（先嵌后切）
- "什么时候不用 RAG？"——装得下就直塞；知识要引用/常更新才上 RAG；
  要改行为去微调（Q4 的三句话版本）
