# S2 学习报告·P18（02 章）—— 实操审计（学生视角 S2：hands-on 型）

- 审计对象：`courses/Part18_rag/tutorial/README.md`、`02_advanced_rag.md`（01 章跳过）+ `scripts/02_contextual_retrieval.py`、`scripts/03_rag_eval.py`
- 实测环境：RTX 4090 / torch 2.x / 模型缓存齐全（Qwen3-Embedding-0.6B、Qwen2.5-0.5B-Instruct、bge-reranker-v2-m3 均在本地 HF cache）；**但本机 huggingface.co 走的代理 192.168.0.105:7890 不可达**（意外构成一档真实弱网环境）
- scratch 留档：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P18R/`（run02.log / run02_offline.log / run03_offline.log / run03_default.log）

## 总分：8.3 / 10

一句话：**教程展示的每一个数字我都能在 `HF_HUB_OFFLINE=1` 下逐位复现（含 238 chunk、四阶梯 0.67/0.61/0.61/0.61、faithfulness 0.56→0.40、判决翻转 A=no/B=yes/C=yes），复现精度是同规模教程里罕见的；扣分全部集中在"网络前提未写明"——模型已在本地缓存、只是代理不通时，脚本自动降级，耗时翻 2.7~19 倍、数字与教程完全对不上，而教程对此最常见场景只字未提。**

无阻断性 bug：4 次运行全部 rc=0，"脚本永不崩"的承诺兑现。

## 卡点清单（6 个：高 2 / 中 2 / 低 2）

### 🔴 高-1：隐含网络前提未文档化，弱网环境下教程数字整体复现不出
模型明明已在本地缓存，但 transformers 默认仍向 huggingface.co 发 HEAD 校验；代理不可达时每模型重试 5 次（1+2+4+8+8s）后抛 ProxyError → 三个模型全部自动降级。后果（默认环境实测）：
- 脚本 02：143s（教程声明 50-55s），mean 从 0.67/0.61/0.61/0.61 变为 0.11/0.11/0.17/0.17
- 脚本 03：188s（教程声明 17s），faithfulness 全 1.00、context_precision 全 0.00（关键词裁判零判别力）

学生看到的输出与教程表格天差地别，却没有任何一处告诉他"设 `HF_HUB_OFFLINE=1` 即可复现教程"。README 只写了"模型缺失时自动降级"，没写"能联网校验但下载不了/代理坏"这个中间态。

### 🔴 高-2：降级警告的指引是死胡同，异常文本截断到失去关键信息
降级时打印的指引是 `huggingface-cli download Qwen/...`——但在"代理坏/无网"场景照做只会再撞一次 ProxyError（模型本就在缓存里）。三行警告的异常摘要是 `ProxyError: (MaxRetryError('HTTPSConnectionPool...Max retr`（`str(e)[:80]` 截断），学生看不出"问题在代理/网络"还是"模型缺失"。建议：检测到本地缓存存在时，警告追加一句"已检测到本地缓存，可尝试 `HF_HUB_OFFLINE=1 python ...`"。

### 🟡 中-1：耗时声明三处口径互相矛盾
- 脚本 02 docstring："GPU 约 2 分钟"；README："~50-55s"；教程 02 章："总耗时 50-55s（实测 52-54s，其中 LLM 定位句 ~28s）"。offline 实测 53s（wall 54.2s，LLM 句 32s）支持教程，docstring 偏保守 2 倍以上，又恰好与坏网络实测（143s）巧合——同一行字在三种环境里对三个不同的数。
- 脚本 03 docstring："GPU 约半分钟"；README/教程："~17s"；offline 实测 10-11s。三个数互不相同。

### 🟡 中-2：02 章正文没有可复制的运行命令块
全章只以链接形式引用脚本（[脚本 02](../scripts/02_contextual_retrieval.py)），运行命令（`cd courses/Part18_rag/scripts && CUDA_VISIBLE_DEVICES=0 python 02_contextual_retrieval.py`）只存在于 README 环境块；连教程 ⚠️ 块里演示的 `RAG18_FORCE_FALLBACK=1` 降级，README 也只给了配脚本 01 的命令。学生读到"跑脚本看 Step 1 打印的示例即可自查"时，得自己回 README 拼命令。

### 🟡 低-1：脚本 03 降级下 context_precision 全 0.00 教程未提
教程练习 Q2 答案精确预告了降级裁判 faithfulness 全 1.00（实测吻合，好评），但同一次降级运行里 context_precision 三问全 0.00（关键词裁判解析不了相关性 prompt）教程没写——虽然是同一结论（降级裁判不可信）的更极端证据，但学生对照教程时会多一处"教程没说的输出"。

### 🟡 低-2：定位句 token 口径微漂
教程写"约 64 token"，脚本常量 `CTX_MAX_TOKENS = 64`（注释"~50 token 的定位句"），docstring 写"~50 token"。上限 64 与典型输出 ~50 是两回事，三处表述建议统一为"上限 64 token（典型 ~50）"。

**YAML/配置对照结论**：Part18_rag 全目录（含 scripts/tutorial）无任何 YAML/JSON/TOML 配置文件——配置全靠脚本内常量（`CHUNK_SIZE=512/CHUNK_OVERLAP=64/RRF_K=60/EVAL_K=20/FUSE_POOL=100` 等，注释齐全）+ 环境变量（`CUDA_VISIBLE_DEVICES`、`RAG18_FORCE_FALLBACK`，以及未被文档化的 `HF_HUB_OFFLINE`）。README 环境块与脚本入口逐项核对一致（cd 路径、脚本名、注释耗时）。未发现配置漂移，唯一的"配置缺失"就是高-1 的网络开关。

### 可拼凑性专项（正面结论）
- 02/03 通过 importlib 按 spec 加载 01 模块，引用的全部共享符号（`CORPUS_FILES/GEN_MODEL/CHUNK_SIZE/CHUNK_OVERLAP/RRF_K/CAND_K/FORCE_FALLBACK/recursive_chunk/embed_texts/dense_search/bm25_scores/rrf_fuse/rerank/relevant_chunks/load_embedder/load_reranker/load_generator/generate_answer/_tokens/QUERIES`）在 01_minimal_rag.py 中逐一 grep 验证存在，无未定义引用。
- 教程输出块与实际 stdout 逐行一致：实验一 Q1 0.75×4 / Q2 0.58 0.42 0.50 0.50 / Q3 0.67 0.67 0.58 0.58，实验二 Q1 0.75 1.00 0.75 1.00 / Q2 0.58 0.50 0.50 0.33，两处 mean 行、失败率行、Step 4 翻转与 W1/W2/W3 判定——**无一字偏差**。
- 练习 1 的 `lambda p: 'yes'` 可直接作为 `faithfulness(answer, contexts, judge)` 的 judge 传入（`_parse_verdict('yes')='yes'`），接口设计与练习承诺一致。

## 分章评分（满分 10）

| 维度 | 得分 | 依据 |
|---|---|---|
| 02 章命令/代码可拼凑性 | 8.5 | 符号与数字全部可复现；扣：正文无命令块（中-2）、网络前提缺失（高-1） |
| 脚本 02 质量 | 9.0 | 累积式复用 01 干净、EVAL_K/FUSE_POOL 与官方口径对齐有注释、降级路径完整、结论打印自带归因 |
| 脚本 03 质量 | 9.0 | judge 可注入接口、unsure 保守口径文档化、噪声实验设计精巧、ragas 可选依赖处理规范 |
| 耗时/环境声明诚实度 | 6.5 | offline 下教程数字精确；扣：docstring 与 README/教程三口径矛盾（中-1）、未写网络前提（高-1） |
| 降级路径设计 | 7.5 | rc=0 兑现、降级数字被教程精确预告；扣：警告指引死胡同+截断（高-2）、context_precision 全 0 未提（低-1） |

## 一致性核对表（docstring vs README vs 教程 vs 实测）

| # | 声明项 | 脚本 docstring | README | 教程 02 章 | 实测·offline 主模式 | 实测·默认（代理坏） | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | 脚本 02 耗时 | 约 2 分钟 | ~50-55s | 50-55s（LLM 句 ~28s） | **53s**（wall 54.2s，LLM 句 32s） | **143s**（全降级） | 教程✓ docstring✗；README 缺前提 |
| 2 | 脚本 03 耗时 | GPU 约半分钟 | ~17s | 总耗时实测 17s | **10s**（wall 11.3s） | **188s**（关键词裁判） | 三口径互异；offline 实测比声明更快 |
| 3 | chunk 数 | — | — | 238 | 238 ✓ | 238 ✓ | ✓ |
| 4 | 实验一 mean | — | — | 0.67/0.61/0.61/0.61 | 0.67/0.61/0.61/0.61 ✓ | 0.11/0.11/0.17/0.17 | offline 逐位✓ |
| 5 | 实验一失败率 | — | — | 全 0.00 | 0.00×4 ✓ | 0.33×4 | offline✓；0.33 恰为教程降级块预告值✓ |
| 6 | 实验二 mean | — | — | 0.67/0.72/0.64/0.67 | 同 ✓（Q1-Q3 逐格一致） | 0.11/0.17/0.17/0.17 | ✓ |
| 7 | faithfulness | — | — | 0.67/0.00/1.00，mean 0.56→0.40 | 完全一致 ✓ | 1.00→1.00 | offline✓；降级值恰为教程练习答案预告✓ |
| 8 | context precision | — | — | [1,1,1,1,0]/全 1，τ=+1.00，其余 n/a | 一致 ✓ | 全 0（教程未提，低-1） | offline✓ |
| 9 | 评测器噪声 | — | — | A=no/B=yes/C=yes；W1=yes/W2=no/W3=no | 完全一致 ✓（含"⚠️ 翻转"字样） | — | ✓ |
| 10 | ragas 缺失行为 | 打印指引跳过 rc=0 | 未装时打印指引跳过 rc=0 | 同左 | 打印 `uv pip install --python .venv ragas` → 跳过，rc=0 ✓ | 同 ✓ | ✓ |
| 11 | YAML/配置 | 常量内嵌 | 环境块仅 env var | 未提配置文件 | 无 YAML；常量+注释自洽 | — | 无漂移；缺 HF_HUB_OFFLINE 文档 |
| 12 | 练习 1 mock 裁判 | judge 形参可注入 | — | `lambda p: 'yes'` | 接口吻合（未单独跑，静态验证） | — | ✓ |

## 只改 3 件事

1. **README 环境块与 02 章开头加"网络开关"一行**：`模型已在本地缓存时建议 export HF_HUB_OFFLINE=1`，并注明"否则代理/弱网下会反复重试（脚本 02 实测 143s、脚本 03 实测 188s）且结果为降级数字"。这是学生最易踩、教程唯一没覆盖的真实环境，一行字消掉全部高-1 困惑。
2. **统一耗时口径**：脚本 02 docstring "约 2 分钟" → "约 1 分钟（offline）；未断网时含重试 2-2.5 分钟"；脚本 03 docstring "约半分钟" → "10-17s"；README/教程标注耗时前提。三处对齐后学生可据此判断自己是不是跑成了降级。
3. **02 章"实测"小节前加可复制命令块**：主模式两条 + `RAG18_FORCE_FALLBACK=1` 降级两条（配 02/03），并让降级警告在检测到本地缓存时提示 `HF_HUB_OFFLINE=1` 而非只给 `huggingface-cli download`（高-2 一并解决）。

## 最喜欢 3 处

1. **教程把降级数字原文写进正文并留档**：⚠️ 块的 0.11→0.11→0.17→0.17、失败率 33%，练习答案里的 grounded=1.00/+幻觉=1.00——我在坏代理环境跑出的降级结果与教程预告**逐位一致**。教程提前替学生跑过最坏的路径并诚实公布，这种"自曝家底"是可信度天花板。
2. **实验二"格式噪声与增益同量级"**：信息一字不差、只换排版（`文档名 · 章节 原文` vs `《文档名》章节：原文`），mean 0.72 vs 0.64——这是全章最有洞察的实测设计，且结论打印与教程解读完全对齐；它教的不是技术，是"单点数字不可信"的评测观。
3. **脚本 03 的 judge 可调用注入**：`faithfulness(answer, contexts, judge)` 把裁判抽象成 `Callable[[str], str]`，练习 1 的 mock 裁判一行 lambda 即可替换、可离线测试——把"评测器本身是被测对象"这个教学点直接焊进了函数签名。
