# S2 学习报告·P01（Bigrams）

> 审计人：S2（实操薄弱型：理论尚可，几乎没从零写过代码；会调 API，不熟 shape/broadcasting）
> 环境：Python 3.12.12（项目 .venv），MPLBACKEND=Agg，全部在 scratch/scratch 工作目录真实运行
> 日期：2026-09-04

---

## 总分：8.5 / 10

| 维度 | 得分 | 说明 |
|------|:---:|------|
| 教程内容质量 | 8.5/10 | 概念讲解到位，broadcasting/等价性两处是亮点；个别代码口径与脚本脱节 |
| 教程↔脚本可复现性 | 8/10 | 7 个脚本全部一次跑通（EXIT=0，无超时），但存在 3 处不一致（见核对表 ❌/⚠️） |
| 作业质量 | 9/10 | 提示充分、测试自动判分、拓展题分层合理；小缺陷：未实现时测试仍显示 PASSED |

---

## 卡点清单

格式：`[严重度]|阶段|现象|解决`

1. **🔴|作业题 1|照教程脚本 02 的"动态推字符集"写法（`chars = sorted(set(''.join(words)))`）实现 `build_bigram_matrix`，首跑测试 FAIL**：测试喂的是 `['emma','olivia','ava']`，动态集合只有 8 个字母，'e' 被映射到索引 2，而测试按固定字母表查 `N[0, 5]`（得 0）。**这是教程脚本从未提示的隐患**——脚本处理全量数据恰好含 26 个字母所以从未暴露。|读测试源码自诊定位，改为硬编码 `enumerate('abcdefghijklmnopqrstuvwxyz')` 后 PASS。建议教程/作业显式警告"勿从输入动态推 stoi"。
2. **🟡|脚本 07|教程 03 说"训练后 loss 应该收敛到约 2.47"，实际跑 07 输出 2.4901，脚本结尾还硬编码打印"最终 loss ≈ 2.47"**，口径自相矛盾。|自己写作业题 5（无正则、同种子）100ep 得 2.4729，定位到 07 在 loss 里加了 `0.01 * (W**2).mean()`，打印的是"含正则 loss"。
3. **🟡|教程 03 §1|教程贴的 one-hot 可视化代码声称"生成 ../images/cell032_output01.png"并注明"完整脚本见 06_neural_network.py"，但脚本 06 里没有任何 matplotlib 代码**。图本身存在，但学生照脚本找不到生成来源。|无（记录）。
4. **🟡|教程 02 §2 vs 脚本 03|教程可视化代码 `savefig('../images/cell011_output00.png')` + `plt.show()`，脚本 03 实际存到 `scripts/bigram_matrix.png`**（且每跑一次就往仓库 scripts/ 目录写 635KB 文件，污染工作区）。|无（记录）。
5. **🟡|教程 02 §5|NLL 核心代码先于"模型平滑"小节出现**：若按 3️⃣ 的未平滑 `P` 直接跑 5️⃣ 的代码，遇到训练集未出现的 bigram 会 `log(0) = -inf`。实操型读者会先跑后读、直接踩坑。|往下读一节才见平滑说明；建议代码自带 `(N+1)` 或加一行警告。
6. **🟢|教程 02 §4|教程说采样输出"可能因 PyTorch 版本略有不同"，实际当前环境输出与教程示例逐字一致**（junide/janasah/p/cony/a），免责声明反而让我怀疑自己跑错了。|对照确认一致。
7. **🟢|作业测试|`test_train_bigram_nn` 在未实现（返回 None）时也显示 PASSED 而非 SKIPPED**，粗看会误判"5 题全过"。|阅读测试源码才知有 `if result is None: return` 跳过逻辑。
8. **🟢|教程 01|"约 228,000 个字符"实为 bigram 总数（实测 228146）**；纯字符总数约 196k（228146−32033 个名字），说法不严谨但量级无害。

---

## 逐章评分

| 章节 | 评分 | 一句话点评 |
|------|:---:|------|
| 01_introduction.md | 9/10 | 门槛低、预期管理好；"概率→log→负→平均"推导链条提前给出，对新手友好 |
| 02_bigram_model.md | 8/10 | Broadcasting 速成是全场最佳；扣分：NLL 代码先于平滑、可视化代码与脚本脱节 |
| 03_neural_network.md | 8.5/10 | 等价性（MLE 视角）讲得透，lr=50 的"特例非通用"解释很负责；扣分：训练循环代码缺正则项、W.exp() 对比承诺未兑现 |
| tutorial/README.md | 9/10 | 导航与学习路线图清楚 |

---

## 教程↔脚本一致性核对表

| 教程位置 | 教程代码 | 对应脚本实测 | 判定 |
|------|------|------|:---:|
| 02 §2 统计 | `N=zeros(27,27,int32)`；`['.']+list(word)+['.']` | 02/03/04/05 同逻辑（仅变量名 word/w、chars/chs 之差） | ✅ |
| 02 §2 可视化 | `savefig('../images/cell011_output00.png')`+`show()` | 03 存 `scripts/bigram_matrix.png`，无 show，多 title/fontsize=8 | ⚠️ 输出路径/形式不一致 |
| 02 §3 概率 | `P=N.float(); P/=P.sum(1,keepdims=True)`（无平滑） | 04/05 实为 `(N+1).float()`（平滑在后文才讲） | ⚠️ 顺序坑 |
| 02 §4 采样 | seed 2147483647，multinomial 采样 | 04 相同；输出逐字复现 junide/janasah/p/cony/a | ✅ |
| 02 §5 NLL | 循环累加 log，"平均 NLL 约 2.45" | 05 实测 2.4544 | ✅ |
| 03 §1 one-hot | `F.one_hot(xs, 27).float()`，形状 (N,27) | 06 相同，实测 (228146, 27) | ✅ |
| 03 §1 可视化 | 声称脚本 06 生成 cell032_output01.png | 06 无任何可视化代码 | ❌ |
| 03 §2 前向 | `W=randn(27,27,requires_grad=True)`（无种子） | 06 用 `Generator().manual_seed(2147483647)` | ⚠️ 教程版不可复现 |
| 03 §3 训练循环 | `loss=-probs[arange,ys].log().mean()`（无正则） | 07 的 loss 含 `+0.01*(W**2).mean()` | ❌ loss 定义不一致 |
| 03 §3 结论 | "loss 收敛到约 2.47" | 07 实测 2.4901（含正则）；作业无正则版实测 2.4729 | ⚠️ 口径不一 |
| 03 §4 W.exp() 对比 | "训练后看 W.exp() 与 N 几乎一样"（附 `detach()` 代码） | 7 个脚本均未实现该对比 | ❌ 承诺未兑现 |
| 03 §4 采样 | 未给种子说明 | 07 用 `2147483647 + 10` | ⚠️ 教程未说明 |

脚本运行记录（scratch/S2_P1，MPLBACKEND=Agg，timeout 120s）：**01–07 全部 EXIT=0，零失败零超时**。关键实测：01 总数 32033；02 总 bigram 228146、Top1 'n.'=6763；05 平均 NLL 2.4544（andrejq 3.4834）；06 初始 loss 3.7590；07 最终 2.4901，采样质量明显差于计数版（mmyazzieelend 之类长怪名）。

---

## 作业元数据 + pytest 输出

工作目录：`students/S2_hands/work/assignment_1/`（data 软链接至仓库 data/，`__main__` 与 test 的 `../../data/names.txt` 相对路径均被正确接住）

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|----|:---:|:---:|:---:|:---:|
| 1 build_bigram_matrix | ❌ 首跑 FAIL → 修复 | 0（读测试源码自诊） | ~4min | FAIL → **PASS** |
| 2 compute_probabilities | ✅ | 0 | ~2min | **PASS** |
| 3 generate_names | ✅ | 0 | ~3min | **PASS** |
| 4 compute_nll_loss | ✅ | 0 | ~2min | **PASS** |
| 5 train_bigram_nn（拓展） | ✅ | 0（README 提示代码已足够） | ~5min | **PASS** |

**最终 pytest 输出（全量，`python -m pytest test_bigram_exercises.py -v`）：**

```
test_bigram_exercises.py::test_build_bigram_matrix PASSED                [ 20%]
test_bigram_exercises.py::test_compute_probabilities PASSED              [ 40%]
test_bigram_exercises.py::test_generate_names PASSED                     [ 60%]
test_bigram_exercises.py::test_compute_nll_loss PASSED                   [ 80%]
test_bigram_exercises.py::test_train_bigram_nn PASSED                    [100%]
============================== 5 passed in 81.53s ==============================
```

数值交叉验证：作业 `compute_nll_loss` = 2.4544（= 脚本 05 ✅）；默认 seed 生成 junide/janasah/p/cony/a（= 教程示例 ✅）；seed=42 生成 ya/syahavilin/dleekahmangonya/tryahe/chen；`train_bigram_nn`（无正则）100ep loss = 2.4729。

---

## 只改 3 件事

1. **统一 07/教程 03 的 loss 口径**：教程 03 §3 训练循环补上 `+ 0.01 * (W**2).mean()`（或明确注明 07 含正则），并把"收敛到约 2.47"改为与 07 实际输出一致（含正则 ≈2.49 / 无正则 ≈2.47），删掉脚本 07 里硬编码的"最终 loss ≈ 2.47"字符串。
2. **兑现/对齐可视化承诺**：脚本 06 补上可选的 one-hot 可视化段（或在教程注明"该图由 notebook 生成，脚本不含"）；统一教程注释与脚本的可视化输出路径，并让脚本把 PNG 写到 scratch/输出目录而不是写回仓库 `scripts/`。
3. **堵住题 1 的字符集坑**：在作业题 1 或教程 02 加一句显式警告"stoi 必须按固定 26 字母表构建，勿从输入动态推导"；同时把 `test_train_bigram_nn` 未实现分支改为 `pytest.skip`（或文档注明"PASSED 可能只是跳过"）。

## 最喜欢 3 处

1. **02 的 Broadcasting 速成**：从右向左对齐规则 + `keepdims` 有/无对照 + `[[1,2,3],[4,5,6]]` 手算示例——对我这种 shape/broadcasting 薄弱的人是全场最有价值的一段，作业题 2 一次过直接受益于此。
2. **03 的"计数版 vs 神经网络版"双栏对照表 + MLE 等价性解释**：把整个 Part 1 收拢成一张表，"神经网络优势不在 Bigram 而在可扩展性"这一定位诚实且有助于建立全局观。
3. **作业的脚手架设计**：函数签名 + docstring 步骤提示 + 自动判分测试 + 拓展题可选，README 的"思考"问题（log(27)≈3.3 的基准对照、NLL 与 bit 编码的关系、smoothing 0/1/10/100 实验）引而不发，适合自测理解深度。
