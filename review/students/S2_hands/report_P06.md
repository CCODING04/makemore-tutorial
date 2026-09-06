# S2 学习报告 · P06（Transformer / GPT）

> 学生画像：S2（实操薄弱型）——按教程边读边抄代码、逐段跑脚本、做作业时只看骨架 docstring 提示，不偷看参考答案。
> 审计日期：2026-09-04。环境：Linux，RTX GPU（`torch.cuda.is_available()=True`），PyTorch 2.6.0+cu124，Python 3.12.12。

---

## 一、总分：9 / 10

教程代码块与 7 个脚本逐段对照**零实质性偏差**；教程声称的"实跑真实日志"本次复跑全部对上（多数完全一致，个别差 ±0.03 以内且教程自带 ≈ 号）；作业 5 题（7 个测试）一次全过。扣 1 分主要给脚本 07 在 GPU 机器上无法快速跑完（time-blocked）和代码块节选的孤立可运行性问题。

---

## 二、卡点清单（按严重度排序）

1. **【卡 · time-blocked】脚本 07 在 GPU 机器上 3 分钟跑不完。**
   `07_scaleup_generate.py` 用 `CPU_MODE = not torch.cuda.is_available()` 切换模式。我的机器有 GPU → 自动走完整版（batch=64, block=256, n_embd=384, 6 层, 5000 步, ~10.58M 参数），`timeout 180` 被杀（rc=124, secs=180，stdout 因缓冲丢失为空）。教程 03 章说"缩小型 <30s 可跑"，但这个承诺**只在无 GPU 的机器上成立**——有 GPU 但只想快速验证的读者反而没得选。建议加环境变量开关（如 `GPT_QUICK=1`）。
2. **【绊】教程代码块是"节选"，孤立复制会 NameError。**
   如 01 章 `get_batch` 代码块里的 `block_size / batch_size / device / train_data / val_data`，02 章 `Head` 里的 `n_embd / block_size`——脚本中都是（闭包外）全局变量，教程代码块内未定义也未提示。笔记惯例可接受，但对"抄了就跑"的 S2 型学生是必踩的坑。
3. **【微】"实跑日志"未标注运行设备。**
   教程 02/03 章的 step0 数字（如 04 脚本 4.2351/4.2337）与我本次 CUDA 复跑（4.2328/4.2301）有微小漂移——教程显然是 CPU 跑的，但正文没说。README 的"看趋势别背数字"其实已覆盖，若日志旁标注设备/版本更严谨。
4. **【微】脚本 03 的 `torch.allclose` 实际带了 `atol=1e-5, rtol=1e-5` 放宽容差**（教程正文只写"用 `torch.allclose` 验证：True"）。合理（for 循环 mean 与 matmul 有 ~1e-8 舍入差，脚本注释也解释了），但教程若提一句更完整。
5. **【风格】所有脚本训练循环用 `for iter in range(...)` 遮蔽内置 `iter`**，教程代码块同款。不影响运行。

无阻断性错误：**教程代码块 ↔ 脚本没有发现任何变量名不一致、逻辑不一致或脚本本身跑不通的问题**。

---

## 三、逐章评分

| 章节 | 评分 | 实操体验（S2 视角） |
|------|:---:|------|
| 01 数据与 Tokenizer | 9.0 | 逐段抄写可跑通；chunk 多样本演示 + get_batch + Bigram + AdamW 训练循环与脚本 02 逐行一致；复跑初始 loss 4.7417 与教程一字不差，val ≈2.50 ✓。扣：节选变量未定义（卡点 2）、日志未标设备。 |
| 02 Attention 从零开始 | 9.5 | v1/v2/v3 三版本 allclose 全 True（复跑 ✓）；Head 代码逐行对照脚本 04 零偏差（scaled、masked_fill、softmax、register_buffer、idx_cond 裁剪全在）；亲和力演示、"放大 8 倍 softmax"演示都可复现；6 条笔记结构清晰。 |
| 03 Transformer Block | 9.0 | LayerNorm 演示（std=1.1180 的 Bessel 校正解释）复跑数字与教程完全一致；Block(pre-norm)/ln_f 与脚本 06 一致；CPU 缩小型参数量 112,193 我用独立小脚本复算**精确匹配**；Dropout 三处位置与脚本 07 一致。扣：07 的 GPU/CPU 自动切换（卡点 1）。 |
| 04 超越 Transformer | 8.5 | 无代码章，encoder/decoder/cross-attention/nanoGPT 三细节/SFT→RM→RLHF 讲得清楚；nanoGPT 片段（4D attention、GeLU、参数分组）本地无法验证，属走读性质。 |
| README | 9.0 | 双列 loss 表（原视频 vs 本仓库）+ † 脚注诚实解释 Phase2 反常，导航/前置知识齐全。 |

---

## 四、一致性核对表（教程代码块 ↔ 脚本逐段对照）

| # | 教程位置 | 代码块内容 | 对应脚本 | 结论 |
|---|---------|-----------|---------|:---:|
| 1 | 01 章「一个 chunk 内含多个样本」演示 | `train_data[:block_size+1]` 偏移打印 | 02 脚本 L66-73 | 一致 |
| 2 | 01 章 get_batch | `data_local/ix/x.stack/y.stack/to(device)` | 02 脚本 L76-83 | 逐行一致（节选依赖全局变量） |
| 3 | 01 章 BigramLanguageModel | Embedding 表当 logits + view(B*T,C) | 02 脚本 L106-125 | 逐行一致 |
| 4 | 01 章 generate | softmax + multinomial + cat | 02 脚本 L127-135 | 逐行一致 |
| 5 | 01 章 训练循环 + AdamW | zero_grad(set_to_none=True)→backward→step | 02 脚本 L152-163 | 一致 |
| 6 | 02 章 v1 for 循环平均 | 双层循环 + mean(0) | 03 脚本 L42-46 | 一致 |
| 7 | 02 章 v2 tril 归一化 @ x | `wei/wei.sum(1,keepdim=True)` | 03 脚本 L53-62 | 一致（脚本 allclose 带容差，见卡点 4） |
| 8 | 02 章 v3 masked_fill+softmax | 全 0 → -inf → softmax → @x | 03 脚本 L70-73 | 一致 |
| 9 | 02 章 代码清理（n_embd+lm_head） | 去掉 vocab_size 参数 | 04 脚本 `__init__(self)` | 一致 |
| 10 | 02 章 位置编码 | pos_emb + 广播相加 | 04 脚本 L143-152 | 一致 |
| 11 | 02 章 Head（核心） | K/Q/V、scaled、tril buffer、遮罩、softmax、wei@v | 04 脚本 L88-111 | 逐行一致 |
| 12 | 02 章 模型插入 sa_head | tok+pos → sa → lm_head | 04 脚本 L136-162 | 一致 |
| 13 | 02 章 generate 裁剪 | `idx_cond = idx[:, -block_size:]` | 04 脚本 L164-173 | 一致 |
| 14 | 02 章 MultiHeadAttention | ModuleList + cat + proj | 05 脚本 L109-121 | 逐行一致 |
| 15 | 03 章 FeedForward | Linear→ReLU→Linear(4×) | 05 脚本 L123-136 | 一致 |
| 16 | 03 章 残差两行 | `x = x + sa(x); x = x + ffwd(x)` | 05 脚本 BlockNoLN L146-149 | 一致 |
| 17 | 03 章 LayerNorm 演示 | `torch.randn(4,5)` 逐行 mean/std | 06 脚本 L91-100 | 一致（复跑 std=1.1180 与教程相同） |
| 18 | 03 章 Block（pre-norm） | `x + sa(ln1(x))`, `x + ffwd(ln2(x))` | 06 脚本 L145-160 | 逐行一致 |
| 19 | 03 章 完整模型含 ln_f | blocks → ln_f → lm_head | 06 脚本 L162-199 | 一致 |
| 20 | 03 章 Dropout 三处位置 | softmax 后 / proj 后 / ffwd 末尾 | 07 脚本 L130, L144, L154 | 一致 |
| 21 | 03 章 参数统计 + 超参表 | sum(p.numel())；CPU 缩小型超参 | 07 脚本 L36-48, L226-232 | 一致；参数量 112,193 独立复算精确匹配 |
| 22 | 03 章 生成起始 | `torch.zeros((1,1), dtype=long, device)` | 07 脚本 L257 | 一致 |

**数字复跑核对**（教程"实跑日志" vs 本次 CUDA 环境）：

| 项 | 教程 | 本次复跑 | 差异 |
|----|------|---------|------|
| 01 总字符/vocab/train | 1,115,394 / 65 / 1,003,854 | 同 | 0 |
| 02 初始 loss | 4.7417 | 4.7417 | 0 |
| 02 最终 val | 2.5017 | 2.4988 | -0.003 |
| 03 三版本 allclose | True×3 | True×3 | 0 |
| 04 最终 val | 2.3871 | 2.3860 | -0.001 |
| 05 Phase1 val | 2.4545 | 2.4545 | 0 |
| 05 Phase2 val | 2.5006 | 2.5010 | +0.0004 |
| 05 Phase3 val | 2.2324 | 2.2290 | -0.003 |
| 06 step0 / 最终 val | 4.2889 / 2.2340 | 4.2889 / 2.2382 | 0 / +0.004 |
| 07 CPU 参数量 | 112,193 | 112,193（独立复算） | 0 |
| 07 GPU 完整版 | "A100 ~15 min / 4090 ~8 min" | 本机 180s 超时（rc=124） | time-blocked |

---

## 五、脚本运行记录（7 个）

目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S2_P6/`（日志 `NN_*.log`，汇总 `run_summary.txt`）

```
01_explore_data          rc=0    secs=1    PASS
02_bigram_baseline       rc=0    secs=3    PASS
03_attention_trick       rc=0    secs=0    PASS
04_self_attention        rc=0    secs=5    PASS
05_multihead_feedforward rc=0    secs=5    PASS
06_layernorm_transformer rc=0    secs=6    PASS
07_scaleup_generate      rc=124  secs=180  TIME-BLOCKED（GPU 完整版 5000 步 > 3 分钟上限，按规则中断）
```

---

## 六、作业元数据 + pytest 输出

- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_6/`（data 软链经 `S2_hands/data` → 原仓库 data，路径解析正常）
- 编辑文件：仅 `transformer_exercises.py`（只编辑 exercises ✓）
- **做题顺序**：先只凭骨架 docstring + 教程所学实现全部 5 题，然后才跑 pytest（先做后看 ✓）
- 每题记录：

| 题目 | 一次过？ | 额外提示次数 | 耗时 | 结果 |
|------|:---:|:---:|------|------|
| 1 Tokenizer + train/val 划分 | 是 | 0（docstring 步骤即答案提纲） | ~1 min | PASS |
| 2 get_batch | 是 | 0 | <1 min | PASS |
| 3 Bigram（含 2 个测试） | 是 | 0 | ~2 min | PASS ×2 |
| 4 Self-Attention（含 2 个测试） | 是 | 0 | ~2 min | PASS ×2 |
| 5 🌟 Transformer Block | 是 | 0 | ~2 min | PASS |

```
$ cd work/assignment_6 && MPLBACKEND=Agg /home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python -m pytest test_transformer_exercises.py -v

============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.1.1 -- .venv/bin/python
plugins: anyio-4.14.2
collected 7 items

test_transformer_exercises.py::test_exercise_1_tokenize PASSED           [ 14%]
test_transformer_exercises.py::test_exercise_2_get_batch PASSED          [ 28%]
test_transformer_exercises.py::test_exercise_3_bigram_loss PASSED        [ 42%]
test_transformer_exercises.py::test_exercise_3_generate PASSED           [ 57%]
test_transformer_exercises.py::test_exercise_4_scaling PASSED            [ 71%]
test_transformer_exercises.py::test_exercise_4_head PASSED               [ 85%]
test_transformer_exercises.py::test_exercise_5_transformer_block PASSED  [100%]

============================== 7 passed in 0.77s ===============================
```

**5 题全 PASS，首轮 7/7，零返工。** 作业难度梯度合理：1-3 抄教程即可，4-5 需要真的理解 Head/Block 签名变化（作业版把 `n_embd/block_size` 显式传入构造函数，比脚本的全局变量写法更工程化——这点差异作业 docstring 交代得清楚，不构成卡点）。

---

## 七、只改 3 件事

1. **给脚本 07 加快速模式开关**（最重要）：`CPU_MODE = os.environ.get('GPT_QUICK') == '1' or not torch.cuda.is_available()`，并在教程 03 章超参表旁注明"有 GPU 的机器默认走完整版（约 8-15 分钟），加 `GPT_QUICK=1` 可 30 秒过一遍"。否则教程承诺的"<30s 缩小型"在 GPU 机器上是死代码。
2. **教程代码块加"节选"提示**：在 01/02/03 章每个依赖全局变量的代码块上方加一句（或代码块首行注释）：`# 节选自脚本 NN；block_size / batch_size / device / n_embd 等为脚本全局变量`，杜绝孤立复制 NameError。
3. **"实跑日志"标注环境**：在 01-03 章的日志块标题注明 `（CPU，单线程，torch 2.x，seed=1337）`，并在 README 数字表加脚注"CUDA 环境复跑数字有 ±0.03 内漂移"。教程反复强调"看趋势别背数字"，把复现条件写明才算闭环。

---

## 八、最喜欢的 3 处

1. **README 的双列 loss 演进表 + † 脚注**：把"原视频 A100 数字"和"本仓库 CPU 数字"并排，还专门解释 Phase2（+前馈）val 反升的收敛节奏问题——这是学生复跑时最容易恐慌的点，教程提前打好了心理预防针。
2. **LayerNorm std=1.1180 的数值细节**（03 章）：不止给出实跑输出，还解释了为什么不是精确的 1.0（Bessel 校正 sqrt(5/4)）——我复跑逐字对上，这种"连小数点都敢让你验证"的写法极大增强了对全部"真实日志"的信任。
3. **脚本 05 的三阶段 `train_phase` 设计**：每个 Phase 前重置 `torch.manual_seed(1337)`，让多头/前馈/残差三档的 loss 严格可比——把"控制变量做消融"的实验思维悄悄教给了学生，且教程 02→03 章引用的 Phase1 数字（2.4545）与复跑完全一致。

---

*S2 · 2026-09-04 · 审计用时约 18 分钟（对照 8 min + 跑脚本 20 min 墙钟/其中 3 min 为 07 超时 + 作业 10 min）*
