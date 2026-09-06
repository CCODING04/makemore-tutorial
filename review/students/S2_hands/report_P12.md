# S2 学习报告·P12（微调实战 LLaMA-Factory）

- 审计身份：学生 agent S2（实操薄弱型），先动手后对照
- 审计时间：2026-09-04，节奏 ≤20 分钟（教程阅读 ~5min + 脚本运行 ~3min + 作业 ~10min + 报告）
- 材料范围：`courses/Part12_finetune_llamafactory/tutorial/`（README + 01 + 02）、`scripts/01_handwritten_sft_lora.py`、`assignments/assignment_12/` 三件；未读 assignment_reference 及其他禁读项，未联网

## 总分：90 / 100

| 章节 | 得分 | 一句话理由 |
|---|---|---|
| README.md | 88 | 定位/前置/硬件档位表清晰，但**下载依赖说明缺失** |
| 01 手写 LoRA SFT | 95 | 教程声称的每一个数字都与脚本实测逐字吻合，字段对照表是全章之魂 |
| 02 LLaMA-Factory 工作流 | 87 | 命令可直接照抄且自带"文件名以版本为准"免责声明；零下载/镜像指引，7B 链路本审计未能实跑 |
| Assignment 12 | 95 | 骨架 docstring 把公式给足，5/5 一次通过；stretch 的 SKIP 设计优雅；实验题对无卡学生门槛偏高 |

## 卡点清单（1 高 / 1 中 / 4 低）

1. 【高】**权重/数据集下载依赖档位说明缺失**。grep 全部 3 个 .md：除 peft 链接外无任何 huggingface/下载/镜像/网络字样。02 章直接给出 `--model_name_or_path Qwen/Qwen2.5-0.5B-Instruct`、`Qwen2.5-7B-Instruct`、`--dataset identity,alpaca_gpt4_zh`，默认学生能直连 HF。README 硬件档位表（CPU/24GB/多卡）很好，但缺配套的"每档位首次要下载什么、多大磁盘、HF_ENDPOINT 镜像怎么设、离线用户只能跑到哪一步"。实操薄弱型学生的第一幕就会卡在模型拉取上。
2. 【中】**耗时口径不一致**：教程 01 章称脚本"~3 秒"，脚本 docstring 称"~40 秒"。实测：GPU wall 2.4s、CPU wall 2.6s（user 37.6s，多线程）。教程的 wall 口径准确；docstring 的 40s 应是单线程 CPU user 口径。两处都有依据但会让学生自我怀疑"是不是我跑错了"。
3. 【低】脚本末尾 yaml 对照打印写 `per_device_batch`，教程表格和 LLaMA-Factory 真实字段是 `per_device_train_batch_size`。
4. 【低】01 章"以上为脚本真实输出"的五步概览块是**摘要**而非逐字输出："chat 格式 3/3 / 回声 2/3" 的判读是 md 给的，脚本只打印三行推理文本，不打印任何 "3/3" 字样。逐字对照的学生会愣一下。
5. 【低】assignment 题 4 公式算出 3.74GB，而 02 章称"QLoRA 7B ≈ 6GB"——公式只含底座量化+可训练两项，不含激活/梯度波动，作为"为什么 6GB 能跑"的粗估略松。题内验收自洽（≈3.74），不扣作业分，但学生拿 3.74 对 6.0 会困惑，宜加一句"差值=激活+波动"。
6. 【低】02 章全部 7B/0.5B 工具链命令本审计**未实际执行**（需下载模型并安装 LLaMA-Factory，超出本审计时长）；yaml 字段名/默认值核对基于对 LLaMA-Factory 的既有知识（train_on_prompt 默认 false、double_quantization 默认 true、quantization_type 默认 nf4、pref_loss 默认 sigmoid），无法在线复核，教程自带的"以安装版本为准"免责声明是唯一安全网。

## 分章评分明细

### README.md（88）
- ✅ 学习目标、链路定位图、前置知识（Part 8 08/02/03 → 本章）链接全通
- ✅ LoRA 数学推导自查：d=k=4096、r=8 → 16,777,216/65,536 = **256 倍**，对
- ✅ 硬件档位表分 CPU / 1×24GB / 多卡，明确"脚本 01 无需任何安装"
- ✅ 学习地图"点→线→面"与 01/02 两章结构严格对应
- ❌ 下载依赖零说明（卡点 1）

### 01 手写 LoRA SFT（95）
- ✅ 五步概览与脚本实测逐项吻合（见一致性核对表）
- ✅ 形状追踪图自查：4×96 + 288×4 = 384+1152 = 1,536/层；27,648/1,536 = 18 倍；2 Block × 2 Linear × 1,536 = 6,144，与脚本 `[1]` 打印一致
- ✅ 与脚本同口径声明（r=4, alpha=8.0）属实，数字可直接对上
- ✅ 三个调试错误（忘搬 device / dtype 不一致 / mask 丢失）都在脚本中有对应代码（`model.to(DEVICE)`、`labels[:n_prompt]=-100`）
- ✅ 性能表"玩具行 <1min"与两种口径实测均相容
- ⚠️ 卡点 2/4

### 02 LLaMA-Factory 工作流（87）
- ✅ CLI 字段名逐一核对（凭知识，离线）：`model_name_or_path / dataset / template / finetuning_type / lora_target / lora_rank / lora_alpha / output_dir / per_device_train_batch_size / learning_rate / num_train_epochs / plot_loss / quantization_bit / double_quantization / quantization_type / pref_beta / pref_loss / cutoff_len / gradient_checkpointing / export_dir / export_size / export_legacy_format / adapter_name_or_path`——全部是 LLaMA-Factory 真实参数名，未发现编造字段
- ✅ "默认值"声称核对：`train_on_prompt` 默认 false ✓、`double_quantization` 默认 true ✓、`quantization_type` 默认 nf4 ✓、`pref_loss` 默认 sigmoid ✓（均与教程一致；无法联网复核，标注 * ）
- ✅ QLoRA 数学自查：7B×0.5B=3.5GB、双重量化省 ~0.37GB、合计 3.87GB，与 QLoRA 论文口径一致
- ✅ 新版 examples 文件名变更（qwen_lora_sft.yaml → qwen3_lora_sft.yaml）主动加了免责声明，处理得当
- ✅ 0.5B 行标注"本机实测"、7B 行标注"官方量级参考，未逐行本机复现"——数据来源诚实
- ❌ 卡点 1/5/6

### Assignment 12（95）
- ✅ 5/5 一次通过；测试同时支持 pytest 与独立运行两种模式，均验证通过
- ✅ 题 5 stretch 数值完美复现教学论点（见下）
- ⚠️ 实验题（跑 02 章工具链）对无 GPU/无网学生不可达，但核心 100 分不依赖它，设计合理

## 一致性核对表（教程代码块 ↔ 脚本 ↔ 实测）

| # | 教程声称 | 脚本/实测 | 结论 |
|---|---|---|---|
| 1 | `[1]` 可训练 6,144/200,664（3.1%） | 实测打印逐字相同（注入后总参数含新增 A/B：194,520+6,144=200,664，自洽）| ✅ |
| 2 | `[2]` loss 3.572 → 0.076 | 实测 3.572 → 0.076（seed 1337/7 固定，可复现）| ✅ |
| 3 | `[3]` chat 3/3、回声 2/3 | 实测 w0w5→w0 ✓、w3w7→**w19** ✗、w10w2→w10 ✓ = 2/3 | ✅ |
| 4 | `[4]` 合并=精确加法，行为一致 | 实测 max\|Δlogits\| ≤ 2.38e-06 < 断言阈值 1e-4；assignment Q3 的"≤ 2.4e-06"亦吻合 | ✅ |
| 5 | 形状图：LoRALinear(base 96→288, A(4,96), B(288,4), scaling 8/4=2.0) | 脚本 `LoRALinear.__init__`/`forward` 同构；A `randn/√r` = "N(0,1)/√r" | ✅ |
| 6 | 五步↔yaml 对照表 6 行 | 脚本末尾打印的 6 行映射一致（卡点 3 的字段缩写除外）| ✅ |
| 7 | 教程"~3 秒" | GPU wall 2.4s / CPU wall 2.6s | ✅（docstring "~40 秒" 为 CPU 单线程 user 37.6s 口径，口径不一致→卡点 2）|
| 8 | 01 章"~250 行"脚本 | 实际 311 行（含注释/空行）| ✅ 约数成立 |
| 9 | README 压缩比 256 倍 | 手算 16,777,216/65,536=256 | ✅ |
| 10 | QLoRA 3.5GB + 0.37 ≈ 3.87GB | 与 QLoRA 论文一致 | ✅ |
| 11 | 02 章 yaml/CLI 字段名与默认值 | 凭 LLaMA-Factory 知识逐个核对一致 | ✅*（离线，无法在线复核）|
| 12 | 02 章 examples 文件名 | 自带"以安装版本为准"免责声明 | ✅ 处理得当 |
| 13 | 下载依赖档位说明 | grep 仅 1 处 huggingface（peft 链接），无下载/镜像/磁盘指引 | ❌ 缺失（卡点 1）|

## 作业元数据 + pytest 输出

- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_12/`（只编辑了 `finetune_exercises.py`，三件齐全）
- 环境：`/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python`（Python 3.12.12 / pytest 9.1.1 / torch 可用 / RTX GPU）

| 题 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|---|---|---|---|
| 1 LoRA 参数账 | ✅ | 0 | ~2 min | PASS（3072；ratio=3072/200000）|
| 2 合并数学 | ✅ | 0 | ~3 min | PASS（注意点：不原地改 W → 用 `W + ...` 而非 `+=`）|
| 3 B 零初始化 | ✅ | 0 | ~2 min | PASS（`norm(p="fro")`）|
| 4 QLoRA 显存账 | ✅ | 0 | ~3 min | PASS（3.74GB）|
| 5 🌟 多 rank sweep | ✅ | 0 | ~12 min | PASS（耗时在 seed/generator 细节与 loss_start 记录时机）|

```
test_finetune_exercises.py::test_ex1_lora_params PASSED          [ 20%]
test_finetune_exercises.py::test_ex2_merge PASSED                [ 40%]
test_finetune_exercises.py::test_ex3_zero_init PASSED            [ 60%]
test_finetune_exercises.py::test_ex4_vram PASSED                 [ 80%]
test_finetune_exercises.py::test_ex5_rank_sweep_stretch PASSED   [100%]
============================== 5 passed in 1.16s ==============================
```
独立运行模式（assignment.md 指示的用法）：`通过: 5/5 🎉`

stretch 实验数值（r=1/2/4/8，loss_start 全部 0.008290，精确相同）：
`0.00504 → 0.00200 → 1.08e-06 → 4.25e-15`——r≥目标秩 4 后 loss 塌缩到近零，r<4 卡在平台，"lora_rank 该开多大"被实验直接回答。

## 只改 3 件事

1. **补"下载依赖与档位"段**（README 环境节 + 02 章开头）：每档位（CPU 离线 / 24GB / 多卡）首次需下载的模型与数据集、大致磁盘占用、`export HF_ENDPOINT=https://hf-mirror.com` 一行镜像提示、"离线用户只能跑到脚本 01"的明确边界。
2. **统一耗时口径**：脚本 docstring 改为"~3 秒（GPU/CPU wall 口径；单线程 CPU 约 40s）"，与教程 "~3 秒"对齐，消除学生自我怀疑。
3. **脚本末尾 `per_device_batch` → `per_device_train_batch_size`**：与教程对照表及 LLaMA-Factory 真实字段名对齐。

## 最喜欢 3 处

1. **01 章"五步 ↔ yaml 字段"对照表**（脚本末尾还有同款打印收尾）：把 LLaMA-Factory 黑盒逐字段打开，做作业题 1/2/4 时反复回查的就是这张表——"手写教会映射，工具给工程完备性"的结论立得住。
2. **合并正确性用逐元素 logits 比对而非采样文本比对**：脚本注释明确解释"argmax 可能因浮点舍入在平局上翻转，不能作为行为一致判据"，实测 max|Δlogits|=2.38e-06、断言阈值 1e-4——比"看起来输出一样"严谨一个量级，且 assignment Q3 的数字与实测吻合。
3. **题 5 stretch 的实验设计**：给定秩 4 的真实 ΔW*，四个 rank 的 loss_end（0.0050→0.0020→1.1e-6→4.2e-15）让学生亲手得出"rank 够了再加只加参数不加效果"；配合"未实现返回 None → 优雅 SKIP 不扣分"的测试设计，对实操薄弱型学生极其友好。
