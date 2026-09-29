# Part 7 · 教程数字与证据对账表

> 教程正文引用的每一个实验数字都能在下述落盘文件中复核。所有实验于 **2026-09-29** 在
> 本机复跑（RTX 4090 ×2 / torch 2.6.0+cu124 / Python 3.12，quick 档 + 固定种子），
> 证据目录 `temp/out/logs/`（gitignored，可按右列命令重跑再生成）。

| 教程位置 | 数字 | 证据文件 | 复跑命令（scripts/ 下） |
|---|---|---|---|
| 01 章 | 压缩率：中文 27→17（1.59）、英文 63→16（3.94）；自训 6400 词表 7.1s/14.3MB；留出集 1.52 vs 官方 1.33；重合度 39.2% | `exp01_bpe_compare.csv`、`run_logs/01_bpe.log` | `python 01_bpe_tokenizer.py` |
| 02 章 | 参数账本 28.98M；loss 8.87→7.23（60 步） | `s1_baseline_first.csv`、`run_logs/`（旧 refine 同配置跑） | `python 02_baseline.py` |
| 03 章 | val ppl 4519.11→472.23（300 步） | `s2_baseline_instrumented.csv` | `python 03_experiment_lab.py` |
| 04 章 | fp32/fp16/bf16 末段 loss 6.4355/6.5192/6.5287；tok/s ≈125k/228k/229k（1.8×） | `s3_amp_fp32.csv`、`s3_amp_fp16+scaler.csv`、`s3_amp_bf16.csv` | `python 04_stability_speed.py` |
| 04 章 | 双卡 DDP 各 ~110.6k tok/s（合计未 ×2） | `exp04_ddp.csv`（⚠️ 教程 04 章 DDP 演示块为控制台记录、未落盘 run_logs，其 111.3k 与 csv 110.6k 为运行间波动；重跑补录待办） | `torchrun --nproc_per_node=2 my_minimind.py --stage 3 --ddp-demo` |
| 05 章 | RoPE 数学检查：保范数 4.77e-07、平移不变 1.91e-06 | `run_logs/05_rope.log` | `python 05_rope.py` |
| 05 章 | 外推 ppl@340/@512：learned 416.31/430.60（+3.4%）vs rope 333.88/319.27（−4.4%） | `exp05_rope_vs_learned.csv`（runner=05_rope） | `python 05_rope.py` |
| 06 章 | KV Cache 一致性最大偏差 1.79e-07（GQA+RoPE+QK-Norm 全开） | `run_logs/06_gqa.log` | `python 06_gqa_qknorm.py` |
| 06 章 | MHA 363.53 vs GQA 352.44；KV 投影 0.524M→0.131M | `exp06_gqa_vs_mha.csv`（runner=06_gqa_qknorm） | `python 06_gqa_qknorm.py` |
| 06 章 | QK-Norm 造病：logits std 8.85→1.00 | `exp06_qk_norm.csv` | `python 06_gqa_qknorm.py` 或 `my_minimind --stage 4` |
| 07 章 | 同参预算 ppl：ReLU 310.53 → SwiGLU 278.41（−10%）；每层 FFN 2.10M vs 2.16M | `exp07_relu_vs_swiglu.csv`、`s5_relu.csv`、`s5_swiglu.csv` | `python my_minimind.py --stage 5` |
| 07 章 | MoE α 扫描：gini 0.533/0.115/0.087；max/mean 2.53/1.34/1.28；任务 loss 0.0065/0.0234/0.0190 | `exp07_moe_load_balance.csv` | `python my_minimind.py --stage 5` |
| 08 章 | 参数账本 28.98M→25.83M（≈官方 26M） | 确定性算术（⚠️ stage 6 输出尚未落盘 run_logs，mm6.log 重跑补录待办） | `python my_minimind.py --stage 6` |
| 09 章 | val ppl 5448.66→278.41；截断率 14%；显存 3.2GB | `s7_pretrain.csv`、`run_logs/mm7.log` | `python my_minimind.py --stage 7` |
| 10 章 | val ppl 442.78→66.43；监督占比 ≈71% | `s8_sft.csv`、`run_logs/mm8.log` | `python my_minimind.py --stage 8` |
| 11 章 | DPO 训练池 acc 56.5→80.5、heldout 62.0→45.0（过拟合现场） | `exp11_dpo_generalization.csv`、`run_logs/mm9.log` | `python my_minimind.py --stage 9` |
| 12 章 | RoPE 四件套 ppl：naive 77.71/84.58、PI 88.16/84.55、NTK 78.53/78.45、YaRN 80.85/77.63（双验收 ✅） | `run_logs/12_rope.log` | `python 12_rope_scaling.py` |
| 12 章 | 迷你 RULER needle acc：128 全 1.000；512 档 naive 0.422/PI 0.500/NTK 0.891/YaRN 1.000 | `run_logs/13_ruler.log`、`images/output_long_context.png` | `python 13_long_context_eval.py` |
| 13 章 | MLA KV 账本 MHA 1.07GB/GQA 0.27GB/MLA 0.08GB 等（确定性算术） | 脚本输出（确定性） | `python 14_mla_nsa_accounting.py` |
| 01 章（论断） | BPE 训练侧平局裁决 = token id 序最小：`aaabdaaabac` 手推 merges 为 aa→ab→aaab→ac（第 2 步 2-2 平局取 ('a','b')、第 4 步全 1 平局取 ('a','c')）；low/lower 第 1 步 9-9 平局取 ('l','o') | tokenizers 0.22.2（venv）/0.23.1（系统）双版本 `train_from_iterator` 实测一致，复现片段见教程 01 章 📝 注 | 同左（玩具语料，秒级） |
| 01 章（论断） | 官方 tokenizer 训练语料 = SFT 对话 `sft_t2t_mini.jsonl`（约 90 万行）——非 pretrain、非"本语料全量"；1.52 vs 1.33 主因是分布匹配 | 官方 [`trainer/train_tokenizer.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_tokenizer.py) 的 `DATA_PATH`（2026-09-29 核对 master） | 官方仓库文件 |
| 09 章（论断） | 单条样本字数：median 230 / mean 270.5 / p90 482（sample 2 万条；全量前 5 万行 median 229/mean 268.7 同分布）；官方 tokenizer 编码留出 200 条 token median ≈179、>338 token 占 15.5%（全量 14%）；seq=340 的"500~580 字"是容量口径（338×1.5~1.7）非典型长度 | 2026-09-29 本机实测（jsonl 逐条 len + tokenizer 编码） | 教程 09 章"seq=340 的来历"段 |
| 09 章（论断） | mini 语料 token 总量 ≈2.6 亿：sample 2 万条官方 tokenizer encode_batch 实测均值 204.8 token/条 × 1,270,238 行（含 bos/eos ≈2.63 亿）；×2 epoch ≈ 5.2 亿 = Chinchilla 26M×20 | 2026-09-29 本机实测（encode_batch 全 2 万条） | 教程 09 章"对账"段 |

**口径备注**：① 单次训练实验（05/06/07/09/10/11 章）的 ppl 绝对值含随机波动（种子固定
则同机可复现），教程引用时保留两位小数并标注"单次短训"；跨模型比较都基于同一评估集。
② 12 章四件套在官方 6400 词表语料上 ppl 量级整体高于旧版字符级口径（77~88 vs 5~14），
教程只使用同语料内的相对排序。③ 消融证据 `exp*.csv` 为追加式（带 date/runner 列），
多次运行可对照漂移。
