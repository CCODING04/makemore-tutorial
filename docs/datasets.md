# 数据集说明与下载指南

> 本课程"开箱即跑"的脚本只依赖仓库内的小文件（`data/names.txt` 228KB、`data/input.txt` 1.1MB）。
> 只有当你想要**贴近原仓库/真实规模**复现时，才需要下面这些大尺寸数据集。
> 每个条目给出：下载页面/命令、体积、格式介绍、对应章节。

## 1. Part 7 — minimind 官方数据（中文 LLM 全流程复现）

| 文件 | 体积 | 用途 | 每行格式（jsonl） |
|---|---|---|---|
| `pretrain_t2t_mini.jsonl` | 1.2 GB | 预训练 | `{"text": "如何才能摆脱拖延症？..."}` |
| `sft_t2t_mini.jsonl` | 1.6 GB | SFT 多轮对话 | `{"conversations": [{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}` |
| `dpo.jsonl` | 53 MB | DPO 偏好对 | `{"chosen": [...], "rejected": [...]}` |

**下载**（国内推荐 ModelScope，断点续传：重跑同一命令即可）：

```bash
pip install modelscope
modelscope download --dataset gongjy/minimind_dataset \
  pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl dpo.jsonl --local_dir ./dataset
# HuggingFace 备选：export HF_ENDPOINT=https://hf-mirror.com 后用 huggingface-cli download
```

- 页面：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
  （HF 镜像：https://huggingface.co/datasets/jingyaogong/minimind_dataset）
- 介绍：minimind 作者整理的中文对话语料，t2t = text-to-text；mini 版是官方推荐的
  最小可复现组合（共 ~2.9GB，单卡 3090 全流程 ≈2.3 小时）。特殊 token：`<|im_start|>`/`<|im_end|>`。
- 对应章节：[Part 7 05 章·复现 minimind 毕业指南](../courses/Part7_minimind/tutorial/05_reproduce_minimind.md)

## 2. Part 8 — train-llm-from-scratch 的英文数据（原版规模复现）

| 数据集 | 体积（下载后） | 用途 | 获取 |
|---|---|---|---|
| The Pile（子集） | 数 GB～数百 GB | 预训练 | HF: `montinger/the-pile-en` 或官方 https://pile.eleuther.ai/ |
| Alpaca | ~40 MB | SFT 指令数据 | `datasets.load_dataset("tzu-alpaca/alpaca-cleaned")` 或 https://huggingface.co/datasets/tatsu-lab/alpaca （52K 条 instruction/input/output） |
| Dolly-15k | ~10 MB | SFT（人工书写） | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| HH-RLHF | ~150 MB | 偏好对（chosen/rejected 对话） | https://huggingface.co/datasets/Anthropic/hh-rlhf |
| UltraFeedback | ~400 MB | 偏好对（64K，GPT-4 标注） | https://huggingface.co/datasets/openbmb/UltraFeedback |
| GSM8K | ~5 MB | RLVR 数学题（8.5K 小学数学题，`#### 42` 答案格式） | `load_dataset("openai/gsm8k", "main")` |

```python
# 统一入口（Part 8 脚本 03 已支持 --original-data 开关加载 Alpaca）
from datasets import load_dataset
load_dataset("openai/gsm8k", "main")            # RLVR 奖励
load_dataset("tatsu-lab/alpaca")                # SFT
load_dataset("Anthropic/hh-rlhf")               # DPO/偏好
```

- 对应章节：[Part 8 README 规模对照表](../courses/Part8_post_training/tutorial/README.md) ·
  03 章 `--original-data` 开关 · 05 章 GSM8K 评估

## 3. Part 13 — 预训练数据工程的参照语料

| 数据集 | 体积 | 介绍 | 获取 |
|---|---|---|---|
| FineWeb | 15T tokens（~9TB） | 当前最佳开源预训练语料之一（C4 级清洗 + 全局 MinHash 去重 + 质量过滤） | https://huggingface.co/datasets/HuggingFaceFW/fineweb （`sample-10BT` 子集 ~28GB 适合学习） |
| FineWeb-Edu | 1.3T tokens | 用 LLM 质量分类器筛出的"教育性"子集 | 同上仓库 `sample-10BT` |
| C4（en） | ~750 万文档 | 经典基线语料（T5 用的 Cleaned MassiveText 子集） | `load_dataset("allenai/c4", "en", streaming=True)` |

> 学习建议：不需要下载全量——用 `streaming=True` 或 `sample-*` 子集体验管线；
> Part 13 的作业与脚本全部使用**脚本内合成的玩具语料**，零下载。

## 4. Part 11/12/14 — 模型权重（非数据，但同为"大文件"）

| 模型 | 体积 | 用途 | 获取 |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | ~1 GB | verl quickstart（单卡 ≥24GB）/ vLLM 对比实验 / Part 14 基线 | HF：`Qwen/Qwen2.5-0.5B-Instruct`（自动下载到 ~/.cache/huggingface） |
| Qwen2.5-7B-Instruct | ~15 GB（safetensors） | QLoRA 微调主菜（4bit 后训练显存 6GB） | HF 同上 |
| minimind 官方权重（26M/64M） | 100-250 MB | Part 7 验收对照 | https://www.modelscope.cn/models/gongjy/minimind-3-pytorch |

> ⚠️ 教学策略：这些权重**不进仓库、不进课程验证流程**——课程脚本全部自包含（合成数据 +
> 从零训练的小模型）；上表仅供你想跑"原版规模"时按图索骥。
