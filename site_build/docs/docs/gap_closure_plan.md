# 缺口补全实施计划（Gap Closure Plan）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 `docs/course_roadmap_v3.md` §6 缺口登记中的全部 7 个可行动缺口（RAG 全链路、Agent/function calling、scaling law、幻觉/安全对齐、lm-eval-harness 实操、YaRN 长上下文、Flash Attention 手写内核），每个新增/扩展内容按 tutorial-creator 标准达到 ≥4.5 分。

**Architecture:** 两新建 Part（Part18_rag、Part19_agents，对应应用线 A1/A2）+ 三处原地扩展（Part7 脚本 11 加 YaRN、Part8 07 章扩幻觉/安全 + 新增两个实操脚本、Part9 新增 FA 内核章、Part13 新增 00 章 scaling law）+ 横切文档同步（根 README / roadmap / 面试指南 / requirements）。每个任务独立交付、独立验证、独立提交。

**Tech Stack:** Python 3.12 / torch 2.6.0+cu124 / triton 3.2（项目 `.venv`，2×RTX 4090）；可选新依赖：`lm_eval[hf]`；需下载模型：Qwen3-Embedding-0.6B（~1.2GB）、（可选）Qwen2.5-7B-Instruct（refusal direction 演示）；已缓存：Qwen/Qwen2.5-0.5B-Instruct。

## Global Constraints

- 运行环境：`/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python`；`export MPLBACKEND=Agg`；CUDA 编译需 `CUDA_HOME=/usr/local/cuda-12.4` 且 PATH 含 `.venv/bin`。
- 硬件预算：所有"核心路径"脚本单卡 4090 ≤30s（冒烟模式）可跑；训练扫描类脚本必须有 `--mode smoke|full` 双模式。
- 质量门禁：每个新增 Part/章按 `skills/tutorial-creator/references/good-tutorial-standard.md` 四维加权 ≥4.5（独立复评，非自评）。
- 脚本规范：自包含、`__file__` 相对路径、固定种子、shape 注释、分段 debug 打印（见 `skills/tutorial-creator/references/scripts-guide.md`）。
- 作业规范：4 核心 + 1 🌟stretch（未实现返回 None 优雅 SKIP ⏭️）+ 3-6 道思考题 `<details>` 完整答案 + 每题验收清单；`assignment_reference/` 参考实现必须全过（见 `assignment-guide.md`）。
- 文风：中文正文、图表英文标题、emoji 规范（🔑⚠️💡📝🧪📊）；教程引用数字必须来自真实运行并标注环境。
- **引用红线（调研已核实）**：① GPU MODE FA 讲座主讲是 Thomas Viehmann（非 Colin Atkinson）；② 以下论断未核实、禁止写入或必须标注"未核实"：DeepSeek-V3 的 N^0.77·D^0.23 指数、MCP 捐赠 Linux Foundation、"LFronTQ" 基准（改用 LongProc）、"AI Studio FA2 系列"（改用 tspeterkim/rishisankar/echen 三资源）；③ SWE-bench 分数只引 swebench.com 官方站并注明脚手架。
- 不改既有 Part 1-17 的已达标内容（除本计划明确列出的衔接点）；每任务完成后单独 commit（不 push，最终统一由用户确认后 push）。

---

### Task 1: Part 7 — YaRN 外推 + 迷你长上下文评测

**Files:**
- Modify: `courses/Part7_minimind/scripts/11_rope_scaling.py`（加 yarn 方案）
- Create: `courses/Part7_minimind/scripts/13_long_context_eval.py`（迷你 RULER）
- Modify: `courses/Part7_minimind/tutorial/02_modern_components.md:255`（"我们教程先不做 scaling"→ 指向脚本 11 的 YaRN 实测）、`05_reproduce_minimind.md:137` 附近（三件套 → 四件套）
- Modify: `courses/Part7_minimind/tutorial/README.md`（导航表加脚本 13）

**Interfaces:**
- Consumes: 现有 `11_rope_scaling.py` 的 `build_rope_cache / apply_rope` 接口与 naive/PI/NTK 三分支
- Produces: `yarn_inv_freq(dim, base, s, alpha=1, beta=32)` 与 `attn_factor = 0.1*ln(s)+1`；`13_long_context_eval.py` 的 `make_kv_task(n_vars, ctx_len)` 与 `needle_accuracy(model, scheme, ctx_len)`

- [ ] **Step 1: 在脚本 11 增加 yarn 分支**（照抄 HF `modeling_rope_utils.py` 的三部件，写清论文 α/β 与 HF `beta_fast/beta_slow` 反向命名的注释）：

```python
def yarn_params(dim, base, s, alpha=1.0, beta=32.0):
    """YaRN: NTK-by-parts 插值 + 注意力温度。论文 2309.00071，实现对照 HF _compute_yarn_parameters。"""
    def find_correction_dim(num_rot, dim, base, max_pos):
        return dim * math.log(max_pos / (num_rot * 2 * math.pi)) / (2 * math.log(base))
    low  = find_correction_dim(beta,      dim // 2, base, 10)   # β_slow=32 → 下界
    high = find_correction_dim(alpha,     dim // 2, base, 128 * s)  # β_fast=1 → 上界（HF 变量名与论文希腊字母相反）
    ramp = clamp((arange(dim // 2) - low) / (high - low), 0, 1)     # 逐维 0(全插值)→1(全外推)
    inv_freq = (1 / (s * f)) * (1 - ramp) + (1 / f) * ramp          # f = base**(-2i/dim)
    attn_factor = 0.1 * math.log(s) + 1                              # √(1/t)，作用于 softmax 前
    return inv_freq, attn_factor
```

在现有 naive/PI/NTK 的 ppl 对比表中加第 ④ 行 `yarn`；温度实现选 `q = q * attn_factor`（嵌入现有 `apply_rope` 加 `attn_scale` 参数）。
- [ ] **Step 2: 单元自检 + 预期排序验证**。运行脚本 11，验收（写进输出）：
  - `s=1` 时 yarn 的 inv_freq 与 naive 逐元素 allclose（温度=1）；
  - `ppl@128: yarn ≈ naive`（温度 1.069 影响极小）；`ppl@256: yarn ≤ ntk < naive`。
- [ ] **Step 3: 新建 `13_long_context_eval.py`（迷你 RULER）**：合成 KV 检索任务（"a=3, b=7, … query: a?"），上下文 {128, 256, 512} 三档 × {naive, pi, ntk, yarn} 四方案画 needle 准确率曲线（matplotlib 英文标题存 `output_long_context.png`，`__file__` 相对路径）。CPU ≤30s。
- [ ] **Step 4: 更新两处教程**。`02_modern_components.md:255` 改为："工业界用 YaRN/NTK scaling 缓解——脚本 11 已实测四种方案（YaRN 的温度因子 √(1/t)=0.1·ln(s)+1 是面试加分点）"；`05_reproduce_minimind.md` 三件套改四件套并贴 Step 2/3 的真实输出（标注环境）。补一段"为什么 NIAH 不够"（RULER 2404.06654：宣称 128K 的模型多在 ~32K 失效；LongBench v2 2412.15204：直接作答最佳仅 50.1%），附 RULER/HELMET/LongBench v2 链接。
- [ ] **Step 5: 验证与提交**：

```bash
cd courses/Part7_minimind/scripts && MPLBACKEND=Agg /path/to/.venv/bin/python 11_rope_scaling.py && MPLBACKEND=Agg .../python 13_long_context_eval.py
git add -A courses/Part7_minimind && git commit -m "feat(P7): YaRN 外推实测 + 迷你 RULER 长上下文评测（缺口 A）"
```

---

### Task 2: Part 9 — Flash Attention 手写 Triton 内核章

**Files:**
- Create: `courses/Part9_cuda_kernels/scripts/09_flash_attention_triton.py`
- Create: `courses/Part9_cuda_kernels/tutorial/05_flash_attention.md`
- Modify: `courses/Part9_cuda_kernels/tutorial/README.md`（导航表加 05 章 + 脚本 09；实测参考表加 FA 行）

**Interfaces:**
- Consumes: 02 章 tiling 直觉、07_triton_kernels.py 的 softmax 内核（开篇复用为动机）
- Produces: `fa_forward(q, k, v, causal=True)` Triton 内核包装函数（bf16 输入、fp32 累加），与 `bench(fn)` 计时协议

- [ ] **Step 1: 写脚本（四段）**。
  段 1 基准：naive attention + `F.scaled_dot_product_attention`，用 `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION])` 逐一锁定后端并**打印实际命中者**（不得假设 4090 命中 flash）。
  段 2 内核核心（online softmax，exp2 技巧）：

```python
m_new = tl.maximum(m, tl.max(qk, 1))          # 行最大值（fp32）
p     = tl.math.exp2(qk * scale * LOG2E - m_new * LOG2E)  # 稳定 softmax 分子
alpha = tl.math.exp2((m - m_new) * LOG2E)     # 旧和的缩放因子
l     = l * alpha + tl.sum(p, 1)              # 分母滚动和
acc   = acc * alpha + tl.dot(p.to(v.dtype), v)# 输出滚动累加
```
  causal 按官方 tutorial 06 的三阶段分解（无 mask 块 → 对角块 → 跳过）；边界用 `tl.load(..., mask=..., other=0)` 且 `qk_scaled = qk*scale + tl.where(valid, 0, -1.0e6)`（勿用 -inf 防 NaN）；`tl.dot` 输入 fp16/bf16、acc fp32；autotune 网格 `BLOCK_M∈{64,128}, BLOCK_N∈{64,128}, num_warps∈{4,8}, num_stages∈{2,3}`。
  段 3 数值验收：与 naive 对照 `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`（bf16），打印最大相对误差（exp2/scale 顺序引入 1e-3 级差异属正常，注释说明）。
  段 4 性能：`torch.cuda.Event` 计时，预热 10 + 测 50，扫 seq={1K,2K,4K}×{causal,non-causal}（d=64 对齐 minimind）；**验收承诺写进输出：教学版前向 ≥ SDPA 最优后端的 50% 合格、>85% 优秀**（依据：FlexAttention 官方为 FA2 的 90% = Triton 路径上限）。
  段 5（各 ~20 行）：FlexAttention `create_block_mask` 复现同一 causal/sliding 结果；SageAttention 只作一段文字（4090 恰是 INT4 甜点卡，2411.10958，4090 上 ~3× FA2）。
- [ ] **Step 2: 写 `05_flash_attention.md`**。结构：学习目标（4 条）→ 前置（02 章 tiling、07 章 Triton softmax）→ 理论（FA1 2205.14135 → FA2 2307.08691 → FA3 2407.08608 仅 Hopper/WGMMA 是 SM90 指令、4090 用不了 → FlexAttention → SageAttention 演进表含 arXiv 号）→ online softmax 逐步推导（含上段代码逐行）→ 实测（贴 Step 1 真实数字 + 环境标注）→ 陷阱（边界 -inf NaN、fp32 累加、causal 块边界行误差定位法）→ 3 概念题 + 2 动手（把内核接回 `Part7/05_full_model.py` 替换 SDPA 测 ppl；改 BLOCK 尺寸画性能曲线）。引用资源：Triton 官方 tutorial 06、GPU MODE Lecture 12（Thomas Viehmann）、tspeterkim/flash-attention-minimal、echen.io CuTe 系列。
- [ ] **Step 3: 验证与提交**：

```bash
export CUDA_HOME=/usr/local/cuda-12.4; cd courses/Part9_cuda_kernels/scripts && ../../.venv/... python 09_flash_attention_triton.py  # rc=0，数值断言过
git add -A courses/Part9_cuda_kernels && git commit -m "feat(P9): 手写 Triton Flash Attention 内核章（缺口 F）"
```

---

### Task 3: Part 8 — 幻觉与安全对齐扩节

**Files:**
- Create: `courses/Part8_post_training/scripts/11_hallucination_safety.py`
- Modify: `courses/Part8_post_training/tutorial/07_evaluation.md`（扩三小节）
- Modify: `courses/Part8_post_training/tutorial/README.md`、根 `README.md` Part 8 行（核心概念加"幻觉/安全"）

**Interfaces:**
- Consumes: `Qwen/Qwen2.5-0.5B-Instruct`（HF 已缓存）与 `Qwen/Qwen2.5-0.5B`（base，需下载 ~1GB）
- Produces: `sample_n(prompt, n, t)`、`semantic_entropy(answers, nli_model)`、`ece(conf, correct)` 三个可复用函数（作业 8 扩展题将引用）

- [ ] **Step 1: 写脚本三实验**。
  实验 A 简化 semantic entropy：内置 30 题（15 个易幻觉 trivia + 15 个稳定事实），`do_sample, T=0.7, n=10`；答案聚类用句向量余弦（≥0.8 同簇，无 NLI 依赖、纯 torch）；SE = 聚类分布熵；报告 SE 与"多数答案是否正确"的 AUROC（手写 rank-based AUROC）。叙事锚点：Farquhar et al. Nature 630 (2024) + SelfCheckGPT（EMNLP 2023）。
  实验 B 温度 sweep：T∈{0.0,0.3,0.7,1.0} × 幻觉率（以多数投票答案对错为标签）曲线——破除"低温=更事实"迷思（2402.05201：任务相关）。
  实验 C 校准对比：base vs Instruct 同一批 20 道 3 选 1 选择题，输出 token 概率为置信度，算 ECE（10 桶）——呼应 GPT-4 报告 Figure 8 "RLHF 损害校准"（注明是 GPT-4 时代结论，DPO 系有 TruthRL 2509.25760 新证据）。
  可选段（🌟，模型缺失则打印提示跳过）：refusal direction——Qwen2.5-7B-Instruct 上 harmful/harmless 各 100 条 diff-in-means 提方向、方向消融解除拒绝（Arditi 2406.11717）。**只做白盒机制演示，不提供端到端黑盒越狱教程。**
- [ ] **Step 2: 07_evaluation.md 扩三小节**（插在"## 5 给自己的模型搭一条最小可信评估线"之后）：① 幻觉：机理 → 检测两族谱（黑盒一致性 vs 白盒 probe：Lookback Lens 2407.07071、SE Probes 2406.15927）→ 缓解（RLHF 校准代价/RAG/弃权训练）；② 安全：refusal 机理 → 攻（GCG 思想一句话）防（Constitutional Classifiers 2501.18837）→ 评测（HarmBench 2402.04249/JailbreakBench；**明确 AdvBench/TruthfulQA 已过时不作主证据**）；③ 中国合规一页纸：双备案制（《生成式 AI 服务管理暂行办法》第 17 条 + 深度合成规定）、2025-09 内容标识办法、面试答法"算法备案+大模型备案+安全评估题库+内容标识"四件套。贴 Step 1 真实输出（环境标注）。
- [ ] **Step 3: 验证与提交**：

```bash
cd courses/Part8_post_training/scripts && MPLBACKEND=Agg .../python 11_hallucination_safety.py  # rc=0（7B 未下载时🌟段自动跳过）
git add -A courses/Part8_post_training && git commit -m "feat(P8): 幻觉与安全对齐扩节——SE/温度sweep/校准三实验+合规一页纸（缺口 H-1）"
```

---

### Task 4: Part 8 — lm-evaluation-harness 实操

**Files:**
- Create: `courses/Part8_post_training/scripts/12_lm_eval_hands_on.py`（python API 驱动 + 自定义 task）
- Create: `courses/Part8_post_training/scripts/mytasks/course_quiz.yaml` + `mytasks/utils.py`
- Modify: `courses/Part8_post_training/tutorial/07_evaluation.md:26-40`（§2 lm-eval-harness 节后接"实操"小节）
- Modify: `requirements.txt`（可选依赖加 `lm_eval[hf]  # v0.4.10+ 默认不装 HF 栈，必须 extras`）

**Interfaces:**
- Consumes: `lm_eval.simple_evaluate(model="hf", model_args=..., tasks=[...], num_fewshot=, limit=, log_samples=)`；自定义任务经 `include_path`
- Produces: `mytasks/course_quiz` 任务（generate_until 型，5 道本课程知识题）

- [ ] **Step 1: 写驱动脚本**：① `lm_eval[hf]` 未安装 → 打印安装指引并退出 rc=0（优雅降级）；② 跑 `arc_easy --limit 100 --num_fewshot 0`（Qwen2.5-0.5B-Instruct）打印 accuracy；③ 同任务 `num_fewshot=5` 对比（分数不可互比的坑写进输出）；④ 加载 `mytasks/course_quiz.yaml`（`doc_to_text` Jinja、`doc_to_choice`、`acc` 指标）跑通。YAML 核心：

```yaml
task: course_quiz
dataset_path: json
dataset_kwargs: {data_files: {test: course_quiz.jsonl}}
output_type: multiple_choice
doc_to_text: "Q: {{question}}\nA:"
doc_to_target: "{{answer}}"
doc_to_choice: "{{choices}}"
metric_list: [{metric: acc}]
```

- [ ] **Step 2: 教程实操小节**：安装坑（v0.4.10 extras 变更、`trust_remote_code` 必须进 `--model_args`、`--batch_size auto` 可能 OOM 用 `auto:N`、seed 四元组 `python,numpy,torch,fewshot`、v0.4.12+ vLLM 最低版本变严）；vLLM 后端用法给出命令但标注"本课环境未装 vLLM，命令为预期行为"；替代品格局表（lm-eval 英文学术 / OpenCompass 中文生态 / lighteval HF 生产）。参考文献加 Biderman 2405.14782。
- [ ] **Step 3: 验证与提交**：

```bash
uv pip install --python .venv "lm_eval[hf]"   # 先装（记录版本号进教程）
cd courses/Part8_post_training/scripts && .../python 12_lm_eval_hands_on.py  # rc=0，两个 fewshot 数字 + 自定义任务 acc 打印
git add -A courses/Part8_post_training requirements.txt && git commit -m "feat(P8): lm-eval-harness 实操+自定义 task（缺口 H-2）"
```

---

### Task 5: Part 13 — Scaling Law 开篇章（00 章）

**Files:**
- Create: `courses/Part13_data_engineering/tutorial/00_scaling_laws.md`
- Create: `courses/Part13_data_engineering/scripts/00_scaling_laws.py`（`--mode fit|scan|epoch` 三子命令）
- Modify: `courses/Part13_data_engineering/tutorial/README.md`（导航表加 00 章为"开篇"，说明建议先读）、`01_dedup_from_scratch.md` 开头加一句前读指引

**Interfaces:**
- Consumes: `data/input.txt`（epoch 实验的 unique 语料）；Part 8 的 `01_gpt_model.py` 风格小 GPT（脚本内自建 1M-10M 参数模型，不 import 跨 Part）
- Produces: `chinchilla_loss(N, D, params)` 与 `fit_chinchilla(records)`（Huber 损失，`scipy.optimize`，返回 E/A/α/B/β）；`records` 格式 `(N, D, final_loss)` 列表

- [ ] **Step 1: 写脚本三模式**。
  `fit`（零 GPU，秒级）：对内置的合成-加噪 Chinchilla 数据（按 E=1.69, A=406.4, α=0.34, B=410.7, β=0.28 生成 + 3% 噪声，注释声明来源 Hoffmann 2203.15556 表格拟合值）跑 `fit_chinchilla`，验收拟合参数相对误差 <5%，ASCII + png 画 isoFLOP 带（C=6ND 等值线上的 loss 谷）；另附零算力替代方案说明（Pythia checkpoint loss 拟合练习）。
  `scan`（冒烟 ≤60s / full 可选）：网格 N∈{1M, 3M, 10M} × D∈{2M, 6M, 20M}（smoke）训小 GPT（cosine schedule、horizon=总 token 数——**必须逐模型设置，这是 Kaplan 偏差的坑，注释讲清**），打印 (N, D, loss) 表并 feed 给 `fit_chinchilla`，让学生自己得出 N:D≈1:10~1:20。
  `epoch`（数据约束，呼应 Muennighoff 2305.16264）：固定 unique tokens（data/input.txt），训 1/2/4/8/16 epoch，画 loss-epoch 曲线，验收 R≤4 时 loss 接近线性外推、R>4 饱和。
- [ ] **Step 2: 写 00 章**。结构：学习目标（4 条：推导/拟合/解释过训练/设计数据预算）→ 问题引入（"数据洗到多干净、攒到多少才够？"直连 Part 13 主题）→ 数学（Kaplan 2001.08361 幂律 → Chinchilla L(N,D)=E+A/N^α+B/D^β 逐步推导 + 三参数含义 → LR schedule 坑）→ 数据约束（R>4 饱和公式化描述）→ 过训练时代（Llama 3 2407.21783：8B 训 15T ≈1875 t/p vs Chinchilla 最优 ~200B；Besiroglu 2401.00448 inference-aware；参照系表 Llama2-70B 28 → Llama3.1-8B 1875 t/p）→ emergent 之争（Schaeffer 2304.15004 vs Wei 博客反方，衔接 Part 8 评估学指标选择）→ MoE scaling 一段（Krajewski 2402.07871 粒度变量，链接 Part 7 MoE 章）→ 实测贴 Step 1 三模式真实输出 → 3 概念题 + 2 动手（用 scan 模式扩网格重拟合；把 epoch 曲线外推回答"加数据 vs 多epoch 谁划算"）→ 参考资源（Lilian Weng 2026-06 博客、CS336、HF datablations）。
- [ ] **Step 3: 验证与提交**：

```bash
cd courses/Part13_data_engineering/scripts && MPLBACKEND=Agg .../python 00_scaling_laws.py --mode fit && .../python 00_scaling_laws.py --mode scan && .../python 00_scaling_laws.py --mode epoch  # 三模式 rc=0
git add -A courses/Part13_data_engineering && git commit -m "feat(P13): scaling law 开篇章——Chinchilla 拟合/isoFLOP/数据约束三实验（缺口 G）"
```

---

### Task 6: 新建 Part 18 — RAG 全链路（应用线 A1）

**Files:**
- Create: `courses/Part18_rag/scripts/01_minimal_rag.py`、`02_contextual_retrieval.py`、`03_rag_eval.py`
- Create: `courses/Part18_rag/tutorial/README.md`、`01_naive_to_hybrid.md`、`02_advanced_rag.md`
- Create: `assignments/assignment_18/{assignment.md, rag_exercises.py, test_rag_exercises.py}` + `assignment_reference/assignment_18/` 同构
- Modify: 根 `README.md`（路线图/课程表/结构树/作业列表加 Part 18）

**Interfaces:**
- Consumes: Qwen3-Embedding-0.6B（需下载，README 环境节说明 + 脚本缺模型时打印指引优雅退出）、Qwen2.5-0.5B-Instruct（已缓存，生成/判分）、本仓库 `docs/*.md` 作为语料（真实、小规模、免下载）
- Produces: `recursive_chunk(text, size=512, overlap=64)`、`bm25_scores(query, chunks)`、`rrf_fuse(list_a, list_b, k=60)`、`faithfulness(answer, contexts, judge)` —— 作业 18 四道核心题与此同名同签名

- [ ] **Step 1: `01_minimal_rag.py`（~250 行，手写五件套）**：递归分块 → embedding（Qwen3-Embedding-0.6B last-token pooling；**降级路径**：模型未下载时用可复现的 hashing trick 向量并打印警告，保证脚本永不崩）→ 手写 BM25（k=1.2, b=0.75，与 Part 13 LSH 近似检索哲学互文）→ RRF 混合（k=60）→ cross-encoder 重排（bge-reranker-v2-m3，同样降级路径）→ Qwen2.5-0.5B 生成。语料 = 本仓库 docs/ 的 8 个 md。输出：3 个 query 的检索对比表（dense/BM25/hybrid/+rerank 的 recall@5 逐级提升）+ 带引用生成。向量检索用 torch 广播暴力 cosine（注释：不手写 ANN，对照 FAISS 一句话）。
- [ ] **Step 2: `02_contextual_retrieval.py`（Anthropic 复刻）**：对每个 chunk 用 Qwen2.5-0.5B-Instruct 生成 50 token 上下文前缀再嵌入，报告 recall@20 前后变化 + BM25 混合后的进一步变化；对照官方数字（5.7%→3.7%→2.9%，加 rerank 1.9%，即 -67%，来源 anthropic.com/engineering/contextual-retrieval）讨论本机量级差异原因（0.5B 上下文生成质量）。附 late chunking 一段讲解（2409.04701，Jina 博客）。
- [ ] **Step 3: `03_rag_eval.py`**：手写 faithfulness（claim 分解 + judge 逐条 entailment，judge 用 0.5B-Instruct）与 context precision（judge 对 (query, chunk) 打分排序）——然后 `pip install ragas`（可选依赖，缺失时打印指引跳过）对同一批输出打分对比，讨论"评测器噪声"。
- [ ] **Step 4: 教程两章 + README**。01 章：五件套逐行 + 数据流 ASCII + 陷阱（chunk 太碎丢上下文/hybrid 权重要网格搜/ embedding 榜单不可迷信——MTEB 维护性研究 2506.21182 与 RTEB）+ 实测贴 Step 1 输出；02 章（认知章）：演进表 naive→advanced→modular→agentic RAG（综述 2501.09136）+ GraphRAG 2404.16130 / HippoRAG 2 2502.14802 / RAPTOR 思想讲解（只讲+图，不实现，标注选做）+ **"什么时候不该用 RAG"专节**（LaRA 2502.09977 无银弹、Self-Route 成本路由（2407.16833）、<200k token 直接塞 prompt 的 context engineering 共识）+ RAGAS 四指标定义。每章 3 概念题 + 2-3 动手 + 前置三级 + 学习目标，格式对齐 good-tutorial-standard。
- [ ] **Step 5: 作业 18（4+1）**：题 1 `recursive_chunk`（边界：空文本/超长段/overlap 一致性）；题 2 `bm25_scores`（IDF 性质、与 TF 的单调性）；题 3 `rrf_fuse`（k→∞ 退化为计数排序、并列名次处理）；题 4 `faithfulness`（给定 judge mock 下全支持=1.0/全矛盾=0.0）；题 5 🌟`hybrid_weight_sweep`（网格搜 dense:sparse 权重画曲线）。测试查性质不查精确值（除确定性纯函数）；思考题 4 道（含"RAG vs 微调 vs 长上下文怎么选"）；参考实现放 `assignment_reference/assignment_18/`。
- [ ] **Step 6: 验证与提交**：

```bash
for s in courses/Part18_rag/scripts/0*.py; do MPLBACKEND=Agg .venv/bin/python $s || exit 1; done
.venv/bin/python assignments/assignment_18/test_rag_exercises.py            # 骨架优雅失败 + 题5 SKIP
cp 参考实现到 /tmp 镜像跑 test                                                # 5/5
git add -A courses/Part18_rag assignments/assignment_18 assignment_reference/assignment_18 README.md && git commit -m "feat: Part 18 RAG 全链路——手写五件套/contextual retrieval/评测（缺口 D/应用线 A1）"
```

---

### Task 7: 新建 Part 19 — Agent 与 Function Calling（应用线 A2）

**Files:**
- Create: `courses/Part19_agents/scripts/01_agent_loop.py`、`02_mini_mcp.py`、`03_tau_mini.py`
- Create: `courses/Part19_agents/tutorial/README.md`、`01_agent_loop.md`、`02_protocols_and_frameworks.md`
- Create: `assignments/assignment_19/{assignment.md, agent_exercises.py, test_agent_exercises.py}` + `assignment_reference/assignment_19/`
- Modify: 根 `README.md`（Part 19 行 + Part 17 行补"姊妹篇"互链）

**Interfaces:**
- Consumes: Qwen2.5-0.5B-Instruct（已缓存；chat template 自带 tool calling 格式）；Part 17 的轨迹概念、Part 14 的 vLLM 定位（仅文字互链）
- Produces: `run_agent(model, tools, query, max_turns)`、`parse_tool_calls(text)`、`TOOL_SPECS`（OpenAI tools JSON schema）—— 作业 19 核心题同名

- [ ] **Step 1: `01_agent_loop.py`（~180 行）**：`while` 循环 + OpenAI tools JSON schema 三工具（calculator / file_read / bash——bash 加超时 10s + 命令白名单 {ls, cat, grep, python}，**沙箱安全是教学点**）+ `parse_tool_calls` 解析 + 本地执行 + 结果以 `tool role` 回填 + 上下文超长裁剪（保 system + 最近 N 轮）。必须包含：一次失败恢复演示（工具报错 → 模型自纠）与终止条件（无 tool_calls / max_turns / 循环检测同工具连续 3 次）。开篇注释：ReAct 轨迹 = Part 17 的训练数据格式（互链）。
- [ ] **Step 2: `02_mini_mcp.py`（~120 行）**：同文件内实现 toy MCP server（stdin/stdout JSON-RPC 2.0，`initialize`/`tools/list`/`tools/call` 三方法，暴露 echo 与 add 两工具，subprocess 启动）+ mini client 握手并调用——"协议标准化"从名词变可调试代码。A2A 只讲 agent card 概念（一段文字 + 官方 JSON-RPC/SSE 设计链接）；AGENTS.md 提一句（agent↔代码库静态说明，与 MCP/A2A 的三层分工表）。
- [ ] **Step 3: `03_tau_mini.py`（τ-bench 微缩）**：内置政策文档（退换货规则 300 字）+ 用户模拟器（脚本化 10 轮对话）+ 3 工具（查订单/改地址/退款）+ `pass^1` 指标（一次通过率）；跑 Qwen2.5-0.5B 报数字。注明 τ²-bench 2506.07982 与"评分修正改分数、复现必须锁版本"的教训。
- [ ] **Step 4: 教程两章 + README**。01 章：agent loop 逐行 + 轨迹解剖（与 Part 17 同格式）+ 中断/重试/上下文管理三件事 + Echo Trap 安全回顾（bash 白名单）+ 实测贴 Step 1/3 输出；02 章（生态认知）：协议三层（MCP=agent↔工具、A2A=agent↔agent 150+组织、AGENTS.md）+ 框架对照表（LangGraph 状态机/OpenAI Agents SDK/smolagents 千行 CodeAgent/极简派 Pi 四工具——引用 Mario Zechner 设计文）+ **多智能体三方辩论专节**（Anthropic orchestrator-worker 15× token vs Cognition "Don't Build Multi-Agents" vs LangChain 调和框架，结论引导到"上下文隔离才是收益来源"）+ 评测现状（SWE-bench Verified 官方站 Opus 4.5 80.9%；**第三方榜单污染警示**）+ agentic RL 去向（GiGPO 2505.10978 verl-agent、AgentRL 2510.04206，2×4090 可训 1.5B——互链 Part 17）。每章 3 概念题 + 2 动手 + 三级前置 + 学习目标。
- [ ] **Step 5: 作业 19（4+1）**：题 1 `tool_spec`（为给定函数生成合法 JSON schema——性质：required 字段齐全、类型合法）；题 2 `parse_tool_calls`（合法/畸形/多调用/无调用四类输入）；题 3 `should_stop(state)`（终止条件状态机：max_turns/无调用/循环检测）；题 4 `pass_at_1(runs)`（τ-bench 指标，数学性质 E[pass^1] 与多次独立 run 的关系）；题 5 🌟`mini_mcp_call`（JSON-RPC 三步握手 mock）。思考题 4 道（含"subagent 收益来自数量还是上下文隔离"）；参考实现进 `assignment_reference/assignment_19/`。
- [ ] **Step 6: 验证与提交**（同 Task 6 模式，三脚本 rc=0 + 作业双模式）：

```bash
git add -A courses/Part19_agents assignments/assignment_19 assignment_reference/assignment_19 README.md && git commit -m "feat: Part 19 Agent/Function Calling——手写 loop/mini-MCP/τ-bench 微缩（缺口 E/应用线 A2）"
```

---

### Task 8: 横切文档同步（缺口登记翻转）

**Files:**
- Modify: `docs/course_roadmap_v3.md`（§6 缺口表 7 行状态全部翻转：RAG/Agent→已建 Part 18/19；scaling law→已建 P13 00 章；幻觉安全/lm-eval→已扩 P8；YaRN→已扩 P7；FA→已扩 P9；修订记录加 v3.1 行）
- Modify: `docs/llm_interview_guide.md`（§1 映射表簇 D/E/G/H 行更新为对应 Part；§2 总评缺口排序改写；§3 链 6 应用链从"❌ 需自补"改为指向 Part 18/19）
- Modify: 根 `README.md` 学习路线（若 Task 6/7 未覆盖的收尾：A1/A2 标注）、`requirements.txt`（可选依赖：`lm_eval[hf]`、`ragas`（RAG 评测对比用，可选）、模型下载说明注释）
- Modify: `assignment_reference/README.md`（登记 18/19）

- [ ] **Step 1: 逐一修改上述文件**（每处引用真实存在的文件路径，改完 `grep -rn "待开\|待定\|🟡" docs/course_roadmap_v3.md` 确认 §6 无残留待办，多模态"归档"行保留）。
- [ ] **Step 2: 提交**：

```bash
git add -A docs README.md requirements.txt assignment_reference/README.md && git commit -m "docs: 缺口登记翻转 v3.1——7 缺口全部落地，应用线 A1/A2 建成"
```

---

### Task 9: 质量门禁 — 全量回归 + tutorial-creator 复评

**Files:**
- 无新文件（验证轮；发现问题则回到对应 Task 修补）

- [ ] **Step 1: 全量回归**：跑既有 18 个脚本 + Part 7/8/9/13/18/19 全部新脚本 + 双卡 torchrun×4 + Part 9 make + 作业 9-19 骨架/参考双模式（复用本轮建立的跑批脚本模式，修掉 `*_exercises.py` glob 会匹配 `test_*_exercises.py` 的坑：`ls | grep -v test_`）。全部 rc 符合预期（脚本 0 / 骨架 1 / 参考 0）。
- [ ] **Step 2: 独立复评**：派 3 个只读评审代理（分组 7+9 / 8+13 / 18+19）按 `good-tutorial-standard.md` 打分——**新增/扩展的每个 Part 必须 ≥4.5 且血肉 ≥4.5**；发现的 P0/P1 当轮修复后复跑受影响验证。
- [ ] **Step 3: 收尾提交**：

```bash
git add -A && git commit -m "fix: 缺口补全质量门禁——复评问题收口，全部新增内容 ≥4.5"
```

之后向用户确认是否 push。

---

## Self-Review（已执行）

1. **覆盖度**：缺口登记 7 项 ↔ Task 1(YaRN)/2(FA)/3(幻觉)/4(lm-eval)/5(scaling)/6(RAG)/7(Agent)，全部映射；文档翻转与质量门禁为 Task 8/9。多模态（已归档）无需任务。✔
2. **占位符扫描**：所有文件路径均已 `ls` 核实真实存在（Part7 `02_modern_components.md:255`、Part8 `07_evaluation.md`、脚本编号接续 11/12/13 均无冲突）；关键代码（YaRN 三部件、online softmax、BM25/RRF 签名、Chinchilla 公式、YAML）已给出实体；`/path/to/.venv` 仅出现在示例命令中且 Task 0 无——全局约束已写明绝对路径。✔
3. **类型一致性**：`rrf_fuse(list_a, list_b, k=60)`、`parse_tool_calls(text)`、`chinchilla_loss/fit_chinchilla` 等跨任务引用的签名在各 Task 的 Produces/Consumes 中一致。✔
4. **风险预案**：新模型下载失败 → 降级路径已定义（Task 6 hashing trick / Task 3 🌟 段跳过）；`lm_eval`/`ragas` 未装 → 优雅退出；2×4090 跑不动真实 128K 评测 → 迷你 RULER 认知层替代（Task 1 Step 3/4）。✔
