# Part 11-14 教程设计调研与规划（拟开章节）

> **生成**：2026-08-30。基于 4 路仓库深调（train-llm-from-scratch 时间线 / rasbt 全书结构 /
> verl+LlamaFactory / Data-Juicer+vLLM）+ GLM-5.3、DeepSeek-V4 SOTA 检索。
> 回答三个问题：Part 8 主源的时间线评价；rasbt 作为续章的评估；Part 11-14 各章的
> "锚定仓库 → 链路定位 → 教程设计（手写 vs 工具对比）→ 环境可行性"。

---

## 0. 版本锁定策略（按审阅意见修正）

**原则：锁 LTS 或 latest，不用任意旧版本**——latest 有新特性、是当下最流行，锁它才有时效价值；
没有 LTS 的项目就锁"最新 release tag"并记录日期。

| 仓库 | 策略 | 本课程环境的具体落点 |
|---|---|---|
| vLLM | latest（0.28.0，pin torch 2.13）→ **独立 venv** | ⚠️ latest 与课程 venv（torch 2.5.1+cu121）不兼容；备选 `vllm==0.6.6` 恰好 pin torch==2.5.1+cu121，**可直接装进课程 venv**（二选一，教程两案都给） |
| verl | latest release tag + Docker（官方镜像 `verlai/verl`） | 版本耦合极重（vllm/torch/transformers 锁步），Docker 是唯一低摩擦路径 |
| LlamaFactory | latest（`pip install -e .`） | 耦合轻（peft/trl/transformers 三件套），跟随 latest 即可 |
| Data-Juicer | latest（`pip install py-data-juicer`） | 文本管线 CPU 可跑，重依赖均为可选 extras |
| PyTorch 本体 | 课程 venv 2.5.1+cu121 保持不动 | 各新章用独立 venv，课程主环境不升级 |

---

## 1. Part 8 主源（train-llm-from-scratch）时间线评价

**回答"内容是否详实、按照时间线发展"：结构上详实且强连贯，更新史上是"两次成型"而非持续跟踪。**

- **内容结构（详实 ✅、时间线 ✅）**：README 按步骤组织——数据准备（4 路数据流）→ 逐块搭模型 →
  预训练 → 生成 → 后训练（SFT→RM→DPO/ORPO/KTO→PPO→GRPO/RLVR）→ 评估 → Chat/UI。
  每阶段**加载上一阶段 checkpoint**（"wrap, do not rewrite"），全程用同一个指标（GSM8K greedy
  准确率）把 Base/SFT/DPO/PPO/GRPO 串成一张表——这是教科书式的"时间线发展"。
- **规模与细节**：13M 演示 / 77M 真预训练（2×L40，2000 步，loss 11.14→3.76）/ 406M 后训练默认；
  公式给到直觉级代码（BT、DPO sigmoid、PPO clip+GAE、GRPO 组归一、k3 KL），非论文级推导。
- **更新史（核实 commit log）**：2025-01 建仓（nanoGPT 式预训练教程）→ 2025-05 加 notebook →
  **9.5 个月停更** → **2026-06-04 一次性 squash 提交全部现代化内容**（pretrain_base、
  SFT/RM/DPO/PPO/GRPO 全套脚本、评估 harness、Streamlit UI、MkDocs——GRPO 在 R1 带
  火它约 17 个月后才补上）→ 6 月中 AMP/梯度检查点/README 重写/真实 MHA bug 修复 →
  8 月仅外观性提交。
- **结论**：不是"冻结的 2024 遗物"，但也**不是 SOTA 跟踪者**。主管线刻意用 2020-2022 经典
  架构（MHA+learned PE；RoPE/GQA 只在 2025 的旁支 notebook），无 flash attention。
  **对本课程的含义**：我们 Part 8 已补齐 GRPO/量化/投机解码/评估学，差距主要剩
  "主源无现代架构"——这正是 Part 7（现代组件）+ Part 8（生命周期）互补设计的合理性证明。
  **行动**：Part 8 README 时效声明按此措辞（"主源 2026-06 现代化后含 GRPO/RLVR；
  架构为教学经典款，现代组件见 Part 7"）。

## 2. rasbt/LLMs-from-scratch 评估：难度与衔接

**难度（对我们培养出的学生）：主线 2.5/10，含 bonus 与续作约 4/10。** 递阶为"更慢更温和的
重走"，不是挑战，而是**查漏与巩固**。

- **结构**：ch01-07（与 Manning 书 1:1）+ 附录 A-E + 大量 bonus 文件夹；**续作**
  `rasbt/reasoning-from-scratch`（5.1k，独立书）才是 RL/蒸馏所在：推理时扩展、
  **RLVR-GRPO from scratch（ch06）**、进阶 GRPO 变体（DeepSeek-V3.2/Olmo3 风格，ch07）、
  **蒸馏（ch08）**、MATH-500/LLM-as-judge 评测。维护极活跃（最后推送 2026-08-29，Apache-2.0）。
- **对已修完本课程 Part 1-10 的学生**：ch01-05 ≈90% 复习（更小规模的 GPT-2 重推）；
  **真有新东西的**：ch06 文本分类微调（从零课少见）、ch07 指令数据工程（JSON 格式规约、
  近重复检测、被动语态改写）+ LLM-as-judge 评测 harness、ch04 的 KV-cache→GQA→MLA→SWA→MoE
  显存估算实验、ch05 GPT→Llama 权重手术与省显存加载、**附录 E 从零写 LoRA**、续作的
  GRPO 变体与蒸馏。
- **结论**：✅ **可以作为"延伸与拓展"（巩固补漏 + 资料库），❌ 不适合作为主线的"下一章"**
  ——主线下一章应是 Part 11 对齐实战（verl）。rasbt 的正确用法：① ch06/AppE/ch07 选学
  （正好填本课程"分类微调/指令数据工程/LoRA 从零"三个小缺口）；② **续作
  reasoning-from-scratch 与 Part 11 互为对照**（我们用工业框架 verl，他用裸 PyTorch 讲透
  同一批算法——一工程一原理，双视角）。

## 3. Part 11-14 逐章设计

### Part 11：对齐实战（锚定 verl；辅助 slime / reasoning-from-scratch）

- **链路定位**：RL 后训练基础设施（RLHF/RLVR），HybridFlow 论文（EuroSys'25）；
  生产用户：DAPO、Seed-Thinking、Doubao；**GLM 系列自 GLM-4.5 起的 slime（8.3k，活跃）
  是同赛道参照**——两框架都证明"后训练 Scaling"是当前旗舰模型的主要增益来源。
- **仓库内容**：混合控制器编程模型（RL 数据流与计算解耦）、FSDP2/Megatron 训练后端 +
  vLLM/SGLang rollout 后端、Ray 编排、算法全家桶（PPO/GRPO/GSPO/DAPO/PRIME…）、
  quickstart=**Qwen2.5-0.5B 在 GSM8K 上单卡 PPO（文档明确 ≥24GB）**。
- **环境可行性**：✅ 单卡 4090 可跑官方 quickstart（micro-batch=1 防 OOM）；双 4090 可做
  FSDP 分片扩展。⚠️ 安装摩擦高 → **Docker 镜像 `verlai/verl` + 锁 release tag**。
- **教程设计（我们模式：教程+脚本+作业）**：
  1. 概念桥接（不写代码）：把 Part 8 手写的 PPO/GRPO 逐概念映射到 verl（rollout 循环+
     权重回同步 HybridEngine、组内优势、KL-to-ref、规则奖励函数、Ray 单控制器数据流）；
  2. 脚本：Docker 内跑 quickstart（0.5B PPO@GSM8K）→ 改 `adv_estimator=grpo` → 自定义
     奖励函数（正则抽取 GSM8K 答案，呼应 Part 8 07 章评估学）→ 双卡 FSDP；
  3. 作业：把 slime 的 README/quickstart 与 verl 做一张概念对照表（面试即用）。
- **手写 vs 工具对比表**：手写 rollout+同步 vs HybridEngine 的权重同步；手写组内标准化 vs
  `compute_advantage`；手写 KL 惩罚 vs ref-policy 机制——5 行代码对应表进教程。

### Part 12：微调实战（锚定 LLaMA-Factory；辅助 unsloth）

- **链路定位**：统一微调工具（SFT/LoRA/QLoRA/DPO/KTO/RM 全阶段覆盖）——训练侧"最后一公里"
  的工业标准入口（Amazon/NVIDIA/阿里云在用）。
- **环境可行性**：✅ 最佳——官方文档 **QLoRA 7B 仅需 6GB 显存**（LoRA bf16 7B=16GB 也装下），
  4090 绰绰有余；安装摩擦低（peft/trl 三件套），首次出结果小时级。
- **教程设计**（先 LlamaFactory 后 verl 的顺序由两仓库特性决定：一小时出结果 vs 半天搭环境）：
  1. `identity` 小模型（0.5B）单 yaml LoRA SFT → LLaMA Board WebUI 建立直觉；
  2. QLoRA 7B 全流程：yaml → 训练 → `export` 合并 → `chat`/`api`（OpenAI 兼容）；
  3. DPO-with-LoRA（UltraFeedback 子集，呼应 Part 8 03 章）；
  4. **手写 vs 工具对比**（核心教学设计）：学生先用 peft 从零写 5 件事——LoRA 层注入
     （target_modules/r/alpha）、chat template 构造+prompt masking（呼应 Part 8 02 章）、
     SFTTrainer 的 packing vs padding、NF4 量化+QLoRA、DPO 的 LoRA 参考策略——然后
     同一数据集用 LlamaFactory yaml 复现，对比（模板正确性/packing/显存）。
  5. 作业：unsloth（75.2k）同一任务加速对比（官方免费 Colab notebook 丰富）。

### Part 13：预训练数据工程（锚定 Data-Juicer）

- **链路定位**：预训练/SFT 数据清洗（上游最上游）——RedPajama/BLOOM 管线的开源复现，
  "好语料和网页垃圾的差距就在这"；**正是 roadmap T8 与面试簇 G 的 ❌ 缺口**。
- **环境可行性**：✅ **CPU 即可**（"laptop 到千节点集群"），文本管线轻依赖（重 extras 按需装），
  最小 demo = 3 条样本过 1 filter + 1 mapper。
- **教程设计**：
  1. **手写先行**（~100 行）：shingling → MinHash 签名 → 分带 LSH → Jaccard 验证
     （datasketch，100 篇小文档）——呼应 Part 9"手写内核"与本课程从零哲学；
  2. 同一管线换 Data-Juicer YAML（`document_minhash_deduplicator` + `words_num_filter`...）
     复跑，对比 5 个被自动化的点：分词/shingling（Cython/C++）、签名计算、LSH 分桶、
     去重簇消解、**YAML 可复现性 + 逐 op 追踪审计**；
  3. 彩蛋教学：2026-08-28 最新提交恰是 `document_minhash_deduplicator` 的 bugfix
     （空 token 样本）——"读一个真实 PR"练习；
  4. 延伸：FineWeb 去重官方教程（Anyscale）、Gopher/C4 启发式规则（呼应指南 §1 簇 G）。

### Part 14：推理部署实战（锚定 vLLM）

- **链路定位**：训练完成后的**服务阶段**（PagedAttention/连续批处理/prefix caching/
  CUDA graphs）——本课程 Part 8 06 章"手写模拟"的工业对应体。
- **环境可行性**：✅ 4090（sm89）官方支持；两条安装路径按 §0 策略：**首选独立 venv 装
  latest（0.28.0，torch 2.13）**；备选 `vllm==0.6.6`（恰好 pin torch==2.5.1+cu121，可复用
  课程 venv，注明该版本有 vllm-flash-attn 解析冲突的已知 issue #11283）。
- **教程设计**（对比表直接可用）：

  | 手写（本课程已做） | vLLM 对应 | 实测指标 |
  |---|---|---|
  | KV cache 字典（Part 7 03） | PagedAttention 块表 | 固定 batch 下 KV 显存 |
  | 顺序 generate 循环 | 连续批处理 | 聚合 tok/s 吞吐 |
  | 每请求冷启动 | prefix caching | 共享前缀 prompt 的 TTFT |
  | 手写 INT8/INT4（Part 8 09） | GPTQ/AWQ/FP8 加载 | 磁盘/显存占用 |
  | 贪心多 token 猜测 | n-gram 投机解码（无需 draft 模型） | tok/s + 接受率 |

  脚本：离线 `LLM.generate`（Qwen2.5-0.5B）→ `vllm serve` + OpenAI client →
  `benchmarks/benchmark_serving.py`（sonnet.txt）出 TTFT/TPOT 分位数 → 量化服务 →
  n-gram 投机解码。作业：把 Part 8 09 的手写投机解码与 vLLM 内置的接受率对比。

## 4. SOTA 情报（GLM-5.3 / DeepSeek-V4）与课程映射

### 关键事实（2026-08 检索）

- **GLM-5.3**（2026-08-14 发布）：**基座与 GLM-5.2 完全相同，全部提升来自后训练**——
  "后训练之王"路线：小规模（十张卡）验证 RL 方法 → 工程系统放大到旗舰模型与超长训练周期
  （后训练 Scaling）。RL 基建是开源的 **slime**（THUDM/slime，8.3k，活跃）。
- **GLM-5.3-Flash**：GLM-5 系**首个原生多模态**（320B/激活 18B），**首个稀疏注意力+线性
  注意力混合架构**的开源前沿模型（长上下文服务成本大降），国产芯片大规模验证。
- **DeepSeek-V4**（2026-04-24 技术报告）：V4-Pro 1.6T/49B、V4-Flash 284B/13B，原生 1M 上下文；
  **CSA/HCA 混合注意力**（1M 场景下 FLOPs 降至 V3.2 的 27%、KV 缓存 10%）；32-33T token 语料；
  与 AMD MI400 软硬协同。
- **DeepSeek-V4 后训练范式细节（方法学级替换，与课程直接相关）**：
  ① **放弃 V3.2 的统一 Mixed RL**，改为两阶段——**专家训练**（数学/代码/Agent/指令跟随
  各训独立专家模型，每个都是 SFT → **GRPO**）→ **同策略蒸馏 OPD**（On-Policy Distillation，
  用**全词表 KL** 把各专家能力统一进大一统模型，取代原 Mixed RL）。GRPO 没有被抛弃，
  而是下沉到专家层——我们 Part 8 手写的 GRPO + Part 11 的 verl 恰好就是这条管线的技能栈；
  ② **V4-Pro 进一步**：rubric-guided RL 数据 + **GRM（生成式奖励模型）本身被 RL 直接优化**
  ——这是 Part 8 03 章 Bradley-Terry 判别式 RM 的下一代形态（面试高价值谈资）；
  ③ OPD 的"全词表 KL"与 Part 8 04 章手写的 k3 KL 惩罚、rasbt 续作 ch08 蒸馏同属一个知识族：
  **蒸馏首次进入主线视野** → 在 Part 11 教程加一节"OPD 是什么"（无需实操，讲清
  "专家 RL + 在策略蒸馏统一"的双层结构即可）。

### 对课程的三个直接映射

1. **后训练 Scaling = Part 8/11 的时代注脚**（最该写进教程的事实）：两个旗舰系列的主要增益
   都来自后训练（GLM-5.3 基座不变；DeepSeek-V4 重写后训练范式）——我们课程 Part 8（手写
   全流程）+ Part 11（工业框架）恰好是这条路线的完整技能栈。Part 8 README/07 章与
   Part 11 教程开头各加一段"为什么后训练是当下主战场"（引 GLM-5.3/DeepSeek-V4 为例）。
2. **slime 进 Part 11 的对照表**：verl vs slime（同为 RL Scaling 框架；slime 有"10 卡验证再
   放大"的方法论叙事，对学习者特别友好）——作业题做概念对照。
2b. **OPD/GRM 进 Part 11 教程与面试素材**：DeepSeek-V4 的"专家 GRPO → OPD 蒸馏统一"双层
   结构、"生成式奖励模型被 RL 优化"两个点，是 Part 8 手写内容与 2026 旗舰实践之间的
   最短桥梁（详见上方 DeepSeek-V4 细节①②③）。
3. **多模态应该涵盖，但保持在 Part 15**：GLM-5.3-Flash 的"原生多模态"和 DeepSeek-V4 的
   通用性证明多模态已是旗舰标配；但 VLM 训练（vision encoder + projector + LLM 联调）是
   独立的大块头，且 verl 已支持 VLM RL（Part 11 可顺手提一句）。**维持 Part 15（多模态 VLM）
   的规划，优先级不变**。稀疏/线性混合注意力 → 记入 Part 7/9 的"延伸阅读"一句
   （GLM-5.3-Flash 与 DeepSeek-V4 的 CSA/HCA 是 Part 7 RoPE/GQA 之后注意力演进的下一代方向）。
4. **Infra 是否有关？——有，且越来越重**：GLM-5.3-Flash 为国产芯片做架构适配、DeepSeek-V4
   与 AMD MI400 协同设计、verl 的 3D-HybridEngine 为"训练↔rollout 转换"省显存——
   说明"模型-硬件协同"已进入主流叙事，Part 9（CUDA）+ Part 10（分布式）的定位被 SOTA
   佐证，无需调整。

## 5. 汇总时间线（建议实施顺序）

```
Part 12 LlamaFactory（1-2 天，最快见效，QLoRA 7B@4090）
→ Part 11 verl（3-4 天，Docker + 0.5B GRPO）
→ Part 13 Data-Juicer（1-2 天，纯 CPU）
→ Part 14 vLLM（2-3 天，独立 venv + benchmark）
（每章都按"教程+脚本+作业 + 手写 vs 工具对比表"交付；rasbt 主书作巩固资料、
  reasoning-from-scratch 与 Part 11 双视角并行）
```
