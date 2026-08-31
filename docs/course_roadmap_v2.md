# 课程路线图 v2 —— 内容优化任务清单

> **生成**：2026-08-30（基于 4 份并行研究简报：分布式训练 / 推理服务 / 长上下文+数据+评估 / minimind 复现细节）
> **流程**：研究 → 本清单 → 学生 agent 评审 → 教师（主 agent）按序实施
> **上一版**：`part7_part8_optimization_plan.md`（其 P0 项并入本清单 T1-T4）
> **后继**：[course_roadmap_v3.md](course_roadmap_v3.md) —— **学习者路线图**（每节点
> 「学习内容安排 / 学习目标设定 / 学习验证」三段式）。本清单使命完成后转为建设任务存档；
> 学习规划请以 v3 为准。

## 研究关键发现（影响任务设计）

1. **minimind 已更新，旧规划过时**：训练脚本在 `trainer/` 目录（非根目录）；数据集改名为
   `pretrain_t2t_mini.jsonl`(1.2GB)/`sft_t2t_mini.jsonl`(1.6GB)/`dpo.jsonl`(53MB)，
   旧名 pretrain_hq/sft_512 已废弃；**无 warmup 参数**（cosine 封装 1.0×→0.1×lr）；
   DPO 默认 lr=4e-8、beta=0.15；26M 配置=hidden 512/8 层/8 头/kv 头 2/vocab 6400/rope 1e6；
   官方实测 3090 单卡 dense 全流程 ≈2.3h/≈¥3。
2. **karpathy/minGPT-ddp 原 404**，官方维护版在 `pytorch/examples/distributed/minGPT-ddp`——Part 10 引用它而非原链接。
3. **FSDP1 已弃用**（PyTorch 2.13），教学应教 `fully_shard`（FSDP2）API。
4. 推理侧关键数字可直接进教程：PagedAttention 解决 60-80% KV 浪费→<4%；Orca 连续批处理 36.9×；
   int8 RTN ppl 几乎无损、int4 g128 +0.3~0.6；投机解码 E[tokens]=(1-α^(γ+1))/(1-α)；KIVI：K 按 channel、V 按 token 量化。
5. 长上下文三件套公式：PI（m→m/s）、NTK（θ'=θ·s^(dim/(dim-2))）、YaRN（ramp+温度 τ=0.1ln(s)+1，400 步到 128k）。
6. 评估学硬证据：GSM1k（arXiv 2405.00332）显示 Mistral −8%/Phi −21% 过拟合——污染不是理论问题。
7. MoE aux loss：L=α·N·Σf_i·P_i，α≈0.01（Switch）；minimind 用 5e-4；DeepSeek-V3 已用无 aux loss 的 bias 法。

---

## 任务清单（按优先级）

> **实施状态（2026-08-30）**：T1-T7 已全部实施并验证（Part 7 新增 05 章+脚本 09/10、
> **Part 11-14 已完整构建**（教程/脚本/作业，"手写→工具"双轨）；全课程脚本审计完成
> （Part 1-14 共 60+ 脚本可运行，修复 Part 4 三处 bug：03 cmp 对非叶子 grad 无护栏、
> 05 缺 retain_grad、06 需 --quick；Part 8 README 时效声明已加）；作业参考答案落位
> [assignment_reference/](../assignment_reference/README.md)（14 套全部实测通过，含
> 4 处原作业测试 bug 修复：a2 初始化尺度 / a3 loss 边界 / a4 签名与 retain_grad / 阈值）；
> Part 8 新增配置表+脚本 09+06/07 章、Part 10 全新上线 6 脚本+4 章+assignment_10）。
> 未实施：T8-T10（见 P2）。

### 🔴 P0 —— 复现断层修补（学生跑完课程≠能复现原仓库）

#### T1 Part 7「复现 minimind 毕业指南」 `courses/Part7_minimind/tutorial/05_reproduce_minimind.md`
- 对照表：课程脚本 ↔ `trainer/` 四脚本（按研究成果修正路径与默认超参）
- 数据：ModelScope 三个 t2t_mini 文件 + jsonl 行格式示例
- 配置放大表：课程缩小版 → 26M(hidden 512) → 64M(hidden 768) → MoE 198M
- 成本预期：3090 实测时长表；`eval_llm.py` + lm-eval-harness 验收
- 进阶实验小节：YaRN/NTK RoPE scaling（PI/NTK/YaRN 公式与数字，呼应 minimind 的 `inference_rope_scaling`）

#### T2 Part 7「三阶段验收」 `scripts/09_eval_demo.py`
- 加载 `temp/ckpt_pretrain.pt` / `ckpt_sft.pt` / `ckpt_dpo.pt`（缺哪个就现场快速训一个缩小版，保持自包含）
- 同一批 prompt 三阶段生成对比 + held-out ppl 曲线 —— 补齐"学完有个交付物"

#### T3 Part 7 MoE 负载均衡实验 `scripts/10_moe_load_balance.py`
- 实现 Switch aux loss（α·N·Σf_i·P_i），对比 α=0 / α=0.01 / minimind 的 5e-4 下专家负载分布（基尼系数/最大负载比）
- 呼应 Part 7 脚本 04（MoE 只讲了概念，现在有实验）

#### T4 Part 8「原版规模/数据模式」对齐
- `02_pretrain.py` 增加 GPU-original 配置档（对照原仓库配置，实施时先核实该仓库 README 的实际超参，不编造）
- README/01 章加配置对照表（cpu-toy / gpu-course / gpu-original）

### 🟠 P1 —— 课程空白补齐

#### T5 Part 8 推理与服务扩章
- 新脚本 `scripts/09_quantize_and_serve.py`：int8/int4 量化 ppl 对比、KV cache 显存计算器、PagedAttention 块表模拟、连续批处理 vs 静态批处理模拟、投机解码机制 demo（验收率 α 与 (1-α^(γ+1))/(1-α) 对照）
- 新教程 `tutorial/06_inference_and_serving.md`：量化（RTN/GPTQ/AWQ 思想）、KV 量化（KIVI：K 按 channel、V 按 token）、PagedAttention/连续批处理、投机解码、部署指标（TTFT/TPOT/goodput/GGUF 命名）、**上线前评估学**（lm-eval-harness、HELM、污染 GSM1k 证据、人工 vs 规则 vs LLM-judge）
- Part 8 README/05 章挂接

#### T6 Part 10 分布式训练（全新 Part，来源 pytorch/examples minGPT-ddp + PyTorch 官方 DDP/FSDP 系列 + nanotron/Ultra-Scale Playbook）
- 脚本（双 4090 实机 torchrun 验证，同时兼容单进程/CPU）：
  1. `01_distributed_basics.py` collectives 广播/归约/all-reduce/all-gather + 进程组
  2. `02_ddp_gpt.py` 标准 DDP 训练 mini GPT（DistributedSampler、set_epoch、no_sync 梯度累积、1 vs 2 GPU 吞吐对比）
  3. `03_zero_memory.py` ZeRO 1/2/3 显存公式 + 优化器状态分片模拟（分 rank 记账）
  4. `04_fsdp_gpt.py` FSDP2 `fully_shard` 实战 + 显存对比
  5. `05_tensor_parallel.py` 手写 Megatron 式列并行/行并行 MLP（f/g 算子，all-reduce 验证等价于单卡）
  6. `06_pipeline_parallel.py` GPipe/1F1B 微批流水线（bubble 公式实测）
- 教程 4 章：并行taxonomy与通信原语 / DDP 深入（桶、重叠、坑）/ 显存与 ZeRO/FSDP / TP/PP 与 3D 并行 + 工业 stack（nanotron/Ultra-Scale Playbook、LLaMA 预训练配置）
- 作业 assignment_10（题 1-4 纯 CPU：all-reduce 均值语义、显存账本计算器、DistributedSampler 模拟、TP 分块数学；🌟 流水线 bubble 计算）

#### T7 评估学（并入 T5 的 06 章，不单独成章）
- 依据研究：lm-eval-harness 两类请求（loglikelihood/generate_until）、HELM 七指标、
  GSM1k 过拟合证据、13/20-gram 去污染惯例、Arena Elo vs LLM-judge 偏差（85% vs 81%，位置/长度偏差）、ppl 跨 tokenizer 不可比

### 🟡 P2 —— 规划归档（本次不实施）

#### T8 数据工程（指南主线 Part 13，或并入 Part 10 04 章）
- FineWeb 流水线、Gopher/C4 启发式规则数字、MinHash+LSH（5-gram、14 band×8 row、阈值≈0.7）、FineWeb-Edu 分类器
- 纯 Python MinHash dedup 小 demo 是可行落地形态
#### T9 多模态（CLIP/ViT→LLM）
#### T10 全部 Part 补「面试直通车」小节 + hints.md 分级提示（沿 part7_part8_optimization_plan P1-5）

## 实施顺序

```
T1 → T2 → T3 → T4（半天量级，先补复现断层）
→ T5（Part 8 扩章）
→ T6（Part 10，工作量最大）
→ 全量验证 + 导航/README 更新
（T7 随 T5 顺带完成；T8-T10 归档 roadmap）
```

---

## 学生 agent 评审结论（2026-08-30）与采纳决定

学生评分：T5=10（最大盲区）> T1=T2=T10=9 > T3=8 = T6=8 > T7=7 > T4=6 = T8=6 > T9=4。

| 学生意见 | 采纳 |
|---|---|
| T5 必须含 vLLM 最小实操（10 行启动 + 概念映射）与环境自检 | ✅ 并入 T5 06 章 |
| T7 评估学别塞 06 章，拆成独立 07 章 | ✅ 改为 `tutorial/07_evaluation.md` |
| T10 面试直通车提前：新章从第一天就带"面试怎么问" | ✅ Part 7 新章 / Part 8 06·07 章 / Part 10 各章直接内嵌，T10 降为增量惯例 |
| T6 标记 TP/PP（脚本 5-6）为进阶可选；补"分布式 Hello World 心智模型 + 常见报错 FAQ"；CPU/单进程兼容写成硬验收 + 无多卡学习路径表 | ✅ |
| T2 给出预期输出样例（pretrain 流利但离题 / SFT 格式正确 / DPO 偏 chosen） | ✅ 写进 09 脚本输出与指南 |
| T3 写死运行预算与指标定义（基尼系数公式） | ✅ |
| T1 补下载断点续传/镜像/租卡成本现实路径 | ✅ |
| T4 降级：一张"配置对照 + 因果"表（参数↑10×时 lr/batch/累积怎么变），不做大任务 | ✅ 并入 Part 8 README |
| T9 多模态移除出路线图 | ✅ 归档为"暂不做" |
| T6 验收硬标准：TP 与单卡误差 <1e-5、bubble 实测对上公式、ZeRO 记账可复算 | ✅ 写进脚本与教程 |

**最终实施序**：T1 → T2 → T3 → T4(轻量) → T5(+T7) → T6 → 全量验证。

> **2026-08-30 第二轮回顾修复**（Part 7/8 不足逐项验证后处理）：
> 已修——P7：03 章 MHA→GQA diff 讲解、09 迷你 DPO 改用"错配上下文"偏好对（ppl 14.2→12.2）、
> 01 章 BPE 三实现对照（+Part 8 反向链接）、新脚本 11_rope_scaling.py（naive/PI/NTK 外推实测）、
> 05 章验证边界声明、assignment_7 实验题×3；P8：03_sft 真实数据开关（--original-data，优雅回退）、
> 09 训练步数 200→500（int8 Δ +0.78→+0.38）、PPO/GRPO 顶部导读、assignment_8 增 06/07 章实验题、
> 406M 单卡可行性备注修正（原"需 24GB+"过于保守）。
> 证伪撤销——"GSM8K 答案抽取脆弱"（实为标准做法）。
> 遗留——Part 9 profiling 实操、vLLM 本机实跑、LoRA 实战（主线 Part 12）、对齐实战（主线 Part 11）；RAG/Agent 移入应用线（见指南 v2 修订）。
>
> **多模态双章已建（2026-08-31）**：Part 15 多模态理解（拼接式 VLM 手写 + 三大方案 +
> CLIP/SigLIP）与 Part 16 图像/视频生成（手写 DDPM + diffusers 工具链 + IP-Adapter 对齐
> + CogVideoX/Wan2.1）——设计依据 [part15_16_multimodal_design.md](part15_16_multimodal_design.md)，
> 跨模态特征对齐主线贯穿理解/生成两侧。
>
> 论文阅读训练：[paper_reading_guide.md](paper_reading_guide.md)（每 Part 一篇代表论文：
> 快读路径 / 公式推理五步法 / 最小复现闭环；配套 tools/verify_paper_formulas.py）。
>
> 参考来源审查：[references_by_part.md](references_by_part.md)（逐 Part 仓库/教程清单，star 与最后推送时间实查）
>
> 拟开 Part 11-14 调研与教程设计：[part11_14_tutorial_design.md](part11_14_tutorial_design.md)
> （各章锚定仓库→链路定位→手写 vs 工具对比→环境可行性；含 GLM-5.3 / DeepSeek-V4 SOTA 映射）
>
> 延伸阅读：[llm_interview_guide.md](llm_interview_guide.md) —— 岗位需求 × 课程映射（✅/🟡/❌）、算法/应用双赛道区分与拟开课程建议（含仓库 star 实查；v2 已修正 RAG/Agent 的赛道归属与框架选型）。

## 四、继续深入学习拓展（2026 前沿地图，2026-08-31 v3 增补）

> 依据 2026-08 岗位深挖（字节 Agent 算法/腾讯混元 RM/B 站长程 Agentic RL/NIO 后训练等
> 2026 在招 JD）与前沿趋势检索。学习法遵循 [paper_reading_guide.md](paper_reading_guide.md)
> 的"最小复现闭环"：每方向先跑通官方 quickstart/玩具复现，再决定深挖深度。

| 方向 | 为什么是当下热点 | 必读论文/报告 | 动手仓库 |
|---|---|---|---|
| **Agentic RL / 长程智能体 RL** | 字节/B站/滴滴/NIO 2026 JD 原文关键词（训练环境+轨迹数据+评测三件套） | AgentGym、AgentBench | verl-agent、Agent Lightning（微软）、Part 11 verl 基础直接续接 |
| **推理模型 / test-time compute** | R1 后全部旗舰跟进；JD 出现"推理模型背景"；s1/test-time scaling | DeepSeek-R1 (2501.12948)、TTS 综述 (2503.24235)、s1 (2501.19393) | rasbt/reasoning-from-scratch（与 Part 11 双视角） |
| **长上下文与高效注意力** | DeepSeek-V4 CSA/HCA、GLM-5.3-Flash 稀疏+线性混合、NSA 获 ACL'25 最佳论文 | NSA (2502.11089)、MLA (2405.04434)、MoBA | NSA 官方实现（知乎"手撕 NSA"有逐行解析） |
| **FP8/低精度训练与推理** | DeepSeek-V3 FP8 训练已验证；推理侧 FP8/W4A16 成标配 | DeepSeek-V3 报告 §3 | Transformer Engine、vLLM 量化路径（Part 14 02 章） |
| **MCP / tool-use 生态** | Anthropic MCP 成为事实标准；Agent JD 高频 | MCP 规范、function calling 文档 | 自写最小 MCP server（呼应应用线 A2） |
| **生成侧进阶**（Part 16 续） | SD3/FLUX 的 rectified flow 已成图像主流；视频生成开源爆发 | SD3/RF (2403.03206)、CogVideoX (2408.06072)、Wan2.1 (2503.20314) | diffusers 全谱系（Part 16 02 章已起步） |
| **评测体系建设** | 应用岗 JD"评测体系"出现率飙升；生成/agentic 评测是空白 | HELM、lm-eval-harness、AgentBench | Part 8 07 章 §5 方法论 + lm-eval-harness 实操 |

> 与已建章节的接口：Part 11（verl）→ Agentic RL；Part 7/9（RoPE/CUDA）→ 高效注意力
> 与 MLA/NSA；Part 14（vLLM/投机解码）→ 低精度与推理优化；Part 15/16 → 原生多模态与
> any-to-any。**每条线的"第一周行动"都是跑通一个官方 quickstart 或玩具复现。**
