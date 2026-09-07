# Part 8: 后训练全流程
## 01. GPT-2 与预训练
- 经典 GPT-2 架构（LayerNorm + learned PE + MHA + ReLU）
- 预训练流水线
## 02. SFT 与 Chat Template
- 监督微调
- Chat Template
- Prompt Masking
## 03. 奖励模型与对齐算法
- Bradley-Terry 奖励模型
- DPO/ORPO/KTO 三种对齐
## 04. 强化学习：PPO 与 GRPO
- PPO（GAE + Clipped Surrogate）
- GRPO（Critic-Free RL）
## 05. 评估与推理部署
- GSM8K 评估
- 全阶段对比
- 交互式 Chat
## 06. 推理与服务
- 量化 int8/int4（GPTQ/AWQ）
- KV 显存与 KIVI
- PagedAttention
- 连续批处理
- 投机解码
## 07. 评估学
- 规则/人工/LLM-judge 三范式
- lm-eval-harness（实操 + 自定义 task）
- HELM
- benchmark 污染（GSM1k）
- ppl 陷阱；**幻觉与安全**（语义熵/SelfCheckGPT
## 08. LoRA 与分类微调
- 从零写 LoRA（低秩分解注入）
- 参数量/显存对比
- 分类微调回顾
## 09. 推理模型与 test-time compute
- R1 四阶段管线
- cold start SFT → 推理 RL → self-consistency