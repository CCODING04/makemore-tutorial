# Part 7: Minimind 复现
## 01. BPE Tokenizer
- 为什么需要 subword
- BPE 算法原理
- 训练 6400 词表
- 压缩率对比
- chat 格式预告
## 02. 现代组件：RMSNorm 与 RoPE
- LayerNorm 回顾
- RMSNorm
- RoPE 旋转位置编码
- 权重绑定
## 03. GQA 与 FFN：SwiGLU、KV Cache、MoE
- MHA 回顾
- GQA/MQA
- KV Cache
- Flash Attention
- SwiGLU
## 04. 训练流水线：Pretrain → SFT → DPO
- 预训练技巧
- SFT + Loss Masking
- DPO
- 完整流水线与部署（章末挂三阶段验收）
## 05. 复现 minimind 毕业指南
- 课程脚本 ↔ 官方 trainer 对照
- 真实数据下载
- 四阶段超参
- 验收与成本；进阶实验：RoPE 外推四件套 + 迷你 RULER 长上下文评测 + MoE 负载均衡