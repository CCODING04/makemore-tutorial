# Part 15: 多模态理解
## 01. 手写拼接式 VLM 四件套
- patch embedding → ViT → mlp2x 投影器 → token 拼接；LLaVA 两阶段训练玩具版
## 02. 三大方案与对齐损失
- 拼接式/门控/early-fusion；CLIP InfoNCE vs SigLIP；动态分辨率与 token 压缩