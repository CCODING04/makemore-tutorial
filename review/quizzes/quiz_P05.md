# 测验 · Part 5（WaveNet 层次化架构）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** 为什么要 Part 3 的字典管理层重构成 PyTorch 风格的层对象？Sequential 容器带来了什么？

- **答案**：字典方案在 forward 里写 `if layer['type'] == 'linear' ... elif ...`，逻辑被类型判断污染，每加一种层就要改 forward；层对象统一 `__call__` 接口后组合自然。Sequential 顺序调用各层，并用 `parameters()` 一行收集全部参数（对每层的 `parameters()` 列表做展平），让"设置 requires_grad、优化器参数列表"等操作一次完成。
- **锚点**：教程 01_pytorchify.md §从 Part 3 的问题出发 / §Sequential 容器

**Q2（数字）** `FlattenConsecutive(2)` 对 `(B, 8, 10)` 做了什么？三层之后形状如何变化？为什么实现用 `view` 而不是 `torch.cat`？

- **答案**：把 `(B, T, C)` 重排成 `(B, T//2, C*2)`：`(B,8,10) → (B,4,20) → (B,2,40) → (B,1,80)`，即 8 字符 → 4 个 bigram → 2 个 fourgram → 1 个 eightgram。`view` 只改 shape/stride，零拷贝，$O(1)$；`cat` 要分配新内存并复制数据，$O(n)$，训练循环中差异会累积。T 不能被 n 整除时直接 assert 报错，所以 block_size 需是 2 的幂（课程用 8）。
- **锚点**：教程 02_wavenet_architecture.md §FlattenConsecutive 层 / §view vs cat

**Q3（概念）** 为什么 Linear 层一行代码都不改就能吃 3D 输入 `(B, T, C)`？这给了 WaveNet 什么自由？

- **答案**：Linear 的仿射变换只作用在**最后一维**：`(B, T, C) @ (C, H) → (B, T, H)`，前置维度自动保留，每个 (b, t) 位置共享同一个权重矩阵——这正是卷积"逐位置共享核"的视角。因此可以不展平 T 维度，在每次 `FlattenConsecutive` 后直接接 Linear，逐层融合。边界：共享权重强制各时间步变换相同，想要位置特化必须显式引入位置信息。
- **锚点**：教程 02_wavenet_architecture.md §关键洞察：Linear 支持多维输入

**Q4（诊断）** WaveNet 的 3D 中间层送进旧的 2D BatchNorm1d 会发生什么？如何修复与验证？

- **答案**：旧实现 `x.mean(dim=0, keepdim=True)` 对 3D 输入得到 `(1, T, C)`，每个时间步**独立**归一化而不是在整个 batch × time 上聚合——它不报错，表现为"loss 在降但性能上不去"的隐藏 bug。修复：2D 输入在 `dim=0`、3D 输入在 `dim=(0,1)` 上同时 reduce；`running_mean/var` 始终保持 1D `(C,)`，eval 模式下 3D 输入要 `unsqueeze` 两次广播成 `(1, 1, C)`。验证：3D 输入跑一遍，检查 running_mean 形状为 `(C,)`、输出均值约 0、标准差约 1。
- **锚点**：教程 03_training_and_bugs.md §BatchNorm1D 的 3D Bug；assignment_5/README.md 题 3

**Q5（对比）** 课程各模型的验证 loss 阶梯是怎样的？放大版 WaveNet 用了什么配置，提升主要来自哪里？

- **答案**：MLP（block 3）约 2.10 → 深层 BN（Part 3，block 3）约 2.07 → WaveNet 小模型（block 8，约 22K 参数）约 2.07 → 放大版（`n_embd=24, n_hidden=128`，batch=128，50K 步）约 1.99~2.00，课程内首次降到 2.0 以下。block 3→8 的小模型就已经到约 2.07，说明主要提升来自更长的上下文（8 字符）；层次化融合的收益在参数效率——直接展平的第一层 `Linear(80, 200)` 需 16000 参数，WaveNet 第一层 `Linear(20, 200)` 只需 4000。
- **锚点**：教程 03_training_and_bugs.md §放大训练 / §性能对比；assignment_5/README.md 深度思考题 Q1 与话术卡 Q5

## 覆盖映射（学习目标 → 题号）

学习目标来源：教程 README.md「学完这一部分你能」。

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 把参数管理重构成 PyTorch 风格的模块化代码 | Q1 | 字典 if/elif → 统一 `__call__` |
| 理解 Sequential 容器和 `parameters()` 统一接口 | Q1 | 一行收集全部参数 |
| 掌握 FlattenConsecutive 层：逐步融合上下文 | Q2 | 形状链与 view/cat 取舍 |
| 理解 Linear 层天然支持多维输入 | Q3 | 只在最后一维做矩阵乘法 |
| 修复 BatchNorm1D 的 3D 输入 bug | Q4 | reduce 维度、running 统计量与广播 |
| 构建 WaveNet 层次化架构，验证 loss 降到 < 2.0 | Q5 | ⚠️ 数字口径不一致，见下注 |

> ⚠️ 正文覆盖不足（数字未同步）：教程 03_training_and_bugs.md「放大训练」表格写放大模型参数量"~170K"，但作业 5 话术卡 Q5 勘误为实测 76,579（并注明"早期教程误写 ~170K，已按实测修正"），教程正文未同步修改。命题时参数量以 76,579 为准、教程表格仅供参考；loss 数字（~1.99/2.00）两处一致。

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| WaveNet 的融合路径 | 8 chars → 4 bigrams → 2 fourgrams → 1 eightgram；树状两两融合，感受野 1→2→4→8 指数增长 |
| FlattenConsecutive(2) 的形状变换 | `(B, T, C) → (B, T/2, C*2)`；`(B,8,10) → (B,4,20)`；T 不整除时 assert |
| view vs cat | view 只改视图零拷贝 $O(1)$；cat 分配复制 $O(n)$；训练循环中优先 view |
| Linear 吃 3D 输入的原因 | 只在最后一维矩阵乘，`(B,T,C) @ (C,H) → (B,T,H)`，逐位置共享权重（卷积视角） |
| BN 3D 修复要点 | 训练在 `dim=(0,1)` reduce；`running_mean/var` 保持 `(C,)`；eval 时 unsqueeze 两次广播 `(1,1,C)` |
| WaveNet 放大版配置与成绩 | `n_embd=24, n_hidden=128`、block 8、batch 128、50K 步 → dev loss 约 1.99（首次 < 2.0） |
| 与直接展平 MLP 比参数量 | 展平第一层 `Linear(80,200)` = 16000 参数；WaveNet 第一层 `Linear(20,200)` = 4000，层次化更省 |
| 与 Attention 的本质区别 | WaveNet 固定局部融合、感受野指数增长；Attention 一层即可全域关注但代价 $O(n^2)$ |
| 为什么叫 WaveNet | 层次融合等价于 dilated causal convolution（膨胀因果卷积），原论文用于音频波形生成 |
| WaveNet 输出如何变 2D | 三层融合后 T=1，logits 是 `(B, 1, vocab)`；在最后一层 Linear 前插 Flatten 收成 `(B, vocab)` |
