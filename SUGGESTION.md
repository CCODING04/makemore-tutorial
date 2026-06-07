# 📋 Makemore Tutorial 改进方案

> **生成日期**：2026-06-07
> **最近验证**：2026-06-07（学生 agent 全面审查）
> **来源**：综合 Charmander 评审 + Codex 改进意见 + STUDENT_FEEDBACK.md + 学生 agent 全面审查
> **定位**：改进计划清单，逐项完成后可勾选

---

## 改进总览

| 优先级 | 类别 | 数量 | 说明 |
|--------|------|------|------|
| 🔴 P0 | 必须修复 | 5 项 | 已确认的 bug 和不一致 |
| 🟡 P1 | 强烈建议 | 12 项 | 内容补充、卡点解释、体验优化 |
| 🟢 P2 | 提升质量 | 7 项 | 统一规范、扩展资源、长期优化 |

---

## 🔴 P0：必须修复

### P0-5：学生 agent 全面审查发现的新问题

- [ ] **Part 2/3 赋值链接错误**
  - 文件：
    - `courses/Part2_mlp/tutorial/README.md` L47
    - `courses/Part2_mlp/tutorial/03_training_and_eval.md` L251
    - `courses/Part3_batchnorm/tutorial/README.md` L46
    - `courses/Part3_batchnorm/tutorial/03_deep_network.md` L292
  - 问题：`../assignment_2/` 解析为 `courses/Part2_mlp/assignment_2/`（不存在）
  - 修复：改为 `../../../assignments/assignment_2/`

---

## ⚠️ REVIEW.md 问题验证结果

> 以下是对 REVIEW.md 中所有问题的**逐项验证**，标注了当前状态。

### 已修复的问题（不需要再处理）

- [x] **Part3 Linear 类缺少 Kaiming gain** — ❌ **已修复**
  - 文件：`courses/Part3_batchnorm/tutorial/03_deep_network.md` L17
  - 当前代码：`self.weight = torch.randn((fan_in, fan_out), generator=g) * (5/3) / fan_in**0.5`
  - **结论**：Kaiming gain `* (5/3)` 已存在，无需修改

- [x] **Assignment 2 evaluate 提示代码参数 bug** — ❌ **已修复**
  - 文件：`assignments/assignment_2/README.md` L161
  - 当前代码：`logits = mlp_forward(X, C, W1, b1, W2, b2)`（无 `Y` 参数）
  - **结论**：代码已正确，无需修改

### 仍需修复的问题

- [x] **Part1 采样示例名与脚本输出不一致** ✅ 已修复 — ⚠️ **仍存在**
  - 文件：`courses/Part1_bigrams/tutorial/02_bigram_model.md` L139-143
  - 当前内容：`junide, janasah, p, cony, a`（来自 Karpathy 原视频）
  - 问题：与 `scripts/04_probability_sampling.py` 实际输出不一致
  - 修复：用脚本实际输出替换示例名

- [x] **Part1 正则化 `.sum()` vs `.mean()` 不一致** ✅ 已修复
  - 文件：`courses/Part1_bigrams/tutorial/03_neural_network.md` L150
  - 教程：`reg_loss = 0.01 * (W ** 2).sum()`
  - 脚本：`scripts/07_gradient_descent.py` 使用 `.mean()`
  - 修复：统一为 `.mean()`（与 Karpathy 原课一致）

- [x] **Part2 教程包含编辑遗留** ✅ 已修复
  - 文件：`courses/Part2_mlp/tutorial/01_introduction.md` L103
  - 内容："我们有三份名字数据：32033 + 3 + 1 = 32033+3+1。不对，让我重新说——"
  - 修复：删除该句

- [x] **Part2 参数量描述错误** ✅ 已修复
  - 文件：`courses/Part2_mlp/tutorial/03_training_and_eval.md` L154, L296
  - 教程："约 1.3 万"（实际 n_embd=2, n_hidden=100 时为 3,481）
  - 修复：改为 "约 3,500" 或加公式说明

- [x] **Part3 momentum 值不一致** ✅ 已修复
  - 文件：`courses/Part3_batchnorm/tutorial/02_batchnorm.md`
  - 内联代码 L73-74：`0.999 * ... + 0.001 * ...`（momentum=0.001）
  - BatchNorm1d 类 L105：`momentum=0.1`
  - 修复：统一或加说明解释两种写法的区别

- [x] **Part3 层结构不一致** ✅ 已修复
  - 教程：`03_deep_network.md` 使用类定义（Linear, Tanh, BatchNorm1d）
  - 脚本：`scripts/05_deep_network.py` 使用字典结构 `{'type': 'linear', 'W': W, 'b': b}`
  - 修复：统一或在教程中说明

---

## 🔴 P0：必须修复

### P0-1：修复仍存在的 REVIEW.md 问题

- [x] Part1 采样示例名与脚本输出不一致 ✅ 已修复
- [x] Part1 正则化 `.sum()` vs `.mean()` 不一致 ✅ 已修复
- [x] Part2 编辑遗留 ✅ 已修复
- [x] Part2 参数量描述错误 ✅ 已修复
- [x] Part3 momentum 值不一致 ✅ 已修复
- [x] Part3 层结构不一致 ✅ 已修复

### P0-2：修复 STUDENT_FEEDBACK.md 中的严重卡点

- [x] **卡点 1：Broadcasting 解释不足** ✅ 已修复
  - 文件：`courses/Part1_bigrams/tutorial/02_bigram_model.md`
  - 修复：在概率矩阵归一化处增加 Broadcasting 速成小节
  - 内容：
    - 为什么 `(27, 27) / (27, 1)` 会自动广播？
    - 广播规则：从右对齐、维度一致或一方为 1
    - `keepdims=True` 的作用和不加的后果

- [x] **卡点 2：`C[X]` 高级索引未解释** ✅ 已修复
  - 文件：`courses/Part2_mlp/tutorial/02_mlp_architecture.md`
  - 修复：在 Embedding 查表处增加形状推导示例
  - 内容：
    - `C.shape=(27,2)`, `X.shape=(N,3)` → `C[X].shape=(N,3,2)`
    - 与 `F.one_hot(X) @ C` 的等价关系

- [x] **卡点 3：`view(-1)` 未解释** ✅ 已修复
  - 文件：`courses/Part2_mlp/tutorial/02_mlp_architecture.md`
  - 修复：在 `emb.view(emb.shape[0], -1)` 处增加解释
  - 内容：`(N, 3, 2) → (N, 6)`，保留 batch 维度，自动计算剩余

### P0-3：新增 requirements.txt

- [x] 创建 `requirements.txt` ✅ 已创建
  ```
  torch
  matplotlib
  pytest
  ```
- [x] README 中更新安装方式 ✅ 已更新

### P0-4：添加章节间导航链接

- [x] 每个 Part 的 `tutorial/README.md` 末尾添加导航链接 ✅ 已添加
- [ ] 格式：
  ```markdown
  ---
  [← 上一章：Part X](../PartX_xxx/tutorial/README.md) | [下一章：Part Y →](../PartY_yyy/tutorial/README.md)
  ```
- [ ] Part 1 无上一章，Part 5 无下一章，只有一侧链接

---

## 🟡 P1：强烈建议

### P1-1：扩充 Part 4 教程内容

Part 4 是最难的章节（手动反向传播），但教程内容最薄（555 行）。需要补充：

- [ ] 每一步反向传播的**形状推导图**
  ```
  dlogits (N, 27)
      ↓ cross entropy backward
  dhpreact (N, 200)
      ↓ batchnorm backward
  dh (N, 200)
      ↓ tanh backward
  dhprebn (N, 200)
      ↓ linear backward
  dembcat (N, 30)
      ↓ reshape backward
  demb (N, 3, 10)
      ↓ embedding backward
  dC (27, 10)
  ```

- [ ] **BatchNorm 反传的完整推导**（Karpathy 花了近 30 分钟讲这个）
- [ ] 手动梯度 vs autograd 梯度的**对比验证步骤**
- [ ] **梯度流表**：

  | forward 变量 | backward 梯度 | shape |
  |---|---|---|
  | logits | dlogits | (N, 27) |
  | hpreact | dhpreact | (N, 200) |
  | h | dh | (N, 200) |
  | embcat | dembcat | (N, 30) |
  | emb | demb | (N, 3, 10) |
  | C | dC | (27, 10) |

- [ ] 添加"常见错误"小节（忘除 batch size、softmax 维度错、embedding 累加等）
- [ ] 添加心理疏导："第一次读不要求全部记住，目标是理解每一步梯度从哪里来"

### P1-2：扩充 Part 5 教程内容

- [ ] 补充 **shape 变化表**：

  | 层 | 输入 shape | 输出 shape |
  |---|---|---|
  | Embedding | `(B, 8)` | `(B, 8, C)` |
  | FlattenConsecutive(2) | `(B, 8, C)` | `(B, 4, 2C)` |
  | Linear | `(B, 4, 2C)` | `(B, 4, H)` |
  | FlattenConsecutive(2) | `(B, 4, H)` | `(B, 2, 2H)` |
  | FlattenConsecutive(2) | `(B, 2, H)` | `(B, 2H)` |
  | Linear output | `(B, 2H)` | `(B, vocab)` |

- [ ] 详细解释 **BatchNorm 3D bug**（默认对 dim=0 求均值，3D 应对 `(0, 1)` 求）
- [ ] 明确说明这不是完整卷积实现，是用 FlattenConsecutive 模拟
- [ ] 增加与 Part 2 MLP 的**性能对比表**

### P1-3：给长训练脚本增加 quick mode

- [ ] 涉及脚本：
  - `Part4/scripts/06_manual_training.py`（200,000 步）
  - `Part5/scripts/07_scaled_wavenet.py`（50,000 步）
  - 其他训练步数 > 10,000 的脚本
- [ ] 实现方式：
  ```python
  import sys
  QUICK = "--quick" in sys.argv
  max_steps = 1000 if QUICK else 200000
  ```
- [ ] 脚本顶部注释说明：`默认 quick 模式（1000步），加 --full 跑完整训练`

### P1-4：补充高概率卡点解释

- [x] **Part 1**：learning rate 为什么可以是 50 ✅ 已添加
  - 说明这是 toy problem 的特例，不是通用经验
- [ ] **Part 2**：学习率搜索的上下文解释
  - "在指数空间扫描候选值，找到 loss 下降最快且稳定的区域"
- [x] **Part 1**：增加"计数模型 vs 神经网络模型"对照表 ✅ 已添加

  | 计数模型 | 神经网络模型 |
  |---|---|
  | bigram count matrix | weight matrix W |
  | row normalize | softmax |
  | NLL evaluation | cross entropy |
  | sampling from P | sampling from softmax |

### P1-5：增加作业分级提示

- [ ] 每个 `assignments/assignment_X/` 新增 `hints.md`：
  ```markdown
  ## Hint 1：方向提示（不看代码也能想到）
  ## Hint 2：关键公式/概念
  ## Hint 3：伪代码骨架
  ## Solution：完整参考实现
  ```
- [ ] 覆盖 assignment 1-5

### P1-6：补充 Part 4、5 的思考题

- [x] Part 4 作业补充 ✅ 已添加
  - "如果省略 bngain 的梯度会怎样？"
  - "手动梯度和 autograd 梯度有微小误差（1e-7），原因是什么？"
- [x] Part 5 作业补充 ✅ 已添加
  - "同样 block_size=8，直接 flatten 的 MLP 和 WaveNet 哪个参数更多？为什么？"
  - "WaveNet 的树状结构和 Transformer 的注意力机制有什么本质区别？"

### P1-7：Part 3 补充诊断速查表

- [ ] 在教程中增加：

  | 诊断对象 | 看什么 | 异常说明 |
  |---|---|---|
  | activation distribution | 是否过度饱和 | tanh 接近 -1 或 1 |
  | gradient distribution | 是否消失/爆炸 | 梯度太小或太大 |
  | weight gradient ratio | 参数更新是否均衡 | 某层训练异常 |
  | update/data ratio | 学习率是否合适 | 太大震荡，太小学不动 |

- [ ] 诊断练习要求"画出对比图"时，提供脚本或代码框架

### P1-8：README 补充

- [x] 增加"推荐学习顺序"小节 ✅ 已添加
- [x] 增加"如何运行全部测试" ✅ 已添加
- [x] 增加"如何判断学完一个 Part"的完成标准 ✅ 已添加
- [x] 明确说明本仓库只覆盖 makemore Part 1-5 ✅ 已添加
- [x] 增加 Micrograd 作为推荐阅读/前置课程的说明 ✅ 已添加

### P1-9：消除脚本间代码重复

- [ ] Part 2 的 `06_visualize_embedding.py` 和 `07_sampling.py` 重复了完整训练循环
- [ ] 方案：用 `torch.save`/`torch.load` 或提取共享模块
- [ ] 或至少在脚本头部加注释说明代码是复制的

### P1-10：统一 Part 1-2 的字符映射风格

- [ ] Part 1 用 `stoi = {s: i+1 for i, s in enumerate(chars)}; stoi['.'] = 0`
- [ ] Part 2 用 `chars = ['.'] + chars; stoi = {s: i for i, s in enumerate(chars)}`
- [ ] 统一为 Part 2 的写法（更简洁）

---

## 🟢 P2：提升质量（后续迭代）

### P2-1：统一 Assignment 模板

每个作业 README 统一为：

```markdown
## 作业目标
## 前置知识
## 题目列表
## 推荐完成顺序
## 如何运行测试
## 思考题
## 常见错误
## 提交检查清单
```

### P2-2：统一 Tutorial 小节结构

每个教程小节统一为：

```markdown
# 标题
## 这一节要解决什么问题
## 直觉解释
## 代码实现
## Shape 检查
## 常见错误
## 运行脚本
## 小结
## 课后练习
```

### P2-3：统一 Scripts 规范

- [ ] 所有脚本顶部统一数据路径：
  ```python
  from pathlib import Path
  ROOT = Path(__file__).resolve().parents[3]
  DATA_PATH = ROOT / "data" / "names.txt"
  ```
- [ ] 所有脚本结尾输出"下一步建议"
- [ ] 训练脚本输出格式统一（step/loss/dev loss/sample names）
- [ ] 图片生成脚本说明保存路径

### P2-4：每个 Part 增加"学完检查清单"

```markdown
## ✅ 学完本 Part 你应该能做到

- [ ] 能解释 XXX 核心概念
- [ ] 能独立运行所有 scripts
- [ ] 能完成 assignment 并通过测试
- [ ] 能回答思考题
- [ ] 能向别人解释 XXX（费曼检验）
```

### P2-5：增加 answers 目录

- [ ] 每个 `assignments/assignment_X/` 增加 `solutions/` 目录
- [ ] 包含完整参考实现
- [ ] README 中说明"先自己做，实在不行再看答案"

### P2-6：README 增加 Windows 兼容说明

- [ ] 补充 PowerShell 下的运行示例
- [ ] 说明路径分隔符差异

### P2-7：检查所有 tutorial 中的相对链接

- [ ] 检查所有 tutorial 中的相对链接是否正确（`../scripts/`、`../../../assignments/`）
- [ ] 检查 Part2 中是否引用了 `Part1_bigram`（应为 `Part1_bigrams`）
- [ ] 检查 Part5 tutorial 中 assignment 链接

---

## Karpathy 原课内容覆盖度分析

### 已覆盖的内容（5 讲）

| Part | 原视频时长 | 仓库覆盖 | 关键知识点 |
|------|-----------|---------|-----------|
| 1 Bigrams | 1h57m | ✅ 完整 | 频率矩阵、Softmax、NLL、梯度下降 |
| 2 MLP | 1h15m | ✅ 完整 | Embedding、MLP、Minibatch SGD、Train/Dev/Test |
| 3 BatchNorm | 1h55m | ✅ 完整 | 激活诊断、Kaiming 初始化、BatchNorm |
| 4 Manual Backprop | 1h55m | ⚠️ 偏薄 | 手动梯度、CrossEntropy 反传、BN 反传 |
| 5 WaveNet | 56min | ⚠️ 偏薄 | FlattenConsecutive、WaveNet、3D BatchNorm |

### 未覆盖的内容

| 课程 | 是否属于 makemore | 建议 |
|------|:-----------------:|------|
| Micrograd（反向传播基础） | ❌ 独立课程 | 建议在 README 中推荐阅读，作为 Part 4 的前置 |
| GPT from Scratch | ❌ 独立课程 | 不需要覆盖，但可在 README 中提及作为进阶 |
| GPT Tokenizer | ❌ 独立课程 | 不需要覆盖 |

### Karpathy 原课中的关键知识点检查

| 知识点 | 是否覆盖 | 位置 |
|--------|:--------:|------|
| Bigram 计数法 | ✅ | Part 1 |
| 神经网络版 Bigram | ✅ | Part 1 |
| Softmax + NLL | ✅ | Part 1 |
| Embedding 查表 | ✅ | Part 2 |
| MLP 前向传播 | ✅ | Part 2 |
| Minibatch SGD | ✅ | Part 2 |
| Train/Dev/Test 划分 | ✅ | Part 2 |
| 学习率搜索 | ✅ | Part 2 |
| 初始 Loss 诊断 | ✅ | Part 3 |
| Tanh 饱和诊断 | ✅ | Part 3 |
| Kaiming 初始化 | ✅ | Part 3 |
| BatchNorm 实现 | ✅ | Part 3 |
| 手动反向传播 | ✅ | Part 4 |
| CrossEntropy 反传 | ✅ | Part 4 |
| BatchNorm 反传 | ✅ | Part 4 |
| FlattenConsecutive | ✅ | Part 5 |
| WaveNet 架构 | ✅ | Part 5 |
| 3D BatchNorm | ✅ | Part 5 |

---

## 执行建议

### 第一阶段：修 bug（P0-1 ~ P0-2）

先修复所有已确认的错误和不一致。这是基础，不修的话学生会困惑。

### 第二阶段：补内容（P0-3 ~ P0-4）

新增 requirements.txt、导航链接。这些是"能自学"的前提。

### 第三阶段：优化体验（P1 系列）

补充卡点解释、quick mode、分级提示、思考题。这些让学习体验更顺畅。

### 第四阶段：长期打磨（P2 系列）

统一规范、模板、答案目录。这些是锦上添花。

---

## 两个评审意见对比

| 改进项 | Charmander | Codex | 结论 |
|--------|:---:|:---:|------|
| 修复 REVIEW.md 已知问题 | ✅ | ✅ | 一致，必须做 |
| Part 4/5 教程内容太薄 | ✅ | ✅ | 一致，P1 |
| 章节导航链接 | ✅ | — | Charmander 独有，P0 |
| Micrograd 推荐阅读 | ✅ | — | Charmander 独有，P1 |
| 脚本代码重复 | ✅ | — | Charmander 独有，P1 |
| requirements.txt | — | ✅ | Codex 独有，P0 |
| quick mode | — | ✅ | Codex 独有，P1 |
| 分级提示/答案 | — | ✅ | Codex 独有，P1 |
| 卡点详细解释 | — | ✅ | Codex 独有，P1 |
| Shape/梯度流表 | — | ✅ | Codex 独有，P1 |
| 统一教程/脚本规范 | — | ✅ | Codex 独有，P2 |
| Assignment 统一模板 | — | ✅ | Codex 独有，P2 |
| broadcasting/keepdim 解释 | — | ✅ | Codex 独有，P1 |
| 学习率 50 解释 | — | ✅ | Codex 独有，P1 |

**结论**：两个评审意见高度互补，没有冲突项。Codex 的建议更细粒度、更具体；Charmander 的建议更偏结构和深度。合并后覆盖全面。

---

*此文档为改进计划，执行时逐项完成后标记 `[x]`。*
