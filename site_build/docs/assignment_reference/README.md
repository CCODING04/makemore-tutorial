# 📝 Assignment Reference — 作业参考答案

> 本目录收录 **Assignment 1-19 的参考答案**（与 `assignments/` 各题的测试文件一一对应）。
> 每个答案都在本课程环境（Python 3.12 + torch 2.6.0+cu124，双 4090）上**实际运行验证通过**。

## ✅ 验证状态（2026-09-01 实测）

| 作业 | 主题 | 参考答案测试结果 | 备注 |
|---|---|---|---|
| 01 | Bigram | 5/5 ✅（NLL=2.4546） | |
| 02 | MLP | 5/5 ✅（tuning val=2.4509） | ⚠️ 原测试初始化 bug 已修（见下） |
| 03 | BatchNorm | 5/5 ✅（dev_loss=2.2296） | ⚠️ 原测试边界 bug 已修 |
| 04 | 手动反传 | 5/5 ✅（全部 max diff <1e-4） | ⚠️ 原测试 3 处 bug 已修 |
| 05 | WaveNet | 5/5 ✅（dev_loss=2.0700） | |
| 06 | Transformer | 7/7 ✅ | |
| 07 | Minimind | 7/7 ✅ | 题 5 需 GPU（自动跳过） |
| 08 | 后训练 | 8/8 ✅ | |
| 09 | CUDA | 5/5 ✅ | 题 5 需 GPU + triton |
| 10 | 分布式 | 5/5 ✅ | 纯 CPU 可跑 |
| 11 | 对齐实战 | 3/3 ✅ | 纯 CPU 可跑 |
| 12 | 微调实战 | 4/4 ✅ | 纯 CPU 可跑 |
| 13 | 数据工程 | 4/4 ✅ | 纯标准库 |
| 14 | 推理部署 | 3/3 ✅ | 纯纸笔数学 |
| 15 | 多模态理解（VLM） | 4/4 ✅ | 纯 CPU 可跑 |
| 16 | 图像/视频生成 | 4/4 ✅ | 纯 CPU 可跑 |
| 17 | Agentic RL | 4/4 ✅ | 纯标准库；test 文件由本轮补齐，reference 目录用 `assignments/assignment_17/` 的测试复核通过（4/4） |
| 18 | RAG 全链路 | 5/5 ✅ | 纯 CPU 可跑（题 5 🌟权重网格搜索含在内）；2026-09-01 实测 |
| 19 | Agent/FC | 5/5 ✅ | 纯标准库（题 5 🌟mini-MCP 为 mock transport）；2026-09-01 实测 |

## 🔧 使用方式

```bash
# 每个目录内：xxx_exercises.py 是完整答案，test_xxx.py 是对应测试（与 assignments/ 相同）
cd assignment_reference/assignment_01
python test_bigram_exercises.py     # 应全部 ✅

# ⚠️ 诚实提示：先自己做完 assignments/ 里的作业再看答案。
# 直接抄答案 = 放弃了这套课程最重要的练习环节。
```

## 🐛 审查中发现并修复的原作业 bug（P0，已同步到 assignments/ 原文件）

1. **Assignment 2 `test_train_step`**：std=1 的 W1/W2 初始化让初始 CE≈18，任何正确实现都过不了
   `<10` 断言——已改为小尺度初始化（`*0.1/*0.01`），初始 CE≈3.37 ≈ ln(27)，与注释意图一致。
   `tuning_experiment` 的骨架提示同步修正（1000 步内 std=1 init 到不了 <2.5，实测 3.64）。
2. **Assignment 3 `test_diagnose_initial_loss`**：`<10` 断言与课程史实矛盾——
   Part 3 脚本 01 实测未修正初始 loss = **26.78**（n_hidden=200 全量数据），正确实现必然
   ~20-30。断言已改为 `>10`（"明显大于 ln(27)"）。
3. **Assignment 4 `test_q1/q2`**：`forward_pass(params, Xb)` 缺 Yb 参数但断言 cache['loss']
   （CE 需要标签）——签名已改为 `(params, Xb, Yb)`，测试调用同步。
4. **Assignment 4 `test_q4`**：① 参数未开 requires_grad → `loss.backward()` 必崩；
   ② `hpreact` 非叶子节点 `.grad` 默认不保存 → 需 `retain_grad()`；
   ③ max-diff 阈值 1e-5 过严（两条独立 float32 计算链实测噪声 ≈4.8e-5）→ 放宽至 1e-4。

## 📌 已知环境依赖

- **GPU（4090）**：assignment_07 题 5（Triton）、assignment_09 题 5（Triton softmax）——
  无 GPU 时这些子项自动跳过，其余全部可跑
- **Part 11-17 的工具链实操**（verl Docker / LLaMA-Factory / vLLM / 多模态与 Agent 框架）
  不在本目录范围——它们的运行方式见各章教程，作业答案只覆盖纯 Python 部分
