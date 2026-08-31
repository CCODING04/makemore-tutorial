# Tutorial Writing Guide

## Structure

Each lesson's tutorial lives in `courses/PartX/tutorial/`:

```
tutorial/
├── README.md           # Chapter navigation + prerequisites
├── 01_introduction.md  # First chapter
├── 02_core_concept.md  # Second chapter
├── 03_advanced.md      # Third chapter (if needed)
└── notes.md            # Learning notes + QA archive (optional but recommended)
```

### When to Split

- **< 3000 lines**: single `tutorial.md` is fine
- **3000+ lines**: split into 2-4 chapter files + `README.md`
- Rule of thumb: 3 files per lesson works well (intro / core / advanced)

## Chapter Template

```markdown
# Part X - Chapter N: Title

> 🧭 Navigation arrow connecting to previous/next chapter

## 学习目标 (Learning Objectives)
- 3-5 concrete outcomes using action verbs
- Example: "理解 XXX 的核心原理", "掌握 YYY 的数学推导", "能够手写 ZZZ 的实现"

## 前置知识 (Prerequisites)
- What the reader should already know (link to previous chapters/lessons)
- Mark which are required vs optional

## 理论背景 (Theoretical Background)

### 问题引入 (Problem Introduction)
Why does this technology exist? What pain point does it solve?
What would happen without it?

### 数学推导 (Mathematical Derivation)
Where does the formula come from? Step-by-step derivation with explanation.
Intuition and analogies.

### 历史脉络 (Historical Context)
Key papers, evolution of the technology.

## 代码实现 (Code Implementation)

### 形状追踪 (Shape Tracking)
Every tensor's shape change. ASCII diagram showing data flow.

### 逐行解释 (Line-by-Line Explanation)
Not just code blocks. Every line has comments. Explain why it's written this way.

### 调试展示 (Debugging Process)
Common errors. How to fix them. Error message interpretation.

## 工程实践 (Engineering Practice)

### 性能分析 (Performance Analysis)
Time complexity, space complexity, real measured data.

### 常见陷阱 (Common Pitfalls)
FAQ format: symptom → cause → solution.

### 最佳实践 (Best Practices)
Industrial approach, configuration recommendations.

## 练习与思考 (Exercises & Thinking)

### 概念检验 (Concept Check)
3-5 Q&A questions with collapsed answers in `<details>`.

### 动手实践 (Hands-on Practice)
Code-writing exercises with clear acceptance criteria.

### 扩展思考 (Extension Thinking)
Guide deeper exploration, connect to other knowledge points.

## 参考资源 (References)
Original papers, official documentation, related tutorials, open-source implementations.

## 学完本章你能... (After This Chapter You Can...)
- List specific skills gained
- Use checkboxes for self-assessment

> 💡 **Insight**: Why this matters conceptually.

> ⚠️ **Pitfall**: Common mistake and how to avoid it.

> 🔑 **Key Concept**: Term — one-sentence definition.

> 📝 **Note**: Additional context or explanation.

## Practice Questions
<details>
<summary>Q: Question text?</summary>
A: Answer with explanation.
</details>

## What's Next
Brief preview of next chapter → link to `02_xxx.md`.
```

## Theory Depth Requirements

### Problem Introduction Pattern
```markdown
## 为什么需要 XXX？

在没有 XXX 之前，我们面临的问题是：
- 痛点 1：具体描述
- 痛点 2：具体描述

XXX 的出现解决了这些问题：
- 解决方案 1
- 解决方案 2

> 💡 类比：用日常生活中的例子帮助理解
```

### Mathematical Derivation Pattern
```markdown
## 数学推导

### 从直觉到公式

我们想要实现的目标是：[直觉描述]

数学上，这可以表示为：[初始公式]

推导过程：
1. 第一步：[解释]
2. 第二步：[解释]
3. ...

最终得到：[最终公式]

> 🔑 关键洞察：公式中每个部分的含义
```

### Historical Context Pattern
```markdown
## 历史脉络

这项技术的发展历程：
- [年份]：[关键论文/事件]
- [年份]：[改进/变体]
- 现在：[当前最佳实践]

> 📚 参考文献：[论文链接]
```

## Code Explanation Depth

### Shape Tracking Pattern
```markdown
## 数据流与形状变化

输入张量形状：(batch=32, seq_len=128, d_model=512)
  ↓ 线性投影
Q 形状：(32, 128, 512)
  ↓ 重塑为多头
Q 形状：(32, 8, 128, 64)  # 8 个头，d_k=64
  ↓ 转置用于注意力计算
Q 形状：(32, 8, 128, 64)

ASCII 图示：
┌─────────────────┐
│ Input (32,128,512) │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Linear → (32,128,512) │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Reshape → (32,8,128,64) │
└─────────────────┘
```

### Line-by-Line Explanation Pattern
```python
# ❌ 不好的方式：贴一大段代码
def complex_function(x, y, z):
    # ... 50 行代码 ...

# ✅ 好的方式：分步解释
# Step 1: 输入预处理
# x 的 shape: (batch, seq_len, d_model)
x_norm = layer_norm(x)  # 归一化，shape 不变

# Step 2: 计算注意力
# Q, K, V 的 shape 都是 (batch, heads, seq_len, d_k)
Q = x_norm @ W_Q  # 线性投影
K = x_norm @ W_K
V = x_norm @ W_V

# Step 3: 注意力分数
# QK^T 的 shape: (batch, heads, seq_len, seq_len)
attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
```

### Debugging Process Pattern
```markdown
## 常见错误与调试

### 错误 1：形状不匹配
**错误信息**：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 512x512)
```

**原因**：[解释为什么会出现这个错误]

**修复方法**：
```python
# 修复前
output = input @ weight  # 错误

# 修复后
output = input @ weight.T  # 正确：转置 weight
```

**预防**：[如何避免这个错误]
```

## Engineering Practice Depth

### Performance Analysis Pattern
```markdown
## 性能分析

### 时间复杂度
- 操作 1：O(n²)
- 操作 2：O(n log n)

### 空间复杂度
- 中间结果：O(n)
- 最终结果：O(1)

### 实测数据
| 场景 | 耗时 | 内存 |
|------|------|------|
| 小规模 | 0.1s | 10MB |
| 中规模 | 1.0s | 100MB |
| 大规模 | 10.0s | 1GB |

> 📊 数据来源：[测试环境说明]
```

### Common Pitfalls Pattern
```markdown
## 常见陷阱

### 陷阱 1：[症状描述]
**症状**：[用户会看到什么]
**原因**：[为什么会发生]
**解法**：
```python
# ❌ 错误做法
...

# ✅ 正确做法
...
```

### 陷阱 2：[症状描述]
...
```

### Best Practices Pattern
```markdown
## 最佳实践

### 实践 1：[名称]
**为什么**：[解释原因]
**怎么做**：[具体步骤]
**工业界做法**：[真实案例]

### 实践 2：[名称]
...
```

## Annotation System

Use consistent emoji markers throughout tutorials:

| Marker | Meaning | Usage |
|--------|---------|-------|
| 🔑 | Key concept | Core ideas that must be remembered |
| ⚠️ | Warning/Pitfall | Common mistakes and how to avoid |
| 💡 | Insight/Tip | Helpful understanding or technique |
| 📝 | Note/Supplement | Additional context or explanation |
| 🔗 | Link | Connection to other parts or resources |
| 🚀 | Performance | Performance-related insights |
| 🧪 | Experiment | Hands-on verification |
| 📊 | Data/Result | Measured results or statistics |

## Navigation Links

Each Part's `tutorial/README.md` should include navigation links at the end:

```markdown
---

[← 上一章：Part X](../../PartX_xxx/tutorial/README.md) | [下一章：Part Y →](../../PartY_yyy/tutorial/README.md)
```

**Rules**:
- First Part: only "下一章" link
- Last Part: only "上一章" link
- Middle Parts: both links
- Use `../../` to navigate from `courses/PartX/tutorial/` to `courses/PartY/tutorial/`

## Common Pain Points

Each chapter should proactively explain concepts that students commonly struggle with:

### PyTorch Basics
- **Broadcasting**: explain rules with shape derivation examples
- **`view(-1)`**: explain automatic dimension calculation
- **`C[X]` advanced indexing**: explain shape derivation (N,3,2)
- **`keepdims=True`**: explain why it's needed

### Training Concepts
- **Learning rate**: explain why different values work for different tasks
- **Loss function**: explain the mathematical meaning
- **Gradient descent**: explain the update rule

### Common Pitfalls
- **`.sum()` vs `.mean()`**: explain when to use which
- **`torch.no_grad()`**: explain why it's needed for evaluation
- **Random seeds**: explain reproducibility

> 💡 Add these explanations **proactively** in the tutorial, not just when students ask. This prevents the most common stuck points.

## Cross-Reference Conventions

| Reference target | Path format |
|---|---|
| Script in same lesson | `../scripts/01_xxx.py` |
| Image in same lesson | `../images/xxx.png` |
| Assignment for this lesson | `../../../assignments/assignment_X/` |
| Previous lesson | `../PartX-1/tutorial/` |
| Data file | `../../../data/names.txt` |

## Writing Style

1. **Language**: Chinese (or target language), conversational but accurate
2. **Emoji markers**: consistent use of 💡 (insight), ⚠️ (pitfall), 🔑 (key concept), 📝 (note)
3. **Code blocks**: always specify language (`python`, `bash`)
4. **Diagrams**: prefer ASCII art over Mermaid (more reliable rendering)
5. **No fabrications**: if unsure about a claim, search and cross-validate before including
6. **Show real outputs**: run scripts and use actual output in tutorial, not hypothetical
7. **Mark approximations**: use `≈` for approximate values, explain why exact match isn't expected
8. **Cite sources**: for non-obvious claims, note the reference (paper, documentation)

### Good ASCII Art Example
```
Input (3 chars)          Embedding (3×2)           Concat (6)
┌───┬───┬───┐           ┌───┬───┐
│ . │ e │ m │    →      │   │   │  e             ┌───────────┐
└───┴───┴───┘           │   │   │  m      →      │ 6 floats  │
                        └───┴───┘                 └───────────┘
```

### Mermaid Fallback (only when ASCII is insufficient)
Use only for complex flowcharts. Avoid in node text:
- ❌ `<br/>`, HTML tags
- ❌ `|`, `()`, `{}` special chars
- ❌ Emoji in node names
- ✅ Plain text + numbers + basic punctuation only

## Accuracy Rules

1. **Never fabricate**: if the transcript is vague, search the web for verification
2. **Show real outputs**: run scripts and use actual output in tutorial, not hypothetical
3. **Mark approximations**: use `≈` for approximate values, explain why exact match isn't expected
4. **Cite sources**: for non-obvious claims, note the reference (paper, documentation)

## Inter-Lesson Continuity

Each lesson should:
- **Open with**: reference to previous lesson's results ("In Part X, we achieved...")
- **Motivate**: explain why the previous approach has limitations
- **Close with**: preview of next lesson ("Next time, we'll solve...")
- **Link assignments**: "Ready to practice? → Assignment X"

## Learning Notes (Optional but Recommended)

Create a `notes.md` file to archive learning Q&A:

```markdown
# Part X 学习笔记

> 记录学习过程中的临时提问与解答，供复习参考。

## 学习总结

**完成时间**：YYYY-MM-DD

### 掌握扎实的知识点
- 知识点 1
- 知识点 2

### 需要加强的知识点
- 知识点 1
- 知识点 2

### 课下建议
1. 建议 1
2. 建议 2

## 章节 QA 记录

### 章节 1：标题 — QA 记录
> 📅 YYYY-MM-DD

**Q**：问题描述？

**A**：详细解答。

---

**Q**：另一个问题？

**A**：详细解答。
```

## FAQ: Common Mistakes Found in Reviews

1. **Tutorial code ≠ script code**: always copy from the actual runnable script, don't write from memory
2. **Parameter counts**: calculate and verify; don't guess
3. **Sample outputs**: run the actual script and paste real output
4. **Variable names**: keep consistent between tutorial inline code and scripts
5. **Self-correction residue**: remove any "wait, let me rephrase" artifacts before publishing
6. **Missing theory**: don't just show code, explain why it's written this way
7. **Missing shape tracking**: every tensor operation should have shape comments
8. **Missing debugging**: show common errors and how to fix them
9. **Missing performance analysis**: include time/space complexity and real measurements
10. **Missing best practices**: include industrial approach and configuration recommendations
