---
name: tutorial-creator
description: >
  Create comprehensive, self-contained tutorial courses from video lectures/notebooks.
  Generates three parallel outputs per lesson: progressive Python scripts, tutorial Markdown,
  and assignments with auto-tests. Includes a dual-perspective (teacher + student) quality
  review system. Use when asked to "create a tutorial", "build a course", "convert lectures
  to tutorials", "generate educational content", "optimize tutorial", "rewrite tutorial",
  "fix tutorial quality", or "review tutorial quality". Also applies to tasks like
  "make a tutorial from this video/notebook/transcript" or "improve existing tutorial".
---

# Tutorial Creator

Convert video lectures or notebooks into a complete, self-contained tutorial course, or optimize existing tutorials to meet high-quality educational standards.

## Overview

Each lesson produces three deliverables via parallel sub-agents:

```
Phase A: 3 parallel sub-agents
├── Scripts    → progressive, runnable Python files
├── Tutorial   → Markdown chapters with theory, code, diagrams, exercises
└── Assignment → TODO skeleton + auto-test with fixed seeds

Phase B: Verification
├── Run all scripts (MPLBACKEND=Agg for matplotlib)
├── Run assignment tests
├── Check cross-references (paths, code sync)
└── Dual-perspective review (teacher + student)

Phase C: Quality Enhancement (for optimization tasks)
├── Theory depth check
├── Code explanation completeness
├── Engineering practice validation
└── Learning outcome verification
```

## What Makes a Good Tutorial

A high-quality tutorial must have **skeleton** (structure), **flesh** (content depth), and **soul** (pedagogical flow):

### 1. Skeleton — Clear Structure
- **Learning objectives**: 3-5 concrete outcomes using action verbs (understand, master, be able to)
- **Prerequisites**: What students need to know, with links to previous content
- **Chapter navigation**: Clear roadmap of what comes first/next
- **Consistent formatting**: Emoji markers, code blocks, visual hierarchy

### 2. Flesh — Content Depth
- **Theory background**: Why this technology? What problem does it solve? Historical context?
- **Mathematical derivation**: Formulas with step-by-step explanation, not just final results
- **Code explanation**: Line-by-line comments, shape tracking, debugging process
- **Engineering practice**: Performance analysis, common pitfalls, best practices
- **Visual aids**: ASCII diagrams, tables, charts for complex concepts

### 3. Soul — Pedagogical Flow
- **Intuition first, math second**: Start with "why" before "how"
- **Progressive complexity**: Simple → Complete → Industrial
- **Cross-references**: Connect to previous parts, show evolution
- **Hands-on verification**: Every claim backed by runnable code and real output

## Project Structure

```
<project-root>/
├── README.md                  # course map + dependencies
├── data/                      # shared datasets
├── tools/                     # helper scripts (e.g. extract_images.py)
├── courses/
│   └── PartX_<name>/
│       ├── <original>.ipynb   # source notebook (optional)
│       ├── images/            # extracted + generated images
│       ├── scripts/           # 01_*.py → 07_*.py (progressive)
│       └── tutorial/          # 01_*.md, 02_*.md, ..., README.md, notes.md
└── assignments/
    └── assignment_X/
        ├── assignment.md      # problem description
        ├── xxx_exercises.py   # TODO skeleton
        └── test_xxx.py        # pytest-compatible tests
```

## Phase A: Sub-agent Prompts

Spawn 3 sub-agents in parallel. Each reads the same source material (transcript + notebook).

### Sub-agent 1: Scripts

Read `references/scripts-guide.md` for detailed guidance. Key rules:

1. **Progressive**: each script builds on the previous one (not rewritten from scratch)
2. **Self-contained**: every script runs independently (own imports, data loading)
3. **Numbered**: `01_description.py` through `06-07_description.py`
4. **Data path**: use `os.path.dirname(os.path.abspath(__file__))` to resolve relative paths
5. **Comments**: docstring header with lesson number, script number, one-line goal
6. **Shape annotations**: every tensor operation should have shape comments
7. **Debug output**: print intermediate results for verification

### Sub-agent 2: Tutorial

Read `references/tutorial-guide.md` for detailed guidance. Key rules:

1. Split into 2-4 chapter files + README.md if content > 3000 lines
2. Reference scripts as `../scripts/01_xxx.py` and images as `../images/xxx.png`
3. Use emoji markers: 💡 insight, ⚠️ pitfall, 🔑 key concept, 📝 note
4. Prefer ASCII art over Mermaid (more reliable cross-platform)
5. Include 2-3 QA exercises per chapter in `<details>` tags
6. **Accuracy**: search and cross-validate any uncertain claims — never fabricate
7. **Theory depth**: include background, motivation, mathematical derivation, historical context
8. **Code depth**: line-by-line explanation, shape tracking, debugging process
9. **Engineering depth**: performance analysis, common pitfalls, best practices

### Sub-agent 3: Assignment

Read `references/assignment-guide.md` for detailed guidance. Key rules:

1. 4 core exercises + 1 stretch (marked 🌟)
2. Fixed random seeds (`torch.manual_seed(42)`) for reproducibility
3. Tests check **shape, dtype, mathematical invariants** — not exact values
4. Stretch exercise gracefully skipped if returns `None`
5. Provide detailed TODO comments (step-level, almost pseudocode)
6. Include 3-5 thinking questions with `<details>` answers

## Phase B: Verification & Review

After all sub-agents complete, run verification:

### Step 1: Run Scripts
```bash
for s in courses/PartX/scripts/*.py; do
  MPLBACKEND=Agg python3 "$s" && echo "✅ $s" || echo "❌ $s"
done
```

### Step 2: Run Tests
```bash
python3 assignments/assignment_X/test_xxx.py
```

### Step 3: Fix Issues
Common problems (from real experience):
- **Data path errors**: scripts use `__file__`-relative paths, not cwd-relative
- **matplotlib Chinese fonts**: use English for chart titles, Chinese for prose
- **TODO functions returning None**: expected behavior, tests must handle gracefully

### Step 4: Dual-Perspective Review

Spawn 2 review sub-agents:

**Teacher Review** — read `references/review-teacher.md`:
- Concept accuracy (verify formulas, claims, examples)
- Code-tutorial consistency (inline code matches actual scripts)
- Inter-lesson continuity
- Theory depth (background, motivation, derivation completeness)
- Code explanation depth (line-by-line, shape tracking, debugging)
- Engineering practice depth (performance, pitfalls, best practices)
- Rate: P0 (must fix) / P1 (should fix) / P2 (nice to have)

**Student Review** — read `references/review-student.md`:
- Learning curve (where will students get stuck?)
- Explanation clarity (broadcasting, tensor ops, new concepts)
- Exercise difficulty progression
- "If I could only fix 3 things" prioritization
- Theory accessibility (is the motivation clear? is the math explained?)
- Code readability (can I follow the logic? are shapes clear?)

### Step 5: Apply Fixes
Process review results in priority order:
1. Fix all P0 issues
2. Fix P1 issues
3. Re-run verification

### Step 6: Final Scoring (必须执行)

**⚠️ 关键步骤：** 完成所有优化后，必须依据 `references/good-tutorial-standard.md` 进行最终评分。

**评分流程：**

1. **读取标准文档**：`references/good-tutorial-standard.md`
2. **逐维度评分**：骨架、血肉、灵魂、练习各维度独立评分
3. **计算总分**：`总分 = (骨架 × 0.2) + (血肉 × 0.4) + (灵魂 × 0.2) + (练习 × 0.2)`
4. **判定等级**：
   - ⭐⭐⭐⭐⭐ 优秀 (4.5-5.0)：可直接发布
   - ⭐⭐⭐⭐ 良好 (3.5-4.4)：需要小幅优化
   - ⭐⭐⭐ 合格 (2.5-3.4)：需要较大优化
   - ⭐⭐ 待改进 (1.5-2.4)：需要重写
   - ⭐ 不合格 (0-1.4)：需要重新设计

5. **输出评分报告**：

```markdown
## 最终评分报告

### 各维度评分

| 维度 | 评分 | 权重 | 加权分 | 主要优点 | 主要不足 |
|------|------|------|--------|----------|----------|
| 骨架 | X/5 | 20% | X.X | ... | ... |
| 血肉 | X/5 | 40% | X.X | ... | ... |
| 灵魂 | X/5 | 20% | X.X | ... | ... |
| 练习 | X/5 | 20% | X.X | ... | ... |
| **总分** | - | - | **X.X/5** | - | - |

### 等级判定

**⭐⭐⭐⭐ 良好** (X.X/5)

### 改进建议（如果总分 < 4.5）

1. [最高优先级改进]
2. [第二优先级改进]
3. [第三优先级改进]

### 质量门禁检查

- [ ] 学习目标：3-5 个，动词开头，具体可衡量
- [ ] 前置知识：三级分类，有链接
- [ ] 章节导航：完整，有链接，形成闭环
- [ ] 格式一致：emoji 规范，代码块有语法高亮
- [ ] 问题引入：痛点清晰，有类比
- [ ] 数学推导：逐步推导，每步有解释
- [ ] 历史脉络：有演进路径和论文链接
- [ ] 逐行注释：每行都有注释，解释"为什么"
- [ ] 数据流追踪：有 shape/type 标注，有 ASCII 图
- [ ] 实测输出：真实输出，有环境标注
- [ ] 调试展示：3+ 个常见错误
- [ ] 性能分析：有复杂度和实测数据
- [ ] 常见陷阱：3+ 个，有症状/原因/解法
- [ ] 最佳实践：有工业做法和配置推荐
- [ ] 直觉优先：先"为什么"再"怎么做"
- [ ] 渐进复杂：简单 → 完整 → 工业
- [ ] 交叉引用：连接前后内容
- [ ] 动手验证：每个概念都有验证
- [ ] 概念检验：3+ 题，有折叠答案
- [ ] 动手实践：2+ 个代码练习
- [ ] 扩展思考：1+ 个引导性问题
```

**重要提示：**
- 评分必须基于 `references/good-tutorial-standard.md` 的具体标准
- 血肉维度权重最高 (40%)，内容深度是核心
- 总分必须 ≥ 3.5 才能达到"良好"标准
- 如果总分 < 3.5，必须返回继续优化

## Phase C: Quality Enhancement (for Optimization Tasks)

When optimizing existing tutorials, run additional quality checks:

### Step 6: Theory Depth Check
```markdown
For each chapter, verify:
- [ ] Problem introduction: Why does this technology exist?
- [ ] Motivation: What pain point does it solve?
- [ ] Mathematical derivation: Are formulas explained step-by-step?
- [ ] Historical context: Key papers, evolution
- [ ] Intuition: Analogies, visual explanations
```

### Step 7: Code Explanation Completeness
```markdown
For each code block, verify:
- [ ] Purpose statement before the code
- [ ] Line-by-line comments for non-obvious operations
- [ ] Shape annotations on tensor operations
- [ ] Debugging process shown (common errors and fixes)
- [ ] Real output displayed (not hypothetical)
```

### Step 8: Engineering Practice Validation
```markdown
For each topic, verify:
- [ ] Performance analysis (time/space complexity)
- [ ] Common pitfalls (FAQ format: symptom → cause → solution)
- [ ] Best practices (industrial approach)
- [ ] Configuration recommendations
```

### Step 9: Learning Outcome Verification
```markdown
For each chapter, verify:
- [ ] Learning objectives are concrete and measurable
- [ ] "学完本章你能..." section lists specific skills
- [ ] Practice questions test understanding
- [ ] Exercises have clear acceptance criteria
```

## Tutorial Chapter Template

Each chapter should follow this structure for maximum pedagogical effectiveness:

```markdown
# Chapter Title

> 🧭 Navigation arrow (connect to previous/next chapter)

## 学习目标 (Learning Objectives)
- 3-5 concrete outcomes using action verbs
- Example: "理解 XXX 的核心原理", "掌握 YYY 的数学推导", "能够手写 ZZZ 的实现"

## 前置知识 (Prerequisites)
- List required knowledge points
- Provide quick review links
- Mark which are required vs optional

## 理论背景 (Theoretical Background)
### 问题引入 (Problem Introduction)
- Why does this technology exist?
- What pain point does it solve?
- What would happen without it?

### 数学推导 (Mathematical Derivation)
- Where does the formula come from?
- Step-by-step derivation with explanation
- Intuition and analogies

### 历史脉络 (Historical Context)
- Key papers
- Evolution of the technology

## 代码实现 (Code Implementation)
### 形状追踪 (Shape Tracking)
- Every tensor's shape change
- ASCII diagram showing data flow

### 逐行解释 (Line-by-Line Explanation)
- Not just code blocks
- Every line has comments
- Explain why it's written this way

### 调试展示 (Debugging Process)
- Common errors
- How to fix them
- Error message interpretation

## 工程实践 (Engineering Practice)
### 性能分析 (Performance Analysis)
- Time complexity
- Space complexity
- Real measured data

### 常见陷阱 (Common Pitfalls)
- FAQ format
- Symptom → Cause → Solution

### 最佳实践 (Best Practices)
- Industrial approach
- Configuration recommendations

## 练习与思考 (Exercises & Thinking)
### 概念检验 (Concept Check)
- 3-5 Q&A questions
- Collapsed answers in `<details>`

### 动手实践 (Hands-on Practice)
- Code-writing exercises
- Clear acceptance criteria

### 扩展思考 (Extension Thinking)
- Guide deeper exploration
- Connect to other knowledge points

## 参考资源 (References)
- Original papers
- Official documentation
- Related tutorials
- Open-source implementations

## 学完本章你能... (After This Chapter You Can...)
- List specific skills gained
- Use checkboxes for self-assessment
```

## Code Explanation Best Practices

### ❌ Bad: Just paste code
```python
def complex_function(x, y, z):
    # ... 50 lines of code ...
```

### ✅ Good: Step-by-step explanation
```python
# Step 1: Input preprocessing
# x shape: (batch, seq_len, d_model)
x_norm = layer_norm(x)  # Normalize, shape unchanged

# Step 2: Compute attention
# Q, K, V shape: (batch, heads, seq_len, d_k)
Q = x_norm @ W_Q  # Linear projection
K = x_norm @ W_K
V = x_norm @ W_V

# Step 3: Attention scores
# QK^T shape: (batch, heads, seq_len, seq_len)
attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
```

### Shape Tracking Pattern
```markdown
Input tensor shape: (batch=32, seq_len=128, d_model=512)
  ↓ Linear projection
Q shape: (32, 128, 512)
  ↓ Reshape for multi-head
Q shape: (32, 8, 128, 64)  # 8 heads, d_k=64
  ↓ Transpose for attention
Q shape: (32, 8, 128, 64)
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

## Cross-Reference Conventions

| Reference target | Path format |
|---|---|
| Script in same lesson | `../scripts/01_xxx.py` |
| Image in same lesson | `../images/xxx.png` |
| Assignment for this lesson | `../../../assignments/assignment_X/` |
| Previous lesson | `../PartX-1/tutorial/` |
| Data file | `../../../data/names.txt` |

## Inter-Lesson Continuity

Each lesson should:
- **Open with**: reference to previous lesson's results ("In Part X, we achieved...")
- **Motivate**: explain why the previous approach has limitations
- **Close with**: preview of next lesson ("Next time, we'll solve...")
- **Link assignments**: "Ready to practice? → Assignment X"
- **Cross-reference**: connect concepts across parts

## Per-Lesson Time Budget

| Step | Estimated Time |
|------|---------------|
| 3 parallel sub-agents | ~8-10 min |
| Verification + fixes | ~5-8 min |
| Dual review | ~10-15 min |
| Apply review fixes | ~5-10 min |
| Quality enhancement (optimization) | ~10-15 min |
| **Total per lesson** | **~30-45 min** (creation) / **~40-55 min** (optimization) |

## Optimization Workflow

When optimizing existing tutorials:

### Step 1: Assess Current State
Read the existing tutorial and identify:
- What's already good (keep it)
- What's missing (add it)
- What's unclear (rewrite it)
- What's outdated (update it)

### Step 2: Plan Improvements
Create a checklist of improvements:
- Theory depth gaps
- Code explanation gaps
- Engineering practice gaps
- Learning outcome gaps

### Step 3: Apply Improvements
Use the 3 parallel sub-agents to:
1. Enhance scripts (add shape annotations, debug output)
2. Rewrite tutorial (add theory, improve explanations)
3. Improve exercises (add thinking questions)

### Step 4: Verify & Review
Run the full verification and review process.

### Step 5: Document Changes
Add a changelog or improvement notes to help track what was changed and why.
