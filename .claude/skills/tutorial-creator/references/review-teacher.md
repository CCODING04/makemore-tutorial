# Teacher Review Guide

## Review Scope

As a teacher/reviewer, evaluate each lesson for **accuracy**, **consistency**, **pedagogical quality**, and **content depth**.

## Rating System

| Level | Meaning | Action |
|-------|---------|--------|
| **P0** | Must fix — factual error or broken code | Fix before any release |
| **P1** | Should fix — inconsistency or misleading content | Fix before public release |
| **P2** | Nice to have — style, polish, minor improvements | Fix in next iteration |

## Review Checklist

### 1. Concept Accuracy (most critical)

- [ ] All mathematical formulas are correct (verify against primary sources)
- [ ] Claims about performance/behavior match actual script outputs
- [ ] Analogies and metaphors are accurate (not misleading)
- [ ] No fabricated content — uncertain claims are marked or removed

### 2. Theory Depth (NEW - Critical for educational quality)

- [ ] **Problem introduction**: Why does this technology exist? What pain point does it solve?
- [ ] **Motivation**: Is the "why" explained before the "how"?
- [ ] **Mathematical derivation**: Are formulas explained step-by-step, not just stated?
- [ ] **Historical context**: Key papers, evolution, current best practices
- [ ] **Intuition**: Are there analogies, visual explanations, or metaphors?
- [ ] **Cross-references**: Does it connect to previous parts and show evolution?

**Common theory depth issues:**
- Jumping straight to code without explaining why
- Stating formulas without derivation
- Missing the "before this technology" context
- No connection to previous learning

### 3. Code Explanation Depth (NEW - Critical for learning)

- [ ] **Line-by-line explanation**: Every non-obvious line has comments
- [ ] **Shape tracking**: Every tensor operation has shape annotations
- [ ] **Purpose statements**: Code blocks have clear "what this does" explanations
- [ ] **Debugging process**: Common errors and fixes are shown
- [ ] **Real outputs**: Script outputs are displayed, not hypothetical
- [ ] **Variable naming**: Consistent between tutorial and scripts

**Common code explanation issues:**
- Code blocks without any comments
- Missing shape annotations on tensor operations
- No explanation of why code is written this way
- Hypothetical outputs instead of real ones

### 4. Engineering Practice Depth (NEW - Critical for real-world application)

- [ ] **Performance analysis**: Time/space complexity explained
- [ ] **Common pitfalls**: FAQ format with symptom → cause → solution
- [ ] **Best practices**: Industrial approach and configuration recommendations
- [ ] **Real-world context**: How is this used in production?

**Common engineering practice issues:**
- No performance analysis
- Missing common pitfalls section
- No mention of industrial best practices
- No configuration recommendations

### 5. Code-Tutorial Consistency

- [ ] Inline code in tutorial matches actual script implementations
- [ ] Variable names are consistent (tutorial uses same names as scripts)
- [ ] Function signatures in assignment hints match skeleton code
- [ ] Numeric examples in tutorial match real script outputs

**Common issues found in practice:**
- Tutorial shows `.sum()` but script uses `.mean()`
- Tutorial claims "1.3万 parameters" but actual count is 3,481
- Tutorial uses class-based layer interface, scripts use dict-based
- Tutorial Kaiming init missing gain factor (e.g., `5/3` for tanh)

### 6. Cross-Lesson Continuity

- [ ] Each lesson opens by connecting to previous lesson's results
- [ ] Code style is consistent across lessons (e.g., character mapping)
- [ ] Difficulty progression is smooth (no sudden jumps)
- [ ] Prerequisites are correctly stated
- [ ] Cross-references to other parts are accurate

### 7. Assignment Quality

- [ ] Hint code compiles and has correct function signatures
- [ ] Tests verify meaningful properties (not trivially passable)
- [ ] Stretch exercises have graceful skip when unimplemented
- [ ] TODO hints are step-level (helpful but not giving away answers)
- [ ] Thinking questions test understanding, not just recall

### 8. Script Quality

- [ ] All scripts run without errors
- [ ] Each script is self-contained (independent execution)
- [ ] Progressive structure is maintained (later scripts build on earlier)
- [ ] No duplicated training code where shared logic would be better
- [ ] Shape annotations present on tensor operations
- [ ] Debug output for intermediate results

### 9. Learning Outcome Quality

- [ ] Learning objectives are concrete and measurable
- [ ] "学完本章你能..." section lists specific skills
- [ ] Practice questions test understanding
- [ ] Exercises have clear acceptance criteria
- [ ] Thinking questions promote deeper exploration

## Depth Assessment Matrix

Rate each chapter on a scale of 1-5 for depth:

| Dimension | 1 (Shallow) | 3 (Adequate) | 5 (Deep) |
|-----------|-------------|--------------|----------|
| **Theory** | Just states facts | Explains why | Derives from first principles |
| **Code** | Shows code only | Some comments | Line-by-line with shapes |
| **Engineering** | No analysis | Basic complexity | Full performance + pitfalls |
| **Practice** | No exercises | Basic exercises | Thinking questions + extension |

**Target**: All dimensions should be ≥ 3 for a publishable tutorial.

## Review Output Format

```markdown
# Review: Part X

## Overall Score: X.X/10

## Depth Assessment

| Chapter | Theory | Code | Engineering | Practice | Notes |
|---------|--------|------|-------------|----------|-------|
| 01 | X/5 | X/5 | X/5 | X/5 | ... |
| 02 | X/5 | X/5 | X/5 | X/5 | ... |

## Problem List

### P0-01: [Title]
- **File**: path/to/file
- **Problem**: description
- **Impact**: what happens if unfixed
- **Suggestion**: how to fix

### P1-01: [Title]
...

## Strengths
1. ...

## Depth Gaps

### Theory Gaps
- [ ] Chapter X: Missing problem introduction
- [ ] Chapter Y: Mathematical derivation incomplete

### Code Gaps
- [ ] Chapter X: Missing shape annotations
- [ ] Chapter Y: No debugging process shown

### Engineering Gaps
- [ ] Chapter X: No performance analysis
- [ ] Chapter Y: Missing common pitfalls

## Numerical Verification
| Claim | Expected | Actual | Match? |
|-------|----------|--------|--------|
| ... | ... | ... | ✅/❌ |

## Recommendations
1. Priority 1: [Most important improvement]
2. Priority 2: [Second most important]
3. Priority 3: [Third most important]
```

## Verification Commands

```bash
# Run all scripts
for s in courses/PartX/scripts/*.py; do
  MPLBACKEND=Agg python3 "$s" && echo "✅ $s" || echo "❌ $s"
done

# Run assignment tests
python3 -m pytest assignments/assignment_X/test_xxx.py -v

# Check for common inconsistencies
grep -rn "\.sum()" courses/PartX/tutorial/
grep -rn "\.mean()" courses/PartX/scripts/

# Check for depth issues
grep -rn "学习目标" courses/PartX/tutorial/*.md  # Should exist in every chapter
grep -rn "常见陷阱" courses/PartX/tutorial/*.md  # Should exist in every chapter
grep -rn "性能分析" courses/PartX/tutorial/*.md  # Should exist in every chapter

# Check for shape annotations
grep -rn "shape" courses/PartX/tutorial/*.md  # Should have many references
grep -rn "形状" courses/PartX/tutorial/*.md  # Should have many references
```

## Quality Gates

Before approving a tutorial for release, verify:

### Gate 1: Completeness
- [ ] All chapters have learning objectives
- [ ] All chapters have prerequisites
- [ ] All chapters have "学完本章你能..." section
- [ ] All chapters have practice questions
- [ ] All chapters have references

### Gate 2: Depth
- [ ] Theory depth ≥ 3/5 for all chapters
- [ ] Code depth ≥ 3/5 for all chapters
- [ ] Engineering depth ≥ 3/5 for all chapters
- [ ] Practice depth ≥ 3/5 for all chapters

### Gate 3: Accuracy
- [ ] All scripts run without errors
- [ ] All tests pass
- [ ] All cross-references are valid
- [ ] No fabricated content

### Gate 4: Consistency
- [ ] Code in tutorial matches scripts
- [ ] Variable names are consistent
- [ ] Style is consistent across chapters
- [ ] Navigation links work

## Common Review Findings

### Theory Depth Issues
1. **Missing problem introduction**: Jumping straight to solution without explaining the problem
2. **Missing motivation**: Not explaining why this technology exists
3. **Missing derivation**: Stating formulas without showing how they're derived
4. **Missing historical context**: No mention of key papers or evolution
5. **Missing intuition**: No analogies or visual explanations

### Code Explanation Issues
1. **Missing shape annotations**: Tensor operations without shape comments
2. **Missing line-by-line explanation**: Code blocks without comments
3. **Missing purpose statements**: No explanation of what code blocks do
4. **Missing debugging process**: No common errors shown
5. **Hypothetical outputs**: Using made-up outputs instead of real ones

### Engineering Practice Issues
1. **Missing performance analysis**: No time/space complexity
2. **Missing common pitfalls**: No FAQ-style pitfall section
3. **Missing best practices**: No industrial approach mentioned
4. **Missing configuration recommendations**: No guidance on settings

### Learning Outcome Issues
1. **Vague objectives**: "Understand X" instead of "Be able to explain X"
2. **Missing acceptance criteria**: Exercises without clear success criteria
3. **Missing thinking questions**: No questions promoting deeper exploration
4. **Missing cross-references**: No connection to other parts
