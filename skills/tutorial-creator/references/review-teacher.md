# Teacher Review Guide

## Review Scope

As a teacher/reviewer, evaluate each lesson for **accuracy**, **consistency**, and **pedagogical quality**.

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

### 2. Code-Tutorial Consistency

- [ ] Inline code in tutorial matches actual script implementations
- [ ] Variable names are consistent (tutorial uses same names as scripts)
- [ ] Function signatures in assignment hints match skeleton code
- [ ] Numeric examples in tutorial match real script outputs

**Common issues found in practice:**
- Tutorial shows `.sum()` but script uses `.mean()`
- Tutorial claims "1.3万 parameters" but actual count is 3,481
- Tutorial uses class-based layer interface, scripts use dict-based
- Tutorial Kaiming init missing gain factor (e.g., `5/3` for tanh)

### 3. Cross-Lesson Continuity

- [ ] Each lesson opens by connecting to previous lesson's results
- [ ] Code style is consistent across lessons (e.g., character mapping)
- [ ] Difficulty progression is smooth (no sudden jumps)
- [ ] Prerequisites are correctly stated

### 4. Assignment Quality

- [ ] Hint code compiles and has correct function signatures
- [ ] Tests verify meaningful properties (not trivially passable)
- [ ] Stretch exercises have graceful skip when unimplemented
- [ ] TODO hints are step-level (helpful but not giving away answers)

### 5. Script Quality

- [ ] All scripts run without errors
- [ ] Each script is self-contained (independent execution)
- [ ] Progressive structure is maintained (later scripts build on earlier)
- [ ] No duplicated training code where shared logic would be better

## Review Output Format

```markdown
# Review: Part X

## Overall Score: X.X/10

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

## Numerical Verification
| Claim | Expected | Actual | Match? |
|-------|----------|--------|--------|
| ... | ... | ... | ✅/❌ |
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
```
