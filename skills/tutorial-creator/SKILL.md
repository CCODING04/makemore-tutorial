---
name: tutorial-creator
description: >
  Create comprehensive, self-contained tutorial courses from video lectures/notebooks.
  Generates three parallel outputs per lesson: progressive Python scripts, tutorial Markdown,
  and assignments with auto-tests. Includes a dual-perspective (teacher + student) quality
  review system. Use when asked to "create a tutorial", "build a course", "convert lectures
  to tutorials", "generate educational content", or "review tutorial quality". Also applies
  to tasks like "make a tutorial from this video/notebook/transcript".
---

# Tutorial Creator

Convert video lectures or notebooks into a complete, self-contained tutorial course.

## Overview

Each lesson produces three deliverables via parallel sub-agents:

```
Phase A: 3 parallel sub-agents
├── Scripts    → progressive, runnable Python files
├── Tutorial   → Markdown chapters with code, diagrams, exercises
└── Assignment → TODO skeleton + auto-test with fixed seeds

Phase B: Verification
├── Run all scripts (MPLBACKEND=Agg for matplotlib)
├── Run assignment tests
├── Check cross-references (paths, code sync)
└── Dual-perspective review (teacher + student)
```

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
│       └── tutorial/          # 01_*.md, 02_*.md, ..., README.md
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

### Sub-agent 2: Tutorial

Read `references/tutorial-guide.md` for detailed guidance. Key rules:

1. Split into 2-4 chapter files + README.md if content > 3000 lines
2. Reference scripts as `../scripts/01_xxx.py` and images as `../images/xxx.png`
3. Use emoji markers: 💡 insight, ⚠️ pitfall, 🔑 key concept
4. Prefer ASCII art over Mermaid (more reliable cross-platform)
5. Include 2-3 QA exercises per chapter in `<details>` tags
6. **Accuracy**: search and cross-validate any uncertain claims — never fabricate

### Sub-agent 3: Assignment

Read `references/assignment-guide.md` for detailed guidance. Key rules:

1. 4 core exercises + 1 stretch (marked 🌟)
2. Fixed random seeds (`torch.manual_seed(42)`) for reproducibility
3. Tests check **shape, dtype, mathematical invariants** — not exact values
4. Stretch exercise gracefully skipped if returns `None`
5. Provide detailed TODO comments (step-level, almost pseudocode)

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
- Rate: P0 (must fix) / P1 (should fix) / P2 (nice to have)

**Student Review** — read `references/review-student.md`:
- Learning curve (where will students get stuck?)
- Explanation clarity (broadcasting, tensor ops, new concepts)
- Exercise difficulty progression
- "If I could only fix 3 things" prioritization

### Step 5: Apply Fixes
Process review results in priority order:
1. Fix all P0 issues
2. Fix P1 issues
3. Re-run verification

## Per-Lesson Time Budget

| Step | Estimated Time |
|------|---------------|
| 3 parallel sub-agents | ~8-10 min |
| Verification + fixes | ~5-8 min |
| Dual review | ~10-15 min |
| Apply review fixes | ~5-10 min |
| **Total per lesson** | **~30-45 min** |
