# Tutorial Writing Guide

## Structure

Each lesson's tutorial lives in `courses/PartX/tutorial/`:

```
tutorial/
├── README.md           # Chapter navigation + prerequisites
├── 01_introduction.md  # First chapter
├── 02_core_concept.md  # Second chapter
└── 03_advanced.md      # Third chapter (if needed)
```

### When to Split

- **< 3000 lines**: single `tutorial.md` is fine
- **3000+ lines**: split into 2-4 chapter files + `README.md`
- Rule of thumb: 3 files per lesson works well (intro / core / advanced)

## Chapter Template

```markdown
# Part X - Chapter N: Title

## Prerequisites
What the reader should already know (link to previous chapters/lessons).

## What You'll Learn
3-5 bullet points of concrete outcomes.

## Core Content

### Section 1
... explanation with code examples ...

> 💡 **Insight**: Why this matters conceptually.

> ⚠️ **Pitfall**: Common mistake and how to avoid it.

> 🔑 **Key Concept**: Term — one-sentence definition.

### Section 2
...

**Practice Questions**
<details>
<summary>Q: Question text?</summary>
A: Answer with explanation.
</details>

## Exercises
2-3 practice problems with progressive difficulty.

## What's Next
Brief preview of next chapter → link to `02_xxx.md`.
```

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
2. **Emoji markers**: consistent use of 💡 (insight), ⚠️ (pitfall), 🔑 (key concept)
3. **Code blocks**: always specify language (`python`, `bash`)
4. **Diagrams**: prefer ASCII art over Mermaid (more reliable rendering)
5. **No fabrications**: if unsure about a claim, search and cross-validate before including

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

## FAQ: Common Mistakes Found in Reviews

1. **Tutorial code ≠ script code**: always copy from the actual runnable script, don't write from memory
2. **Parameter counts**: calculate and verify; don't guess
3. **Sample outputs**: run the actual script and paste real output
4. **Variable names**: keep consistent between tutorial inline code and scripts
5. **Self-correction residue**: remove any "wait, let me rephrase" artifacts before publishing
