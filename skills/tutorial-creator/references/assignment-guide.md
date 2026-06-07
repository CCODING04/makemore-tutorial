# Assignment Design Guide

## File Structure

```
assignments/assignment_X/
├── assignment.md          # Problem description (4 core + 1 stretch)
├── xxx_exercises.py       # TODO skeleton with step-level hints
└── test_xxx.py            # Auto-test (pytest-compatible)
```

## Exercise Design

### Difficulty Progression

| # | Type | Description |
|---|------|-------------|
| 1 | Basic | Implement a core component (data loading, basic calculation) |
| 2 | Core | Build on exercise 1 (model forward pass, loss computation) |
| 3 | Applied | Combine concepts (training loop, evaluation) |
| 4 | Integration | Full pipeline (end-to-end training + evaluation) |
| 5 | 🌟 Stretch | Open-ended challenge (optional, gracefully skipped) |

### Stretch Exercise Pattern

Stretch exercises should return `None` by default, and tests should skip gracefully:

```python
def stretch_exercise(...):
    """🌟 Stretch goal: description"""
    # TODO: your implementation
    return None
```

```python
def test_stretch():
    result = stretch_exercise(...)
    if result is None:
        pytest.skip("Stretch exercise not implemented")
    # ... actual assertions ...
```

## TODO Skeleton Template

```python
"""
Assignment X: Title
Based on Part X of the tutorial series.
"""

import torch
import torch.nn.functional as F
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')


def exercise_1_basic():
    """
    Title of exercise.

    Returns:
        result (torch.Tensor): description of expected output

    Steps:
        1. Load data from _DATA_PATH
        2. Process into required format
        3. Return the result tensor

    Hint:
        Use torch.Tensor operations, check shapes carefully.
    """
    # TODO: Implement
    return None
```

## Test Design Principles

### What to Test

| Dimension | Example |
|-----------|---------|
| **Shape** | `assert result.shape == (N, vocab_size)` |
| **Dtype** | `assert result.dtype == torch.float32` |
| **Mathematical invariants** | `assert torch.allclose(result.sum(dim=1), torch.ones(N))` |
| **Value range** | `assert result.min() >= 0 and result.max() <= 1` |
| **Improvement** | `assert trained_loss < initial_loss` |

### What NOT to Test

- ❌ Exact floating-point values (training is stochastic)
- ❌ Specific tensor element values (unless deterministic)
- ❌ Specific loss values at specific steps (platform-dependent)

### Fixed Seed Strategy

```python
def test_exercise():
    torch.manual_seed(42)
    result = exercise_function(...)
    # Test properties, not exact values
    assert result.shape == expected_shape
    assert result.dtype == torch.float32
```

### Test Function Independence

Each test function must be independent — one failure should not cascade:

```python
def test_exercise_1():
    # Creates its own data, doesn't depend on other exercises
    ...

def test_exercise_2():
    # Creates its own data, independently tests exercise 2
    ...
```

## Data Path Convention

```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')
```

Assignment files are 2 levels deep from project root (assignments/assignment_X/), so data is at `../../data/`.

## Common Issues (from Real Experience)

1. **Hint code bugs**: verify that hint code in `assignment.md` actually matches function signatures in the skeleton. A wrong parameter in a hint is a P0 bug.

2. **Overly generous test thresholds**: `loss < 2.5` might pass with random weights. Calibrate by running with both correct and intentionally broken implementations.

3. **Missing imports**: the skeleton must include all necessary imports, even `os` for path resolution.

## Thinking Questions

Each assignment should include 3-5 thinking questions after the exercises to deepen understanding:

```markdown
## 🤔 思考题

**Q1：** [Question about a key concept]

<details>
<summary>💡 提示</summary>

[Answer with explanation]

</details>

**Q2：** [Question about a common pitfall]

<details>
<summary>💡 提示</summary>

[Answer with explanation]

</details>
```

**Question Types**:
- **Conceptual**: "Why does X work this way?"
- **Comparative**: "What's the difference between A and B?"
- **Debugging**: "What happens if we change X?"
- **Extension**: "How would this scale to larger models?"

**Rules**:
- Use `<details>` tags so students can attempt before seeing the answer
- Provide the answer, not just a hint
- Connect to real-world applications when possible

4. **Inconsistent function signatures**: `assignment.md`, `xxx_exercises.py`, and `test_xxx.py` must all agree on parameter names and order.
