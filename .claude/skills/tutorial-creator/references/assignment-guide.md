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

    Acceptance Criteria:
        - result.shape == (expected_shape)
        - result.dtype == torch.float32
        - result.sum() ≈ expected_value (within tolerance)
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

4. **Inconsistent function signatures**: `assignment.md`, `xxx_exercises.py`, and `test_xxx.py` must all agree on parameter names and order.

## Thinking Questions (Enhanced)

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

**Q3：** [Question about design choices]

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
- **Application**: "Where would you use this in practice?"

**Rules**:
- Use `<details>` tags so students can attempt before seeing the answer
- Provide the answer, not just a hint
- Connect to real-world applications when possible
- Include at least one question that requires deeper thinking

## Acceptance Criteria (NEW)

Each exercise should have clear acceptance criteria:

```python
def exercise_1_basic():
    """
    Title of exercise.

    Returns:
        result (torch.Tensor): description of expected output

    Acceptance Criteria:
        - result.shape == (expected_shape)
        - result.dtype == torch.float32
        - result.sum() ≈ expected_value (within tolerance)
        - result.min() >= 0 and result.max() <= 1
    """
    # TODO: Implement
    return None
```

### Acceptance Criteria Pattern

```markdown
## 验收标准

### 练习 1：[标题]
- [ ] 返回值形状正确：`(expected_shape)`
- [ ] 返回值类型正确：`torch.float32`
- [ ] 数学性质正确：`result.sum() ≈ expected_value`
- [ ] 值范围正确：`0 <= result <= 1`

### 练习 2：[标题]
- [ ] 损失函数下降：`final_loss < initial_loss`
- [ ] 梯度更新正确：`model.parameters()` 有变化
- [ ] 模型输出合理：`output.shape == (batch, vocab_size)`
```

## Assignment Quality Checklist

Before finalizing an assignment, verify:

- [ ] **Clear problem statement**: Each exercise has a clear description
- [ ] **Step-level hints**: TODO comments provide guidance without giving away answers
- [ ] **Acceptance criteria**: Each exercise has clear success criteria
- [ ] **Test independence**: Tests don't depend on each other
- [ ] **Fixed seeds**: Tests use fixed random seeds for reproducibility
- [ ] **Shape/dtype tests**: Tests verify mathematical properties, not exact values
- [ ] **Stretch graceful skip**: Stretch exercises skip if not implemented
- [ ] **Thinking questions**: 3-5 questions with `<details>` answers
- [ ] **Function signatures**: Consistent across assignment.md, exercises.py, and test.py
- [ ] **Import completeness**: All necessary imports are included

## Example: Well-Designed Assignment

```markdown
# Assignment 6: Transformer 实现

## 练习 1：多头注意力实现

实现多头注意力机制，包括：
1. 线性投影生成 Q, K, V
2. 缩放点积注意力
3. 多头拼接与输出投影

**验收标准**：
- [ ] 输出形状正确：`(batch, seq_len, d_model)`
- [ ] 注意力权重和为 1：`attn_weights.sum(dim=-1) ≈ 1`
- [ ] 梯度可以反向传播

## 练习 2：Transformer Block 实现

基于练习 1 的多头注意力，实现完整的 Transformer Block：
1. 多头注意力 + 残差连接 + LayerNorm
2. FFN + 残差连接 + LayerNorm

**验收标准**：
- [ ] 输出形状与输入相同
- [ ] 残差连接正确实现
- [ ] LayerNorm 正确应用

## 🤔 思考题

**Q1：** 为什么注意力分数要除以 √d_k？

<details>
<summary>💡 提示</summary>

当 d_k 较大时，点积 QK^T 的方差为 d_k · Var(q_i k_i)，除以 √d_k 将分布标准化，
避免 softmax 进入梯度饱和区。

</details>

**Q2：** 为什么用 LayerNorm 而不是 BatchNorm？

<details>
<summary>💡 提示</summary>

BatchNorm 对 batch 维度归一化，依赖 batch 统计量；LayerNorm 对特征维度归一化，
不依赖 batch 大小，更适合序列数据和小 batch 训练。

</details>
```

## Assignment Review Checklist

When reviewing assignments, verify:

- [ ] **Problem clarity**: Each exercise has a clear description
- [ ] **Hint quality**: Hints guide without giving away answers
- [ ] **Acceptance criteria**: Each exercise has clear success criteria
- [ ] **Test quality**: Tests verify meaningful properties
- [ ] **Test independence**: Tests don't depend on each other
- [ ] **Stretch handling**: Stretch exercises skip gracefully
- [ ] **Thinking questions**: Questions promote deeper understanding
- [ ] **Signature consistency**: Function signatures match across files
- [ ] **Import completeness**: All necessary imports are included
- [ ] **Data paths**: Correct relative paths to data files
