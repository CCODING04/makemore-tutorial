# Scripts Generation Guide

## Script Template

```python
#!/usr/bin/env python3
"""
Part X - Script N: Short description
Goal: One sentence explaining what this script demonstrates
"""

import torch
import torch.nn.functional as F
import os

def main():
    # Data path (resolve relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', '<datafile>')

    # Load data
    with open(data_path, 'r') as f:
        data = f.read().splitlines()

    # === Core content of this script ===
    # ... progressive evolution from previous script ...

if __name__ == '__main__':
    main()
```

## Progressive Design Pattern

Scripts must be **cumulative**, not rewritten:

```
01_explore_data.py       # Load data, basic stats
02_build_dataset.py      # Add: dataset construction (builds on 01)
03_model_forward.py      # Add: model forward pass (builds on 02)
04_training_loop.py      # Add: training loop (builds on 03)
05_evaluation.py         # Add: evaluation metrics (builds on 04)
06_visualization.py      # Add: visualizations (builds on 05)
07_sampling.py           # Add: generation/sampling (builds on 05)
```

Key principle: scripts 06 and 07 should share training logic with 05 rather than duplicating it.

## Data Path Rules

The depth depends on file location:

| File location | Relative path to data/ |
|---|---|
| `courses/PartX/scripts/` | `os.path.join(script_dir, '..', '..', '..', 'data', 'file')` |
| `assignments/assignment_X/` | `os.path.join(_THIS_DIR, '..', '..', 'data', 'file')` |

**Always** use `os.path.dirname(os.path.abspath(__file__))` — never rely on cwd.

## Number of Scripts

- Target: 5-7 scripts per lesson
- Start with data exploration, end with the most advanced concept
- Each script should take < 30 seconds to run (for quick iteration)

## Matplotlib Scripts

For scripts generating plots:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ... plotting code ...

save_path = os.path.join(script_dir, 'output.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {save_path}")
```

**Pitfall**: Don't use Chinese characters in matplotlib titles/labels — use English for charts, Chinese for tutorial prose.

## Shape Annotation Requirements (NEW - Critical for learning)

Every tensor operation should have shape comments:

```python
# ❌ Bad: No shape information
x = torch.randn(32, 128, 512)
q = x @ W_q
q = q.view(32, 128, 8, 64)
q = q.transpose(1, 2)

# ✅ Good: Shape annotations throughout
# x shape: (batch=32, seq_len=128, d_model=512)
x = torch.randn(32, 128, 512)

# Q = x @ W_q
# x: (32, 128, 512), W_q: (512, 512) → q: (32, 128, 512)
q = x @ W_q

# Reshape for multi-head attention
# q: (32, 128, 512) → (32, 128, 8, 64) [8 heads, d_k=64]
q = q.view(32, 128, 8, 64)

# Transpose for attention computation
# q: (32, 128, 8, 64) → (32, 8, 128, 64) [heads first]
q = q.transpose(1, 2)
```

### Shape Annotation Pattern

For complex operations, show the full shape derivation:

```python
# Shape derivation:
# Input:  (batch, seq_len, d_model)
# Linear: (batch, seq_len, d_model) @ (d_model, d_k) → (batch, seq_len, d_k)
# Reshape: (batch, seq_len, d_k) → (batch, seq_len, heads, d_k // heads)
# Transpose: (batch, seq_len, heads, d_k // heads) → (batch, heads, seq_len, d_k // heads)

# Implementation:
# x shape: (32, 128, 512)
x = torch.randn(32, 128, 512)

# W_q shape: (512, 64), q shape: (32, 128, 64)
q = x @ W_q

# Reshape: (32, 128, 64) → (32, 128, 8, 8) [8 heads, d_k=8]
q = q.view(32, 128, 8, 8)

# Transpose: (32, 128, 8, 8) → (32, 8, 128, 8) [heads first for attention]
q = q.transpose(1, 2)
```

## Debug Output Requirements (NEW - Critical for verification)

Every script should print intermediate results for verification:

```python
# ❌ Bad: Silent execution
def train_model():
    for epoch in range(10):
        loss = train_step()
    return model

# ✅ Good: Debug output for verification
def train_model():
    for epoch in range(10):
        loss = train_step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    # Print final results
    print(f"\nTraining complete!")
    print(f"Final loss: {loss:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
```

### Debug Output Pattern

```python
# Print shape information for verification
print(f"Input shape: {x.shape}")  # Expected: (32, 128, 512)
print(f"Output shape: {q.shape}")  # Expected: (32, 8, 128, 64)

# Print value ranges for verification
print(f"Value range: [{q.min():.4f}, {q.max():.4f}]")
print(f"Mean: {q.mean():.4f}, Std: {q.std():.4f}")

# Print intermediate results for debugging
print(f"\nStep 1: Linear projection")
print(f"  q shape: {q.shape}")
print(f"  q stats: mean={q.mean():.4f}, std={q.std():.4f}")

print(f"\nStep 2: Reshape for multi-head")
print(f"  q shape: {q.shape}")
print(f"  q[0,0,0,:5] = {q[0,0,0,:5].tolist()}")
```

## Common Pitfalls (from Real Experience)

1. **Path resolution**: exec runs from project root, not script dir. Always use `__file__`-relative paths.

2. **Random seeds**: Each script should set its own seed for reproducibility, but don't assume specific numeric outputs across environments.

3. **Global variables**: Wrap everything in `main()` to avoid namespace pollution.

4. **Import consistency**: Every script must have its own imports at the top, even if "obviously" available.

5. **Missing shape annotations**: Tensor operations without shape comments make code hard to follow.

6. **Missing debug output**: Silent scripts make it hard to verify results.

7. **Hypothetical outputs**: Using made-up outputs instead of real ones from script execution.

## Script Quality Checklist

Before finalizing a script, verify:

- [ ] **Self-contained**: Runs independently with own imports
- [ ] **Progressive**: Builds on previous script
- [ ] **Shape annotations**: Every tensor operation has shape comments
- [ ] **Debug output**: Intermediate results are printed
- [ ] **Real outputs**: Script actually runs and produces output
- [ ] **Error handling**: Common errors are handled gracefully
- [ ] **Documentation**: Docstring explains purpose
- [ ] **Data paths**: Uses `__file__`-relative paths

## Example: Well-Documented Script

```python
#!/usr/bin/env python3
"""
Part 6 - Script 2: Multi-Head Attention
Goal: Implement multi-head attention with shape tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    batch_size = 32
    seq_len = 128
    d_model = 512
    num_heads = 8
    d_k = d_model // num_heads  # 64
    
    print("=" * 60)
    print("Multi-Head Attention Implementation")
    print("=" * 60)
    
    # Step 1: Create input tensor
    # x shape: (batch=32, seq_len=128, d_model=512)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nStep 1: Input tensor")
    print(f"  x shape: {x.shape}")  # Expected: (32, 128, 512)
    print(f"  x stats: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Step 2: Linear projections for Q, K, V
    # W_q shape: (d_model, d_model) = (512, 512)
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)
    
    # Q = x @ W_q
    # x: (32, 128, 512), W_q: (512, 512) → Q: (32, 128, 512)
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    
    print(f"\nStep 2: Linear projections")
    print(f"  Q shape: {Q.shape}")  # Expected: (32, 128, 512)
    print(f"  K shape: {K.shape}")  # Expected: (32, 128, 512)
    print(f"  V shape: {V.shape}")  # Expected: (32, 128, 512)
    
    # Step 3: Reshape for multi-head attention
    # Q: (32, 128, 512) → (32, 128, 8, 64) → (32, 8, 128, 64)
    Q = Q.view(batch_size, seq_len, num_heads, d_k)
    Q = Q.transpose(1, 2)
    
    K = K.view(batch_size, seq_len, num_heads, d_k)
    K = K.transpose(1, 2)
    
    V = V.view(batch_size, seq_len, num_heads, d_k)
    V = V.transpose(1, 2)
    
    print(f"\nStep 3: Reshape for multi-head")
    print(f"  Q shape: {Q.shape}")  # Expected: (32, 8, 128, 64)
    print(f"  K shape: {K.shape}")  # Expected: (32, 8, 128, 64)
    print(f"  V shape: {V.shape}")  # Expected: (32, 8, 128, 64)
    
    # Step 4: Compute attention scores
    # Q @ K^T: (32, 8, 128, 64) @ (32, 8, 64, 128) → (32, 8, 128, 128)
    attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    print(f"\nStep 4: Attention scores")
    print(f"  attn_scores shape: {attn_scores.shape}")  # Expected: (32, 8, 128, 128)
    print(f"  attn_scores stats: mean={attn_scores.mean():.4f}, std={attn_scores.std():.4f}")
    
    # Step 5: Apply softmax
    # attn_weights: (32, 8, 128, 128) - attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    print(f"\nStep 5: Attention weights")
    print(f"  attn_weights shape: {attn_weights.shape}")  # Expected: (32, 8, 128, 128)
    print(f"  attn_weights sum: {attn_weights[0, 0, 0, :].sum():.4f}")  # Expected: 1.0
    
    # Step 6: Apply attention to values
    # attn_weights @ V: (32, 8, 128, 128) @ (32, 8, 128, 64) → (32, 8, 128, 64)
    attn_output = attn_weights @ V
    
    print(f"\nStep 6: Attention output")
    print(f"  attn_output shape: {attn_output.shape}")  # Expected: (32, 8, 128, 64)
    
    # Step 7: Concatenate heads
    # attn_output: (32, 8, 128, 64) → (32, 128, 8, 64) → (32, 128, 512)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_model)
    
    print(f"\nStep 7: Concatenate heads")
    print(f"  attn_output shape: {attn_output.shape}")  # Expected: (32, 128, 512)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input:  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print(f"Output: (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print(f"Parameters: {num_heads} heads, d_k={d_k}")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

## Script Review Checklist

When reviewing scripts, verify:

- [ ] **Self-contained**: Runs without external dependencies
- [ ] **Progressive**: Builds on previous script
- [ ] **Shape annotations**: Every tensor operation has shape comments
- [ ] **Debug output**: Intermediate results are printed
- [ ] **Real outputs**: Script actually runs and produces output
- [ ] **Error handling**: Common errors are handled gracefully
- [ ] **Documentation**: Docstring explains purpose
- [ ] **Data paths**: Uses `__file__`-relative paths
- [ ] **Random seeds**: Set for reproducibility
- [ ] **Import consistency**: All imports at top of file
