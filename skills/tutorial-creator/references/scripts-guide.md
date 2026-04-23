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

## Common Pitfalls (from Real Experience)

1. **Path resolution**: exec runs from project root, not script dir. Always use `__file__`-relative paths.
2. **Random seeds**: Each script should set its own seed for reproducibility, but don't assume specific numeric outputs across environments.
3. **Global variables**: Wrap everything in `main()` to avoid namespace pollution.
4. **Import consistency**: Every script must have its own imports at the top, even if "obviously" available.
