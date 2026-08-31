#!/usr/bin/env python3
"""
Script Template — Starting point for tutorial scripts.
Goal: [ONE LINE DESCRIPTION]
"""

import torch
import torch.nn.functional as F
import os


def main():
    # === Data Path (always resolve relative to this script) ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')

    # === Load Data ===
    with open(data_path, 'r') as f:
        words = f.read().splitlines()

    print(f"Loaded {len(words)} words")
    print(f"Example: {words[:3]}")

    # === Core Content ===
    # ... implement here ...


if __name__ == '__main__':
    main()
