#!/usr/bin/env python3
"""
从 Jupyter Notebook (.ipynb) 中提取所有 base64 编码的图片，保存为 PNG 文件。

用法：
    python extract_images.py <notebook_path> <output_dir>

示例：
    python extract_images.py ../courses/Part1_bigrams/makemore_part1_bigrams.ipynb ../courses/Part1_bigrams/images/
"""

import sys
import os
import json
import base64
import re


def extract_images(notebook_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    count = 0
    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue

        for output_idx, output in enumerate(cell.get('outputs', [])):
            # 方式1: output data 中直接包含 image/png
            if 'data' in output and 'image/png' in output['data']:
                b64_data = output['data']['image/png']
                # 可能是字符串或列表
                if isinstance(b64_data, list):
                    b64_data = ''.join(b64_data)
                # 去掉可能的换行和空格
                b64_data = b64_data.strip()

                filename = f"cell{cell_idx:03d}_output{output_idx:02d}.png"
                filepath = os.path.join(output_dir, filename)
                img_bytes = base64.b64decode(b64_data)
                with open(filepath, 'wb') as imgf:
                    imgf.write(img_bytes)
                count += 1
                print(f"  ✓ {filename} ({len(img_bytes)} bytes)")

            # 方式2: display_data 类型
            elif output.get('output_type') == 'display_data':
                if 'data' in output and 'image/png' in output['data']:
                    b64_data = output['data']['image/png']
                    if isinstance(b64_data, list):
                        b64_data = ''.join(b64_data)
                    b64_data = b64_data.strip()

                    filename = f"cell{cell_idx:03d}_display{output_idx:02d}.png"
                    filepath = os.path.join(output_dir, filename)
                    img_bytes = base64.b64decode(b64_data)
                    with open(filepath, 'wb') as imgf:
                        imgf.write(img_bytes)
                    count += 1
                    print(f"  ✓ {filename} ({len(img_bytes)} bytes)")

    return count


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(notebook_path):
        print(f"错误: 找不到 notebook 文件: {notebook_path}")
        sys.exit(1)

    print(f"提取图片: {notebook_path}")
    print(f"输出目录: {output_dir}")
    print()

    count = extract_images(notebook_path, output_dir)

    print()
    print(f"完成! 共提取 {count} 张图片到 {output_dir}")


if __name__ == '__main__':
    main()
