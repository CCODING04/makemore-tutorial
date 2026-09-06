#!/usr/bin/env python3
"""检查 markdown 中的 LaTeX 排版问题（渲染兼容性导向）"""
import re, sys

path = sys.argv[1]
lines = open(path, encoding='utf-8').read().splitlines()
issues = []

# 去掉代码块后再分析行内/行间公式
in_code = False
cleaned = []  # (lineno, line) 非代码块行
fence_lines = []
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith('```'):
        in_code = not in_code
        fence_lines.append(i)
        continue
    if not in_code:
        cleaned.append((i, ln))

# 1) 多行 $$ 块检测：$$ 开但同行未闭合
open_at = None
for i, ln in cleaned:
    n = ln.count('$$')
    if open_at is None:
        # 找开块
        if n == 1 and ln.strip().startswith('$$') and not ln.strip().endswith('$$') or \
           (n % 2 == 1 and not (ln.strip().startswith('$$') and ln.strip().endswith('$$') and n >= 2)):
            if n % 2 == 1:
                open_at = i
                issues.append((i, 'MULTILINE_$$', f'display 公式块跨行（自本行起未闭合）: {ln[:60]}'))
    else:
        if '$$' in ln:
            open_at = None

# 2) 逐行：提取 $...$ 片段，检查公式内中文 / \text{非ASCII}
cn = re.compile(r'[\u4e00-\u9fff]')
for i, ln in cleaned:
    # 去掉 $$ 后按 $ 配对提取
    s = ln.replace('$$', '$')
    parts = s.split('$')
    # 偶数索引是公式内容（若 $ 配对）
    if len(parts) % 2 == 0:
        issues.append((i, 'UNBALANCED_$', f'本行 $ 数量为奇数（{len(parts)-1} 个），配对断裂: {ln[:70]}'))
    for j in range(1, len(parts), 2):
        math = parts[j]
        if cn.search(math):
            issues.append((i, 'CHINESE_IN_MATH', f'公式内含中文: ${math[:50]}$'))
        for m in re.finditer(r'\\text\{([^}]*)\}', math):
            if cn.search(m.group(1)):
                issues.append((i, 'TEXT_WITH_CJK', f'\\text{{}} 包含中文: \\text{{{m.group(1)[:20]}}}'))
        for bad in [r'\\big', r'\{=\}']:
            pass

# 3) 列表/引用块内的 display 公式（缩进 $$ 或 > 后 $$）
for i, ln in cleaned:
    if re.match(r'^\s{2,}\$\$', ln):
        issues.append((i, 'INDENTED_$$', f'列表/缩进内的 display 公式，兼容性差: {ln[:60]}'))
    if ln.lstrip().startswith('>') and '$$' in ln:
        issues.append((i, 'BLOCKQUOTE_$$', f'引用块内的 display 公式，兼容性差: {ln[:60]}'))

# 4) $$ 与正文同段（$$ 前后缺空行）
for k, (i, ln) in enumerate(cleaned):
    if ln.strip().startswith('$$') and k > 0:
        prev = cleaned[k-1][1]
        if prev.strip() and not prev.strip().startswith('$$'):
            issues.append((i, '$$_NO_BLANK_BEFORE', f'$$ 块前缺空行（部分渲染器会并入段落）: {ln[:50]}'))

if not issues:
    print('✅ 未发现问题')
else:
    cur = None
    for i, kind, msg in issues:
        if kind != cur:
            print(f'\n── {kind} ──')
            cur = kind
        print(f'L{i}: {msg}')
    print(f'\n共 {len(issues)} 处问题')
