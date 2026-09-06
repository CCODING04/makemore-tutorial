#!/usr/bin/env python3
"""教材链接检查（合并视图版）：REVIEW 覆盖 REPO；相对链接在合并树上解析。
范围限定 courses/ 与 assignments/（教材本体）。用法：python check_links.py [REPO] [REVIEW]"""
import os, re, sys

repo = sys.argv[1] if len(sys.argv) > 1 else '/home/admin02/Code/WorkSpace/makemore-tutorial'
review = sys.argv[2] if len(sys.argv) > 2 else '/home/admin02/Code/WorkSpace/makemore-tutorial-review'
FENCE = re.compile(r'```.*?```', re.S)
INLINE = re.compile(r'`[^`]*`')

def merged(rel):
    for base in (review, repo):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None

def md_files():
    seen = set()
    for tree in ('courses', 'assignments'):
        for base in (review, repo):
            root = os.path.join(base, tree)
            if not os.path.isdir(root):
                continue
            for dp, _, fs in os.walk(root):
                if '__pycache__' in dp or '.pytest_cache' in dp:
                    continue
                for f in fs:
                    if f.endswith('.md'):
                        rel = os.path.relpath(os.path.join(dp, f), base)
                        if rel not in seen:
                            seen.add(rel)
                            yield rel

bad, checked = [], 0
for rel in sorted(md_files()):
    actual = merged(rel)
    text = FENCE.sub('', open(actual, encoding='utf-8').read())   # 去围栏代码块
    text = INLINE.sub('', text)                                   # 去行内代码（防 r['x']('y') 误报）
    for m in re.finditer(r'\]\(([^)#\s][^)]*)\)', text):
        link = m.group(1).split('#')[0]   # 文件存在即通过；锚点不校验（GitHub slug 规则复杂）
        if not link or link.startswith(('http://', 'https://', '/')):
            continue
        checked += 1
        cand = os.path.normpath(os.path.join(os.path.dirname(rel), link))
        if not merged(cand):
            bad.append(f'{rel} -> {link}')
print(f'检查 {checked} 个相对链接（courses+assignments 合并视图），坏链 {len(bad)} 个')
for b in bad:
    print('  ❌', b)
sys.exit(1 if bad else 0)
