#!/usr/bin/env python3
"""章末指引同步校验（G17）：教程提到的「作业题 N」必须存在于作业分值表。
用法：python check_assignment_refs.py [REPO] [REVIEW]"""
import os, re, sys

repo = sys.argv[1] if len(sys.argv) > 1 else '/home/admin02/Code/WorkSpace/makemore-tutorial'
review = sys.argv[2] if len(sys.argv) > 2 else '/home/admin02/Code/WorkSpace/makemore-tutorial-review'

def merged(rel):
    for base in (review, repo):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None

problems = []
for n in range(1, 20):
    tut_dir = None
    for d in sorted(os.listdir(os.path.join(review, 'courses')) if os.path.isdir(os.path.join(review, 'courses')) else []) + \
             sorted(os.listdir(os.path.join(repo, 'courses'))):
        if re.match(f'Part{n}_', d):
            cand = merged(f'courses/{d}/tutorial')
            tut_dir = cand
            break
    if not tut_dir:
        continue
    mentioned = {}   # 题号 -> 首次出现文件
    for f in sorted(os.listdir(tut_dir)):
        if not f.endswith('.md'):
            continue
        text = open(os.path.join(tut_dir, f), encoding='utf-8').read()
        for m in re.finditer(r'作业题\s*(\d+)|做题\s*(\d+)|题\s*(\d+)\s*[:：（(]', text):
            num = int(next(g for g in m.groups() if g))
            mentioned.setdefault(num, f)
    asg = next((p for p in (merged(f'assignments/assignment_{n}/{name}') for name in ('assignment.md', 'README.md')) if p), None)
    if not asg:
        problems.append(f'P{n:02d}: 作业说明文件缺失')
        continue
    text = open(asg, encoding='utf-8').read()
    declared = set()
    for m in re.finditer(r'^\|\s*(?:🌟\s*)?(\d)\s*\|', text, re.M):
        declared.add(int(m.group(1)))
    for m in re.finditer(r'###\s*(?:🌟\s*)?题\s*(\d)', text):
        declared.add(int(m.group(1)))
    bogus = {x for x in mentioned if x not in declared and 1 <= x <= 9}
    if bogus:
        problems.append(f'P{n:02d}: 教程提到题 {sorted(bogus)}（首现 {sorted({mentioned[x] for x in bogus})[0]}），作业只声明 {sorted(declared)}')

print(f'章末指引同步校验：{len(problems)} 个问题')
for p in problems:
    print('  ❌', p)
sys.exit(1 if problems else 0)
