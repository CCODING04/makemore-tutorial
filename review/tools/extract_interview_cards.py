#!/usr/bin/env python3
"""从作业说明抽取「面试直通车（话术卡）」→ YAML + CSV + Anki 导入文件。
支持两种格式：①旧条目式 - "Q"——A；②话术卡 **Qn："Q"** + 结论/原理/边界。
REVIEW 覆盖优先。用法：python extract_interview_cards.py [REPO] [输出目录]"""
import os, re, sys, glob

repo = sys.argv[1] if len(sys.argv) > 1 else '/home/admin02/Code/WorkSpace/makemore-tutorial'
review = '/home/admin02/Code/WorkSpace/makemore-tutorial-review'
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(review, 'interview')

def section(text):
    m = re.search(r'##\s*[^\n]*面试直通车[^\n]*\n(.*?)(?=\n## |\Z)', text, re.S)
    return m.group(1) if m else None

def parse_legacy(body):
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith('- '):
            continue
        mm = re.match(r'["“](.+?)["”][——\-–]+\s*(.+)', line[2:]) or re.match(r'(.+?)[——\-–]+\s*(.+)', line[2:])
        if mm:
            out.append({'question': mm.group(1).strip(), 'conclusion': mm.group(2).strip(),
                        'principle': '', 'boundary': ''})
    return out

def parse_cards(body):
    out = []
    blocks = re.split(r'\n(?=\*\*Q\d*)', body)
    for b in blocks:
        qm = re.match(r'\*\*Q\d*[：:]?\s*["“]?(.+?)["”]?\s*\*\*', b.strip())
        if not qm:
            continue
        card = {'question': qm.group(1).strip(), 'conclusion': '', 'principle': '', 'boundary': ''}
        for key, zh in (('conclusion', '结论'), ('principle', '原理'), ('boundary', '边界')):
            km = re.search(rf'-\s*\*\*{zh}\*\*[：:]\s*(.+?)(?=\n- \*\*|\n\*\*Q|\Z)', b, re.S)
            if km:
                card[key] = ' '.join(km.group(1).split())
        out.append(card)
    return out

rows = []
for n in range(1, 20):
    rel_dir = f'assignments/assignment_{n}'
    p = None
    for base in (review, repo):
        for name in ('assignment.md', 'README.md'):
            q = os.path.join(base, rel_dir, name)
            if os.path.exists(q):
                p = q
                break
        if p:
            break
    if not p:
        continue
    text = open(p, encoding='utf-8').read()
    body = section(text)
    if not body:
        rows.append({'part': n, 'question': '（缺失）', 'conclusion': '', 'principle': '', 'boundary': '',
                     'source': rel_dir, 'status': 'NO_SECTION'})
        continue
    cards = parse_cards(body) or parse_legacy(body)
    for c in cards:
        src_base = review if p.startswith(review + os.sep) else repo
        c.update(part=n, source=os.path.relpath(p, src_base), status='ok')
        rows.append(c)

os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'cards.yaml'), 'w', encoding='utf-8') as f:
    f.write('# 面试话术卡（话术版：结论→原理→边界；自动抽取）\n')
    for r in rows:
        f.write(f'- part: {r["part"]}\n  question: "{r["question"]}"\n  conclusion: "{r["conclusion"]}"\n'
                f'  principle: "{r["principle"]}"\n  boundary: "{r["boundary"]}"\n  source: {r["source"]}\n')
with open(os.path.join(outdir, 'cards.csv'), 'w', encoding='utf-8') as f:
    f.write('part\tquestion\tconclusion\tprinciple\tboundary\tsource\n')
    for r in rows:
        f.write(f'{r["part"]}\t{r["question"]}\t{r["conclusion"]}\t{r["principle"]}\t{r["boundary"]}\t{r["source"]}\n')
with open(os.path.join(outdir, 'anki_import.csv'), 'w', encoding='utf-8', newline='') as f:
    import csv
    w = csv.writer(f, delimiter='\t')
    for r in rows:
        back = '\n'.join(x for x in (r['conclusion'], r['principle'], r['boundary']) if x)
        w.writerow([f'【P{r["part"]}】{r["question"]}', back or r.get('answer', ''), f'P{r["part"]}::interview'])
ok = sum(1 for r in rows if r.get('status') != 'NO_SECTION')
print(f'{len(rows)} 卡（{ok} 有节）-> cards.yaml / cards.csv / anki_import.csv')
