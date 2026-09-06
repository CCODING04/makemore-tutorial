#!/usr/bin/env python3
"""P-B2：19 个 Part 教程分册 PDF 批量生成（md-to-pdf-cjk 脚本 + 崩溃降级）"""
import os, re, subprocess, sys

V = '/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python'
PDFPY = '/home/admin02/.zcode/skills/md-to-pdf-cjk/scripts/md_to_pdf.py'
REPO = '/home/admin02/Code/WorkSpace/makemore-tutorial'
REVIEW = '/home/admin02/Code/WorkSpace/makemore-tutorial-review'
OUT = os.path.join(REVIEW, 'pdf')
os.makedirs(OUT, exist_ok=True)

TITLES = {1:'Bigrams',2:'MLP',3:'BatchNorm',4:'Backpropagation',5:'WaveNet',6:'Transformer-GPT',
7:'Minimind 复现',8:'后训练全流程',9:'CUDA 内核',10:'分布式训练',11:'对齐实战 verl',12:'微调实战',
13:'数据工程',14:'推理部署 vLLM',15:'多模态理解',16:'图像与视频生成',17:'Agentic RL',18:'RAG 全链路',19:'Agent 与 FC'}

def simplify(md_text):
    """崩溃降级：去掉嵌套强调（粗体/斜体标记）、行内代码与原始 HTML，保留纯文本可读性"""
    t = md_text.replace('**', '').replace('`', '')
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', t)   # 去斜体（防 reportlab 交叉标签）
    t = re.sub(r'<details>\s*|</details>\s*|<summary>|</summary>|<br\s*/?>|</?div[^>]*>', '', t)
    return t

def gen_part(n):
    d = [x for x in os.listdir(os.path.join(REPO, 'courses')) if x.startswith(f'Part{n}_')][0]
    tdir = os.path.join(REPO, 'courses', d, 'tutorial')
    files = [f for f in sorted(os.listdir(tdir)) if f.endswith('.md')]
    files.sort(key=lambda f: (f != 'README.md', f))
    combined = '\n\n'.join(f'\n\n# {f[:-3]}\n\n' + open(os.path.join(tdir, f), encoding='utf-8').read()
                           for f in files)
    tmp = os.path.join(OUT, f'_tmp_Part{n}.md')
    open(tmp, 'w', encoding='utf-8').write(combined)
    out_pdf = os.path.join(OUT, f'Part{n:02d}_{TITLES[n]}.pdf')
    r = subprocess.run([V, PDFPY, tmp, out_pdf], capture_output=True, text=True, timeout=180)
    sys.stdout.write('.') ; sys.stdout.flush()
    pdf_ok = os.path.exists(out_pdf) and os.path.getsize(out_pdf) > 10000
    if pdf_ok:
        return 'ok', ''
    simple = simplify(combined)
    open(tmp, 'w', encoding='utf-8').write(simple)
    r2 = subprocess.run([V, PDFPY, tmp, out_pdf], capture_output=True, text=True, timeout=180)
    pdf_ok2 = os.path.exists(out_pdf) and os.path.getsize(out_pdf) > 10000
    if pdf_ok2:
        return 'ok(simplified)', ''
    return 'fail', (r.stderr or r.stdout)[-120:]

ok, simplified, fails = 0, 0, []
for n in range(1, 20):
    st, err = gen_part(n)
    if st == 'ok': ok += 1
    elif st.startswith('ok'): simplified += 1; ok += 1
    else: fails.append((n, err)); print(f'❌ Part{n:02d}: {err[:80]}')
    print(f'Part{n:02d}: {st}')
print(f'\n完成 {ok}/19（其中简化版 {simplified}）')
