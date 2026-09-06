#!/usr/bin/env python3
"""批量生成 19 个 Part + 总课程的知识脉络图（parse → assemble → markmap 渲染）"""
import json, os, subprocess

K = '/home/admin02/.zcode/skills/knowledge-framework-builder'
V = '/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python'
REPO = '/home/admin02/Code/WorkSpace/makemore-tutorial'
REVIEW = '/home/admin02/Code/WorkSpace/makemore-tutorial-review'
OUT = os.path.join(REVIEW, 'maps')
os.makedirs(OUT, exist_ok=True)

TITLES = {1:'Bigrams',2:'MLP',3:'BatchNorm',4:'Backpropagation',5:'WaveNet',6:'Transformer/GPT',
7:'Minimind 复现',8:'后训练全流程',9:'CUDA 内核',10:'分布式训练',11:'对齐实战 verl',12:'微调实战',
13:'数据工程',14:'推理部署 vLLM',15:'多模态理解',16:'图像/视频生成',17:'Agentic RL',18:'RAG 全链路',19:'Agent 与 FC'}

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-200:]

jobs = []
for n in range(1, 20):
    d = [x for x in os.listdir(os.path.join(REPO, 'courses')) if x.startswith(f'Part{n}_')]
    md = os.path.join(REPO, 'courses', d[0], 'tutorial', 'README.md')
    jobs.append((f'Part{n:02d}', f'Part {n} · {TITLES[n]}', md))
jobs.append(('master', 'makemore 全课程脉络', os.path.join(REPO, 'README.md')))

fails = []
for slug, title, md in jobs:
    w = os.path.join(OUT, slug)
    os.makedirs(w, exist_ok=True)
    inj = os.path.join(w, 'input.json')
    inp = {"course_topic": title,
           "material_files": [{"path": md, "type": "markdown"}],
           "preferences": {"depth_hint": "force_skim",
                           "output_formats": ["markdown", "markmap"], "language": "zh"}}
    open(inj, 'w', encoding='utf-8').write(json.dumps(inp, ensure_ascii=False, indent=1))
    ok = True
    steps = [
        [V, f'{K}/scripts/parse_outline.py', '--input', inj, '--out', f'{w}/outline.json', '--quiet'],
        [V, f'{K}/scripts/assemble_result.py', '--input', inj, '--tree', f'{w}/outline.json', '--out', f'{w}/result.json', '--pretty'],
        [V, f'{K}/scripts/render_outputs.py', f'{w}/result.json', '--formats', 'markdown,markmap', '--out-dir', w],
    ]
    for cmd in steps:
        rc, err = run(cmd)
        if rc != 0:
            fails.append((slug, os.path.basename(cmd[-3]) if len(cmd) > 3 else 'render', err[-150:]))
            ok = False
            break
    if ok:
        print('✅', slug)

print(f'完成 {len(jobs)-len(fails)}/{len(jobs)}')
for x in fails:
    print('FAIL', x)
