#!/usr/bin/env python3
"""生成知识脉络图（markmap 格式）为每个 Part + 全课程总图。

从各 Part 的 README.md 中提取章节导航表，生成交互式思维导图 HTML。
用法：python generate_maps.py
"""
import os
import re

THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS)

PART_TITLES = {
    1: 'Bigrams', 2: 'MLP', 3: 'BatchNorm', 4: 'Backpropagation', 5: 'WaveNet',
    6: 'Transformer/GPT', 7: 'Minimind 复现', 8: '后训练全流程', 9: 'CUDA 内核',
    10: '分布式训练', 11: '对齐实战 verl', 12: '微调实战', 13: '数据工程',
    14: '推理部署 vLLM', 15: '多模态理解', 16: '图像/视频生成', 17: 'Agentic RL',
    18: 'RAG 全链路', 19: 'Agent 与 FC',
}

MARKMAP_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - 知识脉络图</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; }}
#mindmap {{ width: 100vw; height: 100vh; }}
.toolbar {{ position: fixed; top: 10px; right: 10px; z-index: 100; display: flex; gap: 8px; }}
.toolbar button {{ padding: 8px 16px; background: #18E299; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }}
.toolbar button:hover {{ background: #0fa76e; }}
.toolbar a {{ padding: 8px 16px; background: #f0f3f6; color: #1f2328; border: 1px solid #d0d7de; border-radius: 8px; text-decoration: none; font-size: 14px; }}
</style>
</head>
<body>
<div class="toolbar">
  <a href="/index.html">返回首页</a>
  <button onclick="markmap.fit()">适应屏幕</button>
</div>
<svg id="mindmap"></svg>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdn.jsdelivr.net/npm/markmap-view@0.15"></script>
<script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.15"></script>
<script>
const md = `{markdown}`;
(async function() {{
  const {{ Transformer }} = markmap;
  const transformer = new Transformer();
  const {{ root }} = transformer.transform(md);
  const {{ Markmap }} = markmap;
  const svg = document.getElementById('mindmap');
  const mm = Markmap.create(svg, {{
    autoFit: true,
    duration: 300,
    maxWidth: 300,
    initialExpandLevel: 2,
  }}, root);
  window.markmap = mm;
}})();
</script>
</body>
</html>"""


def extract_chapters_from_readme(readme_path):
    """从 README.md 提取章节信息。"""
    if not os.path.exists(readme_path):
        return []
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chapters = []
    # 匹配表格行：| 序号 | 章节 | 内容 | 对应脚本 |
    for match in re.finditer(r'\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*([^|]+)\|', content):
        num = match.group(1)
        title = match.group(2)
        link = match.group(3)
        desc = match.group(4).strip()
        chapters.append({'num': num, 'title': title, 'link': link, 'desc': desc})
    
    return chapters


def generate_part_map(part_num, part_dir):
    """为单个 Part 生成 markmap markdown。"""
    readme_path = os.path.join(part_dir, 'tutorial', 'README.md')
    chapters = extract_chapters_from_readme(readme_path)
    
    title = PART_TITLES.get(part_num, f'Part {part_num}')
    lines = [f'# Part {part_num}: {title}']
    
    if chapters:
        for ch in chapters:
            lines.append(f'## {ch["num"]}. {ch["title"]}')
            if ch['desc']:
                # 提取关键概念（用逗号分隔）
                concepts = [c.strip() for c in ch['desc'].split('、')]
                for concept in concepts[:5]:  # 最多5个概念
                    if concept:
                        lines.append(f'- {concept}')
    else:
        # 如果没有章节信息，列出 tutorial 目录下的文件
        tutorial_dir = os.path.join(part_dir, 'tutorial')
        if os.path.exists(tutorial_dir):
            for f in sorted(os.listdir(tutorial_dir)):
                if f.endswith('.md') and f != 'README.md':
                    name = f.replace('.md', '').replace('_', ' ').title()
                    lines.append(f'## {name}')
    
    return '\n'.join(lines)


def generate_master_map():
    """生成全课程总图的 markmap markdown。"""
    lines = ['# LLM 算法工程师面试课程', '']
    
    # 阶段一：神经网络地基
    lines.append('## 阶段一：神经网络地基')
    for p in range(1, 6):
        lines.append(f'- Part {p}: {PART_TITLES[p]}')
    
    # 阶段二：现代架构
    lines.append('## 阶段二：现代架构')
    for p in range(6, 8):
        lines.append(f'- Part {p}: {PART_TITLES[p]}')
    
    # 阶段三：后训练与对齐
    lines.append('## 阶段三：后训练与对齐')
    lines.append(f'- Part 8: {PART_TITLES[8]}')
    
    # 阶段四：系统与工程
    lines.append('## 阶段四：系统与工程')
    for p in range(9, 11):
        lines.append(f'- Part {p}: {PART_TITLES[p]}')
    
    # 阶段五：工业实战
    lines.append('## 阶段五：工业实战')
    for p in range(11, 15):
        lines.append(f'- Part {p}: {PART_TITLES[p]}')
    
    # 多模态与生成
    lines.append('## 多模态与生成')
    for p in range(15, 17):
        lines.append(f'- Part {p}: {PART_TITLES[p]}')
    
    # 应用线
    lines.append('## 应用线')
    for p in range(17, 20):
        lines.append(f'- Part {p}: {PART_TITLES[p]}')
    
    return '\n'.join(lines)


def main():
    maps_dir = os.path.join(REPO_ROOT, 'maps')
    os.makedirs(maps_dir, exist_ok=True)
    
    # 为每个 Part 生成 markmap
    for part_num in range(1, 20):
        part_dir = os.path.join(REPO_ROOT, 'courses', f'Part{part_num}_*')
        # 查找实际目录
        import glob
        matches = glob.glob(part_dir)
        if not matches:
            print(f'  跳过 Part {part_num}: 目录不存在')
            continue
        
        part_dir = matches[0]
        md = generate_part_map(part_num, part_dir)
        
        # 生成 HTML
        part_maps_dir = os.path.join(maps_dir, f'Part{part_num:02d}')
        os.makedirs(part_maps_dir, exist_ok=True)
        
        html = MARKMAP_TEMPLATE.format(
            title=f'Part {part_num}: {PART_TITLES.get(part_num, "")}',
            markdown=md.replace('`', '\\`').replace('\\', '\\\\')
        )
        
        html_path = os.path.join(part_maps_dir, 'framework.markmap.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # 生成纯 markdown 大纲
        md_path = os.path.join(part_maps_dir, 'framework.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f'  Part {part_num}: 生成 {len(md.splitlines())} 行')
    
    # 生成全课程总图
    master_dir = os.path.join(maps_dir, 'master')
    os.makedirs(master_dir, exist_ok=True)
    
    master_md = generate_master_map()
    html = MARKMAP_TEMPLATE.format(
        title='LLM 算法工程师面试课程 - 全课程总图',
        markdown=master_md.replace('`', '\\`').replace('\\', '\\\\')
    )
    
    with open(os.path.join(master_dir, 'framework.markmap.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    
    with open(os.path.join(master_dir, 'framework.md'), 'w', encoding='utf-8') as f:
        f.write(master_md)
    
    print(f'  全课程总图: 生成 {len(master_md.splitlines())} 行')
    try:
        print(f'✅ 知识脉络图生成完成 → {maps_dir}')
    except UnicodeEncodeError:
        print(f'[OK] 知识脉络图生成完成 -> {maps_dir}')


if __name__ == '__main__':
    main()
