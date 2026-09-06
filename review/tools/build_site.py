#!/usr/bin/env python3
"""课程站点构建：REPO 为底 + REVIEW 覆盖 → site_build/docs → mkdocs build。
依赖：pip install mkdocs mkdocs-material（当前环境网络阻塞，联网后执行）。
用法：python build_site.py [--serve]
"""
import os, shutil, subprocess, sys

REPO = '/home/admin02/Code/WorkSpace/makemore-tutorial'
REVIEW = '/home/admin02/Code/WorkSpace/makemore-tutorial-review'
BUILD = os.path.join(REVIEW, 'site_build')

def overlay(src_root, dst_root, trees=('courses', 'assignments')):
    for tree in trees:
        src = os.path.join(src_root, tree)
        if not os.path.isdir(src):
            continue
        for dp, _, fs in os.walk(src):
            if '__pycache__' in dp or '.pytest_cache' in dp:
                continue
            rel = os.path.relpath(dp, src_root)
            os.makedirs(os.path.join(dst_root, rel), exist_ok=True)
            for f in fs:
                shutil.copy2(os.path.join(dp, f), os.path.join(dst_root, rel, f))

def auto_nav(docs):
    """按 Part 生成 nav：README 为页，各章为子页。"""
    import re
    nav = [{'首页': 'index.md'}]
    parts = sorted(d for d in os.listdir(docs) if d.startswith('Part'))
    for p in parts:
        tdir = os.path.join(docs, p, 'tutorial')
        if not os.path.isdir(tdir):
            continue
        title = re.sub(r'^Part(\d+)_.*', r'Part \1', p)
        files = sorted(f for f in os.listdir(tdir) if f.endswith('.md'))
        ordered = [f for f in files if f == 'README.md'] + [f for f in files if f != 'README.md']
        nav.append({title: [os.path.join(p, 'tutorial', f) for f in ordered]})
    extra = [('作业与参考答案', 'assignments_guide.md')] if os.path.exists(os.path.join(docs, 'assignments_guide.md')) else []
    nav.extend(dict([e]) for e in extra)
    return nav

def main():
    if os.path.exists(BUILD):
        shutil.rmtree(BUILD)
    docs = os.path.join(BUILD, 'docs')
    os.makedirs(docs)
    overlay(REPO, docs)
    overlay(REVIEW, docs)   # REVIEW 覆盖在后 = 优化稿生效
    # 首页：根 README + 审计说明
    shutil.copy2(os.path.join(REPO, 'README.md'), os.path.join(docs, 'index.md'))
    for extra in ('assignments', 'docs'):
        pass
    nav = auto_nav(docs)
    import yaml
    data = {   # 与 site/mkdocs.yml 等价；此处直接构造以避开 !!python/name 标签
        'site_name': 'makemore 中文教程（审计优化版）',
        'theme': {'name': 'material',
                  'features': ['navigation.tabs', 'navigation.search', 'content.code.copy'],
                  'palette': [{'media': '(prefers-color-scheme: light)'},
                              {'media': '(prefers-color-scheme: dark)', 'scheme': 'slate'}]},
        'markdown_extensions': [
            {'pymdownx.superfences': {'custom_fences': [
                {'name': 'mermaid', 'class': 'mermaid',
                 'format': 'pymdownx.superfences.fence_code_format'}]}},
            {'pymdownx.arithmatex': {'generic': True}},
            'tables', 'admonition'],
        'extra_javascript': ['https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'],
        'docs_dir': docs,
        'site_dir': os.path.join(BUILD, 'site_html'),
        'nav': nav,
    }
    tmp_cfg = os.path.join(BUILD, 'mkdocs.yml')
    with open(tmp_cfg, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    try:
        import mkdocs  # noqa
    except ImportError:
        print('⚠️ 未安装 mkdocs：请先 `uv pip install mkdocs mkdocs-material`（当前网络阻塞，联网后执行）')
        print(f'✅ 合并文档树已就绪：{docs}（{sum(len(fs) for _,_,fs in os.walk(docs))} 个文件）')
        return 1
    subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '-f', tmp_cfg], check=True)
    print(f'✅ 站点构建完成：{os.path.join(BUILD, "site_html")}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
