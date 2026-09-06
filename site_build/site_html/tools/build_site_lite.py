#!/usr/bin/env python3
"""零依赖课程站点渲染器 v2（亮/暗双主题 + 站内链接绝对化）。

REPO 为底 + REVIEW 覆盖 → 合并树 → 静态 HTML（site_build/site_html/）。
- 站内 .md 链接在渲染期解析为站点绝对路径（先按当前页目录，再按站点根）
- 亮色默认 + 亮/暗切换（localStorage 记忆）
- 数学 KaTeX / Mermaid 浏览器端渲染（无网优雅降级）
用法：python build_site_lite.py [--serve [端口]]
"""
import html
import os
import re
import shutil
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS)   # 仓库根（tools/ 的上一级）

PART_TITLES = {
    1: 'Bigrams', 2: 'MLP', 3: 'BatchNorm', 4: 'Backpropagation', 5: 'WaveNet',
    6: 'Transformer/GPT', 7: 'Minimind 复现', 8: '后训练全流程', 9: 'CUDA 内核',
    10: '分布式训练', 11: '对齐实战 verl', 12: '微调实战', 13: '数据工程',
    14: '推理部署 vLLM', 15: '多模态理解', 16: '图像/视频生成', 17: 'Agentic RL',
    18: 'RAG 全链路', 19: 'Agent 与 FC',
}

BUILD = os.path.join(REPO_ROOT, 'site_build')

# ---------- 合并树 ----------
def merge_tree():
    docs = os.path.join(BUILD, 'docs')
    if os.path.exists(docs):
        shutil.rmtree(docs)
    os.makedirs(docs)
    for base in (REPO_ROOT,):   # review 分支：优化内容已并入仓库自身
        for tree in ('courses', 'assignments', 'widgets', 'docs', 'assignment_reference', 'tools', 'maps', 'quizzes'):
            src = os.path.join(base, tree)
            if not os.path.isdir(src):
                continue
            for dp, dirs, fs in os.walk(src):
                dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache')]
                rel = os.path.relpath(dp, base)
                dst = os.path.join(docs, rel)
                os.makedirs(dst, exist_ok=True)
                for f in fs:
                    shutil.copy2(os.path.join(dp, f), os.path.join(dst, f))
    shutil.copy2(os.path.join(REPO_ROOT, 'README.md'), os.path.join(docs, 'index.md'))
    return docs


def code_block(code, lang):
    """带语法高亮的代码块；Pygments 不可用/未知语言时回退纯转义。"""
    if lang:
        try:
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name
            from pygments.formatters import HtmlFormatter
            try:
                lexer = get_lexer_by_name(lang, stripnl=False)
            except Exception:
                lexer = None
            if lexer is not None:
                return highlight(code, lexer, HtmlFormatter(cssclass='highlight', nowrap=False))
        except Exception:
            pass
    return f'<pre><code>{esc(code)}</code></pre>'

# ---------- Markdown 渲染 ----------
def esc(s):
    return html.escape(s, quote=False)

PAGE_DIR = ''   # 当前渲染页目录（相对站点根），供 rewrite_link 使用

def rewrite_link(href):
    """站内链接 → 站点绝对路径。外链/纯锚点原样。"""
    pure, _, frag = href.partition('#')
    if pure.startswith(('http://', 'https://', '/')):
        return href
    cand = ''
    if pure:
        # normpath 在 Windows 上产生反斜杠，会混进 URL，统一回正斜杠
        cand = os.path.normpath(os.path.join(PAGE_DIR, pure)).replace(os.sep, '/')
        if not os.path.exists(os.path.join(BUILD, 'docs', cand)):
            cand2 = os.path.normpath(pure).replace(os.sep, '/')  # 站点根相对（根 README 风格 courses/xxx）
            if os.path.exists(os.path.join(BUILD, 'docs', cand2)):
                cand = cand2
    if pure.endswith('.md') and cand:
        cand = cand[:-3] + '.html'
        if cand == 'README.html':  # 仓库根 README 在站点中即首页
            cand = 'index.html'
    out = '/' + cand if cand else '/'
    return out + (('#' + frag) if frag else '')

def inline(s):
    s = esc(s)
    s = s.replace('\\|', '|')  # 还原表格转义竖线
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*(.+?)\*\*',
               lambda m: '<strong>' + re.sub(r'\*([^*\s][^*]*)\*(?!\*)', r'<em>\1</em>', m.group(1)) + '</strong>', s)
    s = re.sub(r'(?<!\*)\*([^*\s][^*]*)\*(?!\*)', r'<em>\1</em>', s)
    s = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',
               lambda m: f'<img src="{rewrite_link(m.group(2))}" alt="{m.group(1)}" style="max-width:100%">', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
               lambda m: f'<a href="{rewrite_link(m.group(2))}">{m.group(1)}</a>', s)
    return s

def render_blocks(md):
    lines = md.split('\n')
    out, i = [], 0
    in_code, code_buf, code_lang = False, [], ''
    list_open = None
    while i < len(lines):
        ln = lines[i]
        # 非列表内容抵达时先闭合未关的列表（防 <p>/<table>/<details> 嵌进 <ul>）
        if list_open and ln.strip() and not re.match(r'^\s*([-*+]|\d+\.)\s+', ln):
            out.append(f'</{list_open}>')
            list_open = None
        m = re.match(r'^```(\w*)\s*$', ln)
        if m:
            if not in_code:
                in_code, code_buf, code_lang = True, [], m.group(1)
            else:
                body = '\n'.join(code_buf)
                if code_lang == 'mermaid':
                    out.append(f'<pre class="mermaid">{esc(body)}</pre>')
                else:
                    out.append(code_block(body, code_lang))
                in_code = False
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue
        st = ln.strip()
        if st in ('<details>', '<details>'):
            out.append('<details>')
            i += 1
            continue
        if st == '</details>':
            out.append('</details>')
            i += 1
            continue
        if st.startswith('<summary>'):
            if '</summary>' not in st:
                buf = [st]
                while i + 1 < len(lines) and '</summary>' not in buf[-1]:
                    i += 1
                    buf.append(lines[i])
                inner = ' '.join(x.strip() for x in buf)
            else:
                inner = st
            inner = re.sub(r'^<summary>', '', inner)
            inner = re.sub(r'</summary>\s*$', '', inner)
            out.append(f'<summary>{inline(inner.strip())}</summary>')
            i += 1
            continue
        if st.startswith('<br') or st.startswith('<div') or st == '</div>':
            out.append(st)
            i += 1
            continue
        m = re.match(r'^(#{1,6})\s+(.*)', ln)
        if m:
            lvl = len(m.group(1))
            out.append(f'<h{lvl}>{inline(m.group(2))}</h{lvl}>')
            i += 1
            continue
        if '|' in ln and i + 1 < len(lines) and re.match(r'^\s*\|?[\s:|-]+\|?\s*$', lines[i + 1]) and '-' in lines[i + 1]:
            def split_row(row_ln):
                # \| 是竖线的转义（保护不被当作分列符），切完再还原
                body = row_ln.strip().strip('|').replace('\\|', '\x00')
                return [c.strip().replace('\x00', '|') for c in body.split('|')]
            header = split_row(ln)
            i += 2
            rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                rows.append(split_row(lines[i]))
                i += 1
            t = ['<div class="tbl-wrap"><table class="md-table">', '<thead><tr>']
            t += [f'<th>{inline(c)}</th>' for c in header]
            t.append('</tr></thead><tbody>')
            for r in rows:
                t.append('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>')
            t.append('</tbody></table></div>')
            out.append('\n'.join(t))
            continue
        if re.match(r'^\s*>\s?', ln):
            buf = []
            while i < len(lines) and re.match(r'^\s*>\s?', lines[i]):
                buf.append(re.sub(r'^\s*>\s?', '', lines[i]))
                i += 1
            out.append('<blockquote>' + render_blocks('\n'.join(buf)) + '</blockquote>')
            continue
        m = re.match(r'^(\s*)([-*+]|\d+\.)\s+(.*)', ln)
        if m:
            items = [m.group(3)]
            i += 1
            while i < len(lines):
                ln2 = lines[i]
                m2 = re.match(r'^\s*([-*+]|\d+\.)\s+(.*)', ln2)
                if m2:                       # 同级/缩进的新列表项
                    items.append(m2.group(2))
                    i += 1
                    continue
                if (not ln2.strip()
                        or re.match(r'^(#{1,6}\s|```|\s*>\s?)', ln2)
                        or '|' in ln2
                        or ln2.lstrip().startswith('<')):
                    break                    # 列表结束
                items[-1] += ' ' + ln2.strip()   # 续行 → 并回当前项（修"——"断裂）
                i += 1
            tag = 'ol' if re.match(r'\d+\.', m.group(2)) else 'ul'
            if list_open != tag:
                if list_open:
                    out.append(f'</{list_open}>')
                out.append(f'<{tag}>')
                list_open = tag
            for it in items:
                out.append(f'<li>{inline(it)}</li>')
            continue
        if not st:
            if list_open:
                out.append(f'</{list_open}>')
                list_open = None
            i += 1
            continue
        if re.match(r'^\s*---+\s*$', ln):
            out.append('<hr>')
            i += 1
            continue
        buf = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,6}\s|```|\s*>\s?|(\s*[-*+]\s|\s*\d+\.\s))', lines[i]) and '|' not in lines[i] and not lines[i].lstrip().startswith('<'):
            buf.append(lines[i])
            i += 1
        out.append(f'<p>{inline(" ".join(b.strip() for b in buf))}</p>')
    if list_open:
        out.append(f'</{list_open}>')
    if in_code:
        out.append(f'<pre><code>{esc(chr(10).join(code_buf))}</code></pre>')
    return '\n'.join(out)

PAGE = """<!DOCTYPE html>
<html lang="zh" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#ffffff">
<title>{title} · makemore 教程</title>
<link rel="stylesheet" href="/_assets/style.css?v=31">
<script>
(function(){try{var t=localStorage.getItem('mm-theme');if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}})();
window.MathJax = {tex: {inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$']]}};
</script>
<script defer src="{mathjax_src}" onerror="var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';document.head.appendChild(s);"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js" onerror="window.__noMermaid=1"></script>
<script>document.addEventListener('DOMContentLoaded',function(){if(window.mermaid)mermaid.initialize({startOnLoad:true});});</script>
</head>
<body>
<header class="topbar">
  <a class="brand" href="/index.html">🌱 makemore 教程 <span class="badge">审计优化版</span></a>
  <div class="tools">
    <button id="sideBtn" title="收起/展开目录">☰ 目录</button>
    <button id="fontMinus" title="减小字号">A−</button>
    <button id="fontReset" title="标准字号">A</button>
    <button id="fontPlus" title="增大字号">A+</button>
    <button id="themeBtn" title="切换亮/暗主题">🌙 夜间</button>
  </div>
</header>
<div class="layout">
<nav class="side">{nav}</nav>
<main class="content">{body}
<nav class="pager">{pager}</nav>
</main>
</div>
<script>
(function(){var b=document.getElementById('sideBtn');
function isNarrow(){return window.matchMedia('(max-width:900px)').matches;}
try{if(localStorage.getItem('mm-side')==='off')document.body.classList.add('no-side');}catch(e){}
b.onclick=function(){
  if(isNarrow()){document.body.classList.toggle('side-open');}
  else{var off=document.body.classList.toggle('no-side');
       try{localStorage.setItem('mm-side',off?'off':'on');}catch(e){}}
};
window.addEventListener('resize',function(){
  if(!isNarrow())document.body.classList.remove('side-open');
});
// 点侧栏链接后自动关抽屉（回到阅读）
document.querySelectorAll('.side').forEach(function(s){
  s.addEventListener('click',function(e){
    if(e.target.tagName==='A'&&isNarrow())document.body.classList.remove('side-open');
  });
});})();
(function(){var LEVELS=5,KEY='mm-fs',idx=1;   /* 0:87.5% 1:100% 2:112.5% 3:125% 4:137.5% */
try{var v=parseInt(localStorage.getItem(KEY));if(v>=0&&v<LEVELS)idx=v;}catch(e){}
function apply(){document.documentElement.setAttribute('data-fs',String(idx));
document.documentElement.style.fontSize='';
try{localStorage.setItem(KEY,idx);}catch(e){}}
document.getElementById('fontMinus').onclick=function(){idx=Math.max(0,idx-1);apply();};
document.getElementById('fontReset').onclick=function(){idx=1;apply();};
document.getElementById('fontPlus').onclick=function(){idx=Math.min(LEVELS-1,idx+1);apply();};
apply();})();
(function(){var b=document.getElementById('themeBtn');
function paint(){var t=document.documentElement.getAttribute('data-theme');
b.textContent = t==='dark' ? '☀️ 日间' : '🌙 夜间';}
b.onclick=function(){var t=document.documentElement.getAttribute('data-theme');
var n=t==='dark'?'light':'dark';
document.documentElement.setAttribute('data-theme',n);
try{localStorage.setItem('mm-theme',n);}catch(e){}
paint();};
paint();})();
</script>
</body>
</html>"""

CSS = """
/* === 设计系统：Mintlify 风格（awesome-design-md 参照）===
   白底近黑字 / 绿色点缀 / 5-8% 透明度边框 / 三权重 400-500-600 / 代码块常驻深底 */
:root, :root[data-theme=light]{
  --bg:#ffffff; --bg2:#ffffff; --card:#ffffff; --card2:#fafafa;
  --fg:#0d0d0d; --fg2:#333333; --fg3:#666666; --fg4:#888888;
  --line:rgba(0,0,0,0.05); --line2:rgba(0,0,0,0.08);
  --brand:#18E299; --brand-deep:#0fa76e; --brand-tint:#d4fae8;
  --link:#0fa76e; --link-hover:#18E299; --focus:#18E299;
  --code-bg:#f0f3f6; --pre-bg:#f6f8fa; --pre-fg:#1f2328; --pre-line:#d0d7de;
  --warn:#c37d0d; --tag:#3772cf;
}
:root[data-theme=dark]{
  --bg:#0d0d0d; --bg2:#0d0d0d; --card:#141414; --card2:#141414;
  --fg:#ededed; --fg2:#a0a0a0; --fg3:#a0a0a0; --fg4:#666666;
  --line:rgba(255,255,255,0.08); --line2:rgba(255,255,255,0.14);
  --brand:#18E299; --brand-deep:#18E299; --brand-tint:#0d2b22;
  --link:#18E299; --link-hover:#5cf0c0; --focus:#18E299;
  --code-bg:#1a1f24; --pre-bg:#141414; --pre-fg:#ededed;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
html{font-size:100%}
:root[data-fs="0"]{font-size:87.5%}
:root[data-fs="1"]{font-size:100%}
:root[data-fs="2"]{font-size:112.5%}
:root[data-fs="3"]{font-size:125%}
:root[data-fs="4"]{font-size:137.5%}
body{margin:0;font-family:Inter,system-ui,-apple-system,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;font-size:1.0625rem;font-weight:400;background:var(--bg);color:var(--fg2);line-height:1.8}
h1,h2,h3,h4{color:var(--fg);font-weight:600}
.topbar{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center;padding:10px 22px;background:color-mix(in srgb, var(--bg2) 86%, transparent);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.brand{color:var(--fg);text-decoration:none;font-weight:600;font-size:15px;letter-spacing:-0.2px}
.badge{font-size:11px;font-weight:500;color:var(--brand-deep);background:var(--brand-tint);border:none;border-radius:9999px;padding:2px 10px;margin-left:8px}
.tools{display:flex;gap:6px}
.tools button{background:transparent;color:var(--fg);border:1px solid var(--line2);border-radius:8px;padding:5px 10px;cursor:pointer;font-size:13px;font-weight:500;min-height:34px}
.tools button:hover{border-color:var(--brand);color:var(--brand-deep)}
#themeBtn{min-width:88px}
#sideBtn{min-width:76px}
#sideBtn{display:none}
@media (max-width:900px){#sideBtn{display:inline-block}}
.layout{display:flex;max-width:1240px;margin:0 auto}
.side{width:264px;flex:none;background:var(--bg);border-right:1px solid var(--line);padding:16px 12px;position:sticky;top:53px;height:calc(100vh - 53px);overflow-y:auto;scrollbar-width:thin}
body.no-side .side{display:none}
body.no-side .layout{max-width:980px}
.nav-home{margin-bottom:10px}
.nav-home a{color:var(--fg);font-weight:600;font-size:15px}
.nav-sec{margin:2px 0}
.nav-sec summary{list-style:none;padding:3px 8px;cursor:pointer;user-select:none;font-size:15px;font-weight:600;color:var(--fg);line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nav-sec summary:hover{color:var(--accent)}
details.nav-sec summary::-webkit-details-marker{display:none}
details.nav-sec summary::before{content:'▸ ';color:var(--fg4)}
details.nav-sec[open] summary::before{content:'▾ ';}
details.nav-sec[open] summary{color:var(--fg)}
details.nav-sec .nav-item{padding:0 0 0 10px}
.nav-item a{display:block;color:var(--fg2);padding:2.5px 8px;border-radius:6px;font-size:15px;font-weight:400;text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.45}
.nav-item a:hover{color:var(--brand-deep);background:var(--brand-tint)}
.nav-item a.cur{color:var(--brand-deep);background:var(--brand-tint);font-weight:600}
:root[data-theme=dark] .nav-item a:hover,:root[data-theme=dark] .nav-item a.cur{background:#182b24}
:root[data-theme=dark] .nav-item a.cur{color:var(--brand)}
.nav-sec .nav-item{padding:0}
.content{flex:1;min-width:0;max-width:800px;margin:0 auto;padding:36px 40px 80px}
.content h1{font-size:2em;font-weight:600;letter-spacing:-0.4px;line-height:1.25;border-bottom:1px solid var(--line);padding-bottom:12px;color:var(--fg)}
.content h2{font-size:1.4em;font-weight:600;letter-spacing:-0.2px;margin-top:40px;padding-bottom:6px;border-bottom:1px solid var(--line);color:var(--fg)}
.content h3{font-size:1.15em;font-weight:600;margin-top:30px;color:var(--fg)}
.content p{margin:14px 0}
.content a{color:var(--link);text-decoration:none;font-weight:500}
.content a:hover{color:var(--link-hover);text-decoration:underline}
.content strong{font-weight:600;color:var(--fg)}
.content ul,.content ol{padding-left:22px;margin:12px 0}
.content li{margin:5px 0}
.content pre{background:var(--pre-bg);color:var(--pre-fg);border:1px solid var(--pre-line,var(--line));border-radius:12px;padding:16px 18px;overflow-x:auto;line-height:1.6;font-size:0.9375em}
.content code{background:var(--code-bg);padding:2px 7px;border-radius:6px;font-size:0.86em;font-family:'Geist Mono',ui-monospace,SFMono-Regular,Consolas,monospace}
.content pre code{background:none;padding:0;font-size:1em}
blockquote{border-left:3px solid var(--brand);background:var(--card2);margin:16px 0;padding:10px 18px;border-radius:0 12px 12px 0}
blockquote p{margin:8px 0}
.tbl-wrap{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:12px}
table.md-table{border-collapse:collapse;width:100%;font-size:0.92em;margin:0}
.md-table th,.md-table td{border:none;padding:9px 14px;text-align:left;border-bottom:1px solid var(--line)}
.md-table th{background:var(--card2);font-weight:600;color:var(--fg)}
.md-table tr:last-child td{border-bottom:none}
details{border:1px solid var(--line);border-radius:12px;padding:12px 18px;margin:12px 0;background:var(--card2)}
summary{cursor:pointer;color:var(--fg);font-weight:600}
details[open] summary{color:var(--link);border-bottom:1px dashed var(--line);padding-bottom:6px;margin-bottom:6px}
hr{border:none;border-top:1px solid var(--line);margin:32px 0}
.derivation{background:var(--card2);border:1px solid var(--line);border-left:3px solid var(--brand);border-radius:12px;padding:6px 24px 14px;margin:20px 0}
.derivation .d-title{font-weight:600;color:var(--link);margin:12px 0 4px}
.pager{display:flex;justify-content:space-between;margin-top:52px;border-top:1px solid var(--line);padding-top:20px;gap:12px}
.pager a{color:var(--fg);text-decoration:none;padding:10px 16px;border:1px solid var(--line2);border-radius:9999px;background:var(--card);font-size:14px;font-weight:500}
.pager a:hover{border-color:var(--brand);color:var(--brand-deep)}
img{border-radius:12px;border:1px solid var(--line)}
body.no-side .side{display:none}
body.no-side .layout{max-width:900px}
/* 窄屏：侧栏变抽屉（顶栏 ☰ 唤出），不再藏死 */
@media (max-width:900px){
  .layout{display:block}
  .side{display:none;position:fixed;top:53px;left:0;right:0;bottom:0;width:auto;height:auto;z-index:20;
        background:var(--bg);padding:12px;border-right:none;box-shadow:0 8px 30px rgba(0,0,0,0.12)}
  body.side-open .side{display:block}
  body.side-open{overflow:hidden}
  .content{max-width:100%;padding:18px 16px 70px}
  .content h1{font-size:1.45em}
  .content h2{font-size:1.22em}
  .content pre,.highlight pre{padding:12px;border-radius:10px;max-width:100vw}
  .tbl-wrap{margin:12px -16px;width:calc(100% + 32px)}
  .pager{flex-direction:column}
  .pager a{text-align:center}
  .topbar{padding:10px 14px}
}
@media (max-width:480px){
  body{font-size:15px}
  .content{padding:16px 12px 60px}
  .tbl-wrap{margin:12px -12px;width:calc(100% + 24px)}
}
/* Pygments：代码块常驻暗底（Mintlify 文档风） */
.highlight{background:var(--pre-bg);border:1px solid var(--line);border-radius:12px;margin:16px 0}
.highlight pre{margin:0;border:none;background:transparent}
.highlight code{display:block;padding:16px 18px;overflow-x:auto;line-height:1.6;font-size:0.875em;font-family:'Geist Mono',ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--pre-fg)}
:root[data-theme=light] .highlight .k,:root[data-theme=light] .highlight .kd,:root[data-theme=light] .highlight .kn,:root[data-theme=light] .highlight .ow,:root[data-theme=light] .highlight .kr{color:#cf222e}
:root[data-theme=light] .highlight .s1,:root[data-theme=light] .highlight .s2,:root[data-theme=light] .highlight .sa,:root[data-theme=light] .highlight .sd,:root[data-theme=light] .highlight .se{color:#0a306c}
:root[data-theme=light] .highlight .mi,:root[data-theme=light] .highlight .mf,:root[data-theme=light] .highlight .mh,:root[data-theme=light] .highlight .il{color:#0550ae}
:root[data-theme=light] .highlight .c1,:root[data-theme=light] .highlight .ch,:root[data-theme=light] .highlight .cm{color:#6e7781;font-style:italic}
:root[data-theme=light] .highlight .nf,:root[data-theme=light] .highlight .fm{color:#8250df}
:root[data-theme=light] .highlight .nb,:root[data-theme=light] .highlight .bp{color:#953800}
:root[data-theme=light] .highlight .o,:root[data-theme=light] .highlight .p{color:#1f2328}
:root[data-theme=light] .highlight .nn,:root[data-theme=light] .highlight .nc{color:#953800}
:root[data-theme=light] .highlight .nd{color:#8250df}
:root[data-theme=dark] .highlight .k,:root[data-theme=dark] .highlight .kd,:root[data-theme=dark] .highlight .kn,:root[data-theme=dark] .highlight .ow,:root[data-theme=dark] .highlight .kr{color:#ff7b72}
:root[data-theme=dark] .highlight .s1,:root[data-theme=dark] .highlight .s2,:root[data-theme=dark] .highlight .sa,:root[data-theme=dark] .highlight .sd,:root[data-theme=dark] .highlight .se{color:#a5d6ff}
:root[data-theme=dark] .highlight .mi,:root[data-theme=dark] .highlight .mf,:root[data-theme=dark] .highlight .mh,:root[data-theme=dark] .highlight .il{color:#79c0ff}
:root[data-theme=dark] .highlight .c1,:root[data-theme=dark] .highlight .ch,:root[data-theme=dark] .highlight .cm{color:#8b949e;font-style:italic}
:root[data-theme=dark] .highlight .nf,:root[data-theme=dark] .highlight .fm{color:#d2a8ff}
:root[data-theme=dark] .highlight .nb,:root[data-theme=dark] .highlight .bp{color:#ffa657}
:root[data-theme=dark] .highlight .o,:root[data-theme=dark] .highlight .p{color:#c9d1d9}
:root[data-theme=dark] .highlight .nn,:root[data-theme=dark] .highlight .nc{color:#ffa657}
:root[data-theme=dark] .highlight .nd{color:#d2a8ff}
.content pre:not(.highlight pre){color:var(--pre-fg)}
"""

def write_favicon(site):
    """生成站点图标 favicon.ico（品牌绿圆角方块 + 白色 M），纯标准库实现，无外部依赖。"""
    import struct
    W = H = 32
    R = 7  # 圆角半径
    green = (0x99, 0xE2, 0x18, 255)   # 品牌绿 #18E299（BGRA）
    white = (255, 255, 255, 255)
    transparent = (0, 0, 0, 0)
    # M 字模（13x7 点阵），2 倍放大后居中
    M = ["X...........X",
         "XX.........XX",
         "X.X.......X.X",
         "X..X.....X..X",
         "X...X...X...X",
         "X....X.X....X",
         "X.....X.....X"]
    gx, gy = (W - 26) // 2, (H - 14) // 2

    def pixel(x, y):
        dx, dy = min(x, W - 1 - x), min(y, H - 1 - y)
        if dx < R and dy < R and (R - dx) ** 2 + (R - dy) ** 2 > R * R:
            return transparent
        tx, ty = (x - gx) // 2, (y - gy) // 2
        if 0 <= ty < len(M) and 0 <= tx < len(M[0]) and M[ty][tx] == 'X':
            return white
        return green

    xor = b''.join(bytes(pixel(x, y)) for y in range(H - 1, -1, -1) for x in range(W))
    and_mask = b'\x00' * (W // 8 * H)
    bmp = struct.pack('<IiiHHIIiiII', 40, W, H * 2, 1, 32, 0, len(xor) + len(and_mask), 0, 0, 0, 0)
    img = bmp + xor + and_mask
    ico = struct.pack('<HHH', 0, 1, 1) + struct.pack('<BBBBHHII', W, H, 0, 0, 1, 32, len(img), 22) + img
    open(os.path.join(site, 'favicon.ico'), 'wb').write(ico)

def build():
    docs = merge_tree()
    site = os.path.join(BUILD, 'site_html')
    if os.path.exists(site):
        shutil.rmtree(site)
    pages = []
    for dp, dirs, fs in os.walk(docs):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache')]
        for f in sorted(fs):
            # 统一为正斜杠：Windows 的 os.sep 是反斜杠，会让 nav/链接的
            # `courses/Part.../` 正则与 startswith 判断全部失效（导航丢失）
            r = os.path.relpath(os.path.join(dp, f), docs).replace(os.sep, '/')
            if f.endswith('.md') and not r.startswith('data'):
                pages.append(r)

    # 知识脉络图索引页（maps/ 下各 Part 的 markmap 交互图）
    titles = {1:'Bigrams',2:'MLP',3:'BatchNorm',4:'Backpropagation',5:'WaveNet',6:'Transformer/GPT',
    7:'Minimind 复现',8:'后训练全流程',9:'CUDA 内核',10:'分布式训练',11:'对齐实战 verl',12:'微调实战',
    13:'数据工程',14:'推理部署 vLLM',15:'多模态理解',16:'图像/视频生成',17:'Agentic RL',18:'RAG 全链路',19:'Agent 与 FC'}
    rows = ''.join(
        f'<tr><td>Part {k}</td><td>{esc(v)}</td>'
        f'<td><a href="/maps/Part{k:02d}/framework.markmap.html">交互式思维导图 ↗</a></td>'
        f'<td><a href="/maps/Part{k:02d}/framework.html">大纲页面</a></td></tr>'
        for k, v in titles.items())
    maps_md = (
        '# 🗺️ 知识脉络图（19 Part + 全课程总图）\n\n'
        '每张图为可折叠/缩放的交互式思维导图（基于各章教程大纲生成，零改写零幻觉）。\n\n'
        '| Part | 主题 | 交互图 | 大纲 |\n|---|---|---|---|\n'
        + ''.join(f'| Part {k} | {v} | [交互图](/maps/Part{k:02d}/framework.markmap.html) | [大纲](/maps/Part{k:02d}/framework.html) |\n' for k, v in titles.items())
        + '| 全课程 | 总脉络 | [总图](/maps/master/framework.markmap.html) | [大纲](/maps/master/framework.html) |\n')
    os.makedirs(os.path.join(docs, 'maps'), exist_ok=True)
    os.makedirs(os.path.join(docs, 'quizzes'), exist_ok=True)
    open(os.path.join(docs, 'maps', 'index.md'), 'w', encoding='utf-8').write(maps_md)
    pages.append('maps/index.md')
    pages.append('quizzes/quiz_index.md')
    if not os.path.exists(os.path.join(docs, 'quizzes', 'quiz_index.md')):
        open(os.path.join(docs, 'quizzes', 'quiz_index.md'), 'w', encoding='utf-8').write(
            '# 📝 逐 Part 测验与闪卡\n\n（生成中——education 技能按 Part 产出后自动汇总。）\n')
    # 保序去重：maps/index 等页面既被扫描收录又被无条件 append 时会重复，
    # 导致页数虚增、pager 邻居指向自身
    pages = list(dict.fromkeys(pages))
    pages.sort(key=lambda r: (0 if r == 'index.md' else 1, r))

    def nav_html(cur):
        def sort_key(rel):
            name = os.path.basename(rel)[:-3]
            if name == 'README':
                return (0, '')
            return (1, name)
        parts = {}
        for rel in pages:
            m = re.match(r'(courses/Part\d+_[^/]+)/', rel)
            if m:
                parts.setdefault(m.group(1), []).append(rel)

        def item(rel):
            name = os.path.basename(rel)[:-3]
            disp = 'README · 导览' if name == 'README' else name
            cls = ' class="cur"' if rel == cur else ''
            return f'<div class="nav-item"><a{cls} href="/{rel[:-3]}.html">{esc(disp)}</a></div>'

        def section(title, rels, open_if):
            op = ' open' if open_if else ''
            buf.append(f'<details class="nav-sec"{op}><summary>{esc(title)}</summary>')
            for rel in sorted(rels, key=sort_key):
                buf.append(item(rel))
            buf.append('</details>')

        buf = ['<div class="nav-home"><a href="/index.html">📑 课程首页</a></div>']
        for k in sorted(parts, key=lambda x: int(re.search(r'Part(\d+)', x).group(1))):
            num = int(re.search(r'Part(\d+)', k).group(1))
            title = f'Part {num} · {PART_TITLES.get(num, "")}'
            section(title, parts[k], cur.startswith(k + '/'))
        section('参考文档', [r for r in pages if r.startswith('docs/')], cur.startswith('docs/'))
        asg_rels = [r for r in pages
                    if re.match(r'assignments/assignment_\d+/[^/]+\.md', r)
                    and os.path.basename(r) in ('assignment.md', 'README.md')]
        buf.append('<details class="nav-sec"><summary>课后作业（19 套）</summary>')
        for rel in sorted(asg_rels):
            m = re.match(r'assignments/(assignment_\d+)/', rel)
            cls = ' class="cur"' if rel == cur else ''
            buf.append(f'<div class="nav-item"><a{cls} href="/{rel[:-3]}.html">{esc(m.group(1).capitalize())}</a></div>')
        buf.append('</details>')
        return '\n'.join(buf)

    mathjax_src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'  # 本地包缺失，CDN + 无网降级

    global PAGE_DIR
    n_ok = 0
    for idx, rel in enumerate(pages):
        PAGE_DIR = os.path.dirname(rel)
        text = open(os.path.join(docs, rel), encoding='utf-8').read()
        body = render_blocks(text)
        title = '课程首页 · 审计优化版' if rel == 'index.md' else os.path.basename(rel)[:-3]
        pager = ''
        if idx > 0:
            pager += f'<a href="/{pages[idx-1][:-3]}.html">← {esc(os.path.basename(pages[idx-1])[:-3])}</a>'
        if idx < len(pages) - 1:
            pager += f'<a href="/{pages[idx+1][:-3]}.html">{esc(os.path.basename(pages[idx+1])[:-3])} →</a>'
        html_out = (PAGE.replace('{title}', esc(title)).replace('{nav}', nav_html(rel))
                        .replace('{body}', body).replace('{pager}', pager)
                        .replace('{mathjax_src}', mathjax_src))
        dst = os.path.join(site, rel[:-3] + '.html')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, 'w', encoding='utf-8').write(html_out)
        n_ok += 1
    for dp, dirs, fs in os.walk(docs):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache')]
        for f in fs:
            if f.endswith('.md'):
                continue
            src_f = os.path.join(dp, f)
            rel = os.path.relpath(src_f, docs)
            dst_f = os.path.join(site, rel)
            os.makedirs(os.path.dirname(dst_f), exist_ok=True)
            shutil.copy2(src_f, dst_f)
    assets = os.path.join(site, '_assets')
    os.makedirs(assets, exist_ok=True)
    open(os.path.join(assets, 'style.css'), 'w', encoding='utf-8').write(CSS)
    write_favicon(site)
    print(f'✅ 渲染 {n_ok} 个页面 → {site}')
    return site

def serve(port):
    site = os.path.join(BUILD, 'site_html')
    os.chdir(site)
    import socket
    ip = next((ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith('127.')), '127.0.0.1')
    print('站点已启动：')
    print(f'  本机访问   http://127.0.0.1:{port}/')
    print(f'  局域网访问 http://{ip}:{port}/')
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    ThreadingHTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler).serve_forever()

if __name__ == '__main__':
    if '--serve' in sys.argv:
        port = int(sys.argv[sys.argv.index('--serve') + 1]) if len(sys.argv) > sys.argv.index('--serve') + 1 else 8000
        build()
        serve(port)
    else:
        build()
        print('预览：python build_site_lite.py --serve 8000')
