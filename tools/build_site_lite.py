#!/usr/bin/env python3
"""零依赖课程站点渲染器 v3.1（GitHub Primer 风格 UI + 站内链接绝对化 + 搜索 + 进度追踪）。

REPO 为底 + REVIEW 覆盖 → 合并树 → 静态 HTML（site_build/site_html/）。
- 站内 .md 链接在渲染期解析为站点绝对路径（先按当前页目录，再按站点根）
- GitHub Primer 风格：系统字体栈 / #0969da 蓝 / 全边框表格 / 代码块卡片（语言标签+复制）
- 亮色默认 + 亮/暗切换（localStorage 记忆，未设置时跟随系统偏好）
- 数学 KaTeX / Mermaid 浏览器端渲染（无网优雅降级；KaTeX 经 auto-render 处理 $$/$/\\(...\\)/\\[...\\]）
- 站内搜索（Ctrl+K 或 /，↑↓ 键盘导航，结果高亮 + Part 标签）
- 学习进度追踪（localStorage）
- 面包屑导航
用法：python build_site_lite.py [--serve [端口]]

版本历史：
- v0.0.1: 初始版本（review 分支未修改前）
- v1.0.0: 添加搜索、进度追踪、面包屑导航、知识脉络图、测验系统
- v1.1.0: UI 重构为 GitHub Primer 风格 + 代码块复制/语言标签 + 搜索键盘导航
- v1.2.0: 数学渲染 MathJax → KaTeX（0.18.x auto-render，首屏渲染更快；保留无网降级提示）
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

# 源码/文本文件 → 生成 GitHub 风格查看页（xxx.py → xxx.py.html，正文链接自动改写）
CODE_EXTS = {'.py': 'python', '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
             '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml', '.json': 'json',
             '.js': 'javascript', '.css': 'css', '.txt': '', '.cfg': 'ini',
             '.ini': 'ini', '.csv': '', '.ipynb': 'json'}
VIEWER_MAX_BYTES = 256 * 1024   # 超大文件不生成查看页（保留原文件直链）


def viewer_target(rel_cand):
    """若该文件应有查看页，返回 xxx.html 路径；否则 None。

    跳过 data/（大体量语料）与超过 256KB 的文件——这些保留原始直链，
    由 serve 端以 text/plain 呈现。
    """
    if rel_cand.startswith('data/'):
        return None
    ext = os.path.splitext(rel_cand)[1].lower()
    if ext not in CODE_EXTS:
        return None
    full = os.path.join(BUILD, 'docs', rel_cand)
    try:
        if os.path.getsize(full) > VIEWER_MAX_BYTES:
            return None
    except OSError:
        return None
    return rel_cand + '.html'

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
                out = highlight(code, lexer, HtmlFormatter(cssclass='highlight', nowrap=False))
                # data-lang 供前端代码卡头部显示语言标签
                return out.replace('<div class="highlight">',
                                   f'<div class="highlight" data-lang="{esc(lang)}">', 1)
        except Exception:
            pass
    return f'<pre><code>{esc(code)}</code></pre>'

# ---------- Markdown 渲染 ----------
def esc(s):
    return html.escape(s, quote=False)

PAGE_DIR = ''   # 当前渲染页目录（相对站点根），供 rewrite_link 使用

# 段内断行规则：连续行若以枚举序号（①-⑳、1、）或行首 emoji（提示符）开头，
# 视为新的逻辑点独立成段——课程 md 惯用"一行一个要点"的写法，
# 并段渲染会把 ①②③ 挤成一大段，不利阅读。
SEG_BREAK_RE = re.compile(
    r'^(?:[\u2460-\u2473\u3251-\u325F\u32B1-\u32BF]'   # ①-⑳ 等带圈/括号数字
    r'|\d{1,2}[\u3001\uFF0E]'                           # 1、 2．（全角顿号/句点）
    r'|[\u2300-\u27BF\u2B00-\u2BFF\U0001F000-\U0001FAFF][\uFE0F]?\s)'  # 行首 emoji + 空格
)

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
    elif cand:
        v = viewer_target(cand)
        if v:
            cand = v   # 源码/文本文件 → 对应查看页（viewer_target 已验证源文件存在且会生成）
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
        while i < len(lines):
            nxt = lines[i]
            if (not nxt.strip()
                    or re.match(r'^(#{1,6}\s|```|\s*>\s?|(\s*[-*+]\s|\s*\d+\.\s))', nxt)
                    or '|' in nxt
                    or nxt.lstrip().startswith('<')
                    or SEG_BREAK_RE.match(nxt.strip())):
                break   # 枚举行/提示行独立成段，其余按原规则断段
            buf.append(nxt)
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
<link rel="stylesheet" href="/_assets/style.css?v=37">
<script>
(function(){try{var t=localStorage.getItem('mm-theme');
if(!t&&window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)t='dark';
if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}})();
</script>
<link rel="stylesheet" href="{katex_css}" onerror="window.__noKatex=1">
<script defer src="{katex_js}" onerror="window.__noKatex=1"></script>
<script defer src="{katex_auto}"></script>
<script>
document.addEventListener('DOMContentLoaded',function(){
  function katexOffline(){document.querySelectorAll('.content p,.content li,.content td').forEach(function(e){if(e.textContent.indexOf('$')>=0){e.style.color='#9a6700';e.title='数学公式需要联网加载 KaTeX';}});}
  if(window.__noKatex||!window.renderMathInElement){katexOffline();return;}
  try{renderMathInElement(document.body,{delimiters:[
    {left:'$$',right:'$$',display:true},
    {left:'\\\\[',right:'\\\\]',display:true},
    {left:'\\\\(',right:'\\\\)',display:false},
    {left:'$',right:'$',display:false}
  ],throwOnError:false});}catch(err){katexOffline();}
});
</script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js" onerror="document.querySelectorAll('pre.mermaid').forEach(function(e){e.style.color='#9a6700';e.textContent='[离线] 流程图需要联网加载 Mermaid';});"></script>
<script>document.addEventListener('DOMContentLoaded',function(){if(window.mermaid)mermaid.initialize({startOnLoad:true});});</script>
</head>
<body>
<header class="topbar">
  <div class="topbar-l">
    <a class="brand" href="/index.html">🌱 <span>makemore 教程</span> <span class="version">v1.0.0</span></a>
    <button id="searchBtn" class="search-pill" title="搜索 (Ctrl+K 或 /)">
      <svg class="s-ico" aria-hidden="true" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-1.06 1.06ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"/></svg><span class="s-txt">搜索文档</span><kbd>/</kbd>
    </button>
  </div>
  <div class="tools">
    <button id="sideBtn" title="收起/展开目录">☰ 目录</button>
    <button id="fontMinus" title="减小字号">A−</button>
    <button id="fontReset" title="标准字号">A</button>
    <button id="fontPlus" title="增大字号">A+</button>
    <button id="themeBtn" title="切换亮/暗主题">🌙 夜间</button>
  </div>
</header>
<div id="searchModal" class="search-modal" style="display:none">
  <div class="search-box">
    <div class="search-input-row">
      <svg aria-hidden="true" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" class="search-input-ico"><path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-1.06 1.06ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"/></svg>
      <input id="searchInput" type="text" placeholder="搜索课程内容..." autocomplete="off">
      <kbd class="search-esc">esc</kbd>
    </div>
    <div id="searchResults" class="search-results"></div>
    <div class="search-foot"><span><kbd>↑</kbd><kbd>↓</kbd> 选择</span><span><kbd>↵</kbd> 打开</span><span><kbd>esc</kbd> 关闭</span></div>
  </div>
</div>
<div class="layout">
<nav class="side">{nav}</nav>
<main class="content">
<div class="breadcrumb">{breadcrumb}</div>
{body}
<div class="progress-bar">
  <span id="progressText" class="progress-text">学习进度</span>
  <button id="markComplete" class="btn-success">✓ 标记为已完成</button>
</div>
<nav class="pager">{pager}</nav>
</main>
<aside class="toc" id="toc"></aside>
</div>
<button id="backToTop" title="返回顶部">↑</button>
<script>
// 侧栏
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
document.querySelectorAll('.side').forEach(function(s){
  s.addEventListener('click',function(e){
    if(e.target.tagName==='A'&&isNarrow())document.body.classList.remove('side-open');
  });
});})();
// 字号
(function(){var LEVELS=5,KEY='mm-fs',idx=1;
try{var v=parseInt(localStorage.getItem(KEY));if(v>=0&&v<LEVELS)idx=v;}catch(e){}
function apply(){document.documentElement.setAttribute('data-fs',String(idx));
try{localStorage.setItem(KEY,idx);}catch(e){}}
document.getElementById('fontMinus').onclick=function(){idx=Math.max(0,idx-1);apply();};
document.getElementById('fontReset').onclick=function(){idx=1;apply();};
document.getElementById('fontPlus').onclick=function(){idx=Math.min(LEVELS-1,idx+1);apply();};
apply();})();
// 主题
(function(){var b=document.getElementById('themeBtn');
function paint(){var t=document.documentElement.getAttribute('data-theme');
b.textContent=t==='dark'?'☀️ 日间':'🌙 夜间';}
b.onclick=function(){var t=document.documentElement.getAttribute('data-theme');
var n=t==='dark'?'light':'dark';
document.documentElement.setAttribute('data-theme',n);
try{localStorage.setItem('mm-theme',n);}catch(e){}
paint();};
paint();})();
// 搜索（Ctrl+K / / 唤起，↑↓ 导航，Enter 打开，结果高亮 + Part 标签）
(function(){
  var modal=document.getElementById('searchModal');
  var input=document.getElementById('searchInput');
  var results=document.getElementById('searchResults');
  var searchBtn=document.getElementById('searchBtn');
  var index=null,sel=0,cur=null;
  function loadIndex(){
    if(index)return Promise.resolve(index);
    return fetch('/_assets/search-index.json').then(function(r){return r.json();}).then(function(d){index=d;return d;});
  }
  function openSearch(){modal.style.display='flex';input.value='';results.innerHTML='';sel=0;cur=null;
    setTimeout(function(){input.focus();},0);}
  function closeSearch(){modal.style.display='none';}
  searchBtn.onclick=openSearch;
  modal.onclick=function(e){if(e.target===modal)closeSearch();};
  function escRe(s){return s.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&');}
  function hi(text,q){
    try{var re=new RegExp('('+escRe(q)+')','ig');return text.replace(re,'<mark>$1</mark>');}
    catch(e){return text;}
  }
  function clip(s,q){
    var i=s.toLowerCase().indexOf(q.toLowerCase());
    if(i===-1)return s.substring(0,90);
    var start=Math.max(0,i-36),end=Math.min(s.length,i+q.length+54);
    return (start>0?'...':'')+s.substring(start,end)+(end<s.length?'...':'');
  }
  function render(items,q){
    if(!items.length){results.innerHTML='<div class="search-empty">没有找到相关内容，换个关键词试试</div>';cur=null;return;}
    results.innerHTML=items.map(function(item,n){
      var chip=item.part>0?'<span class="search-part">Part '+item.part+'</span>':'';
      return '<a href="'+item.url+'" data-n="'+n+'" class="search-result-item'+(n===0?' cur':'')+'">'
        +'<div class="search-result-title">'+chip+hi(item.title,q)+'</div>'
        +'<div class="search-result-snippet">'+hi(clip(item.snippet,q),q)+'</div></a>';
    }).join('');
    sel=0;cur=items[0];
    results.querySelectorAll('.search-result-item').forEach(function(el){
      el.addEventListener('mouseenter',function(){
        sel=+el.getAttribute('data-n');
        results.querySelectorAll('.search-result-item').forEach(function(x){x.classList.toggle('cur',x===el);});
        cur=el.getAttribute('href');
      });
    });
  }
  input.oninput=function(){
    var q=input.value.trim();
    if(!q){results.innerHTML='';cur=null;return;}
    loadIndex().then(function(data){
      render(data.filter(function(item){
        return item.title.toLowerCase().indexOf(q.toLowerCase())!==-1||item.snippet.toLowerCase().indexOf(q.toLowerCase())!==-1;
      }).slice(0,12),q);
    });
  };
  function move(d){
    var els=results.querySelectorAll('.search-result-item');
    if(!els.length)return;
    sel=(sel+d+els.length)%els.length;
    els.forEach(function(el,n){el.classList.toggle('cur',n===sel);});
    els[sel].scrollIntoView({block:'nearest'});
    cur=els[sel].getAttribute('href');
  }
  input.onkeydown=function(e){
    if(e.key==='ArrowDown'){e.preventDefault();move(1);}
    else if(e.key==='ArrowUp'){e.preventDefault();move(-1);}
    else if(e.key==='Enter'&&cur){e.preventDefault();window.location.href=cur;}
  };
  document.addEventListener('keydown',function(e){
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();}
    else if(e.key==='/'&&modal.style.display==='none'){
      var t=e.target.tagName;
      if(t!=='INPUT'&&t!=='TEXTAREA'&&!e.target.isContentEditable){e.preventDefault();openSearch();}
    }
    if(e.key==='Escape')closeSearch();
  });
})();
// 进度追踪
(function(){
  var KEY='mm-progress';var progress={};
  try{progress=JSON.parse(localStorage.getItem(KEY))||{};}catch(e){}
  var currentPath=window.location.pathname;
  var markBtn=document.getElementById('markComplete');
  var progressText=document.getElementById('progressText');
  function updateUI(){
    var done=Object.keys(progress).filter(function(k){return progress[k];}).length;
    progressText.textContent='已完成 '+done+' 个章节';
    if(progress[currentPath]){markBtn.textContent='✓ 已完成';markBtn.classList.add('done');}
    else{markBtn.textContent='✓ 标记为已完成';markBtn.classList.remove('done');}
  }
  markBtn.onclick=function(){
    if(progress[currentPath]){delete progress[currentPath];}
    else{progress[currentPath]=true;}
    try{localStorage.setItem(KEY,JSON.stringify(progress));}catch(e){}
    updateUI();
  };
  updateUI();
})();
// 代码块 → GitHub 式卡片（语言标签 + 复制按钮）
(function(){
  function copyText(code,btn){
    function ok(){btn.textContent='✓ 已复制';btn.classList.add('ok');
      setTimeout(function(){btn.textContent='复制';btn.classList.remove('ok');},1600);}
    function fallback(){var ta=document.createElement('textarea');ta.value=code;
      ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);
      ta.select();try{document.execCommand('copy');ok();}catch(e){}document.body.removeChild(ta);}
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(code).then(ok,fallback);
    }else{fallback();}
  }
  function wrap(block,lang){
    if(block.closest('.code-card')||block.closest('.file-view'))return;
    var card=document.createElement('div');card.className='code-card';
    var head=document.createElement('div');head.className='code-head';
    var name=document.createElement('span');name.className='code-lang';name.textContent=lang||'text';
    var btn=document.createElement('button');btn.className='code-copy';btn.type='button';btn.textContent='复制';
    btn.onclick=function(){copyText(block.querySelector('pre').innerText,btn);};
    head.appendChild(name);head.appendChild(btn);
    block.parentNode.insertBefore(card,block);
    card.appendChild(head);card.appendChild(block);
  }
  document.querySelectorAll('.content .highlight').forEach(function(h){wrap(h,h.getAttribute('data-lang'));});
  document.querySelectorAll('.content > pre').forEach(function(p){
    if(!p.classList.contains('mermaid'))wrap(p,'');
  });
})();
// 标题锚点 + 右侧本页目录（滚动高亮）+ 引用块 Alert 分色 + 图片点击放大
(function(){
  var used={};
  function slug(t){
    var s=t.trim().replace(/\\s+/g,'-').replace(/[^\\u4e00-\\u9fa5A-Za-z0-9_-]/g,'');
    if(!s)s='sec';
    if(used[s]){used[s]++;s+='-'+used[s];}else used[s]=1;
    return s;
  }
  var heads=[].slice.call(document.querySelectorAll('.content h2,.content h3'));
  var labels=heads.map(function(h){return h.textContent.trim();});
  heads.forEach(function(h,i){
    h.id=h.id||slug(labels[i]);
    var a=document.createElement('a');a.className='anchor';a.href='#'+h.id;a.textContent='#';
    h.insertBefore(a,h.firstChild);
  });
  var toc=document.getElementById('toc');
  if(toc&&heads.length>2){
    var html='<div class="toc-title">On this page</div>';
    heads.forEach(function(h,i){
      html+='<a href="#'+h.id+'" data-id="'+h.id+'"'+(h.tagName==='H3'?' class="lv3"':'')+'>'+labels[i]+'</a>';
    });
    toc.innerHTML=html;
    var links=[].slice.call(toc.querySelectorAll('a'));
    function spy(){
      var y=window.scrollY+140,cur='';
      heads.forEach(function(h){if(h.getBoundingClientRect().top+window.scrollY<=y)cur=h.id;});
      links.forEach(function(l){l.classList.toggle('cur',l.getAttribute('data-id')===cur);});
    }
    window.addEventListener('scroll',spy,{passive:true});spy();
  }
  document.querySelectorAll('.content blockquote').forEach(function(bq){
    var t=bq.textContent.trim();
    if(t.indexOf('⚠️')===0||t.indexOf('❗')===0)bq.classList.add('bq-warn');
    else if(t.indexOf('🔑')===0||t.indexOf('✅')===0)bq.classList.add('bq-key');
    else{
      var ems=['🎯','📖','💡','🎬','🎛','📝','⭐','🧭','📌'];
      for(var i=0;i<ems.length;i++){if(t.indexOf(ems[i])===0){bq.classList.add('bq-note');break;}}
    }
  });
  var ov=null;
  document.querySelectorAll('.content img').forEach(function(im){
    im.addEventListener('click',function(e){
      e.preventDefault();
      if(!ov){ov=document.createElement('div');ov.className='img-overlay';
        ov.appendChild(document.createElement('img'));document.body.appendChild(ov);
        ov.addEventListener('click',function(){ov.style.display='none';});}
      ov.querySelector('img').src=im.src;ov.style.display='flex';
    });
  });
  document.addEventListener('keydown',function(e){if(e.key==='Escape'&&ov)ov.style.display='none';});
})();
// 返回顶部
(function(){
  var btn=document.getElementById('backToTop');
  window.addEventListener('scroll',function(){btn.style.display=window.scrollY>300?'block':'none';});
  btn.onclick=function(){window.scrollTo({top:0,behavior:'smooth'});};
})();
</script>
</body>
</html>"""

CSS = """
/* === 设计系统：GitHub Primer 风格 ===
   字体/配色/组件对齐 github.com：系统字体栈 + Primer 色板 + 全边框表格 + 代码块卡片 */
:root, :root[data-theme=light]{
  --bg:#ffffff; --bg2:#ffffff; --card:#ffffff; --card2:#f6f8fa;
  --fg:#1f2328; --fg2:#1f2328; --fg3:#59636e; --fg4:#818b98;
  --line:#d1d9e0; --line2:#afb8c1;
  --accent:#0969da; --accent-subtle:#ddf4ff; --accent-soft:rgba(9,105,218,0.15);
  --success:#1f883d; --success-hover:#1a7f37;
  --btn-bg:#f6f8fa; --btn-bg-hover:#eef1f4;
  --btn-line:rgba(31,35,40,0.15); --btn-line-hover:rgba(31,35,40,0.25);
  --code-bg:rgba(175,184,193,0.2); --pre-bg:#f6f8fa; --pre-fg:#1f2328; --pre-line:#d1d9e0;
  --warn:#9a6700; --mark-bg:#fff8c5;
  --overlay:rgba(31,35,40,0.4);
  --mono:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono','PingFang SC','Microsoft YaHei',monospace;
  --topbar-h:56px;
}
:root[data-theme=dark]{
  --bg:#0d1117; --bg2:#010409; --card:#0d1117; --card2:#161b22;
  --fg:#e6edf3; --fg2:#e6edf3; --fg3:#8b949e; --fg4:#6e7681;
  --line:#30363d; --line2:#3d444d;
  --accent:#58a6ff; --accent-subtle:rgba(56,139,253,0.15); --accent-soft:rgba(56,139,253,0.15);
  --success:#238636; --success-hover:#2ea043;
  --btn-bg:#21262d; --btn-bg-hover:#262c36;
  --btn-line:#3d444d; --btn-line-hover:#525a64;
  --code-bg:rgba(110,118,129,0.4); --pre-bg:#161b22; --pre-fg:#e6edf3; --pre-line:#30363d;
  --warn:#d29922; --mark-bg:rgba(187,128,9,0.55);
  --overlay:rgba(1,4,9,0.6);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
:root[data-fs="0"]{font-size:87.5%}
:root[data-fs="1"]{font-size:100%}
:root[data-fs="2"]{font-size:112.5%}
:root[data-fs="3"]{font-size:125%}
:root[data-fs="4"]{font-size:137.5%}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;font-size:16px;font-weight:400;background:var(--bg);color:var(--fg2);line-height:1.6}
::selection{background:var(--accent-subtle)}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:var(--line2);border-radius:6px;border:1px solid var(--bg)}
::-webkit-scrollbar-thumb:hover{background:var(--fg4)}
::-webkit-scrollbar-track{background:transparent}
button{font-family:inherit}
button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
h1,h2,h3,h4{color:var(--fg);font-weight:600}
kbd{display:inline-block;padding:1px 6px;font-size:11px;font-family:var(--mono);color:var(--fg3);background:var(--bg);border:1px solid var(--line2);border-radius:6px;line-height:1.5}
/* === 顶栏（GitHub header 风）=== */
.topbar{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center;gap:12px;min-height:var(--topbar-h);padding:0 20px;background:var(--bg2);border-bottom:1px solid var(--line)}
:root[data-theme=light] .topbar{background:color-mix(in srgb, var(--bg2) 88%, transparent);backdrop-filter:blur(12px)}
.topbar-l{display:flex;align-items:center;gap:16px;min-width:0}
.brand{color:var(--fg);text-decoration:none;font-weight:600;font-size:15px;letter-spacing:-0.2px;white-space:nowrap}
.version{font-size:11px;font-weight:500;color:var(--fg3);margin-left:2px;border:1px solid var(--line);border-radius:999px;padding:0 8px;line-height:1.6}
.search-pill{display:flex;align-items:center;gap:8px;width:250px;min-height:32px;padding:0 10px;background:var(--card2);border:1px solid var(--line2);border-radius:6px;color:var(--fg3);font-size:14px;cursor:pointer;text-align:left}
.search-pill:hover{border-color:var(--accent)}
.search-pill .s-txt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.search-pill .s-ico{flex:none;color:var(--fg4)}
.tools{display:flex;gap:6px}
.tools button{background:var(--btn-bg);color:var(--fg);border:1px solid var(--btn-line);border-radius:6px;padding:3px 12px;cursor:pointer;font-size:14px;font-weight:500;min-height:32px;line-height:1.5}
.tools button:hover{background:var(--btn-bg-hover);border-color:var(--btn-line-hover)}
#themeBtn{min-width:88px}
#sideBtn{display:none}
@media (max-width:900px){
  #sideBtn{display:inline-block}
  .search-pill{width:auto;padding:0 9px}
  .search-pill .s-txt,.search-pill kbd{display:none}
  .tools button{padding:3px 8px;font-size:13px}
  #themeBtn{min-width:0}
}
/* === 布局 === */
.layout{display:flex;max-width:1280px;margin:0 auto}
.side{width:276px;flex:none;background:var(--bg);border-right:1px solid var(--line);padding:16px 12px;position:sticky;top:var(--topbar-h);height:calc(100vh - var(--topbar-h));overflow-y:auto;scrollbar-width:thin}
body.no-side .side{display:none}
body.no-side .layout{max-width:980px}
.nav-home{margin-bottom:12px;padding-left:8px}
.nav-home a{color:var(--fg);font-weight:600;font-size:14px}
.nav-sec{margin:2px 0}
.nav-sec summary{list-style:none;padding:5px 8px;cursor:pointer;user-select:none;font-size:14px;font-weight:600;color:var(--fg);line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-radius:6px}
.nav-sec summary:hover{color:var(--accent)}
details.nav-sec summary::-webkit-details-marker{display:none}
details.nav-sec summary::before{content:'▸ ';color:var(--fg4)}
details.nav-sec[open] summary::before{content:'▾ '}
.nav-item a{display:block;color:var(--fg3);padding:4px 8px 4px 22px;border-radius:6px;font-size:14px;font-weight:400;text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.5}
.nav-item a:hover{color:var(--fg);background:var(--card2)}
.nav-item a.cur{color:var(--accent);background:var(--accent-subtle);font-weight:600}
/* === 内容区（GitHub markdown-body 风）=== */
.content{flex:1;min-width:0;max-width:820px;margin:0 auto;padding:32px 40px 80px}
.content h1{font-size:2em;font-weight:600;line-height:1.25;border-bottom:1px solid var(--line);padding-bottom:.3em;margin:16px 0 24px;color:var(--fg)}
.content h2{font-size:1.5em;font-weight:600;margin:28px 0 16px;padding-bottom:.3em;border-bottom:1px solid var(--line);color:var(--fg)}
.content h3{font-size:1.25em;font-weight:600;margin:24px 0 16px;color:var(--fg)}
.content h4{font-size:1em;font-weight:600;margin:24px 0 16px;color:var(--fg)}
.content p{margin:16px 0}
.content a{color:var(--accent);text-decoration:none;font-weight:400}
.content a:hover{text-decoration:underline}
.content strong{font-weight:600;color:var(--fg)}
.content ul,.content ol{padding-left:2em;margin:16px 0}
.content li{margin:4px 0}
.content code{background:var(--code-bg);padding:.2em .4em;border-radius:6px;font-size:85%;font-family:var(--mono)}
.content pre{background:var(--pre-bg);color:var(--pre-fg);border:1px solid var(--pre-line);border-radius:6px;padding:16px;overflow-x:auto;line-height:1.45;font-size:85%;font-family:var(--mono)}
.content pre code{background:none;padding:0;font-size:100%;border-radius:0}
.content blockquote{color:var(--fg3);border-left:.25em solid var(--line2);background:transparent;margin:16px 0;padding:0 1em}
.content blockquote p{margin:8px 0}
.tbl-wrap{overflow-x:auto;margin:16px 0}
table.md-table{border-collapse:collapse;width:100%;font-size:16px;margin:0}
.md-table th,.md-table td{border:1px solid var(--line);padding:6px 13px;text-align:left}
.md-table th{background:var(--card2);font-weight:600;color:var(--fg)}
.md-table tr:last-child td{border-bottom:1px solid var(--line)}
details{border:1px solid var(--line);border-radius:6px;padding:8px 16px;margin:16px 0;background:transparent}
summary{cursor:pointer;color:var(--fg);font-weight:600}
details[open] summary{border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:8px;border-radius:0}
hr{border:none;height:1px;background:var(--line);margin:24px 0}
.derivation{background:var(--card2);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;padding:6px 20px 12px;margin:20px 0}
.derivation .d-title{font-weight:600;color:var(--accent);margin:12px 0 4px}
img{border-radius:6px;border:1px solid var(--line);max-width:100%}
/* === 代码块卡片（语言标签 + 复制）=== */
.code-card{border:1px solid var(--pre-line);border-radius:6px;margin:16px 0;background:var(--pre-bg);overflow:hidden}
.code-head{display:flex;justify-content:space-between;align-items:center;padding:5px 12px;background:var(--card2);border-bottom:1px solid var(--pre-line)}
.code-lang{font-family:var(--mono);font-size:12px;font-weight:500;color:var(--fg3);text-transform:lowercase}
.code-copy{font-size:12px;font-weight:500;color:var(--fg3);background:var(--btn-bg);border:1px solid var(--btn-line);border-radius:6px;padding:2px 10px;cursor:pointer;line-height:1.6}
.code-copy:hover{color:var(--fg);border-color:var(--btn-line-hover)}
.code-copy.ok{color:var(--success);border-color:var(--success)}
.code-card .highlight,.code-card pre{margin:0;border:none;border-radius:0;background:transparent;padding:12px 16px}
.content .highlight{background:var(--pre-bg);border:1px solid var(--pre-line);border-radius:6px;margin:16px 0}
.content .highlight pre{margin:0;border:none;background:transparent}
.highlight code{font-family:inherit;font-size:100%;background:none;padding:0}
.highlight pre,pre.mermaid{font-family:var(--mono)}
pre.mermaid{display:flex;justify-content:center;background:var(--pre-bg);border:1px solid var(--pre-line);border-radius:6px;padding:16px;overflow-x:auto}
/* Pygments：GitHub 语法配色 */
:root[data-theme=light] .highlight .k,:root[data-theme=light] .highlight .kd,:root[data-theme=light] .highlight .kn,:root[data-theme=light] .highlight .ow,:root[data-theme=light] .highlight .kr{color:#cf222e}
:root[data-theme=light] .highlight .s1,:root[data-theme=light] .highlight .s2,:root[data-theme=light] .highlight .sa,:root[data-theme=light] .highlight .sd,:root[data-theme=light] .highlight .se{color:#0a306c}
:root[data-theme=light] .highlight .mi,:root[data-theme=light] .highlight .mf,:root[data-theme=light] .highlight .mh,:root[data-theme=light] .highlight .il{color:#0550ae}
:root[data-theme=light] .highlight .c1,:root[data-theme=light] .highlight .ch,:root[data-theme=light] .highlight .cm{color:#59636e}
:root[data-theme=light] .highlight .nf,:root[data-theme=light] .highlight .fm{color:#8250df}
:root[data-theme=light] .highlight .nb,:root[data-theme=light] .highlight .bp{color:#953800}
:root[data-theme=light] .highlight .o,:root[data-theme=light] .highlight .p{color:#1f2328}
:root[data-theme=light] .highlight .nn,:root[data-theme=light] .highlight .nc{color:#953800}
:root[data-theme=light] .highlight .nd{color:#8250df}
:root[data-theme=dark] .highlight .k,:root[data-theme=dark] .highlight .kd,:root[data-theme=dark] .highlight .kn,:root[data-theme=dark] .highlight .ow,:root[data-theme=dark] .highlight .kr{color:#ff7b72}
:root[data-theme=dark] .highlight .s1,:root[data-theme=dark] .highlight .s2,:root[data-theme=dark] .highlight .sa,:root[data-theme=dark] .highlight .sd,:root[data-theme=dark] .highlight .se{color:#a5d6ff}
:root[data-theme=dark] .highlight .mi,:root[data-theme=dark] .highlight .mf,:root[data-theme=dark] .highlight .mh,:root[data-theme=dark] .highlight .il{color:#79c0ff}
:root[data-theme=dark] .highlight .c1,:root[data-theme=dark] .highlight .ch,:root[data-theme=dark] .highlight .cm{color:#8b949e}
:root[data-theme=dark] .highlight .nf,:root[data-theme=dark] .highlight .fm{color:#d2a8ff}
:root[data-theme=dark] .highlight .nb,:root[data-theme=dark] .highlight .bp{color:#ffa657}
:root[data-theme=dark] .highlight .o,:root[data-theme=dark] .highlight .p{color:#c9d1d9}
:root[data-theme=dark] .highlight .nn,:root[data-theme=dark] .highlight .nc{color:#ffa657}
:root[data-theme=dark] .highlight .nd{color:#d2a8ff}
/* === 面包屑 === */
.breadcrumb{font-size:14px;color:var(--fg3);margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--line)}
.breadcrumb a{color:var(--fg3);text-decoration:none}
.breadcrumb a:hover{color:var(--accent);text-decoration:underline}
.breadcrumb span{margin:0 7px;color:var(--fg4)}
/* === 进度 / 分页 / 返回顶部 === */
.progress-bar{margin-top:36px;padding:12px 16px;background:var(--card2);border:1px solid var(--line);border-radius:6px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.progress-text{font-size:14px;color:var(--fg3)}
.btn-success{padding:3px 16px;background:var(--success);color:#ffffff;border:1px solid rgba(31,35,40,0.15);border-radius:6px;cursor:pointer;font-size:14px;font-weight:500;min-height:32px}
.btn-success:hover{background:var(--success-hover)}
.btn-success.done{background:var(--card2);color:var(--fg3);border:1px solid var(--line2)}
.pager{display:flex;justify-content:space-between;margin-top:40px;border-top:1px solid var(--line);padding-top:20px;gap:12px}
.pager a{color:var(--fg);text-decoration:none;padding:3px 16px;border:1px solid var(--btn-line);border-radius:6px;background:var(--btn-bg);font-size:14px;font-weight:500;min-height:32px;display:inline-flex;align-items:center}
.pager a:hover{background:var(--btn-bg-hover);border-color:var(--btn-line-hover)}
#backToTop{display:none;position:fixed;bottom:24px;right:24px;width:40px;height:40px;background:var(--btn-bg);color:var(--fg);border:1px solid var(--btn-line);border-radius:6px;cursor:pointer;font-size:16px;z-index:100;box-shadow:0 1px 3px rgba(0,0,0,0.12)}
#backToTop:hover{background:var(--btn-bg-hover);border-color:var(--btn-line-hover)}
/* === 搜索（GitHub command palette 风）=== */
.search-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:var(--overlay);z-index:1000;display:flex;justify-content:center;padding:12vh 16px 0}
.search-box{width:100%;max-width:640px;background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 24px 48px -12px rgba(0,0,0,0.4);overflow:hidden;max-height:64vh;display:flex;flex-direction:column}
.search-input-row{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--line)}
.search-input-ico{flex:none;color:var(--fg4)}
.search-input-row input{flex:1;border:none;font-size:15px;background:transparent;color:var(--fg);outline:none;font-family:inherit}
.search-input-row input::placeholder{color:var(--fg4)}
.search-results{overflow-y:auto;padding:8px;flex:1}
.search-result-item{display:block;padding:10px 12px;text-decoration:none;border-radius:6px;margin-bottom:2px}
.search-result-item.cur{background:var(--accent-subtle)}
.search-result-title{font-weight:600;color:var(--fg);font-size:14px;margin-bottom:3px}
.search-part{font-size:11px;font-weight:500;color:var(--accent);background:var(--accent-subtle);border-radius:999px;padding:0 8px;margin-right:8px;vertical-align:1px}
:root[data-theme=dark] .search-part{background:rgba(56,139,253,0.15)}
.search-result-snippet{font-size:13px;color:var(--fg3);line-height:1.5}
.search-result-item.cur .search-result-snippet{color:var(--fg2)}
mark{background:var(--mark-bg);color:inherit;border-radius:3px;padding:0 1px}
.search-empty{padding:28px 12px;text-align:center;color:var(--fg3);font-size:14px}
.search-foot{display:flex;gap:18px;padding:8px 16px;border-top:1px solid var(--line);font-size:12px;color:var(--fg4);background:var(--card2)}
.search-foot kbd{margin-right:3px}
/* === 标题锚点（GitHub hover # 风）=== */
.content h2 .anchor,.content h3 .anchor{float:left;margin-left:-1.4em;padding-right:.35em;color:var(--fg4);opacity:0;font-size:.8em;line-height:inherit;text-decoration:none;font-weight:400}
.content h2:hover .anchor,.content h3:hover .anchor{opacity:1;color:var(--accent)}
/* === 引用块 Alert 分色（按首 emoji 自动识别）=== */
.content blockquote.bq-note{border-left-color:var(--accent)}
.content blockquote.bq-warn{border-left-color:var(--warn)}
.content blockquote.bq-key{border-left-color:var(--success)}
/* === 右侧本页目录（宽屏显示）=== */
.toc{display:none}
@media (min-width:1360px){
  .layout{max-width:1460px}
  body.no-side .layout{max-width:1100px}
  .toc{display:block;width:250px;flex:none;padding:32px 12px 80px;position:sticky;top:var(--topbar-h);height:calc(100vh - var(--topbar-h));overflow-y:auto;font-size:13px;scrollbar-width:thin}
}
.toc-title{font-weight:600;color:var(--fg);margin-bottom:10px;font-size:14px}
.toc a{display:block;color:var(--fg3);text-decoration:none;padding:3px 0 3px 12px;border-left:2px solid transparent;line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.toc a:hover{color:var(--accent)}
.toc a.cur{color:var(--accent);border-left-color:var(--accent);font-weight:500}
.toc a.lv3{padding-left:26px;font-size:12.5px}
/* === 图片点击放大 === */
.content img{cursor:zoom-in}
.img-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:var(--overlay);z-index:1100;display:flex;align-items:center;justify-content:center;cursor:zoom-out}
.img-overlay img{max-width:92vw;max-height:92vh;border-radius:6px;box-shadow:0 24px 48px -12px rgba(0,0,0,0.5)}
/* === 源码查看页（xxx.py → xxx.py.html）=== */
.file-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:7px 12px;background:var(--card2);border:1px solid var(--line);border-bottom:none;border-radius:6px 6px 0 0}
.file-name{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--fg);word-break:break-all}
.file-raw{font-size:12px;font-weight:500;color:var(--fg3);text-decoration:none;border:1px solid var(--btn-line);border-radius:6px;padding:2px 10px;background:var(--btn-bg);white-space:nowrap;flex:none}
.file-raw:hover{color:var(--fg);border-color:var(--btn-line-hover)}
.file-view .highlight{margin:0;border-radius:0 0 6px 6px}
.file-view .highlight pre{padding:12px 0 12px 16px}
/* === 窄屏 === */
@media (max-width:900px){
  .layout{display:block}
  .side{display:none;position:fixed;top:var(--topbar-h);left:0;right:0;bottom:0;width:auto;height:auto;z-index:20;
        background:var(--bg);padding:12px;border-right:none;box-shadow:0 8px 30px rgba(0,0,0,0.12)}
  body.side-open .side{display:block}
  body.side-open{overflow:hidden}
  .content{max-width:100%;padding:18px 16px 70px}
  .content h1{font-size:1.5em}
  .content h2{font-size:1.3em}
  .tbl-wrap{margin:12px -16px;width:calc(100% + 32px)}
  .pager{flex-direction:column}
  .pager a{text-align:center;justify-content:center}
  .topbar{padding:0 12px}
  .brand .version{display:none}
}
@media (max-width:480px){
  .content{padding:16px 12px 60px}
  .tbl-wrap{margin:12px -12px;width:calc(100% + 24px)}
}
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

def extract_title(text, fallback):
    """从 Markdown 内容提取第一个 H1 标题，否则用 fallback。"""
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('# '):
            # 去掉 # 和 emoji 前缀
            title = line[2:].strip()
            title = re.sub(r'^[\U0001F300-\U0001FAFF\u2600-\u27BF]+\s*', '', title)
            return title if title else fallback
    return fallback

def build():
    docs = merge_tree()
    site = os.path.join(BUILD, 'site_html')
    os.makedirs(site, exist_ok=True)
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

    katex_base = 'https://cdn.jsdelivr.net/npm/katex@0.18.7/dist'  # 本地包缺失，CDN + 无网降级
    katex_css, katex_js, katex_auto = (katex_base + '/katex.min.css', katex_base + '/katex.min.js',
                                       katex_base + '/contrib/auto-render.min.js')

    def make_breadcrumb(rel):
        """生成面包屑导航 HTML。"""
        parts = ['<a href="/index.html">首页</a>']
        if rel == 'index.md':
            return ''
        # 解析路径层级
        segs = rel.replace('.md', '').split('/')
        if segs[0] == 'courses' and len(segs) >= 2:
            m = re.match(r'Part(\d+)', segs[1])
            if m:
                num = int(m.group(1))
                parts.append(f'<a href="/courses/{segs[1]}/tutorial/README.html">Part {num} · {PART_TITLES.get(num, "")}</a>')
                if len(segs) >= 4 and segs[2] == 'tutorial':
                    title = extract_title(open(os.path.join(docs, rel), encoding='utf-8').read(), segs[3])
                    parts.append(esc(title))
        elif segs[0] == 'assignments':
            parts.append('课后作业')
            if len(segs) >= 2:
                parts.append(esc(segs[1].capitalize()))
        elif segs[0] == 'docs':
            parts.append('参考文档')
            title = extract_title(open(os.path.join(docs, rel), encoding='utf-8').read(), segs[-1])
            parts.append(esc(title))
        elif segs[0] == 'maps':
            parts.append('知识脉络图')
        elif segs[0] == 'quizzes':
            parts.append('测验系统')
        return ' <span>/</span> '.join(parts)

    global PAGE_DIR
    n_ok = 0
    for idx, rel in enumerate(pages):
        PAGE_DIR = os.path.dirname(rel)
        text = open(os.path.join(docs, rel), encoding='utf-8').read()
        body = render_blocks(text)
        title = '课程首页' if rel == 'index.md' else extract_title(text, os.path.basename(rel)[:-3])
        breadcrumb = make_breadcrumb(rel)
        pager = ''
        if idx > 0:
            pager += f'<a href="/{pages[idx-1][:-3]}.html">← {esc(os.path.basename(pages[idx-1])[:-3])}</a>'
        if idx < len(pages) - 1:
            pager += f'<a href="/{pages[idx+1][:-3]}.html">{esc(os.path.basename(pages[idx+1])[:-3])} →</a>'
        html_out = (PAGE.replace('{title}', esc(title)).replace('{nav}', nav_html(rel))
                        .replace('{breadcrumb}', breadcrumb)
                        .replace('{body}', body).replace('{pager}', pager)
                        .replace('{katex_css}', katex_css).replace('{katex_js}', katex_js)
                        .replace('{katex_auto}', katex_auto))
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
    # 源码/文本文件 → GitHub 风格查看页（正文里的 xxx.py 链接已在 rewrite_link 改指到此处）
    n_view = 0
    for dp, dirs, fs in os.walk(docs):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache')]
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), docs).replace(os.sep, '/')
            tgt = viewer_target(rel)
            if not tgt:
                continue
            PAGE_DIR = os.path.dirname(rel)
            try:
                text = open(os.path.join(docs, rel), encoding='utf-8').read()
            except UnicodeDecodeError:
                text = open(os.path.join(docs, rel), encoding='utf-8', errors='replace').read()
            lang = CODE_EXTS[os.path.splitext(rel)[1].lower()]
            fname = os.path.basename(rel)
            body = ('<div class="file-view">'
                    f'<div class="file-head"><span class="file-name">{esc(fname)}</span>'
                    f'<a class="file-raw" href="/{rel}">⬇ 原始文件</a></div>'
                    + code_block(text, lang) + '</div>')
            breadcrumb = make_breadcrumb(rel) or '<a href="/index.html">首页</a>'
            breadcrumb += f' <span>/</span> {esc(fname)}'
            html_out = (PAGE.replace('{title}', esc(fname)).replace('{nav}', nav_html(tgt))
                        .replace('{breadcrumb}', breadcrumb)
                        .replace('{body}', body).replace('{pager}', '')
                        .replace('{katex_css}', katex_css).replace('{katex_js}', katex_js)
                        .replace('{katex_auto}', katex_auto))
            open(os.path.join(site, tgt), 'w', encoding='utf-8').write(html_out)
            n_view += 1
    # 生成搜索索引
    search_index = []
    for rel in pages:
        text = open(os.path.join(docs, rel), encoding='utf-8').read()
        title = '课程首页' if rel == 'index.md' else extract_title(text, os.path.basename(rel)[:-3])
        # 提取纯文本（去掉 markdown 语法）
        plain = re.sub(r'```[\s\S]*?```', '', text)  # 去代码块
        plain = re.sub(r'`[^`]+`', '', plain)  # 去行内代码
        plain = re.sub(r'!\[.*?\]\(.*?\)', '', plain)  # 去图片
        plain = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', plain)  # 链接保留文字
        plain = re.sub(r'[#*_~>|]', '', plain)  # 去 markdown 符号
        plain = re.sub(r'\s+', ' ', plain).strip()
        # 取前500字作为摘要
        snippet = plain[:500]
        # 确定所属 Part
        part_match = re.match(r'courses/Part(\d+)', rel)
        part_num = int(part_match.group(1)) if part_match else 0
        search_index.append({
            'url': '/' + rel[:-3] + '.html',
            'title': title,
            'snippet': snippet,
            'part': part_num
        })
    import json
    assets = os.path.join(site, '_assets')
    os.makedirs(assets, exist_ok=True)
    open(os.path.join(assets, 'search-index.json'), 'w', encoding='utf-8').write(json.dumps(search_index, ensure_ascii=False))
    open(os.path.join(assets, 'style.css'), 'w', encoding='utf-8').write(CSS)
    write_favicon(site)
    try:
        print(f'✅ 渲染 {n_ok} 个页面 + {n_view} 个源码查看页 → {site}')
    except UnicodeEncodeError:
        print(f'[OK] 渲染 {n_ok} 个页面 + {n_view} 个源码查看页 -> {site}')
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
