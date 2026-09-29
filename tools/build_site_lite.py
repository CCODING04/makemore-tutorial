#!/usr/bin/env python3
"""零依赖课程站点渲染器 v3.3（全站统一首页设计语言：渐变徽章/胶囊/eyebrow）。

REPO 为底 + REVIEW 覆盖 → 合并树 → 静态 HTML（site_build/site_html/）。
- 站内 .md 链接在渲染期解析为站点绝对路径（先按当前页目录，再按站点根）
- GitHub Primer 风格：系统字体栈 / #0969da 蓝 / 全边框表格 / 代码块卡片（语言标签+复制）
- 亮色默认 + 亮/暗切换（localStorage 记忆，未设置时跟随系统偏好）
- 数学 KaTeX / Mermaid 浏览器端渲染（无网优雅降级；KaTeX 经 auto-render 处理 $$/$/\\(...\\)/\\[...\\]）
- 站内搜索（Ctrl+K 或 /，↑↓ 键盘导航，结果高亮 + Part 标签）
- 学习进度追踪（localStorage）
- 面包屑导航
用法：python build_site_lite.py [--serve [端口]] [--watch [端口]] [--force]
  默认      构建站点（源文件无改动时自动跳过）
  --serve   构建后启动本地预览（改动检测同上）
  --watch   构建后启动预览并持续监听源文件，改动即自动重建（改完 md 存盘 → 刷新浏览器即见）
  --force   忽略改动检测，强制全量重建

版本历史：
- v0.0.1: 初始版本（review 分支未修改前）
- v1.0.0: 添加搜索、进度追踪、面包屑导航、知识脉络图、测验系统
- v1.1.0: UI 重构为 GitHub Primer 风格 + 代码块复制/语言标签 + 搜索键盘导航
- v1.2.0: 数学渲染 MathJax → KaTeX（0.18.x auto-render，首屏渲染更快；保留无网降级提示）
- v1.3.0: 源文件指纹增量检测——无改动跳过重建；--watch 服务中监听 md/图片改动自动重建；--force 强制全量
- v1.3.1: 侧栏导航一键「展开全部/收起全部」按钮（localStorage 跨页记忆展开偏好）
- v1.3.2: 展开按钮与「课程首页」同行；☰ 目录按钮桌面端可见——整个侧栏可收起聚焦内容（localStorage 记忆）
- v1.4.0: 信息流可视化体系——```plot 页内交互函数图（悬停读数/描线动画/主题自适应）、```chunk 训练样本
  可视化（token 色块）、═══/=== 横幅输出块自动转「运行输出」卡、text/图示块隐藏语言标签、
  ```viz authored 可视化组件四型（flow 流程 / tree 树 / panel 模块面板 / highway 残差流主干，
  语义分色 input/op/green/mid/output + 图例 + src 原文一致性校验与过期提示）；
  对齐表格 text 块自动转 HTML 表格。生成方法沉淀为 .zcode/skills/viz-block（含踩坑实录）
- v1.4.1: ```widget 页内 iframe 嵌入 widgets/ 独立交互组件（组件名 + 可选高度；文件缺失回退提示框）；
  widgets 主题跟随站点亮/暗切换（同源读取父页 data-theme，独立打开回退系统偏好）
- v1.4.2: widget 独立页右上角注入「↩ 返回章节」按钮（构建期按 WIDGET_HOME 映射注入，源文件不动；
  iframe 内嵌时自动隐藏）
- v1.6.0: 全站统一首页（hero/Part 卡片）设计语言——主渐变 token grad-brand（蓝→绿）贯穿；顶栏品牌渐变徽章 +
  搜索/工具按钮胶囊化 + 聚焦光晕；侧栏 Part 数字徽章（nav-badge，全完成转绿 §/✎ 中性徽章）+ 渐变主胶囊课程首页按钮 +
  当前项渐变条；面包屑中段 chip 化（eyebrow 徽章）；正文 h2 渐变竖条、行内 code 无边框胶囊、代码卡语言徽章 + 透明头、
  表格现代行线风（圆角外框 + 表头浅蓝底 + 行 hover）、hr 渐变线；页脚完成区 hero 化（渐变底 + 渐变胶囊按钮）；
  pager hover 渐变边条、TOC 渐变当前条
- v1.5.0: UI 升级为现代课程风——跨文档 View Transitions 页面过渡（正文命名 content，框架静止仅正文动；
  主题切换圆形揭示 types:['theme']；不支持 VT 的浏览器由 no-vt 标记走正文进场动画兜底）；
  设计 token 扩充（radius/shadow/side-bg/brand）；侧栏微底色 + chevron 旋转 + 当前项指示条 + 完成态绿勾
  + Part 计数胶囊（运行时按 localStorage 进度）；pager 卡片化（真实标题 + hover 动效）；卡片族统一圆角阴影；
  引用块 Alert 浅色 wash；表格行 hover；阅读进度条 + 返回顶部进度环；搜索弹窗入场动画；窄屏侧栏抽屉平移；
  首页 hero + Part 课程地图网格（构建期统计 + courseMap JSON + 继续学习/进度环运行时填充）
- v1.4.3: 排版对齐 Typora 默认主题 github.css 实测规格——字体栈 Open Sans 打头（Typora 自带字体）、
  正文 #333 + antialiased、行高 1.6 / 无字距、p 与列表等 0.8em 边距、标题 bold + 1rem 边距
  （h1 2.25em / h2 1.75em / h3 1.5em / h4 1.25em）、行内 code 0.9em+边框芯片式、
  代码块 0.9em、内容宽 860px（#write 规格）、blockquote 4px 边框 15px 内边距、表格 th/td 6px 13px；块级间距改 GitHub 式单向 margin——块统一 16px 下边距、标题 24px 上边距、li+li 0.25em、首元素去上边距
"""
import hashlib
import html
import json
import os
import re
import shutil
import sys
import threading
import time

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
STAMP = os.path.join(BUILD, '.last_build.json')   # 上次构建的源指纹（机器本地状态，不入库）

# 建站源树：merge_tree 与改动指纹共用这份清单
SRC_TREES = ('courses', 'assignments', 'widgets', 'docs', 'assignment_reference', 'tools', 'maps', 'quizzes')

# widget 独立页 → 所属教程章节（返回按钮的目标）。以教程页实际引用为准；iframe 内嵌时按钮自动隐藏。
WIDGET_HOME = {
    'attention_heatmap':    ('/courses/Part6_transformer/tutorial/02_attention_from_scratch.html', 'P6·02 注意力'),
    'anim_attention_flow':  ('/courses/Part6_transformer/tutorial/02_attention_from_scratch.html', 'P6·02 注意力'),
    'softmax_temperature':  ('/courses/Part1_bigrams/tutorial/02_bigram_model.html', 'P1·02 Bigram'),
    'kv_cache_memory':      ('/courses/Part14_inference_vllm/tutorial/02_vllm_serving.html', 'P14·02 vLLM'),
    'anim_kv_cache_growth': ('/courses/Part14_inference_vllm/tutorial/02_vllm_serving.html', 'P14·02 vLLM'),
    'pipeline_bubble':      ('/courses/Part10_distributed/tutorial/04_tp_pp_and_beyond.html', 'P10·04 TP/PP'),
    'lora_inject':          ('/courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.html', 'P12·01 LoRA'),
    'ddpm_schedule':        ('/courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.html', 'P16·01 DDPM'),
    'anim_ddpm_diffusion':  ('/courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.html', 'P16·01 DDPM'),
    'lsh_s_curve':          ('/courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.html', 'P13·01 去重'),
    'moe_aux_loss':         ('/courses/Part7_minimind/tutorial/03_gqa_and_ffn.html', 'P7·03 MoE'),
    'dpo':                  ('/courses/Part7_minimind/tutorial/04_training_pipeline.html', 'P7·04 流水线'),
    'transformer_block_archify': ('/courses/Part6_transformer/tutorial/03_transformer_block.html', 'P6·03 Block'),
    'mlp_archify':          ('/courses/Part2_mlp/tutorial/02_mlp_architecture.html', 'P2·02 MLP'),
    'attention_qkv_archify': ('/courses/Part6_transformer/tutorial/02_attention_from_scratch.html', 'P6·02 注意力'),
    'gpt2_archify':         ('/courses/Part8_post_training/tutorial/01_gpt_and_pretrain.html', 'P8·01 GPT'),
    'verl_roles_archify':   ('/courses/Part11_alignment_verl/tutorial/02_verl_quickstart.html', 'P11·02 verl'),
    'lora_archify':         ('/courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.html', 'P12·01 LoRA'),
    'minhash_archify':      ('/courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.html', 'P13·01 去重'),
    'hybrid_rag_archify':   ('/courses/Part18_rag/tutorial/01_naive_to_hybrid.html', 'P18·01 RAG'),
    'agent_loop_archify':   ('/courses/Part19_agents/tutorial/01_agent_loop.html', 'P19·01 Agent'),
}


def inject_widget_back(dst_f):
    """构建期向 site 里的 widget 独立页注入右上角「返回章节」按钮（源 widgets/ 不动）。

    target=_parent 使按钮即便被误放进 iframe 也能跳转父页；被 ```widget 内嵌时脚本将其隐藏。
    """
    name = os.path.splitext(os.path.basename(dst_f))[0]
    home = WIDGET_HOME.get(name)
    if not home:
        return
    href, label = home
    with open(dst_f, encoding='utf-8') as f:
        text = f.read()
    if 'backToPart' in text or '</body>' not in text:
        return
    snippet = (
        '<a id="backToPart" href="' + href + '" target="_parent" title="返回对应章节">↩ ' + label + '</a>'
        '<style>#backToPart{position:fixed;top:10px;right:10px;z-index:9999;'
        'font:500 12.5px/1 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;'
        'padding:8px 13px;border:1px solid rgba(31,35,40,.15);border-radius:999px;'
        'background:#ffffff;color:#0969da;text-decoration:none;box-shadow:0 1px 6px rgba(20,30,70,.18)}'
        ':root[data-theme=dark] #backToPart{background:#21262d;border-color:#3d444d;color:#58a6ff}</style>'
        '<script>(function(){try{if(window.parent&&window.parent!==window){'
        'document.getElementById("backToPart").style.display="none";}}catch(e){}})();</script>'
    )
    text = text.replace('</body>', snippet + '</body>', 1)
    with open(dst_f, 'w', encoding='utf-8') as f:
        f.write(text)


# 主题跟随站点切换的 widget（iframe 内默认配色同步父页 data-theme，独立打开不受影响）
THEME_SYNC_WIDGETS = {
    'transformer_block_archify', 'mlp_archify', 'attention_qkv_archify', 'gpt2_archify',
    'verl_roles_archify', 'lora_archify', 'minhash_archify', 'hybrid_rag_archify',
    'agent_loop_archify',
}


def inject_widget_theme(dst_f):
    """构建期向 site 里的 widget 独立页注入「主题跟随」脚本（源 widgets/ 不动）。

    仅在被 iframe 内嵌时生效：读取父页 <html data-theme>，调用 Archify.theme.apply 同步配色，
    并隐藏 iframe 内的主题切换按钮（独立切换只在独立打开时提供）；父页切换夜间模式实时跟随。
    """
    with open(dst_f, encoding='utf-8') as f:
        text = f.read()
    if 'archifyThemeSync' in text or '</body>' not in text:
        return
    snippet = (
        '<script id="archifyThemeSync">(function(){try{'
        'if(!window.parent||window.parent===window)return;'
        'var pdoc=window.parent.document;'
        'function pm(){var t=pdoc.documentElement.getAttribute("data-theme");'
        'return t==="dark"?"dark":(t==="light"?"light":null);}'
        'function sync(){var t=pm();if(!t)return;'
        'try{if(window.Archify&&Archify.theme&&Archify.theme.apply){'
        'Archify.theme.apply(t);'
        'try{localStorage.removeItem("archify-theme");}catch(_){}}'
        'else{document.documentElement.setAttribute("data-theme",t);}}catch(_){}}'
        'sync();'
        'try{new MutationObserver(sync).observe(pdoc.documentElement,'
        '{attributes:true,attributeFilter:["data-theme"]});}catch(_){}'
        'var b=document.getElementById("btn-theme");'
        'if(b){b.style.display="none";}'
        '}catch(e){}})();</script>'
    )
    text = text.replace('</body>', snippet + '</body>', 1)
    with open(dst_f, 'w', encoding='utf-8') as f:
        f.write(text)


def src_fingerprint():
    """源文件指纹：SRC_TREES + README.md 的（相对路径, 大小, mtime）SHA256。

    任何 md/图片/源码的新增、修改、删除都会改变指纹；~2000 个文件 stat 扫描 <0.1s。
    注意 site_build/ 不在源树内，构建产物不会反馈进指纹（无自激振荡）。
    """
    h = hashlib.sha256()
    n = 0
    for name in ['README.md'] + list(SRC_TREES):
        p = os.path.join(REPO_ROOT, name)
        if os.path.isfile(p):
            paths = [p]
        elif os.path.isdir(p):
            paths = []
            for dp, dirs, fs in os.walk(p):
                dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache')]
                paths.extend(os.path.join(dp, f) for f in fs)
        else:
            continue
        for fp in sorted(paths):
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, REPO_ROOT).replace(os.sep, '/')
            h.update(f'{rel}|{st.st_size}|{int(st.st_mtime)}\n'.encode())
            n += 1
    return h.hexdigest(), n


def read_stamp():
    try:
        with open(STAMP, encoding='utf-8') as f:
            return json.load(f).get('fingerprint')
    except Exception:
        return None


def write_stamp(fingerprint):
    os.makedirs(BUILD, exist_ok=True)
    with open(STAMP, 'w', encoding='utf-8') as f:
        json.dump({'fingerprint': fingerprint, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f)


def ensure_built(force=False):
    """源文件自上次构建无改动则跳过重建。返回是否实际执行了构建。"""
    fp, n = src_fingerprint()
    if not force and read_stamp() == fp and os.path.isfile(os.path.join(BUILD, 'site_html', 'index.html')):
        print(f'[OK] 源文件无改动（{n} 个文件指纹一致），跳过重建；--force 可强制')
        return False
    build()
    write_stamp(src_fingerprint()[0])   # 构建后重取，覆盖构建期间发生的改动
    return True

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
    # 只排除构建产物/大文件目录：数据集（refine/dataset 2.9GB）、CUDA 编译产物、
    # 训练 checkpoint（refine/temp/refine_out）——三者都可由脚本再生成，不该进站点
    skip_dirs = {'__pycache__', '.pytest_cache', 'dataset', 'bin', 'refine_out', 'temp'}
    for base in (REPO_ROOT,):   # review 分支：优化内容已并入仓库自身
        for tree in SRC_TREES:
            src = os.path.join(base, tree)
            if not os.path.isdir(src):
                continue
            for dp, dirs, fs in os.walk(src):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
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


def plot_widget(body):
    """```plot JSON 配置 → 页内 Canvas 交互图（渲染逻辑见 _assets/plot.js）。

    配置：{title, subtitle, note, x:[min,max], y:[min,max],
           curves:[{expr, label}], hlines:[{y,label}], points:[[x,y,label]]}
    expr 为以 x 为自变量的 JS 表达式（如 "x/(1+Math.exp(-x))"）。
    解析失败时回退为纯代码块，构建不中断。
    """
    try:
        cfg = json.loads(body)
    except Exception as e:
        return ('<div class="plot-error">[plot 配置解析失败：'
                + esc(str(e)) + ']</div>' + code_block(body, ''))
    attr = html.escape(json.dumps(cfg, ensure_ascii=False), quote=True)
    return f'<div class="inline-plot" data-plot="{attr}"></div>'


def widget_embed(body):
    """```widget → 页内 iframe 嵌入 widgets/ 下的独立交互组件。

    body 首个非空行 = 组件名（不含 .html，仅限 [a-z0-9_-]）；可选第二行 = iframe 高度 px。
    组件文件不存在时不生成死链，回退为提示框。
    """
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return '<div class="plot-error">[widget 用法：```widget 换行 组件名 换行 高度px```]</div>'
    name = re.sub(r'[^a-z0-9_-]', '', lines[0].lower())
    height = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else 1180
    if not name or not os.path.isfile(os.path.join(REPO_ROOT, 'widgets', name + '.html')):
        return f'<div class="plot-error">[widget 不存在：widgets/{esc(name)}.html]</div>'
    return ('<div class="widget-embed"><div class="widget-bar">'
            f'<span>🎛️ 交互演示 · {esc(name)}.html</span>'
            f'<a href="/widgets/{name}.html" target="_blank">↗ 新窗口打开</a></div>'
            f'<iframe src="/widgets/{name}.html" loading="lazy" title="{esc(name)}" '
            f'style="height:{height}px"></iframe></div>')


def chunk_widget(body):
    """```chunk JSON → 页内"训练样本可视化"组件（渲染逻辑见 _assets/plot.js）。

    配置：{title, subtitle, sequence:[...], samples:N}
    生成 N 行样本：#k 行 x=sequence[0..k]（末位标记为新增）→ y=sequence[k+1]。
    """
    try:
        cfg = json.loads(body)
    except Exception as e:
        return ('<div class="plot-error">[chunk 配置解析失败：'
                + esc(str(e)) + ']</div>' + code_block(body, ''))
    attr = html.escape(json.dumps(cfg, ensure_ascii=False), quote=True)
    return f'<div class="chunk-viz" data-chunk="{attr}"></div>'


# ---------- viz：基于原文 authored 生成的可视化组件（流程/树）----------
VIZ_KINDS = {          # 语义分色：不同属性 → 不同颜色（chunk.html 视觉语法）
    'input':  ('viz-k-input',  '输入'),
    'op':     ('viz-k-op',     '变换'),
    'mid':    ('viz-k-mid',    '中间结果'),
    'output': ('viz-k-output', '输出'),
    'green':  ('viz-k-green',  '计算'),
}
LAST_PLAIN = {'body': None, 'index': -1}   # 紧邻的纯文本块，供 viz 锚定原文
RENDER_CTX = {'page': ''}                  # 当前渲染页面（viz 一致性警告定位用）


def _viz_legend(cfg, used):
    labels = dict(cfg.get('legend', {}))
    parts = []
    for k in dict.fromkeys(used):     # 去重且保序
        cls, default = VIZ_KINDS.get(k, ('', k))
        parts.append(f'<span class="cv-lg"><i class="cv-sw {cls}"></i>{esc(labels.get(k, default))}</span>')
    return '<div class="viz-legend">' + ''.join(parts) + '</div>' if parts else ''


def _viz_src(cfg, stale):
    src = cfg.get('src')
    if not src:
        return '<div class="viz-stale">⚠️ 未声明原文（src），无法校验一致性</div>' if stale else ''
    warn = ('<div class="viz-stale">⚠️ 原文已变化，本图示可能过期，请重新生成</div>'
            if stale else '')
    return (warn + '<details class="viz-src"><summary>原始文本</summary>'
            f'<pre>{esc(src)}</pre></details>')


def _viz_flow(cfg):
    """type=flow：rows=[{chain:[[文本,类型],...]} | {node:[文本,类型], note:...} | {gap:true}]"""
    used, rows_html, first_content = [], [], True
    for row in cfg.get('rows', []):
        if row.get('gap'):
            rows_html.append('<div class="fl-gap"></div>')
            continue
        if not first_content:
            rows_html.append('<div class="fl-varr">▼</div>')
        first_content = False
        if 'chain' in row:
            segs = []
            for item in row['chain']:
                text, kind = (item if isinstance(item, list) else (item, 'mid'))
                used.append(kind)
                cls, _ = VIZ_KINDS.get(kind, VIZ_KINDS['mid'])
                segs.append(f'<span class="viz-node {cls}">{esc(text)}</span>')
            rows_html.append('<div class="fl-row">'
                             + '<span class="fl-harr">→</span>'.join(segs) + '</div>')
        else:
            text, kind = row['node']
            used.append(kind)
            cls, _ = VIZ_KINDS.get(kind, VIZ_KINDS['mid'])
            note = (f'<span class="fl-note">{esc(row["note"])}</span>' if row.get('note') else '')
            rows_html.append(f'<div class="fl-row"><span class="viz-node {cls}">{esc(text)}</span>{note}</div>')
    return ('<div class="viz-card">'
            + ('<div class="viz-head"><span class="viz-title">' + esc(cfg.get('title', ''))
               + ('</span><span class="viz-sub">' + esc(cfg['subtitle']) + '</span>' if cfg.get('subtitle') else '</span>')
               + '</div>' if cfg.get('title') or cfg.get('subtitle') else '')
            + '<div class="viz-body">' + ''.join(rows_html) + '</div>'
            + _viz_legend(cfg, used)
            + '</div>')


def _viz_panel(cfg):
    """type=panel：PPT 图示风的模块面板——容器框内若干子模块卡（徽章+名称+描述+标签+图标），
    可选堆叠指示器（双色条缩略块 ×N + ↓ + ⋯ 表示 Block 重复）。"""
    used = []
    mods = []
    for m in cfg.get('modules', []):
        kind = m.get('kind', 'op')
        used.append(kind)
        cls, _ = VIZ_KINDS.get(kind, VIZ_KINDS['op'])
        tag = ''
        if m.get('tag'):
            tcls, _ = VIZ_KINDS.get(m.get('tagKind', kind), VIZ_KINDS['op'])
            tag = f'<span class="pm-tag {tcls}">{esc(m["tag"])}</span>'
        mods.append(
            '<div class="pm-module">'
            f'<span class="pm-badge {cls}">{esc(m.get("badge", ""))}</span>'
            '<div class="pm-content">'
            f'<span class="pm-name">{esc(m.get("name", ""))}</span>'
            f'<span class="pm-desc">{esc(m.get("desc", ""))}{tag}</span>'
            '</div>'
            + (f'<span class="pm-icon">{m["icon"]}</span>' if m.get("icon") else '')
            + '</div>')
    stack = ''
    if cfg.get('stack'):
        s = cfg['stack']
        for k in s.get('segments', ['op']):
            used.append(k)
        segs = ''.join(
            f'<span class="sb-seg {VIZ_KINDS.get(k, VIZ_KINDS["op"])[0]}"></span>'
            for k in s.get('segments', ['op']))
        thumb = f'<div class="sb-block">{segs}</div>'
        arrow = '<div class="sb-arrow">↓</div>'
        n = int(s.get('count', 3))
        diagram = (thumb + arrow) * max(0, n - 1) + thumb
        if s.get('ellipsis', True):
            diagram += '<div class="sb-ellipsis">⋯</div>'
        stack = ('<div class="pm-stack">'
                 f'<div class="pm-stack-label">{esc(s.get("label", ""))}</div>'
                 f'<div class="sb-wrap">{diagram}</div></div>')
    container = ''.join(mods)
    if cfg.get('container'):
        container = (f'<div class="pm-container"><span class="pm-container-label">'
                     f'{esc(cfg["container"])}</span>' + container + '</div>')
    legend = _viz_legend(cfg, used + (['↓'] if cfg.get('stack') else []))
    if cfg.get('stack'):
        legend = legend.replace(  # ↓ 项渲染为箭头图例
            '<i class="cv-sw "></i>↓',
            '<span class="sb-arrow">↓</span>堆叠方向') if '↓' in legend else legend
    return ('<div class="viz-card viz-panel">'
            + ('<div class="viz-head"><span class="viz-title">' + esc(cfg.get('title', ''))
               + ('</span><span class="viz-sub">' + esc(cfg['subtitle']) + '</span>' if cfg.get('subtitle') else '</span>')
               + '</div>' if cfg.get('title') or cfg.get('subtitle') else '')
            + '<div class="viz-body pm-body">' + container + stack + '</div>'
            + legend + '</div>')


def viz_widget(body):
    """```viz JSON → 语义分色可视化组件。

    type=flow 构建期直接渲染；type=tree 交给 _assets/plot.js 测量布局并画连接线。
    组件须紧跟其依据的原文 text 块；src 字段是原文副本，构建期比对——
    不一致说明 markdown 已修改，页面内与构建日志都会提示需要重新生成。
    """
    try:
        cfg = json.loads(body)
    except Exception as e:
        return ('<div class="plot-error">[viz 配置解析失败：'
                + esc(str(e)) + ']</div>' + code_block(body, ''))
    stale = False
    src = cfg.get('src')
    if src is not None:
        if LAST_PLAIN['index'] == -1 or LAST_PLAIN['body'] != src.strip():
            stale = True
            page = RENDER_CTX.get('page', '?')
            print(f'⚠️ viz 原文不匹配（{page} / {cfg.get("title", "")}）：markdown 可能已变化，请重新生成 viz')
        elif LAST_PLAIN['index'] >= 0:
            # 原文块紧邻其前 → 由 viz 组件内的折叠原文取而代之
            LAST_PLAIN['replaced'] = True
    tail = _viz_src(cfg, stale)
    if cfg.get('type') == 'flow':
        return _viz_flow(cfg) + tail
    if cfg.get('type') == 'panel':
        return _viz_panel(cfg) + tail
    if cfg.get('type') == 'highway':
        attr = html.escape(json.dumps(cfg, ensure_ascii=False), quote=True)
        return f'<div class="viz-highway" data-viz="{attr}"></div>' + tail
    if cfg.get('type') == 'tree':
        attr = html.escape(json.dumps(cfg, ensure_ascii=False), quote=True)
        stale_banner = ('<div class="viz-stale">⚠️ 原文已变化，本图示可能过期，请重新生成</div>'
                        if stale else '')
        return f'<div class="viz-tree" data-viz="{attr}"></div>' + stale_banner
    return '<div class="plot-error">[viz 未知类型，支持 flow / tree]</div>'



def table_widget(body):
    """空格对齐的列式 text 块 → 真 HTML 表格（列宽不再依赖等宽对齐）。不可解析返回 ''。"""
    rows = [re.split(r'\s{2,}|\t+', ln.strip()) for ln in body.split('\n') if ln.strip()]
    if len(rows) < 2:
        return ''
    counts = [len(r) for r in rows]
    if min(counts) < 2 or max(counts) - min(counts) > 1:
        return ''
    t = ['<div class="tbl-wrap"><table class="md-table">', '<thead><tr>']
    t += [f'<th>{inline(c)}</th>' for c in rows[0]]
    t.append('</tr></thead><tbody>')
    for r in rows[1:]:
        t.append('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>')
    t.append('</tbody></table></div>')
    return ''.join(t)

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
                elif code_lang == 'plot':
                    out.append(plot_widget(body))
                elif code_lang == 'chunk':
                    out.append(chunk_widget(body))
                elif code_lang == 'viz':
                    out.append(viz_widget(body))
                    if LAST_PLAIN.get('replaced') and LAST_PLAIN['index'] < len(out) - 1:
                        out[LAST_PLAIN['index']] = ''   # 原文块由 viz 内的折叠原文取代
                    LAST_PLAIN.update({'body': None, 'index': -1, 'replaced': False})
                elif code_lang == 'widget':
                    out.append(widget_embed(body))
                elif code_lang in ('', 'text'):
                    # ═══ 标题 ═══ / === 标题 === 横幅开头的脚本输出块 → 统一「运行输出」卡片
                    m2 = re.match(r'^\s*[═━─=]{2,}\s*(\S.*?)\s*[═━─=]{2,}\s*$',
                                  body.split('\n', 1)[0])
                    if m2:
                        rest = body.split('\n', 1)[1].rstrip('\n') if '\n' in body else ''
                        out.append('<div class="out-card"><div class="out-head">'
                                   f'<span class="out-title">{esc(m2.group(1))}</span>'
                                   '<span class="out-head-r"><span class="out-tag">运行输出</span>'
                                   '<button class="code-copy" type="button">复制</button></span></div>'
                                   f'<pre><code>{esc(rest)}</code></pre></div>')
                        LAST_PLAIN.update({'body': None, 'index': -1, 'replaced': False})
                    else:
                        tbl = table_widget(body)
                        out.append(tbl if tbl else code_block(body, ''))
                        # 记录紧邻文本块（含被转成表格的），供后续 ```viz 锚定/校验并取而代之
                        LAST_PLAIN.update({'body': body.strip(), 'index': len(out) - 1,
                                           'replaced': False})
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
<link rel="stylesheet" href="/_assets/style.css?v=54">
<script>
(function(){try{var t=localStorage.getItem('mm-theme');
if(!t&&window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)t='dark';
if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}
// 无 View Transitions 能力的浏览器打标：正文进场动画兜底（有 VT 时让位给导航过渡）
try{if(!(window.CSS&&CSS.supports&&CSS.supports('selector(html:active-view-transition)')))
document.documentElement.classList.add('no-vt');}catch(e){document.documentElement.classList.add('no-vt');}})();
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
<script defer src="/_assets/plot.js?v=5"></script>
</head>
<body>
<header class="topbar">
  <div class="topbar-l">
    <a class="brand" href="/index.html"><span class="brand-logo">🌱</span><span>makemore 教程</span><span class="version">v1.0.0</span></a>
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
<button id="backToTop" title="返回顶部" aria-label="返回顶部">↑</button>
<div id="readProgress" aria-hidden="true"></div>
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
// 导航一键展开/收起（偏好跨页记忆）
(function(){var btn=document.getElementById('navToggleAll');if(!btn)return;
var secs=function(){return Array.prototype.slice.call(document.querySelectorAll('nav.side details.nav-sec'));};
function refresh(){var s=secs();btn.textContent=(s.length&&s.every(function(d){return d.open}))?'收起全部':'展开全部';}
btn.onclick=function(){var s=secs(),open=s.some(function(d){return !d.open});
  s.forEach(function(d){d.open=open});
  try{localStorage.setItem('mm-nav-all',open?'1':'0');}catch(e){}
  refresh();};
secs().forEach(function(d){d.addEventListener('toggle',refresh)});
try{var v=localStorage.getItem('mm-nav-all');
if(v==='1')secs().forEach(function(d){d.open=true;});
else if(v==='0')secs().forEach(function(d){d.open=!!d.querySelector('.cur');});}catch(e){}
refresh();})();
// 字号
(function(){var LEVELS=5,KEY='mm-fs',idx=1;
try{var v=parseInt(localStorage.getItem(KEY));if(v>=0&&v<LEVELS)idx=v;}catch(e){}
function apply(){document.documentElement.setAttribute('data-fs',String(idx));
try{localStorage.setItem(KEY,idx);}catch(e){}}
document.getElementById('fontMinus').onclick=function(){idx=Math.max(0,idx-1);apply();};
document.getElementById('fontReset').onclick=function(){idx=1;apply();};
document.getElementById('fontPlus').onclick=function(){idx=Math.min(LEVELS-1,idx+1);apply();};
apply();})();
// 主题（View Transitions 圆形揭示；不支持/减少动效偏好时直接切换）
(function(){var b=document.getElementById('themeBtn');
var meta=document.querySelector('meta[name="theme-color"]');
function paint(){var t=document.documentElement.getAttribute('data-theme');
b.textContent=t==='dark'?'☀️ 日间':'🌙 夜间';
if(meta)meta.setAttribute('content',t==='dark'?'#0d1117':'#ffffff');}
function flip(){var n=document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';
document.documentElement.setAttribute('data-theme',n);
try{localStorage.setItem('mm-theme',n);}catch(e){}
paint();}
b.onclick=function(){
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(!document.startViewTransition||reduce){flip();return;}
  var r=b.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2;
  var R=Math.hypot(Math.max(x,window.innerWidth-x),Math.max(y,window.innerHeight-y));
  var vt;
  try{vt=document.startViewTransition({update:flip,types:['theme']});}
  catch(e){vt=document.startViewTransition(flip);}
  vt.ready.then(function(){
    document.documentElement.animate(
      {clipPath:['circle(0px at '+x+'px '+y+'px)','circle('+R+'px at '+x+'px '+y+'px)']},
      {duration:480,easing:'cubic-bezier(.2,.7,.3,1)',pseudoElement:'::view-transition-new(root)'});
  }).catch(function(){});
};
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
// 进度追踪（标记完成 ↔ 侧栏绿勾 + Part 计数联动）
(function(){
  var KEY='mm-progress';var progress={};
  try{progress=JSON.parse(localStorage.getItem(KEY))||{};}catch(e){}
  var currentPath=window.location.pathname;
  var markBtn=document.getElementById('markComplete');
  var progressText=document.getElementById('progressText');
  function refreshNav(){
    document.querySelectorAll('.nav-item a').forEach(function(a){
      a.classList.toggle('done',!!progress[a.getAttribute('href')]);
    });
    document.querySelectorAll('details.nav-sec').forEach(function(d){
      var links=d.querySelectorAll('.nav-item a');
      if(!links.length)return;
      var c=d.querySelector('.nav-count');
      if(!c){c=document.createElement('span');c.className='nav-count';d.querySelector('summary').appendChild(c);}
      var done=0;
      links.forEach(function(a){if(a.classList.contains('done'))done++;});
      c.textContent=done+'/'+links.length;
      c.classList.toggle('all',done===links.length);
      d.classList.toggle('sec-done',done===links.length&&links.length>0);
    });
  }
  function updateUI(){
    var done=Object.keys(progress).filter(function(k){return progress[k];}).length;
    progressText.textContent='已完成 '+done+' 个章节';
    if(progress[currentPath]){markBtn.textContent='✓ 已完成';markBtn.classList.add('done');}
    else{markBtn.textContent='✓ 标记为已完成';markBtn.classList.remove('done');}
    refreshNav();
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
    if(block.closest('.code-card')||block.closest('.file-view')||block.closest('.out-card'))return;
    var card=document.createElement('div');card.className='code-card';
    var head=document.createElement('div');head.className='code-head';
    var name=document.createElement('span');name.className='code-lang';name.textContent=lang||'';
    var btn=document.createElement('button');btn.className='code-copy';btn.type='button';btn.textContent='复制';
    head.appendChild(name);head.appendChild(btn);
    block.parentNode.insertBefore(card,block);
    card.appendChild(head);card.appendChild(block);
  }
  document.querySelectorAll('.content .highlight').forEach(function(h){wrap(h,h.getAttribute('data-lang'));});
  document.querySelectorAll('.content > pre').forEach(function(p){
    if(!p.classList.contains('mermaid'))wrap(p,'');
  });
  // 事件委托统一接线：运行时包装卡 / 构建期 out-card 的复制按钮都走这里；
  // pre 必须从按钮所在卡片容器里取——裸 pre 图示块自身就是 block，block.querySelector('pre') 为 null
  document.addEventListener('click',function(e){
    var btn=e.target.closest('.code-copy');
    if(!btn)return;
    var card=btn.closest('.code-card,.out-card');
    var pre=card&&card.querySelector('pre');
    if(pre)copyText(pre.innerText,btn);
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
// 阅读进度条 + 返回顶部（进度环）
(function(){
  var bar=document.getElementById('readProgress');
  var btn=document.getElementById('backToTop');
  var raf=false;
  function onScroll(){
    if(raf)return;raf=true;
    requestAnimationFrame(function(){
      raf=false;
      var doc=document.documentElement;
      var max=doc.scrollHeight-window.innerHeight;
      var p=max>0?Math.min(1,window.scrollY/max):0;
      if(bar)bar.style.width=(p*100).toFixed(2)+'%';
      btn.classList.toggle('show',window.scrollY>300);
      btn.style.setProperty('--p',(p*100).toFixed(1)+'%');
    });
  }
  window.addEventListener('scroll',onScroll,{passive:true});
  window.addEventListener('resize',onScroll);
  onScroll();
  btn.onclick=function(){
    var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({top:0,behavior:reduce?'auto':'smooth'});
  };
})();
</script>
</body>
</html>"""

PLOT_JS = r"""/* 页内交互函数图渲染器：```plot 围栏块（data-plot JSON）→ Canvas 交互图。
   交互模式参照 sigmoid 动画：悬停十字线 + 实时读数 + 渐近线 + 关键点；
   额外支持多曲线、入场描线动画、亮/暗主题自适应、触屏。 */
(function () {
  'use strict';

  function dark() { return document.documentElement.getAttribute('data-theme') === 'dark'; }
  function cv(name, fb) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fb;
  }
  function fmt(v) { return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3); }

  function initPlot(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-plot')); }
    catch (e) { el.textContent = '[plot 配置异常]'; return; }

    // --- DOM：标题头 + 画布 + 图例 ---
    el.innerHTML = '';
    var head = document.createElement('div'); head.className = 'ip-head';
    var t = document.createElement('span'); t.className = 'ip-title'; t.textContent = cfg.title || '';
    head.appendChild(t);
    if (cfg.subtitle) { var sub = document.createElement('span'); sub.className = 'ip-sub'; sub.textContent = cfg.subtitle; head.appendChild(sub); }
    var wrap = document.createElement('div'); wrap.className = 'ip-wrap';
    var cvs = document.createElement('canvas'); wrap.appendChild(cvs);
    var legend = document.createElement('div'); legend.className = 'ip-legend';
    el.appendChild(head); el.appendChild(wrap); el.appendChild(legend);
    if (cfg.note) { var note = document.createElement('div'); note.className = 'ip-note'; note.textContent = cfg.note; el.appendChild(note); }

    var curves = (cfg.curves || []).map(function (c, i) {
      var item = document.createElement('span'); item.className = 'ip-item';
      var dot = document.createElement('i'); dot.className = 'ip-dot';
      var lab = document.createElement('span'); lab.textContent = c.label || ('f' + (i + 1));
      var val = document.createElement('code'); val.textContent = '—';
      item.appendChild(dot); item.appendChild(lab); item.appendChild(val);
      legend.appendChild(item);
      return { fn: new Function('x', 'return (' + (c.expr || '0') + ');'),
               label: c.label || ('f' + (i + 1)), valEl: val, dotEl: dot, col: null };
    });

    var ctx = cvs.getContext('2d');
    var reveal = 0, hover = null, raf = null, layout = null;

    function palette(i) {
      var p = [cv('--accent', '#0969da'), cv('--success', '#1a7f37'),
               dark() ? '#d2a8ff' : '#8250df', cv('--warn', '#9a6700')];
      return p[i % p.length];
    }

    function ranges() {
      var R = { x0: (cfg.x || [-6, 6])[0], x1: (cfg.x || [-6, 6])[1], y0: -1, y1: 1 };
      if (cfg.y) { R.y0 = cfg.y[0]; R.y1 = cfg.y[1]; }
      else {
        var mn = Infinity, mx = -Infinity;
        curves.forEach(function (c) {
          for (var k = 0; k <= 200; k++) {
            var v; try { v = c.fn(R.x0 + (R.x1 - R.x0) * k / 200); } catch (e) { continue; }
            if (isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
          }
        });
        if (!isFinite(mn)) { mn = -1; mx = 1; }
        if (mx - mn < 1e-6) { mn -= 1; mx += 1; }
        var pad = (mx - mn) * 0.12; R.y0 = mn - pad; R.y1 = mx + pad;
      }
      return R;
    }

    function draw() {
      var rect = wrap.getBoundingClientRect();
      if (rect.width < 10) return;
      var w = rect.width, h = rect.height || w * 0.5625;
      var dpr = window.devicePixelRatio || 1;
      var pw = Math.round(w * dpr), ph = Math.round(h * dpr);
      if (cvs.width !== pw || cvs.height !== ph) { cvs.width = pw; cvs.height = ph; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var R = ranges();
      var pad = { top: 18, bottom: 26, left: 46, right: 16 };
      var plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom;
      if (plotW < 20 || plotH < 20) return;
      function X(x) { return pad.left + (x - R.x0) / (R.x1 - R.x0) * plotW; }
      function Y(y) { return pad.top + plotH - (y - R.y0) / (R.y1 - R.y0) * plotH; }
      layout = { R: R, pad: pad, plotW: plotW, plotH: plotH, w: w, h: h };

      var isDark = dark();
      var cFg = cv('--fg', '#1f2328'), cFg3 = cv('--fg3', '#59636e'), cFg4 = cv('--fg4', '#818b98');
      var cGrid = isDark ? 'rgba(240,246,252,0.08)' : '#eef1f4';
      var cDanger = isDark ? '#ff7b72' : '#cf222e';
      var mono = cv('--mono', 'monospace');
      curves.forEach(function (c, i) { c.col = palette(i); c.dotEl.style.background = c.col; });

      ctx.clearRect(0, 0, w, h);

      // --- 网格 + 刻度 ---
      var xs = cfg.xticks || Math.max(0.5, +(((R.x1 - R.x0) / 8).toPrecision(2)));
      var ys = cfg.yticks || Math.max(0.25, +(((R.y1 - R.y0) / 5).toPrecision(2)));
      function nice(v, step) { return Math.abs(v / step) < 1e-6 ? 0 : Math.abs(step) < 1 ? v.toFixed(2) : String(Math.round(v * 100) / 100); }
      ctx.strokeStyle = cGrid; ctx.lineWidth = 1;
      ctx.font = '11px ' + mono; ctx.fillStyle = cFg4;
      var axisY = Math.min(Math.max(Y(0), pad.top), pad.top + plotH);   // y=0 轴（越界则贴边）
      var axisX = Math.min(Math.max(X(0), pad.left), pad.left + plotW);
      for (var gx = Math.ceil(R.x0 / xs) * xs; gx <= R.x1 + 1e-9; gx += xs) {
        var px = X(gx);
        ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, pad.top + plotH); ctx.stroke();
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(nice(gx, xs), px, Math.min(axisY + 5, pad.top + plotH + 6));
      }
      for (var gy = Math.ceil(R.y0 / ys) * ys; gy <= R.y1 + 1e-9; gy += ys) {
        var py = Y(gy);
        ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(pad.left + plotW, py); ctx.stroke();
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillText(nice(gy, ys), pad.left - 7, py);
      }

      // --- 坐标轴 ---
      ctx.strokeStyle = cFg3; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(pad.left, axisY); ctx.lineTo(pad.left + plotW, axisY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(axisX, pad.top); ctx.lineTo(axisX, pad.top + plotH); ctx.stroke();

      // --- 渐近线 / 参考线（虚线）---
      (cfg.hlines || []).forEach(function (hl) {
        var hy = Y(hl.y);
        if (hy < pad.top || hy > pad.top + plotH) return;
        ctx.setLineDash([4, 5]); ctx.strokeStyle = cDanger; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(pad.left, hy); ctx.lineTo(pad.left + plotW, hy); ctx.stroke();
        ctx.setLineDash([]);
        if (hl.label) {
          ctx.fillStyle = cDanger; ctx.font = '11px ' + mono;
          ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
          ctx.fillText(hl.label, pad.left + 6, hy - 3);
        }
      });

      // --- 曲线（reveal 入场描线：只画到 x 进度处）---
      var xEnd = R.x0 + (R.x1 - R.x0) * reveal;
      curves.forEach(function (c) {
        ctx.beginPath(); ctx.strokeStyle = c.col; ctx.lineWidth = 2.6;
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        var started = false;
        for (var k = 0; k <= 300; k++) {
          var x = R.x0 + (xEnd - R.x0) * k / 300;
          var y; try { y = c.fn(x); } catch (e) { started = false; continue; }
          if (!isFinite(y)) { started = false; continue; }
          var yy = Math.min(Math.max(Y(y), pad.top - 20), pad.top + plotH + 20);
          if (!started) { ctx.moveTo(X(x), yy); started = true; } else { ctx.lineTo(X(x), yy); }
        }
        ctx.stroke();
      });

      // --- 关键点（随 reveal 淡入）---
      ctx.save(); ctx.globalAlpha = Math.max(0, reveal * 2 - 1);
      (cfg.points || []).forEach(function (p) {
        var px = X(p[0]), py = Y(p[1]);
        ctx.fillStyle = cFg;
        ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = cv('--card', '#fff'); ctx.lineWidth = 2; ctx.stroke();
        if (p[2]) {
          ctx.fillStyle = cFg3; ctx.font = '12px ' + mono;
          ctx.textAlign = 'left';
          if (p[3] === 'below') { ctx.textBaseline = 'top'; ctx.fillText(p[2], px + 9, py + 8); }
          else { ctx.textBaseline = 'bottom'; ctx.fillText(p[2], px + 9, py - 4); }
        }
      });
      ctx.restore();

      // --- 悬停：十字线 + 各曲线取值点 + 悬浮读数框 ---
      if (hover !== null && hover >= R.x0 && hover <= R.x1) {
        ctx.setLineDash([3, 5]); ctx.strokeStyle = curves.length ? curves[0].col : cFg3;
        ctx.globalAlpha = 0.35; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(X(hover), pad.top); ctx.lineTo(X(hover), pad.top + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, 0); ctx.lineTo(pad.left, 0); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
        var rows = ['x = ' + hover.toFixed(2)];
        var cols = [null];
        curves.forEach(function (c) {
          var v; try { v = c.fn(hover); } catch (e) { v = NaN; }
          rows.push((c.label.length > 18 ? c.label.slice(0, 17) + '…' : c.label) + ' = ' + (isFinite(v) ? fmt(v) : '—'));
          cols.push(c.col);
          c.valEl.textContent = isFinite(v) ? fmt(v) : '—';
          if (isFinite(v)) {
            var py2 = Math.min(Math.max(Y(v), pad.top), pad.top + plotH);
            ctx.fillStyle = c.col;
            ctx.beginPath(); ctx.arc(X(hover), py2, 5, 0, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = cv('--card', '#fff'); ctx.lineWidth = 2; ctx.stroke();
          }
        });
        // 读数框（顶部，按光标左右分侧，避免遮挡曲线）
        ctx.font = '12px ' + mono;
        var tw = 0; rows.forEach(function (r) { tw = Math.max(tw, ctx.measureText(r).width); });
        tw += 46;
        var th = rows.length * 17 + 12;
        var lx = X(hover) > pad.left + plotW / 2 ? pad.left + 6 : pad.left + plotW - tw - 6;
        var ly = pad.top + 6;
        ctx.fillStyle = isDark ? 'rgba(22,27,34,0.92)' : 'rgba(255,255,255,0.92)';
        ctx.strokeStyle = cv('--line', '#d1d9e0'); ctx.lineWidth = 1;
        if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(lx, ly, tw, th, 6); ctx.fill(); ctx.stroke(); }
        else { ctx.strokeRect(lx, ly, tw, th); ctx.fillRect(lx, ly, tw, th); }
        rows.forEach(function (r, ri) {
          var cy2 = ly + 14 + ri * 17;
          if (cols[ri]) { ctx.fillStyle = cols[ri]; ctx.fillRect(lx + 10, cy2 - 3.5, 7, 7); }
          ctx.fillStyle = ri === 0 ? cFg : cFg3;
          ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(r, lx + (cols[ri] ? 24 : 10), cy2);
        });
      } else {
        curves.forEach(function (c) { c.valEl.textContent = '—'; });
      }
    }

    // --- 入场描线动画 ---
    function animate() {
      if (raf) return;
      var t0 = null;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var p = Math.min(1, (ts - t0) / 700);
        reveal = 1 - Math.pow(1 - p, 3);
        draw();
        if (p < 1) raf = requestAnimationFrame(step); else raf = null;
      }
      raf = requestAnimationFrame(step);
    }

    // --- 事件：悬停 / 触屏 ---
    function onPoint(clientX) {
      if (!layout) return;
      var r = cvs.getBoundingClientRect();
      var px = clientX - r.left;
      var R = layout.R;
      if (px < layout.pad.left || px > layout.pad.left + layout.plotW) { hover = null; }
      else { hover = R.x0 + (px - layout.pad.left) / layout.plotW * (R.x1 - R.x0); }
      draw();
    }
    function off() { hover = null; draw(); }
    cvs.addEventListener('mousemove', function (e) { onPoint(e.clientX); });
    cvs.addEventListener('mouseleave', off);
    cvs.addEventListener('touchmove', function (e) { e.preventDefault(); onPoint(e.touches[0].clientX); }, { passive: false });
    cvs.addEventListener('touchend', off);

    // --- 尺寸 / 主题 / 字号 变化重绘 ---
    if ('ResizeObserver' in window) { new ResizeObserver(function () { draw(); }).observe(wrap); }
    else { window.addEventListener('resize', function () { draw(); }); }
    new MutationObserver(function () { draw(); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-fs'] });

    // --- 首次进入视口 → 描线动画 ---
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { io.disconnect(); animate(); }
      }, { rootMargin: '80px' });
      io.observe(el);
    } else { reveal = 1; }

    draw();
  }

  /* ---------- ```chunk：训练样本可视化（x token 色块 → y 高亮）---------- */
  function initChunk(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-chunk')); }
    catch (e) { el.textContent = '[chunk 配置异常]'; return; }
    var seq = cfg.sequence || [];
    var n = Math.min(cfg.samples || seq.length - 1, seq.length - 1);

    el.innerHTML = '';
    var head = document.createElement('div'); head.className = 'cv-head';
    var t = document.createElement('span'); t.className = 'cv-title'; t.textContent = cfg.title || '';
    head.appendChild(t);
    el.appendChild(head);
    if (cfg.subtitle) {
      var sub = document.createElement('div'); sub.className = 'cv-sub'; sub.textContent = cfg.subtitle;
      el.appendChild(sub);
    }
    var list = document.createElement('div'); list.className = 'cv-list';
    for (var i = 0; i < n; i++) {
      var row = document.createElement('div'); row.className = 'cv-row';
      row.style.transitionDelay = (i * 45) + 'ms';
      var id = document.createElement('span'); id.className = 'cv-id'; id.textContent = '#' + (i + 1);
      var xs = document.createElement('div'); xs.className = 'cv-xs';
      for (var j = 0; j <= i; j++) {
        var tok = document.createElement('span');
        tok.className = 'cv-tok' + (j === i ? ' is-new' : '');
        tok.textContent = seq[j];
        xs.appendChild(tok);
      }
      var arrow = document.createElement('span'); arrow.className = 'cv-arrow'; arrow.textContent = '→';
      var yl = document.createElement('span'); yl.className = 'cv-yl'; yl.textContent = 'y=';
      var yv = document.createElement('span'); yv.className = 'cv-y'; yv.textContent = seq[i + 1];
      row.appendChild(id); row.appendChild(xs); row.appendChild(arrow);
      row.appendChild(yl); row.appendChild(yv);
      list.appendChild(row);
    }
    el.appendChild(list);
    var lg = document.createElement('div'); lg.className = 'cv-legend';
    lg.innerHTML = '<span class="cv-lg"><i class="cv-sw sw-old"></i>已有 token</span>'
      + '<span class="cv-lg"><i class="cv-sw sw-new"></i>新增 token ✦</span>'
      + '<span class="cv-lg"><i class="cv-sw sw-y"></i>输出 y</span>';
    el.appendChild(lg);
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { io.disconnect(); el.classList.add('cv-in'); }
      }, { rootMargin: '60px' });
      io.observe(el);
    } else { el.classList.add('cv-in'); }
  }

  /* ---------- ```viz type=tree：树状结构（JS 测量布局 + SVG 连接线）---------- */
  function initTree(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-viz')); }
    catch (e) { el.textContent = '[viz 配置异常]'; return; }

    el.innerHTML = '';
    var KIND_CLS = { input: 'viz-k-input', op: 'viz-k-op', mid: 'viz-k-mid', output: 'viz-k-output' };
    var KIND_LABEL = { input: '输入', op: '变换', mid: '中间结果', output: '输出' };
    var legendOv = cfg.legend || {};

    if (cfg.title || cfg.subtitle) {
      var head = document.createElement('div'); head.className = 'viz-head';
      var t = document.createElement('span'); t.className = 'viz-title'; t.textContent = cfg.title || '';
      head.appendChild(t);
      if (cfg.subtitle) { var s = document.createElement('span'); s.className = 'viz-sub'; s.textContent = cfg.subtitle; head.appendChild(s); }
      el.appendChild(head);
    }
    var stage = document.createElement('div'); stage.className = 'tr-stage';
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'tr-svg');
    stage.appendChild(svg);
    var levelEls = [];
    var used = {};
    (cfg.levels || []).forEach(function (level) {
      var row = document.createElement('div'); row.className = 'tr-level';
      level.forEach(function (item) {
        var text = item[0], kind = item[1] || 'mid';
        used[kind] = 1;
        var slot = document.createElement('div'); slot.className = 'tr-slot';
        var chip = document.createElement('span');
        chip.className = 'viz-node ' + (KIND_CLS[kind] || 'viz-k-mid');
        chip.textContent = text;
        slot.appendChild(chip);
        row.appendChild(slot);
      });
      stage.appendChild(row);
      levelEls.push(row);
    });
    el.appendChild(stage);

    var legend = document.createElement('div'); legend.className = 'viz-legend';
    Object.keys(used).forEach(function (k) {
      var item = document.createElement('span'); item.className = 'cv-lg';
      var sw = document.createElement('i'); sw.className = 'cv-sw ' + (KIND_CLS[k] || 'viz-k-mid');
      var lb = document.createElement('span'); lb.textContent = legendOv[k] || KIND_LABEL[k] || k;
      item.appendChild(sw); item.appendChild(lb);
      legend.appendChild(item);
    });
    el.appendChild(legend);
    if (cfg.src) {
      var det = document.createElement('details'); det.className = 'viz-src';
      var sum = document.createElement('summary'); sum.textContent = '原始文本';
      var pre = document.createElement('pre'); pre.textContent = cfg.src;
      det.appendChild(sum); det.appendChild(pre);
      el.appendChild(det);
    }

    function draw() {
      svg.innerHTML = '';
      var st = stage.getBoundingClientRect();
      if (st.width < 10) return;
      svg.setAttribute('viewBox', '0 0 ' + st.width + ' ' + st.height);
      var chipPos = levelEls.map(function (row) {
        return Array.from(row.querySelectorAll('.viz-node')).map(function (c) {
          var r = c.getBoundingClientRect();
          return { x: r.left - st.left + r.width / 2, top: r.top - st.top, bottom: r.top - st.top + r.height };
        });
      });
      var stroke = cv('--line2', '#afb8c1');
      for (var i = 0; i + 1 < chipPos.length; i++) {
        var parents = chipPos[i], kids = chipPos[i + 1];
        parents.forEach(function (p, pi) {
          [2 * pi, 2 * pi + 1].forEach(function (ki) {
            if (!kids[ki]) return;
            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', p.x); line.setAttribute('y1', p.bottom);
            line.setAttribute('x2', kids[ki].x); line.setAttribute('y2', kids[ki].top);
            line.setAttribute('stroke', stroke); line.setAttribute('stroke-width', '1.5');
            svg.appendChild(line);
          });
        });
      }
    }
    if ('ResizeObserver' in window) { new ResizeObserver(draw).observe(stage); }
    else { window.addEventListener('resize', draw); }
    new MutationObserver(draw).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme', 'data-fs'] });
    requestAnimationFrame(draw);
  }

  /* ---------- ```viz type=highway：残差流主干 + 子层盒（取样/写回双线 + ⊕ 并入）---------- */
  function initHighway(el) {
    if (el.__plotInit) return;
    el.__plotInit = true;
    var cfg;
    try { cfg = JSON.parse(el.getAttribute('data-viz')); }
    catch (e) { el.textContent = '[viz 配置异常]'; return; }
    var KIND_CLS = { input: 'viz-k-input', op: 'viz-k-op', mid: 'viz-k-mid', output: 'viz-k-output', green: 'viz-k-green' };
    var KIND_LABEL = { input: '输入', op: '变换', mid: '中间结果', output: '输出', green: '计算' };
    var legendOv = cfg.legend || {};

    el.innerHTML = '';
    if (cfg.title || cfg.subtitle) {
      var head = document.createElement('div'); head.className = 'viz-head';
      var t = document.createElement('span'); t.className = 'viz-title'; t.textContent = cfg.title || '';
      head.appendChild(t);
      if (cfg.subtitle) { var s = document.createElement('span'); s.className = 'viz-sub'; s.textContent = cfg.subtitle; head.appendChild(s); }
      el.appendChild(head);
    }
    var stage = document.createElement('div'); stage.className = 'hw-stage';
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'hw-svg');
    stage.appendChild(svg);
    var trunk = document.createElement('div'); trunk.className = 'hw-trunk';
    var from = document.createElement('span'); from.className = 'viz-node viz-k-input'; from.textContent = (cfg.trunk || {}).from || '输入';
    var spacer = document.createElement('span'); spacer.className = 'hw-mid';
    var to = document.createElement('span'); to.className = 'viz-node viz-k-output'; to.textContent = (cfg.trunk || {}).to || '输出';
    trunk.appendChild(from); trunk.appendChild(spacer); trunk.appendChild(to);
    stage.appendChild(trunk);
    var branches = document.createElement('div'); branches.className = 'hw-branches';
    var used = {};
    (cfg.branches || []).forEach(function (b) {
      var kind = b.kind || 'mid';
      used[kind] = 1;
      var slot = document.createElement('div'); slot.className = 'hw-slot';
      var box = document.createElement('div'); box.className = 'hw-block ' + kind;
      var title = document.createElement('div'); title.className = 'hw-btitle'; title.textContent = b.label;
      box.appendChild(title);
      if (b.note) { var sub = document.createElement('div'); sub.className = 'hw-bsub'; sub.textContent = b.note; box.appendChild(sub); }
      slot.appendChild(box);
      branches.appendChild(slot);
    });
    stage.appendChild(branches);
    if (cfg.note) { var fn = document.createElement('div'); fn.className = 'hw-foot'; fn.textContent = cfg.note; stage.appendChild(fn); }
    el.appendChild(stage);
    if (cfg.formula && cfg.formula.length) {
      var fw = document.createElement('div'); fw.className = 'hw-formula';
      cfg.formula.forEach(function (fch, i) {
        if (i) { var a = document.createElement('span'); a.className = 'hw-farrow'; a.textContent = '─►'; fw.appendChild(a); }
        var c = document.createElement('span'); c.className = 'hw-fchip'; c.textContent = fch; fw.appendChild(c);
      });
      el.appendChild(fw);
    }
    var legend = document.createElement('div'); legend.className = 'viz-legend';
    var i0 = document.createElement('span'); i0.className = 'cv-lg';
    i0.innerHTML = '<b class="hw-leg-add">⊕</b>' + (legendOv.add || '加法并入：x ← x + Block(x)');
    legend.appendChild(i0);
    var i1 = document.createElement('span'); i1.className = 'cv-lg';
    i1.innerHTML = '<i class="hw-leg-line"></i>' + (legendOv.trunk || '残差流主干（梯度直通）');
    legend.appendChild(i1);
    Object.keys(used).forEach(function (k) {
      var item = document.createElement('span'); item.className = 'cv-lg';
      item.innerHTML = '<i class="cv-sw ' + (KIND_CLS[k] || 'viz-k-mid') + '"></i>' + (legendOv[k] || KIND_LABEL[k] || k);
      legend.appendChild(item);
    });
    el.appendChild(legend);
    if (cfg.src) {
      var det = document.createElement('details'); det.className = 'viz-src';
      det.innerHTML = '<summary>原始文本</summary><pre></pre>';
      det.querySelector('pre').textContent = cfg.src;
      el.appendChild(det);
    }

    function draw() {
      svg.innerHTML = '';
      var st = stage.getBoundingClientRect();
      if (st.width < 10) return;
      svg.setAttribute('viewBox', '0 0 ' + st.width + ' ' + st.height);
      var cFg4 = cv('--fg4', '#818b98');
      var cWarn = cv('--warn', '#9a6700');
      var cCard = cv('--card', '#ffffff');
      var NS = 'http://www.w3.org/2000/svg';
      var f = from.getBoundingClientRect(), t2 = to.getBoundingClientRect();
      var x1 = f.right - st.left + 4, x2 = t2.left - st.left - 10;
      var y = f.top - st.top + f.height / 2;
      var tl = document.createElementNS(NS, 'line');
      tl.setAttribute('x1', x1); tl.setAttribute('y1', y); tl.setAttribute('x2', x2); tl.setAttribute('y2', y);
      tl.setAttribute('stroke', cFg4); tl.setAttribute('stroke-width', '2.5');
      svg.appendChild(tl);
      var ta = document.createElementNS(NS, 'path');
      ta.setAttribute('d', 'M ' + x2 + ' ' + (y - 5) + ' L ' + x2 + ' ' + (y + 5) + ' L ' + (x2 + 9) + ' ' + y + ' Z');
      ta.setAttribute('fill', cFg4);
      svg.appendChild(ta);
      Array.from(branches.querySelectorAll('.hw-block')).forEach(function (box) {
        var r = box.getBoundingClientRect();
        var cx = r.left - st.left + r.width / 2;
        var topY = r.top - st.top - 3;
        var jx1 = cx - 26, jx2 = cx + 26;
        function vline(x, yFrom, yTo, arrowAtBottom) {
          var l = document.createElementNS(NS, 'line');
          l.setAttribute('x1', x); l.setAttribute('y1', yFrom);
          l.setAttribute('x2', x); l.setAttribute('y2', yTo);
          l.setAttribute('stroke', cFg4); l.setAttribute('stroke-width', '1.8');
          svg.appendChild(l);
          var a = document.createElementNS(NS, 'path');
          var tipY = arrowAtBottom ? yTo - 1 : yFrom + 1;
          var baseY = arrowAtBottom ? yTo - 9 : yFrom + 9;
          a.setAttribute('d', 'M ' + (x - 4.5) + ' ' + baseY + ' L ' + (x + 4.5) + ' ' + baseY + ' L ' + x + ' ' + tipY + ' Z');
          a.setAttribute('fill', cFg4);
          svg.appendChild(a);
        }
        vline(jx1, y + 10, topY, true);
        vline(jx2, topY, y - 10, false);
        var c1 = document.createElementNS(NS, 'circle');
        c1.setAttribute('cx', cx); c1.setAttribute('cy', y); c1.setAttribute('r', '8');
        c1.setAttribute('fill', cCard); c1.setAttribute('stroke', cWarn); c1.setAttribute('stroke-width', '2');
        svg.appendChild(c1);
        var pl1 = document.createElementNS(NS, 'line');
        pl1.setAttribute('x1', cx - 4); pl1.setAttribute('y1', y); pl1.setAttribute('x2', cx + 4); pl1.setAttribute('y2', y);
        pl1.setAttribute('stroke', cWarn); pl1.setAttribute('stroke-width', '1.8');
        svg.appendChild(pl1);
        var pl2 = document.createElementNS(NS, 'line');
        pl2.setAttribute('x1', cx); pl2.setAttribute('y1', y - 4); pl2.setAttribute('x2', cx); pl2.setAttribute('y2', y + 4);
        pl2.setAttribute('stroke', cWarn); pl2.setAttribute('stroke-width', '1.8');
        svg.appendChild(pl2);
      });
    }
    if ('ResizeObserver' in window) { new ResizeObserver(draw).observe(stage); }
    else { window.addEventListener('resize', draw); }
    new MutationObserver(draw).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme', 'data-fs'] });
    requestAnimationFrame(draw);
  }
  function boot() {
    document.querySelectorAll('.inline-plot').forEach(initPlot);
    document.querySelectorAll('.chunk-viz').forEach(initChunk);
    document.querySelectorAll('.viz-tree').forEach(initTree);
    document.querySelectorAll('.viz-highway').forEach(initHighway);
  }
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', boot); }
  else { boot(); }
})();
"""

CSS = """
/* === 设计系统：GitHub Primer 风格 ===
   字体/配色/组件对齐 github.com：系统字体栈 + Primer 色板 + 全边框表格 + 代码块卡片 */
:root, :root[data-theme=light]{
  --bg:#ffffff; --bg2:#ffffff; --card:#ffffff; --card2:#f6f8fa;
  --fg:#333333; --fg2:#333333; --fg3:#59636e; --fg4:#818b98;
  --line:#d1d9e0; --line2:#afb8c1;
  --accent:#0969da; --accent-subtle:#ddf4ff; --accent-soft:rgba(9,105,218,0.15);
  --success:#1f883d; --success-hover:#1a7f37;
  --btn-bg:#f6f8fa; --btn-bg-hover:#eef1f4;
  --btn-line:rgba(31,35,40,0.15); --btn-line-hover:rgba(31,35,40,0.25);
  --code-bg:rgba(175,184,193,0.2); --pre-bg:#f6f8fa; --pre-fg:#1f2328; --pre-line:#d1d9e0;
  --warn:#9a6700; --mark-bg:#fff8c5; --danger:#cf222e; --success-subtle:#dafbe1;
  --overlay:rgba(31,35,40,0.4);
  --mono:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono','PingFang SC','Microsoft YaHei',monospace;
  --topbar-h:56px;
  --side-bg:#f6f8fa; --brand:#18e299;
  --grad-brand:linear-gradient(135deg,var(--accent) 0%,var(--brand) 100%);
  --radius-s:6px; --radius-m:10px; --radius-l:14px;
  --shadow-1:0 1px 3px rgba(31,35,40,0.05),0 4px 14px -6px rgba(31,35,40,0.08);
  --shadow-2:0 4px 10px rgba(31,35,40,0.08),0 16px 40px -12px rgba(31,35,40,0.16);
  --shadow-btn:0 1px 2px rgba(31,35,40,0.1);
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
  --warn:#d29922; --mark-bg:rgba(187,128,9,0.55); --danger:#da3633; --success-subtle:rgba(63,185,80,0.18);
  --overlay:rgba(1,4,9,0.6);
  --side-bg:#010409; --brand:#18e299;
  --grad-brand:linear-gradient(135deg,var(--accent) 0%,var(--brand) 100%);
  --shadow-1:0 1px 3px rgba(1,4,9,0.4),0 4px 14px -6px rgba(1,4,9,0.5);
  --shadow-2:0 4px 10px rgba(1,4,9,0.5),0 16px 40px -12px rgba(1,4,9,0.7);
  --shadow-btn:0 1px 2px rgba(1,4,9,0.4);
}
/* === 页面切换动画：跨文档 View Transitions（Chrome 126+/Safari 18.2+/FF 141+，
      不支持的浏览器自动忽略 → 无动画直切，纯渐进增强）===
      正文区命名 content：导航时顶栏/侧栏(root)静止，仅正文淡出/上移进场 */
@view-transition{navigation:auto}
.content{view-transition-name:content}
@media (prefers-reduced-motion:no-preference){
  ::view-transition-old(root),::view-transition-new(root){animation:none}
  ::view-transition-old(content){animation:vt-out .16s ease both}
  ::view-transition-new(content){animation:vt-in .26s ease both}
  @keyframes vt-out{to{opacity:0}}
  @keyframes vt-in{from{opacity:0;transform:translateY(10px)}}
}
/* 主题切换：JS 以 types:['theme'] 触发，从按钮位置圆形揭示；
   覆盖 root 默认动画避免与页面导航过渡互相干扰 */
html:active-view-transition-type(theme) ::view-transition-old(root),
html:active-view-transition-type(theme) ::view-transition-new(root){animation:none;mix-blend-mode:normal}
/* 不支持 View Transitions 的浏览器（head 内脚本打 no-vt 标记）：正文进场动画兜底 */
@media (prefers-reduced-motion:no-preference){
  .no-vt .content{animation:page-in .25s ease both}
}
@keyframes page-in{from{opacity:0;transform:translateY(8px)}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
:root[data-fs="0"]{font-size:87.5%}
:root[data-fs="1"]{font-size:100%}
:root[data-fs="2"]{font-size:112.5%}
:root[data-fs="3"]{font-size:125%}
:root[data-fs="4"]{font-size:137.5%}
body{margin:0;font-family:'Open Sans','Clear Sans','Helvetica Neue',Helvetica,Arial,'Segoe UI Emoji','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;font-size:16px;font-weight:400;background:var(--bg);color:var(--fg2);line-height:1.6;-webkit-font-smoothing:antialiased}
::selection{background:var(--accent-subtle)}
code,pre,kbd,.hw-formula{letter-spacing:normal}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:var(--line2);border-radius:6px;border:1px solid var(--bg)}
::-webkit-scrollbar-thumb:hover{background:var(--fg4)}
::-webkit-scrollbar-track{background:transparent}
button{font-family:inherit}
button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
h1,h2,h3,h4{color:var(--fg);font-weight:600}
kbd{display:inline-block;padding:1px 6px;font-size:11px;font-family:var(--mono);color:var(--fg3);background:var(--bg);border:1px solid var(--line2);border-radius:6px;line-height:1.5}
/* === 顶栏（GitHub header 风）=== */
.topbar{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center;gap:12px;min-height:var(--topbar-h);padding:0 20px;border-bottom:1px solid var(--line);
        background:color-mix(in srgb, var(--bg2) 86%, transparent);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
.topbar-l{display:flex;align-items:center;gap:16px;min-width:0}
.brand{color:var(--fg);text-decoration:none;font-weight:700;font-size:15px;letter-spacing:-0.2px;white-space:nowrap;display:flex;align-items:center;gap:9px}
.brand-logo{flex:none;width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;
            background:var(--grad-brand);box-shadow:var(--shadow-btn);font-size:15px;line-height:1}
.brand:hover .brand-logo{transform:translateY(-1px);box-shadow:var(--shadow-2)}
.brand .brand-logo{transition:transform .16s ease,box-shadow .16s ease}
.version{font-size:11px;font-weight:500;color:var(--fg3);margin-left:2px;border:1px solid var(--line);border-radius:999px;padding:0 8px;line-height:1.6}
.search-pill{display:flex;align-items:center;gap:8px;width:250px;min-height:34px;padding:0 14px;background:var(--card2);border:1px solid var(--line2);border-radius:999px;color:var(--fg3);font-size:14px;cursor:pointer;text-align:left;
             transition:border-color .16s ease,box-shadow .16s ease}
.search-pill:hover{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.search-pill .s-txt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.search-pill .s-ico{flex:none;color:var(--accent)}
.tools{display:flex;gap:6px}
.tools button{background:var(--btn-bg);color:var(--fg);border:1px solid var(--btn-line);border-radius:999px;padding:3px 13px;cursor:pointer;font-size:14px;font-weight:500;min-height:32px;line-height:1.5;
             transition:transform .15s ease,box-shadow .15s ease,border-color .15s ease,color .15s ease}
.tools button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px);box-shadow:var(--shadow-btn)}
.tools button:active{transform:none;box-shadow:none}
#themeBtn{min-width:88px}
@media (max-width:900px){
  .search-pill{width:auto;padding:0 9px}
  .search-pill .s-txt,.search-pill kbd{display:none}
  .tools button{padding:3px 8px;font-size:13px}
  #themeBtn{min-width:0}
}
/* === 布局 === */
.layout{display:flex;max-width:1280px;margin:0 auto}
.side{width:276px;flex:none;background:var(--side-bg);border-right:1px solid var(--line);padding:16px 12px;position:sticky;top:var(--topbar-h);height:calc(100vh - var(--topbar-h));overflow-y:auto;scrollbar-width:thin}
body.no-side .side{display:none}
body.no-side .layout{max-width:980px}
.nav-home{margin:2px 0 14px;padding:0 4px;display:flex;align-items:center;gap:8px}
.nav-home .nav-home-btn{flex:1;display:flex;align-items:center;justify-content:center;gap:7px;min-height:34px;
  background:var(--grad-brand);color:#ffffff;font-weight:600;font-size:13.5px;border-radius:999px;text-decoration:none;
  box-shadow:var(--shadow-btn);transition:transform .16s ease,box-shadow .16s ease,filter .16s ease}
.nav-home .nav-home-btn:hover{transform:translateY(-1px);box-shadow:var(--shadow-2);color:#ffffff;text-decoration:none;filter:brightness(1.05)}
.nav-home button{font-size:12px;padding:2px 12px;border:1px solid var(--btn-line);border-radius:999px;background:var(--btn-bg);color:var(--fg3);cursor:pointer;line-height:1.7;flex:none}
.nav-home button:hover{color:var(--accent);border-color:var(--accent)}
.nav-sec{margin:2px 0}
.nav-sec summary{list-style:none;padding:5px 8px;cursor:pointer;user-select:none;font-size:14px;font-weight:600;color:var(--fg);line-height:1.5;border-radius:6px;
                 display:flex;align-items:center;gap:6px}
.nav-sec summary:hover{color:var(--accent)}
details.nav-sec summary::-webkit-details-marker{display:none}
.nav-txt{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
details.nav-sec summary::before{content:'';flex:none;width:6px;height:6px;margin-left:2px;
  border-right:1.5px solid var(--fg4);border-bottom:1.5px solid var(--fg4);transform:rotate(-45deg);
  transition:transform .18s ease}
details.nav-sec[open] summary::before{transform:rotate(45deg)}
.nav-badge{flex:none;min-width:20px;height:20px;padding:0 5px;border-radius:6px;display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;color:#ffffff;background:var(--accent);font-variant-numeric:tabular-nums}
.nav-badge.nb-ico{background:var(--card2);color:var(--fg3);border:1px solid var(--line2);font-weight:600;font-size:11.5px}
details.nav-sec.sec-done .nav-badge{background:var(--success)}
details.nav-sec.sec-done .nav-badge.nb-ico{background:var(--success-subtle);color:var(--success);border-color:var(--success)}
.nav-count{flex:none;font-size:10.5px;font-weight:600;color:var(--fg4);background:var(--bg);border:1px solid var(--line);
  border-radius:999px;padding:0 7px;line-height:1.7;font-variant-numeric:tabular-nums}
.nav-count.all{color:var(--success);border-color:var(--success)}
.nav-item a{display:block;position:relative;color:var(--fg3);padding:4px 26px 4px 22px;border-radius:6px;font-size:14px;font-weight:400;text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.5}
.nav-item a:hover{color:var(--fg);background:var(--card2)}
.nav-item a.cur{color:var(--accent);background:var(--accent-subtle);font-weight:600}
.nav-item a.cur::before{content:'';position:absolute;left:-2px;top:50%;transform:translateY(-50%);width:3px;height:18px;border-radius:2px;background:var(--grad-brand)}
.nav-item a.done::after{content:'✓';position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:11px;font-weight:700;color:var(--success)}
.nav-item a.done{color:var(--fg4)}
.nav-item a.done.cur,.nav-item a.done:hover{color:var(--accent)}
/* === 内容区（GitHub markdown-body 风）=== */
.content{flex:1;min-width:0;max-width:860px;margin:0 auto;padding:30px 30px 100px}
.content h1{font-size:2.25em;font-weight:bold;line-height:1.2;padding-bottom:.3em;margin:24px 0 16px;color:var(--fg);
            border-bottom:2px solid;border-image:linear-gradient(90deg,var(--accent) 0%,transparent 72%) 1}
.content h2{font-size:1.75em;font-weight:bold;line-height:1.225;margin:26px 0 16px;padding:0 0 .3em 14px;position:relative;border-bottom:1px solid var(--line);color:var(--fg)}
.content h2::before{content:'';position:absolute;left:0;top:.28em;width:4px;height:.78em;border-radius:2px;background:var(--grad-brand)}
.content h3{font-size:1.5em;font-weight:bold;line-height:1.43;margin:24px 0 16px;color:var(--fg)}
.content h4{font-size:1.25em;font-weight:bold;margin:24px 0 16px;color:var(--fg)}
.content>*:first-child{margin-top:0}
.content p{margin:0 0 16px}
.content a{color:var(--accent);text-decoration:none;font-weight:400}
.content a:hover{text-decoration:underline}
.content strong{font-weight:600;color:var(--fg)}
.content ul,.content ol{padding-left:30px;margin:0 0 16px}
.content li{margin:0}
.content li+li{margin-top:0.25em}
.content code{background:var(--card2);border:none;border-radius:6px;padding:2px 7px;font-size:0.9em;font-family:var(--mono);color:var(--fg)}
.content pre{background:var(--pre-bg);color:var(--pre-fg);border:1px solid var(--pre-line);border-radius:6px;padding:16px;overflow-x:auto;line-height:1.45;font-size:90%;font-family:var(--mono)}
.content pre code{background:none;padding:0;font-size:100%;border-radius:0;border:none}
.content blockquote{color:var(--fg3);border-left:4px solid var(--line);background:transparent;margin:0 0 16px;padding:0 15px}
.content blockquote p{margin:8px 0}
.tbl-wrap{overflow-x:auto;margin:0 0 16px;border:1px solid var(--line);border-radius:var(--radius-m);box-shadow:var(--shadow-1);background:var(--card)}
table.md-table{border-collapse:collapse;width:100%;font-size:16px;margin:0}
.md-table th,.md-table td{border:none;border-bottom:1px solid var(--line);padding:9px 14px;text-align:left}
.md-table th{background:var(--accent-subtle);font-weight:600;color:var(--fg);border-bottom:1px solid var(--line2)}
.md-table tr:last-child td{border-bottom:none}
.md-table tr:hover td{background:var(--card2)}
details{border:1px solid var(--line);border-radius:var(--radius-m);padding:8px 16px;margin:0 0 16px;background:transparent}
summary{cursor:pointer;color:var(--fg);font-weight:600}
/* 正文折叠块：+ 旋转为 × 的展开指示（viz-src 摘要行保持原样）*/
.content details:not(.viz-src) summary{list-style:none;position:relative;padding-left:26px;user-select:none}
.content details:not(.viz-src) summary::-webkit-details-marker{display:none}
.content details:not(.viz-src) summary::before{content:'';position:absolute;left:9px;top:50%;width:7px;height:7px;margin-top:-5px;
  border-right:1.5px solid var(--accent);border-bottom:1.5px solid var(--accent);transform:rotate(-45deg);transition:transform .18s ease}
.content details:not(.viz-src)[open] summary::before{transform:rotate(45deg)}
details[open] summary{border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:8px;border-radius:0}
hr{border:none;height:2px;border-radius:1px;background:linear-gradient(90deg,var(--line2),transparent);margin:28px 0}
.derivation{background:var(--card2);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;padding:6px 20px 12px;margin:0 0 16px}
.derivation .d-title{font-weight:600;color:var(--accent);margin:12px 0 4px}
img{border-radius:6px;border:1px solid var(--line);max-width:100%}
/* === 代码块卡片（语言标签 + 复制）=== */
.code-card{border:1px solid var(--pre-line);border-radius:6px;margin:0 0 16px;background:var(--pre-bg);overflow:hidden}
.code-head{display:flex;justify-content:space-between;align-items:center;padding:7px 12px 7px 10px;background:transparent;border-bottom:1px solid var(--pre-line)}
.code-lang{font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--accent);background:var(--accent-subtle);border-radius:999px;padding:2px 10px;text-transform:lowercase;line-height:1.5}
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
/* === 面包屑（eyebrow 式章节徽章）=== */
.breadcrumb{font-size:13px;color:var(--fg3);margin-bottom:18px;display:flex;align-items:center;flex-wrap:wrap;gap:2px 0}
.breadcrumb a{color:var(--fg3);text-decoration:none}
.breadcrumb a:hover{color:var(--accent);text-decoration:underline}
.breadcrumb span{margin:0 7px;color:var(--fg4)}
.bc-chip{display:inline-flex;align-items:center;gap:5px;font-size:12.5px;font-weight:600;color:var(--accent);
         background:var(--accent-subtle);border-radius:999px;padding:2px 11px;letter-spacing:.3px}
.breadcrumb .bc-chip a,.bc-chip a{color:var(--accent)}
.breadcrumb .bc-chip a:hover{text-decoration:none;filter:brightness(1.08)}
.bc-chip::before{content:'';width:6px;height:6px;border-radius:2px;background:var(--grad-brand);flex:none}
/* === 进度 / 分页 / 返回顶部 / 阅读进度 === */
.progress-bar{margin-top:36px;padding:16px 20px;border:1px solid var(--line);border-radius:var(--radius-l);display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;box-shadow:var(--shadow-1);
              background:linear-gradient(135deg,var(--accent-subtle),transparent 55%),var(--card2)}
.progress-text{font-size:13.5px;color:var(--fg3);font-weight:500}
.btn-success{padding:4px 20px;background:var(--grad-brand);color:#ffffff;border:1px solid transparent;border-radius:999px;cursor:pointer;font-size:14px;font-weight:600;min-height:34px;
             transition:transform .15s ease,box-shadow .15s ease,filter .15s ease}
.btn-success:hover{filter:brightness(1.06);transform:translateY(-1px);box-shadow:var(--shadow-btn)}
.btn-success:active{animation:pop .25s ease}
@keyframes pop{50%{transform:scale(.94)}}
.btn-success.done{background:var(--card);color:var(--fg3);border:1px solid var(--line2);font-weight:500}
.pager{display:flex;justify-content:space-between;align-items:stretch;margin-top:44px;border-top:1px solid var(--line);padding-top:24px;gap:14px}
.pg-spacer{flex:1}
.pg-card{flex:1;min-width:0;max-width:440px;display:flex;flex-direction:column;gap:4px;padding:12px 16px;border:1px solid var(--line);border-radius:var(--radius-m);
  background:var(--card);text-decoration:none;box-shadow:var(--shadow-1);position:relative;
  transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}
.pg-card:hover{transform:translateY(-2px);border-color:var(--accent);box-shadow:var(--shadow-2)}
.pg-card::after{content:'';position:absolute;top:0;bottom:0;width:3px;background:var(--grad-brand);opacity:0;transition:opacity .18s ease}
.pg-prev::after{left:0;border-radius:var(--radius-m) 0 0 var(--radius-m)}
.pg-next::after{right:0;border-radius:0 var(--radius-m) var(--radius-m) 0}
.pg-card:hover::after{opacity:1}
.pg-label{font-size:12px;font-weight:600;color:var(--accent);letter-spacing:.4px;display:flex;align-items:center;gap:5px}
.pg-title{font-size:15px;font-weight:600;color:var(--fg);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pg-card:hover .pg-title{color:var(--accent)}
.pg-arrow{font-weight:400;transition:transform .16s ease}
.pg-prev:hover .pg-arrow{transform:translateX(-3px)}
.pg-next:hover .pg-arrow{transform:translateX(3px)}
.pg-next{text-align:right;align-items:flex-end}
#readProgress{position:fixed;top:var(--topbar-h);left:0;height:3px;width:0;z-index:15;pointer-events:none;border-radius:0 2px 2px 0;
  background:linear-gradient(90deg,var(--accent),var(--brand));transition:width .08s linear}
#backToTop{position:fixed;bottom:28px;right:28px;width:44px;height:44px;background:var(--card);color:var(--accent);border:1px solid var(--line);
  border-radius:50%;cursor:pointer;font-size:17px;z-index:100;box-shadow:var(--shadow-2);
  display:flex;align-items:center;justify-content:center;
  opacity:0;transform:translateY(10px) scale(.9);pointer-events:none;
  transition:opacity .22s ease,transform .22s ease,border-color .22s ease}
#backToTop.show{opacity:1;transform:none;pointer-events:auto}
#backToTop.show:hover{border-color:var(--accent);transform:translateY(-2px)}
#backToTop::before{content:'';position:absolute;inset:-1px;border-radius:50%;
  background:conic-gradient(var(--brand) var(--p,0%),transparent 0);
  -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 3px),#000 calc(100% - 2.5px));
  mask:radial-gradient(farthest-side,transparent calc(100% - 3px),#000 calc(100% - 2.5px))}
/* === 搜索（GitHub command palette 风）=== */
.search-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:var(--overlay);z-index:1000;display:flex;justify-content:center;padding:12vh 16px 0;
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px)}
.search-box{width:100%;max-width:640px;background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 24px 48px -12px rgba(0,0,0,0.4);overflow:hidden;max-height:64vh;display:flex;flex-direction:column;transition:border-color .16s ease,box-shadow .16s ease}
.search-modal:focus-within .search-box{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft),0 24px 48px -12px rgba(0,0,0,0.4)}
@media (prefers-reduced-motion:no-preference){
  .search-modal{animation:sm-fade .16s ease}
  .search-box{animation:sm-in .18s cubic-bezier(.2,.7,.3,1)}
  @keyframes sm-fade{from{opacity:0}}
  @keyframes sm-in{from{opacity:0;transform:translateY(-10px) scale(.97)}}
}
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
.content blockquote.bq-note{border-left-color:var(--accent);background:var(--accent-subtle);border-radius:0 var(--radius-s) var(--radius-s) 0}
.content blockquote.bq-warn{border-left-color:var(--warn);background:rgba(154,103,0,0.06);border-radius:0 var(--radius-s) var(--radius-s) 0}
:root[data-theme=dark] .content blockquote.bq-warn{background:rgba(210,153,34,0.08)}
.content blockquote.bq-key{border-left-color:var(--success);background:var(--success-subtle);border-radius:0 var(--radius-s) var(--radius-s) 0}
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
.toc a.cur{color:var(--accent);border-left-color:transparent;font-weight:500;position:relative}
.toc a.cur::before{content:'';position:absolute;left:-2px;top:4px;bottom:4px;width:3px;border-radius:2px;background:var(--grad-brand)}
.toc a.lv3{padding-left:26px;font-size:12.5px}
/* === 图片点击放大 === */
.content img{cursor:zoom-in}
.img-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:var(--overlay);z-index:1100;display:flex;align-items:center;justify-content:center;cursor:zoom-out}
.img-overlay img{max-width:92vw;max-height:92vh;border-radius:6px;box-shadow:0 24px 48px -12px rgba(0,0,0,0.5)}
/* === 源码查看页（xxx.py → xxx.py.html）=== */
.file-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:7px 12px;background:var(--card2);border:1px solid var(--line);border-bottom:none;border-radius:var(--radius-m) var(--radius-m) 0 0}
.file-name{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--fg);word-break:break-all}
.file-raw{font-size:12px;font-weight:500;color:var(--fg3);text-decoration:none;border:1px solid var(--btn-line);border-radius:6px;padding:2px 10px;background:var(--btn-bg);white-space:nowrap;flex:none}
.file-raw:hover{color:var(--fg);border-color:var(--btn-line-hover)}
.file-view .highlight{margin:0;border-radius:0 0 var(--radius-m) var(--radius-m)}
.file-view .highlight pre{padding:12px 0 12px 16px}
/* === 页内交互函数图（```plot 围栏块，渲染见 _assets/plot.js）=== */
.inline-plot{border:1px solid var(--line);border-radius:6px;margin:0 0 16px;background:var(--card);overflow:hidden}
.ip-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;padding:8px 14px;background:var(--card2);border-bottom:1px solid var(--line)}
.ip-title{font-size:14px;font-weight:600;color:var(--fg)}
.ip-sub{font-size:12px;color:var(--fg3)}
.ip-wrap{position:relative;width:100%;aspect-ratio:16/9;min-height:230px}
.inline-plot canvas{display:block;width:100%;height:100%;cursor:crosshair;touch-action:none}
.ip-legend{display:flex;flex-wrap:wrap;gap:6px 18px;padding:8px 14px;border-top:1px solid var(--line);font-size:13px;color:var(--fg3)}
.ip-item{display:inline-flex;align-items:center;gap:7px}
.ip-dot{width:10px;height:10px;border-radius:3px;display:inline-block;flex:none}
.ip-item code{background:var(--code-bg);padding:1px 8px;border-radius:6px;font-size:12px;font-family:var(--mono);color:var(--fg);min-width:52px;text-align:center}
.ip-note{padding:6px 14px 10px;font-size:12px;color:var(--fg4)}
.plot-error{border:1px solid var(--warn);border-radius:6px;padding:8px 14px;margin:16px 0;color:var(--warn);font-size:13px}
/* === 页内训练样本可视化（```chunk 围栏块，渲染见 _assets/plot.js）=== */
.chunk-viz{border:1px solid var(--line);border-radius:6px;margin:0 0 16px;background:var(--card);overflow:hidden}
.cv-head{padding:8px 14px;background:var(--card2);border-bottom:1px solid var(--line)}
.cv-title{font-size:14px;font-weight:600;color:var(--fg)}
.cv-sub{padding:8px 14px;font-size:13px;color:var(--fg3);border-bottom:1px solid var(--line)}
.cv-list{display:flex;flex-direction:column;gap:4px;padding:12px}
.cv-row{display:flex;align-items:center;gap:8px 12px;padding:6px 10px;border-radius:6px;flex-wrap:wrap;
        opacity:0;transform:translateY(6px);transition:opacity .35s ease,transform .35s ease,background .15s}
.cv-in .cv-row{opacity:1;transform:none}
.cv-row:hover{background:var(--card2)}
.cv-id{display:inline-flex;align-items:center;justify-content:center;min-width:36px;height:24px;background:var(--card2);
       border:1px solid var(--line);color:var(--fg3);font-size:12px;font-weight:600;border-radius:12px;padding:0 10px;
       font-family:var(--mono);flex:none;transition:color .15s,border-color .15s}
.cv-row:hover .cv-id{color:var(--accent);border-color:var(--accent)}
.cv-xs{display:flex;align-items:center;gap:4px;flex-wrap:wrap;flex:1 1 auto;min-width:100px}
.cv-tok{display:inline-flex;align-items:center;justify-content:center;min-width:30px;height:26px;padding:0 8px;
        border-radius:6px;font-family:var(--mono);font-size:13px;font-weight:500;
        background:var(--accent-subtle);color:var(--accent);transition:transform .12s}
.cv-row:hover .cv-tok{transform:translateY(-1px)}
.cv-tok.is-new{background:var(--accent);color:#ffffff;font-weight:600;box-shadow:0 1px 6px var(--accent-soft)}
.cv-tok.is-new::after{content:'✦';font-size:9px;margin-left:3px;opacity:.75;font-weight:400}
.cv-arrow{color:var(--fg4);font-size:16px;flex:none}
.cv-yl{color:var(--fg3);font-size:12px;flex:none;margin-right:-6px}
.cv-y{display:inline-flex;align-items:center;justify-content:center;min-width:40px;height:28px;padding:0 12px;
      border-radius:6px;background:var(--danger);color:#ffffff;font-family:var(--mono);font-size:14px;font-weight:600;
      flex:none;transition:transform .12s}
.cv-row:hover .cv-y{transform:scale(1.05)}
.cv-legend{display:flex;flex-wrap:wrap;gap:8px 20px;padding:8px 14px;border-top:1px solid var(--line);font-size:12px;color:var(--fg3)}
.cv-lg{display:inline-flex;align-items:center;gap:7px}
.cv-sw{width:16px;height:16px;border-radius:5px;display:inline-block;flex:none}
.cv-sw.sw-old{background:var(--accent-subtle);border:1px solid var(--accent)}
.cv-sw.sw-new{background:var(--accent)}
.cv-sw.sw-y{background:var(--danger)}
/* === 运行输出卡片（═══ 标题 ═══ 横幅块自动转换）=== */
.out-card{border:1px solid var(--pre-line);border-radius:6px;margin:0 0 16px;background:var(--pre-bg);overflow:hidden}
.widget-embed{margin:18px 0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--card)}
.widget-embed .widget-bar{display:flex;justify-content:space-between;align-items:center;padding:6px 12px;border-bottom:1px solid var(--line);background:var(--card2);font-size:12.5px;color:var(--fg3)}
.widget-embed .widget-bar a{color:var(--fg3);text-decoration:none}
.widget-embed .widget-bar a:hover{color:var(--accent)}
.widget-embed iframe{display:block;width:100%;border:0}
.out-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:7px 12px;background:var(--card2);border-bottom:1px solid var(--pre-line)}
.out-title{font-size:13px;font-weight:600;color:var(--fg)}
.out-head-r{display:flex;align-items:center;gap:8px;flex:none}
.out-tag{font-size:11px;color:var(--fg4);border:1px solid var(--line);border-radius:999px;padding:0 8px;line-height:1.7}
.out-card pre{margin:0;border:none;border-radius:0;background:transparent;padding:12px 14px;overflow-x:auto;
              font-size:85%;line-height:1.45;font-family:var(--mono);color:var(--pre-fg)}
.out-card pre code{background:none;padding:0;font-size:100%;font-family:inherit}
.code-lang:empty{display:none}
/* === 页内流程可视化（流程型 text 图示块自动转换，chunk 风格视觉语法）=== */
.flow-card{border:1px solid var(--line);border-radius:6px;margin:16px 0;background:var(--card);padding:16px 14px;overflow-x:auto}
.fl-col{display:flex;flex-direction:column;align-items:center;min-width:max-content;margin:0 auto}
.fl-row{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:6px;max-width:100%}
.fl-node{display:inline-flex;align-items:center;padding:3px 12px;border-radius:6px;background:var(--accent-subtle);
         color:var(--accent);font-family:var(--mono);font-size:13px;font-weight:500;line-height:1.7;
         text-align:center;max-width:100%;word-break:break-word}
.fl-harr{color:var(--fg4);font-size:15px;flex:none}
.fl-varr{color:var(--fg4);font-size:13px;line-height:1;padding:4px 0}
.fl-note{font-size:12px;color:var(--fg3)}
.fl-gap{height:8px}
/* === viz 组件（基于原文 authored 生成：语义分色 + 图例 + 标题）=== */
.viz-card,.viz-tree{border:1px solid var(--line);border-radius:6px;margin:0 0 16px;background:var(--card);overflow:hidden}
.viz-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;padding:8px 14px;background:var(--card2);border-bottom:1px solid var(--line)}
.viz-title{font-size:14px;font-weight:600;color:var(--fg)}
.viz-sub{font-size:12px;color:var(--fg3)}
.viz-body{padding:16px 14px;display:flex;flex-direction:column;align-items:center;overflow-x:auto}
.viz-node{display:inline-flex;align-items:center;padding:3px 12px;border-radius:6px;font-family:var(--mono);
          font-size:13px;font-weight:500;line-height:1.7;text-align:center;max-width:100%;word-break:break-word}
.viz-k-input{background:var(--card2);border:1px solid var(--line2);color:var(--fg)}
.viz-k-mid{background:var(--accent-subtle);color:var(--accent)}
.viz-k-op{background:var(--accent);color:#ffffff;font-weight:600}
.viz-k-output{background:var(--danger);color:#ffffff;font-weight:600}
.viz-legend{display:flex;flex-wrap:wrap;gap:8px 20px;padding:8px 14px;border-top:1px solid var(--line);font-size:12px;color:var(--fg3)}
.viz-legend .cv-sw.viz-k-input{border:1px solid var(--line2);background:var(--card2)}
.viz-legend .cv-sw.viz-k-mid{background:var(--accent-subtle)}
.viz-legend .cv-sw.viz-k-op{background:var(--accent)}
.viz-legend .cv-sw.viz-k-output{background:var(--danger)}
.viz-src{border-top:1px solid var(--line)}
.viz-src summary{font-size:12px;color:var(--fg4);cursor:pointer;padding:6px 14px;user-select:none}
.viz-src summary:hover{color:var(--fg3)}
.viz-src pre{margin:0;padding:0 14px 12px;font-size:12px;line-height:1.5;font-family:var(--mono);color:var(--fg3);overflow-x:auto}
.viz-stale{padding:8px 14px;border-top:1px solid var(--warn);color:var(--warn);font-size:12px;background:var(--card2)}
/* panel：PPT 图示风模块面板（viz type=panel）*/
.viz-k-green{background:var(--success);color:#ffffff;font-weight:600}
.viz-legend .cv-sw.viz-k-green{background:var(--success)}
.pm-body{display:block;padding:16px}
.pm-container{position:relative;border:2px solid var(--line2);border-radius:8px;padding:16px 14px 6px;background:var(--card2)}
.pm-container:hover{border-color:var(--fg4)}
.pm-container-label{position:absolute;top:-10px;left:14px;background:var(--card);padding:0 10px;font-size:13px;font-weight:600;color:var(--fg)}
.pm-module{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin-bottom:10px;flex-wrap:wrap;transition:border-color .15s,box-shadow .15s}
.pm-module:hover{border-color:var(--line2);box-shadow:0 1px 6px rgba(0,0,0,0.07)}
.pm-badge{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#ffffff;font-weight:700;flex:none;font-size:15px}
.pm-badge.viz-k-op{background:var(--accent)}
.pm-badge.viz-k-green{background:var(--success)}
.pm-badge.viz-k-input{background:var(--fg4)}
.pm-badge.viz-k-output{background:var(--danger)}
.pm-content{flex:1;display:flex;flex-direction:column;gap:3px;min-width:140px}
.pm-name{font-size:15px;font-weight:600;color:var(--fg)}
.pm-desc{font-size:13px;color:var(--fg3);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pm-tag{font-family:var(--mono);font-size:12px;font-weight:500;padding:1px 10px;border-radius:6px}
.pm-tag.viz-k-op{background:var(--accent-subtle);color:var(--accent)}
.pm-tag.viz-k-green{background:var(--success-subtle);color:var(--success)}
.pm-tag.viz-k-mid{background:var(--accent-subtle);color:var(--accent)}
.pm-icon{font-size:24px;opacity:.55;flex:none}
.pm-stack{margin-top:14px;border:1px solid var(--line2);border-radius:8px;padding:12px;background:var(--card2);display:flex;flex-direction:column;align-items:center;gap:8px}
.pm-stack-label{font-size:13px;font-weight:500;color:var(--fg);text-align:center}
.sb-wrap{display:flex;flex-direction:column;align-items:center;gap:3px}
.sb-block{display:flex;width:130px;height:30px;border-radius:8px;overflow:hidden;border:1px solid var(--line2);background:var(--card)}
.sb-seg{flex:1;opacity:.75}
.sb-seg.viz-k-op{background:var(--accent)}
.sb-seg.viz-k-green{background:var(--success)}
.sb-seg.viz-k-output{background:var(--danger)}
.sb-arrow{color:var(--fg4);font-size:17px;line-height:1}
.sb-ellipsis{color:var(--fg4);letter-spacing:3px;font-size:15px}
/* highway：残差流主干 + 子层盒（viz type=highway）*/
.viz-highway{border:1px solid var(--line);border-radius:6px;margin:0 0 16px;background:var(--card);overflow:hidden}
.hw-stage{position:relative;padding:16px 12px 6px;overflow-x:auto}
.hw-trunk{display:flex;align-items:center;gap:14px}
.hw-mid{flex:1}
.hw-branches{display:flex;gap:12px;margin-top:52px;padding:0 4px}
.hw-slot{flex:1;display:flex;justify-content:center;min-width:0}
.hw-block{border:1.5px solid var(--fg4);border-radius:8px;padding:9px 18px;text-align:center;background:var(--card);max-width:100%}
.hw-block.op{border-color:var(--accent);background:var(--accent-subtle)}
.hw-block.green{border-color:var(--success);background:var(--success-subtle)}
.hw-block.mid{border-style:dashed;background:transparent}
.hw-btitle{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--fg)}
.hw-block.op .hw-btitle{color:var(--accent)}
.hw-block.green .hw-btitle{color:var(--success)}
.hw-bsub{font-size:11.5px;color:var(--fg3);margin-top:2px;white-space:nowrap}
.hw-foot{margin-top:12px;font-size:12px;color:var(--fg4);text-align:center}
.hw-formula{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:6px;padding:10px 12px;border-top:1px dashed var(--line);font-family:var(--mono);font-size:12.5px}
.hw-fchip{background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:2px 10px;color:var(--fg);white-space:nowrap}
.hw-farrow{color:var(--fg4)}
.hw-svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.hw-leg-line{display:inline-block;width:20px;height:2px;background:var(--fg4);vertical-align:middle;margin-right:1px}
.hw-leg-add{color:var(--warn);font-size:15px;font-weight:700;margin-right:1px}
/* 树布局（viz type=tree，连接线由 plot.js 以 SVG 绘制）*/
.tr-stage{position:relative;padding:14px 10px;overflow-x:auto}
.tr-level{display:flex;justify-content:space-around;gap:6px;margin:14px 0}
.tr-slot{flex:1;display:flex;justify-content:center;min-width:0}
.tr-svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
/* === 卡片族统一层次（后置覆盖：统一圆角与浅阴影，代码/图表/输出/嵌入卡一致）=== */
.code-card,.out-card,.inline-plot,.chunk-viz,.viz-card,.viz-tree,.viz-highway,.flow-card,.widget-embed,.file-view{border-radius:var(--radius-m);box-shadow:var(--shadow-1)}
.code-card .code-head{border-radius:var(--radius-m) var(--radius-m) 0 0}
/* === 首页 hero + Part 课程地图 === */
.hero{position:relative;margin:6px 0 26px;padding:34px 34px 30px;border-radius:var(--radius-l);overflow:hidden;
      border:1px solid var(--line);box-shadow:var(--shadow-1);
      background:linear-gradient(135deg,var(--accent-subtle),transparent 55%),var(--card2)}
.hero::before{content:'';position:absolute;top:-70px;right:-50px;width:280px;height:280px;border-radius:50%;pointer-events:none;
      background:radial-gradient(circle,color-mix(in srgb,var(--brand) 20%,transparent),transparent 70%)}
.hero-eyebrow{font-size:13px;font-weight:600;color:var(--accent);letter-spacing:1.5px;margin-bottom:8px}
.hero .hero-title{font-size:2.1em;font-weight:bold;line-height:1.2;margin:0 0 12px;color:var(--fg)}
.hero-sub{font-size:15.5px;color:var(--fg3);max-width:620px;margin:0 0 20px}
.hero-stats{display:flex;flex-wrap:wrap;gap:8px 30px;margin:0 0 22px}
.hero-stat{display:flex;flex-direction:column;gap:1px}
.hs-num{font-size:22px;font-weight:700;color:var(--fg);font-variant-numeric:tabular-nums;line-height:1.2}
.hs-label{font-size:12.5px;color:var(--fg4)}
.hero-cta{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.btn-primary{display:inline-flex;align-items:center;gap:7px;padding:9px 24px;border-radius:999px;background:var(--accent);color:#ffffff;
  font-size:15px;font-weight:600;text-decoration:none;border:1px solid transparent;box-shadow:var(--shadow-btn);
  transition:transform .16s ease,box-shadow .16s ease}
.btn-primary:hover{transform:translateY(-1px);box-shadow:var(--shadow-2);color:#ffffff;text-decoration:none}
.btn-ghost{display:inline-flex;align-items:center;gap:8px;padding:8px 18px;border-radius:999px;border:1px solid var(--line2);
  background:var(--card);color:var(--fg);font-size:14px;font-weight:500;cursor:pointer;line-height:1.5;
  transition:border-color .16s ease,transform .16s ease,color .16s ease}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
.hero-progress{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--fg3)}
.hp-ring{width:36px;height:36px;border-radius:50%;flex:none;position:relative;display:flex;align-items:center;justify-content:center;
  background:conic-gradient(var(--brand) var(--p,0%),var(--line) 0)}
.hp-ring::before{content:'';position:absolute;inset:4px;border-radius:50%;background:var(--card2)}
.hp-ring span{position:relative;font-size:10px;font-weight:700;color:var(--fg3);font-variant-numeric:tabular-nums}
.part-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));gap:14px;margin:18px 0 10px}
.part-card{display:flex;flex-direction:column;gap:9px;padding:15px 16px 13px;border:1px solid var(--line);border-radius:var(--radius-m);
  background:var(--card);text-decoration:none;box-shadow:var(--shadow-1);position:relative;overflow:hidden;
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease}
.part-card::after{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--grad-brand);opacity:0;transition:opacity .18s ease}
.part-card:hover{transform:translateY(-3px);border-color:var(--accent);box-shadow:var(--shadow-2)}
.part-card:hover::after{opacity:1}
.pc-head{display:flex;align-items:flex-start;gap:10px}
.pc-num{flex:none;width:34px;height:34px;border-radius:var(--radius-s);display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:14px;font-weight:700;color:#ffffff;background:var(--accent)}
.part-card.pc-done .pc-num{background:var(--success)}
.pc-title{font-size:14.5px;font-weight:600;color:var(--fg);line-height:1.35;padding-top:5px;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.part-card:hover .pc-title{color:var(--accent)}
.pc-meta{display:flex;align-items:center;justify-content:space-between;font-size:12.5px;color:var(--fg4)}
.pc-prog{font-weight:600;font-variant-numeric:tabular-nums}
.part-card.pc-done .pc-prog{color:var(--success)}
.pc-bar{height:4px;border-radius:2px;background:var(--card2);overflow:hidden}
.pc-bar i{display:block;height:100%;width:0;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--brand));
  transition:width .5s ease}
@media (max-width:600px){
  .hero{padding:24px 20px}
  .hero .hero-title{font-size:1.7em}
  .hero-stats{gap:6px 20px}
}
/* === 窄屏 === */
@media (max-width:900px){
  .layout{display:block}
  .side{display:block;position:fixed;top:var(--topbar-h);left:0;bottom:0;width:min(320px,86vw);height:auto;z-index:20;
        background:var(--side-bg);padding:12px;border-right:1px solid var(--line);box-shadow:var(--shadow-2);
        transform:translateX(-102%);visibility:hidden;
        transition:transform .28s cubic-bezier(.2,.7,.3,1),visibility 0s .28s}
  body.side-open .side{transform:none;visibility:visible;transition:transform .28s cubic-bezier(.2,.7,.3,1)}
  body.side-open{overflow:hidden}
  body.side-open::before{content:'';position:fixed;inset:0;z-index:19;background:var(--overlay)}
  @media (prefers-reduced-motion:no-preference){
    body.side-open::before{animation:sm-fade .2s ease}
  }
  .content{max-width:100%;padding:18px 16px 70px}
  .content h1{font-size:1.5em}
  .content h2{font-size:1.3em}
  .tbl-wrap{margin:12px -16px;width:calc(100% + 32px);border-radius:0;border-left:none;border-right:none}
  .pager{flex-direction:column}
  .pg-spacer{display:none}
  .pg-card{max-width:100%}
  .pg-next{text-align:left;align-items:flex-start}
  .topbar{padding:0 12px}
  .brand .version{display:none}
  #backToTop{bottom:20px;right:20px}
}
@media (max-width:480px){
  .content{padding:16px 12px 60px}
  .tbl-wrap{margin:12px -12px;width:calc(100% + 24px);border-radius:0;border-left:none;border-right:none}
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

    def sort_key(rel):
        name = os.path.basename(rel)[:-3]
        if name == 'README':
            return (0, '')
        return (1, name)

    # Part → 教程页清单（侧栏与首页课程地图共用）
    parts = {}
    for rel in pages:
        m = re.match(r'(courses/Part\d+_[^/]+)/', rel)
        if m:
            parts.setdefault(m.group(1), []).append(rel)
    asg_rels = [r for r in pages
                if re.match(r'assignments/assignment_\d+/[^/]+\.md', r)
                and os.path.basename(r) in ('assignment.md', 'README.md')]

    def nav_html(cur):
        def item(rel):
            name = os.path.basename(rel)[:-3]
            disp = 'README · 导览' if name == 'README' else name
            cls = ' class="cur"' if rel == cur else ''
            return f'<div class="nav-item"><a{cls} href="/{rel[:-3]}.html">{esc(disp)}</a></div>'

        def section(title, rels, open_if, badge=''):
            op = ' open' if open_if else ''
            buf.append(f'<details class="nav-sec"{op}><summary>{badge}<span class="nav-txt">{esc(title)}</span></summary>')
            for rel in sorted(rels, key=sort_key):
                buf.append(item(rel))
            buf.append('</details>')

        buf = ['<div class="nav-home"><a class="nav-home-btn" href="/index.html">📑 课程首页</a>'
               '<button id="navToggleAll" type="button">展开全部</button></div>']
        for k in sorted(parts, key=lambda x: int(re.search(r'Part(\d+)', x).group(1))):
            num = int(re.search(r'Part(\d+)', k).group(1))
            section(PART_TITLES.get(num, ''), parts[k], cur.startswith(k + '/'),
                    badge=f'<span class="nav-badge">{num}</span>')
        section('参考文档', [r for r in pages if r.startswith('docs/')], cur.startswith('docs/'),
                badge='<span class="nav-badge nb-ico">§</span>')
        buf.append('<details class="nav-sec"><summary><span class="nav-badge nb-ico">✎</span>'
                   f'<span class="nav-txt">课后作业（{len(asg_rels)} 套）</span></summary>')
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
        """生成面包屑导航 HTML（中段 chip 徽章化，与首页 eyebrow 语言一致）。"""
        parts = ['<a href="/index.html">首页</a>']
        if rel == 'index.md':
            return ''
        # 解析路径层级
        segs = rel.replace('.md', '').split('/')
        if segs[0] == 'courses' and len(segs) >= 2:
            m = re.match(r'Part(\d+)', segs[1])
            if m:
                num = int(m.group(1))
                parts.append(f'<span class="bc-chip"><a href="/courses/{segs[1]}/tutorial/README.html">Part {num} · {PART_TITLES.get(num, "")}</a></span>')
                if len(segs) >= 4 and segs[2] == 'tutorial':
                    title = extract_title(open(os.path.join(docs, rel), encoding='utf-8').read(), segs[3])
                    parts.append(esc(title))
        elif segs[0] == 'assignments':
            parts.append('<span class="bc-chip">课后作业</span>')
            if len(segs) >= 2:
                parts.append(esc(segs[1].capitalize()))
        elif segs[0] == 'docs':
            parts.append('<span class="bc-chip">参考文档</span>')
            title = extract_title(open(os.path.join(docs, rel), encoding='utf-8').read(), segs[-1])
            parts.append(esc(title))
        elif segs[0] == 'maps':
            parts.append('<span class="bc-chip">知识脉络图</span>')
        elif segs[0] == 'quizzes':
            parts.append('<span class="bc-chip">测验系统</span>')
        return ' <span>/</span> '.join(parts)

    # 全站标题映射：pager 卡片 / 首页 courseMap / 搜索索引共用
    titles = {}
    for rel in pages:
        titles[rel] = '课程首页' if rel == 'index.md' else extract_title(
            open(os.path.join(docs, rel), encoding='utf-8').read(), os.path.basename(rel)[:-3])

    def hero_html():
        """课程首页 hero + Part 课程地图（完成度由页面 JS 按 localStorage 进度填充）。"""
        ordered = sorted(parts, key=lambda x: int(re.search(r'Part(\d+)', x).group(1)))
        n_chapters = sum(len(parts[k]) for k in ordered)
        wdir = os.path.join(docs, 'widgets')
        n_widgets = len([f for f in os.listdir(wdir)
                         if f.endswith('.html') and 'visual-check' not in f]) if os.path.isdir(wdir) else 0
        start_href = f'/{sorted(parts[ordered[0]], key=sort_key)[0][:-3]}.html' if ordered else '/index.html'
        cards, course_map = [], []
        for k in ordered:
            num = int(re.search(r'Part(\d+)', k).group(1))
            rels = sorted(parts[k], key=sort_key)
            cards.append(
                f'<a class="part-card" data-part="{num}" href="/{rels[0][:-3]}.html">'
                f'<span class="pc-head"><span class="pc-num">{num}</span>'
                f'<span class="pc-title">{esc(PART_TITLES.get(num, ""))}</span></span>'
                f'<span class="pc-meta"><span>{len(rels)} 章</span><span class="pc-prog">0/{len(rels)}</span></span>'
                f'<span class="pc-bar"><i></i></span></a>')
            for r in rels:
                course_map.append({'part': num, 'url': '/' + r[:-3] + '.html', 'title': titles[r]})
        cmap = json.dumps(course_map, ensure_ascii=False, separators=(',', ':'))
        hero = f'''<section class="hero">
<div class="hero-eyebrow">NEURAL NETWORKS: ZERO TO HERO · 中文实战教程</div>
<div class="hero-title">🌱 Makemore 中文教程</div>
<p class="hero-sub">基于 Andrej Karpathy 的 Zero to Hero 系列，从字符级语言模型起步，一路手写到大语言模型——代码可跑、数字可复现。</p>
<div class="hero-stats">
<div class="hero-stat"><span class="hs-num">{len(ordered)}</span><span class="hs-label">Part 课程</span></div>
<div class="hero-stat"><span class="hs-num">{n_chapters}</span><span class="hs-label">章教程</span></div>
<div class="hero-stat"><span class="hs-num">{n_widgets}</span><span class="hs-label">交互演示</span></div>
<div class="hero-stat"><span class="hs-num">{len(asg_rels)}</span><span class="hs-label">课后作业</span></div>
</div>
<div class="hero-cta">
<a class="btn-primary" href="{start_href}">开始学习 →</a>
<button class="btn-ghost" id="resumeBtn" type="button" style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">继续上次学习</button>
<div class="hero-progress"><div class="hp-ring" id="hpRing"><span id="hpPct">0%</span></div><span id="hpText">已完成 0 / {len(course_map)} 章</span></div>
</div>
</section>
<h2 id="课程地图">🗺 课程地图</h2>
<div class="part-grid">{''.join(cards)}</div>
<script type="application/json" id="courseMap">{cmap}</script>
<script>''' + """(function(){
  var map;
  try{map=JSON.parse(document.getElementById('courseMap').textContent);}catch(e){return;}
  var prog={};
  try{prog=JSON.parse(localStorage.getItem('mm-progress'))||{};}catch(e){}
  var total=map.length,done=0,byPart={};
  map.forEach(function(m){
    var ok=!!prog[m.url];if(ok)done++;
    if(!byPart[m.part])byPart[m.part]={t:0,d:0};
    byPart[m.part].t++;if(ok)byPart[m.part].d++;
  });
  var pct=total?Math.round(done/total*100):0;
  var ring=document.getElementById('hpRing');
  if(ring){ring.style.setProperty('--p',pct+'%');
    var lab=document.getElementById('hpPct');if(lab)lab.textContent=pct+'%';
    var txt=document.getElementById('hpText');if(txt)txt.textContent='已完成 '+done+' / '+total+' 章';}
  document.querySelectorAll('.part-card').forEach(function(card){
    var s=byPart[card.getAttribute('data-part')];
    if(!s)return;
    var p=card.querySelector('.pc-prog');
    if(p)p.textContent=s.d+'/'+s.t;
    if(s.t&&s.d===s.t)card.classList.add('pc-done');
    var bar=card.querySelector('.pc-bar i');
    if(bar)bar.style.width=(s.t?Math.round(s.d/s.t*100):0)+'%';
  });
  var btn=document.getElementById('resumeBtn');
  if(btn){
    var next=null;
    for(var i=0;i<map.length;i++){if(!prog[map[i].url]){next=map[i];break;}}
    if(next){btn.textContent='继续：'+next.title;btn.onclick=function(){window.location.href=next.url;};}
    else{btn.textContent='✓ 全部完成，去复习';btn.onclick=function(){window.location.href=map.length?map[0].url:'/index.html';};}
  }
})();""" + '</script>\n'
        return hero

    global PAGE_DIR
    n_ok = 0
    for idx, rel in enumerate(pages):
        PAGE_DIR = os.path.dirname(rel)
        RENDER_CTX['page'] = rel
        LAST_PLAIN.update({'body': None, 'index': -1, 'replaced': False})
        text = open(os.path.join(docs, rel), encoding='utf-8').read()
        if rel == 'index.md':
            # hero 已带课程标题，剥离 README 首行 h1 避免重复
            text = re.sub(r'^\s*# .*\r?\n', '', text, count=1)
        body = render_blocks(text)
        title = titles[rel]
        breadcrumb = make_breadcrumb(rel)
        pager = '<span class="pg-spacer"></span>' if idx == 0 else ''
        if idx > 0:
            pager += (f'<a class="pg-card pg-prev" href="/{pages[idx-1][:-3]}.html">'
                      f'<span class="pg-label"><span class="pg-arrow">←</span>上一页</span>'
                      f'<span class="pg-title">{esc(titles[pages[idx-1]])}</span></a>')
        if idx < len(pages) - 1:
            pager += (f'<a class="pg-card pg-next" href="/{pages[idx+1][:-3]}.html">'
                      f'<span class="pg-label">下一页<span class="pg-arrow">→</span></span>'
                      f'<span class="pg-title">{esc(titles[pages[idx+1]])}</span></a>')
        else:
            pager += '<span class="pg-spacer"></span>'
        if rel == 'index.md':
            body = hero_html() + body
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
            if rel.startswith('widgets') and f.endswith('.html'):
                inject_widget_back(dst_f)
                if os.path.splitext(f)[0] in THEME_SYNC_WIDGETS:
                    inject_widget_theme(dst_f)
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
        title = titles[rel]
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
    assets = os.path.join(site, '_assets')
    os.makedirs(assets, exist_ok=True)
    open(os.path.join(assets, 'search-index.json'), 'w', encoding='utf-8').write(json.dumps(search_index, ensure_ascii=False))
    open(os.path.join(assets, 'style.css'), 'w', encoding='utf-8').write(CSS)
    open(os.path.join(assets, 'plot.js'), 'w', encoding='utf-8').write(PLOT_JS)
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

def watch_serve(port):
    """本地服务 + 每秒轮询源指纹：md/图片存盘后约 1~2s 自动重建，刷新浏览器即见。"""
    def loop():
        while True:
            time.sleep(1.0)
            fp, _ = src_fingerprint()
            if fp == read_stamp():
                continue
            print('[WATCH] 检测到源文件改动，重新构建 ...')
            try:
                build()
                write_stamp(src_fingerprint()[0])
                print('[WATCH] 构建完成，刷新浏览器查看最新内容')
            except Exception as e:
                print(f'[WATCH] 构建失败（保留旧站点）：{e}')
    threading.Thread(target=loop, daemon=True).start()
    serve(port)


def _port_after(flag):
    """从 sys.argv 里取 flag 后面的端口号（缺省 8000，非数字则忽略）。"""
    argv = sys.argv
    i = argv.index(flag)
    if i + 1 < len(argv) and argv[i + 1].isdigit():
        return int(argv[i + 1])
    return 8000


if __name__ == '__main__':
    try:   # 重定向/管道下 stdout 变全缓冲，WATCH 消息会滞后，改为行缓冲
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if '--watch' in sys.argv:
        ensure_built('--force' in sys.argv)
        watch_serve(_port_after('--watch'))
    elif '--serve' in sys.argv:
        ensure_built('--force' in sys.argv)
        serve(_port_after('--serve'))
    else:
        ensure_built('--force' in sys.argv)
        print('预览：python build_site_lite.py --serve 8000   （--watch 改动自动重建 / --force 强制全量）')
