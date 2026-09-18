#!/usr/bin/env python3
"""定制模板构建器：从 archify 校验规格的语义与几何生成站内 widget 单文件 HTML。

设计规则见 ../NOTES.md「定制模板做对的事」：
- 设计 token 化（亮/暗双主题，[data-theme=dark] 与站点注入脚本天然兼容）
- 徽章 + 大标题 + 导语 + 三张数据卡 → SVG 图 → 真实类别图例 → 三张结论卡
- 动效分寸：顺序管线可入场分级；汇合/闭环图禁排序、全通路同速常速流动
- 悬停即教学（⊕ 高亮两路输入）；reduced-motion 停动画；零依赖
用法：python3 build_custom.py   （输出到仓库 widgets/*.html）
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WIDGETS = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'widgets'))

# ---------------------------------------------------------------- CSS 模板
STYLE = ''':root {
  --bg:#F7F8FA; --surface:#FFFFFF; --surface-2:#F0F2F5;
  --border:#E5E7EB; --border-strong:#D1D5DB;
  --text:#101318; --text-secondary:#6B7280; --text-muted:#9CA3AF;
  --brand:#4D6BFE; --brand-soft:#EEF1FF;
  --green:#10B981; --violet:#8B5CF6; --amber:#F59E0B;
  --shadow-sm:0 1px 2px rgba(16,19,24,.04); --shadow-md:0 2px 8px rgba(16,19,24,.06);
  --radius:12px;
}
:root[data-theme="dark"] {
  --bg:#0D1117; --surface:#161B22; --surface-2:#21262D;
  --border:#30363D; --border-strong:#3D444D;
  --text:#E6EDF3; --text-secondary:#9198A1; --text-muted:#6E7681;
  --brand:#58A6FF; --brand-soft:#12233C;
  --green:#3FB950; --violet:#AB7DF8; --amber:#E3B341;
  --shadow-sm:0 1px 2px rgba(0,0,0,.3); --shadow-md:0 2px 8px rgba(0,0,0,.35);
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]):not([data-theme="dark"]) {
  --bg:#0D1117; --surface:#161B22; --surface-2:#21262D; --border:#30363D; --border-strong:#3D444D;
  --text:#E6EDF3; --text-secondary:#9198A1; --text-muted:#6E7681;
  --brand:#58A6FF; --brand-soft:#12233C; --green:#3FB950; --violet:#AB7DF8; --amber:#E3B341;
  --shadow-sm:0 1px 2px rgba(0,0,0,.3); --shadow-md:0 2px 8px rgba(0,0,0,.35);
}}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Noto Sans SC",Helvetica,Arial,sans-serif;
  min-height:100vh; display:flex; flex-direction:column; align-items:center;
  padding:36px 20px 48px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
.container { max-width:1080px; width:100%; }
.header { margin-bottom:24px; animation:fadeUp .6s ease both; }
.badge { display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600;
  color:var(--brand); background:var(--brand-soft); border:1px solid color-mix(in srgb,var(--brand) 22%,transparent);
  border-radius:999px; padding:5px 14px; margin-bottom:14px; }
.badge .pulse { width:6px; height:6px; border-radius:50%; background:var(--brand); animation:pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }
h1 { font-size:clamp(22px,3.6vw,32px); font-weight:700; letter-spacing:-.025em; margin-bottom:12px; line-height:1.25; }
h1 em { font-style:normal; color:var(--brand); font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:.82em; }
.subtitle { font-size:14.5px; color:var(--text-secondary); max-width:880px; }
.subtitle strong { color:var(--text); font-weight:600; }
.subtitle code { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12px;
  background:var(--surface-2); border:1px solid var(--border); border-radius:5px; padding:1px 6px; color:var(--text); }
.stat-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin-top:20px; }
.stat-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:15px 17px; box-shadow:var(--shadow-sm); animation:fadeUp .6s ease both; }
.stat-card:nth-child(2){animation-delay:.08s} .stat-card:nth-child(3){animation-delay:.16s}
.stat-label { font-size:11.5px; font-weight:600; color:var(--text-muted); letter-spacing:.04em; margin-bottom:5px; }
.stat-value { font-size:21px; font-weight:700; }
.stat-value.brand{color:var(--brand)} .stat-value.green{color:var(--green)}
.stat-value.violet{color:var(--violet)} .stat-value.amber{color:var(--amber)}
.stat-sub { font-size:12.5px; color:var(--text-secondary); margin-top:3px; }
.diagram-wrap { background:var(--surface); border:1px solid var(--border); border-radius:14px;
  box-shadow:var(--shadow-md); padding:18px 20px 14px; margin-top:22px; animation:fadeUp .6s ease both; animation-delay:.1s; }
.diagram-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }
.diagram-header h2 { font-size:15.5px; font-weight:700; }
.diagram-header .tag { font-size:11px; font-weight:600; color:var(--text-muted);
  background:var(--surface-2); border:1px solid var(--border); border-radius:999px; padding:3px 10px; }
svg { width:100%; height:auto; display:block; }
svg text { font-family:inherit; text-anchor:middle; dominant-baseline:middle; pointer-events:none; }
.node rect { fill:var(--surface); stroke:var(--border-strong); stroke-width:1.2; }
.node { cursor:default; }
.node[data-hl], .node--trig { cursor:pointer; }
.node text.t { font-size:12.5px; font-weight:600; fill:var(--text); }
.node text.s { font-size:10.5px; fill:var(--text-secondary); font-weight:400; }
.node text.s2 { font-size:10px; fill:var(--text-muted); font-weight:400; }
.node text.mono, .elabel.mono { font-family:ui-monospace,Menlo,Consolas,monospace; }
.acc-blue text.t { fill:var(--brand); } .acc-blue rect { stroke:color-mix(in srgb,var(--brand) 40%,var(--border-strong)); }
.acc-green text.t { fill:var(--green); } .acc-green rect { stroke:color-mix(in srgb,var(--green) 40%,var(--border-strong)); }
.acc-violet text.t { fill:var(--violet); } .acc-violet rect { stroke:color-mix(in srgb,var(--violet) 40%,var(--border-strong)); }
.acc-amber text.t { fill:var(--amber); } .acc-amber rect { stroke:color-mix(in srgb,var(--amber) 45%,var(--border-strong)); }
.acc-hero rect { fill:var(--brand-soft); stroke:color-mix(in srgb,var(--brand) 55%,var(--border-strong)); }
.acc-hero text.t { fill:var(--brand); }
.link { stroke:var(--border-strong); stroke-width:1.6; fill:none; }
.link--dashed { stroke-dasharray:5 4; }
.flow { stroke:var(--brand); stroke-width:1.6; fill:none; stroke-dasharray:7 5; opacity:.85; animation:dashFlow 1s linear infinite; }
.flow--green { stroke:var(--green); } .flow--violet { stroke:var(--violet); } .flow--amber { stroke:var(--amber); }
@keyframes dashFlow { to { stroke-dashoffset:-12; } }
.elabel { font-size:10.5px; fill:var(--text-secondary); font-weight:500; }
.band { fill:none; stroke:var(--border-strong); stroke-width:1.2; stroke-dasharray:5 5; }
.band-label { font-size:11.5px; fill:var(--text-muted); font-weight:600; text-anchor:start; }
#arrow path { fill:var(--border-strong); } #arrowBrand path { fill:var(--brand); }
#arrowGreen path { fill:var(--green); } #arrowViolet path { fill:var(--violet); } #arrowAmber path { fill:var(--amber); }
.node,.link,.flow,.elabel,.band,.band-label { transition:opacity .2s; }
svg.fade .node, svg.fade .band, svg.fade .band-label, svg.fade .link, svg.fade .flow, svg.fade .elabel { animation:nodeIn .5s ease backwards; }
svg.sg .node, svg.sg .band { animation:nodeIn .45s ease backwards; }
svg.sg .link, svg.sg .flow, svg.sg .elabel { animation:nodeIn .45s ease backwards; }
.d0{animation-delay:0s}.d1{animation-delay:.08s}.d2{animation-delay:.18s}.d3{animation-delay:.28s}
.d4{animation-delay:.38s}.d5{animation-delay:.48s}.d6{animation-delay:.58s}.d7{animation-delay:.68s}
.d8{animation-delay:.78s}.d9{animation-delay:.88s}
@keyframes nodeIn { from{opacity:0; translate:0 6px} to{opacity:1; translate:0 0} }
.legend { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:12px; padding-top:11px; border-top:1px solid var(--border); }
.legend .item { display:inline-flex; align-items:center; gap:6px; font-size:11.5px; color:var(--text-secondary); }
.sw { width:13px; height:13px; border-radius:4px; border:1.5px solid var(--border-strong); background:var(--surface); flex:none; }
.sw.hero { background:var(--brand-soft); border-color:var(--brand); }
.sw.blue { border-color:var(--brand); } .sw.green { border-color:var(--green); }
.sw.violet { border-color:var(--violet); } .sw.amber { border-color:var(--amber); }
.sw.band-s { width:16px; border-style:dashed; border-radius:5px; }
.sw.dash { width:16px; height:3px; border:none; border-radius:0; background:repeating-linear-gradient(90deg,var(--brand) 0 6px,transparent 6px 10px); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; margin-top:20px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:15px 17px; box-shadow:var(--shadow-sm); animation:fadeUp .6s ease both; }
.card:nth-child(2){animation-delay:.08s} .card:nth-child(3){animation-delay:.16s}
.card h3 { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:700; margin-bottom:7px; }
.dot { width:8px; height:8px; border-radius:50%; flex:none; }
.dot.cyan{background:#22D3EE} .dot.emerald{background:var(--green)} .dot.violet{background:var(--violet)}
.dot.amber{background:var(--amber)} .dot.rose{background:#F43F5E}
.card ul { list-style:none; }
.card li { font-size:12.5px; color:var(--text-secondary); padding-left:13px; position:relative; margin-bottom:4px; }
.card li::before { content:""; position:absolute; left:0; top:8px; width:5px; height:5px; border-radius:50%; background:var(--border-strong); }
.card code { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px;
  background:var(--surface-2); border:1px solid var(--border); border-radius:4px; padding:0 5px; }
footer { margin-top:22px; font-size:11.5px; color:var(--text-muted); text-align:center; }
footer code { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:10.5px; }
@keyframes fadeUp { from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none} }
@media (prefers-reduced-motion: reduce) {
  .flow { animation:none; opacity:.55; } .badge .pulse { animation:none; }
  .stat-card,.card,.diagram-wrap,.header { animation:none; }
  svg.fade .node,svg.fade .band,svg.fade .band-label,svg.fade .link,svg.fade .flow,svg.fade .elabel,
  svg.sg .node,svg.sg .band,svg.sg .link,svg.sg .flow,svg.sg .elabel { animation:none; }
}'''

HOVER_CSS_RULE = 'svg.hl-{g} [data-rel]:not([data-rel~="{g}"]) {{ opacity:.18; }}'
JS = '''(function(){
  document.querySelectorAll('[data-hl]').forEach(function(el){
    var svg = el.closest('svg'); if(!svg) return;
    var g = el.getAttribute('data-hl');
    el.addEventListener('mouseenter', function(){ svg.classList.add('hl-'+g); });
    el.addEventListener('mouseleave', function(){ svg.classList.remove('hl-'+g); });
  });
})();'''

MARKERS = '''<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="arrowBrand" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="arrowGreen" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="arrowViolet" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="arrowAmber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"/></marker>
</defs>'''


class D:
    """图定义累加器"""
    def __init__(self, stagger):
        self.parts = []
        self.groups = set()
        self.stagger = stagger
        self._di = 0

    def node(self, x, y, w, h, title, sub, acc=None, rel='', hid=None, sub2=None,
             mono=False, delay=None, hl=None):
        cls = 'node' + ((' acc-' + acc) if acc else '')
        relattr = f' data-rel="{rel}"' if rel else ''
        idattr = f' id="{hid}"' if hid else ''
        if hl is None and hid:
            hl = hid
        hattr = f' data-hl="{hl}"' if hl else ''
        if hl:
            self.groups.add(hl)
        if self.stagger:
            self._di += 1
            dattr = f' class="{cls} d{min(self._di, 9)}"'
        else:
            dattr = f' class="{cls}"'
        cx = x + w / 2
        if sub2:
            ty, sy, s2y = y + h * 0.30, y + h * 0.55, y + h * 0.76
        else:
            ty, sy = y + h * 0.42, y + h * 0.72
        scls = 's mono' if mono else 's'
        t = [f'<g{dattr}{relattr}{idattr}{hattr}>',
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>',
             f'<text class="t" x="{cx}" y="{ty}">{title}</text>',
             f'<text class="{scls}" x="{cx}" y="{sy}">{sub}</text>']
        if sub2:
            t.append(f'<text class="s2" x="{cx}" y="{s2y}">{sub2}</text>')
        t.append('</g>')
        self.parts.append('\n      '.join(t))

    def edge(self, d, label=None, lx=None, ly=None, anchor='middle', flow=None,
             rel='', dashed=False, marker='arrow', mono=True, delay=None, width=None):
        # flow: None=不动画 / 'blue'|'green'|'violet'|'amber'
        link_marker = f' marker-end="url(#{marker})"' if marker else ''
        style = f' stroke-width="{width}"' if width else ''
        dcls = ' d%d' % min(delay, 9) if (self.stagger and delay) else ''
        self.parts.append(
            f'<path class="link{"--dashed" if dashed else ""}{dcls}" d="{d}"{link_marker}{style}'
            + (f' data-rel="{rel}"' if rel else '') + '/>')
        if flow:
            fcls = 'flow' + (f' flow--{flow}' if flow != 'blue' else '')
            self.parts.append(f'<path class="{fcls}{dcls}" d="{d}"'
                              + (f' data-rel="{rel}"' if rel else '') + '/>')
        if label:
            a = '' if anchor == 'middle' else f' text-anchor="{anchor}"'
            m = ' mono' if mono else ''
            self.parts.append(f'<text class="elabel{m}{dcls}" x="{lx}" y="{ly}"{a}'
                              + (f' data-rel="{rel}"' if rel else '') + f'>{label}</text>')

    def band(self, x, y, w, h, label, ly):
        self.parts.append(f'<rect class="band d0" x="{x}" y="{y}" width="{w}" height="{h}" rx="14"/>')
        self.parts.append(f'<text class="band-label d0" x="{x + 16}" y="{ly}">{label}</text>')

    def raw(self, s):
        self.parts.append(s)


def esc(s):
    return s  # 内容为受控字面量，无需转义


def render_page(dg, meta):
    groups_css = '\n  '.join(HOVER_CSS_RULE.format(g=g) for g in sorted(dg.groups))
    mode_class = 'sg' if dg.stagger else 'fade'
    hover_targets = ''.join(f'\n    <span class="item"><span class="sw {sw}"></span>{tx}</span>'
                            for sw, tx in meta['legend'])
    stats = ''
    for lab, val, cls, sub in meta['stats']:
        vc = f' stat-value {cls}' if cls else ' stat-value'
        stats += (f'<div class="stat-card"><div class="stat-label">{lab}</div>'
                  f'<div class="{vc.strip()}">{val}</div><div class="stat-sub">{sub}</div></div>\n      ')
    cards = ''
    for dot, title, items in meta['cards']:
        lis = ''.join(f'<li>{it}</li>' for it in items)
        cards += (f'<div class="card"><h3><span class="dot {dot}"></span>{title}</h3>'
                  f'<ul>{lis}</ul></div>\n    ')
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="color-scheme" content="light dark" />
<title>{meta['title_plain']}</title>
<style>
{STYLE}
  {groups_css}
</style>
</head>
<body>
<div class="container">
  <header class="header">
    <div class="badge"><span class="pulse"></span>{meta['badge']}</div>
    <h1>{meta['h1']}</h1>
    <p class="subtitle">{meta['subtitle']}</p>
    <div class="stat-row">
      {stats.strip()}
    </div>
  </header>

  <div class="diagram-wrap">
    <div class="diagram-header">
      <h2>{meta['diagram_title']}</h2>
      <span class="tag">Archify 规格 → 定制模板</span>
    </div>
    <svg viewBox="0 0 {meta['vb'][0]} {meta['vb'][1]}" xmlns="http://www.w3.org/2000/svg" class="{mode_class}">
      {MARKERS}
      {chr(10).join('      ' + p for p in dg.parts)}
    </svg>
    <div class="legend">{hover_targets}
    </div>
  </div>

  <div class="cards">
    {cards.strip()}
  </div>

  <footer>{meta['footer']}</footer>
</div>
<script>
{JS}
</script>
</body>
</html>
'''


# ================================================================ 图定义
def block_d():
    d = D(stagger=False)
    d.band(24, 64, 1012, 100, '残差流主干 · x 只经加法（梯度直通）', 56)
    # 主干
    d.node(40, 76, 120, 60, '输入 x', '(B, T, C)', rel='trunk')
    d.edge('M 160 106 L 384 106', 'x', 272, 128, flow='blue', rel='trunk j1')
    d.node(390, 76, 130, 60, '⊕ 残差相加', 'x + sa(ln1(x))', acc='hero', rel='j1', hid='j1', mono=True)
    d.edge('M 520 106 L 684 106', 'x + sa(x)', 602, 128, flow='blue', rel='trunk j2')
    d.node(690, 76, 130, 60, '⊕ 残差相加', 'x + ffwd(ln2(x))', acc='hero', rel='j2', hid='j2', mono=True)
    d.edge('M 820 106 L 904 106', 'x+sa+ffwd', 862, 128, flow='blue', rel='trunk')
    d.node(910, 76, 150, 60, '输出 x', '→ 下一层 Block', rel='trunk')
    # 支路①
    d.edge('M 100 136 V 286 H 126', '读 x', 92, 214, anchor='end', flow='blue', rel='b1 j1', mono=False)
    d.node(130, 256, 90, 60, 'LayerNorm', 'ln1', acc='blue', rel='b1')
    d.edge('M 220 286 L 296 286', 'ln1(x)', 258, 272, flow='blue', rel='b1')
    d.node(300, 256, 180, 60, 'Self-Attention', '① 通信 · token 交换信息', acc='blue', rel='b1')
    d.edge('M 390 256 V 142', 'sa(ln1(x))', 400, 205, anchor='start', flow='blue', rel='b1 j1')
    # 支路②
    d.edge('M 505 136 V 286 H 586', '读 x', 497, 214, anchor='end', flow='green', rel='b2 j2', mono=False)
    d.node(590, 256, 90, 60, 'LayerNorm', 'ln2', acc='green', rel='b2')
    d.edge('M 680 286 L 726 286', 'ln2(x)', 703, 272, flow='green', rel='b2')
    d.node(730, 256, 170, 60, 'FeedForward (MLP)', '② 计算 · 逐 token 思考', acc='green', rel='b2')
    d.edge('M 815 256 V 142', 'ffwd(ln2(x))', 826, 205, anchor='start', flow='green', rel='b2 j2')
    return d, dict(
        name='transformer_block_archify', vb=(1060, 330),
        badge='Part 6 · Transformer / GPT',
        title_plain='Transformer Block · 残差连接 — x = x + f(x)',
        h1='Transformer Block <em>x = x + f(x)</em>',
        subtitle='<strong>残差流主干</strong>从输入直通输出——x 只经过<strong>加法</strong>；两个子层从主干上<strong>读 x</strong>、'
                 '算出增量再<strong>写回</strong>：<code>x = x + self.sa(self.ln1(x))</code>，<code>x = x + self.ffwd(self.ln2(x))</code>（Pre-LN）。'
                 '两个 ⊕ 是<strong>同步点</strong>：两路输入都到达后才输出，支路并行、没有先后。',
        diagram_title='Block 内部结构 · 残差主干 + 两条支路',
        stats=[('核心公式', 'x + f(x)', 'brand', '主干只做加法 · 子层只添增量'),
               ('子层分工', '① 通信 → ② 计算', '', 'Self-Attention · FeedForward'),
               ('⊕ 同步点', '两路到齐才输出', 'green', 'x 与 f(x) 并行汇合，无先后')],
        legend=[('hero', '⊕ 残差汇合点（同步：两路到齐才输出）'), ('blue', '① 通信支路（Attention）'),
                ('green', '② 计算支路（FFN）'), ('band-s', '残差流主干（x 只经加法）'),
                ('dash', '常速流动 = 数据流（不表示先后）')],
        cards=[('cyan', '为什么梯度能直通',
                ['加法把上游梯度<code>原样复制</code>给两路，不是对半分',
                 '恒等支路梯度恒为 1：主干梯度不衰减（Part 4）',
                 '初始化时子层很小 → 深层也能训练']),
               ('emerald', 'Pre-LN 顺序别写反',
                ['先归一化再进子层：<code>x + sa(ln1(x))</code>',
                 '写成 <code>ln(x + sa(x))</code> 是 post-norm',
                 'LN 在支路上、不在主干上——主干只做加法']),
               ('violet', '堆叠成深层网络',
                ['Block 重复 <code>n_layer</code> 次，输出直进下一层',
                 '现代 LLM 把 ReLU MLP 换成 SwiGLU（Part 7）',
                 '先通信、再计算是每个 Block 的固定节奏'])],
        footer='语义源：archify 规格 <code>p06_transformer_block</code>（showcase 校验）· 汇合图：禁入场排序，全通路同速流动 · 悬停 ⊕ 看两路输入')


def mlp_d():
    d = D(stagger=True)
    d.band(150, 62, 706, 116, 'MLP 网络结构（Bengio 2003 两层）', 54)
    xs = [30, 168, 306, 444, 582, 720, 858]
    defs = [('输入索引', '[5, 13, 13]', None, 'ctx'),
            ('C · Embedding', '(27×2) 查表', 'blue', 'p'),
            ('拼接 concat', '(N,3,2)→(N,6)', None, 'ctx'),
            ('W1 + b1', 'Linear 6→100', 'blue', 'p'),
            ('tanh', '激活函数', None, 'ctx'),
            ('W2 + b2', 'Linear 100→27', 'blue', 'p'),
            ('logits→softmax', '下一字符概率', 'green', 'ctx')]
    ids = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
    for i, (x, (t, s, a, r)) in enumerate(zip(xs, defs)):
        d.node(x, 90, 118, 60, t, s, acc=a, rel=r, hid=ids[i] if r == 'p' else None,
               hl='p' if r == 'p' else None)
    for i in range(6):
        d.edge(f'M {xs[i]+118} 120 L {xs[i+1]-6} 120', flow='blue', delay=i + 1, rel='ctx')
    return d, dict(
        name='mlp_archify', vb=(1006, 200),
        badge='Part 2 · MLP',
        title_plain='MLP 网络结构（Bengio 2003）',
        h1='两层 MLP <em>查表 → 拼接 → 两次线性 + tanh</em>',
        subtitle='课程第一张完整网络图：3 个字符索引经 <code>C</code> 查表得到向量、拼接成 6 维，'
                 '经 <code>W1(6→100)</code> + <code>tanh</code> 再 <code>W2(100→27)</code>，'
                 'softmax 给出下一个字符的概率。悬停蓝色节点看<strong>要训练的参数</strong>在哪。',
        diagram_title='前向形状链',
        stats=[('可训练参数', 'C · W1 · W2', 'brand', '创建时必须 requires_grad=True'),
               ('隐藏层', '100 维', '', 'tanh 激活（Bengio 2003 设置）'),
               ('输出', '27 类概率', 'green', 'softmax(logits) → 下一字符')],
        legend=[('blue', '可训练参数（悬停高亮）'), ('', '运算层（拼接 / 激活）'),
                ('green', '输出概率'), ('band-s', '网络结构'), ('dash', '数据流')],
        cards=[('cyan', '参数在哪',
                ['C、W1、W2 是要训练的参数',
                 'C 创建时必须 <code>requires_grad=True</code>',
                 '漏掉会在 <code>loss.backward()</code> 直接报错']),
               ('emerald', '形状链',
                ['<code>(N,3)</code> → 查表 <code>(N,3,2)</code> → 拼接 <code>(N,6)</code>',
                 '<code>Linear 6→100</code> → tanh → <code>Linear 100→27</code>',
                 '输出 logits <code>(N,27)</code> 经 softmax 成概率']),
               ('amber', 'C[X] 高级索引',
                ['<code>emb = C[X]</code>：按索引行取出向量',
                 '这是 Part 2 最核心的操作之一',
                 '查表对梯度可导——梯度回传到 C 的对应行'])],
        footer='语义源：archify 规格 <code>p02_mlp</code> · 顺序管线：允许入场分级 + 主路常速流动')


def gpt2_d():
    d = D(stagger=True)
    xs = [28, 196, 364, 532, 700, 868]
    defs = [('token ids', '(B, T)', None, 'ctx'),
            ('token+pos embed', '相加 · 可学习位置', 'blue', 'ctx'),
            ('Block × N', 'Pre-LN · 残差主干', 'hero', 'blk'),
            ('LayerNorm final', '最后一层 LN', None, 'ctx'),
            ('lm_head', 'n_embed → vocab', 'blue', 'ctx'),
            ('logits', '(B, T, vocab)', 'green', 'ctx')]
    for x, (t, s, a, r) in zip(xs, defs):
        d.node(x, 80, 152, 62, t, s, acc=a, rel=r, hid='blk' if r == 'blk' else None)
    for i in range(5):
        rel = 'blk' if i in (1, 2) else 'ctx'
        d.edge(f'M {xs[i]+152} 111 L {xs[i+1]-6} 111', flow='blue', delay=i + 1, rel=rel)
    return d, dict(
        name='gpt2_archify', vb=(1048, 175),
        badge='Part 8 · 后训练全流程',
        title_plain='GPT-2 模型架构（Pre-LN）',
        h1='GPT-2 前向结构 <em>ids → embed → Block×N → lm_head</em>',
        subtitle='后训练所有章节的参考架构：token 与位置 embedding <strong>相加</strong>（不是拼接），'
                 'Pre-LN Block 堆叠 N 层，final LayerNorm 后接 <code>lm_head</code> 预测下一个 token。'
                 '<code>forward_hidden()</code> 返回 final LN 之后、lm_head 之前的 hidden state——'
                 '这是 SFT / RM / 蒸馏共同的 hook 点。',
        diagram_title='模型级前向主路',
        stats=[('位置编码', '可学习 · 相加', '', '参数表，不用 RoPE'),
               ('Block 结构', 'Pre-LN', 'brand', '先归一化再进子层（内部见 Part 6）'),
               ('关键 hook', 'forward_hidden()', 'green', 'SFT / RM / 蒸馏的接管点')],
        legend=[('hero', 'Block × N（Pre-LN · 残差主干）'), ('blue', '参数层（Embedding / lm_head）'),
                ('green', '输出 logits'), ('dash', '数据流（顺序管线，入场分级）')],
        cards=[('cyan', '设计选择',
                ['位置编码：可学习参数表、相加、不用 RoPE',
                 'MHA 标准（不用 GQA）；MLP 4x + ReLU（不用 SwiGLU）',
                 'Pre-LN：先 LayerNorm 再进子层，训练更稳定']),
               ('rose', '后训练关键 hook',
                ['<code>forward_hidden()</code>：final LN 后、lm_head 前的 hidden state',
                 'SFT 换 lm_head、RM 加 reward_head、蒸馏读 hidden',
                 'Block 内部残差结构见 Part 6 03 章交互图']),
               ('violet', '为什么要看模型级',
                ['后训练的所有角色都共享这一个 backbone',
                 '知道 hook 点在哪，SFT/DPO/PPO 的差异才看得懂',
                 '形状链：(B,T) → (B,T,C) → (B,T,vocab)'])],
        footer='语义源：archify 规格 <code>p08_gpt2</code> · 顺序管线：允许入场分级 + 主路常速流动')


def attention_d():
    d = D(stagger=False)
    d.node(30, 40, 150, 64, 'Q：cat ★', '提问 · 在找什么', acc='blue', rel='qk')
    d.node(30, 250, 150, 64, 'K：token 标签', '⟨bos⟩/the/cat/sat', acc='blue', rel='qk')
    d.edge('M 180 72 H 212 V 163 H 256', flow='blue', rel='qk sc')
    d.edge('M 180 282 H 228 V 191 H 256', flow='blue', rel='qk sc')
    d.node(260, 145, 160, 64, '打分 q·Kᵀ', '逐 token 点积', rel='sc', hid='sc')
    d.edge('M 420 177 H 486', '4 个分数', 453, 163, flow='blue', rel='sc w')
    d.node(490, 145, 170, 64, 'softmax 权重', '0.05 / 0.10 / 0.60 / 0.25', rel='w')
    d.node(490, 20, 150, 64, 'V：内容', 'token 携带的内容', acc='violet', rel='v')
    d.edge('M 660 163 H 716', '权重 w', 688, 149, flow='blue', rel='w j')
    d.edge('M 565 84 H 770 V 141', '内容 v', 660, 74, flow='violet', rel='v j')
    d.node(720, 145, 150, 64, '⊕ 加权求和', '按 w 混合各 V', acc='hero', rel='j', hid='j')
    d.edge('M 870 177 H 916', 'Σwᵢvᵢ', 893, 163, flow='blue', rel='j out')
    d.node(920, 145, 110, 64, '新表示', '带全文语境', acc='green', rel='out')
    return d, dict(
        name='attention_qkv_archify', vb=(1040, 330),
        badge='Part 6 · Transformer / GPT',
        title_plain='Self-Attention — QK 打分 → softmax → 加权 V',
        h1='Self-Attention <em>Q·K → softmax → Σwᵢvᵢ</em>',
        subtitle='一次前向的三步：<strong>Q 与 K 两路共同汇入</strong>打分（q·Kᵀ 逐 token 点积），softmax 归一成权重，'
                 '再与 <strong>V 两路共同汇入</strong>加权求和。K 不流向 V——权重只决定"读多少"。'
                 '悬停 ⊕ 或打分节点，看它等待的两路输入。',
        diagram_title='两处汇合的诚实 DAG',
        stats=[('汇合点 ×2', '打分 · 求和', 'brand', 'Q+K 打分；权重+V 求和'),
               ('q=cat 权重', '0.60 最高', '', '行和 1.00（softmax）'),
               ('输出', 'Σwᵢvᵢ', 'green', '带全文语境的新表示')],
        legend=[('blue', 'Q / K（查询与标签）'), ('violet', 'V（内容）'), ('hero', '⊕ 汇合点（两路到齐）'),
                ('green', '输出'), ('dash', '常速流动 = 数据流（不表示先后）')],
        cards=[('cyan', '两处汇合',
                ['打分要 Q、K 两路都到：<code>q·Kᵀ</code> 逐 token 点积',
                 '求和要权重、V 两路都到：新表示 = <code>Σ wᵢ·vᵢ</code>',
                 'K 不流向 V——权重只决定"读多少"']),
               ('emerald', '权重从哪来',
                ['q = cat 时 w = 0.05 / 0.10 / 0.60 / 0.25',
                 '每行独立 softmax，行和恒为 1.00',
                 '自己对自己的权重最高（0.60）']),
               ('violet', '扩到全序列',
                ['8 个 token 各当一次提问者 → 8×8 权重矩阵',
                 '因果 mask：只能看过去，不能看未来',
                 '逐帧动画见本章演示组件'])],
        footer='语义源：archify 规格 <code>p06_attention_qkv</code> · 汇合图：禁入场排序，全通路同速流动')


def verl_d():
    d = D(stagger=False)
    d.node(40, 110, 180, 70, 'Actor（训练）', 'FSDP2 · 更新参数', acc='blue', rel='loop')
    d.edge('M 220 145 H 296', '训练后权重', 258, 131, flow='blue', rel='loop sync')
    d.node(300, 110, 190, 70, '权重回同步', '3D-HybridEngine', acc='hero', rel='sync', hid='sync')
    d.edge('M 490 145 H 566', '刷新推理引擎', 528, 131, flow='blue', rel='sync loop')
    d.node(570, 110, 250, 70, 'Rollout（生成）', 'vLLM · 采样 G 个回答', acc='green', rel='loop')
    d.edge('M 695 110 V 50 H 130 V 106', '采样 G 个回答 → 训练信号', 412, 42, flow='green',
           rel='loop sync', marker='arrowGreen')
    d.node(40, 260, 180, 70, 'Ref（参考）', '❄️ 冻结 SFT · 算 KL', acc='violet', rel='ref')
    d.edge('M 130 260 V 184', '参考 logprobs（KL）', 30, 225, anchor='end', rel='ref',
           dashed=True, marker='arrowViolet')
    return d, dict(
        name='verl_roles_archify', vb=(850, 360),
        badge='Part 11 · 对齐实战 verl',
        title_plain='verl 三角色架构（GRPO 视角）',
        h1='verl 三角色 <em>权重同步 + 训练闭环</em>',
        subtitle='Actor 训练后的权重经 <strong>3D-HybridEngine</strong> 回同步刷新 vLLM 推理引擎，'
                 'Rollout 采样出的 G 个回答回流成训练信号——三角色连成一个<strong>闭环</strong>；'
                 '冻结的 Ref 只出参考 logprobs 算 KL。悬停中间节点看整条环路。',
        diagram_title='三角色与权重/回答的双向流动',
        stats=[('训练闭环', 'Actor ⇄ Rollout', 'brand', '权重下行 · 回答回流'),
               ('同步引擎', '3D-HybridEngine', '', '训练态 ↔ 推理态免重载'),
               ('GRPO 视角', '3 角色', 'violet', 'PPO 还需第 4 角色 critic')],
        legend=[('blue', 'Actor（训练侧）'), ('hero', '权重回同步通道'), ('green', 'Rollout（推理侧）'),
                ('violet', 'Ref（冻结参考）'), ('dash', '常速流动 = 闭环运转（不表示先后）')],
        cards=[('cyan', '角色对照（手写版）',
                ['Actor = 训练循环：更新模型参数',
                 'Rollout = <code>for step: 采样 G 个回答</code>',
                 'Ref = ref 模型：计算 KL 惩罚']),
               ('rose', 'PPO 视角提醒',
                ['这张图是 GRPO 视角的三角色',
                 'PPO（<code>adv_estimator=gae</code>）还需第 4 个角色 critic',
                 'GRPO 砍掉 critic 正是它省显存的原因']),
               ('emerald', '为什么要有 weight sync',
                ['训练态（FSDP2 分片）与推理态（vLLM）格式不同',
                 '每步把新权重刷进推理引擎再采样',
                 'HybridEngine 让这一步不用重载整个模型'])],
        footer='语义源：archify 规格 <code>p11_verl_roles</code> · 闭环图：禁入场排序，环路常速流动')


def lora_d():
    d = D(stagger=False)
    d.node(30, 60, 130, 64, 'x', '(batch, seq, 96)', rel='io')
    d.edge('M 160 92 H 226', '直通', 193, 78, flow='violet', rel='io base', mono=False)
    d.node(230, 30, 190, 64, 'base_layer ❄️', 'Linear(96→288) · W(288,96)', acc='violet', rel='base')
    d.edge('M 95 124 V 222 H 226', '旁路 🔥', 148, 175, anchor='end', flow='amber', rel='io la', mono=False)
    d.node(230, 190, 190, 64, 'lora_A 🔥', '(4,96) · x@A.T → (…,4)', acc='amber', rel='la')
    d.edge('M 420 222 H 466', '@ B.T', 443, 208, flow='amber', rel='la')
    d.node(470, 190, 190, 64, 'lora_B 🔥', '(288,4) · 零初始化', acc='amber', rel='la')
    d.edge('M 420 62 H 700 V 126 H 756', 'base_out', 560, 48, flow='violet', rel='base j', marker=None)
    d.edge('M 660 222 H 720 V 158 H 756', '× α/r = 2.0 相加', 742, 244, flow='amber', rel='la j', mono=False)
    d.node(760, 110, 200, 64, '⊕ output', 'base_out + lora_out', acc='hero', rel='j', hid='j')
    return d, dict(
        name='lora_archify', vb=(980, 280),
        badge='Part 12 · 微调实战',
        title_plain='LoRA 双分支前向（r=4，α=8）',
        h1='LoRA 前向 <em>ΔW = α/r · BA</em>',
        subtitle='冻结的 <code>base_layer</code> 直通主路；旁路上 <code>lora_A → lora_B</code> 算出低秩增量，'
                 '两路在 ⊕ 处<strong>相加</strong>（B 零初始化 → 训练起点 ΔW=0，起点无损）。'
                 '悬停 ⊕ 看它等待的两路输入。',
        diagram_title='冻结主路 + 可训练旁路',
        stats=[('本层可训练', '1,536', 'amber', '4×96 + 288×4（原始 27,648，压缩 18×）'),
               ('缩放因子', 'α/r = 2.0', '', '学习强度与 r 解耦'),
               ('融合方式', '⊕ 相加', 'brand', 'base_out + lora_out（两路到齐）')],
        legend=[('violet', '冻结权重 ❄️（不更新）'), ('amber', '可训练低秩支路 🔥'),
                ('hero', '⊕ 相加（两路到齐）'), ('dash', '常速流动 = 数据流（不表示先后）')],
        cards=[('cyan', '参数账本',
                ['本层可训练 <code>4×96 + 288×4 = 1,536</code>',
                 '本层原始 <code>288×96 = 27,648</code>（压缩 18 倍）',
                 '全模型可训练 1,536 × 4 层 = 6,144']),
               ('amber', '初始化关键',
                ['<code>lora_B</code> 零初始化 → 起点 ΔW=0，起点无损',
                 'scaling α/r 把"学习强度"与 r 解耦',
                 'A 用高斯初始化：~N(0,1)/√r']),
               ('violet', '注入位置',
                ['原论文只注 Wq、Wv；QLoRA 实验表明 all 更好',
                 '<code>lora_target: all</code> = 所有 Linear 都注入',
                 '合并时 <code>W += (α/r)·BA</code>，部署零开销'])],
        footer='语义源：archify 规格 <code>p12_lora</code> · 汇合图：禁入场排序，全通路同速流动')


def minhash_d():
    d = D(stagger=True)
    xs = [50, 310, 570, 810]
    defs = [('① shingling', '文档 → 3-gram 集合', 'Jaccard 比较单位', 'blue', 'ctx'),
            ('② MinHash', '64 哈希取最小', '→ 64 维签名', 'blue', 'ctx'),
            ('③ 分带 LSH', '16 段 × 4 维', '任一段全同 → 候选对', 'amber', 'lsh'),
            ('④ Jaccard 验证', '只对候选对精算', '≥ 0.5 判重', 'green', 'ctx')]
    for x, (t, s, s2, a, r) in zip(xs, defs):
        d.node(x, 90, 200, 72, t, s, acc=a, rel=r, sub2=s2, hid='lsh' if r == 'lsh' else None)
    d.edge('M 250 126 L 304 126', '词组集合', 277, 112, flow='blue', delay=1, rel='ctx')
    d.edge('M 510 126 L 564 126', '64 维签名', 537, 112, flow='blue', delay=2, rel='lsh')
    d.edge('M 770 126 L 804 126', '候选对', 787, 112, flow='blue', delay=3, rel='lsh')
    return d, dict(
        name='minhash_archify', vb=(1060, 200),
        badge='Part 13 · 数据工程',
        title_plain='MinHash + 分带 LSH 去重四阶段管线',
        h1='去重管线 <em>签名 → 候选 → 验证</em>',
        subtitle='四阶段管线：①② 把文档压成 <strong>64 维签名</strong>（某维最小哈希相等的概率 = Jaccard），'
                 '③ 分带 LSH 用 <strong>16 段 × 4 维</strong>快速筛出候选对，④ 只对候选对精算 Jaccard（≥ 0.5 判重）——'
                 '避免全量 O(n²) 两两比较。',
        diagram_title='四阶段去重管线',
        stats=[('签名维度', 'k = 64', 'blue', '= b × r = 16 bands × 4 rows'),
               ('候选概率', '1-(1-Jʳ)ᵇ', 'amber', 'J 越高必然命中（S 曲线）'),
               ('判重阈值', 'J ≥ 0.5', 'green', '只对候选对精算')],
        legend=[('blue', '签名阶段（①②）'), ('amber', '候选分桶（③ LSH）'),
                ('green', '验证门（④ 阈值判定）'), ('dash', '数据流（顺序管线，入场分级）')],
        cards=[('cyan', '两步加速',
                ['MinHash 把 Jaccard 变成 64 维签名的比较',
                 '分带 LSH 只对候选对精算，避免 O(n²) 两两比较',
                 'J 越高越必然命中候选']),
               ('amber', '本课取值',
                ['<code>k = 64 = b × r = 16 × 4</code>（assert 强制）',
                 '哈希族 <code>(a·h+b) mod P</code>，seed=7 固定',
                 '<code>zlib.crc32</code> 跨进程稳定，数字可复现']),
               ('emerald', '等效阈值',
                 ['拐点 <code>J* = (1/b)^(1/r)</code>，本课配置 ≈ 0.50',
                 '恰好等于脚本的判重阈值 0.5',
                 'S 曲线交互演示见上一节'])],
        footer='语义源：archify 规格 <code>p13_minhash</code> · 顺序管线：允许入场分级 + 主路常速流动')


def rag_d():
    d = D(stagger=False)
    d.node(30, 150, 190, 64, '离线索引', '语料 8 篇 → 238 chunks', acc='violet', rel='idx')
    d.edge('M 220 172 H 260 V 104 H 296', '向量', 240, 130, flow='violet', rel='idx dr', mono=False)
    d.edge('M 220 172 H 260 V 240 H 296', '词表', 240, 225, flow='violet', rel='idx br', mono=False)
    d.node(300, 40, 160, 64, 'dense 检索', 'q_vec@chunk_mat · top-10', acc='blue', rel='dr')
    d.node(300, 260, 160, 64, 'BM25 检索', '打分 238 chunk · top-10', acc='blue', rel='br')
    d.edge('M 460 72 H 540 V 146', 'top-10', 500, 58, flow='blue', rel='dr j')
    d.edge('M 460 292 H 540 V 198', 'top-10', 500, 305, flow='blue', rel='br j')
    d.node(540, 150, 120, 64, 'RRF 融合', '只融名次 k=60', acc='hero', rel='j', hid='j')
    d.edge('M 660 182 H 696', 'hybrid', 678, 168, flow='blue', rel='j rr')
    d.node(700, 150, 150, 64, 'bge-reranker', '成对打分 top-5→3', acc='blue', rel='rr')
    d.edge('M 850 182 H 886', 'top-3', 868, 168, flow='blue', rel='rr out')
    d.node(890, 150, 130, 64, 'Qwen2.5 生成', '[1][2][3]→answer', acc='green', rel='out')
    return d, dict(
        name='hybrid_rag_archify', vb=(1040, 350),
        badge='Part 18 · RAG 全链路',
        title_plain='Hybrid RAG 数据流',
        h1='Hybrid RAG <em>双路召回 → RRF → 重排 → 生成</em>',
        subtitle='<strong>离线</strong>把语料切块、向量化建索引；<strong>在线</strong>查询进来后 dense（语义）与 '
                 'BM25（关键词）两路各自召回 top-10，RRF 只用<strong>名次</strong>融合（不比分数口径），'
                 '重排取 top-3 作为生成证据。悬停 ⊕ 融合点看两路召回。',
        diagram_title='离线索引 × 在线查询双时间线',
        stats=[('双路召回', 'dense ∥ BM25', 'blue', '语义与关键词互补'),
               ('融合规则', 'RRF · k=60', 'brand', '只融名次，不比分数口径'),
               ('生成证据', 'top-3', 'green', '重排后进 Qwen2.5-0.5B')],
        legend=[('violet', '离线索引（向量库 / 词表）'), ('blue', '检索 / 重排组件'),
                ('hero', 'RRF 融合（两路到齐）'), ('green', '生成'), ('dash', '常速流动 = 数据流（不表示先后）')],
        cards=[('rose', '两条时间线',
                ['离线：语料 → chunk → 向量矩阵（不随查询变化）',
                 '在线：query 进来 → 双路召回 → 融合重排 → 生成',
                 '换 Embedding 模型只需重建索引这一条线']),
               ('cyan', '为什么混合',
                ['dense 抓语义，BM25 抓关键词，两路互补',
                 'RRF 只用名次融合（k=60），避免分数口径问题',
                 '重排对 (query, chunk) 成对精算']),
               ('amber', '降级路径',
                ['无 GPU / 无模型时 <code>hash_embed → (238, 256)</code>',
                 '管线结构不变，只换向量化实现',
                 '本课语料 8 篇 md，约 86k 字符'])],
        footer='语义源：archify 规格 <code>p18_hybrid_rag</code> · 汇合图：禁入场排序，全通路同速流动')


def agent_d():
    d = D(stagger=False)
    d.node(30, 80, 110, 60, 'user query', '第 N 轮', rel='main')
    d.edge('M 140 110 H 176', flow='blue', rel='main')
    d.node(180, 80, 160, 60, '① chat_template', '注入工具 schema', acc='blue', rel='main')
    d.edge('M 340 110 H 376', '每轮', 358, 96, flow='blue', rel='main cyc')
    d.node(380, 80, 150, 60, '② generate', '贪心 / 采样', acc='hero', rel='cyc', hid='cyc')
    d.edge('M 530 110 H 566', flow='blue', rel='cyc dec')
    d.node(570, 80, 150, 60, '③ parse_calls', '有调用？', acc='blue', rel='dec')
    d.edge('M 720 110 H 766', '无调用 ✅', 743, 96, flow='green', rel='dec out', marker='arrowGreen', mono=False)
    d.node(770, 80, 140, 60, '最终答案 ✅', '循环结束', acc='green', rel='out')
    d.edge('M 645 140 V 180 H 580 V 246', '有调用', 610, 170, flow='blue', rel='dec cyc', mono=False)
    d.node(500, 250, 160, 60, '④ execute_tool', '白名单 bash', acc='amber', rel='cyc')
    d.edge('M 660 280 H 696', '结果贴回', 678, 236, flow='amber', rel='cyc', mono=False)
    d.node(700, 250, 170, 60, '⑤ messages +=', 'tool_calls + result', acc='violet', rel='cyc')
    d.edge('M 785 310 V 336 H 455 V 146', '下一轮（循环）', 620, 328, flow='blue', rel='cyc')
    return d, dict(
        name='agent_loop_archify', vb=(930, 360),
        badge='Part 19 · Agent 与 FC',
        title_plain='Agent Loop — 说话还是填工具调用单',
        h1='Agent Loop <em>while 循环就是全部骨架</em>',
        subtitle='五个部件连成一个循环：parse 出"<strong>有调用</strong>"就执行工具、把结果贴回 messages、'
                 '回到 ② 进入下一轮；"<strong>无调用</strong>"才输出最终答案。② 每轮都会重新执行。'
                 '悬停 ② 看整条循环；安全边界（白名单/超时）都在执行侧。',
        diagram_title='循环骨架与两个出口',
        stats=[('循环体', '②③④⑤', 'brand', '生成 → 解析 → 执行 → 回填'),
               ('出口', '2 个', 'green', '无调用=答案；或触发终止条件'),
               ('终止条件', '3 个', 'amber', '无调用 / max_turns / 复读检测')],
        legend=[('blue', 'Agent 运行时（①②③）'), ('amber', '工具执行（④）'),
                ('violet', '对话状态（⑤ messages）'), ('green', '出口（答案）'),
                ('dash', '常速流动 = 循环运转（不表示先后）')],
        cards=[('cyan', '循环本质',
                ['把工具结果拼回上下文，再让模型决定下一步',
                 '全部智能在"决定下一步"里，其余是电话线路',
                 '工具 schema 塞进 prompt——模型"说话"或"填单"']),
               ('rose', '三个终止条件',
                ['① 无 tool_calls（答案 / 放弃）',
                 '② <code>max_turns</code> 上限',
                 '③ 同调用复读 3 次（循环检测）']),
               ('amber', '安全边界',
                ['模型只能"下单"（name + arguments JSON）',
                 '超时 / 白名单 / 路径检查都在执行侧',
                 '模型输出是不可信输入——边界必须放执行侧'])],
        footer='语义源：archify 规格 <code>p19_agent_loop</code> · 闭环图：禁入场排序，环路常速流动')


DIAGRAMS = [block_d, mlp_d, gpt2_d, attention_d, verl_d, lora_d, minhash_d, rag_d, agent_d]


def main():
    os.makedirs(WIDGETS, exist_ok=True)
    for fn in DIAGRAMS:
        d, meta = fn()
        html = render_page(d, meta)
        out = os.path.join(WIDGETS, meta['name'] + '.html')
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'{meta["name"]}.html  {len(html)//1024}KB  stagger={d.stagger}  groups={sorted(d.groups)}')


if __name__ == '__main__':
    main()
