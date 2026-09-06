#!/usr/bin/env python3
"""渲染器补丁：details 块正确处理 + 字号调节按钮 + 阅读排版微调"""
p = '/home/admin02/Code/WorkSpace/makemore-tutorial-review/tools/build_site_lite.py'
s = open(p, encoding='utf-8').read()
n = 0

# 1) details 块：开/闭标签独立成行时按标签处理；summary 行内文本走 inline()
old = """        st = ln.strip()
        if st.startswith(('<details>', '</details>', '<summary', '</summary', '<br', '<div')):
            out.append(st)
            i += 1
            continue"""
new = """        st = ln.strip()
        if st in ('<details>', '<details>'):
            out.append('<details>')
            i += 1
            continue
        if st == '</details>':
            out.append('</details>')
            i += 1
            continue
        m_sum = re.match(r'<summary>(.*)</summary>\\s*$', st)
        if m_sum:
            out.append(f'<summary>{inline(m_sum.group(1))}</summary>')
            i += 1
            continue
        if st.startswith('<br') or st.startswith('<div'):
            out.append(st)
            i += 1
            continue"""
if old in s:
    s = s.replace(old, new); n += 1

# 2) 段落收集在任何 '<' 开头的原始 HTML 行前停止（防吞 </details>）
old2 = "not re.match(r'^(#{1,6}\\s|```|\\s*>\\s?|(\\s*[-*+]\\s|\\s*\\d+\\.\\s))', lines[i]) and '|' not in lines[i]:"
new2 = "not re.match(r'^(#{1,6}\\s|```|\\s*>\\s?|(\\s*[-*+]\\s|\\s*\\d+\\.\\s))', lines[i]) and '|' not in lines[i] and not lines[i].lstrip().startswith('<'):"
cnt = s.count(old2)
s = s.replace(old2, new2); n += cnt

# 3) 顶栏字号调节按钮
old3 = """  <button id="themeBtn" title="切换亮/暗主题">🌙 夜间</button>
</header>"""
new3 = """  <div class="tools">
    <button id="fontMinus" title="减小字号">A−</button>
    <button id="fontReset" title="标准字号">A</button>
    <button id="fontPlus" title="增大字号">A+</button>
    <button id="themeBtn" title="切换亮/暗主题">🌙 夜间</button>
  </div>
</header>"""
if old3 in s:
    s = s.replace(old3, new3); n += 1

# 4) 字号调节 JS（三档 15/16.5/18px，localStorage 记忆）
old4 = """(function(){var b=document.getElementById('themeBtn');"""
new4 = """(function(){var SIZES=[15,16.5,18],KEY='mm-fs',idx=1;
try{var v=parseInt(localStorage.getItem(KEY));if(v>=0&&v<SIZES.length)idx=v;}catch(e){}
function apply(){document.documentElement.style.fontSize=SIZES[idx]+'px';
try{localStorage.setItem(KEY,idx);}catch(e){}}
document.getElementById('fontMinus').onclick=function(){idx=Math.max(0,idx-1);apply();};
document.getElementById('fontReset').onclick=function(){idx=1;apply();};
document.getElementById('fontPlus').onclick=function(){idx=Math.min(SIZES.length-1,idx+1);apply();};
apply();})();
(function(){var b=document.getElementById('themeBtn');"""
if old4 in s:
    s = s.replace(old4, new4, 1); n += 1

# 5) CSS：按钮组 + details 紧凑 + 基准字号
old5 = """#themeBtn{background:var(--code-bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}"""
new5 = """.tools{display:flex;gap:6px}
.tools button{background:var(--code-bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:13px;min-width:40px}
.tools button:hover{border-color:var(--accent);color:var(--accent)}
#themeBtn{min-width:88px}
details{margin:10px 0}
details p{margin:8px 0}"""
if old5 in s:
    s = s.replace(old5, new5); n += 1

open(p, 'w', encoding='utf-8').write(s)
print(f'applied {n} patches')
