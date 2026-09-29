# -*- coding: utf-8 -*-
"""一次性脚本：生成站点自托管字体子集（assets/fonts）。

用途：把 Noto Serif SC 700（衬线标题）与 IBM Plex Mono 400（等宽）按站点实际用字子集化，
产出 2 个 woff2 + OFL.txt，由 tools/build_site_lite.py 的 build() 复制到 _assets/fonts/。

用法：
    & 'G:\\Workspace\\python_env\\.venv\\Scripts\\python.exe' tools\\build_fonts.py

一次性依赖：brotli（fontTools 读写 woff2 必需，仅生成端用，站点运行时零依赖）。
新增章节若用到生僻字，重跑本脚本即可补齐。
"""
import io
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_HTML = os.path.join(REPO_ROOT, 'site_build', 'site_html')
CACHE_DIR = os.path.join(REPO_ROOT, 'tools', '.fontcache')
OUT_DIR = os.path.join(REPO_ROOT, 'assets', 'fonts')
CDN = 'https://cdn.jsdelivr.net/npm/@fontsource'

# 站点正文/图标常用符号区段（按码位补齐；字体里没有的字符会被自动忽略）
RANGES = [
    (0x0020, 0x007E),  # ASCII 可打印
    (0x00A0, 0x00FF),  # Latin-1 补充
    (0x2000, 0x206F),  # 常用标点（含破折号、省略号、引号）
    (0x2070, 0x209F),  # 上下标
    (0x20A0, 0x20BF),  # 货币符号
    (0x2100, 0x214F),  # 字母式符号（№ ™ ℉ 等）
    (0x2150, 0x218F),  # 数字形式（Ⅰ ⅰ ⅓ 等）
    (0x2190, 0x21FF),  # 箭头
    (0x2200, 0x22FF),  # 数学运算符（≤ ≥ ≈ ≠ ∞ 等）
    (0x2460, 0x24FF),  # 带圈字符（① ⒈ 等）
    (0x2500, 0x257F),  # 制表符（目录树）
    (0x25A0, 0x25FF),  # 几何图形（■ ● ▲ 等）
    (0x2600, 0x26FF),  # 杂项符号（☰ ★ 等）
    (0x3000, 0x303F),  # CJK 标点
    (0xFF00, 0xFFEF),  # 全角字符
]

# 只做拉丁子集的等宽字体：ASCII + 常用标点 + 箭头
MONO_RANGES = [(0x0020, 0x007E), (0x00A0, 0x00FF), (0x2000, 0x206F), (0x2190, 0x21FF)]

FONTS = [
    {
        'src': 'noto-serif-sc-chinese-simplified-700-normal.woff2',
        'pkg': 'noto-serif-sc@5',
        'out': 'noto-serif-sc-700.woff2',
        'ranges': RANGES,
        'budget': 1.2 * 1024 * 1024,
    },
    {
        'src': 'ibm-plex-mono-latin-400-normal.woff2',
        'pkg': 'ibm-plex-mono@5',
        'out': 'ibm-plex-mono-400.woff2',
        'ranges': MONO_RANGES,
        'budget': 80 * 1024,
    },
]


def fetch(url, dest):
    """下载到缓存目录（已存在则跳过）。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print('  缓存命中 %s' % os.path.basename(dest))
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print('  下载 %s' % url)
    req = urllib.request.Request(url, headers={'User-Agent': 'makemore-site-fontbuilder'})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as fh:
        fh.write(resp.read())
    print('    -> %s (%.1f KB)' % (os.path.basename(dest), os.path.getsize(dest) / 1024))
    return dest


def collect_charset():
    """收集站点已构建页面里出现的全部字符。"""
    if not os.path.isdir(SITE_HTML):
        print('[!] 未找到 %s，请先执行：python tools/build_site_lite.py --force' % SITE_HTML)
        sys.exit(1)
    chars = set()
    n = 0
    for root, _dirs, files in os.walk(SITE_HTML):
        for name in files:
            if not name.endswith('.html'):
                continue
            n += 1
            try:
                with io.open(os.path.join(root, name), 'r', encoding='utf-8', errors='ignore') as fh:
                    chars.update(fh.read())
            except OSError:
                pass
    if not n:
        print('[!] %s 下没有 HTML，请先执行构建' % SITE_HTML)
        sys.exit(1)
    print('  扫描 %d 个页面，原始字符集 %d 个' % (n, len(chars)))
    return chars


def unicodes_for(chars, ranges):
    codes = {ord(c) for c in chars if not c.isspace() or c == ' '}
    for lo, hi in ranges:
        codes.update(range(lo, hi + 1))
    codes.discard(0x0A)
    codes.discard(0x0D)
    codes.discard(0x09)
    return sorted(c for c in codes if c >= 0x20)


def subset_font(spec, src_path, chars, out_path):
    from fontTools import subset
    options = subset.Options()
    options.flavor = 'woff2'
    options.hinting = False
    options.desubroutinize = False
    options.layout_features = ['*']
    options.prune_unicode_ranges = True
    options.name_IDs = ['*']
    options.notdef_outline = True

    font = subset.load_font(src_path, options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=unicodes_for(chars, spec['ranges']))
    subsetter.subset(font)
    subset.save_font(font, out_path, options)
    font.close()
    return os.path.getsize(out_path)


def write_ofl(noto_license, mono_license):
    parts = [
        'fonts in this directory are subsetted copies of open-source fonts,',
        'redistributed under the SIL Open Font License 1.1 (OFL-1.1).',
        '',
        '=' * 72,
        'Noto Serif SC (Regular/Bold) — subset',
        'Copyright 2015-2024 Google LLC. Noto is a trademark of Google LLC.',
        '=' * 72,
        noto_license.strip(),
        '',
        '=' * 72,
        'IBM Plex Mono — subset (latin)',
        'Copyright 2017 IBM Corp. All rights reserved.',
        '=' * 72,
        mono_license.strip(),
        '',
        'Source packages: https://github.com/notofonts/noto-cjk , https://github.com/IBM/plex',
    ]
    with io.open(os.path.join(OUT_DIR, 'OFL.txt'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(parts) + '\n')


def main():
    print('== 收集字符集 ==')
    chars = collect_charset()
    os.makedirs(OUT_DIR, exist_ok=True)

    print('== 下载源字体（缓存于 tools/.fontcache/）==')
    licenses = []
    for spec in FONTS:
        spec['src_path'] = fetch('%s/%s/files/%s' % (CDN, spec['pkg'], spec['src']),
                                 os.path.join(CACHE_DIR, spec['src']))
        lic_name = spec['pkg'].replace('@', '-') + '-LICENSE'
        licenses.append(fetch('%s/%s/LICENSE' % (CDN, spec['pkg']),
                              os.path.join(CACHE_DIR, lic_name)))

    print('== 子集化 ==')
    over = []
    for spec in FONTS:
        out_path = os.path.join(OUT_DIR, spec['out'])
        size = subset_font(spec, spec['src_path'], chars, out_path)
        flag = 'OK' if size <= spec['budget'] else '超出预算 %.0f KB' % (spec['budget'] / 1024)
        print('  %-32s %8.1f KB  [%s]' % (spec['out'], size / 1024, flag))
        if size > spec['budget']:
            over.append((spec['out'], size))

    with io.open(licenses[0], 'r', encoding='utf-8', errors='ignore') as fh:
        noto_lic = fh.read()
    with io.open(licenses[1], 'r', encoding='utf-8', errors='ignore') as fh:
        mono_lic = fh.read()
    write_ofl(noto_lic, mono_lic)
    print('  %-32s %8d 字节' % ('OFL.txt', os.path.getsize(os.path.join(OUT_DIR, 'OFL.txt'))))

    if over:
        print('')
        print('[!] 以下字体超出体积预算，按方案 3.2 兜底：')
        for name, size in over:
            print('    %s = %.1f KB' % (name, size / 1024))
        print('    若衬线 > 1.5MB，则改用「只有 h1/h2/.hero-title 用衬线」的方案。')
    print('')
    print('完成：%s' % OUT_DIR)


if __name__ == '__main__':
    main()