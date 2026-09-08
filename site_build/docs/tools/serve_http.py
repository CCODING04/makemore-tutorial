#!/usr/bin/env python3
"""课程站点 HTTP 服务：静态文件 + 容错重定向。
- 路径只含 * / 空白（复制 markdown 时带上的星号）→ 302 回首页
- 源码/文本类扩展名一律 text/plain; charset=utf-8（否则 Windows 注册表
  可能把 .py 映射成二进制，浏览器表现为下载/空白打不开）
- 其余同 SimpleHTTPRequestHandler
用法：python serve_http.py [端口]   （需先 cd 到站点目录）
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# 源码/文本类 → text/plain（注意不要放 .css/.js/.json：样式表必须 text/css，
# 否则 Chromium 严格模式整表拒绝；.js/.json 保持原生 MIME）
TEXT_EXTS = {'.py', '.sh', '.bash', '.zsh', '.yaml', '.yml', '.toml',
             '.txt', '.cfg', '.ini', '.csv', '.ipynb', '.log', '.md'}

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-cache, no-store, must-revalidate')
        super().end_headers()
    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in TEXT_EXTS:
            return 'text/plain; charset=utf-8'
        return super().guess_type(path)
    def send_head(self):
        path = self.path
        # 复制 markdown 时带上的星号/多余斜杠 → 回首页
        if '*' in path and re.fullmatch(r'[/*\s]+', path):
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return None
        return super().send_head()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
