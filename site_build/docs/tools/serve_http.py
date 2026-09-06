#!/usr/bin/env python3
"""课程站点 HTTP 服务：静态文件 + 容错重定向。
- 路径只含 * / 空白（复制 markdown 时带上的星号）→ 302 回首页
- 其余同 SimpleHTTPRequestHandler
用法：python serve_http.py [端口]   （需先 cd 到站点目录）
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-cache, no-store, must-revalidate')
        super().end_headers()
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
