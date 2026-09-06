#!/usr/bin/env bash
# 课程站点局域网服务：构建 → 0.0.0.0 服务（端口默认 8000）
# 用法：bash tools/serve_site.sh [端口]
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
PORT=${1:-8000}
"$PY" "$REPO/tools/build_site_lite.py"
cd "$REPO/site_build/site_html"
IP=$(hostname -I | awk '{print $1}')
echo
echo "──────────────────────────────────────────────"
echo "  📖 课程站点已就绪"
echo "   本机访问   http://127.0.0.1:$PORT/"
echo "   局域网访问 http://$IP:$PORT/"
echo "──────────────────────────────────────────────"
exec "$PY" "$REPO/tools/serve_http.py" "$PORT"
