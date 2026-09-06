#!/usr/bin/env bash
# 课程站点局域网服务：构建 → 自动接管端口 → 0.0.0.0 服务
# 用法：bash serve_site.sh [端口]   默认 8000
# 若端口已被占用（多半是上一次的本站点实例），会自动停掉旧实例后启动。
set -e
REVIEW=/home/admin02/Code/WorkSpace/makemore-tutorial-review
PY=/home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python
PORT=${1:-8000}

"$PY" "$REVIEW/tools/build_site_lite.py"

# 自动接管：停掉已占用该端口的旧实例（按 ss 精确找监听 pid，避免误杀）
OLDS=$(ss -tlnp 2>/dev/null | grep -E ":$PORT[[:space:]]" | grep -oP 'pid=\K[0-9]+' | sort -u || true)
if [ -n "$OLDS" ]; then
  echo "↻ 端口 $PORT 已被占用（pid: $(echo $OLDS | tr '\n' ' ')），自动停掉旧实例…"
  kill $OLDS 2>/dev/null || true
  sleep 1
  # 仍在则补一刀
  OLDS2=$(ss -tlnp 2>/dev/null | grep -E ":$PORT[[:space:]]" | grep -oP 'pid=\K[0-9]+' | sort -u || true)
  [ -n "$OLDS2" ] && kill -9 $OLDS2 2>/dev/null || true
fi

cd "$REVIEW/site_build/site_html"
IP=$(hostname -I | awk '{print $1}')
echo
echo "──────────────────────────────────────────────"
echo "  📖 课程站点已就绪"
echo "   本机访问   http://127.0.0.1:$PORT/"
echo "   局域网访问 http://$IP:$PORT/"
echo "   （手机/其他电脑浏览器直接打开局域网地址；Ctrl+C 停止）"
echo "──────────────────────────────────────────────"
exec "$PY" "$REVIEW/tools/serve_http.py" "$PORT"
