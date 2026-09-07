@echo off
REM 课程站点局域网服务：构建 → 自动接管端口 → 0.0.0.0 服务
REM 本文件为 UTF-8 编码，先切控制台代码页，否则中文/制表线按 GBK 解析会乱码
chcp 65001 >nul
REM 用法：serve_site.bat [端口]   默认 8000
REM 若端口已被占用（多半是上一次的本站点实例），会自动停掉旧实例后启动。

setlocal enabledelayedexpansion

set PORT=%1
if "%PORT%"=="" set PORT=8000

REM 获取当前目录
set SCRIPT_DIR=%~dp0
set REVIEW_DIR=%SCRIPT_DIR%..

REM Python 环境：按优先级查找（可通过环境变量 PYTHON_ENV 覆盖）
REM 1. 环境变量 PYTHON_ENV 指定的路径
REM 2. 仓库同级 .venv 目录
REM 3. 常见自定义路径
REM 4. 系统 python
set "PYTHON="
if defined PYTHON_ENV (
    if exist "%PYTHON_ENV%\Scripts\python.exe" set "PYTHON=%PYTHON_ENV%\Scripts\python.exe"
)
if not defined PYTHON if exist "%REVIEW_DIR%\.venv\Scripts\python.exe" set "PYTHON=%REVIEW_DIR%\.venv\Scripts\python.exe"
if not defined PYTHON if exist "G:\Workspace\python_env\.venv\Scripts\python.exe" set "PYTHON=G:\Workspace\python_env\.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

REM 自动接管：先停掉已占用该端口的旧实例（旧实例的工作目录在 site_html，
REM 不先杀掉会导致构建时 rmtree 报 WinError 32，重建失败只能用旧内容）
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo 端口 %PORT% 已被占用（pid: %%a），自动停掉旧实例...
    taskkill /PID %%a /F >nul 2>&1
)

REM 等待 Windows 释放被杀进程的工作目录句柄，否则构建的 rmtree 可能报 WinError 32
ping -n 3 127.0.0.1 >nul

REM 构建站点
set "SITE_DIR=%REVIEW_DIR%\site_build\site_html"
"%PYTHON%" "%REVIEW_DIR%\tools\build_site_lite.py"
if not exist "%SITE_DIR%\index.html" (
    echo 构建似乎未完成（目录可能仍被占用），3 秒后自动重试一次...
    ping -n 4 127.0.0.1 >nul
    "%PYTHON%" "%REVIEW_DIR%\tools\build_site_lite.py"
)

REM 启动 HTTP 服务（外壳不 cd 进站点目录：目录锁只由下面的 python 持有，
REM 下次重启杀掉它即可释放，避免外壳 cmd 残留锁死 site_html）
REM 获取本机 IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /R /C:"IPv4.*[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*"') do (
    set IP=%%a
    set IP=!IP: =!
    goto :found_ip
)
:found_ip

echo.
echo ──────────────────────────────────────────────
echo   📖 课程站点已就绪
echo    本机访问   http://127.0.0.1:%PORT%/
echo    局域网访问 http://%IP%:%PORT%/
echo    （手机/其他电脑浏览器直接打开局域网地址；Ctrl+C 停止）
echo ──────────────────────────────────────────────

"%PYTHON%" -c "import os,sys; os.chdir(r'%SITE_DIR%'); sys.argv=['serve_http.py','%PORT%']; exec(open(r'%REVIEW_DIR%\tools\serve_http.py',encoding='utf-8').read())"
