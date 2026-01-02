@echo off
REM Cloudflare Tunnel 起動スクリプト (Windows版)
REM Usage: start_tunnel.bat

echo 🚀 競輪予測アプリを起動します...
echo.

REM 依存関係チェック
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Pythonが見つかりません。インストールしてください。
    pause
    exit /b 1
)

where cloudflared >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ cloudflaredが見つかりません。
    echo インストール方法: CLOUDFLARE_TUNNEL_SETUP.mdを参照してください
    pause
    exit /b 1
)

REM プロジェクトディレクトリに移動
cd /d "%~dp0"

echo 📦 依存関係をチェック中...
if not exist ".venv\" (
    echo ⚠️  仮想環境が見つかりません。作成します...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM Pythonサーバーをバックグラウンドで起動
echo 🐍 Webサーバーを起動中... (ポート8000)
start /B python web_app.py
timeout /t 3 /nobreak >nul

echo ✅ サーバー起動完了
echo.
echo 🌐 Cloudflare Tunnelを起動中...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo ⚠️  停止するには Ctrl+C を2回押してください
echo.

REM トンネル起動
cloudflared tunnel --url http://localhost:8000

REM 終了処理
echo.
echo 🛑 Cloudflare Tunnelを停止しました
echo 🧹 サーバーを停止中...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq web_app.py*" >nul 2>nul
echo ✅ 完全に停止しました
pause
