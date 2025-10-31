@echo off
chcp 65001 > nul
:: 競輪予測アプリの起動スクリプト（Windows用）

echo ========================================
echo    競輪予測アプリ 起動スクリプト
echo ========================================
echo.

:: 依存関係のチェック
echo [1/4] 依存関係を確認中...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo 依存関係をインストール中...
    pip install -r requirements.txt
)

:: モデルファイルのチェック
echo [2/4] モデルファイルを確認中...
if not exist "backend\models\model.pkl" (
    echo モデルを学習中（初回のみ）...
    python backend\train_model.py
)

:: IPアドレスの取得
echo [3/4] ネットワーク情報を取得中...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set IP_ADDRESS=%%a
    goto :found
)
:found
set IP_ADDRESS=%IP_ADDRESS: =%

:: サーバーの起動
echo [4/4] サーバーを起動中...
echo.
echo ========================================
echo    サーバーが起動しました！
echo ========================================
echo.
echo 📱 iPhoneからアクセスする方法:
echo.
echo   1. iPhoneとこのPCを同じWi-Fiネットワークに接続
echo.
echo   2. iPhoneのSafariで以下のURLを開く:
echo      http://%IP_ADDRESS%:5000
echo.
echo   3. PWAとしてインストールする場合:
echo      - 共有ボタン（□↑）をタップ
echo      - 「ホーム画面に追加」を選択
echo.
echo 🖥️  このPCからアクセスする場合:
echo      http://localhost:5000
echo.
echo サーバーを停止するには Ctrl+C を押してください
echo.
echo ========================================
echo.

:: Flaskサーバーの起動
python backend\app.py
