#!/bin/bash
# Cloudflare Tunnel 起動スクリプト
# Usage: ./start_tunnel.sh

set -e

echo "🚀 競輪予測アプリを起動します..."
echo ""

# 依存関係チェック
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3が見つかりません。インストールしてください。"
    exit 1
fi

if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflaredが見つかりません。"
    echo "インストール方法: CLOUDFLARE_TUNNEL_SETUP.mdを参照してください"
    exit 1
fi

# プロジェクトディレクトリに移動
cd "$(dirname "$0")"

echo "📦 依存関係をチェック中..."
if [ ! -d ".venv" ]; then
    echo "⚠️  仮想環境が見つかりません。作成しますか？ (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    else
        echo "仮想環境なしで続行します..."
    fi
else
    source .venv/bin/activate
fi

# Pythonサーバーをバックグラウンドで起動
echo "🐍 Webサーバーを起動中... (ポート8000)"
python3 web_app.py &
SERVER_PID=$!

# サーバーの起動を待つ
sleep 3

# サーバーが起動したか確認
if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ サーバーの起動に失敗しました"
    exit 1
fi

echo "✅ サーバー起動完了 (PID: $SERVER_PID)"
echo ""
echo "🌐 Cloudflare Tunnelを起動中..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# トンネル起動（Ctrl+Cで終了するまで）
cloudflared tunnel --url http://localhost:8000

# Ctrl+C後のクリーンアップ
echo ""
echo "🛑 Cloudflare Tunnelを停止しました"
echo "🧹 サーバーを停止中..."
kill $SERVER_PID 2>/dev/null || true
echo "✅ 完全に停止しました"
