#!/bin/bash
# 競輪予測アプリの起動スクリプト

# 色の定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   競輪予測アプリ 起動スクリプト${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 依存関係のチェック
echo -e "${YELLOW}[1/4] 依存関係を確認中...${NC}"
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}依存関係をインストール中...${NC}"
    pip install -r requirements.txt
fi

# モデルファイルのチェック
echo -e "${YELLOW}[2/4] モデルファイルを確認中...${NC}"
if [ ! -f "backend/models/model.pkl" ]; then
    echo -e "${YELLOW}モデルを学習中（初回のみ）...${NC}"
    python backend/train_model.py
fi

# IPアドレスの取得
echo -e "${YELLOW}[3/4] ネットワーク情報を取得中...${NC}"
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# サーバーの起動
echo -e "${YELLOW}[4/4] サーバーを起動中...${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   サーバーが起動しました！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "📱 ${BLUE}iPhoneからアクセスする方法:${NC}"
echo ""
echo -e "  1. iPhoneとこのPCを${YELLOW}同じWi-Fiネットワーク${NC}に接続"
echo ""
echo -e "  2. iPhoneのSafariで以下のURLを開く:"
echo -e "     ${GREEN}http://${IP_ADDRESS}:5000${NC}"
echo ""
echo -e "  3. PWAとしてインストールする場合:"
echo -e "     - 共有ボタン（□↑）をタップ"
echo -e "     - 「ホーム画面に追加」を選択"
echo ""
echo -e "🖥️  ${BLUE}このPCからアクセスする場合:${NC}"
echo -e "     ${GREEN}http://localhost:5000${NC}"
echo ""
echo -e "${YELLOW}サーバーを停止するには Ctrl+C を押してください${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Flaskサーバーの起動
cd "$(dirname "$0")"
python backend/app.py
