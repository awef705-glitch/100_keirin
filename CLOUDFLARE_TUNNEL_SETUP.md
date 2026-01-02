# Cloudflare Tunnel で完全無料デプロイ

## 特徴
- ✅ **完全無料・無制限**
- ✅ デプロイ失敗なし
- ✅ ローカルで動くのでデバッグ簡単
- ✅ iPhone完結（トンネル経由でアクセス）
- ✅ 無料枠の心配不要

---

## セットアップ手順

### 1. Cloudflaredをインストール

**Windows (WSL)**:
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

または、Windows版をダウンロード：
https://github.com/cloudflare/cloudflared/releases

---

### 2. Cloudflareにログイン

```bash
cloudflared tunnel login
```

ブラウザが開くので、Cloudflareアカウントでログイン（無料アカウントでOK）

---

### 3. トンネルを作成

```bash
# トンネル作成
cloudflared tunnel create keirin-app

# 出力例:
# Tunnel credentials written to /home/user/.cloudflared/UUID.json
# Created tunnel keirin-app with id UUID
```

---

### 4. トンネルを起動

#### **方法A: 簡易版（すぐ使える）**

```bash
# 1. ローカルでサーバー起動
cd /mnt/c/Users/awef7/Documents/00_GitHub/00_Me/100_keirin
python web_app.py

# 2. 別のターミナルでトンネル起動
cloudflared tunnel --url http://localhost:8000
```

→ `https://ランダムURL.trycloudflare.com` が発行される
→ このURLにiPhoneからアクセス！

#### **方法B: 固定URL版（推奨）**

```bash
# config.yml作成
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: keirin-app
credentials-file: /home/user/.cloudflared/UUID.json

ingress:
  - hostname: keirin.YOUR-DOMAIN.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# トンネル起動
cloudflared tunnel run keirin-app
```

---

### 5. iPhoneからアクセス

1. サーバー起動: `python web_app.py`
2. トンネル起動: `cloudflared tunnel --url http://localhost:8000`
3. 表示されたURLをiPhoneのSafariで開く
4. ホーム画面に追加で完了！

---

## 🎯 起動スクリプト（簡単版）

`start_tunnel.sh`を作成：

```bash
#!/bin/bash
# サーバー起動
python web_app.py &
SERVER_PID=$!

# トンネル起動
cloudflared tunnel --url http://localhost:8000

# 終了時にサーバーも停止
kill $SERVER_PID
```

実行：
```bash
chmod +x start_tunnel.sh
./start_tunnel.sh
```

---

## メリット

| 項目 | Railway/Render | Cloudflare Tunnel |
|------|---------------|-------------------|
| 料金 | 無料枠あり（制限） | **完全無料・無制限** |
| デプロイ | 複雑・失敗しやすい | **不要（ローカル実行）** |
| デバッグ | 困難 | **簡単（ローカル）** |
| 起動時間 | 遅い | **即座** |
| 無料枠 | 使い切る可能性 | **無制限** |

---

## トラブルシューティング

### Q: URLが毎回変わる
A: 固定URL版を使う（独自ドメイン設定）

### Q: トンネルが切れる
A: 自動再起動スクリプトを使う

### Q: ローカルサーバーが起動しない
A: 依存関係をインストール：`pip install -r requirements.txt`

---

## 注意点

- PCを起動している間のみ利用可能
- インターネット接続が必要
- 完全オフラインにはならない

→ 完全オフライン対応には「GitHub Pages + ONNX」を推奨
